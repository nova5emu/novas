// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/gpu_write_trace_dump.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <fmt/format.h>
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/zstd_compression.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace VideoCommon::GpuWriteTrace {
namespace {
constexpr size_t MaxArtifactBytes = 32 * 1024 * 1024;
constexpr size_t MaxResidentBytes = 64 * 1024 * 1024;
constexpr size_t MaxInstructions = 128 * 1024;
struct Artifact {
    u64 shader{}, pipeline{}, revision{};
    std::vector<u8> ir, spirv; // Compressed until a top window requests this shader.
    bool written{};
};
struct Dumps {
    std::mutex mutex;
    std::map<u64, std::vector<std::shared_ptr<Artifact>>> shaders;
    size_t resident_bytes{};
    u64 revision{};
    std::filesystem::path directory;
};
std::unique_ptr<Dumps> dumps;

bool Write(const std::filesystem::path& path, std::span<const u8> bytes) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.flush();
    return file.good();
}
} // namespace

void ResetComputeDumps(bool active) {
    dumps.reset();
    if (!active) return;
    dumps = std::make_unique<Dumps>();
    dumps->directory = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) /
        "gpu_write_trace" / fmt::format("{}", std::chrono::system_clock::now().time_since_epoch().count());
}

std::string PrepareComputeDump(const Shader::IR::Program& program) {
    std::string text = fmt::format(
        "stage=compute ir=post_translate_pre_spirv block={}x{}x{} native_global_bounds={}\n",
        program.workgroup_size[0], program.workgroup_size[1], program.workgroup_size[2],
        program.info.native_global_bounds);
    for (size_t i = 0; i < program.info.storage_buffers_descriptors.size(); ++i) {
        const auto& descriptor = program.info.storage_buffers_descriptors[i];
        const Shader::StorageWriteFootprint not_collected;
        const auto& footprint = i < program.info.storage_write_footprints.size()
            ? program.info.storage_write_footprints[i] : not_collected;
        text += fmt::format("binding={} cb={}+{:#x} count={} written={} indirect_offset={:#x} "
                            "known={} reason={} envelope=[{:#x},{:#x}) symbolic={} terms={} coherence_eligible={}\n",
                            i, descriptor.cbuf_index, descriptor.cbuf_offset, descriptor.count,
                            descriptor.is_written, descriptor.indirect_offset, footprint.known,
                            Shader::StorageWriteReasonName(footprint.reason), footprint.begin, footprint.end,
                            footprint.symbolic, footprint.terms.size(), footprint.coherence_eligible);
        for (size_t term_index = 0; term_index < (std::min)(footprint.terms.size(), size_t{64}); ++term_index) {
            const auto& term = footprint.terms[term_index];
            text += fmt::format("store_term binding={} index={} width={} constant={:#x} "
                                "wg_coeff={},{},{} local_coeff={},{},{} needs_base={} "
                                "base_residual_mask={:#x} arithmetic=u32_modular truncated={}\n",
                                i, term_index, term.width, term.constant,
                                term.coefficients[0], term.coefficients[1], term.coefficients[2],
                                term.coefficients[3], term.coefficients[4], term.coefficients[5],
                                term.needs_base, ~term.base_mask, footprint.terms.size() > 64);
        }
    }
    size_t instructions{};
    for (const auto* block : program.blocks) {
        instructions += block->Instructions().size();
        if (instructions > MaxInstructions) {
            return text + "IR_OMITTED instruction_limit=131072 (SPIR-V retained if within size limit)\n";
        }
    }
    auto ir = Shader::IR::DumpProgram(program);
    if (text.size() > MaxArtifactBytes || ir.size() > MaxArtifactBytes - text.size()) {
        return text + "IR_OMITTED byte_limit=33554432\n";
    }
    return text + ir;
}

void RegisterComputeDump(u64 shader, u64 pipeline, std::string ir, std::span<const u32> spirv) {
    if (ir.empty() || spirv.empty() || ir.size() > MaxArtifactBytes || spirv.size_bytes() > MaxArtifactBytes) {
        LOG_WARNING(HW_GPU, "GPU_WRITE_TRACE dump_capture_skipped shader={:016x} pipeline={:016x} reason=size_limit",
                    shader, pipeline);
        return;
    }
    auto artifact = std::make_shared<Artifact>();
    artifact->shader = shader;
    artifact->pipeline = pipeline;
    artifact->ir = Common::Compression::CompressDataZSTD(
        reinterpret_cast<const u8*>(ir.data()), ir.size(), 1);
    artifact->spirv = Common::Compression::CompressDataZSTD(
        reinterpret_cast<const u8*>(spirv.data()), spirv.size_bytes(), 1);
    if (artifact->ir.empty() || artifact->spirv.empty()) {
        LOG_WARNING(HW_GPU, "GPU_WRITE_TRACE dump_capture_skipped shader={:016x} reason=compression", shader);
        return;
    }
    bool full{};
    {
        std::scoped_lock lock{dumps->mutex};
        const auto found = dumps->shaders.find(shader);
        if (found != dumps->shaders.end()) {
            for (const auto& previous : found->second) {
                if (previous->pipeline == pipeline && previous->ir == artifact->ir &&
                    previous->spirv == artifact->spirv) return;
            }
        }
        const size_t bytes = artifact->ir.size() + artifact->spirv.size();
        full = bytes > MaxResidentBytes - dumps->resident_bytes;
        if (!full) {
            artifact->revision = ++dumps->revision;
            dumps->resident_bytes += bytes;
            dumps->shaders[shader].push_back(std::move(artifact));
        }
    }
    if (full) {
        LOG_WARNING(HW_GPU, "GPU_WRITE_TRACE dump_capture_skipped shader={:016x} pipeline={:016x} reason=64MiB_cache_limit",
                    shader, pipeline);
    }
}

void DumpTopComputeShaders(std::span<const u64> hashes) {
    std::vector<std::shared_ptr<Artifact>> selected;
    std::vector<u64> missing;
    {
        std::scoped_lock lock{dumps->mutex};
        for (u64 hash : hashes.first((std::min<size_t>)(hashes.size(), 8))) {
            const auto found = dumps->shaders.find(hash);
            if (found == dumps->shaders.end()) {
                missing.push_back(hash);
                continue;
            }
            for (const auto& artifact : found->second) {
                if (!artifact->written) selected.push_back(artifact);
            }
        }
    }
    for (u64 hash : missing) {
        LOG_WARNING(HW_GPU, "GPU_WRITE_TRACE dump_missing shader={:016x} (no retained compilation artifact)", hash);
    }
    if (selected.empty()) return;
    if (!Common::FS::CreateDirs(dumps->directory)) {
        LOG_WARNING(HW_GPU, "GPU_WRITE_TRACE dump_failed directory={}", Common::FS::PathToUTF8String(dumps->directory));
        return;
    }
    // No tracker/registry lock during decompression or disk I/O. Only the reporting thread writes.
    for (const auto& artifact : selected) {
        const auto stem = fmt::format("{:016x}-p{:016x}-v{}", artifact->shader, artifact->pipeline, artifact->revision);
        const auto ir_path = dumps->directory / (stem + ".ir.txt");
        const auto spv_path = dumps->directory / (stem + ".spv");
        const auto ir = Common::Compression::TryDecompressDataZSTD(artifact->ir, MaxArtifactBytes);
        const auto spv = Common::Compression::TryDecompressDataZSTD(artifact->spirv, MaxArtifactBytes);
        const bool success = ir && spv && Write(ir_path, *ir) && Write(spv_path, *spv);
        if (success) {
            std::scoped_lock lock{dumps->mutex};
            artifact->written = true;
        }
        LOG_INFO(HW_GPU, "GPU_WRITE_TRACE dump shader={:016x} pipeline={:016x} revision={} success={} ir={} spv={}",
                 artifact->shader, artifact->pipeline, artifact->revision, success,
                 Common::FS::PathToUTF8String(ir_path), Common::FS::PathToUTF8String(spv_path));
    }
}
} // namespace VideoCommon::GpuWriteTrace
