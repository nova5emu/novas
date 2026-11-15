// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <fmt/format.h>

#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_optimization.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_handle_program.h"
#include "video_core/shader_notify.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"

namespace Vulkan {

using Shader::ImageBufferDescriptor;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_WORDS_OFFSET;
using Tegra::Texture::TexturePair;

namespace {
std::vector<ByteAddressedRange> ByteAddressedMappedRanges(Tegra::MemoryManager& memory,
                                                        GPUVAddr address, u64 size) {
    if (size == 0 || address > (std::numeric_limits<u64>::max)() - size ||
        !memory.IsWithinGPUAddressRange(address + size - 1)) {
        throw Shader::RuntimeError("Unknown packed surface access range {:#x}+{:#x}", address, size);
    }
    std::vector<ByteAddressedRange> ranges;
    while (size != 0) {
        const u64 count{(std::min)(size, 0x1000 - (address & 0xfff))};
        const auto mapped{memory.GpuToCpuAddress(address)};
        if (!mapped || *mapped > (std::numeric_limits<u64>::max)() - count) {
            throw Shader::RuntimeError("Unmapped packed surface access range at {:#x}", address);
        }
        if (!ranges.empty() && ranges.back().address + ranges.back().size == *mapped) {
            ranges.back().size += count;
        } else {
            ranges.push_back({*mapped, count});
        }
        address += count;
        size -= count;
    }
    return ranges;
}

// EDEN_NX2_UPSCALER_TRACE=1 waits for the first native dispatch using subgroup shuffles (software
// HMMA) at or after frame EDEN_NX2_UPSCALER_TRACE_FRAME (default 600), then logs the resources of
// every native compute dispatch in the next EDEN_NX2_UPSCALER_TRACE_FRAMES frames (default 3), at
// most 4000 lines. It prints guest addresses and hashes of constant data, never the data.
struct UpscalerTrace {
    bool enabled{};
    u64 first_frame{};
    u64 frames{};
    u64 armed_frame{};
    bool armed{};
    std::atomic<u64> frame{};
    u32 lines{};
};

UpscalerTrace& GetUpscalerTrace() {
    static UpscalerTrace trace{[] {
        const auto number{[](const char* name, u64 fallback) {
            const char* const value{std::getenv(name)};
            return value ? u64{std::strtoull(value, nullptr, 10)} : fallback;
        }};
        const char* const enabled{std::getenv("EDEN_NX2_UPSCALER_TRACE")};
        return UpscalerTrace{
            .enabled = enabled != nullptr && *enabled != '\0' && *enabled != '0',
            .first_frame = number("EDEN_NX2_UPSCALER_TRACE_FRAME", 600),
            .frames = number("EDEN_NX2_UPSCALER_TRACE_FRAMES", 3),
        };
    }()};
    return trace;
}

u64 HashGuestMemory(Tegra::MemoryManager& memory, GPUVAddr address, u32 size) {
    std::vector<u8> bytes(size);
    memory.ReadBlockUnsafe(address, bytes.data(), bytes.size());
    u64 hash{0xcbf29ce484222325ULL};
    for (const u8 byte : bytes) {
        hash = (hash ^ byte) * 0x100000001b3ULL;
    }
    return hash;
}

void TraceUpscalerDispatch(UpscalerTrace& trace, const Shader::Info& info, u64 shader_hash,
                           Tegra::Engines::KeplerCompute& kepler_compute,
                           Tegra::MemoryManager& gpu_memory, BufferCache& buffer_cache,
                           TextureCache& texture_cache,
                           std::span<const VideoCommon::ImageViewInOut> views) {
    static constexpr u32 LINE_LIMIT = 4000;
    const auto log{[&trace](std::string_view line) {
        if (trace.lines++ < LINE_LIMIT) {
            LOG_INFO(Render_Vulkan, "Upscaler trace {}", line);
        }
    }};
    const auto& qmd{kepler_compute.launch_description};
    const auto& cbuf0{qmd.const_buffer_config[0]};
    const bool has_cbuf0{(qmd.const_buffer_enable_mask & 1) != 0 && cbuf0.size != 0 &&
                         gpu_memory.IsFullyMappedRange(cbuf0.Address(), cbuf0.size)};
    log(fmt::format("f={} prog={:#x} hash={:016x} grid={}x{}x{} block={}x{}x{} cb0={:#x}+{:#x} "
                    "cb0_fnv={:016x}",
                    trace.frame.load(std::memory_order_relaxed),
                    kepler_compute.regs.code_loc.Address(), shader_hash, qmd.grid_dim_x.Value(),
                    qmd.grid_dim_y.Value(), qmd.grid_dim_z.Value(), qmd.block_dim_x.Value(),
                    qmd.block_dim_y.Value(), qmd.block_dim_z.Value(), cbuf0.Address(),
                    cbuf0.size.Value(),
                    has_cbuf0 ? HashGuestMemory(gpu_memory, cbuf0.Address(), cbuf0.size) : 0));

    struct Range {
        GPUVAddr begin;
        GPUVAddr end;
    };
    std::vector<Range> ssbos;
    for (size_t index = 0; index < info.storage_buffers_descriptors.size(); ++index) {
        const auto& desc{info.storage_buffers_descriptors[index]};
        GPUVAddr slot{qmd.const_buffer_config[desc.cbuf_index].Address() + desc.cbuf_offset};
        if (desc.indirect_offset != ~0U) {
            slot = gpu_memory.Read<u64>(slot) + desc.indirect_offset;
        }
        const GPUVAddr address{gpu_memory.Read<u64>(slot)};
        const auto device_address{address ? gpu_memory.GpuToCpuAddress(address) : std::nullopt};
        u32 size{};
        if (device_address) {
            const u32 layout_size{static_cast<u32>(gpu_memory.GetMemoryLayoutSize(address))};
            const u64 next{gpu_memory.Read<u64>(slot + 8)};
            const bool next_is_size{(next >> 32) == 0 && static_cast<u32>(next) != 0 &&
                                    static_cast<u32>(next) <= layout_size};
            size = next_is_size ? static_cast<u32>(next) : (std::min)(layout_size, 8U << 20);
        }
        ssbos.push_back({address, address + size});
        log(fmt::format("  ssbo{} cb{}+{:#x} {} va={:#x} size={:#x}{} image_gpu_dirty={} "
                        "buffer_gpu_dirty={}",
                        index, desc.cbuf_index, desc.cbuf_offset, desc.is_written ? "w" : "r",
                        address, size, device_address ? "" : " null",
                        device_address && texture_cache.IsRegionGpuModified(*device_address, size),
                        device_address && buffer_cache.IsRegionGpuModified(*device_address, size)));
    }

    size_t view_index{};
    const auto log_views{[&](const auto& descriptors, std::string_view kind) {
        for (const auto& desc : descriptors) {
            for (u32 element = 0; element < desc.count; ++element, ++view_index) {
                const auto& view{texture_cache.GetImageView(views[view_index].id)};
                const auto* const image{view.BackingImage()};
                std::string aliases;
                if (image) {
                    const GPUVAddr end{image->gpu_addr + image->guest_size_bytes};
                    for (size_t ssbo = 0; ssbo < ssbos.size(); ++ssbo) {
                        if (ssbos[ssbo].begin < end && image->gpu_addr < ssbos[ssbo].end) {
                            aliases += fmt::format(" alias=ssbo{}", ssbo);
                        }
                    }
                }
                log(fmt::format("  {}{} cb{}+{:#x} tic={} {} va={:#x} size={:#x} {}x{} fmt={}{}",
                                kind, view_index, desc.cbuf_index, desc.cbuf_offset,
                                views[view_index].index, image ? "" : "null",
                                image ? image->gpu_addr : 0, image ? image->guest_size_bytes : 0,
                                view.size.width, view.size.height,
                                static_cast<u32>(view.format), aliases));
            }
        }
    }};
    log_views(info.texture_buffer_descriptors, "texbuf");
    log_views(info.image_buffer_descriptors, "imgbuf");
    log_views(info.texture_descriptors, "tex");
    log_views(info.image_descriptors, "img");
}
} // namespace

void ComputePipeline::TickUpscalerTrace() {
    auto& trace{GetUpscalerTrace()};
    if (trace.enabled) {
        if (trace.frame.fetch_add(1, std::memory_order_relaxed) == 0) {
            LOG_INFO(Render_Vulkan, "Upscaler trace enabled, arming from frame {}",
                     trace.first_frame);
        }
    }
}

ComputePipeline::ComputePipeline(const Device& device_, Scheduler& scheduler, vk::PipelineCache& pipeline_cache_,
                                 DescriptorPool& descriptor_pool,
                                 GuestDescriptorQueue& guest_descriptor_queue_,
                                 DescriptorBufferRing& descriptor_buffer_ring_,
                                 Common::ThreadWorker* thread_worker,
                                 PipelineStatistics* pipeline_statistics,
                                 VideoCore::ShaderNotify* shader_notify, const Shader::Info& info_,
                                 vk::ShaderModule spv_module_, u64 shader_hash_,
                                 bool trace_hmma_pipeline)
    : device{device_},
      pipeline_cache(pipeline_cache_), guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, info{info_},
      shader_hash{shader_hash_}, spv_module(std::move(spv_module_)) {
    if (shader_notify) {
        shader_notify->MarkShaderBuilding();
    }
    std::copy_n(info.constant_buffer_used_sizes.begin(), uniform_buffer_sizes.size(),
                uniform_buffer_sizes.begin());
    num_descriptor_entries = NumDescriptorEntries(info);

    DescriptorLayoutBuilder builder{device};
    builder.Add(info, VK_SHADER_STAGE_COMPUTE_BIT);

    uses_push_descriptor = builder.CanUsePushDescriptor();
    uses_descriptor_buffer = builder.CanUseDescriptorBuffer() && descriptor_buffer_ring.IsValid();
    descriptor_set_layout =
        builder.CreateDescriptorSetLayout(uses_push_descriptor, uses_descriptor_buffer);
    if (uses_descriptor_buffer) {
        descriptor_buffer_layout = builder.MakeDescriptorBufferLayout(*descriptor_set_layout);
        if (!descriptor_buffer_ring.CanAllocate(descriptor_buffer_layout.size)) {
            LOG_DEBUG(Render_Vulkan,
                        "Compute shader {:016X} needs {} descriptor bytes per dispatch, falling "
                        "back to sets",
                        shader_hash, descriptor_buffer_layout.size);
            uses_descriptor_buffer = false;
            descriptor_buffer_layout = {};
            descriptor_set_layout = builder.CreateDescriptorSetLayout(false);
        }
    }
    pipeline_layout = builder.CreatePipelineLayout(*descriptor_set_layout);
    if (!uses_descriptor_buffer) {
        descriptor_update_template =
            builder.CreateTemplate(*descriptor_set_layout, *pipeline_layout, uses_push_descriptor);
        if (!uses_push_descriptor) {
            descriptor_allocator =
                descriptor_pool.Allocator(device, scheduler, *descriptor_set_layout, info);
        }
    }

    auto func{[this, shader_notify, pipeline_statistics, trace_hmma_pipeline] {
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
            .pNext = nullptr,
            .requiredSubgroupSize = GuestWarpSize,
        };
        // Only warp-level ops observe the subgroup size. Forcing 32 on other shaders makes Intel
        // compile SIMD32, which can spill a register-heavy kernel for minutes (Mario Kart World).
        const bool uses_warp_ops = info.uses_subgroup_vote || info.uses_subgroup_invocation_id ||
                                   info.uses_subgroup_shuffles || info.uses_subgroup_mask ||
                                   info.uses_quad_shuffles || info.uses_fswzadd;
        VkPipelineCreateFlags flags{};
        if (device.IsKhrPipelineExecutablePropertiesEnabled() && Settings::values.renderer_debug.GetValue()) {
            flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        }
        if (uses_descriptor_buffer) {
            flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }
        if (BuildsUnoptimized(device.GetDriverID(), shader_hash)) {
            flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        }
        const VkComputePipelineCreateInfo compute_ci{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = flags,
            .stage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = device.IsExtSubgroupSizeControlSupported() && uses_warp_ops &&
                                 device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT)
                             ? &subgroup_size_ci
                             : nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = *spv_module,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            },
            .layout = *pipeline_layout,
            .basePipelineHandle = 0,
            .basePipelineIndex = 0,
        };
        const auto build_start = trace_hmma_pipeline ? std::chrono::steady_clock::now()
                                                    : std::chrono::steady_clock::time_point{};
        if (trace_hmma_pipeline) {
            LOG_INFO(Render_Vulkan,
                     "HMMA driver-build begin s6:{:016x} path=software-loop-dontinline "
                     "required_subgroup={}",
                     shader_hash, compute_ci.stage.pNext ? GuestWarpSize : 0);
        }
        try {
            pipeline = device.GetLogical().CreateComputePipeline(compute_ci, *pipeline_cache);
        } catch (const vk::Exception& exception) {
            LOG_CRITICAL(Render_Vulkan,
                         "Compute shader {:016X} pipeline creation failed: {}",
                         shader_hash, exception.what());
            std::scoped_lock lock{build_mutex};
            is_built = true;
            build_condvar.notify_one();
            if (shader_notify) {
                shader_notify->MarkShaderComplete();
            }
            return;
        }
        if (trace_hmma_pipeline) {
            LOG_INFO(Render_Vulkan, "HMMA driver-build end s6:{:016x} cpu_ms={:.3f}", shader_hash,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - build_start).count());
        }

        // Log compute pipeline creation
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogPipelineStateChange(
                "ComputePipeline created"
            );
        }

        if (pipeline_statistics) {
            pipeline_statistics->Collect(device, *pipeline);
        }
        // Nothing reads the module once the pipeline exists; the driver keeps its SPIR-V while it
        // lives.
        spv_module.reset();
        std::scoped_lock lock{build_mutex};
        is_built = true;
        build_condvar.notify_one();
        if (shader_notify) {
            shader_notify->MarkShaderComplete();
        }
    }};
    build_task = std::move(func);
    if (thread_worker) {
        build_task_done = false;
        thread_worker->QueueWork([this] {
            RunBuild();
            // The last access to this pipeline from the worker.
            build_task_done.store(true, std::memory_order::release);
        });
    } else {
        RunBuild();
    }
}

void ComputePipeline::RunBuild() {
    // A draw or dispatch that needs the pipeline builds it on the scheduler thread instead of
    // waiting for the worker queue, which can hold hundreds of builds, to reach it.
    if (build_claimed.exchange(true)) {
        return;
    }
    build_task();
    build_task = nullptr;
}

bool ComputePipeline::Configure(Tegra::Engines::KeplerCompute& kepler_compute,
                                Tegra::MemoryManager& gpu_memory, Scheduler& scheduler,
                                BufferCache& buffer_cache, TextureCache& texture_cache,
                                TextureCacheRuntime& texture_runtime,
                                std::optional<ByteAddressedShadow>& shadow) {
    shadow.reset();
    guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries, uses_descriptor_buffer);

    buffer_cache.SetComputeUniformBufferState(info.constant_buffer_mask, &uniform_buffer_sizes);
    buffer_cache.UnbindComputeStorageBuffers();
    const bool footprint_image_writes = VideoCommon::GpuWriteTrace::apply_footprints &&
        std::ranges::any_of(info.image_descriptors, [](const auto& desc) { return desc.is_written; });
    size_t ssbo_index{};
    for (const auto& desc : info.storage_buffers_descriptors) {
        ASSERT(desc.count == 1);
        buffer_cache.BindComputeStorageBuffer(ssbo_index, desc.cbuf_index, desc.cbuf_offset,
                                              desc.is_written, desc.indirect_offset);
        if (VideoCommon::GpuWriteTrace::enabled || VideoCommon::GpuWriteTrace::profile_footprints) {
            const auto& launch = kepler_compute.launch_description;
            const auto* recipe = ssbo_index < info.storage_write_footprints.size()
                ? &info.storage_write_footprints[ssbo_index] : nullptr;
            const Shader::StorageWriteFootprint not_collected;
            const auto& footprint = recipe ? *recipe : not_collected;
            buffer_cache.SetComputeWriteTrace(ssbo_index, {
                .shader = shader_hash,
                .binding = static_cast<u32>(ssbo_index),
                .cbuf = desc.cbuf_index,
                .cbuf_offset = desc.cbuf_offset,
                .grid = {launch.grid_dim_x.Value(), launch.grid_dim_y.Value(), launch.grid_dim_z.Value()},
                .block = {launch.block_dim_x.Value(), launch.block_dim_y.Value(), launch.block_dim_z.Value()},
                .indirect = kepler_compute.GetIndirectComputeAddress().has_value(),
                .known = footprint.known,
                .reason = static_cast<u8>(footprint.reason),
                .begin = footprint.begin,
                .end = footprint.end,
            }, recipe, footprint_image_writes);
        }
        ++ssbo_index;
    }

    texture_cache.SynchronizeDescriptors(true);

    boost::container::small_vector<VideoCommon::ImageViewInOut, 64> views;
    boost::container::small_vector<VideoCommon::SamplerId, 64> samplers;

    const auto& qmd{kepler_compute.launch_description};
    const auto& cbufs{qmd.const_buffer_config};
    const bool via_header_index{qmd.linked_tsc != 0};
    const auto read_handle{[&](const auto& desc, u32 index) {
        if (desc.handle_program != ~0U) {
            // Read by the shader from a storage buffer: computed for this dispatch.
            return TexturePair(VideoCommon::EvaluateTextureHandle(
                                   info, desc.handle_program, gpu_memory,
                                   [&](u32 cbuf) -> GPUVAddr {
                                       return cbuf < cbufs.size() &&
                                                      ((qmd.const_buffer_enable_mask >> cbuf) &
                                                       1) != 0
                                                  ? cbufs[cbuf].Address()
                                                  : GPUVAddr{};
                                   }),
                               via_header_index);
        }
        ASSERT(((qmd.const_buffer_enable_mask >> desc.cbuf_index) & 1) != 0);
        const u32 index_offset{index << desc.size_shift};
        const u32 offset{desc.cbuf_offset + index_offset};
        const GPUVAddr addr{cbufs[desc.cbuf_index].Address() + offset};
        if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&> ||
                      std::is_same_v<decltype(desc), const Shader::TextureBufferDescriptor&>) {
            if (desc.has_secondary) {
                ASSERT(((qmd.const_buffer_enable_mask >> desc.secondary_cbuf_index) & 1) != 0);
                const u32 secondary_offset{desc.secondary_cbuf_offset};
                const GPUVAddr separate_addr{cbufs[desc.secondary_cbuf_index].Address() +
                                             secondary_offset};
                const u32 lhs_raw{gpu_memory.Read<u32>(addr) << desc.shift_left};
                const u32 rhs_raw{gpu_memory.Read<u32>(separate_addr) << desc.secondary_shift_left};
                return TexturePair(lhs_raw | rhs_raw, via_header_index);
            }
        }
        return TexturePair(gpu_memory.Read<u32>(addr), via_header_index);
    }};
    const auto add_image{[&](const auto& desc, bool blacklist) {
        for (u32 index = 0; index < desc.count; ++index) {
            const auto handle{read_handle(desc, index)};
            views.push_back({
                .index = handle.first,
                .blacklist = blacklist,
                .id = {},
            });
        }
    }};
    for (const auto& desc : info.texture_buffer_descriptors) {
        add_image(desc, false);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        add_image(desc, false);
    }
    for (const auto& desc : info.texture_descriptors) {
        for (u32 index = 0; index < desc.count; ++index) {
            const auto handle{read_handle(desc, index)};
            views.push_back({handle.first});

            VideoCommon::SamplerId sampler = texture_cache.GetSamplerId(handle.second, true);
            samplers.push_back(sampler);
        }
    }
    for (const auto& desc : info.image_descriptors) {
        add_image(desc, desc.is_written);
    }
    texture_cache.FillImageViews(std::span(views.data(), views.size()), true);
    if (auto& trace{GetUpscalerTrace()}; trace.enabled && kepler_compute.IsNativeMode()) {
        const u64 frame{trace.frame.load(std::memory_order_relaxed)};
        if (!trace.armed && frame >= trace.first_frame && info.uses_subgroup_shuffles) {
            trace.armed = true;
            trace.armed_frame = frame;
            LOG_INFO(Render_Vulkan, "Upscaler trace armed at frame {} by {:016x}", frame,
                     shader_hash);
        }
        if (trace.armed && frame > trace.armed_frame && frame <= trace.armed_frame + trace.frames) {
            TraceUpscalerDispatch(trace, info, shader_hash, kepler_compute, gpu_memory,
                                  buffer_cache, texture_cache, views);
        }
    }

    if (RecordingSampledTextures()) {
        sampled_images.clear();
        const size_t first = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                             Shader::NumDescriptors(info.image_buffer_descriptors);
        for (size_t index = first; index < first + Shader::NumDescriptors(info.texture_descriptors);
             ++index) {
            sampled_images.push_back({views[index].index, views[index].id, samplers[index - first]});
        }
    }

    std::optional<size_t> shadow_index;
    size_t image_index{Shader::NumDescriptors(info.texture_buffer_descriptors) +
                       Shader::NumDescriptors(info.image_buffer_descriptors) +
                       Shader::NumDescriptors(info.texture_descriptors)};
    written_images.clear();
    for (const auto& desc : info.image_descriptors) {
        for (u32 i = 0; i < desc.count; ++i, ++image_index) {
            if (desc.is_written) {
                written_images.push_back(views[image_index].id);
            }
            const auto& view{texture_cache.GetImageView(views[image_index].id)};
            if (!desc.is_byte_addressed || view.format != PixelFormat::R16_FLOAT) {
                continue;
            }
            if (shadow_index || !kepler_compute.IsNativeMode() ||
                kepler_compute.GetIndirectComputeAddress() || desc.count != 1 ||
                desc.is_read || !desc.is_written || desc.type != Shader::TextureType::Color2D ||
                desc.format != Shader::ImageFormat::R32_UINT || info.uses_global_memory) {
                throw Shader::RuntimeError("Unsupported packed R16 surface dispatch or unbounded global access");
            }
            shadow_index = image_index;
        }
    }
    if (shadow_index) {
        const auto max_dim{device.GetMaxComputeWorkGroupCount()};
        if (qmd.grid_dim_x == 0 || qmd.grid_dim_y == 0 || qmd.grid_dim_z == 0 ||
            qmd.grid_dim_x > max_dim[0] || qmd.grid_dim_y > max_dim[1] ||
            qmd.grid_dim_z > max_dim[2]) {
            throw Shader::RuntimeError("Invalid packed R16 surface dispatch dimensions");
        }
        const auto& target_view{texture_cache.GetImageView(views[*shadow_index].id)};
        const auto* target{target_view.BackingImage()};
        if (!target) {
            throw Shader::RuntimeError("Unknown packed R16 surface backing");
        }
        const auto target_ranges{ByteAddressedMappedRanges(gpu_memory, target->gpu_addr,
                                                          target->guest_size_bytes)};
        for (size_t i = 0; i < views.size(); ++i) {
            if (i == *shadow_index) {
                continue;
            }
            const auto& view{texture_cache.GetImageView(views[i].id)};
            if (view.IsBuffer()) {
                continue;
            }
            const auto* image{view.BackingImage()};
            if (!image) {
                throw Shader::RuntimeError("Unknown packed R16 surface peer image at binding {}", i);
            }
            for (const auto& a : target_ranges) {
                for (const auto& b : ByteAddressedMappedRanges(gpu_memory, image->gpu_addr,
                                                               image->guest_size_bytes)) {
                    if (a.Overlaps(b)) {
                        throw Shader::RuntimeError("Aliased packed R16 surface peer image at binding {}", i);
                    }
                }
            }
        }
        image_index = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                      Shader::NumDescriptors(info.image_buffer_descriptors) +
                      Shader::NumDescriptors(info.texture_descriptors);
        for (const auto& desc : info.image_descriptors) {
            for (u32 i = 0; i < desc.count; ++i, ++image_index) {
                if (image_index != *shadow_index) {
                    static_cast<void>(texture_cache.GetImageView(views[image_index].id)
                        .StorageView(desc.type, desc.format, desc.is_byte_addressed));
                }
            }
        }
        shadow = texture_runtime.CreateByteAddressedShadow(target_view);
    }

    buffer_cache.UnbindComputeTextureBuffers();
    size_t index{};
    const auto add_buffer{[&](const auto& desc) {
        constexpr bool is_image = std::is_same_v<decltype(desc), const ImageBufferDescriptor&>;
        for (u32 i = 0; i < desc.count; ++i) {
            bool is_written{false};
            if constexpr (is_image) {
                is_written = desc.is_written;
            }
            ImageView& image_view = texture_cache.GetImageView(views[index].id);
            PixelFormat format{image_view.format};
            if constexpr (is_image) {
                if (const auto explicit_format{PixelFormatFromImageFormat(desc.format)}) {
                    format = *explicit_format;
                }
            }
            buffer_cache.BindComputeTextureBuffer(index, image_view.GpuAddr(),
                                                  image_view.BufferSize(), format,
                                                  is_written, is_image);
            ++index;
        }
    }};
    std::ranges::for_each(info.texture_buffer_descriptors, add_buffer);
    std::ranges::for_each(info.image_buffer_descriptors, add_buffer);

    buffer_cache.UpdateComputeBuffers();
    buffer_cache.BindHostComputeBuffers();
    if (buffer_cache.any_buffer_uploaded) {
        buffer_cache.runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }

    RescalingPushConstant rescaling;
    const VideoCommon::SamplerId* samplers_it{samplers.data()};
    const VideoCommon::ImageViewInOut* views_it{views.data()};
    if (!shadow) {
        PushImageDescriptors(texture_cache, guest_descriptor_queue, info, rescaling, samplers_it,
                             views_it);
    } else {
        auto sampled_info{info};
        sampled_info.image_descriptors.clear();
        PushImageDescriptors(texture_cache, guest_descriptor_queue, sampled_info, rescaling,
                             samplers_it, views_it);
        for (const auto& desc : info.image_descriptors) {
            bool is_rescaled{};
            for (u32 i = 0; i < desc.count; ++i) {
                const size_t view_index{static_cast<size_t>(views_it - views.data())};
                auto& view{texture_cache.GetImageView((views_it++)->id)};
                const VkImageView storage{view_index == *shadow_index ? shadow->view :
                    view.StorageView(desc.type, desc.format, desc.is_byte_addressed)};
                guest_descriptor_queue.AddImage(storage);
                is_rescaled |= texture_cache.IsRescaling(view);
            }
            rescaling.PushImage(is_rescaled);
        }
    }

    if (guest_descriptor_queue.UpdateCount() != num_descriptor_entries) {
        LOG_CRITICAL(Render_Vulkan, "Compute descriptor count mismatch: expected {}, got {}",
                     num_descriptor_entries, guest_descriptor_queue.UpdateCount());
        return false;
    }

    if (!is_built.load(std::memory_order::relaxed)) {
        // Wait for the pipeline to be built
        scheduler.Record([this](vk::CommandBuffer) {
            RunBuild();
            std::unique_lock lock{build_mutex};
            build_condvar.wait(lock, [this] { return is_built.load(std::memory_order::relaxed); });
        });
    }

    // Log compute pipeline binding
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogPipelineBind(true, "compute pipeline");
    }

    const DescriptorUpdateEntry* const descriptor_data{guest_descriptor_queue.UpdateData()};
    VkDeviceSize descriptor_buffer_offset{};
    u32 descriptor_buffer_chunk{};
    if (uses_descriptor_buffer) {
        const DescriptorBufferRing::Allocation alloc{
            descriptor_buffer_ring.Allocate(scheduler, descriptor_buffer_layout.size)};
        if (!alloc.host) {
            LOG_DEBUG(Render_Vulkan, "Failed to reserve descriptor memory, skipping dispatch");
            return false;
        }
        WriteDescriptorBuffer(device, descriptor_buffer_layout, descriptor_data, alloc.host);
        descriptor_buffer_offset = alloc.offset;
        descriptor_buffer_chunk = alloc.chunk;
    }

    const bool bind_descriptor_buffer{
        uses_descriptor_buffer && scheduler.UpdateDescriptorBufferChunk(descriptor_buffer_chunk)};

    if (shadow) {
        image_index = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                      Shader::NumDescriptors(info.image_buffer_descriptors) +
                      Shader::NumDescriptors(info.texture_descriptors);
        for (const auto& desc : info.image_descriptors) {
            for (u32 i = 0; i < desc.count; ++i, ++image_index) {
                if (desc.is_written) {
                    texture_cache.MarkModification(
                        texture_cache.GetImageView(views[image_index].id).image_id);
                }
            }
        }
    }

    const bool is_rescaling = !info.texture_descriptors.empty() || !info.image_descriptors.empty();
    if (HangTrace::ResourcesEnabled()) {
        // Copy metadata now: descriptor queue storage can be reused after the worker consumes it.
        const auto buffers = guest_descriptor_queue.TraceBuffers();
        const auto& cb0 = qmd.const_buffer_config[0];
        const u64 cb0_address = cb0.Address();
        std::array<u64, 2> values{};
        u64 valid{};
        constexpr std::array<u32, 2> offsets{0xc0, 0x150};
        for (size_t i = 0; i < offsets.size(); ++i) {
            if ((qmd.const_buffer_enable_mask & 1) &&
                cb0.size.Value() >= offsets[i] + sizeof(u64) &&
                gpu_memory.IsFullyMappedRange(cb0_address + offsets[i], sizeof(u64))) {
                // Observe guest RAM without introducing a GPU readback into the experiment.
                gpu_memory.ReadBlockUnsafe(cb0_address + offsets[i], &values[i], sizeof(u64));
                valid |= u64{1} << i;
            }
        }
        const u64 cb0_size = cb0.size.Value();
        const u64 descriptor_mode = uses_descriptor_buffer ? 2 : (uses_push_descriptor ? 1 : 0);
        const u64 robust = device.IsRobustBufferAccessEnabled();
        const u64 global = info.uses_global_memory;
        const u64 native_bounds = info.native_global_bounds;
        std::vector<std::array<u64, 8>> sources;
        u64 entry{}, binding{};
        for (const auto& desc : info.constant_buffer_descriptors) {
            sources.push_back({shader_hash, entry, binding++, desc.count,
                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, desc.index, 0, ~u64{0}});
            entry += desc.count;
        }
        for (const auto& desc : info.storage_buffers_descriptors) {
            sources.push_back({shader_hash, entry, binding++, desc.count,
                               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, desc.cbuf_index,
                               desc.cbuf_offset, desc.indirect_offset});
            entry += desc.count;
        }
        std::vector<HangTrace::BufferAllocation> allocations;
        for (const auto& b : buffers) {
            allocations.push_back(HangTrace::FindBuffer(b.buffer));
        }
        std::array<u64, 3> upload_values{};
        u64 upload_entry = ~u64{0};
        for (const auto& source : sources) {
            if (source[4] != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || source[5] != 0) {
                continue;
            }
            for (const auto& b : buffers) {
                if (b.entry == source[1]) {
                    upload_entry = b.entry;
                    upload_values = HangTrace::SampleUpload(b.buffer, b.offset, b.size);
                }
            }
        }
        scheduler.Record([buffers, allocations, upload_values, upload_entry,
                          values, valid, cb0_address, cb0_size, descriptor_mode,
                          robust, global, native_bounds, descriptor_buffer_chunk,
                          descriptor_buffer_offset, sources = std::move(sources),
                          shader = shader_hash](vk::CommandBuffer cmdbuf) {
            HangTrace::Resource("compute-descriptors cmd,shader,mode,chunk,offset,global,native_bounds,robust",
                {HangTrace::Handle(*cmdbuf), shader, descriptor_mode, descriptor_buffer_chunk,
                 descriptor_buffer_offset, global, native_bounds, robust});
            HangTrace::Resource("compute-cb0-guest cmd,shader,addr,size,valid,c0_c0,c0_150,robust",
                {HangTrace::Handle(*cmdbuf), shader, cb0_address, cb0_size, valid,
                 values[0], values[1], robust});
            for (const auto& source : sources) {
                HangTrace::Resource("compute-source shader,entry,binding,count,type,cbuf,offset,indirect",
                                    source);
            }
            HangTrace::Resource("compute-cb0-upload cmd,shader,entry,valid,c0_c0,c0_150",
                {HangTrace::Handle(*cmdbuf), shader, upload_entry, upload_values[0],
                 upload_values[1], upload_values[2]});
            for (size_t i = 0; i < buffers.size(); ++i) {
                const auto& b = buffers[i];
                const auto& a = allocations[i];
                HangTrace::Resource("compute-allocation buffer,generation,size,memory,memory_offset,allocation_size,properties,worker_generation",
                    {b.buffer, a.generation, a.size, a.memory, a.offset, a.allocation_size,
                     a.properties, HangTrace::FindBuffer(b.buffer).generation});
                HangTrace::Resource("compute-buffer cmd,shader,entry,buffer,offset,range,address,mode",
                    {HangTrace::Handle(*cmdbuf), shader, b.entry, b.buffer, b.offset,
                     b.size, b.address, descriptor_mode});
            }
        });
    }
    scheduler.Record([this, descriptor_data, is_rescaling, descriptor_buffer_offset,
                      descriptor_buffer_chunk, bind_descriptor_buffer,
                      rescaling_data = rescaling.Data()](vk::CommandBuffer cmdbuf) {
        if (bind_descriptor_buffer) {
            const VkDescriptorBufferBindingInfoEXT binding_info{
                descriptor_buffer_ring.BindingInfo(descriptor_buffer_chunk)};
            cmdbuf.BindDescriptorBuffersEXT(binding_info);
        }
        if (!pipeline) {
            return;
        }
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
        HangTrace::Bind(true, HangTrace::Handle(*pipeline), 0, shader_hash);
        if (!descriptor_set_layout) {
            return;
        }
        if (is_rescaling) {
            cmdbuf.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 RESCALING_LAYOUT_WORDS_OFFSET, sizeof(rescaling_data),
                                 rescaling_data.data());
        }
        if (uses_descriptor_buffer) {
            const u32 buffer_index{};
            cmdbuf.SetDescriptorBufferOffsetsEXT(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout,
                                                 0, buffer_index, descriptor_buffer_offset);
        } else if (uses_push_descriptor) {
            cmdbuf.PushDescriptorSetWithTemplateKHR(*descriptor_update_template, *pipeline_layout,
                                                    0, descriptor_data);
        } else {
            const VkDescriptorSet descriptor_set{descriptor_allocator.Commit()};
            const vk::Device& dev{device.GetLogical()};
            dev.UpdateDescriptorSet(descriptor_set, *descriptor_update_template, descriptor_data);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0,
                                      descriptor_set, nullptr);
        }
    });
    return true;
}

} // namespace Vulkan
