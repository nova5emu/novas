// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <optional>

#include "common/common_types.h"
#include "common/settings.h"
#include "video_core/engines/fermi_2d.h"

// The Legend of Zelda: Ocarina of Time's anti-aliasing. After tonemapping, the game runs a
// single-frame tensor network on the tonemapped image and an output pass that sharpens the result
// hard. With the "Ocarina of Time anti-aliasing" setting at No sharpening, the network is skipped
// and its output pass is replaced by a copy of the tonemapped image
// (docs/nx2/oot-network-bypass.md).
//
// Shader hashes, native program keys and constant buffer offsets are those of the game's main
// with build id E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348; a game update needs them re-checked.
namespace VideoCommon::OotNetwork {

constexpr u64 ProgramId = 0x0400486024DEC000ULL;

// Native program keys of the network's compute passes.
constexpr u64 PrepassProgramKey = 0x8475d671e6967428ULL;
constexpr u64 NetworkProgramKey = 0x42593194bb0b2b32ULL;
constexpr u64 OutputProgramKey = 0xda6df3e54c716df2ULL;
constexpr std::array<u64, 3> OutputProgramKeys{
    OutputProgramKey,
    0x6f9367bd41dc0b58ULL,
    0xd8efd9f0c95abd80ULL,
};

constexpr bool IsOutputProgram(u64 key) {
    return key == OutputProgramKeys[0] || key == OutputProgramKeys[1] ||
           key == OutputProgramKeys[2];
}

// Translated Android output hash, kept separate from native program keys.
constexpr u64 HandheldOutputShaderHash = 0xe40f5efb5480e692ULL;

constexpr bool ShouldSkipIntermediate(u64 native_key, bool output_replaced) {
    return output_replaced && (native_key == PrepassProgramKey || native_key == NetworkProgramKey);
}

using CopyEngine = Tegra::Engines::Fermi2D;

constexpr bool IsCopySurfaceValid(const CopyEngine::Surface& surface) {
    return surface.Address() != 0 && surface.format == Tegra::RenderTargetFormat::B10G11R11_FLOAT &&
           surface.linear == CopyEngine::MemoryLayout::BlockLinear && surface.depth == 1 &&
           surface.layer == 0 && surface.width > 0 && surface.width <= 8192 &&
           surface.height > 0 && surface.height <= 8192 && surface.block_width.Value() == 0 &&
           surface.block_height.Value() <= 5 && surface.block_depth.Value() == 0;
}

inline std::optional<CopyEngine::Config> MakeCopyConfig(const CopyEngine::Surface& source,
                                                     const CopyEngine::Surface& destination) {
    if (!IsCopySurfaceValid(source) || !IsCopySurfaceValid(destination) ||
        source.Address() == destination.Address()) {
        return std::nullopt;
    }
    // Relative aspect tolerance is 2%, with a scale between 0.5x and 4x per axis.
    const s64 aspect_error = s64{source.width} * destination.height -
                             s64{destination.width} * source.height;
    const s64 aspect_reference = s64{destination.width} * source.height;
    if (aspect_error * 100 < -aspect_reference * 2 || aspect_error * 100 > aspect_reference * 2 ||
        source.width > destination.width * 2 || destination.width > source.width * 4 ||
        source.height > destination.height * 2 || destination.height > source.height * 4) {
        return std::nullopt;
    }
    const bool scaled = source.width != destination.width || source.height != destination.height;
    return CopyEngine::Config{
        .operation = CopyEngine::Operation::SrcCopy,
        .filter = scaled ? CopyEngine::Filter::Bilinear : CopyEngine::Filter::Point,
        .must_accelerate = true,
        .dst_x0 = 0,
        .dst_y0 = 0,
        .dst_x1 = static_cast<s32>(destination.width),
        .dst_y1 = static_cast<s32>(destination.height),
        .src_x0 = 0,
        .src_y0 = 0,
        .src_x1 = static_cast<s32>(source.width),
        .src_y1 = static_cast<s32>(source.height),
    };
}

/// Constant buffer 0 offset of the output pass's texture handle for the B10G11R11 image it writes
/// (the scene image the rest of the frame uses).
constexpr u32 OutputImageHandleOffset = 0xd0;

/// Fragment shaders of the game's tonemap passes; render target 0 of their draws is the
/// network's input.
constexpr bool IsTonemapShader(u64 hash) {
    return hash == 0xbded5ce917183214ULL || // outdoors
           hash == 0x6cb06edf14be1d79ULL || // indoors
           hash == 0x3ac011bd2d772994ULL;   // outdoors at another time of day
}

/// Owned by the rasterizer: the tonemap pass draws on the 3D engine, and the network's passes do
/// not all run on the same compute engine.
struct BypassState {
    std::atomic<GPUVAddr> tonemapped_image{}; ///< render target 0 of the latest tonemap draw
    std::atomic<bool> output_replaced{};      ///< the latest output pass was replaced
    std::atomic<u32> reported_output_states{};
    std::atomic<u32> reported_qmd_releases{};

    void SetTonemappedSurface(std::optional<CopyEngine::Surface> surface) {
        std::unique_lock lock{tonemap_mutex, std::try_to_lock};
        if (!lock) {
            tonemapped_image.store(0, std::memory_order_relaxed);
            output_replaced.store(false, std::memory_order_relaxed);
            return;
        }
        tonemap_surface = surface;
        tonemapped_image.store(surface ? surface->Address() : 0, std::memory_order_relaxed);
        if (!surface) {
            output_replaced.store(false, std::memory_order_relaxed);
        }
    }

    std::optional<CopyEngine::Surface> GetTonemappedSurface() {
        std::unique_lock lock{tonemap_mutex, std::try_to_lock};
        if (!lock) {
            return std::nullopt;
        }
        if (tonemap_surface && tonemap_surface->Address() !=
                                  tonemapped_image.load(std::memory_order_relaxed)) {
            return std::nullopt;
        }
        return tonemap_surface;
    }

private:
    std::mutex tonemap_mutex;
    std::optional<CopyEngine::Surface> tonemap_surface;
};

/// Whether the running game is Ocarina of Time with its anti-aliasing at No sharpening.
inline bool BypassEnabled(u64 program_id) {
    return program_id == ProgramId &&
           Settings::values.oot_anti_aliasing.GetValue() == Settings::OotAntiAliasing::NoSharpening;
}

} // namespace VideoCommon::OotNetwork
