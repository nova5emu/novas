// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstring>
#include <optional>
#include <span>
#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"

namespace VideoCommon {

inline constexpr u32 MaxStorageBaseSnapshotSize = 64 * 1024;

inline bool StorageBaseVectorFits(u32 offset, u64 size) {
    const u64 vector_begin = offset & ~u32{15};
    return (offset & 3) == 0 && vector_begin <= size && 16 <= size - vector_begin;
}

inline bool StorageBaseRangesOverlap(u64 a, u64 a_size, u64 b, u64 b_size) {
    if (!a_size || !b_size) return false;
    return a <= b ? b - a < a_size : a - b < b_size;
}

// The entire streamed UBO must come from coherent backing, not just its descriptor word.
inline Shader::StorageWriteReason StorageBaseSnapshotEligibility(
    u32 offset, u32 size, bool valid_binding, bool gpu_modified, bool writable_alias) {
    using Reason = Shader::StorageWriteReason;
    if (!valid_binding || !StorageBaseVectorFits(offset, size)) return Reason::BaseSnapshotBounds;
    if (size > MaxStorageBaseSnapshotSize) return Reason::BaseSnapshotLimit;
    if (gpu_modified) return Reason::BaseGpuModified;
    if (writable_alias) return Reason::BaseWritableAlias;
    return Reason::None;
}

inline std::optional<u32> ReadStorageBaseSnapshot(std::span<const u8> bytes, u32 offset) {
    if (!StorageBaseVectorFits(offset, bytes.size())) return std::nullopt;
    u32 value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

} // namespace VideoCommon
