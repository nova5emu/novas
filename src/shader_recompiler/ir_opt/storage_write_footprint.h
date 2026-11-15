// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <compare>
#include <optional>
#include "shader_recompiler/shader_info.h"

namespace Shader::IR {
struct Program;
}

namespace Shader::Optimization {
struct StorageWriteInterval {
    u64 begin{}, end{};
    auto operator<=>(const StorageWriteInterval&) const = default;
};
struct EvaluatedStorageWriteFootprint {
    bool known{};
    StorageWriteReason reason{StorageWriteReason::NotCollected};
    std::vector<StorageWriteInterval> intervals;
};

EvaluatedStorageWriteFootprint EvaluateStorageWriteFootprint(
    const StorageWriteFootprint& footprint, const std::array<u32, 3>& grid,
    const std::array<u32, 3>& block, bool indirect, std::optional<u32> base_low, u64 binding_size);

bool StorageWriteFootprintsEnabled();
bool StorageWriteFootprintProfilingEnabled();
bool ValidStorageWriteIntervals(std::span<const StorageWriteInterval> intervals, u64 binding_size);
// Input is a sorted, disjoint proof. Merging only adds potentially written bytes.
bool CoalesceStorageWriteIntervals(std::vector<StorageWriteInterval>& intervals,
                                   u64 max_gap = 128, size_t max_intervals = 64);
bool CanApplyStorageWriteFootprint(const StorageWriteFootprint& recipe,
                                  const EvaluatedStorageWriteFootprint& evaluated,
                                  u64 binding_address, u64 binding_size, bool split);

// Read-only analysis of the IR. Unknown remains the default for every unsupported access.
void CollectStorageWriteFootprints(IR::Program& program);
void CollectStorageWriteFootprintsIfEnabled(IR::Program& program);
}
