// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include "common/common_types.h"

namespace Tegra::SemaphoreAcquire {

// Preserve SemaphoreTrigger's values; SEM_EXECUTE uses a different encoding.
enum Operation : u32 {
    Equal = 1,
    Gequal = 4,
    MaskNonzero = 8,
    SignedDifferenceGequal = 0x100,
    MaskZero = 0x101,
};

constexpr std::optional<u32> NormalizeExecute(u32 operation) {
    switch (operation) {
    case 0: return Equal;
    case 2: return Gequal;
    case 3: return SignedDifferenceGequal;
    case 4: return MaskNonzero;
    case 5: return MaskZero;
    default: return std::nullopt;
    }
}

constexpr bool Matches(u32 operation, u32 value, u32 current) {
    switch (operation) {
    case Equal: return current == value;
    case Gequal: return current >= value;
    case MaskNonzero: return (current & value) != 0;
    case SignedDifferenceGequal: return ((current - value) & 0x80000000U) == 0;
    case MaskZero: return (current & value) == 0;
    default: return false;
    }
}

} // namespace Tegra::SemaphoreAcquire
