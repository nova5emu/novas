// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Core::NceFaultPolicy {

enum class Decision { Unsupported, Read, Write };

constexpr Decision Classify(bool access_error, u64 esr, bool cached_backing,
                            bool pc_equals_address, bool deferred_enabled) {
    if (!access_error || !cached_backing || pc_equals_address || ((esr >> 26) & 0x3f) != 0x24)
        return Decision::Unsupported;
    const bool write = (esr & (u64{1} << 6)) != 0;
    const u64 family = esr & 0x3c;
    if (family == 0x0c) return write ? Decision::Write : Decision::Read;
    if (deferred_enabled && write && family == 0x04) return Decision::Write;
    return Decision::Unsupported;
}

} // namespace Core::NceFaultPolicy
