// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>

#include "common/common_types.h"

namespace Core::NceFaultStats {

/// NovaS diagnostics: what the guest instruction was when an NCE access fault reached the GPU's
/// cached memory (Memory::InvalidateNCE with rasterizer pages). Written from the signal handler
/// (plain atomics only), read by the Android memory line.
enum Class : std::size_t {
    Read,          ///< ESR WnR clear
    StoreSingle,   ///< STR/STUR/STRB/STRH (immediate, unscaled, pre/post, register offset)
    StorePair,     ///< STP/STNP
    StoreSimd,     ///< single or pair, SIMD/FP register
    StoreExclusive,///< STXR/STLXR/STLR and the other exclusive/ordered forms
    StoreAtomic,   ///< CAS, SWP, LDADD and the other LSE atomics
    StoreOther,    ///< any other write
    Count,
};

inline std::array<std::atomic<u64>, Count> counts{};
inline thread_local Class last_class = Read;

inline Class Classify(u32 inst, bool is_write) {
    if (!is_write) {
        return Read;
    }
    if ((inst & 0x3F000000) == 0x08000000) {
        return StoreExclusive;
    }
    if ((inst & 0x3B200C00) == 0x38200000) {
        return StoreAtomic;
    }
    const bool simd = (inst & 0x04000000) != 0;
    if ((inst & 0x3A000000) == 0x28000000) {
        return simd ? StoreSimd : StorePair;
    }
    if ((inst & 0x3B000000) == 0x39000000 || (inst & 0x3B200000) == 0x38000000 ||
        (inst & 0x3B200C00) == 0x38200800) {
        return simd ? StoreSimd : StoreSingle;
    }
    return StoreOther;
}

} // namespace Core::NceFaultStats
