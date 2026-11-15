// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <span>

#include "common/atomic_ops.h"
#include "common/common_types.h"

namespace Core::ExactStoreBacking {

enum class Validation { Valid, InvalidOperands, Unaligned, PageCrossing };
struct Shape {
    u32 element_size{};
    bool pair{};
    bool zero_block{};
};

inline Validation Validate(u64 address, u8* target, std::span<const u8> bytes,
                           Shape shape) noexcept {
    const u64 size = bytes.size();
    if (!target || size == 0 || (size & (size - 1)) != 0) return Validation::InvalidOperands;
    if (shape.zero_block) {
        if (size < 4 || size > 2048 || (address & (size - 1)) != 0 || shape.pair ||
            shape.element_size != (std::min<u64>)(size, 16) ||
            !std::all_of(bytes.begin(), bytes.end(), [](u8 b) { return b == 0; }))
            return Validation::InvalidOperands;
    } else if (size > 32 || shape.element_size != (shape.pair ? size / 2 : size) ||
               shape.element_size == 0 || shape.element_size > 16) {
        return Validation::InvalidOperands;
    }
    const bool in_block = (address & 15) + size <= 16;
    const bool pair_halves = shape.pair && !in_block && shape.element_size >= 8 &&
                             (address & (shape.element_size - 1)) == 0;
    const bool cross_blocks = !in_block && !pair_halves && size <= 16;
    const bool zero_blocks = shape.zero_block && size >= 16;
    if (((address ^ reinterpret_cast<uintptr_t>(target)) & 15) != 0 ||
        (!zero_blocks && !in_block && !pair_halves && !cross_blocks))
        return Validation::Unaligned;
    if (size > 4096 - (address & 4095)) return Validation::PageCrossing;
    return Validation::Valid;
}

template <typename T>
inline void StoreScalar(u8* target, const u8* bytes) noexcept {
    T value{};
    std::memcpy(&value, bytes, sizeof(T));
#if defined(_MSC_VER)
    T expected{}, actual{};
    while (!Common::AtomicCompareAndSwap(reinterpret_cast<T*>(target), value, expected, actual))
        expected = actual;
#else
    static_assert(__atomic_always_lock_free(sizeof(T), 0));
    __atomic_store_n(reinterpret_cast<T*>(target), value, __ATOMIC_RELAXED);
#endif
}

inline void Store128(u8* target, const u8* bytes) noexcept {
    u128 value{}, expected{}, actual{};
    std::memcpy(value.data(), bytes, sizeof(value));
    while (!Common::AtomicCompareAndSwap(reinterpret_cast<u64*>(target), value, expected, actual))
        expected = actual;
}

// Caller holds the mapping/admission proof. No allocation, cache calls, logging or GPU waits.
// A rejected shape does not touch backing. Masked CAS preserves neighboring CPU-visible bytes.
inline bool Commit(u64 address, u8* target, std::span<const u8> bytes, Shape shape) noexcept {
    if (Validate(address, target, bytes, shape) != Validation::Valid) return false;
    const u64 size = bytes.size();
    const bool natural = (address & (size - 1)) == 0;
    const bool in_block = (address & 15) + size <= 16;
    const bool pair_halves = shape.pair && !in_block && shape.element_size >= 8 &&
                             (address & (shape.element_size - 1)) == 0;
    if (pair_halves || (shape.zero_block && size >= 16)) {
        for (u64 offset = 0; offset < size; offset += shape.element_size) {
            if (shape.element_size == 8) StoreScalar<u64>(target + offset, bytes.data() + offset);
            else Store128(target + offset, bytes.data() + offset);
        }
        return true;
    }
    if (!natural) {
        u64 done = 0;
        while (done < size) {
            u8* const part = target + done;
            const u64 offset = reinterpret_cast<uintptr_t>(part) & 15;
            const u64 length = (std::min<u64>)(size - done, 16 - offset);
            auto* const block = reinterpret_cast<u64*>(part - offset);
            u128 expected = Common::AtomicLoad128(block);
            while (true) {
                u128 desired = expected;
                std::memcpy(reinterpret_cast<u8*>(desired.data()) + offset, bytes.data() + done, length);
                u128 actual{};
                if (Common::AtomicCompareAndSwap(block, desired, expected, actual)) break;
                expected = actual;
            }
            done += length;
        }
        return true;
    }
    switch (size) {
    case 1: StoreScalar<u8>(target, bytes.data()); break;
    case 2: StoreScalar<u16>(target, bytes.data()); break;
    case 4: StoreScalar<u32>(target, bytes.data()); break;
    case 8: StoreScalar<u64>(target, bytes.data()); break;
    case 16: Store128(target, bytes.data()); break;
    default: return false;
    }
    return true;
}

} // namespace Core::ExactStoreBacking

#ifdef HAS_NCE
#include "core/arm/nce/nce_exact_store.h"

namespace Core::NceExactStore {

inline Result ValidateBackingStore(DAddr address, std::span<const u8> bytes,
                                    const Backing& backing) noexcept {
    using ExactStoreBacking::Validation;
    switch (ExactStoreBacking::Validate(address, backing.pointer, bytes,
            {backing.element_size, backing.pair, backing.zero_block})) {
    case Validation::Valid: return Result::Success;
    case Validation::InvalidOperands: return Result::InvalidOperands;
    case Validation::Unaligned: return Result::Unaligned;
    case Validation::PageCrossing: return Result::PageCrossing;
    }
    return Result::InvalidOperands;
}

inline bool CommitBackingStore(DAddr address, std::span<const u8> bytes,
                                const Backing& backing) noexcept {
    return ExactStoreBacking::Commit(address, backing.pointer, bytes,
        {backing.element_size, backing.pair, backing.zero_block});
}

} // namespace Core::NceExactStore
#endif
