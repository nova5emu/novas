// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include "common/common_types.h"
#include "common/gpu_transition_profile.h"

namespace Core::NceExactStore {
using Result = Common::GpuTransitionProfile::DeferredFallbackReason;

// DC ZVA has a bounded, implicit payload; ordinary queued stores remain at 32 B.
inline constexpr std::array<u8, 2048> DcZvaZeros{};

struct Store {
    u64 address{};
    u64 writeback_value{};
    std::array<u8, 32> bytes{};
    u32 size{};
    u32 base_register{};
    bool writeback{};
    bool pair{};
    bool simd{};
    bool zero_block{};
};
} // namespace Core::NceExactStore

#ifdef HAS_NCE

#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <cstddef>

namespace Common {
struct PageTable;
}

namespace Core::NceExactStore {

// Consumed by diagnostics outside the signal handler. Fallbacks count rasterizer faults only.
inline std::atomic<u64> emulated_single{};
inline std::atomic<u64> emulated_pair{};
inline std::atomic<u64> emulated_simd_single{};
inline std::atomic<u64> emulated_simd_pair{};
inline std::atomic<u64> emulated_dc_zva{}, emulated_dc_zva_bytes{};
inline std::atomic<u64> dc_zva_detected{};
// DCZID_EL0[4:0]: DZP in bit 4, log2(block size in words) in bits 3:0.
inline std::array<std::atomic<u64>, 32> dc_zva_dczid{};
inline std::array<std::atomic<u64>, static_cast<std::size_t>(Result::Count)> fallbacks{};
static_assert(std::atomic<u64>::is_always_lock_free);
static_assert(std::atomic<bool>::is_always_lock_free);

struct OtherOpcode {
    // Zero is empty; otherwise key - 1 is the full instruction, including Rt.
    std::atomic<u64> key{}, count{};
};
// Host context only, once per deferred fault (not its fallback retry). Keys are never reset;
// diagnostics may exchange counts with zero while producers continue to increment them.
inline std::array<OtherOpcode, 32> other_opcodes{};
inline std::atomic<u64> other_opcodes_seen{}, other_opcodes_overflow{}, opcode_read_failed{};

inline void RecordOtherOpcode(u32 instruction) {
    other_opcodes_seen.fetch_add(1, std::memory_order_relaxed);
    const u64 key = static_cast<u64>(instruction) + 1;
    for (auto& entry : other_opcodes) {
        u64 expected = entry.key.load(std::memory_order_relaxed);
        if (expected == 0 &&
            entry.key.compare_exchange_strong(expected, key, std::memory_order_relaxed)) {
            expected = key;
        }
        if (expected == key) {
            entry.count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    other_opcodes_overflow.fetch_add(1, std::memory_order_relaxed);
}


enum class FaultReason : std::size_t {
    SignalCode, ExceptionClass, FaultStatus, PcEqualsAddress, Count
};
inline constexpr std::array fault_reason_names{"signal_code", "exception_class", "fault_status",
                                               "pc_equals_address"};
static_assert(fault_reason_names.size() == static_cast<std::size_t>(FaultReason::Count));
inline std::array<std::atomic<u64>, static_cast<std::size_t>(FaultReason::Count)> unsupported_reasons{};
inline std::array<std::atomic<u64>, 64> unsupported_ec{}, unsupported_fsc{};
inline std::array<std::atomic<u64>, 2> unsupported_wnr{};
// si_code buckets: SEGV_ACCERR, SEGV_MAPERR, everything else.
inline std::array<std::atomic<u64>, 3> unsupported_si_code{};
inline std::atomic<u64> retry_bypassed{}, retry_mismatch{};

struct Timing {
    std::atomic<u64> count{}, total_ns{}, max_ns{};

    void Add(u64 ns) {
        count.fetch_add(1, std::memory_order_relaxed);
        total_ns.fetch_add(ns, std::memory_order_relaxed);
        auto maximum = max_ns.load(std::memory_order_relaxed);
        while (maximum < ns &&
               !max_ns.compare_exchange_weak(maximum, ns, std::memory_order_relaxed)) {
        }
    }
};

// Host context only. No clock reads or timing updates from the signal handler.
inline u64 NowNs() {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}
// Host: return from guest through CompleteExactStore, including rejected attempts. Resume/total
// end just before reentering guest on the same core/thread; switched threads count as skipped.
inline Timing timing_host{}, timing_resume{}, timing_total{};
// All queued callbacks, including rejections. Queue includes enqueueing; wake ends at wait return.
inline Timing timing_gpu_queue{}, timing_gpu_callback{}, timing_gpu_wake{}, timing_gpu_round_trip{};
inline std::atomic<u64> resume_skipped{};
inline std::atomic<u64> inline_copies{}, inline_upload_copies{}, inline_render_pass_ends{};
inline thread_local bool in_exact_inline{};

class InlineScope {
public:
    InlineScope() : previous{in_exact_inline} {
        in_exact_inline = true;
    }
    ~InlineScope() {
        in_exact_inline = previous;
    }

private:
    bool previous;
};

inline constexpr auto& fallback_names = Common::GpuTransitionProfile::deferred_fallback_names;
static_assert(fallback_names.size() == static_cast<std::size_t>(Result::Count));


// Per-page store budget (signal handler: lock-free, clock_gettime is async-signal-safe).
// Pages are hashed into a small table; a collision only resets the count of the other page.
// EDEN_NX2_NCE_STORE_BUDGET=<n>: stores per page and window before taking the download path.
// Off by default: on the RedMagic Astra each such download waited ~4.5 ms (186 a second), far
// more than emulating every store (~57 us each). Read once, outside the signal handler.
inline const u32 StoreBudget = [] {
    const char* value = std::getenv("EDEN_NX2_NCE_STORE_BUDGET");
    return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0U;
}();
inline constexpr u64 BudgetWindowNs = 16'000'000;
struct PageBudgetEntry {
    std::atomic<u64> tag{}; ///< page | window << 40 is not needed: page and window kept apart
    std::atomic<u64> window{};
    std::atomic<u32> count{};
};
inline std::array<PageBudgetEntry, 4096> page_budget{};

inline bool WithinPageBudget(u64 page) {
    if (StoreBudget == 0) {
        return true;
    }
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    const u64 window = (static_cast<u64>(now.tv_sec) * 1'000'000'000ULL +
                        static_cast<u64>(now.tv_nsec)) / BudgetWindowNs;
    auto& entry = page_budget[(page >> 12) & (page_budget.size() - 1)];
    if (entry.tag.load(std::memory_order_relaxed) != page ||
        entry.window.load(std::memory_order_relaxed) != window) {
        entry.tag.store(page, std::memory_order_relaxed);
        entry.window.store(window, std::memory_order_relaxed);
        entry.count.store(1, std::memory_order_relaxed);
        return true;
    }
    return entry.count.fetch_add(1, std::memory_order_relaxed) + 1 <= StoreBudget;
}

inline void CountFallback(Result result) noexcept {
    fallbacks[static_cast<std::size_t>(result)].fetch_add(1, std::memory_order_relaxed);
}

struct Backing {
    const Common::PageTable* page_table{};
    u64 address{};
    u8* pointer{};
    u16 block{};
    u32 element_size{};
    bool pair{};
    bool zero_block{};
    // B's temporary Ready path has a stable CPU identity/protection proof.
    bool allow_clean_page{};
};

} // namespace Core::NceExactStore

#endif
