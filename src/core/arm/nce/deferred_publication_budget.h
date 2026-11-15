// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include "common/common_types.h"

namespace Core {

// Host-continuation policy only. A rejected publication may complete exactly under Ready;
// a failed exact completion still requires the ordered native fallback.
class DeferredPublicationBudget {
public:
    enum class Decision { Allow, PublicationCost, CorePublicationCost, FaultDensity };
    static constexpr u32 SampleCount = 32;
    static constexpr u32 SlowSampleLimit = 8;
    static constexpr u64 MeanLimitNs = 25'000;
    static constexpr u64 CoreMeanLimitNs = 100'000;
    static constexpr u64 WindowNs = 100'000'000;
    static constexpr u64 CooldownNs = WindowNs;
    static constexpr u32 FaultLimit = 4096;
    static constexpr size_t SlotCount = 256;
    struct CoreEvent {
        bool activated{};
        u32 slow_samples{};
        u64 max_ns{};
    };

    std::optional<CoreEvent> TakeCoreEvent() {
        auto event = core_event;
        core_event.reset();
        return event;
    }

    Decision BeforeAttempt(PAddr physical_page, u64 now) {
        if (core_expensive) {
            if (now - core_stopped_at < CooldownNs) return Decision::CorePublicationCost;
            core_expensive = false;
            core_block = {};
            core_event = CoreEvent{false, stopped_slow_samples, stopped_max_ns};
        }
        auto& entry = entries[Slot(physical_page)];
        if (!entry.valid || entry.page != physical_page) {
            entry = {};
            entry.valid = true;
            entry.page = physical_page;
            entry.window_start = now;
        }
        if (entry.expensive) {
            if (now - entry.expensive_since < CooldownNs) return Decision::PublicationCost;
            entry.expensive = false;
            entry.block = {};
        }
        if (now - entry.window_start >= WindowNs) {
            entry.window_start = now;
            entry.faults = 0;
        }
        if (entry.faults == FaultLimit) return Decision::FaultDensity;
        ++entry.faults;
        return Decision::Allow;
    }

    // Observe successful journal publications, excluding synchronous exact completions.
    // The decision affects a future instruction.
    void Observe(PAddr physical_page, u64 publication_ns, u64 now) {
        if (publication_ns == 0) return;
        if (AddSample(core_block, publication_ns, CoreMeanLimitNs, now)) {
            core_expensive = true;
            core_stopped_at = now;
            stopped_slow_samples = core_block.slow_samples;
            stopped_max_ns = core_block.max_ns;
            core_event = CoreEvent{true, stopped_slow_samples, stopped_max_ns};
        }
        if (core_block.samples == SampleCount) core_block = {};
        auto& entry = entries[Slot(physical_page)];
        if (!entry.valid || entry.page != physical_page) return;
        if (AddSample(entry.block, publication_ns, MeanLimitNs, now)) {
            entry.expensive = true;
            entry.expensive_since = now;
        }
        if (entry.block.samples == SampleCount) entry.block = {};
    }

private:
    struct Block {
        u64 started_at{}, max_ns{};
        u32 samples{}, slow_samples{};
    };
    static bool AddSample(Block& block, u64 duration, u64 limit, u64 now) {
        if (block.samples && now - block.started_at >= WindowNs) block = {};
        if (!block.samples) block.started_at = now;
        ++block.samples;
        block.slow_samples += duration > limit;
        block.max_ns = (std::max)(block.max_ns, duration);
        return block.samples == SampleCount && block.slow_samples >= SlowSampleLimit;
    }
    static constexpr size_t Slot(PAddr page) { return (page >> 12) % SlotCount; }
    struct Entry {
        PAddr page{};
        u64 window_start{}, expensive_since{};
        u32 faults{};
        Block block{};
        bool valid{}, expensive{};
    };
    // Page collisions never discard the core block or cooldown.
    std::array<Entry, SlotCount> entries{};
    Block core_block{};
    u64 core_stopped_at{}, stopped_max_ns{};
    u32 stopped_slow_samples{};
    bool core_expensive{};
    std::optional<CoreEvent> core_event;
};

} // namespace Core
