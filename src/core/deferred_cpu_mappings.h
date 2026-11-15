// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>
#include "common/common_types.h"

namespace Core {

// CPU identities for B. Device mappings have a separate registry. Every caller holding this
// gate must finish without waiting for the GPU; outer transitions drain before taking it.
class DeferredCpuMappings {
public:
    enum class Status {
        Eligible, Disabled, MappingChanged, Alias, NotPrivateHeap,
        Shared, Transfer, Code, CapturedWriter,
    };
    struct Snapshot { u64 generation{}; };

    explicit DeferredCpuMappings(u64 physical_size = 0, bool enabled_ = false)
        : enabled{enabled_}, counts(enabled_ ? physical_size / PageSize : 0),
          exclusions(counts.size()), generations(counts.size()) {}

    bool Enabled() const { return enabled; }

    template <typename F>
    decltype(auto) WithGate(F&& operation) {
        std::scoped_lock lock{gate};
        return operation();
    }

    // The callback must additionally validate this VA's current pointer/protection under the
    // gate. A count of one by itself cannot prove which VA maps the expected physical page.
    template <typename F>
    Status WithPage(const void* owner, VAddr address, PAddr physical, F&& operation) {
        if (!enabled) return Status::Disabled;
        std::scoped_lock lock{gate};
        if (exhausted || (physical & PageMask) || physical / PageSize >= counts.size())
            return Status::MappingChanged;
        const size_t index = physical / PageSize;
        if (exclusions[index] != 0) return static_cast<Status>(exclusions[index]);
        if (counts[index] != 1) return counts[index] ? Status::Alias : Status::MappingChanged;
        const auto it = heaps.find(owner);
        const VAddr page = address & ~PageMask;
        if (it == heaps.end() || page < it->second.first || page >= it->second.second)
            return Status::NotPrivateHeap;
        operation(Snapshot{generations[index]});
        return Status::Eligible;
    }

    // OldPhysical resolves under the gate; Mutation changes the real CPU table under it too.
    template <typename OldPhysical, typename Mutation>
    void WithMapping(VAddr address, u64 page_count, std::optional<PAddr> physical,
                     OldPhysical&& old_physical, Mutation&& mutation) {
        if (!enabled) { mutation(); return; }
        std::scoped_lock lock{gate};
        AdvanceGeneration();
        for (u64 i = 0; i < page_count; ++i) {
            if (const auto old = old_physical(address + i * PageSize)) Remove(*old);
            if (physical) Add(*physical + i * PageSize);
        }
        mutation();
    }

    void AllowHeapRange(const void* owner, VAddr begin, VAddr end) {
        if (!enabled) return;
        std::scoped_lock lock{gate};
        AdvanceGeneration();
        if (begin <= end && !(begin & PageMask) && !(end & PageMask)) heaps[owner] = {begin, end};
        else { heaps.erase(owner); exhausted = true; }
    }

    // Finalization still needs the original table for MakePageGroup. Remove the physical
    // references before releasing those page groups, without traversing unmapped VA holes.
    void RemoveFinalizedRange(PAddr physical, u64 size) {
        if (!enabled) return;
        std::scoped_lock lock{gate};
        AdvanceGeneration();
        if ((physical & PageMask) || (size & PageMask)) { exhausted = true; return; }
        for (u64 offset = 0; offset < size; offset += PageSize) Remove(physical + offset);
    }

    void RemoveOwner(const void* owner) {
        if (!enabled) return;
        std::scoped_lock lock{gate};
        heaps.erase(owner);
        AdvanceGeneration();
    }

    // A durable captured writer must be excluded before exposing its pointer. This metadata
    // is not a drain: the caller already holds the outer Ready transition or writer permission.
    void ExcludePhysical(PAddr physical, u64 size, Status reason) {
        if (!enabled || size == 0) return;
        std::scoped_lock lock{gate};
        AdvanceGeneration();
        ExcludePhysicalLocked(physical, size, reason);
    }

    template <typename Resolve>
    void ExcludeVirtualRange(VAddr address, u64 size, Resolve&& resolve, Status reason) {
        if (!enabled || !size) return;
        std::scoped_lock lock{gate};
        AdvanceGeneration();
        if (size - 1 > (std::numeric_limits<u64>::max)() - address) { exhausted = true; return; }
        const VAddr last = (address + size - 1) & ~PageMask;
        for (VAddr page = address & ~PageMask;; page += PageSize) {
            if (const auto physical = resolve(page)) ExcludePhysicalLocked(*physical, PageSize, reason);
            if (page == last) break;
        }
    }

private:
    static constexpr u64 PageSize = 4096;
    static constexpr u64 PageMask = PageSize - 1;
    void ExcludePhysicalLocked(PAddr physical, u64 size, Status reason) {
        if (reason < Status::Shared || reason > Status::CapturedWriter ||
            size - 1 > (std::numeric_limits<u64>::max)() - physical) {
            exhausted = true;
            return;
        }
        const u64 first = physical / PageSize;
        const u64 last = (physical + size - 1) / PageSize;
        if (last >= exclusions.size()) { exhausted = true; return; }
        for (u64 page = first; page <= last; ++page) {
            // Shared/transfer have priority over a generic permission exclusion. Nothing
            // reopens when a captured writer releases its object or a physical page is reused.
            if (!exclusions[page] || static_cast<u8>(reason) < exclusions[page])
                exclusions[page] = static_cast<u8>(reason);
            generations[page] = generation;
        }
    }
    void AdvanceGeneration() {
        if (generation == (std::numeric_limits<u64>::max)()) exhausted = true;
        else ++generation;
    }
    void Add(PAddr physical) {
        const u64 index = physical / PageSize;
        if ((physical & PageMask) || index >= counts.size() ||
            counts[index] == (std::numeric_limits<u32>::max)()) { exhausted = true; return; }
        ++counts[index];
        generations[index] = generation;
    }
    void Remove(PAddr physical) {
        const u64 index = physical / PageSize;
        if ((physical & PageMask) || index >= counts.size() || counts[index] == 0) {
            exhausted = true;
            return;
        }
        --counts[index];
        generations[index] = generation;
    }
    const bool enabled;
    std::mutex gate;
    std::vector<u32> counts;
    std::vector<u8> exclusions;
    std::vector<u64> generations;
    std::map<const void*, std::pair<VAddr, VAddr>> heaps;
    u64 generation{1};
    bool exhausted{};
};
} // namespace Core
