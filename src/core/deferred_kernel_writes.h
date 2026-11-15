// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "video_core/deferred_write_journal.h"

namespace Core {

// Process-lifetime roles for kernel/JIT locked accesses and native NCE fallback accesses.
// Old physical identities stay excluded too: a concurrent old writer may retain its pointer.
class DeferredKernelWrites {
public:
    using Permission = std::shared_ptr<VideoCommon::DeferredWrites::Journal::WriterPermission>;
    static constexpr size_t FastTableMask = (1u << 18) - 1;

    // Tagged cache of permanent VA roles, not of a particular physical mapping. WithMapping
    // prepares every replacement identity before exposing it. Collisions only cause misses.
    const std::atomic<u64>* EnableFastTable() {
        std::scoped_lock lock{mutex};
        if (!fast_storage) {
            fast_storage = std::make_unique<std::atomic<u64>[]>(FastTableMask + 1);
            fast_table.store(fast_storage.get(), std::memory_order_release);
        }
        return fast_storage.get();
    }

    [[nodiscard]] bool IsPreparedFast(VAddr address, u64 size) const {
        if (!ValidRange(address, size)) return false;
        if (!size) return true;
        const auto* table = fast_table.load(std::memory_order_acquire);
        if (!table) return false;
        const u64 last = (address + size - 1) / PageSize;
        for (u64 page = address / PageSize;; ++page) {
            if (table[page & FastTableMask].load(std::memory_order_acquire) != page + 1)
                return false;
            if (page == last) return true;
        }
    }

    // Resolve and reserve run under the registry mutex and must not wait for the GPU.
    // PreparePin runs without it, before the caller acquires its kernel locks.
    struct IgnorePhysicalPage {
        void operator()(PAddr, bool) const {}
    };

    template <typename Resolve, typename Reserve, typename PreparePin,
              typename Observe = IgnorePhysicalPage>
    [[nodiscard]] bool Prepare(VAddr address, u64 size, Resolve&& resolve, Reserve&& reserve,
                               PreparePin&& prepare_pin, Observe&& observe = {},
                               bool observe_pages = false) {
        if (!ValidRange(address, size)) return false;
        if (!size) return true;
        const VAddr last = (address + size - 1) & ~PageMask;
        std::vector<Permission> pending;
        {
            std::scoped_lock lock{mutex};
            for (VAddr page = address & ~PageMask;; page += PageSize) {
                const std::optional<PAddr> physical = resolve(page);
                auto& role = virtual_pages[page];
                role = physical;
                if (physical) {
                    const bool newly_excluded = observe_pages && !physical_pages.contains(*physical);
                    auto permission = ReservePhysical(*physical, reserve);
                    if (!permission) return false;
                    if (observe_pages) observe(*physical, newly_excluded);
                    if (!permission->Ready()) pending.push_back(std::move(permission));
                }
                if (page == last) break;
            }
        }
        for (const auto& permission : pending) {
            if (!prepare_pin(permission) || !permission->Ready()) return false;
        }
        // A remap while preparing is allowed only through WithMapping, which installs a
        // ready exclusion before exposing the new identity. Merely finding a role is not Ready.
        if (!IsPrepared(address, size)) return false;
        if (auto* table = fast_table.load(std::memory_order_acquire)) {
            for (u64 page = address / PageSize;; ++page) {
                table[page & FastTableMask].store(page + 1, std::memory_order_release);
                if (page == last / PageSize) break;
            }
        }
        return true;
    }

    // Existing outer mapping transition must already have drained: this path never waits.
    // Invoke mutate while holding the registry mutex, so first registration cannot miss it.
    template <typename Reserve, typename Mutate>
    [[nodiscard]] bool WithMapping(VAddr address, u64 size, std::optional<PAddr> physical,
                                   Reserve&& reserve, Mutate&& mutate) {
        if (!ValidRange(address, size) || (address & PageMask) || (size & PageMask) ||
            (physical && ((*physical & PageMask) || !ValidRange(*physical, size)))) {
            return false;
        }
        std::scoped_lock lock{mutex};
        if (!size) {
            mutate();
            return true;
        }
        const VAddr last = address + size - 1;
        auto first = virtual_pages.lower_bound(address);
        auto end = first;
        for (; end != virtual_pages.end() && end->first <= last; ++end) {
            if (!physical) continue;
            auto permission = ReservePhysical(*physical + (end->first - address), reserve);
            if (!permission || !permission->MarkReady()) return false;
        }
        mutate();
        for (auto it = first; it != end; ++it) {
            it->second = physical ? std::make_optional(*physical + (it->first - address))
                                  : std::nullopt;
        }
        return true;
    }

    [[nodiscard]] bool IsPrepared(VAddr address, u64 size) const {
        if (!ValidRange(address, size)) return false;
        if (!size) return true;
        const VAddr last = (address + size - 1) & ~PageMask;
        std::scoped_lock lock{mutex};
        for (VAddr page = address & ~PageMask;; page += PageSize) {
            const auto role = virtual_pages.find(page);
            if (role == virtual_pages.end()) return false;
            if (role->second) {
                const auto pin = physical_pages.find(*role->second);
                if (pin == physical_pages.end() || !pin->second->Ready()) return false;
            }
            if (page == last) return true;
        }
    }

    // Teardown only: no guest access/Prepare may run concurrently with Clear.
    void Clear() {
        std::map<PAddr, Permission> retired;
        {
            std::scoped_lock lock{mutex};
            virtual_pages.clear();
            physical_pages.swap(retired);
            if (fast_storage) {
                for (size_t i = 0; i <= FastTableMask; ++i)
                    fast_storage[i].store(0, std::memory_order_release);
            }
        }
    }

    [[nodiscard]] size_t VirtualPageCount() const {
        std::scoped_lock lock{mutex};
        return virtual_pages.size();
    }

    [[nodiscard]] size_t PhysicalPageCount() const {
        std::scoped_lock lock{mutex};
        return physical_pages.size();
    }

private:
    static constexpr u64 PageSize = 4096;
    static constexpr u64 PageMask = PageSize - 1;

    static bool ValidRange(u64 address, u64 size) {
        return !size || size - 1 <= (std::numeric_limits<u64>::max)() - address;
    }

    template <typename Reserve>
    Permission ReservePhysical(PAddr physical, Reserve& reserve) {
        if (physical & PageMask) return {};
        if (const auto it = physical_pages.find(physical); it != physical_pages.end()) {
            return it->second;
        }
        auto permission = reserve(physical);
        if (permission) physical_pages.emplace(physical, permission);
        return permission;
    }

    mutable std::mutex mutex;
    std::map<VAddr, std::optional<PAddr>> virtual_pages;
    std::map<PAddr, Permission> physical_pages;
    std::unique_ptr<std::atomic<u64>[]> fast_storage;
    std::atomic<std::atomic<u64>*> fast_table{};
};

} // namespace Core
