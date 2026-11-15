// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/gpu_transition_profile.h"
#include "common/range_mutex.h"
#include "common/scratch_buffer.h"
#include "common/thread_worker.h"
#include "common/sparse_large_vector.h"

namespace VideoCommon::CpuEpochs {
enum class Exclusion : u8;
}
namespace Core {

constexpr size_t DEVICE_PAGEBITS = 12ULL;
constexpr size_t DEVICE_PAGESIZE = 1ULL << DEVICE_PAGEBITS;
constexpr size_t DEVICE_PAGEMASK = DEVICE_PAGESIZE - 1ULL;

class DeviceMemory;

namespace Memory {
class Memory;
}

template <typename DTraits>
struct DeviceMemoryManagerAllocator;

struct Asid {
    size_t id;
};

struct DeviceMemoryPhysicalSpan {
    u64 offset; ///< Byte offset from the start of DRAM, not a guest physical address.
    u64 size;
};

template <typename Traits>
class DeviceMemoryManager {
    using DeviceInterface = typename Traits::DeviceInterface;
    using DeviceMethods = typename Traits::DeviceMethods;

public:
    DeviceMemoryManager(const DeviceMemory& device_memory);
    ~DeviceMemoryManager();

    static constexpr bool HAS_FLUSH_INVALIDATION = true;
    static constexpr size_t AS_BITS = Traits::device_virtual_bits;

    void BindInterface(DeviceInterface* device_inter);

    /// Marks the memory behind [address, address + size) cold (Common::HostMemory::HintCold),
    /// from a low-priority thread: data the host GPU has its own copy of, e.g. uploaded textures.
    void HintCold(DAddr address, size_t size);
    void ExcludeCpuEpochExport(DAddr address, u64 size) const;
    void ExcludeCpuEpochRange(DAddr address, u64 size,
                             VideoCommon::CpuEpochs::Exclusion reason) const;

    DAddr Allocate(size_t size);
    void AllocateFixed(DAddr start, size_t size);
    void Free(DAddr start, size_t size);

    void Map(DAddr address, VAddr virtual_address, size_t size, Asid asid, bool track = false);

    // Fresh backing only: validation, permission and publication share mapping_guard.
    // permit must not map/unmap, take cache locks or wait for the GPU.
    template <typename Permit>
    bool TryMapUnbacked(DAddr address, VAddr virtual_address, size_t size, Asid asid,
                        bool track, Permit&& permit) {
        std::scoped_lock lock{mapping_guard};
        if (size == 0 || (address & page_mask) != 0 || (virtual_address & page_mask) != 0 ||
            (size & page_mask) != 0 || address >= device_as_size || size > device_as_size - address) {
            return false;
        }
        const size_t first = address >> page_bits;
        const size_t pages = size >> page_bits;
        for (size_t page = first; page < first + pages; ++page) {
            if (tracked_entries[page].compressed_physical_ptr != 0) {
                return false;
            }
        }
        if (!permit()) {
            return false;
        }
        MapImpl(address, virtual_address, size, asid, track);
        return true;
    }

    // The caller owns the physical pages until every device mapping is retired. Like
    // TryMapUnbacked, permit runs under mapping_guard before any translation is published.
    template <typename Permit>
    bool TryMapPhysicalUnbacked(DAddr address, std::span<const DeviceMemoryPhysicalSpan> spans,
                               Permit&& permit) {
        std::scoped_lock lock{mapping_guard};
        if (spans.empty() || address == 0 || (address & page_mask) != 0 ||
            address >= device_as_size) {
            return false;
        }
        const u64 physical_limit = std::min<u64>(physical_size,
                                                u64{compressed_device_addr.size()} << page_bits);
        u64 size = 0;
        for (const auto& span : spans) {
            if (span.size == 0 || (span.offset & page_mask) != 0 ||
                (span.size & page_mask) != 0 || span.offset >= physical_limit ||
                span.size > physical_limit - span.offset ||
                span.size > device_as_size - address - size) {
                return false;
            }
            size += span.size;
        }
        const size_t first = address >> page_bits;
        const size_t pages = size >> page_bits;
        for (size_t page = first; page < first + pages; ++page) {
            if (tracked_entries[page].compressed_physical_ptr != 0) {
                return false;
            }
        }
        if (!permit()) {
            return false;
        }
        MapPhysicalImpl(address, spans, size);
        return true;
    }

    void Unmap(DAddr address, size_t size);

    void TrackContinuityImpl(DAddr address, VAddr virtual_address, size_t size, Asid asid);
    void TrackContinuity(DAddr address, VAddr virtual_address, size_t size, Asid asid) {
        std::scoped_lock lk(mapping_guard);
        TrackContinuityImpl(address, virtual_address, size, asid);
    }

    // Write / Read
    template <typename T>
    T* GetPointer(DAddr address);

    template <typename T>
    const T* GetPointer(DAddr address) const;

    template <typename Func>
    void ApplyOpOnPAddr(PAddr address, Common::ScratchBuffer<u32>& buffer, Func&& operation) {
        DAddr subbits = DAddr(address & page_mask);
        const u32 base = compressed_device_addr[(address >> page_bits)];
        if ((base >> MULTI_FLAG_BITS) == 0) [[likely]] {
            const DAddr d_address = (DAddr(base) << page_bits) + subbits;
            operation(d_address);
            return;
        }
        InnerGatherDeviceAddresses(buffer, address);
        for (u32 value : buffer) {
            operation((DAddr(value) << page_bits) + subbits);
        }
    }

    /// Whether p points into the emulated physical memory, the only memory with device addresses.
    bool IsPhysicalPointer(const u8* p) const {
        return (GetRawPhysicalAddr(p) >> page_bits) < compressed_device_addr.size();
    }

    template <typename Func>
    void ApplyOpOnPointer(const u8* p, Common::ScratchBuffer<u32>& buffer, Func&& operation) {
        PAddr address = GetRawPhysicalAddr<u8>(p);
        ApplyOpOnPAddr(address, buffer, operation);
    }

#ifdef HAS_NCE
    // Runs only for a single, matching device mapping contained in one page. The callback holds
    // mapping_guard: it must not map/unmap, gather aliases, or acquire rasterizer cache locks.
    template <typename Operation>
    bool ApplyOpOnSingleMapping(const u8* pointer, size_t size, Operation&& operation) {
        return ApplyOpOnSingleMappingImpl(pointer, size, std::forward<Operation>(operation), false);
    }

    template <typename Operation>
    bool TryApplyOpOnSingleMapping(const u8* pointer, size_t size, Operation&& operation) {
        return ApplyOpOnSingleMappingImpl(pointer, size, std::forward<Operation>(operation), true);
    }

private:
    template <typename Operation>
    bool ApplyOpOnSingleMappingImpl(const u8* pointer, size_t size, Operation&& operation, bool diagnostic) {
        std::unique_lock lock{mapping_guard, std::defer_lock};
        if (diagnostic) {
            if (!lock.try_lock()) {
                Common::GpuTransitionProfile::RecordDiagnosticDrop(
                    Common::GpuTransitionProfile::DiagnosticSite::DeviceMappingInspection);
                return false;
            }
        } else {
            lock.lock();
        }
        if (!pointer || !IsPhysicalPointer(pointer)) {
            return false;
        }
        const PAddr physical = GetRawPhysicalAddr(pointer);
        const size_t offset = physical & page_mask;
        if (size == 0 || size > page_size - offset) {
            return false;
        }
        const u32 base = compressed_device_addr[physical >> page_bits];
        if (base == 0 || (base & MULTI_FLAG) != 0) {
            return false;
        }
        const DAddr address = (DAddr{base} << page_bits) + offset;
        if (address >= device_as_size || GetPointer<u8>(address) != pointer) {
            return false;
        }
        operation(address);
        return true;
    }

public:
#endif

    PAddr GetPhysicalRawAddressFromDAddr(DAddr address) const {
        PAddr subbits = PAddr(address & page_mask);
        auto paddr = tracked_entries[(address >> page_bits)].compressed_physical_ptr;
        if (paddr == 0) {
            return 0;
        }
        return (PAddr(paddr - 1) << page_bits) + subbits;
    }

    // One mapped page, stable while operation(pointer, physical) runs. Cache locks precede
    // mapping_guard; operation must not remap, acquire cache locks or wait for the GPU.
    template <typename Operation>
    bool ApplyOpOnMappedMemory(DAddr address, size_t size, Operation&& operation) {
        std::scoped_lock lock{mapping_guard};
        if (address >= device_as_size || size == 0 || size > page_size - (address & page_mask)) {
            return false;
        }
        auto* pointer = GetPointer<u8>(address);
        if (!pointer || !IsPhysicalPointer(pointer)) {
            return false;
        }
        operation(pointer, GetRawPhysicalAddr(pointer));
        return true;
    }

    // Diagnostic visitor. Cache callers must not wait for mapping mutations or retain pointers.
    template <typename Operation>
    bool TryInspectPhysicalRange(DAddr address, u64 size, Operation&& operation) {
        constexpr u64 max_pages = 4096;
        std::unique_lock lock{mapping_guard, std::try_to_lock};
        if (!lock || address >= device_as_size || size == 0 || size > device_as_size - address ||
            size > max_pages * page_size - (address & page_mask)) {
            return false;
        }
        bool complete = true;
        while (size != 0) {
            const u64 part = (std::min<u64>)(size, page_size - (address & page_mask));
            auto* pointer = GetPointer<u8>(address);
            if (pointer && IsPhysicalPointer(pointer)) {
                operation(GetRawPhysicalAddr(pointer), address, part);
            } else {
                complete = false;
            }
            address += part;
            size -= part;
        }
        return complete;
    }

    template <typename T>
    void Write(DAddr address, T value);

    template <typename T>
    T Read(DAddr address) const;

    u8* GetSpan(const DAddr src_addr, const std::size_t size);
    const u8* GetSpan(const DAddr src_addr, const std::size_t size) const;

    void ReadBlock(DAddr address, void* dest_pointer, size_t size);
    void ReadBlockUnsafe(DAddr address, void* dest_pointer, size_t size);
    void WriteBlock(DAddr address, const void* src_pointer, size_t size);
    void WriteBlockUnsafe(DAddr address, const void* src_pointer, size_t size);

    Asid RegisterProcess(Memory::Memory* memory);
    void UnregisterProcess(Asid id);

    void UpdatePagesCachedCount(DAddr addr, size_t size, s32 delta);

    // New batch API to update multiple ranges with a single lock acquisition.
    void UpdatePagesCachedBatch(std::span<const std::pair<DAddr, size_t>> ranges, s32 delta);

    /// Traps the next guest CPU read of these pages, to learn which GPU writes the CPU reads.
    void ArmReadTraps(DAddr addr, size_t size);

private:
    void MapImpl(DAddr address, VAddr virtual_address, size_t size, Asid asid, bool track);
    void MapPhysicalImpl(DAddr address, std::span<const DeviceMemoryPhysicalSpan> spans, size_t size);
    struct TranslationEntry {
        DAddr guest_page{};
        u8* host_ptr{};
    };

    /// A thread's last four page translations of one manager. Several threads read through
    /// them (the GPU thread, the fencing thread, service threads): one cache shared by all was
    /// cleared by Map on a service thread between the GPU thread's null check of an entry and
    /// its use, and the GPU thread copied from address 0 (title 1 crashed every few minutes).
    struct TranslationCache {
        const void* owner{};
        u64 generation{};
        std::array<TranslationEntry, 4> slots{};
        u32 cursor{};
    };
    static inline thread_local TranslationCache t_cache{};

    /// This thread's translations, emptied when Map or Unmap ran since they were made.
    TranslationCache& Translations() {
        const u64 current = translation_generation.load(std::memory_order_acquire);
        if (t_cache.owner != this || t_cache.generation != current) {
            t_cache = TranslationCache{.owner = this, .generation = current};
        }
        return t_cache;
    }

    // Internal helper that performs the update assuming the caller already holds the necessary lock.
    void UpdatePagesCachedCountNoLock(DAddr addr, size_t size, s32 delta);

private:
    static constexpr size_t device_virtual_bits = Traits::device_virtual_bits;
    static constexpr size_t device_as_size = 1ULL << device_virtual_bits;
    static constexpr size_t physical_min_bits = 32;
    static constexpr size_t physical_max_bits = 35; // covers the 12 GiB layout plus nx2 graphics server memory
    static constexpr size_t page_bits = 12;
    static constexpr size_t page_size = 1ULL << page_bits;
    static constexpr size_t page_mask = page_size - 1ULL;
    static constexpr u32 physical_address_base = 1U << page_bits;
    static constexpr u32 MULTI_FLAG_BITS = 31;
    static constexpr u32 MULTI_FLAG = 1U << MULTI_FLAG_BITS;
    static constexpr u32 MULTI_MASK = ~MULTI_FLAG;

public:
    // Raw physical offset <-> host pointer (used by the deferred-writes consumer).
    template <typename T>
    T* GetPointerFromRaw(PAddr addr) {
        return reinterpret_cast<T*>(physical_base + addr);
    }

    template <typename T>
    const T* GetPointerFromRaw(PAddr addr) const {
        return reinterpret_cast<T*>(physical_base + addr);
    }

    template <typename T>
    PAddr GetRawPhysicalAddr(const T* ptr) const {
        return static_cast<PAddr>(reinterpret_cast<uintptr_t>(ptr) - physical_base);
    }

private:

    void WalkBlock(const DAddr addr, const std::size_t size, auto on_unmapped, auto on_memory,
                   auto increment);

    void CutContinuityAt(size_t page);

    void InnerGatherDeviceAddresses(Common::ScratchBuffer<u32>& buffer, PAddr address);

    std::unique_ptr<DeviceMemoryManagerAllocator<Traits>> impl;

    const uintptr_t physical_base;
    const size_t physical_size;
    const DeviceMemory* device_memory_ptr;
    DeviceInterface* device_inter;
    std::unique_ptr<Common::ThreadWorker> cold_hint_worker;

    struct TrackedEntry {
        VAddr cpu_backing_address;
        u32 continuity_tracker;
        u32 compressed_physical_ptr;
    };
    Common::SparseLargeVector<u32> compressed_device_addr;
    Common::SparseLargeVector<TrackedEntry> tracked_entries;

    // Process memory interfaces

    std::deque<size_t> id_pool;
    std::deque<Memory::Memory*> registered_processes;

    // Memory protection management

    static constexpr size_t guest_max_as_bits = 39;
    static constexpr size_t guest_as_size = 1ULL << guest_max_as_bits;
    static constexpr size_t guest_mask = guest_as_size - 1ULL;
    static constexpr size_t asid_start_bit = guest_max_as_bits;

    std::pair<Asid, VAddr> ExtractCPUBacking(size_t page_index) {
        auto content = tracked_entries[page_index].cpu_backing_address;
        const VAddr address = content & guest_mask;
        const Asid asid{static_cast<size_t>(content >> asid_start_bit)};
        return std::make_pair(asid, address);
    }

    constexpr void InsertCPUBacking(size_t page_index, VAddr address, Asid asid) {
        tracked_entries.GetUnchecked(page_index).cpu_backing_address = address | (asid.id << asid_start_bit);
    }

    std::atomic<u64> translation_generation{1}; ///< Bumped by Map and Unmap
    using CounterType = u8;
    using CounterAtomicType = std::atomic_uint8_t;
    static constexpr size_t subentries = 8 / sizeof(CounterType);
    static constexpr size_t subentries_mask = subentries - 1;
    static constexpr size_t subentries_shift =
        std::countr_zero(sizeof(u64)) - std::countr_zero(sizeof(CounterType));
    class CounterEntry final {
    public:
        CounterEntry() = default;

        CounterAtomicType& Count(std::size_t page) {
            return values[page & subentries_mask];
        }

        const CounterAtomicType& Count(std::size_t page) const {
            return values[page & subentries_mask];
        }

    private:
        std::array<CounterAtomicType, subentries> values{};
    };
    static_assert(sizeof(CounterEntry) == subentries * sizeof(CounterType),
                  "CounterEntry should be 8 bytes!");

    static constexpr size_t num_counter_entries =
        (1ULL << (device_virtual_bits - page_bits)) / subentries;
    using CachedPages = std::array<CounterEntry, num_counter_entries>;
    std::unique_ptr<CachedPages> cached_pages;
    Common::RangeMutex counter_guard;
    std::mutex mapping_guard;


};

} // namespace Core
