// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include "common/common_types.h"

namespace VideoCommon::CpuEpochAdmission {
struct Result;
}
namespace VideoCommon::CpuEpochs {
class Coordinator;
}

namespace VideoCommon::NcePageUsageProfile {
enum class Use : u32 {
    Uniform = 1,
    Vertex = 2,
    Index = 4,
    StorageRead = 8,
    TexelRead = 16,
    Indirect = 32,
    GpuWrite = 64,
    OtherBuffer = 128,
    Texture = 256,
};
constexpr u32 Bit(Use use) {
    return static_cast<u32>(use);
}
struct BindingInfo {
    u64 device_address{}, size{}, buffer_id{};
    u32 kind{}, stage{}, index{};
    bool written{};
};
struct PageSample {
    PAddr physical{};
    u64 faults{}, first_pc{}, recent_pc{}, first_tid{}, recent_tid{}, device_base{};
    u32 ever_mask{}, window_mask{}, core_mask{};
    bool aliases{}, device_known{};
    u64 read_use_version{}, read_use_events{}, fault_bursts_approx{}, epoch_races{};
    std::array<u64, 64> cpu_mask{}, gpu_declared_mask{};
    u32 cpu_bytes{}, gpu_declared_bytes{}, overlap_bytes{};
    BindingInfo last_binding{}, last_writable_binding{};
};
struct Totals {
    u64 faults{}, unresolved{}, dropped{}, range_overflow{}, invalid_usage{}, alias_unknown{},
        coverage_gaps{}, coverage_gap_bytes{};
    u64 cpu_writes{}, cpu_write_bytes{}, unresolved_cpu_writes{}, dropped_cpu_writes{};
};
// Bounded storage is prepared only at session configuration, never by fault or usage hooks.
class Tracker {
public:
    static constexpr size_t PageBits = 12, HashCapacity = 32768, Capacity = 16384, Probes = 16;
    void Reset(bool active = true);
    bool Initialized() const noexcept;
    void Fault(PAddr physical, u64 pc, u64 tid, u32 core);
    void UnresolvedFault();
    void CpuWrite(PAddr physical, u64 size);
    void UnresolvedCpuWrite();
    void Binding(PAddr physical, u64 size, const BindingInfo& info);
    void UsePhysical(PAddr physical, u64 size, DAddr device, u32 mask);
    std::optional<PageSample> Get(PAddr physical) const;
    std::optional<PageSample> Take(size_t dense_index);
    size_t Watched() const;
    Totals TakeTotals();
    void CoverageGap(u64 bytes);

private:
    struct AtomicBinding {
        std::atomic<u64> device{}, size{}, buffer_id{};
        std::atomic<u32> kind{}, stage{}, index{};
        std::atomic<bool> written{};
        void Store(const BindingInfo& info) noexcept;
        BindingInfo Load() const noexcept;
    };
    struct Entry {
        std::atomic<u64> key{}, faults{}, first_pc{}, recent_pc{}, first_tid{}, recent_tid{},
            device{};
        std::atomic<u64> read_use_version{}, read_use_events{}, fault_bursts{}, epoch_races{};
        std::atomic<u64> last_fault_version{~u64{0}};
        std::atomic<u32> ever_mask{}, window_mask{}, core_mask{};
        std::atomic<bool> initialized{}, aliases{};
        std::atomic<u32> footprint_slot{};
        AtomicBinding binding{}, writable_binding{};
    };
    struct Footprint {
        std::array<std::atomic<u64>, 64> cpu{}, gpu_declared{};
    };
    static void MarkBits(std::array<std::atomic<u64>, 64>& bits, u32 begin, u32 end);
    static void CountBits(PageSample& sample);
    bool MarkFootprint(Entry& entry, u32 begin, u32 end, bool gpu);
    template <typename Operation>
    u64 VisitRange(PAddr physical, u64 size, Operation&& operation);
    Entry* Find(u64 page);
    const Entry* Find(u64 page) const;
    PageSample Read(const Entry& entry, bool footprint = true) const;
    void Mark(Entry& entry, u64 page, u64 first_page, u64 device_base, bool device_valid, u32 mask,
              u32 byte_begin, u32 byte_end);
    std::unique_ptr<Entry[]> entries;
    std::unique_ptr<Footprint[]> footprints;
    std::unique_ptr<std::atomic<u32>[]> dense;
    std::atomic<u64> claimed{}, fault_count{}, unresolved{}, dropped{}, range_overflow{},
        invalid_usage{}, alias_unknown{}, coverage_gaps{}, coverage_gap_bytes{};
    std::atomic<u64> cpu_writes{}, cpu_write_bytes{}, unresolved_cpu_writes{}, dropped_cpu_writes{};
};
bool Enabled() noexcept;
bool CpuEpochAuditEnabled() noexcept;
bool IsCpuEpochStorageCandidate(const PageSample& sample);
bool IsCpuEpochStorageCandidate(PAddr physical);
CpuEpochAdmission::Result EvaluateCpuEpochAudit(const PageSample* sample, PAddr physical, u64 size,
                                               bool pending_readbacks, bool gpu_dirty);
void AuditCpuEpoch(PAddr physical, u64 size, bool pending_readbacks, bool gpu_dirty);
bool NeedsMapping() noexcept;
void CoverageGap(u64 bytes);
void Configure();
void Fault(PAddr physical, u64 pc, u64 tid, u32 core);
void UnresolvedFault();
void CpuWrite(PAddr physical, u64 size);
void UnresolvedCpuWrite();
void Binding(PAddr physical, u64 size, const BindingInfo& info);
void UsePhysical(PAddr physical, u64 size, DAddr device, u32 mask);
void Report(const CpuEpochs::Coordinator* epochs = nullptr);
template <typename Memory>
void UseDeviceRange(Memory& memory, DAddr device, u64 size, u32 mask,
                    const BindingInfo* info = nullptr) {
    if (!NeedsMapping() || !size) {
        return;
    }
    if (!memory.TryInspectPhysicalRange(
            device, size, [mask, info](PAddr physical, DAddr mapped_device, u64 length) {
                UsePhysical(physical, length, mapped_device, mask);
                if (info) {
                    Binding(physical, length, *info);
                }
            })) {
        CoverageGap(size);
    }
}
} // namespace VideoCommon::NcePageUsageProfile
