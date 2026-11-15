// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "common/logging.h"
#include "video_core/cpu_epoch_admission.h"
#include "video_core/cpu_epoch_coordinator.h"
#include "video_core/nce_page_usage_profile.h"

namespace VideoCommon::NcePageUsageProfile {
namespace {
constexpr u64 PageSize = 1ULL << Tracker::PageBits;
constexpr u32 KnownMask = 511;
constexpr u32 BufferReads = Bit(Use::Uniform) | Bit(Use::Vertex) | Bit(Use::Index) |
                            Bit(Use::StorageRead) | Bit(Use::TexelRead) | Bit(Use::Indirect) |
                            Bit(Use::OtherBuffer);
std::atomic<bool> enabled{};
std::atomic<bool> cpu_epoch_audit{};
struct EpochAuditCounters {
    std::atomic<u64> evaluations{}, candidates{}, candidate_downloads{}, candidate_readbacks{};
    std::array<std::atomic<u64>, 8> rejections{};
} epoch_audit;
constexpr std::array epoch_reasons{"unclassified",
                                   "spatial_overlap_or_unknown",
                                   "observed_alias",
                                   "cpu_identity_unproven",
                                   "pending_readbacks_or_unknown",
                                   "gpu_data_unresolved",
                                   "writer_barrier_missing",
                                   "download_required"};
Tracker tracker;
std::atomic_flag reporting = ATOMIC_FLAG_INIT;
u64 last_report{};
u64 Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
struct Runs {
    std::string text;
    bool truncated{};
};
Runs EncodeRuns(const std::array<u64, 64>& bits) {
    Runs result;
    size_t count = 0;
    const auto set = [&](u32 byte) { return (bits[byte / 64] & (1ULL << (byte % 64))) != 0; };
    for (u32 byte = 0; byte < PageSize;) {
        if (!set(byte)) {
            ++byte;
            continue;
        }
        if (count == 8) {
            result.truncated = true;
            break;
        }
        const auto begin = byte;
        while (byte < PageSize && set(byte)) {
            ++byte;
        }
        if (count++) {
            result.text += ',';
        }
        result.text += std::to_string(begin) + '-' + std::to_string(byte - 1);
    }
    if (result.text.empty()) {
        result.text = "none";
    }
    return result;
}
size_t Hash(u64 page) {
    return (page * 11400714819323198485ULL) % Tracker::HashCapacity;
}
} // namespace
void Tracker::AtomicBinding::Store(const BindingInfo& info) noexcept {
    device.store(info.device_address, std::memory_order_relaxed);
    size.store(info.size, std::memory_order_relaxed);
    buffer_id.store(info.buffer_id, std::memory_order_relaxed);
    kind.store(info.kind, std::memory_order_relaxed);
    stage.store(info.stage, std::memory_order_relaxed);
    index.store(info.index, std::memory_order_relaxed);
    written.store(info.written, std::memory_order_relaxed);
}
BindingInfo Tracker::AtomicBinding::Load() const noexcept {
    return {device.load(std::memory_order_relaxed),    size.load(std::memory_order_relaxed),
            buffer_id.load(std::memory_order_relaxed), kind.load(std::memory_order_relaxed),
            stage.load(std::memory_order_relaxed),     index.load(std::memory_order_relaxed),
            written.load(std::memory_order_relaxed)};
}
bool Tracker::Initialized() const noexcept {
    return entries && footprints && dense;
}
void Tracker::Reset(bool active) {
    const auto footprint_count = Watched();
    claimed = 0;
    fault_count = 0;
    unresolved = 0;
    dropped = 0;
    range_overflow = 0;
    invalid_usage = 0;
    alias_unknown = 0;
    coverage_gaps = 0;
    coverage_gap_bytes = 0;
    cpu_writes = 0;
    cpu_write_bytes = 0;
    unresolved_cpu_writes = 0;
    dropped_cpu_writes = 0;
    if (!active) {
        entries.reset();
        footprints.reset();
        dense.reset();
        return;
    }
    if (!Initialized()) {
        auto new_entries = std::make_unique<Entry[]>(HashCapacity);
        auto new_footprints = std::make_unique<Footprint[]>(Capacity);
        auto new_dense = std::make_unique<std::atomic<u32>[]>(Capacity);
        entries = std::move(new_entries);
        footprints = std::move(new_footprints);
        dense = std::move(new_dense);
    }
    for (size_t i = 0; i < Capacity; ++i) {
        dense[i] = 0;
    }
    for (size_t i = 0; i < footprint_count; ++i) {
        auto& footprint = footprints[i];
        for (auto& word : footprint.cpu) {
            word = 0;
        }
        for (auto& word : footprint.gpu_declared) {
            word = 0;
        }
    }
    for (size_t i = 0; i < HashCapacity; ++i) {
        auto& e = entries[i];
        e.key = 0;
        e.faults = 0;
        e.first_pc = 0;
        e.recent_pc = 0;
        e.first_tid = 0;
        e.recent_tid = 0;
        e.device = 0;
        e.ever_mask = 0;
        e.window_mask = 0;
        e.core_mask = 0;
        e.initialized = false;
        e.aliases = false;
        e.footprint_slot = 0;
        e.binding.Store({});
        e.writable_binding.Store({});
        e.read_use_version = 0;
        e.read_use_events = 0;
        e.fault_bursts = 0;
        e.epoch_races = 0;
        e.last_fault_version = ~u64{0};
    }
}
Tracker::Entry* Tracker::Find(u64 page) {
    return const_cast<Entry*>(std::as_const(*this).Find(page));
}
const Tracker::Entry* Tracker::Find(u64 page) const {
    if (!Initialized()) {
        return nullptr;
    }
    for (size_t probe = 0; probe < Probes; ++probe) {
        const auto& e = entries[(Hash(page) + probe) % HashCapacity];
        const auto key = e.key.load(std::memory_order_acquire);
        if (key == page + 1) {
            return &e;
        }
        if (!key) {
            return nullptr;
        }
    }
    return nullptr;
}
void Tracker::Fault(PAddr physical, u64 pc, u64 tid, u32 core) {
    fault_count.fetch_add(1, std::memory_order_relaxed);
    if (!Initialized()) {
        dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 page = physical >> PageBits;
    Entry* selected = nullptr;
    for (size_t probe = 0; probe < Probes; ++probe) {
        const auto index = (Hash(page) + probe) % HashCapacity;
        auto& e = entries[index];
        auto key = e.key.load(std::memory_order_acquire);
        if (!key) {
            if (e.key.compare_exchange_strong(key, page + 1, std::memory_order_acq_rel)) {
                const auto position = claimed.fetch_add(1, std::memory_order_relaxed);
                if (position >= Capacity) {
                    break;
                }
                e.footprint_slot.store(static_cast<u32>(position + 1), std::memory_order_relaxed);
                e.first_pc = pc;
                e.first_tid = tid;
                e.initialized.store(true, std::memory_order_release);
                dense[position].store(static_cast<u32>(index + 1), std::memory_order_release);
                selected = &e;
                break;
            }
        }
        if (key == page + 1) {
            if (e.initialized.load(std::memory_order_acquire)) {
                selected = &e;
            }
            break;
        }
    }
    if (!selected) {
        dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto version = selected->read_use_version.load(std::memory_order_relaxed);
    auto previous = selected->last_fault_version.load(std::memory_order_relaxed);
    if (previous == ~u64{0} || version > previous) {
        if (selected->last_fault_version.compare_exchange_strong(previous, version,
                                                                 std::memory_order_relaxed)) {
            selected->fault_bursts.fetch_add(1, std::memory_order_relaxed);
        } else if (previous != version) {
            selected->epoch_races.fetch_add(1, std::memory_order_relaxed);
        }
    }
    selected->recent_pc.store(pc, std::memory_order_relaxed);
    selected->recent_tid.store(tid, std::memory_order_relaxed);
    if (core < 32) {
        selected->core_mask.fetch_or(1U << core, std::memory_order_relaxed);
    }
    selected->faults.fetch_add(1, std::memory_order_relaxed);
}
void Tracker::UnresolvedFault() {
    fault_count.fetch_add(1, std::memory_order_relaxed);
    unresolved.fetch_add(1, std::memory_order_relaxed);
}
template <typename Operation>
u64 Tracker::VisitRange(PAddr physical, u64 size, Operation&& operation) {
    if (!size || size - 1 > std::numeric_limits<u64>::max() - physical) {
        return 0;
    }
    const auto count = Watched();
    const u64 first = physical >> PageBits, last = (physical + size - 1) >> PageBits;
    u64 visited = 0;
    const auto visit = [&](Entry& e, u64 page) {
        if (!e.initialized.load(std::memory_order_acquire)) {
            coverage_gaps.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const u32 begin = page == first ? static_cast<u32>(physical & (PageSize - 1)) : 0;
        const u32 end = page == last
                            ? static_cast<u32>(((physical + size - 1) & (PageSize - 1)) + 1)
                            : static_cast<u32>(PageSize);
        operation(e, page, begin, end);
        ++visited;
    };
    if (last - first + 1 < count) {
        for (u64 page = first; page <= last; ++page) {
            if (auto* e = Find(page)) {
                visit(*e, page);
            }
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            const auto index = dense[i].load(std::memory_order_acquire);
            if (!index) {
                coverage_gaps.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            auto& e = entries[index - 1];
            const auto page = e.key.load(std::memory_order_relaxed) - 1;
            if (page >= first && page <= last) {
                visit(e, page);
            }
        }
    }
    return visited;
}
void Tracker::MarkBits(std::array<std::atomic<u64>, 64>& bits, u32 begin, u32 end) {
    if (begin >= end || end > PageSize) {
        return;
    }
    const u32 first = begin / 64, last = (end - 1) / 64;
    for (u32 i = first; i <= last; ++i) {
        const u32 low = i == first ? begin % 64 : 0;
        const u32 high = i == last ? (end - 1) % 64 + 1 : 64;
        const u64 mask = (~u64{0} << low) & (~u64{0} >> (64 - high));
        bits[i].fetch_or(mask, std::memory_order_relaxed);
    }
}
bool Tracker::MarkFootprint(Entry& e, u32 begin, u32 end, bool gpu) {
    const auto slot = e.footprint_slot.load(std::memory_order_relaxed);
    if (!slot || slot > Capacity) {
        coverage_gaps.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto& footprint = footprints[slot - 1];
    MarkBits(gpu ? footprint.gpu_declared : footprint.cpu, begin, end);
    return true;
}
void Tracker::CpuWrite(PAddr physical, u64 size) {
    if (!size) {
        return;
    }
    cpu_writes.fetch_add(1, std::memory_order_relaxed);
    cpu_write_bytes.fetch_add(size, std::memory_order_relaxed);
    if (size - 1 > std::numeric_limits<u64>::max() - physical) {
        range_overflow.fetch_add(1, std::memory_order_relaxed);
        dropped_cpu_writes.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    bool incomplete = false;
    const auto visited = VisitRange(physical, size, [&](Entry& e, u64, u32 begin, u32 end) {
        incomplete |= !MarkFootprint(e, begin, end, false);
    });
    const auto pages = ((physical + size - 1) >> PageBits) - (physical >> PageBits) + 1;
    if (incomplete || visited != pages) {
        dropped_cpu_writes.fetch_add(1, std::memory_order_relaxed);
    }
}
void Tracker::UnresolvedCpuWrite() {
    unresolved_cpu_writes.fetch_add(1, std::memory_order_relaxed);
}
void Tracker::Binding(PAddr physical, u64 size, const BindingInfo& info) {
    if (size && size - 1 > std::numeric_limits<u64>::max() - physical) {
        range_overflow.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    VisitRange(physical, size, [&](Entry& e, u64, u32, u32) {
        e.binding.Store(info);
        if (info.written) {
            e.writable_binding.Store(info);
        }
    });
}
void Tracker::Mark(Entry& e, u64 page, u64 first_page, u64 base, bool valid, u32 mask,
                   u32 byte_begin, u32 byte_end) {
    if (!e.initialized.load(std::memory_order_acquire)) {
        coverage_gaps.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if ((mask & BufferReads) && !(mask & Bit(Use::GpuWrite))) {
        e.read_use_version.fetch_add(1, std::memory_order_relaxed);
        e.read_use_events.fetch_add(1, std::memory_order_relaxed);
    }
    if (mask & Bit(Use::GpuWrite)) {
        MarkFootprint(e, byte_begin, byte_end, true);
    }
    e.ever_mask.fetch_or(mask, std::memory_order_relaxed);
    e.window_mask.fetch_or(mask, std::memory_order_relaxed);
    const u64 offset = (page - first_page) << PageBits;
    if (!valid || offset > std::numeric_limits<u64>::max() - base || base + offset == ~u64{0}) {
        alias_unknown.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 encoded = base + offset + 1;
    auto previous = e.device.load(std::memory_order_relaxed);
    if (!previous) {
        e.device.compare_exchange_strong(previous, encoded, std::memory_order_relaxed);
    }
    if (previous && previous != encoded) {
        e.aliases.store(true, std::memory_order_relaxed);
    }
}
void Tracker::UsePhysical(PAddr physical, u64 size, DAddr device, u32 mask) {
    if (!size) {
        return;
    }
    if (mask & ~KnownMask) {
        invalid_usage.fetch_add(1, std::memory_order_relaxed);
    }
    mask &= KnownMask;
    if (!mask) {
        return;
    }
    if (size - 1 > std::numeric_limits<u64>::max() - physical) {
        range_overflow.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 first = physical >> PageBits;
    const u64 partial = physical & (PageSize - 1);
    const bool valid = device >= partial && size - 1 <= std::numeric_limits<u64>::max() - device;
    const u64 base = valid ? device - partial : 0;
    VisitRange(physical, size, [&](Entry& e, u64 page, u32 begin, u32 end) {
        Mark(e, page, first, base, valid, mask, begin, end);
    });
}
PageSample Tracker::Read(const Entry& e, bool footprint) const {
    const auto device = e.device.load(std::memory_order_relaxed);
    PageSample result{(e.key.load(std::memory_order_relaxed) - 1) << PageBits,
                      e.faults.load(std::memory_order_relaxed),
                      e.first_pc.load(std::memory_order_relaxed),
                      e.recent_pc.load(std::memory_order_relaxed),
                      e.first_tid.load(std::memory_order_relaxed),
                      e.recent_tid.load(std::memory_order_relaxed),
                      device ? device - 1 : 0,
                      e.ever_mask.load(std::memory_order_relaxed),
                      e.window_mask.load(std::memory_order_relaxed),
                      e.core_mask.load(std::memory_order_relaxed),
                      e.aliases.load(std::memory_order_relaxed),
                      device != 0,
                      e.read_use_version.load(std::memory_order_relaxed),
                      e.read_use_events.load(std::memory_order_relaxed),
                      e.fault_bursts.load(std::memory_order_relaxed),
                      e.epoch_races.load(std::memory_order_relaxed)};
    result.last_binding = e.binding.Load();
    result.last_writable_binding = e.writable_binding.Load();
    const auto slot = e.footprint_slot.load(std::memory_order_relaxed);
    if (footprint && slot && slot <= Capacity) {
        for (size_t i = 0; i < 64; ++i) {
            result.cpu_mask[i] = footprints[slot - 1].cpu[i].load(std::memory_order_relaxed);
            result.gpu_declared_mask[i] =
                footprints[slot - 1].gpu_declared[i].load(std::memory_order_relaxed);
        }
        CountBits(result);
    }
    return result;
}
void Tracker::CountBits(PageSample& sample) {
    sample.cpu_bytes = sample.gpu_declared_bytes = sample.overlap_bytes = 0;
    for (size_t i = 0; i < 64; ++i) {
        sample.cpu_bytes += std::popcount(sample.cpu_mask[i]);
        sample.gpu_declared_bytes += std::popcount(sample.gpu_declared_mask[i]);
        sample.overlap_bytes += std::popcount(sample.cpu_mask[i] & sample.gpu_declared_mask[i]);
    }
}
std::optional<PageSample> Tracker::Get(PAddr physical) const {
    const auto* e = Find(physical >> PageBits);
    return e && e->initialized.load(std::memory_order_acquire) ? std::optional{Read(*e)}
                                                               : std::nullopt;
}
std::optional<PageSample> Tracker::Take(size_t i) {
    if (i >= Watched()) {
        return std::nullopt;
    }
    const auto index = dense[i].load(std::memory_order_acquire);
    if (!index) {
        return std::nullopt;
    }
    auto& e = entries[index - 1];
    auto result = Read(e, false);
    const auto slot = e.footprint_slot.load(std::memory_order_relaxed);
    if (slot && slot <= Capacity) {
        for (size_t word = 0; word < 64; ++word) {
            result.cpu_mask[word] =
                footprints[slot - 1].cpu[word].exchange(0, std::memory_order_relaxed);
            result.gpu_declared_mask[word] =
                footprints[slot - 1].gpu_declared[word].exchange(0, std::memory_order_relaxed);
        }
        CountBits(result);
    }
    result.faults = e.faults.exchange(0, std::memory_order_relaxed);
    result.window_mask = e.window_mask.exchange(0, std::memory_order_relaxed);
    result.core_mask = e.core_mask.exchange(0, std::memory_order_relaxed);
    result.read_use_events = e.read_use_events.exchange(0, std::memory_order_relaxed);
    result.fault_bursts_approx = e.fault_bursts.exchange(0, std::memory_order_relaxed);
    result.epoch_races = e.epoch_races.exchange(0, std::memory_order_relaxed);
    return result;
}
size_t Tracker::Watched() const {
    return std::min<u64>(claimed.load(std::memory_order_relaxed), Capacity);
}
Totals Tracker::TakeTotals() {
    return {fault_count.exchange(0),
            unresolved.exchange(0),
            dropped.exchange(0),
            range_overflow.exchange(0),
            invalid_usage.exchange(0),
            alias_unknown.exchange(0),
            coverage_gaps.exchange(0),
            coverage_gap_bytes.exchange(0),
            cpu_writes.exchange(0),
            cpu_write_bytes.exchange(0),
            unresolved_cpu_writes.exchange(0),
            dropped_cpu_writes.exchange(0)};
}
void Tracker::CoverageGap(u64 bytes) {
    coverage_gaps.fetch_add(1, std::memory_order_relaxed);
    coverage_gap_bytes.fetch_add(bytes, std::memory_order_relaxed);
}
bool NeedsMapping() noexcept {
    return Enabled() && tracker.Watched() != 0;
}
void CoverageGap(u64 bytes) {
    if (Enabled()) {
        tracker.CoverageGap(bytes);
    }
}
bool Enabled() noexcept {
    return enabled.load(std::memory_order_relaxed);
}
bool CpuEpochAuditEnabled() noexcept {
    return cpu_epoch_audit.load(std::memory_order_relaxed);
}
namespace {
bool IsObservedStorageClass(const PageSample& sample) {
    const auto storage_binding = [&](const BindingInfo& binding) {
        return (binding.kind == 3 || binding.kind == 4) && binding.buffer_id && binding.size &&
               binding.size <= std::numeric_limits<u64>::max() - binding.device_address &&
               (binding.device_address <= sample.device_base
                    ? sample.device_base - binding.device_address < binding.size
                    : binding.device_address - sample.device_base < PageSize);
    };
    if (!sample.device_known || !storage_binding(sample.last_binding)) {
        return false;
    }
    if (sample.ever_mask == Bit(Use::StorageRead)) {
        return !sample.last_binding.written && !sample.last_writable_binding.kind &&
               !sample.gpu_declared_bytes;
    }
    return sample.ever_mask == (Bit(Use::StorageRead) | Bit(Use::GpuWrite)) &&
           storage_binding(sample.last_writable_binding) && sample.last_writable_binding.written;
}
} // namespace

bool IsCpuEpochStorageCandidate(const PageSample& sample) {
    return IsObservedStorageClass(sample) && !sample.aliases && sample.cpu_bytes &&
           !sample.overlap_bytes &&
           (sample.ever_mask == Bit(Use::StorageRead) || sample.gpu_declared_bytes);
}
bool IsCpuEpochStorageCandidate(PAddr physical) {
    if (!CpuEpochAuditEnabled())
        return false;
    const auto sample = tracker.Get(physical);
    return sample && IsCpuEpochStorageCandidate(*sample);
}
CpuEpochAdmission::Result EvaluateCpuEpochAudit(const PageSample* sample, PAddr physical, u64 size,
                                                bool pending_readbacks, bool gpu_dirty) {
    const u64 offset = physical & (PageSize - 1);
    const bool bounded = size && size <= PageSize - offset;
    const bool read_only =
        sample && sample->ever_mask == Bit(Use::StorageRead) && IsObservedStorageClass(*sample);
    bool disjoint = sample && sample->cpu_bytes && (read_only || sample->gpu_declared_bytes) &&
                    !sample->overlap_bytes && bounded;
    if (disjoint) {
        for (u64 byte = offset; byte < offset + size; ++byte) {
            if (sample->gpu_declared_mask[byte / 64] & (u64{1} << (byte % 64))) {
                disjoint = false;
                break;
            }
        }
    }
    const CpuEpochAdmission::Inputs input{
        .classified = sample && IsObservedStorageClass(*sample),
        .no_spatial_overlap = disjoint,
        .observed_alias = sample && sample->aliases,
        // This callback holds cache/mapping locks. Acquiring the CPU identity gate here would
        // reverse the producer's lock order; a future outer transition must supply its proof.
        .identity_proven = false,
        .no_pending_readbacks = !pending_readbacks,
        .gpu_dirty = gpu_dirty,
        .writer_barrier_ready = false,
        .gpu_data_resolved = !gpu_dirty,
    };
    return CpuEpochAdmission::Evaluate(input);
}
void AuditCpuEpoch(PAddr physical, u64 size, bool pending_readbacks, bool gpu_dirty) {
    if (!CpuEpochAuditEnabled()) {
        return;
    }
    const auto sample = tracker.Get(physical);
    const auto result = EvaluateCpuEpochAudit(sample ? &*sample : nullptr, physical, size,
                                              pending_readbacks, gpu_dirty);
    epoch_audit.evaluations.fetch_add(1, std::memory_order_relaxed);
    if (result.observed_candidate) {
        epoch_audit.candidates.fetch_add(1, std::memory_order_relaxed);
        epoch_audit.candidate_downloads.fetch_add(result.requires_download,
                                                  std::memory_order_relaxed);
        epoch_audit.candidate_readbacks.fetch_add(pending_readbacks, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < epoch_reasons.size(); ++i) {
        if (result.reasons & (u32{1} << i)) {
            epoch_audit.rejections[i].fetch_add(1, std::memory_order_relaxed);
        }
    }
}
void Configure() {
    enabled = false;
    cpu_epoch_audit = false;
    last_report = Now();
    const auto flag = [](const char* name) {
        const auto* v = std::getenv(name);
        return v && v[0] == '1' && v[1] == '\0';
    };
    const bool requested_audit = flag("EDEN_NX2_CPU_EPOCHS");
    const bool requested =
        flag("EDEN_NX2_GPU_PROFILE") || flag("EDEN_DKB_FRAME_TRACE") || requested_audit;
    tracker.Reset(requested);
    cpu_epoch_audit = requested_audit;
    enabled = requested;
    epoch_audit.evaluations = 0;
    epoch_audit.candidates = 0;
    epoch_audit.candidate_downloads = 0;
    epoch_audit.candidate_readbacks = 0;
    for (auto& count : epoch_audit.rejections) {
        count = 0;
    }
    if (CpuEpochAuditEnabled()) {
        LOG_INFO(HW_GPU, "CPU_EPOCH_CONFIG enabled=1 mode=native_storage_prototype "
                         "selection=storage_properties masks=0x8,0x48 uniform_enabled=false "
                         "gpu_read_policy=materialize_before_rw "
                         "native_writer_barrier=mprotect_membarrier");
    }
    if (Enabled()) {
        LOG_INFO(
            HW_GPU,
            "CPU_PAGE_PROFILE_CONFIG capacity={} masks=uniform:1,vertex:2,index:4,"
            "storage_read:8,texel_read:16,indirect:32,gpu_write:64,other_buffer:128,texture:256 "
            "history=after_first_fault physical_identity=true permission_changes=false "
            "footprint_bytes={} "
            "binding_kind=graphics_uniform:1,compute_uniform:2,graphics_storage:3,compute_storage:"
            "4,multirange:5 multirange_stage=draw:0,dispatch:1",
            Tracker::Capacity, Tracker::Capacity * 2 * 64 * sizeof(u64));
    }
}
void Fault(PAddr page, u64 pc, u64 tid, u32 core) {
    if (Enabled()) {
        tracker.Fault(page, pc, tid, core);
    }
}
void CpuWrite(PAddr physical, u64 size) {
    if (Enabled()) {
        tracker.CpuWrite(physical, size);
    }
}
void UnresolvedCpuWrite() {
    if (Enabled()) {
        tracker.UnresolvedCpuWrite();
    }
}
void Binding(PAddr physical, u64 size, const BindingInfo& info) {
    if (Enabled()) {
        tracker.Binding(physical, size, info);
    }
}
void UnresolvedFault() {
    if (Enabled()) {
        tracker.UnresolvedFault();
    }
}
void UsePhysical(PAddr physical, u64 size, DAddr device, u32 mask) {
    if (Enabled()) {
        tracker.UsePhysical(physical, size, device, mask);
    }
}
void Report(const CpuEpochs::Coordinator* epochs) {
    if (!Enabled() || reporting.test_and_set(std::memory_order_acquire)) {
        return;
    }
    const u64 now = Now();
    if (now - last_report < 10'000'000'000ULL) {
        reporting.clear(std::memory_order_release);
        return;
    }
    const auto elapsed = now - last_report;
    last_report = now;
    if (CpuEpochAuditEnabled()) {
        LOG_INFO(HW_GPU,
                 "CPU_EPOCH_ADMISSION window_ms={} evaluations={} observed_candidate_faults={} "
                 "candidate_download_required={} candidate_pending_readbacks={} "
                 "ownership_counters=CPU_EPOCH_COORDINATOR "
                 "mode=admission_audit coverage=validated_exact_callback "
                 "footprint=current_store_and_independent_window_samples",
                 elapsed / 1'000'000, epoch_audit.evaluations.exchange(0),
                 epoch_audit.candidates.exchange(0), epoch_audit.candidate_downloads.exchange(0),
                 epoch_audit.candidate_readbacks.exchange(0));
        for (size_t i = 0; i < epoch_reasons.size(); ++i) {
            LOG_INFO(HW_GPU, "CPU_EPOCH_REJECT reason={} evaluations={} reasons_can_overlap=true",
                     epoch_reasons[i], epoch_audit.rejections[i].exchange(0));
        }
    }
    std::array<u64, 512> faults{}, pages{}, class_bursts{}, class_reads{};
    std::vector<PageSample> hottest;
    u64 watched_faults = 0, holes = 0, read_events = 0, bursts = 0, epoch_races = 0;
    u64 cpu_bytes = 0, gpu_declared_bytes = 0, overlap_bytes = 0;
    const auto count = tracker.Watched();
    for (size_t i = 0; i < count; ++i) {
        const auto sample = tracker.Take(i);
        if (!sample) {
            ++holes;
            continue;
        }
        cpu_bytes += sample->cpu_bytes;
        gpu_declared_bytes += sample->gpu_declared_bytes;
        overlap_bytes += sample->overlap_bytes;
        read_events += sample->read_use_events;
        bursts += sample->fault_bursts_approx;
        epoch_races += sample->epoch_races;
        class_bursts[sample->ever_mask] += sample->fault_bursts_approx;
        class_reads[sample->ever_mask] += sample->read_use_events;
        if (!sample->faults) {
            continue;
        }
        watched_faults += sample->faults;
        faults[sample->ever_mask] += sample->faults;
        ++pages[sample->ever_mask];
        if (hottest.size() < 32) {
            hottest.push_back(*sample);
        } else {
            auto cold =
                std::min_element(hottest.begin(), hottest.end(),
                                 [](const auto& a, const auto& b) { return a.faults < b.faults; });
            if (sample->faults > cold->faults) {
                *cold = *sample;
            }
        }
    }
    const auto total = tracker.TakeTotals();
    LOG_INFO(HW_GPU,
             "CPU_PAGE_PROFILE window_ms={} faults={} watched_faults={} unresolved={} dropped={} "
             "watched_pages={} dense_holes={} range_overflow={} invalid_usage={} alias_unknown={} "
             "coverage_gaps={} coverage_gap_bytes={} capacity={} probes={} "
             "classification=observed_since_first_fault_not_generation_or_readonly_proof "
             "window_cut=independent_atomics",
             elapsed / 1'000'000, total.faults, watched_faults, total.unresolved, total.dropped,
             count, holes, total.range_overflow, total.invalid_usage, total.alias_unknown,
             total.coverage_gaps, total.coverage_gap_bytes, Tracker::Capacity, Tracker::Probes);
    const bool complete = !total.unresolved_cpu_writes && !total.dropped_cpu_writes &&
                          !total.coverage_gaps && !total.dropped && !total.range_overflow &&
                          !total.invalid_usage && !holes;
    LOG_INFO(
        HW_GPU,
        "CPU_PAGE_PROFILE_FOOTPRINT cpu_writes={} cpu_write_bytes={} unresolved_cpu_writes={} "
        "dropped_cpu_writes={} cpu_unique_bytes={} gpu_declared_unique_bytes={} "
        "spatial_overlap_bytes={} reported_coverage_complete={} scope=watched_since_first_fault "
        "cpu_bytes_scope=successful_decoded_stores gpu_bytes_scope=conservative_declared_ranges "
        "window_cut=independent_word_exchanges actual_gpu_execution_or_hazard_proof=false",
        total.cpu_writes, total.cpu_write_bytes, total.unresolved_cpu_writes,
        total.dropped_cpu_writes, cpu_bytes, gpu_declared_bytes, overlap_bytes, complete);
    LOG_INFO(HW_GPU,
             "CPU_PAGE_PROFILE_EPOCH read_use_events={} fault_bursts_approx={} epoch_races={} "
             "model=first_fault_then_observed_buffer_read_usage lifetime_versions=true "
             "not_permission_or_c1_savings_proof=true",
             read_events, bursts, epoch_races);
    for (size_t mask = 0; mask < faults.size(); ++mask) {
        if (faults[mask]) {
            LOG_INFO(HW_GPU,
                     "CPU_PAGE_PROFILE_CLASS ever_mask={:#x} pages={} faults={} "
                     "fraction_all_faults_approx={} read_use_events={} fault_bursts_approx={} "
                     "observed_buffer_reads_only={} "
                     "mask0=unobserved_not_unused",
                     mask, pages[mask], faults[mask],
                     total.faults ? static_cast<double>(faults[mask]) / total.faults : 0.0,
                     class_reads[mask], class_bursts[mask],
                     mask != 0 && (mask & (Bit(Use::GpuWrite) | Bit(Use::Texture) |
                                           Bit(Use::OtherBuffer))) == 0);
        }
    }
    std::sort(hottest.begin(), hottest.end(),
              [](const auto& a, const auto& b) { return a.faults > b.faults; });
    for (const auto& p : hottest) {
        const auto exclusion = epochs && epochs->Requested() ? epochs->TryExclusions(p.physical) :
                                                               std::nullopt;
        const auto cpu_runs = EncodeRuns(p.cpu_mask);
        const auto gpu_runs = EncodeRuns(p.gpu_declared_mask);
        const bool proxy =
            p.ever_mask != 0 &&
            !(p.ever_mask & (Bit(Use::GpuWrite) | Bit(Use::Texture) | Bit(Use::OtherBuffer))) &&
            p.fault_bursts_approx != 0;
        LOG_INFO(HW_GPU,
                 "CPU_PAGE_PROFILE_HOT physical={:#x} faults={} first_pc={:#x} recent_pc={:#x} "
                 "first_tid={} recent_tid={} core_mask={:#x} ever_mask={:#x} window_mask={:#x} "
                 "aliases={} device_known={} device_base={:#x} read_use_version={} "
                 "read_use_events={} fault_bursts_approx={} epoch_races={} c1_proxy_defined={} "
                 "faults_per_burst_approx={} epoch_exclusion_reported={} epoch_exclusion_mask={:#x} "
                 "metadata=independent_atomic_samples",
                 p.physical, p.faults, p.first_pc, p.recent_pc, p.first_tid, p.recent_tid,
                 p.core_mask, p.ever_mask, p.window_mask, p.aliases, p.device_known, p.device_base,
                 p.read_use_version, p.read_use_events, p.fault_bursts_approx, p.epoch_races, proxy,
                 proxy ? static_cast<double>(p.faults) / p.fault_bursts_approx : 0.0,
                 exclusion.has_value(), static_cast<u32>(exclusion.value_or(0)));
        LOG_INFO(HW_GPU,
                 "CPU_PAGE_PROFILE_BYTES physical={:#x} cpu_bytes={} gpu_declared_bytes={} "
                 "overlap_bytes={} cpu_runs={} cpu_runs_truncated={} gpu_declared_runs={} "
                 "gpu_declared_runs_truncated={} run_offsets=inclusive_bytes spatial_only=true",
                 p.physical, p.cpu_bytes, p.gpu_declared_bytes, p.overlap_bytes, cpu_runs.text,
                 cpu_runs.truncated, gpu_runs.text, gpu_runs.truncated);
        const auto& binding = p.last_binding;
        const auto& writer = p.last_writable_binding;
        LOG_INFO(HW_GPU,
                 "CPU_PAGE_PROFILE_BINDING physical={:#x} last_kind={} last_stage={} last_index={} "
                 "last_device={:#x} last_size={} last_buffer_id={} last_written={} writer_kind={} "
                 "writer_stage={} writer_index={} writer_device={:#x} writer_size={} "
                 "writer_buffer_id={} writer_written={} metadata=independent_atomic_samples",
                 p.physical, binding.kind, binding.stage, binding.index, binding.device_address,
                 binding.size, binding.buffer_id, binding.written, writer.kind, writer.stage,
                 writer.index, writer.device_address, writer.size, writer.buffer_id,
                 writer.written);
    }
    reporting.clear(std::memory_order_release);
}
} // namespace VideoCommon::NcePageUsageProfile
