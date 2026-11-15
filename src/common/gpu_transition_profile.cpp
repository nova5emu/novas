// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/gpu_transition_profile.h"

#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <set>
#include <string_view>
#include <utility>
#include "common/logging.h"
#include "common/caller_stacks.h"

namespace Common::GpuTransitionProfile {
namespace {
constexpr std::array site_names{
    "memory_map_pages", "memory_unmap_pages", "memory_protect_region", "heap_resize",
    "page_table_finalize", "process_finalize", "device_map", "device_unmap", "buffer_join",
    "buffer_delete", "texture_register", "texture_refresh", "texture_refresh_upload", "dma_copy",
    "nv_as_map_queued", "nv_as_map_executed", "nv_as_unmap_queued", "nv_as_unmap_executed",
    "nv_as_release_queued", "nv_as_release_executed", "nv_sparse_create_queued", "nv_sparse_create_executed",
    "nv_sparse_destroy_queued", "nv_sparse_destroy_executed", "nv_sparse_tile_queued", "nv_sparse_tile_executed",
    "nv_sparse_back_mapped", "nv_as_map_reuse", "heap_map_call", "heap_unmap_call",
    "heap_map_mutation", "heap_unmap_mutation", "heap_destroy_unmap", "sparse_write_call",
    "sparse_block_claim", "sparse_backing_allocate_call", "sparse_backing_allocated",
    "sparse_backing_map_failure", "sparse_backing_discarded", "sparse_backing_released",
    "sparse_wait_pending", "sparse_map_request", "kernel_sync_physical_exclude",
    "jit_exclusive_exclude", "nce_fallback_exclude", "locked_access_remap_exclude",
    "legacy_writer_physical_exclude",
};
static_assert(site_names.size() == SiteCount);
constexpr std::array suspension_site_names{
    "svc_memory", "svc_user_buffer", "ipc_receive", "ipc_cleanup", "ipc_async",
    "thread_finalize", "process_finalize", "ro", "loader", "nvdrv", "sparse", "producer_recovery",
    "legacy_writer",
};
static_assert(suspension_site_names.size() == SuspensionSiteCount);
constexpr std::array deferred_status_names{
    "published", "full", "invalid_write", "rejected", "sequence_exhausted", "lease_unavailable",
    "suspended",
};
static_assert(deferred_status_names.size() == DeferredPublishStatusCount);
constexpr std::array deferred_drain_names{"submit_list", "acquire", "readback", "transition"};
static_assert(deferred_drain_names.size() == DeferredDrainSiteCount);
std::chrono::steady_clock::time_point last_report;
std::mutex budget_pages_mutex;
std::array<std::set<PAddr>, deferred_fallback_names.size()> budget_pages;
std::mutex exclusive_inline_mutex;
std::vector<std::shared_ptr<ExclusiveInlineCounter>> exclusive_inline_counters;
struct LeaseDiagnostic {
    const char* stage{};
    const char* cause{};
    PAddr physical_page{};
    u64 calls{};
};
struct RejectedOpcode {
    u64 pc{};
    u32 instruction{};
    u64 calls{};
};
std::mutex producer_diagnostics_mutex;
std::array<LeaseDiagnostic, 64> lease_diagnostics;
std::array<RejectedOpcode, 64> rejected_opcodes;
u64 lease_diagnostic_overflow{}, rejected_opcode_overflow{};

void ReportProducerDiagnostics(double seconds) {
    std::array<LeaseDiagnostic, 64> leases;
    std::array<RejectedOpcode, 64> opcodes;
    u64 lease_overflow{}, opcode_overflow{};
    {
        auto lock = TryDiagnosticLock(producer_diagnostics_mutex, DiagnosticSite::ProducerSnapshot);
        if (!lock) {
            return;
        }
        leases = lease_diagnostics;
        opcodes = rejected_opcodes;
        lease_overflow = std::exchange(lease_diagnostic_overflow, 0);
        opcode_overflow = std::exchange(rejected_opcode_overflow, 0);
        lease_diagnostics = {};
        rejected_opcodes = {};
    }
    for (const auto& entry : leases) {
        if (!entry.calls) break;
        LOG_INFO(HW_GPU, "GPU_LEASE_DIAGNOSTIC window={:.3f}s stage={} cause={} physical_page={:#x} calls={}",
                 seconds, entry.stage, entry.cause, entry.physical_page, entry.calls);
    }
    for (const auto& entry : opcodes) {
        if (!entry.calls) break;
        LOG_INFO(HW_GPU, "GPU_NCE_REJECTED_OPCODE window={:.3f}s reason=unsupported_instruction pc={:#x} opcode={:08x} calls={}",
                 seconds, entry.pc, entry.instruction, entry.calls);
    }
    LOG_INFO(HW_GPU, "GPU_PRODUCER_DIAGNOSTICS window={:.3f}s lease_overflow={} rejected_opcode_overflow={} slots=64",
             seconds, lease_overflow, opcode_overflow);
}

u64 TakeExclusiveInlineHits() {
    auto lock = TryDiagnosticLock(exclusive_inline_mutex, DiagnosticSite::ExclusiveSnapshot);
    if (!lock) {
        return 0;
    }
    u64 result{};
    for (auto it = exclusive_inline_counters.begin(); it != exclusive_inline_counters.end();) {
        const u64 hits = (*it)->hits.load(std::memory_order_relaxed);
        result += hits - (*it)->reported;
        (*it)->reported = hits;
        if (it->use_count() == 1) it = exclusive_inline_counters.erase(it);
        else ++it;
    }
    return result;
}

void UpdateMaximum(std::atomic<u64>& target, u64 value) noexcept {
    u64 previous = target.load(std::memory_order_relaxed);
    while (previous < value &&
           !target.compare_exchange_weak(previous, value, std::memory_order_relaxed)) {
    }
}

u64 ElapsedNs(u64 begin) noexcept {
    const u64 now = SuspensionNowNs();
    return begin != 0 && now >= begin ? now - begin : 0;
}

bool Flag(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
}
} // namespace

namespace {
std::array<std::atomic<u64>, DiagnosticSiteCount> diagnostic_drops;
constexpr std::array diagnostic_site_names{
    "journal_inspection", "journal_snapshot", "device_mapping_inspection", "legacy_pages", "legacy_fault",
    "legacy_snapshot", "writer_nce_fault", "writer_nce_coverage", "writer_nce_snapshot", "lease_detail",
    "rejected_opcode", "producer_snapshot", "caller_stacks", "exclusive_snapshot", "budget_pages", "budget_snapshot",
};
static_assert(diagnostic_site_names.size() == DiagnosticSiteCount);
}
void RecordDiagnosticDrop(DiagnosticSite site) noexcept {
    diagnostic_drops[static_cast<size_t>(site)].fetch_add(1, std::memory_order_relaxed);
}
std::array<u64, DiagnosticSiteCount> TakeDiagnosticDrops() noexcept {
    std::array<u64, DiagnosticSiteCount> result{};
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = diagnostic_drops[i].exchange(0, std::memory_order_relaxed);
    }
    return result;
}
std::unique_lock<std::mutex> TryDiagnosticLock(std::mutex& mutex, DiagnosticSite site) {
    std::unique_lock lock{mutex, std::try_to_lock};
    if (!lock) {
        RecordDiagnosticDrop(site);
    }
    return lock;
}
void RecordLeaseDiagnostic(const char* stage, const char* cause, PAddr physical_page) {
    if (!enabled.load(std::memory_order_relaxed)) return;
    auto lock = TryDiagnosticLock(producer_diagnostics_mutex, DiagnosticSite::LeaseDetail);
    if (!lock) {
        return;
    }
    for (auto& entry : lease_diagnostics) {
        if (!entry.calls) {
            entry = {stage, cause, physical_page, 1};
            return;
        }
        if (std::string_view{entry.stage} == stage && std::string_view{entry.cause} == cause &&
            entry.physical_page == physical_page) {
            ++entry.calls;
            return;
        }
    }
    ++lease_diagnostic_overflow;
}

void RecordRejectedNceOpcode(u32 instruction, u64 pc) {
    if (!enabled.load(std::memory_order_relaxed)) return;
    auto lock = TryDiagnosticLock(producer_diagnostics_mutex, DiagnosticSite::RejectedOpcode);
    if (!lock) {
        return;
    }
    for (auto& entry : rejected_opcodes) {
        if (!entry.calls) {
            entry = {pc, instruction, 1};
            return;
        }
        if (entry.pc == pc && entry.instruction == instruction) {
            ++entry.calls;
            return;
        }
    }
    ++rejected_opcode_overflow;
}

void RecordBudgetExcludedPage(DeferredFallbackReason reason, PAddr page) {
    if (!enabled.load(std::memory_order_relaxed)) return;
    if (reason != DeferredFallbackReason::PublicationCost &&
        reason != DeferredFallbackReason::PublicationCoreCost &&
        reason != DeferredFallbackReason::FaultDensity) return;
    auto lock = TryDiagnosticLock(budget_pages_mutex, DiagnosticSite::BudgetPages);
    if (!lock) {
        return;
    }
    budget_pages[static_cast<size_t>(reason)].insert(page & ~PAddr{4095});
}

void RecordBudgetCoreEvent(bool activated, u32 slow, u64 max_ns, u32 core_index) {
    LOG_INFO(HW_GPU, "GPU_DEFERRED_BUDGET core={} core_event={} slow_samples={} block_max_ms={:.3f}",
             core_index, activated ? "activated" : "rehabilitated", slow, max_ns / 1e6);
}
Counters counters;
Counters byte_counters;
SuspensionCounters suspension_counters;
SvcCounters svc_counters;
DeferredCounters deferred_counters;
ExclusiveCounters exclusive_counters;

std::shared_ptr<ExclusiveInlineCounter> RegisterExclusiveInlineCounter() {
    if (!Flag("EDEN_NX2_DEFERRED_WRITES") && !Flag("EDEN_NX2_GPU_PROFILE")) return {};
    auto counter = std::make_shared<ExclusiveInlineCounter>();
    std::scoped_lock lock{exclusive_inline_mutex};
    exclusive_inline_counters.push_back(counter);
    return counter;
}

void ExclusiveCounters::Record(bool fast, u64 elapsed_ns) noexcept {
    calls.fetch_add(1, std::memory_order_relaxed);
    if (fast) fast_hits.fetch_add(1, std::memory_order_relaxed);
    hook_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
}

ExclusiveSnapshot ExclusiveCounters::TakeSnapshot() noexcept {
    return {calls.exchange(0, std::memory_order_relaxed),
            fast_hits.exchange(0, std::memory_order_relaxed),
            hook_ns.exchange(0, std::memory_order_relaxed)};
}

void RecordExclusiveHook(bool fast, u64 begin_ns) noexcept {
    if (enabled.load(std::memory_order_relaxed))
        exclusive_counters.Record(fast, ElapsedNs(begin_ns));
}

void DeferredCounters::RecordPublish(DeferredPublishStatus status, u64 bytes,
                                     u64 ring_occupancy) noexcept {
    publications[static_cast<size_t>(status)].fetch_add(1, std::memory_order_relaxed);
    if (status == DeferredPublishStatus::Published) {
        published_bytes.fetch_add(bytes, std::memory_order_relaxed);
        UpdateMaximum(ring_peak, ring_occupancy);
    } else if (status == DeferredPublishStatus::Full) {
        RecordBackpressure();
    }
}

void DeferredCounters::RecordDrain(DeferredDrainSite site) noexcept {
    drains[static_cast<size_t>(site)].fetch_add(1, std::memory_order_relaxed);
}

void DeferredCounters::RecordApplied(u64 count) noexcept {
    applied.fetch_add(count, std::memory_order_relaxed);
}

void DeferredCounters::RecordRetired(u64 count) noexcept {
    retired.fetch_add(count, std::memory_order_relaxed);
}

void DeferredCounters::RecordBackpressure(u64 count) noexcept {
    backpressure.fetch_add(count, std::memory_order_relaxed);
}

void DeferredCounters::RecordRecovery(bool ready) noexcept {
    recovery_attempts.fetch_add(1, std::memory_order_relaxed);
    if (ready) recovered.fetch_add(1, std::memory_order_relaxed);
}

void DeferredCounters::RecordFallback(DeferredFallbackReason reason) noexcept {
    if (reason != DeferredFallbackReason::Success)
        fallbacks[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void DeferredCounters::RecordPhase(DeferredProducerPhase phase, u64 ns) noexcept {
    auto& sample = phases[static_cast<size_t>(phase)];
    sample.calls.fetch_add(1, std::memory_order_relaxed);
    sample.total_ns.fetch_add(ns, std::memory_order_relaxed);
    UpdateMaximum(sample.max_ns, ns);
}

void DeferredCounters::RecordSuspendedExact(bool success) noexcept {
    (success ? suspended_exact_success : suspended_exact_failed).fetch_add(1, std::memory_order_relaxed);
}

DeferredSnapshot DeferredCounters::TakeSnapshot() noexcept {
    DeferredSnapshot result{};
    for (size_t i = 0; i < phases.size(); ++i) {
        result.phases[i] = {phases[i].calls.exchange(0, std::memory_order_relaxed),
                            phases[i].total_ns.exchange(0, std::memory_order_relaxed),
                            phases[i].max_ns.exchange(0, std::memory_order_relaxed)};
    }
    result.suspended_exact_success = suspended_exact_success.exchange(0, std::memory_order_relaxed);
    result.suspended_exact_failed = suspended_exact_failed.exchange(0, std::memory_order_relaxed);
    for (size_t i = 0; i < fallbacks.size(); ++i)
        result.fallbacks[i] = fallbacks[i].exchange(0, std::memory_order_relaxed);
    for (size_t i = 0; i < publications.size(); ++i) {
        result.publications[i] = publications[i].exchange(0, std::memory_order_relaxed);
    }
    result.published_bytes = published_bytes.exchange(0, std::memory_order_relaxed);
    result.ring_peak = ring_peak.exchange(0, std::memory_order_relaxed);
    result.backpressure = backpressure.exchange(0, std::memory_order_relaxed);
    for (size_t i = 0; i < drains.size(); ++i) {
        result.drains[i] = drains[i].exchange(0, std::memory_order_relaxed);
    }
    result.applied = applied.exchange(0, std::memory_order_relaxed);
    result.retired = retired.exchange(0, std::memory_order_relaxed);
    result.recovery_attempts = recovery_attempts.exchange(0, std::memory_order_relaxed);
    result.recovered = recovered.exchange(0, std::memory_order_relaxed);
    return result;
}

void SvcCounters::Record(u32 svc, bool candidate, bool participated, bool new_epoch) noexcept {
    auto& sample = samples[svc < SvcSlotCount - 1 ? svc : SvcSlotCount - 1];
    sample.calls.fetch_add(1, std::memory_order_relaxed);
    if (candidate) sample.candidates.fetch_add(1, std::memory_order_relaxed);
    if (participated) sample.participations.fetch_add(1, std::memory_order_relaxed);
    if (new_epoch) sample.new_epochs.fetch_add(1, std::memory_order_relaxed);
}

SvcSnapshot SvcCounters::TakeSnapshot() noexcept {
    SvcSnapshot result{};
    for (size_t i = 0; i < samples.size(); ++i) {
        auto& sample = samples[i];
        result[i] = {
            sample.calls.exchange(0, std::memory_order_relaxed),
            sample.candidates.exchange(0, std::memory_order_relaxed),
            sample.participations.exchange(0, std::memory_order_relaxed),
            sample.new_epochs.exchange(0, std::memory_order_relaxed),
        };
    }
    return result;
}

void SuspensionCounters::Record(SuspensionSite site, SuspensionEvent event) noexcept {
    samples[static_cast<size_t>(site)].events[static_cast<size_t>(event)].fetch_add(
        1, std::memory_order_relaxed);
}

void SuspensionCounters::RecordReady(SuspensionSite site, bool prepared, u64 wait_ns) noexcept {
    Record(site, prepared ? SuspensionEvent::ReadyPrepared : SuspensionEvent::ReadyJoined);
    auto& sample = samples[static_cast<size_t>(site)];
    sample.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    UpdateMaximum(sample.wait_max_ns, wait_ns);
}

void SuspensionCounters::RecordEpochReady(SuspensionSite origin, u64 draining_ns) noexcept {
    auto& sample = samples[static_cast<size_t>(origin)];
    sample.epochs_ready.fetch_add(1, std::memory_order_relaxed);
    sample.draining_ready_ns.fetch_add(draining_ns, std::memory_order_relaxed);
    UpdateMaximum(sample.draining_ready_max_ns, draining_ns);
}

SuspensionSnapshot SuspensionCounters::TakeSnapshot() noexcept {
    SuspensionSnapshot result{};
    for (size_t i = 0; i < samples.size(); ++i) {
        auto& source = samples[i];
        auto& destination = result[i];
        const auto event = [&](SuspensionEvent kind) {
            return source.events[static_cast<size_t>(kind)].exchange(0, std::memory_order_relaxed);
        };
        destination.participations = event(SuspensionEvent::Participation);
        destination.new_epochs = event(SuspensionEvent::NewEpoch);
        destination.ready_prepared = event(SuspensionEvent::ReadyPrepared);
        destination.ready_joined = event(SuspensionEvent::ReadyJoined);
        destination.pending = event(SuspensionEvent::Pending);
        destination.failures = event(SuspensionEvent::Failure);
        destination.wait_ns = source.wait_ns.exchange(0, std::memory_order_relaxed);
        destination.wait_max_ns = source.wait_max_ns.exchange(0, std::memory_order_relaxed);
        destination.epochs_ready = source.epochs_ready.exchange(0, std::memory_order_relaxed);
        destination.draining_ready_ns = source.draining_ready_ns.exchange(0, std::memory_order_relaxed);
        destination.draining_ready_max_ns =
            source.draining_ready_max_ns.exchange(0, std::memory_order_relaxed);
    }
    return result;
}

u64 SuspensionNowNs() noexcept {
    if (!enabled.load(std::memory_order_relaxed)) return 0;
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void RecordSuspensionReady(SuspensionSite site, bool prepared, u64 caller_begin_ns) noexcept {
    if (!enabled.load(std::memory_order_relaxed)) return;
    suspension_counters.RecordReady(site, prepared, ElapsedNs(caller_begin_ns));
}

void RecordSuspensionEpochReady(SuspensionSite origin, u64 epoch_begin_ns) noexcept {
    if (!enabled.load(std::memory_order_relaxed)) return;
    suspension_counters.RecordEpochReady(origin, ElapsedNs(epoch_begin_ns));
}

Snapshot Counters::TakeSnapshot() noexcept {
    Snapshot result{};
    for (size_t i = 0; i < counts.size(); ++i) {
        result[i] = counts[i].exchange(0, std::memory_order_relaxed);
    }
    return result;
}

namespace {
thread_local bool legacy_scope_active{};
thread_local LegacyWriterSite legacy_scope_site{LegacyWriterSite::Unknown};
thread_local std::array<u64, static_cast<size_t>(LegacyWriterSite::Count)> legacy_stack_calls{};
}

LegacyWriterCounters legacy_writer_counters;

void LegacyWriterCounters::RecordCall(LegacyWriterSite site, u64 bytes) {
    auto& sample = calls[static_cast<size_t>(site)];
    sample.count.fetch_add(1, std::memory_order_relaxed);
    sample.bytes.fetch_add(bytes, std::memory_order_relaxed);
    constexpr std::array<u64, 5> limits{16, 64, 256, 4096, 65536};
    const auto bucket = static_cast<size_t>(std::lower_bound(limits.begin(), limits.end(), bytes) - limits.begin());
    sample.sizes[bucket].fetch_add(1, std::memory_order_relaxed);
}

void LegacyWriterCounters::RecordPhysical(LegacyWriterSite site, PAddr page, bool newly_excluded) {
    page &= ~PAddr{4095};
    auto lock = TryDiagnosticLock(mutex, DiagnosticSite::LegacyPages);
    if (!lock) {
        return;
    }
    first_sites.try_emplace(page, site);
    auto& entry = pages[static_cast<size_t>(site)];
    // A page first excluded in this window stays new across later preparations.
    if (newly_excluded) {
        entry.added.insert(page);
        entry.existing.erase(page);
    } else if (!entry.added.contains(page)) {
        entry.existing.insert(page);
    }
}

void LegacyWriterCounters::RecordFault(PAddr page) {
    page &= ~PAddr{4095};
    auto lock = TryDiagnosticLock(mutex, DiagnosticSite::LegacyFault);
    if (!lock) {
        return;
    }
    const auto found = first_sites.find(page);
    if (found == first_sites.end()) return;
    const auto index = static_cast<size_t>(found->second);
    ++samples[index].nce_faults;
    pages[index].faulted.insert(page);
}

LegacyWriterSnapshot LegacyWriterCounters::TakeSnapshot() {
    auto lock = TryDiagnosticLock(mutex, DiagnosticSite::LegacySnapshot);
    if (!lock) {
        return {};
    }
    auto result = samples;
    for (size_t i = 0; i < pages.size(); ++i) {
        result[i].calls = calls[i].count.exchange(0, std::memory_order_relaxed);
        result[i].bytes = calls[i].bytes.exchange(0, std::memory_order_relaxed);
        for (size_t bucket = 0; bucket < result[i].sizes.size(); ++bucket)
            result[i].sizes[bucket] = calls[i].sizes[bucket].exchange(0, std::memory_order_relaxed);
        result[i].new_pages = pages[i].added.size();
        result[i].existing_pages = pages[i].existing.size();
        result[i].nce_fault_pages = pages[i].faulted.size();
        pages[i] = {};
    }
    samples = {};
    return result;
}

void LegacyWriterCounters::Reset() {
    std::scoped_lock lock{mutex};
    samples = {};
    pages = {};
    first_sites.clear();
    for (auto& entry : calls) {
        entry.count.store(0, std::memory_order_relaxed);
        entry.bytes.store(0, std::memory_order_relaxed);
        for (auto& bucket : entry.sizes) bucket.store(0, std::memory_order_relaxed);
    }
}

LegacyWriterScope::LegacyWriterScope(LegacyWriterSite site, u64 bytes) {
    if (!enabled.load(std::memory_order_relaxed) || legacy_scope_active) return;
    outer = true;
    legacy_scope_active = true;
    legacy_scope_site = site;
    legacy_writer_counters.RecordCall(site, bytes);
    const auto index = static_cast<size_t>(site);
    if (legacy_stack_calls[index]++ % 128 == 0)
        Common::CallerStacks::Sample(legacy_writer_site_names[index], 1);
}

LegacyWriterScope::~LegacyWriterScope() {
    if (outer) {
        legacy_scope_active = false;
        legacy_scope_site = LegacyWriterSite::Unknown;
    }
}

LegacyWriterSite CurrentLegacyWriterSite() noexcept { return legacy_scope_site; }

void RecordLegacyPhysicalPage(PAddr page, bool newly_excluded) {
    if (enabled.load(std::memory_order_relaxed))
        legacy_writer_counters.RecordPhysical(CurrentLegacyWriterSite(), page, newly_excluded);
}

void RecordLegacyNceFault(PAddr page) {
    if (enabled.load(std::memory_order_relaxed)) legacy_writer_counters.RecordFault(page);
}
WriterNceCounters writer_nce_counters;

void WriterNceCounters::RecordCoverage(bool physical_resolved, bool device_resolved) {
    auto lock = TryDiagnosticLock(mutex, DiagnosticSite::WriterNceCoverage);
    if (!lock) {
        return;
    }
    ++sample.total;
    sample.physical_resolved += physical_resolved;
    sample.device_resolved += device_resolved;
}

void WriterNceCounters::RecordFault(PAddr page, u64 origin_mask, u64 first_reason_mask, bool complete) {
    page &= ~PAddr{4095};
    auto lock = TryDiagnosticLock(mutex, DiagnosticSite::WriterNceFault);
    if (!lock) {
        return;
    }
    if (!complete) ++sample.incomplete_inspection;
    if (origin_mask == 0 && complete) ++sample.no_live_writer;
    for (size_t i = 0; i < sample.origins.size(); ++i) {
        if ((origin_mask & (u64{1} << i)) == 0) continue;
        ++sample.origins[i].faults;
        origin_pages[i].insert(page);
    }
    for (size_t i = 0; i < sample.first_reasons.size(); ++i) {
        if ((first_reason_mask & (u64{1} << i)) == 0) continue;
        ++sample.first_reasons[i].faults;
        reason_pages[i].insert(page);
    }
}

WriterNceSnapshot WriterNceCounters::TakeSnapshot() {
    auto lock = TryDiagnosticLock(mutex, DiagnosticSite::WriterNceSnapshot);
    if (!lock) {
        return {};
    }
    auto result = sample;
    for (size_t i = 0; i < origin_pages.size(); ++i) {
        result.origins[i].pages = origin_pages[i].size();
        origin_pages[i].clear();
    }
    for (size_t i = 0; i < reason_pages.size(); ++i) {
        result.first_reasons[i].pages = reason_pages[i].size();
        reason_pages[i].clear();
    }
    sample = {};
    return result;
}

void WriterNceCounters::Reset() { TakeSnapshot(); }

void RecordWriterNceCoverage(bool physical_resolved, bool device_resolved) {
    if (enabled.load(std::memory_order_relaxed))
        writer_nce_counters.RecordCoverage(physical_resolved, device_resolved);
}

void RecordWriterNceFault(PAddr page, u64 origin_mask, u64 first_reason_mask, bool complete) {
    if (enabled.load(std::memory_order_relaxed))
        writer_nce_counters.RecordFault(page, origin_mask, first_reason_mask, complete);
}
namespace {
struct WatchdogSlot {
    std::atomic<WatchdogPhase> phase{WatchdogPhase::Idle};
    std::atomic<u64> begin_ns{};
    WatchdogPhaseSample Read() const {
        return {phase.load(std::memory_order_relaxed), begin_ns.load(std::memory_order_relaxed)};
    }
    void Write(WatchdogPhaseSample sample) {
        begin_ns.store(sample.begin_ns, std::memory_order_relaxed);
        phase.store(sample.phase, std::memory_order_relaxed);
    }
};
std::array<WatchdogSlot, 4> watchdog_cores;
WatchdogSlot watchdog_gpu;
std::atomic<u64> watchdog_frame_ns{}, watchdog_frame_count{};
std::atomic<u64> watchdog_query_active{}, watchdog_query_begin_ns{}, watchdog_query_max_ns{};
constexpr std::array watchdog_phase_names{
    "idle", "host_continuation", "recovery", "decode", "writer_inspection", "publish",
    "native_fallback", "budget_exact", "gpu_tick_work", "gpu_callback", "gpu_lock_acquire", "gpu_diagnostic",
    "gpu_finish", "grant_queued",
};
}

WatchdogCoreScope::WatchdogCoreScope(size_t core, WatchdogPhase phase) {
    if (!enabled.load(std::memory_order_relaxed) || core >= watchdog_cores.size()) return;
    slot = static_cast<u32>(core);
    previous = watchdog_cores[slot].Read();
    watchdog_cores[slot].Write({phase, SuspensionNowNs()});
}
WatchdogCoreScope::~WatchdogCoreScope() {
    if (slot < watchdog_cores.size()) watchdog_cores[slot].Write(previous);
}
WatchdogGpuScope::WatchdogGpuScope(WatchdogPhase phase) {
    if (!enabled.load(std::memory_order_relaxed)) return;
    active = true;
    previous = watchdog_gpu.Read();
    watchdog_gpu.Write({phase, SuspensionNowNs()});
}
WatchdogGpuScope::~WatchdogGpuScope() {
    if (active) watchdog_gpu.Write(previous);
}
WatchdogInspectionScope::WatchdogInspectionScope() : begin{SuspensionNowNs()} {
    if (!begin) return;
    watchdog_query_begin_ns.store(begin, std::memory_order_relaxed);
    watchdog_query_active.fetch_add(1, std::memory_order_relaxed);
}
WatchdogInspectionScope::~WatchdogInspectionScope() {
    if (!begin) return;
    const u64 end = SuspensionNowNs();
    if (end >= begin) UpdateMaximum(watchdog_query_max_ns, end - begin);
    watchdog_query_active.fetch_sub(1, std::memory_order_relaxed);
}
void ResetWatchdog() {
    for (auto& slot : watchdog_cores) slot.Write({});
    watchdog_gpu.Write({});
    watchdog_frame_ns.store(SuspensionNowNs(), std::memory_order_relaxed);
    watchdog_frame_count.store(0, std::memory_order_relaxed);
    watchdog_query_active.store(0, std::memory_order_relaxed);
    watchdog_query_begin_ns.store(0, std::memory_order_relaxed);
    watchdog_query_max_ns.store(0, std::memory_order_relaxed);
}
void WatchdogFrameTick() {
    if (!enabled.load(std::memory_order_relaxed)) return;
    watchdog_frame_ns.store(SuspensionNowNs(), std::memory_order_relaxed);
    watchdog_frame_count.fetch_add(1, std::memory_order_relaxed);
}
WatchdogSnapshot TakeWatchdogSnapshot() {
    WatchdogSnapshot snapshot;
    snapshot.frame_ns = watchdog_frame_ns.load(std::memory_order_relaxed);
    snapshot.frame_count = watchdog_frame_count.load(std::memory_order_relaxed);
    snapshot.query_active = watchdog_query_active.load(std::memory_order_relaxed);
    snapshot.query_last_begin_ns = watchdog_query_begin_ns.load(std::memory_order_relaxed);
    snapshot.query_max_ns = watchdog_query_max_ns.load(std::memory_order_relaxed);
    for (size_t i = 0; i < watchdog_cores.size(); ++i) snapshot.cores[i] = watchdog_cores[i].Read();
    snapshot.gpu = watchdog_gpu.Read();
    return snapshot;
}
void ReportWatchdog() {
    if (!enabled.load(std::memory_order_relaxed)) return;
    const u64 now = SuspensionNowNs();
    const auto snapshot = TakeWatchdogSnapshot();
    if (!snapshot.frame_ns || now < snapshot.frame_ns || now - snapshot.frame_ns < 5'000'000'000) return;
    const auto age_ms = [now](u64 begin) { return begin && now >= begin ? (now - begin) / 1e6 : 0.0; };
    LOG_INFO(HW_GPU, "GPU_WATCHDOG tick_stale_ms={:.3f} ticks={} gpu_phase={} gpu_phase_ms={:.3f} "
                    "query_active={} query_last_begin_age_ms={:.3f} query_max_ms_lifetime={:.3f} "
                    "snapshot=independent_atomic_loads phases_are_operation_markers=1",
             age_ms(snapshot.frame_ns), snapshot.frame_count,
             watchdog_phase_names[static_cast<size_t>(snapshot.gpu.phase)], age_ms(snapshot.gpu.begin_ns),
             snapshot.query_active, age_ms(snapshot.query_last_begin_ns), snapshot.query_max_ns / 1e6);
    for (size_t i = 0; i < snapshot.cores.size(); ++i) {
        LOG_INFO(HW_GPU, "GPU_WATCHDOG core={} phase={} phase_ms={:.3f}", i,
                 watchdog_phase_names[static_cast<size_t>(snapshot.cores[i].phase)],
                 age_ms(snapshot.cores[i].begin_ns));
    }
}
void Configure() {
    {
        std::scoped_lock lock{producer_diagnostics_mutex};
        lease_diagnostics = {};
        rejected_opcodes = {};
        lease_diagnostic_overflow = rejected_opcode_overflow = 0;
    }
    legacy_writer_counters.Reset();
    writer_nce_counters.Reset();
    const bool active = Flag("EDEN_NX2_DEFERRED_WRITES") || Flag("EDEN_NX2_GPU_PROFILE");
    enabled.store(active, std::memory_order_relaxed);
    ResetWatchdog();
    TakeDiagnosticDrops();
    {
        std::scoped_lock lock{budget_pages_mutex};
        for (auto& pages : budget_pages) pages.clear();
    }
    if (!active) return;
    counters.TakeSnapshot();
    byte_counters.TakeSnapshot();
    suspension_counters.TakeSnapshot();
    svc_counters.TakeSnapshot();
    deferred_counters.TakeSnapshot();
    exclusive_counters.TakeSnapshot();
    TakeExclusiveInlineHits();
    last_report = std::chrono::steady_clock::now();
    LOG_INFO(HW_GPU, "GPU_TRANSITIONS enabled=1 counts=function_entries not_completed_transactions=1 "
                    "protect=explicit_ProtectRegion_only window_seconds=10 "
                    "coverage=after_gpu_init shutdown_tail_may_be_unreported=1");
    LOG_INFO(HW_GPU, "GPU_SUSPENSIONS enabled=1 window_seconds=10 "
                    "wait=successful_caller_elapsed epoch_time=once_per_ready_epoch_by_creator "
                    "pending=nonready_attempts fields_independently_sampled=1");
    LOG_INFO(HW_GPU, "GPU_DEFERRED_WRITES enabled=1 window_seconds=10 "
                    "ring_peak=live_records_after_commit_including_history "
                    "backpressure=full_attempts_plus_explicit_events drains=cut_requests "
                    "budget_rejection_counts=all_declined_publications budget_exact_in_suspended_exact=1 "
                    "fields_independently_sampled=1");
}

bool Report() {
    if (!enabled.load(std::memory_order_relaxed)) return false;
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    if (seconds < 10.0) return false;
    last_report = now;
    ReportProducerDiagnostics(seconds);
    const auto writer_faults = writer_nce_counters.TakeSnapshot();
    LOG_INFO(HW_GPU, "GPU_WRITER_NCE window={:.3f}s total={} physical_resolved={} physical_unresolved={} "
                     "device_resolved={} device_unresolved={} no_live_writer_in_resolved_spaces={} incomplete_inspection={} matches=nonexclusive_live_permissions",
             seconds, writer_faults.total, writer_faults.physical_resolved,
             writer_faults.total - writer_faults.physical_resolved, writer_faults.device_resolved,
             writer_faults.total - writer_faults.device_resolved, writer_faults.no_live_writer, writer_faults.incomplete_inspection);
    constexpr std::array origin_names{"temporary", "nce_fallback_persistent", "kernel_jit", "legacy_writer"};
    for (size_t i = 0; i < writer_faults.origins.size(); ++i) {
        LOG_INFO(HW_GPU, "GPU_WRITER_NCE window={:.3f}s writer_origin={} faults={} physical_pages={}",
                 seconds, origin_names[i], writer_faults.origins[i].faults, writer_faults.origins[i].pages);
    }
    for (size_t i = 0; i < writer_faults.first_reasons.size(); ++i) {
        LOG_INFO(HW_GPU, "GPU_WRITER_NCE window={:.3f}s first_fallback_reason={} faults={} physical_pages={}",
                 seconds, i < deferred_fallback_names.size() ? deferred_fallback_names[i] : "unknown", writer_faults.first_reasons[i].faults,
                 writer_faults.first_reasons[i].pages);
    }
    const auto legacy = legacy_writer_counters.TakeSnapshot();
    for (size_t i = 0; i < legacy.size(); ++i) {
        const auto& sample = legacy[i];
        LOG_INFO(HW_GPU, "GPU_LEGACY_WRITERS window={:.3f}s site={} attribution=first_legacy_touch new_wins_within_window=1 "
                        "calls={} bytes={} size_le16={} size_17_64={} size_65_256={} size_257_4096={} "
                        "size_4097_65536={} size_gt65536={} new_physical_pages={} existing_physical_pages={} "
                        "nce_faults={} nce_fault_pages={}",
                 seconds, legacy_writer_site_names[i], sample.calls, sample.bytes,
                 sample.sizes[0], sample.sizes[1], sample.sizes[2], sample.sizes[3], sample.sizes[4],
                 sample.sizes[5], sample.new_pages, sample.existing_pages, sample.nce_faults, sample.nce_fault_pages);
    }
    const auto exclusive = exclusive_counters.TakeSnapshot();
    const u64 inline_hits = TakeExclusiveInlineHits();
    LOG_INFO(HW_GPU, "GPU_DEFERRED_WRITES window={:.3f}s jit_exclusive_hook_calls={} "
                    "jit_exclusive_hook_calls_per_second={:.3f} jit_exclusive_fast_hits={} "
                    "jit_exclusive_host_fast_hits={} jit_exclusive_inline_fast_hits={} "
                    "jit_exclusive_hook_ms={:.3f} jit_exclusive_hook_avg_ns={:.1f} "
                    "jit_exclusive_timing_scope=host_callback",
             seconds, exclusive.hook_calls, exclusive.hook_calls / seconds,
             exclusive.fast_hits + inline_hits, exclusive.fast_hits, inline_hits,
             exclusive.hook_ns / 1e6,
             exclusive.hook_calls ? static_cast<double>(exclusive.hook_ns) / exclusive.hook_calls : 0.0);
    std::array<std::set<PAddr>, deferred_fallback_names.size()> excluded_pages;
    {
        auto lock = TryDiagnosticLock(budget_pages_mutex, DiagnosticSite::BudgetSnapshot);
        if (lock) {
            excluded_pages.swap(budget_pages);
        }
    }
    for (const auto reason : {DeferredFallbackReason::PublicationCost,
                              DeferredFallbackReason::PublicationCoreCost,
                              DeferredFallbackReason::FaultDensity}) {
        const auto index = static_cast<size_t>(reason);
        LOG_INFO(HW_GPU, "GPU_DEFERRED_WRITES window={:.3f}s budget_reason={} excluded_pages={}",
                 seconds, deferred_fallback_names[index], excluded_pages[index].size());
    }
    const auto deferred = deferred_counters.TakeSnapshot();
    for (size_t i = 0; i < deferred.phases.size(); ++i) {
        const auto& phase = deferred.phases[i];
        LOG_INFO(HW_GPU, "GPU_DEFERRED_WRITES window={:.3f}s producer_phase={} calls={} "
                        "total_ms={:.3f} max_ms={:.3f}",
                 seconds, deferred_producer_phase_names[i], phase.calls,
                 phase.total_ns / 1e6, phase.max_ns / 1e6);
    }
    LOG_INFO(HW_GPU, "GPU_DEFERRED_WRITES window={:.3f}s suspended_exact_success={} "
                    "suspended_exact_success_per_second={:.3f} suspended_exact_failed={}",
             seconds, deferred.suspended_exact_success, deferred.suspended_exact_success / seconds,
             deferred.suspended_exact_failed);
    for (size_t i = 1; i < deferred.fallbacks.size(); ++i) {
        LOG_INFO(HW_GPU, "GPU_DEFERRED_WRITES window={:.3f}s fallback_reason={} fallbacks={} "
                        "fallbacks_per_second={:.3f}",
                 seconds, deferred_fallback_names[i], deferred.fallbacks[i], deferred.fallbacks[i] / seconds);
    }
    LOG_INFO(HW_GPU, "GPU_DEFERRED_WRITES window={:.3f}s published={} published_per_second={:.3f} "
                    "published_bytes={} published_bytes_per_second={:.3f} ring_peak={} "
                    "backpressure={} applied={} retired={} recovery_attempts={} recovered={}",
             seconds, deferred.publications[static_cast<size_t>(DeferredPublishStatus::Published)],
             deferred.publications[static_cast<size_t>(DeferredPublishStatus::Published)] / seconds,
             deferred.published_bytes, deferred.published_bytes / seconds, deferred.ring_peak,
             deferred.backpressure, deferred.applied, deferred.retired,
             deferred.recovery_attempts, deferred.recovered);
    for (size_t i = 1; i < deferred.publications.size(); ++i) {
        LOG_INFO(HW_GPU, "GPU_DEFERRED_WRITES window={:.3f}s status={} attempts={} "
                        "attempts_per_second={:.3f}",
                 seconds, deferred_status_names[i], deferred.publications[i],
                 deferred.publications[i] / seconds);
    }
    for (size_t i = 0; i < deferred.drains.size(); ++i) {
        LOG_INFO(HW_GPU, "GPU_DEFERRED_WRITES window={:.3f}s drain_site={} cuts={} "
                        "cuts_per_second={:.3f}",
                 seconds, deferred_drain_names[i], deferred.drains[i], deferred.drains[i] / seconds);
    }
    const auto snapshot = counters.TakeSnapshot();
    const auto byte_snapshot = byte_counters.TakeSnapshot();
    for (size_t i = 0; i < snapshot.size(); ++i) {
        LOG_INFO(HW_GPU, "GPU_TRANSITIONS window={:.3f}s site={} calls={} calls_per_second={:.3f} "
                        "bytes={} bytes_per_second={:.3f}",
                 seconds, site_names[i], snapshot[i], snapshot[i] / seconds,
                 byte_snapshot[i], byte_snapshot[i] / seconds);
    }
    const auto suspensions = suspension_counters.TakeSnapshot();
    for (size_t i = 0; i < suspensions.size(); ++i) {
        const auto& sample = suspensions[i];
        LOG_INFO(HW_GPU, "GPU_SUSPENSIONS window={:.3f}s site={} participations={} "
                        "participations_per_second={:.3f} new_epochs={} new_epochs_per_second={:.3f} ready_prepared={} "
                        "ready_joined={} pending={} failures={} wait_ms={:.3f} wait_max_ms={:.3f} "
                        "epochs_ready={} draining_ready_ms={:.3f} draining_ready_max_ms={:.3f}",
                 seconds, suspension_site_names[i], sample.participations,
                 sample.participations / seconds, sample.new_epochs, sample.new_epochs / seconds, sample.ready_prepared,
                 sample.ready_joined, sample.pending, sample.failures, sample.wait_ns / 1.0e6,
                 sample.wait_max_ns / 1.0e6, sample.epochs_ready, sample.draining_ready_ns / 1.0e6,
                 sample.draining_ready_max_ns / 1.0e6);
    }
    const auto svcs = svc_counters.TakeSnapshot();
    for (size_t i = 0; i < svcs.size(); ++i) {
        const auto& sample = svcs[i];
        if (!(sample.calls || sample.candidates || sample.participations || sample.new_epochs)) continue;
        LOG_INFO(HW_GPU, "GPU_SUSPENSION_SVC window={:.3f}s svc={:#04x} overflow={} calls={} "
                        "calls_per_second={:.3f} guard_candidates={} participations={} new_epochs={} "
                        "new_epochs_per_second={:.3f}",
                 seconds, i, i == SvcSlotCount - 1, sample.calls, sample.calls / seconds,
                 sample.candidates, sample.participations, sample.new_epochs, sample.new_epochs / seconds);
    }
    const auto dropped = TakeDiagnosticDrops();
    for (size_t i = 0; i < dropped.size(); ++i) {
        LOG_INFO(HW_GPU, "GPU_DIAGNOSTIC_DROPS window={:.3f}s site={} dropped={} "
                         "sampling=best_effort retained_on_snapshot_contention=1",
                 seconds, diagnostic_site_names[i], dropped[i]);
    }
    return true;
}

} // namespace Common::GpuTransitionProfile
