// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <map>
#include <mutex>
#include <set>
#include "common/common_types.h"

namespace Common::GpuTransitionProfile {

enum class Site : size_t {
    MemoryMap,
    MemoryUnmap,
    MemoryProtect,
    HeapResize,
    PageTableFinalize,
    ProcessFinalize,
    DeviceMap,
    DeviceUnmap,
    BufferJoin,
    BufferDelete,
    TextureRegister,
    TextureRefresh,
    TextureRefreshUpload,
    DmaCopy,
    NvAsMapQueued,
    NvAsMapExecuted,
    NvAsUnmapQueued,
    NvAsUnmapExecuted,
    NvAsReleaseQueued,
    NvAsReleaseExecuted,
    NvSparseCreateQueued,
    NvSparseCreateExecuted,
    NvSparseDestroyQueued,
    NvSparseDestroyExecuted,
    NvSparseTileQueued,
    NvSparseTileExecuted,
    NvSparseBackMapped,
    NvAsMapReuse,
    HeapMapCall,
    HeapUnmapCall,
    HeapMapMutation,
    HeapUnmapMutation,
    HeapDestroyUnmap,
    SparseWriteCall,
    SparseBlockClaim,
    SparseBackingAllocateCall,
    SparseBackingAllocated,
    SparseBackingMapFailure,
    SparseBackingDiscarded,
    SparseBackingReleased,
    SparseWaitPending,
    SparseMapRequest,
    KernelSyncPhysicalExclude,
    JitExclusiveExclude,
    NceFallbackExclude,
    LockedAccessRemapExclude,
    LegacyWriterExclude,
    Count,
};
inline constexpr size_t SiteCount = static_cast<size_t>(Site::Count);
using Snapshot = std::array<u64, SiteCount>;

class Counters {
public:
    void Record(Site site) noexcept {
        Add(site, 1);
    }
    void Add(Site site, u64 amount) noexcept {
        counts[static_cast<size_t>(site)].fetch_add(amount, std::memory_order_relaxed);
    }
    Snapshot TakeSnapshot() noexcept;

private:
    std::array<std::atomic<u64>, SiteCount> counts{};
};

enum class SuspensionSite : size_t {
    SvcMemory,
    SvcUserBuffer,
    IpcReceive,
    IpcCleanup,
    IpcAsync,
    ThreadFinalize,
    ProcessFinalize,
    Ro,
    Loader,
    Nvdrv,
    Sparse,
    ProducerRecovery,
    LegacyWriter,
    Count,
};
enum class SuspensionEvent : size_t {
    Participation,
    NewEpoch,
    ReadyPrepared,
    ReadyJoined,
    Pending,
    Failure,
    Count,
};
inline constexpr size_t SuspensionSiteCount = static_cast<size_t>(SuspensionSite::Count);

inline constexpr size_t SvcSlotCount = 257; // Last slot aggregates unsupported/out-of-range IDs.
struct SvcSample {
    u64 calls{};
    u64 candidates{};
    u64 participations{};
    u64 new_epochs{};
};
using SvcSnapshot = std::array<SvcSample, SvcSlotCount>;

class SvcCounters {
public:
    void Record(u32 svc, bool candidate, bool participated, bool new_epoch) noexcept;
    SvcSnapshot TakeSnapshot() noexcept;

private:
    struct AtomicSample {
        std::atomic<u64> calls{};
        std::atomic<u64> candidates{};
        std::atomic<u64> participations{};
        std::atomic<u64> new_epochs{};
    };
    std::array<AtomicSample, SvcSlotCount> samples{};
};

struct SuspensionSample {
    u64 participations{};
    u64 new_epochs{};
    u64 ready_prepared{};
    u64 ready_joined{};
    u64 pending{};
    u64 failures{};
    u64 wait_ns{};
    u64 wait_max_ns{};
    u64 epochs_ready{};
    u64 draining_ready_ns{};
    u64 draining_ready_max_ns{};
};
using SuspensionSnapshot = std::array<SuspensionSample, SuspensionSiteCount>;

class SuspensionCounters {
public:
    void Record(SuspensionSite site, SuspensionEvent event) noexcept;
    void RecordReady(SuspensionSite site, bool prepared, u64 wait_ns) noexcept;
    void RecordEpochReady(SuspensionSite origin, u64 draining_ns) noexcept;
    // Each atomic resets independently; a concurrent event can straddle reporting windows.
    SuspensionSnapshot TakeSnapshot() noexcept;

private:
    struct AtomicSample {
        std::array<std::atomic<u64>, static_cast<size_t>(SuspensionEvent::Count)> events{};
        std::atomic<u64> wait_ns{};
        std::atomic<u64> wait_max_ns{};
        std::atomic<u64> epochs_ready{};
        std::atomic<u64> draining_ready_ns{};
        std::atomic<u64> draining_ready_max_ns{};
    };
    std::array<AtomicSample, SuspensionSiteCount> samples{};
};

enum class DeferredPublishStatus : size_t {
    Published,
    Full,
    InvalidWrite,
    Rejected,
    SequenceExhausted,
    LeaseUnavailable,
    Suspended,
    Count,
};
enum class DeferredDrainSite : size_t {
    SubmitList,
    Acquire,
    Readback,
    Transition,
    Count,
};

enum class DeferredFallbackReason : size_t {
    Success, Disabled, Read, MissingEsr, UnsupportedFault, UnsupportedInstruction,
    InvalidOperands, PageCrossing, Unaligned, AddressMismatch, MappingChanged, AliasedMapping,
    NoGpuBuffer, NoGpuBufferBounds, CleanPage, BusyBuffer, CpuModified, PendingWrites,
    PendingQueries, GpuTexture, BackendUnsupported, WrongThread, PageBudget,
    DcZvaProhibited, DcZvaBlockSize, LeaseUnavailable, QueueFull, Suspended,
    IneligibleAlias, IneligibleShared, IneligibleCode, IneligibleCaptured, IneligibleCpuState,
    CommitRejected, IneligibleTransfer, LeaseExisting, LeaseClosing, LeaseTransition,
    LeaseFreshIdentity, LeaseWriterExcluded, LeaseGenerationExhausted,
    PublicationCost, FaultDensity, PublicationCoreCost, Count,
};
inline constexpr std::array deferred_fallback_names{
    "success", "disabled", "read", "missing_esr", "unsupported_fault", "unsupported_instruction",
    "invalid_operands", "page_crossing", "unaligned", "address_mismatch", "mapping_changed",
    "aliased_mapping", "no_gpu_buffer", "no_gpu_buffer_bounds", "clean_page", "busy_buffer",
    "cpu_modified", "pending_writes", "pending_queries", "gpu_texture", "backend_unsupported",
    "wrong_thread", "page_budget", "dc_zva_prohibited", "dc_zva_block_size", "lease_unavailable",
    "full", "suspended", "ineligible_alias", "ineligible_shared", "ineligible_code",
    "ineligible_captured", "ineligible_cpu_state", "commit_rejected", "ineligible_transfer",
    "lease_existing", "lease_closing", "lease_transition", "lease_fresh_identity",
    "lease_writer_excluded", "lease_generation_exhausted",
    "publication_cost", "fault_density", "publication_core_cost",
};
static_assert(deferred_fallback_names.size() == static_cast<size_t>(DeferredFallbackReason::Count));
inline constexpr size_t DeferredPublishStatusCount = static_cast<size_t>(DeferredPublishStatus::Count);
inline constexpr size_t DeferredDrainSiteCount = static_cast<size_t>(DeferredDrainSite::Count);

enum class DeferredProducerPhase : size_t {
    Recovery, LeaseGrant, SuspensionReady, SuspendedExact, NativeFallback, PublishAttempt,
    PublishCpuAdmission, PublishDeviceAdmission, PublishLeaseLookup, PublishCommit,
    LeaseGrantQueue, LeaseGrantCallback, LeaseGrantReturn, LeaseGrantAvoided, BudgetExact,
    BudgetExactSuccess, BudgetExactFailed, Count,
};
inline constexpr std::array deferred_producer_phase_names{
    "recovery", "lease_grant", "suspension_ready", "suspended_exact", "native_fallback", "publish_attempt",
    "publish_cpu_admission", "publish_device_admission", "publish_lease_lookup", "publish_commit",
    "lease_grant_queue", "lease_grant_callback", "lease_grant_return", "lease_grant_avoided", "budget_exact",
    "budget_exact_success", "budget_exact_failed",
};
static_assert(deferred_producer_phase_names.size() == static_cast<size_t>(DeferredProducerPhase::Count));
struct DeferredPhaseSample { u64 calls{}, total_ns{}, max_ns{}; };

struct DeferredSnapshot {
    std::array<DeferredPhaseSample, deferred_producer_phase_names.size()> phases{};
    u64 suspended_exact_success{};
    u64 suspended_exact_failed{};
    std::array<u64, deferred_fallback_names.size()> fallbacks{};
    std::array<u64, DeferredPublishStatusCount> publications{};
    u64 published_bytes{};
    u64 ring_peak{};
    u64 backpressure{};
    std::array<u64, DeferredDrainSiteCount> drains{};
    u64 applied{};
    u64 retired{};
    u64 recovery_attempts{};
    u64 recovered{};
};

class DeferredCounters {
public:
    // Occupancy is the publishing ring's live records, including retained history, after commit.
    // Full counts one backpressure event; rejected attempts never contribute bytes or occupancy.
    void RecordPublish(DeferredPublishStatus status, u64 bytes = 0, u64 ring_occupancy = 0) noexcept;
    void RecordDrain(DeferredDrainSite site) noexcept;
    void RecordApplied(u64 count = 1) noexcept;
    void RecordRetired(u64 count) noexcept;
    void RecordBackpressure(u64 count = 1) noexcept;
    void RecordRecovery(bool ready) noexcept;
    void RecordFallback(DeferredFallbackReason reason) noexcept;
    void RecordPhase(DeferredProducerPhase phase, u64 ns) noexcept;
    void RecordSuspendedExact(bool success) noexcept;
    // Fields reset independently, like SuspensionCounters. Maxima cover samples in this window.
    DeferredSnapshot TakeSnapshot() noexcept;

private:
    struct PhaseCounters { std::atomic<u64> calls{}, total_ns{}, max_ns{}; };
    std::array<PhaseCounters, deferred_producer_phase_names.size()> phases{};
    std::atomic<u64> suspended_exact_success{}, suspended_exact_failed{};
    std::array<std::atomic<u64>, deferred_fallback_names.size()> fallbacks{};
    std::array<std::atomic<u64>, DeferredPublishStatusCount> publications{};
    std::atomic<u64> published_bytes{};
    std::atomic<u64> ring_peak{};
    std::atomic<u64> backpressure{};
    std::array<std::atomic<u64>, DeferredDrainSiteCount> drains{};
    std::atomic<u64> applied{};
    std::atomic<u64> retired{};
    std::atomic<u64> recovery_attempts{};
    std::atomic<u64> recovered{};
};

struct ExclusiveSnapshot {
    u64 hook_calls{};
    u64 fast_hits{};
    u64 hook_ns{};
};

class ExclusiveCounters {
public:
    void Record(bool fast, u64 elapsed_ns) noexcept;
    ExclusiveSnapshot TakeSnapshot() noexcept;
private:
    std::atomic<u64> calls{};
    std::atomic<u64> fast_hits{};
    std::atomic<u64> hook_ns{};
};

// One executing JIT is the only writer. Reporting only loads cumulative hits; it never resets
// the word, so generated code can increment without a locked RMW or a shared cache line.
struct alignas(64) ExclusiveInlineCounter {
    std::atomic<u64> hits{};
    u64 reported{};
};
std::shared_ptr<ExclusiveInlineCounter> RegisterExclusiveInlineCounter();

// Configured before guest/GPU workers start; not a live toggle.
inline std::atomic<bool> enabled{};
extern Counters counters;
extern Counters byte_counters;
extern SuspensionCounters suspension_counters;
extern SvcCounters svc_counters;
extern DeferredCounters deferred_counters;
extern ExclusiveCounters exclusive_counters;
void RecordExclusiveHook(bool fast, u64 begin_ns) noexcept;
void RecordBudgetExcludedPage(DeferredFallbackReason reason, PAddr page);
void RecordBudgetCoreEvent(bool activated, u32 slow, u64 max_ns, u32 core_index = 0);

inline void RecordDeferredFallback(DeferredFallbackReason reason) noexcept {
    if (enabled.load(std::memory_order_relaxed)) deferred_counters.RecordFallback(reason);
}

inline void RecordDeferredPublish(DeferredPublishStatus status, u64 bytes = 0,
                                  u64 ring_occupancy = 0) noexcept {
    if (enabled.load(std::memory_order_relaxed)) {
        deferred_counters.RecordPublish(status, bytes, ring_occupancy);
    }
}

inline void RecordDeferredDrain(DeferredDrainSite site) noexcept {
    if (enabled.load(std::memory_order_relaxed)) deferred_counters.RecordDrain(site);
}

inline void RecordDeferredApplied(u64 count = 1) noexcept {
    if (enabled.load(std::memory_order_relaxed)) deferred_counters.RecordApplied(count);
}

inline void RecordDeferredRetired(u64 count) noexcept {
    if (enabled.load(std::memory_order_relaxed)) deferred_counters.RecordRetired(count);
}

inline void RecordDeferredBackpressure(u64 count = 1) noexcept {
    if (enabled.load(std::memory_order_relaxed)) deferred_counters.RecordBackpressure(count);
}

inline void RecordDeferredRecovery(bool recovered) noexcept {
    if (enabled.load(std::memory_order_relaxed)) deferred_counters.RecordRecovery(recovered);
}

inline void RecordSvc(u32 svc, bool candidate, bool participated, bool new_epoch) noexcept {
    if (enabled.load(std::memory_order_relaxed)) {
        svc_counters.Record(svc, candidate, participated, new_epoch);
    }
}

inline void Record(Site site) noexcept {
    if (enabled.load(std::memory_order_relaxed)) counters.Record(site);
}

inline void RecordRange(Site site, u64 bytes) noexcept {
    if (!enabled.load(std::memory_order_relaxed)) return;
    counters.Record(site);
    byte_counters.Add(site, bytes);
}

inline void RecordSuspension(SuspensionSite site, SuspensionEvent event) noexcept {
    if (enabled.load(std::memory_order_relaxed)) suspension_counters.Record(site, event);
}

inline void RecordSuspensionParticipation(SuspensionSite site, bool new_epoch) noexcept {
    if (!enabled.load(std::memory_order_relaxed)) return;
    suspension_counters.Record(site, SuspensionEvent::Participation);
    if (new_epoch) suspension_counters.Record(site, SuspensionEvent::NewEpoch);
}

// Off returns zero without reading a clock. Starts must survive in the caller/epoch metadata.
[[nodiscard]] u64 SuspensionNowNs() noexcept;

// Host continuation only. Disabled profiling does not read the clock.
class DeferredProducerScope {
public:
    explicit DeferredProducerScope(DeferredProducerPhase phase_) noexcept
        : phase{phase_}, begin{SuspensionNowNs()} {}
    ~DeferredProducerScope() {
        if (begin) deferred_counters.RecordPhase(phase, SuspensionNowNs() - begin);
    }
    DeferredProducerScope(const DeferredProducerScope&) = delete;
    DeferredProducerScope& operator=(const DeferredProducerScope&) = delete;
private:
    DeferredProducerPhase phase;
    u64 begin;
};
void RecordSuspensionReady(SuspensionSite site, bool prepared, u64 caller_begin_ns) noexcept;
// Exactly once per Draining -> Ready, attributed to the epoch's original creator.
void RecordSuspensionEpochReady(SuspensionSite origin, u64 epoch_begin_ns) noexcept;

enum class LegacyWriterSite : size_t {
    Unknown, Write, WriteBlock, WriteBlockUnsafe, ZeroBlock, CopyBlock, Kernel, Audio, Jit,
    ReadDebugMemory, WriteDebugMemory, ReadDebugIoMemory, IpcUserBuffer, TransferMemory, CodeMemory,
    LinearToUser, UserToLinear, KernelToLinear, HeapToHeap, HeapToHeapUnchecked, Count,
};
inline constexpr std::array legacy_writer_site_names{
    "unknown", "write", "write_block", "write_block_unsafe", "zero_block", "copy_block",
    "kernel", "audio", "jit",
    "read_debug_memory", "write_debug_memory", "read_debug_io_memory", "ipc_user_buffer",
    "transfer_memory", "code_memory", "linear_to_user", "user_to_linear", "kernel_to_linear",
    "heap_to_heap", "heap_to_heap_unchecked",
};
static_assert(legacy_writer_site_names.size() == static_cast<size_t>(LegacyWriterSite::Count));
struct LegacyWriterSample {
    u64 calls{}, bytes{};
    std::array<u64, 6> sizes{};
    u64 new_pages{}, existing_pages{}, nce_faults{}, nce_fault_pages{};
};
using LegacyWriterSnapshot = std::array<LegacyWriterSample, static_cast<size_t>(LegacyWriterSite::Count)>;
class LegacyWriterCounters {
public:
    void RecordCall(LegacyWriterSite site, u64 bytes);
    void RecordPhysical(LegacyWriterSite site, PAddr page, bool newly_excluded);
    void RecordFault(PAddr page);
    LegacyWriterSnapshot TakeSnapshot();
    void Reset();
private:
    std::mutex mutex;
    LegacyWriterSnapshot samples{};
    struct Calls {
        std::atomic<u64> count{}, bytes{};
        std::array<std::atomic<u64>, 6> sizes{};
    };
    std::array<Calls, static_cast<size_t>(LegacyWriterSite::Count)> calls;
    struct Pages { std::set<PAddr> added, existing, faulted; };
    std::array<Pages, static_cast<size_t>(LegacyWriterSite::Count)> pages;
    std::map<PAddr, LegacyWriterSite> first_sites;
};
extern LegacyWriterCounters legacy_writer_counters;
class LegacyWriterScope {
public:
    LegacyWriterScope(LegacyWriterSite site, u64 bytes);
    ~LegacyWriterScope();
    LegacyWriterScope(const LegacyWriterScope&) = delete;
    LegacyWriterScope& operator=(const LegacyWriterScope&) = delete;
private:
    bool outer{};
};
LegacyWriterSite CurrentLegacyWriterSite() noexcept;
void RecordLegacyPhysicalPage(PAddr page, bool newly_excluded);
void RecordLegacyNceFault(PAddr page);
inline constexpr size_t WriterNceOriginCount = 4;
struct WriterNceSample { u64 faults{}, pages{}; };
struct WriterNceSnapshot {
    u64 total{}, physical_resolved{}, device_resolved{}, no_live_writer{}, incomplete_inspection{};
    std::array<WriterNceSample, WriterNceOriginCount> origins{};
    std::array<WriterNceSample, deferred_fallback_names.size() + 1> first_reasons{};
};
class WriterNceCounters {
public:
    void RecordCoverage(bool physical_resolved, bool device_resolved);
    void RecordFault(PAddr page, u64 origin_mask, u64 first_reason_mask, bool complete = true);
    WriterNceSnapshot TakeSnapshot();
    void Reset();
private:
    std::mutex mutex;
    WriterNceSnapshot sample{};
    std::array<std::set<PAddr>, WriterNceOriginCount> origin_pages;
    std::array<std::set<PAddr>, deferred_fallback_names.size() + 1> reason_pages;
};
extern WriterNceCounters writer_nce_counters;
void RecordWriterNceCoverage(bool physical_resolved, bool device_resolved);
void RecordWriterNceFault(PAddr page, u64 origin_mask, u64 first_reason_mask, bool complete = true);
enum class WatchdogPhase : u32 {
    Idle, HostContinuation, Recovery, Decode, WriterInspection, Publish, NativeFallback,
    BudgetExact, GpuTickWork, GpuCallback, GpuLockAcquire, GpuDiagnostic, GpuFinish, GrantQueued,
};
struct WatchdogPhaseSample {
    WatchdogPhase phase{WatchdogPhase::Idle};
    u64 begin_ns{};
};
struct WatchdogSnapshot {
    u64 frame_ns{}, frame_count{}, query_active{}, query_last_begin_ns{}, query_max_ns{};
    std::array<WatchdogPhaseSample, 4> cores{};
    WatchdogPhaseSample gpu;
};
class WatchdogCoreScope {
public:
    WatchdogCoreScope(size_t core, WatchdogPhase phase);
    ~WatchdogCoreScope();
    WatchdogCoreScope(const WatchdogCoreScope&) = delete;
    WatchdogCoreScope& operator=(const WatchdogCoreScope&) = delete;
private:
    u32 slot{4};
    WatchdogPhaseSample previous;
};
class WatchdogGpuScope {
public:
    explicit WatchdogGpuScope(WatchdogPhase phase);
    ~WatchdogGpuScope();
    WatchdogGpuScope(const WatchdogGpuScope&) = delete;
    WatchdogGpuScope& operator=(const WatchdogGpuScope&) = delete;
private:
    bool active{};
    WatchdogPhaseSample previous;
};
class WatchdogInspectionScope {
public:
    WatchdogInspectionScope();
    ~WatchdogInspectionScope();
    WatchdogInspectionScope(const WatchdogInspectionScope&) = delete;
    WatchdogInspectionScope& operator=(const WatchdogInspectionScope&) = delete;
private:
    u64 begin{};
};
void ResetWatchdog();
void WatchdogFrameTick();
WatchdogSnapshot TakeWatchdogSnapshot();
void ReportWatchdog();
// Stage and cause strings must have static storage duration.
void RecordLeaseDiagnostic(const char* stage, const char* cause, PAddr physical_page);
void RecordRejectedNceOpcode(u32 instruction, u64 pc);
enum class DiagnosticSite : size_t {
    JournalInspection, JournalSnapshot, DeviceMappingInspection, LegacyPages, LegacyFault,
    LegacySnapshot, WriterNceFault, WriterNceCoverage, WriterNceSnapshot, LeaseDetail,
    RejectedOpcode, ProducerSnapshot, CallerStacks, ExclusiveSnapshot, BudgetPages, BudgetSnapshot, Count,
};
inline constexpr size_t DiagnosticSiteCount = static_cast<size_t>(DiagnosticSite::Count);
void RecordDiagnosticDrop(DiagnosticSite site) noexcept;
std::array<u64, DiagnosticSiteCount> TakeDiagnosticDrops() noexcept;
[[nodiscard]] std::unique_lock<std::mutex> TryDiagnosticLock(std::mutex& mutex, DiagnosticSite site);
void Configure();
bool Report();

} // namespace Common::GpuTransitionProfile
