// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <future>
#include <thread>
#include <vector>
#include "common/gpu_transition_profile.h"

TEST_CASE("NCE fallback profile includes rejections before publication and resets independently",
          "[deferred_writes][gpu_transition_profile][deferred_nce_publication]") {
    using namespace Common::GpuTransitionProfile;
    DeferredCounters local;
    local.RecordFallback(DeferredFallbackReason::IneligibleAlias);
    local.RecordFallback(DeferredFallbackReason::IneligibleShared);
    local.RecordFallback(DeferredFallbackReason::QueueFull);
    local.RecordFallback(DeferredFallbackReason::Success);
    local.RecordPublish(DeferredPublishStatus::Full);
    const auto snapshot = local.TakeSnapshot();
    CHECK(snapshot.fallbacks[static_cast<size_t>(DeferredFallbackReason::IneligibleAlias)] == 1);
    CHECK(snapshot.fallbacks[static_cast<size_t>(DeferredFallbackReason::IneligibleShared)] == 1);
    CHECK(snapshot.fallbacks[static_cast<size_t>(DeferredFallbackReason::QueueFull)] == 1);
    CHECK(snapshot.fallbacks[0] == 0);
    CHECK(snapshot.publications[static_cast<size_t>(DeferredPublishStatus::Full)] == 1);
    CHECK(snapshot.backpressure == 1);
    CHECK(local.TakeSnapshot().fallbacks[static_cast<size_t>(DeferredFallbackReason::QueueFull)] == 0);
}

TEST_CASE("Exclusive hook profile separates fast callbacks from preparation time",
          "[gpu_transition_profile][deferred_exclusive_fast]") {
    Common::GpuTransitionProfile::ExclusiveCounters local;
    local.Record(true, 7);
    local.Record(false, 103);
    const auto sample = local.TakeSnapshot();
    CHECK(sample.hook_calls == 2);
    CHECK(sample.fast_hits == 1);
    CHECK(sample.hook_ns == 110);
    const auto empty = local.TakeSnapshot();
    CHECK(empty.hook_calls == 0);
    CHECK(empty.fast_hits == 0);
    CHECK(empty.hook_ns == 0);
}

TEST_CASE("Deferred profile separates committed bytes failures cuts and retained history",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    DeferredCounters local;
    local.RecordPublish(DeferredPublishStatus::Published, 8, 1);
    local.RecordPublish(DeferredPublishStatus::Published, 32, 7);
    local.RecordPublish(DeferredPublishStatus::Published, 4, 3);
    for (size_t i = 1; i < DeferredPublishStatusCount; ++i) {
        local.RecordPublish(static_cast<DeferredPublishStatus>(i), 1000, 999);
    }
    local.RecordDrain(DeferredDrainSite::SubmitList);
    local.RecordDrain(DeferredDrainSite::Acquire);
    local.RecordDrain(DeferredDrainSite::Acquire);
    local.RecordDrain(DeferredDrainSite::Readback);
    local.RecordDrain(DeferredDrainSite::Transition);
    local.RecordApplied(3);
    local.RecordRetired(1); // Applied history can remain pinned by a readback.
    local.RecordBackpressure(2);
    local.RecordRecovery(false);
    local.RecordRecovery(true);
    const auto sample = local.TakeSnapshot();
    CHECK(sample.publications[static_cast<size_t>(DeferredPublishStatus::Published)] == 3);
    for (size_t i = 1; i < DeferredPublishStatusCount; ++i) CHECK(sample.publications[i] == 1);
    CHECK(sample.published_bytes == 44);
    CHECK(sample.ring_peak == 7);
    CHECK(sample.backpressure == 3);
    CHECK(sample.drains[static_cast<size_t>(DeferredDrainSite::SubmitList)] == 1);
    CHECK(sample.drains[static_cast<size_t>(DeferredDrainSite::Acquire)] == 2);
    CHECK(sample.drains[static_cast<size_t>(DeferredDrainSite::Readback)] == 1);
    CHECK(sample.drains[static_cast<size_t>(DeferredDrainSite::Transition)] == 1);
    CHECK(sample.applied == 3);
    CHECK(sample.retired == 1);
    CHECK(sample.recovery_attempts == 2);
    CHECK(sample.recovered == 1);
    const auto cleared = local.TakeSnapshot();
    for (const auto count : cleared.publications) CHECK(count == 0);
    for (const auto count : cleared.drains) CHECK(count == 0);
    CHECK(cleared.published_bytes == 0);
    CHECK(cleared.ring_peak == 0);
    CHECK(cleared.backpressure == 0);
    CHECK(cleared.applied == 0);
    CHECK(cleared.retired == 0);
    CHECK(cleared.recovery_attempts == 0);
    CHECK(cleared.recovered == 0);
    local.RecordPublish(DeferredPublishStatus::Published, 16, 2);
    CHECK(local.TakeSnapshot().ring_peak == 2);
}

TEST_CASE("Deferred counters accumulate concurrent producers and consumer snapshots",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    DeferredCounters local;
    constexpr u64 attempts = 2000;
    constexpr u64 workers = 4;
    std::vector<std::jthread> threads;
    for (u64 core = 0; core < workers; ++core) {
        threads.emplace_back([&, core] {
            for (u64 i = 0; i < attempts; ++i) {
                local.RecordPublish(DeferredPublishStatus::Published, 4, core + 1);
                local.RecordPublish(DeferredPublishStatus::Full);
                local.RecordDrain(DeferredDrainSite::SubmitList);
                local.RecordApplied();
                local.RecordRetired(1);
                local.RecordRecovery(i % 2 == 0);
            }
        });
    }
    DeferredSnapshot total{};
    const auto collect = [&] {
        const auto sample = local.TakeSnapshot();
        for (size_t i = 0; i < DeferredPublishStatusCount; ++i) {
            total.publications[i] += sample.publications[i];
        }
        for (size_t i = 0; i < DeferredDrainSiteCount; ++i) total.drains[i] += sample.drains[i];
        total.published_bytes += sample.published_bytes;
        if (sample.ring_peak > total.ring_peak) total.ring_peak = sample.ring_peak;
        total.backpressure += sample.backpressure;
        total.applied += sample.applied;
        total.retired += sample.retired;
        total.recovery_attempts += sample.recovery_attempts;
        total.recovered += sample.recovered;
    };
    for (unsigned i = 0; i < 32; ++i) {
        collect();
        std::this_thread::yield();
    }
    threads.clear();
    collect();
    CHECK(total.publications[static_cast<size_t>(DeferredPublishStatus::Published)] == attempts * workers);
    CHECK(total.publications[static_cast<size_t>(DeferredPublishStatus::Full)] == attempts * workers);
    CHECK(total.publications[static_cast<size_t>(DeferredPublishStatus::Rejected)] == 0);
    CHECK(total.published_bytes == attempts * workers * 4);
    CHECK(total.ring_peak == workers);
    CHECK(total.backpressure == attempts * workers);
    CHECK(total.drains[static_cast<size_t>(DeferredDrainSite::SubmitList)] == attempts * workers);
    CHECK(total.drains[static_cast<size_t>(DeferredDrainSite::Acquire)] == 0);
    CHECK(total.applied == attempts * workers);
    CHECK(total.retired == attempts * workers);
    CHECK(total.recovery_attempts == attempts * workers);
    CHECK(total.recovered == attempts * workers / 2);
}

TEST_CASE("Abandoned suspension recovery has its own preparation profile site",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    SuspensionCounters local;
    local.Record(SuspensionSite::ProducerRecovery, SuspensionEvent::Participation);
    local.RecordReady(SuspensionSite::ProducerRecovery, true, 50);
    const auto sample = local.TakeSnapshot();
    const auto& recovery = sample[static_cast<size_t>(SuspensionSite::ProducerRecovery)];
    CHECK(recovery.participations == 1);
    CHECK(recovery.ready_prepared == 1);
    CHECK(recovery.new_epochs == 0);
    CHECK(recovery.wait_ns == 50);
    CHECK(sample[static_cast<size_t>(SuspensionSite::SvcMemory)].participations == 0);
}

TEST_CASE("Disabled deferred profile wrappers do not record events", "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    const bool previous_enabled = enabled.exchange(false, std::memory_order_relaxed);
    deferred_counters.TakeSnapshot();
    RecordDeferredPublish(DeferredPublishStatus::Published, 16, 4);
    RecordDeferredPublish(DeferredPublishStatus::Full);
    RecordDeferredDrain(DeferredDrainSite::Readback);
    RecordDeferredApplied(2);
    RecordDeferredRetired(1);
    RecordDeferredBackpressure();
    RecordDeferredRecovery(true);
    const auto sample = deferred_counters.TakeSnapshot();
    enabled.store(previous_enabled, std::memory_order_relaxed);
    for (const auto count : sample.publications) CHECK(count == 0);
    for (const auto count : sample.drains) CHECK(count == 0);
    CHECK(sample.published_bytes == 0);
    CHECK(sample.ring_peak == 0);
    CHECK(sample.backpressure == 0);
    CHECK(sample.applied == 0);
    CHECK(sample.retired == 0);
    CHECK(sample.recovery_attempts == 0);
    CHECK(sample.recovered == 0);
}

TEST_CASE("SVC profile distinguishes queries candidates participants and new epochs by ID",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    SvcCounters local;
    local.Record(0x06, false, false, false); // QueryMemory never requests a global guard.
    local.Record(0x1d, false, false, false); // SignalProcessWideKey closes outside the lock.
    local.Record(0x16, true, false, false);  // B disabled, but profile enabled.
    local.Record(0x16, true, true, true);
    local.Record(0x16, true, true, false);   // Nested participant in an existing epoch.
    local.Record(0x80, true, true, true);
    local.Record(0xffffffff, false, false, false);
    const auto snapshot = local.TakeSnapshot();
    CHECK(snapshot[0x06].calls == 1);
    CHECK(snapshot[0x06].candidates == 0);
    CHECK(snapshot[0x06].participations == 0);
    CHECK(snapshot[0x1d].calls == 1);
    CHECK(snapshot[0x1d].new_epochs == 0);
    CHECK(snapshot[0x16].calls == 3);
    CHECK(snapshot[0x16].candidates == 3);
    CHECK(snapshot[0x16].participations == 2);
    CHECK(snapshot[0x16].new_epochs == 1);
    CHECK(snapshot[0x80].new_epochs == 1);
    CHECK(snapshot.back().calls == 1);
    for (const auto& sample : local.TakeSnapshot()) {
        CHECK(sample.calls == 0);
        CHECK(sample.candidates == 0);
        CHECK(sample.participations == 0);
        CHECK(sample.new_epochs == 0);
    }
}

TEST_CASE("SVC counters aggregate concurrent cores without mixing IDs",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    SvcCounters local;
    std::vector<std::thread> cores;
    for (u32 core = 0; core < 4; ++core) {
        cores.emplace_back([&, core] {
            for (unsigned i = 0; i < 2000; ++i) {
                local.Record(core & 1 ? 0x16 : 0x03, true, true, (i % 2) == 0);
            }
        });
    }
    for (auto& core : cores) core.join();
    const auto snapshot = local.TakeSnapshot();
    for (const u32 id : {0x03, 0x16}) {
        CHECK(snapshot[id].calls == 4000);
        CHECK(snapshot[id].candidates == 4000);
        CHECK(snapshot[id].participations == 4000);
        CHECK(snapshot[id].new_epochs == 2000);
    }
    CHECK(snapshot[0x06].calls == 0);
}

TEST_CASE("GPU transition counters separate sites and reset snapshots", "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    Counters local;
    for (const auto count : local.TakeSnapshot()) CHECK(count == 0);
    local.Record(Site::MemoryMap);
    local.Record(Site::MemoryMap);
    local.Record(Site::MemoryUnmap);
    local.Record(Site::TextureRefresh);
    auto snapshot = local.TakeSnapshot();
    CHECK(snapshot[static_cast<size_t>(Site::MemoryMap)] == 2);
    CHECK(snapshot[static_cast<size_t>(Site::MemoryUnmap)] == 1);
    CHECK(snapshot[static_cast<size_t>(Site::TextureRefresh)] == 1);
    CHECK(snapshot[static_cast<size_t>(Site::TextureRefreshUpload)] == 0);
    for (const auto count : local.TakeSnapshot()) CHECK(count == 0);
    local.Record(Site::TextureRefreshUpload);
    snapshot = local.TakeSnapshot();
    CHECK(snapshot[static_cast<size_t>(Site::MemoryMap)] == 0);
    CHECK(snapshot[static_cast<size_t>(Site::TextureRefreshUpload)] == 1);
}

TEST_CASE("Mapping profile separates reuse queued execution and actual backing bytes",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    Counters calls, bytes;
    const auto record = [&](Site site, u64 size) { calls.Record(site); bytes.Add(site, size); };
    record(Site::NvAsMapQueued, 0x20000);
    record(Site::NvAsMapReuse, 0x20000);
    record(Site::NvAsMapExecuted, 0x20000);
    record(Site::HeapMapCall, 0x3000);
    record(Site::HeapMapMutation, 0x1000); // Only the uncovered page was mapped.
    record(Site::SparseWriteCall, 4);
    record(Site::SparseBlockClaim, 4 * 1024 * 1024);
    auto sample = calls.TakeSnapshot();
    auto sizes = bytes.TakeSnapshot();
    CHECK(sample[static_cast<size_t>(Site::DeviceMap)] == 0);
    CHECK(sample[static_cast<size_t>(Site::NvAsMapExecuted)] == 1);
    CHECK(sample[static_cast<size_t>(Site::NvAsMapReuse)] == 1);
    CHECK(sizes[static_cast<size_t>(Site::HeapMapCall)] == 0x3000);
    CHECK(sizes[static_cast<size_t>(Site::HeapMapMutation)] == 0x1000);
    CHECK(sizes[static_cast<size_t>(Site::SparseWriteCall)] == 4);
    CHECK(sizes[static_cast<size_t>(Site::SparseBlockClaim)] == 4 * 1024 * 1024);
    CHECK(sample[static_cast<size_t>(Site::SparseBackingAllocated)] == 0);
    record(Site::NvSparseTileExecuted, 0x1000); // Completion may land in the following window.
    sample = calls.TakeSnapshot();
    sizes = bytes.TakeSnapshot();
    CHECK(sample[static_cast<size_t>(Site::NvAsMapQueued)] == 0);
    CHECK(sizes[static_cast<size_t>(Site::NvAsMapQueued)] == 0);
    CHECK(sample[static_cast<size_t>(Site::NvSparseTileExecuted)] == 1);
    CHECK(sizes[static_cast<size_t>(Site::NvSparseTileExecuted)] == 0x1000);
}

TEST_CASE("Mapping byte counters accumulate concurrent multi-page mutations",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    Counters local;
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < 4; ++i) {
        workers.emplace_back([&] {
            for (unsigned j = 0; j < 1000; ++j) local.Add(Site::SparseBackingReleased, 4096);
        });
    }
    for (auto& worker : workers) worker.join();
    CHECK(local.TakeSnapshot()[static_cast<size_t>(Site::SparseBackingReleased)] == 4000 * 4096);
    CHECK(local.TakeSnapshot()[static_cast<size_t>(Site::SparseBackingReleased)] == 0);
}

TEST_CASE("Suspension counters separate caller waits from once-per-epoch preparation",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    SuspensionCounters local;
    const auto creator = SuspensionSite::SvcMemory;
    const auto joiner = SuspensionSite::IpcReceive;
    local.Record(creator, SuspensionEvent::Participation);
    local.Record(creator, SuspensionEvent::NewEpoch);
    local.Record(creator, SuspensionEvent::Pending);
    local.RecordReady(creator, true, 100);
    local.Record(joiner, SuspensionEvent::Participation);
    local.RecordReady(joiner, false, 40);
    local.Record(joiner, SuspensionEvent::Participation);
    local.RecordReady(joiner, false, 10);
    local.RecordEpochReady(creator, 80);
    local.Record(SuspensionSite::Ro, SuspensionEvent::Failure);

    const auto snapshot = local.TakeSnapshot();
    const auto& origin = snapshot[static_cast<size_t>(creator)];
    CHECK(origin.participations == 1);
    CHECK(origin.new_epochs == 1);
    CHECK(origin.ready_prepared == 1);
    CHECK(origin.ready_joined == 0);
    CHECK(origin.pending == 1);
    CHECK(origin.wait_ns == 100);
    CHECK(origin.wait_max_ns == 100);
    CHECK(origin.epochs_ready == 1);
    CHECK(origin.draining_ready_ns == 80);
    CHECK(origin.draining_ready_max_ns == 80);
    const auto& joined = snapshot[static_cast<size_t>(joiner)];
    CHECK(joined.participations == 2);
    CHECK(joined.new_epochs == 0);
    CHECK(joined.ready_joined == 2);
    CHECK(joined.wait_ns == 50);
    CHECK(joined.wait_max_ns == 40);
    CHECK(joined.epochs_ready == 0);
    CHECK(joined.draining_ready_ns == 0);
    CHECK(snapshot[static_cast<size_t>(SuspensionSite::Ro)].failures == 1);
    for (const auto& sample : local.TakeSnapshot()) {
        CHECK(sample.participations == 0);
        CHECK(sample.new_epochs == 0);
        CHECK(sample.ready_prepared == 0);
        CHECK(sample.ready_joined == 0);
        CHECK(sample.pending == 0);
        CHECK(sample.failures == 0);
        CHECK(sample.wait_ns == 0);
        CHECK(sample.wait_max_ns == 0);
        CHECK(sample.epochs_ready == 0);
        CHECK(sample.draining_ready_ns == 0);
        CHECK(sample.draining_ready_max_ns == 0);
    }
}

TEST_CASE("Suspension counters accumulate concurrent participants and reset maxima",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    SuspensionCounters local;
    constexpr u64 workers = 4;
    constexpr u64 attempts = 1000;
    std::vector<std::jthread> threads;
    for (u64 worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            local.Record(SuspensionSite::SvcMemory, SuspensionEvent::NewEpoch);
            for (u64 i = 0; i < attempts; ++i) {
                local.Record(SuspensionSite::SvcMemory, SuspensionEvent::Participation);
                local.RecordReady(SuspensionSite::SvcMemory, i % 2 == 0, worker + 1);
                if (i % 10 == 0) local.Record(SuspensionSite::SvcMemory, SuspensionEvent::Pending);
            }
            local.RecordEpochReady(SuspensionSite::SvcMemory, worker + 10);
            local.Record(SuspensionSite::SvcMemory, SuspensionEvent::Failure);
        });
    }
    threads.clear(); // jthread joins before sampling; no assertions in worker threads.
    const auto sample = local.TakeSnapshot()[static_cast<size_t>(SuspensionSite::SvcMemory)];
    CHECK(sample.participations == workers * attempts);
    CHECK(sample.new_epochs == workers);
    CHECK(sample.ready_prepared == workers * attempts / 2);
    CHECK(sample.ready_joined == workers * attempts / 2);
    CHECK(sample.pending == workers * attempts / 10);
    CHECK(sample.failures == workers);
    CHECK(sample.wait_ns == attempts * 10);
    CHECK(sample.wait_max_ns == 4);
    CHECK(sample.epochs_ready == workers);
    CHECK(sample.draining_ready_ns == 46);
    CHECK(sample.draining_ready_max_ns == 13);
    local.RecordReady(SuspensionSite::SvcMemory, false, 1);
    local.RecordEpochReady(SuspensionSite::SvcMemory, 2);
    const auto next = local.TakeSnapshot()[static_cast<size_t>(SuspensionSite::SvcMemory)];
    CHECK(next.participations == 0);
    CHECK(next.ready_joined == 1);
    CHECK(next.wait_ns == 1);
    CHECK(next.wait_max_ns == 1);
    CHECK(next.epochs_ready == 1);
    CHECK(next.draining_ready_ns == 2);
    CHECK(next.draining_ready_max_ns == 2);
}

TEST_CASE("Concurrent suspension snapshots do not lose additive event counts",
          "[gpu_transition_profile]") {
    using namespace Common::GpuTransitionProfile;
    SuspensionCounters local;
    constexpr u64 attempts = 2000;
    std::jthread worker([&] {
        for (u64 i = 0; i < attempts; ++i) {
            local.Record(SuspensionSite::IpcAsync, SuspensionEvent::Participation);
            local.RecordReady(SuspensionSite::IpcAsync, false, 3);
        }
    });
    u64 participations{}, ready{}, wait{};
    const auto collect = [&] {
        const auto sample = local.TakeSnapshot()[static_cast<size_t>(SuspensionSite::IpcAsync)];
        participations += sample.participations;
        ready += sample.ready_joined;
        wait += sample.wait_ns;
    };
    for (u64 i = 0; i < 32; ++i) {
        collect();
        std::this_thread::yield();
    }
    worker.join();
    collect();
    CHECK(participations == attempts);
    CHECK(ready == attempts);
    CHECK(wait == attempts * 3);
}

TEST_CASE("NCE host phase counters separate transition waits and exact transient successes",
          "[gpu_transition_profile][deferred_writes][deferred_nce_publication]") {
    using namespace Common::GpuTransitionProfile;
    DeferredCounters local;
    local.RecordPhase(DeferredProducerPhase::SuspensionReady, 700);
    local.RecordPhase(DeferredProducerPhase::SuspensionReady, 300);
    local.RecordPhase(DeferredProducerPhase::NativeFallback, 12);
    local.RecordSuspendedExact(true);
    local.RecordSuspendedExact(false);
    const auto sample = local.TakeSnapshot();
    const auto& ready = sample.phases[static_cast<size_t>(DeferredProducerPhase::SuspensionReady)];
    CHECK(ready.calls == 2);
    CHECK(ready.total_ns == 1000);
    CHECK(ready.max_ns == 700);
    CHECK(sample.phases[static_cast<size_t>(DeferredProducerPhase::NativeFallback)].total_ns == 12);
    CHECK(sample.suspended_exact_success == 1);
    CHECK(sample.suspended_exact_failed == 1);
    CHECK(sample.publications[static_cast<size_t>(DeferredPublishStatus::Published)] == 0);
    const auto reset = local.TakeSnapshot();
    CHECK(reset.phases[static_cast<size_t>(DeferredProducerPhase::SuspensionReady)].calls == 0);
    CHECK(reset.phases[static_cast<size_t>(DeferredProducerPhase::SuspensionReady)].max_ns == 0);
    CHECK(reset.suspended_exact_success == 0);
}

TEST_CASE("Legacy writer attribution survives windows while physical page counts deduplicate",
          "[gpu_transition_profile][deferred_writes]") {
    using namespace Common::GpuTransitionProfile;
    LegacyWriterCounters local;
    local.RecordCall(LegacyWriterSite::CopyBlock, 65537);
    local.RecordPhysical(LegacyWriterSite::CopyBlock, 0x4001, true);
    local.RecordPhysical(LegacyWriterSite::CopyBlock, 0x4008, true);
    local.RecordPhysical(LegacyWriterSite::CopyBlock, 0x4001, false);
    local.RecordPhysical(LegacyWriterSite::WriteBlock, 0x4000, false);
    local.RecordFault(0x4001);
    local.RecordFault(0x4008);
    local.RecordFault(0x9000);
    const auto first = local.TakeSnapshot();
    const auto& copy = first[static_cast<size_t>(LegacyWriterSite::CopyBlock)];
    CHECK(copy.calls == 1);
    CHECK(copy.bytes == 65537);
    CHECK(copy.sizes[5] == 1);
    CHECK(copy.new_pages == 1);
    CHECK(copy.existing_pages == 0);
    CHECK(copy.nce_faults == 2);
    CHECK(copy.nce_fault_pages == 1);
    CHECK(first[static_cast<size_t>(LegacyWriterSite::WriteBlock)].existing_pages == 1);
    CHECK(first[static_cast<size_t>(LegacyWriterSite::WriteBlock)].nce_faults == 0);
    local.RecordFault(0x4002);
    const auto second = local.TakeSnapshot();
    CHECK(second[static_cast<size_t>(LegacyWriterSite::CopyBlock)].nce_faults == 1);
    CHECK(second[static_cast<size_t>(LegacyWriterSite::CopyBlock)].new_pages == 0);
    local.Reset();
    local.RecordFault(0x4000);
    CHECK(local.TakeSnapshot()[static_cast<size_t>(LegacyWriterSite::CopyBlock)].nce_faults == 0);
}

TEST_CASE("Legacy writer outer scopes suppress nested calls and disabled profiling records nothing",
          "[gpu_transition_profile][deferred_writes]") {
    using namespace Common::GpuTransitionProfile;
    const bool previous = enabled.exchange(true);
    legacy_writer_counters.Reset();
    {
        LegacyWriterScope copy{LegacyWriterSite::CopyBlock, 4097};
        CHECK(CurrentLegacyWriterSite() == LegacyWriterSite::CopyBlock);
        {
            LegacyWriterScope nested{LegacyWriterSite::WriteBlock, 4097};
            CHECK(CurrentLegacyWriterSite() == LegacyWriterSite::CopyBlock);
            RecordLegacyPhysicalPage(0x4000, true);
        }
        CHECK(CurrentLegacyWriterSite() == LegacyWriterSite::CopyBlock);
    }
    CHECK(CurrentLegacyWriterSite() == LegacyWriterSite::Unknown);
    const auto sample = legacy_writer_counters.TakeSnapshot();
    CHECK(sample[static_cast<size_t>(LegacyWriterSite::CopyBlock)].calls == 1);
    CHECK(sample[static_cast<size_t>(LegacyWriterSite::CopyBlock)].sizes[4] == 1);
    CHECK(sample[static_cast<size_t>(LegacyWriterSite::WriteBlock)].calls == 0);
    enabled.store(false);
    {
        LegacyWriterScope disabled{LegacyWriterSite::WriteBlock, 1};
        RecordLegacyPhysicalPage(0x5000, true);
        RecordLegacyNceFault(0x4000);
    }
    const auto off = legacy_writer_counters.TakeSnapshot();
    CHECK(off[static_cast<size_t>(LegacyWriterSite::WriteBlock)].calls == 0);
    CHECK(off[static_cast<size_t>(LegacyWriterSite::WriteBlock)].new_pages == 0);
    CHECK(off[static_cast<size_t>(LegacyWriterSite::CopyBlock)].nce_faults == 0);
    legacy_writer_counters.Reset();
    enabled.store(previous);
}
TEST_CASE("Writer NCE intersections count each live origin and first reason once per fault",
          "[gpu_transition_profile][deferred_writes]") {
    using namespace Common::GpuTransitionProfile;
    WriterNceCounters local;
    const u64 unknown = u64{1} << static_cast<size_t>(DeferredFallbackReason::Count);
    local.RecordCoverage(false, false);
    local.RecordCoverage(true, false);
    local.RecordCoverage(true, true);
    local.RecordCoverage(true, true);
    local.RecordFault(0x4001, 15, unknown);
    local.RecordFault(0x4008, 15, unknown);
    local.RecordFault(0x5000, 0, 0, false);
    const auto sample = local.TakeSnapshot();
    CHECK(sample.total == 4);
    CHECK(sample.physical_resolved == 3);
    CHECK(sample.device_resolved == 2);
    CHECK(sample.incomplete_inspection == 1);
    CHECK(sample.no_live_writer == 0);
    for (const auto& origin : sample.origins) {
        CHECK(origin.faults == 2);
        CHECK(origin.pages == 1);
    }
    CHECK(sample.first_reasons[static_cast<size_t>(DeferredFallbackReason::Count)].faults == 2);
    CHECK(sample.first_reasons[static_cast<size_t>(DeferredFallbackReason::Count)].pages == 1);
    CHECK(local.TakeSnapshot().origins[0].faults == 0);
}
TEST_CASE("Watchdog restores nested host and GPU phases without sampling emulator state",
          "[gpu_transition_profile][deferred_writes]") {
    using namespace Common::GpuTransitionProfile;
    const bool previous = enabled.exchange(true);
    ResetWatchdog();
    const auto initial = TakeWatchdogSnapshot();
    CHECK(initial.frame_ns != 0);
    CHECK(initial.frame_count == 0);
    {
        WatchdogCoreScope host{2, WatchdogPhase::HostContinuation};
        const auto host_begin = TakeWatchdogSnapshot().cores[2].begin_ns;
        CHECK(TakeWatchdogSnapshot().cores[2].phase == WatchdogPhase::HostContinuation);
        {
            WatchdogCoreScope query{2, WatchdogPhase::WriterInspection};
            WatchdogInspectionScope inspection;
            CHECK(TakeWatchdogSnapshot().cores[2].phase == WatchdogPhase::WriterInspection);
            CHECK(TakeWatchdogSnapshot().query_active == 1);
        }
        CHECK(TakeWatchdogSnapshot().cores[2].phase == WatchdogPhase::HostContinuation);
        CHECK(TakeWatchdogSnapshot().cores[2].begin_ns == host_begin);
        CHECK(TakeWatchdogSnapshot().query_active == 0);
        WatchdogGpuScope work{WatchdogPhase::GpuTickWork};
        {
            WatchdogGpuScope callback{WatchdogPhase::GpuCallback};
            CHECK(TakeWatchdogSnapshot().gpu.phase == WatchdogPhase::GpuCallback);
        }
        CHECK(TakeWatchdogSnapshot().gpu.phase == WatchdogPhase::GpuTickWork);
    }
    CHECK(TakeWatchdogSnapshot().cores[2].phase == WatchdogPhase::Idle);
    CHECK(TakeWatchdogSnapshot().gpu.phase == WatchdogPhase::Idle);
    WatchdogFrameTick();
    CHECK(TakeWatchdogSnapshot().frame_count == 1);
    enabled.store(false);
    {
        WatchdogCoreScope disabled{2, WatchdogPhase::Publish};
        WatchdogGpuScope gpu_disabled{WatchdogPhase::GpuCallback};
        WatchdogInspectionScope no_query;
        WatchdogFrameTick();
        CHECK(TakeWatchdogSnapshot().cores[2].phase == WatchdogPhase::Idle);
        CHECK(TakeWatchdogSnapshot().gpu.phase == WatchdogPhase::Idle);
        CHECK(TakeWatchdogSnapshot().query_active == 0);
        CHECK(TakeWatchdogSnapshot().frame_count == 1);
    }
    enabled.store(previous);
}

TEST_CASE("Diagnostic try locks discard contention instead of waiting for the holder",
          "[gpu_transition_profile][deferred_writes]") {
    using namespace Common::GpuTransitionProfile;
    std::mutex mutex;
    std::promise<void> held, release;
    auto released = release.get_future().share();
    std::jthread holder([&] {
        std::scoped_lock lock{mutex};
        held.set_value();
        released.wait();
    });
    held.get_future().wait();
    TakeDiagnosticDrops();
    auto query = std::async(std::launch::async, [&] {
        auto lock = TryDiagnosticLock(mutex, DiagnosticSite::LegacyFault);
        return lock.owns_lock();
    });
    const auto status = query.wait_for(std::chrono::seconds{1});
    release.set_value();
    holder.join();
    CHECK(status == std::future_status::ready);
    CHECK_FALSE(query.get());
    CHECK(TakeDiagnosticDrops()[static_cast<size_t>(DiagnosticSite::LegacyFault)] == 1);
    auto available = TryDiagnosticLock(mutex, DiagnosticSite::LegacyFault);
    CHECK(available.owns_lock());
}