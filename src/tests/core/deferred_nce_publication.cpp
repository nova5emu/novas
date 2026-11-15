// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "core/arm/nce/nce_store_backing.h"
#include "core/arm/nce/deferred_publication_budget.h"
#include "core/deferred_cpu_mappings.h"
#include "video_core/deferred_write_journal.h"

namespace {
using namespace VideoCommon::DeferredWrites;
struct ProducerFixture {
    Core::DeferredCpuMappings mappings{0x10000, true};
    Journal journal{2, 2};
    alignas(16) std::array<u8, 4096> backing{};
    int table{};

    ProducerFixture() {
        mappings.WithMapping(0x10000, 1, 0x4000,
            [](VAddr) { return std::optional<PAddr>{}; }, [] {});
        mappings.AllowHeapRange(&table, 0x10000, 0x11000);
    }

    std::optional<Journal::Lease> Grant() {
        std::optional<Journal::Lease> lease;
        mappings.WithPage(&table, 0x10000, 0x4000, [&](const auto& snapshot) {
            lease = journal.GrantLease(0x4000, 0x8000, snapshot.generation);
        });
        return lease;
    }

    Publication Publish(const Journal::Lease& lease, u8 value, size_t core = 0) {
        Publication result{PublishStatus::Rejected};
        mappings.WithPage(&table, 0x10000, 0x4000, [&](const auto& snapshot) {
            Write write;
            write.physical_address = 0x4003;
            write.device_address = 0x8003;
            write.mapping_generation = snapshot.generation;
            write.lease_generation = lease.generation;
            write.size = 4;
            std::fill_n(write.bytes.begin(), write.size, value);
            result = journal.TryPublishLeased(core, write, [&]() noexcept {
                return Core::ExactStoreBacking::Commit(write.device_address, backing.data() + 3,
                    std::span{write.bytes}.first(write.size), {4, false, false});
            });
        });
        return result;
    }
};
} // namespace

TEST_CASE("Publication cost stops the next store without retrying the committed sample",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    Core::DeferredPublicationBudget budget;
    using Budget = Core::DeferredPublicationBudget;
    for (u32 i = 0; i < Budget::SampleCount; ++i) {
        REQUIRE(budget.BeforeAttempt(0x4000, i) == Budget::Decision::Allow);
        budget.Observe(0x4000, Budget::MeanLimitNs + 1, i);
    }
    CHECK(budget.BeforeAttempt(0x4000, Budget::SampleCount) == Budget::Decision::PublicationCost);
    CHECK(budget.BeforeAttempt(0x5000, Budget::SampleCount) == Budget::Decision::Allow);
}

TEST_CASE("Page publication cooldown expires without observing exact completions",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    using Budget = Core::DeferredPublicationBudget;
    Budget budget;
    for (u32 i = 0; i < Budget::SampleCount; ++i) {
        REQUIRE(budget.BeforeAttempt(0x4000, i) == Budget::Decision::Allow);
        budget.Observe(0x4000, Budget::MeanLimitNs + 1, i);
    }
    const u64 stopped_at = Budget::SampleCount - 1;
    // These rejected attempts can complete synchronously, without journal cost samples.
    CHECK(budget.BeforeAttempt(0x4000, stopped_at + 1) == Budget::Decision::PublicationCost);
    CHECK(budget.BeforeAttempt(0x4000, stopped_at + Budget::CooldownNs - 1) ==
          Budget::Decision::PublicationCost);
    CHECK(budget.BeforeAttempt(0x4000, stopped_at + Budget::CooldownNs) == Budget::Decision::Allow);
    for (u32 i = 0; i < Budget::SampleCount; ++i) {
        budget.Observe(0x4000, Budget::MeanLimitNs, stopped_at + Budget::CooldownNs + i);
    }
    CHECK(budget.BeforeAttempt(0x4000, stopped_at + Budget::CooldownNs + Budget::SampleCount) ==
          Budget::Decision::Allow);
    CHECK_FALSE(budget.TakeCoreEvent());
}

TEST_CASE("Page cooldown does not extend the core cooldown during exact completions",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    using Budget = Core::DeferredPublicationBudget;
    Budget budget;
    for (u32 i = 0; i < Budget::SampleCount; ++i) {
        REQUIRE(budget.BeforeAttempt(0x4000, i) == Budget::Decision::Allow);
        budget.Observe(0x4000, Budget::CoreMeanLimitNs + 1, i);
    }
    const auto activated = budget.TakeCoreEvent();
    REQUIRE(activated);
    CHECK(activated->activated);
    const u64 stopped_at = Budget::SampleCount - 1;
    CHECK(budget.BeforeAttempt(0x4000, stopped_at + Budget::CooldownNs - 1) ==
          Budget::Decision::CorePublicationCost);
    CHECK(budget.BeforeAttempt(0x4000, stopped_at + Budget::CooldownNs) == Budget::Decision::Allow);
    const auto recovered = budget.TakeCoreEvent();
    REQUIRE(recovered);
    CHECK_FALSE(recovered->activated);
    CHECK(budget.BeforeAttempt(0x5000, stopped_at + Budget::CooldownNs + 1) == Budget::Decision::Allow);
}

TEST_CASE("Cheap publications renew cost samples and fault windows independently",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    Core::DeferredPublicationBudget budget;
    using Budget = Core::DeferredPublicationBudget;
    for (u32 i = 0; i < Budget::SampleCount * 3; ++i) {
        REQUIRE(budget.BeforeAttempt(0x4000, i * Budget::WindowNs) == Budget::Decision::Allow);
        budget.Observe(0x4000, Budget::MeanLimitNs, i * Budget::WindowNs);
    }
    for (u32 i = 1; i < Budget::FaultLimit; ++i) {
        REQUIRE(budget.BeforeAttempt(0x4000, (Budget::SampleCount * 3 - 1) * Budget::WindowNs) ==
                Budget::Decision::Allow);
    }
    CHECK(budget.BeforeAttempt(0x4000, (Budget::SampleCount * 3 - 1) * Budget::WindowNs) ==
          Budget::Decision::FaultDensity);
    CHECK(budget.BeforeAttempt(0x4000, Budget::SampleCount * 3 * Budget::WindowNs) ==
          Budget::Decision::Allow);
}

TEST_CASE("Budget collisions forget only performance samples and zero observations are ignored",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    Core::DeferredPublicationBudget budget;
    using Budget = Core::DeferredPublicationBudget;
    constexpr PAddr first = 0x4000, collision = first + Budget::SlotCount * 4096;
    for (u32 i = 0; i < Budget::SampleCount * 2; ++i) {
        REQUIRE(budget.BeforeAttempt(first, 0) == Budget::Decision::Allow);
        budget.Observe(first, 0, 0);
    }
    REQUIRE(budget.BeforeAttempt(collision, 1) == Budget::Decision::Allow);
    budget.Observe(first, Budget::MeanLimitNs * Budget::SampleCount * 2, 1);
    CHECK(budget.BeforeAttempt(first, 2) == Budget::Decision::Allow);
}

TEST_CASE("Cost fallback drains accepted writes before the native retry overwrites them",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    ProducerFixture fixture;
    Core::DeferredPublicationBudget budget;
    using Budget = Core::DeferredPublicationBudget;
    const auto lease = fixture.Grant();
    REQUIRE(lease);
    REQUIRE(budget.BeforeAttempt(0x4000, 0) == Budget::Decision::Allow);
    REQUIRE(fixture.Publish(*lease, 7).status == PublishStatus::Published);
    for (u32 i = 0; i < Budget::SampleCount; ++i)
        budget.Observe(0x4000, Budget::MeanLimitNs + 1, 0);
    REQUIRE(budget.BeforeAttempt(0x4000, 1) == Budget::Decision::PublicationCost);
    CHECK(fixture.journal.CaptureCut() == 1);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto pin = fixture.journal.BeginWriterPermission(ranges);
    REQUIRE(pin);
    CHECK_FALSE(pin->MarkReady());
    REQUIRE(fixture.journal.DrainThrough(pin->Watermark(), [](const Write& write) {
        return write.bytes[0] == 7;
    }) == DrainResult::Complete);
    REQUIRE(pin->MarkReady());
    fixture.backing[3] = 9; // Only after Ready may the old writer commit.
    CHECK(fixture.Publish(*lease, 11).status == PublishStatus::LeaseUnavailable);
    CHECK(fixture.backing[3] == 9);
    CHECK_FALSE(fixture.Grant());
}

TEST_CASE("Core cost survives collisions and rehabilitates after cooldown",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    using Budget = Core::DeferredPublicationBudget;
    Budget budget;
    constexpr PAddr first = 0x4000, collision = first + Budget::SlotCount * 4096;
    for (u32 i = 0; i < Budget::SampleCount; ++i) {
        const auto page = i % 2 ? first : collision;
        REQUIRE(budget.BeforeAttempt(page, i) == Budget::Decision::Allow);
        budget.Observe(page, Budget::CoreMeanLimitNs + 1, i);
    }
    const auto activated = budget.TakeCoreEvent();
    REQUIRE(activated);
    CHECK(activated->activated);
    CHECK(activated->slow_samples == Budget::SampleCount);
    CHECK(activated->max_ns == Budget::CoreMeanLimitNs + 1);
    CHECK(budget.BeforeAttempt(first, 32) == Budget::Decision::CorePublicationCost);
    CHECK(budget.BeforeAttempt(0x9000, Budget::CooldownNs + 30) == Budget::Decision::CorePublicationCost);
    CHECK(budget.BeforeAttempt(0x9000, Budget::CooldownNs + 31) == Budget::Decision::Allow);
    const auto recovered = budget.TakeCoreEvent();
    REQUIRE(recovered);
    CHECK_FALSE(recovered->activated);
    for (u32 i = 0; i < Budget::SampleCount; ++i)
        budget.Observe(0x9000, Budget::MeanLimitNs, Budget::CooldownNs + 32 + i);
    CHECK(budget.BeforeAttempt(0xa000, Budget::CooldownNs + 100) == Budget::Decision::Allow);
    CHECK_FALSE(budget.TakeCoreEvent());
}

TEST_CASE("Page cost requires eight slow samples in a complete block",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    using Budget = Core::DeferredPublicationBudget;
    for (u32 slow_count : {7U, 8U}) {
        Budget budget;
        for (u32 i = 0; i < Budget::SampleCount; ++i) {
            REQUIRE(budget.BeforeAttempt(0x4000, i) == Budget::Decision::Allow);
            budget.Observe(0x4000, i < slow_count ? Budget::MeanLimitNs + 1 : Budget::MeanLimitNs, i);
        }
        CHECK(budget.BeforeAttempt(0x4000, 32) ==
              (slow_count == 8 ? Budget::Decision::PublicationCost : Budget::Decision::Allow));
        CHECK_FALSE(budget.TakeCoreEvent());
    }
}
TEST_CASE("An isolated spike leaves page and core admission open",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    using Budget = Core::DeferredPublicationBudget;
    Budget budget;
    for (u32 i = 0; i < Budget::SampleCount; ++i) {
        REQUIRE(budget.BeforeAttempt(0x4000, i) == Budget::Decision::Allow);
        budget.Observe(0x4000, i == 0 ? 12'000'000 : 1'000, i);
    }
    CHECK(budget.BeforeAttempt(0x4000, 32) == Budget::Decision::Allow);
    CHECK_FALSE(budget.TakeCoreEvent());
}

TEST_CASE("Incomplete slow blocks expire at the window boundary",
          "[deferred_writes][deferred_nce_publication][deferred_publication_budget]") {
    using Budget = Core::DeferredPublicationBudget;
    Budget budget;
    for (u32 i = 0; i < Budget::SlowSampleLimit; ++i) {
        REQUIRE(budget.BeforeAttempt(0x4000, i) == Budget::Decision::Allow);
        budget.Observe(0x4000, Budget::CoreMeanLimitNs + 1, i);
    }
    for (u32 i = 0; i < Budget::SampleCount; ++i) {
        REQUIRE(budget.BeforeAttempt(0x4000, Budget::WindowNs + i) == Budget::Decision::Allow);
        budget.Observe(0x4000, 1'000, Budget::WindowNs + i);
    }
    CHECK(budget.BeforeAttempt(0x4000, Budget::WindowNs + 32) == Budget::Decision::Allow);
    CHECK_FALSE(budget.TakeCoreEvent());
}
TEST_CASE("NCE publication retains exact bytes and old readback cannot undo its CPU commit",
          "[deferred_writes][deferred_nce_publication]") {
    ProducerFixture fixture;
    fixture.backing.fill(1);
    const auto lease = fixture.Grant();
    REQUIRE(lease);
    REQUIRE(fixture.journal.FindLease(0x4000, 0x8000, lease->mapping_generation));
    auto old = fixture.journal.BeginReadback();
    REQUIRE(fixture.Publish(*lease, 7).status == PublishStatus::Published);
    CHECK(fixture.backing[2] == 1);
    CHECK(fixture.backing[3] == 7);
    CHECK(fixture.backing[6] == 7);
    CHECK(fixture.backing[7] == 1);
    fixture.journal.WriteReadback(*old, 0x4000, 16, [&](u64 offset, u64 size) {
        std::memset(fixture.backing.data() + offset, 9, size);
    });
    CHECK(fixture.backing[2] == 9);
    CHECK(fixture.backing[3] == 7);
    CHECK(fixture.backing[7] == 9);
    REQUIRE(fixture.Publish(*lease, 11, 1).status == PublishStatus::Published);
    std::vector<u8> applied;
    CHECK(fixture.journal.DrainThrough(2, [&](const Write& write) {
        applied.push_back(write.bytes[0]);
        return true;
    }) == DrainResult::Complete);
    CHECK(applied == std::vector<u8>{7, 11});
}

TEST_CASE("NCE full suspended and closing admissions do not touch backing",
          "[deferred_writes][deferred_nce_publication]") {
    ProducerFixture fixture;
    const auto lease = fixture.Grant();
    REQUIRE(lease);
    REQUIRE(fixture.Publish(*lease, 1).status == PublishStatus::Published);
    REQUIRE(fixture.Publish(*lease, 2).status == PublishStatus::Published);
    CHECK(fixture.Publish(*lease, 3).status == PublishStatus::Full);
    CHECK(fixture.backing[3] == 2);
    auto suspension = fixture.journal.BeginSuspension();
    CHECK_FALSE(fixture.journal.FindLease(0x4000, 0x8000, lease->mapping_generation));
    CHECK(fixture.Publish(*lease, 4, 1).status == PublishStatus::Suspended);
    CHECK(fixture.backing[3] == 2);
    REQUIRE(suspension.TryPrepare([&](Cut cut) {
        return fixture.journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete;
    }) == SuspensionResult::Ready);
    suspension.Reset();
    CHECK(fixture.Publish(*lease, 5, 1).status == PublishStatus::LeaseUnavailable);
    CHECK(fixture.backing[3] == 2);
}

TEST_CASE("NCE alias or captured writer rejects the same backing despite a cached lease",
          "[deferred_writes][deferred_nce_publication]") {
    ProducerFixture fixture;
    const auto lease = fixture.Grant();
    REQUIRE(lease);
    SECTION("A second CPU mapping") {
        fixture.mappings.WithMapping(0x20000, 1, 0x4000,
            [](VAddr) { return std::optional<PAddr>{}; }, [] {});
    }
    SECTION("A captured writer") {
        fixture.mappings.ExcludePhysical(0x4000, 4096, Core::DeferredCpuMappings::Status::CapturedWriter);
    }
    CHECK(fixture.Publish(*lease, 7).status == PublishStatus::Rejected);
    CHECK(fixture.backing[3] == 0);
    CHECK(fixture.journal.CaptureCut() == 0);
}

TEST_CASE("NCE lease hints cannot authorize publication after a writer or closure intervenes",
          "[deferred_writes][deferred_nce_publication]") {
    ProducerFixture fixture;
    const auto lease = fixture.Grant();
    REQUIRE(lease);
    SECTION("Writer starts without any pending records") {
        const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
        auto permission = fixture.journal.BeginWriterPermission(ranges);
        REQUIRE(permission);
        CHECK_FALSE(fixture.journal.FindLease(0x4000, 0x8000, lease->mapping_generation));
        CHECK(fixture.Publish(*lease, 7).status == PublishStatus::LeaseUnavailable);
    }
    SECTION("Consumer closes the lease") {
        const auto closure = fixture.journal.CloseLease(*lease);
        REQUIRE(closure);
        CHECK_FALSE(fixture.journal.FindLease(0x4000, 0x8000, lease->mapping_generation));
        CHECK(fixture.Publish(*lease, 7).status == PublishStatus::LeaseUnavailable);
        REQUIRE(fixture.journal.FinishClosure(*closure));
    }
    CHECK(fixture.backing[3] == 0);
}

TEST_CASE("NCE mapping generations reject ABA without penalizing unrelated pages",
          "[deferred_writes][deferred_nce_publication]") {
    ProducerFixture fixture;
    const auto lease = fixture.Grant();
    REQUIRE(lease);
    fixture.mappings.WithMapping(0x20000, 1, 0x7000,
        [](VAddr) { return std::optional<PAddr>{}; }, [] {});
    REQUIRE(fixture.Publish(*lease, 3).status == PublishStatus::Published);
    fixture.mappings.WithMapping(0x10000, 1, 0x5000,
        [](VAddr) { return std::optional<PAddr>{0x4000}; }, [] {});
    fixture.mappings.WithMapping(0x10000, 1, 0x4000,
        [](VAddr) { return std::optional<PAddr>{0x5000}; }, [] {});
    CHECK(fixture.Publish(*lease, 7).status == PublishStatus::LeaseUnavailable);
    CHECK(fixture.backing[3] == 3);
}

TEST_CASE("A Ready exact store preserves future admission without waiting for the transaction owner",
          "[deferred_writes][deferred_nce_publication]") {
    ProducerFixture fixture;
    const auto lease = fixture.Grant();
    REQUIRE(lease);
    REQUIRE(fixture.Publish(*lease, 7).status == PublishStatus::Published);
    auto transaction = fixture.journal.BeginSuspension();
    auto exact = transaction.Fork();
    REQUIRE(exact.TryPrepare([&](Cut cut) {
        return fixture.journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete;
    }) == SuspensionResult::Ready);
    REQUIRE(transaction.IsReady());
    const std::array<u8, 4> payload{9, 9, 9, 9};
    bool committed{};
    CHECK(fixture.mappings.WithPage(&fixture.table, 0x10003, 0x4000, [&](const auto&) {
        committed = Core::ExactStoreBacking::Commit(0x8003, fixture.backing.data() + 3,
                                                    payload, {4, false, false});
    }) == Core::DeferredCpuMappings::Status::Eligible);
    REQUIRE(committed);
    CHECK(fixture.backing[3] == 9);
    CHECK(fixture.journal.CaptureCut() == 1); // Exact legacy commit, not another B publication.
    exact.Reset();
    CHECK(fixture.journal.GetSuspensionState() == SuspensionState::SuspendedReady);
    transaction.Reset();
    const auto new_lease = fixture.Grant();
    REQUIRE(new_lease); // No permanent native-fallback writer was installed.
    REQUIRE(fixture.Publish(*new_lease, 11).status == PublishStatus::Published);
}

TEST_CASE("A remap during a Ready exact-store queue rejects the captured physical identity",
          "[deferred_writes][deferred_nce_publication]") {
    ProducerFixture fixture;
    auto transaction = fixture.journal.BeginSuspension();
    REQUIRE(transaction.TryPrepare([](Cut) { return true; }) == SuspensionResult::Ready);
    auto exact = transaction.Fork();
    fixture.mappings.WithMapping(0x10000, 1, 0x5000,
        [](VAddr) { return std::optional<PAddr>{0x4000}; }, [] {});
    bool called{};
    CHECK(fixture.mappings.WithPage(&fixture.table, 0x10003, 0x4000,
        [&](const auto&) { called = true; }) == Core::DeferredCpuMappings::Status::MappingChanged);
    CHECK_FALSE(called);
    CHECK(fixture.backing[3] == 0);
    CHECK(fixture.journal.CaptureCut() == 0);
}

TEST_CASE("Retention sampling distinguishes fixed capacity history and live exclusions",
          "[deferred_writes][deferred_nce_publication]") {
    ProducerFixture fixture;
    const auto initial = fixture.journal.GetRetentionSnapshot();
    REQUIRE(initial.fixed_bytes != 0);
    const auto lease = fixture.Grant();
    REQUIRE(lease);
    auto readback = fixture.journal.BeginReadback();
    REQUIRE(fixture.Publish(*lease, 3).status == PublishStatus::Published);
    REQUIRE(fixture.journal.DrainThrough(1, [](const Write&) { return true; }) == DrainResult::Complete);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto pin = fixture.journal.BeginWriterPermission(ranges);
    REQUIRE(pin->MarkReady());
    const auto retained = fixture.journal.GetRetentionSnapshot();
    CHECK(retained.fixed_bytes == initial.fixed_bytes);
    CHECK(retained.live_records == 1);
    CHECK(retained.pending_records == 0);
    CHECK(retained.leases == 0);
    CHECK(retained.writer_permissions == 1);
    CHECK(retained.readbacks == 1);
    readback.reset();
    pin.reset();
    REQUIRE(fixture.journal.RetireThrough(1));
    const auto cleared = fixture.journal.GetRetentionSnapshot();
    CHECK(cleared.live_records == 0);
    CHECK(cleared.writer_permissions == 0);
    CHECK(cleared.readbacks == 0);
}
