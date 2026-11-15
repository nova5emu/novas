// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/deferred_write_journal.h"
#include "video_core/deferred_consumer.h"
#include "core/hle/kernel/svc_deferred_transition.h"

namespace {
using namespace VideoCommon::DeferredWrites;

Write ByteWrite(u8 value) {
    Write write;
    write.physical_address = 0x4000;
    write.device_address = 0x8000;
    write.mapping_generation = 1;
    write.lease_generation = 1;
    write.size = 1;
    write.bytes[0] = value;
    return write;
}

Publication Publish(Journal& journal, std::size_t core, u8 value) {
    return journal.TryPublish(core, ByteWrite(value), []() noexcept { return true; });
}
} // namespace

TEST_CASE("Legacy writer excludes grants even when the page initially has no lease",
          "[deferred_writes][deferred_writers]") {
    auto journal = std::make_shared<Journal>(1, 4);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4008, 4}};
    auto writer = BeginSharedWriterPermission(journal, ranges);
    REQUIRE(writer);
    REQUIRE(writer->MarkReady());
    CHECK(writer->Watermark() == 0);
    CHECK_FALSE(journal->GrantLease(0x4000, 0x8000, 1));
    CHECK_FALSE(journal->GrantLease(0x4000, 0x18000, 2));
    CHECK(Publish(*journal, 0, 9).status == PublishStatus::LeaseUnavailable);
    REQUIRE(journal->GrantLease(0x5000, 0x9000, 1));
    auto callback = writer;
    writer.reset();
    CHECK_FALSE(journal->GrantLease(0x4000, 0x8000, 1));
    callback.reset();
    CHECK(journal->GrantLease(0x4000, 0x8000, 1));
}

TEST_CASE("Legacy fallback applies the affected prefix before commit and revokes old tokens",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    auto write = ByteWrite(7);
    write.lease_generation = lease->generation;
    u8 backing{};
    u8 gpu{};
    REQUIRE(journal.TryPublishLeased(0, write, [&]() noexcept {
        backing = 7;
        return true;
    }).sequence == 1);
    const std::array ranges{TransitionRange{TransitionSpace::Device, 0x8000, 4}};
    auto writer = journal.BeginWriterPermission(ranges);
    REQUIRE(writer);
    REQUIRE(writer->Watermark() == 1);
    CHECK_FALSE(writer->MarkReady());
    CHECK(journal.TryPublishLeased(0, write, []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
    REQUIRE(journal.DrainThrough(writer->Watermark(), [&](const Write& entry) {
        gpu = entry.bytes[0];
        return true;
    }) == DrainResult::Complete);
    REQUIRE(writer->MarkReady());
    CHECK(gpu == 7);
    backing = 9;
    writer.reset();
    const auto next = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(next);
    CHECK(next->generation != lease->generation);
    CHECK(journal.TryPublishLeased(0, write, []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
    unsigned reapplied{};
    REQUIRE(journal.DrainThrough(1, [&](const Write&) {
        ++reapplied;
        return true;
    }) == DrainResult::Complete);
    CHECK(reapplied == 0);
    CHECK(backing == 9);
}

TEST_CASE("Unrelated pending writes do not force a fresh writer to drain",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    auto write = ByteWrite(7);
    write.lease_generation = lease->generation;
    REQUIRE(journal.TryPublishLeased(0, write, []() noexcept { return true; }).sequence == 1);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x5000, 4096}};
    auto writer = journal.BeginWriterPermission(ranges);
    REQUIRE(writer);
    CHECK(writer->Watermark() == 0);
    CHECK(writer->MarkReady());
    CHECK(journal.AppliedThrough() == 0);
}

TEST_CASE("Cancelling an unprepared writer preserves the original lease and pending record",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    auto write = ByteWrite(7);
    write.lease_generation = lease->generation;
    REQUIRE(journal.TryPublishLeased(0, write, []() noexcept { return true; }).sequence == 1);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    {
        auto writer = journal.BeginWriterPermission(ranges);
        REQUIRE(writer);
        CHECK_FALSE(writer->MarkReady());
    }
    CHECK(journal.TryPublishLeased(0, write, []() noexcept { return true; }).sequence == 2);
}

TEST_CASE("Overlapping writer exclusions keep a cached lease blocked until the last cancellation",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    const auto hint = journal.FindLease(0x4000, 0x8000, 1);
    REQUIRE(hint);
    auto write = ByteWrite(9);
    write.lease_generation = hint->generation;
    const std::array physical{TransitionRange{TransitionSpace::Physical, 0x4fff, 1}};
    const std::array device{TransitionRange{TransitionSpace::Device, 0x8008, 4}};
    auto first = journal.BeginWriterPermission(physical);
    auto second = journal.BeginWriterPermission(device);
    REQUIRE(first);
    REQUIRE(second);
    unsigned commits{};
    const auto commit = [&]() noexcept { ++commits; return true; };
    CHECK_FALSE(journal.FindLease(0x4000, 0x8000, 1));
    CHECK(journal.TryPublishLeased(0, write, commit).status == PublishStatus::LeaseUnavailable);
    first.reset();
    CHECK_FALSE(journal.FindLease(0x4000, 0x8000, 1));
    CHECK(journal.TryPublishLeased(0, write, commit).status == PublishStatus::LeaseUnavailable);
    CHECK(commits == 0);
    second.reset();
    REQUIRE(journal.FindLease(0x4000, 0x8000, 1));
    CHECK(journal.TryPublishLeased(0, write, commit).status == PublishStatus::Published);
    CHECK(commits == 1);
}

TEST_CASE("Lease diagnostics distinguish lookup rejection and stale commit validation",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    LeaseLookupStatus status{};
    CHECK_FALSE(journal.FindLease(0x4000, 0x8000, 1, &status));
    CHECK(status == LeaseLookupStatus::Missing);
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    REQUIRE(journal.FindLease(0x4000, 0x8000, 1, &status));
    CHECK(status == LeaseLookupStatus::Found);
    CHECK_FALSE(journal.FindLease(0x4000, 0x9000, 1, &status));
    CHECK(status == LeaseLookupStatus::DeviceMismatch);
    CHECK_FALSE(journal.FindLease(0x4000, 0x8000, 2, &status));
    CHECK(status == LeaseLookupStatus::MappingMismatch);
    auto write = ByteWrite(7);
    write.lease_generation = lease->generation + 1;
    unsigned commits{};
    const auto commit = [&]() noexcept { ++commits; return true; };
    CHECK(journal.TryPublishLeased(0, write, commit, &status).status == PublishStatus::LeaseUnavailable);
    CHECK(status == LeaseLookupStatus::GenerationMismatch);
    write.lease_generation = lease->generation;
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto writer = journal.BeginWriterPermission(ranges);
    REQUIRE(writer);
    CHECK_FALSE(journal.FindLease(0x4000, 0x8000, 1, &status));
    CHECK(status == LeaseLookupStatus::WriterExcluded);
    CHECK(journal.TryPublishLeased(0, write, commit, &status).status == PublishStatus::LeaseUnavailable);
    CHECK(status == LeaseLookupStatus::WriterExcluded);
    writer.reset();
    const auto closure = journal.CloseLease(*lease);
    REQUIRE(closure);
    CHECK_FALSE(journal.FindLease(0x4000, 0x8000, 1, &status));
    CHECK(status == LeaseLookupStatus::Closing);
    CHECK(journal.TryPublishLeased(0, write, commit, &status).status == PublishStatus::LeaseUnavailable);
    CHECK(status == LeaseLookupStatus::Closing);
    CHECK(commits == 0);
}

TEST_CASE("Host grant precheck distinguishes persistent writers and preserves ordered fallback",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    auto write = ByteWrite(7);
    write.lease_generation = lease->generation;
    REQUIRE(journal.TryPublishLeased(0, write, []() noexcept { return true; }).status ==
            PublishStatus::Published);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto temporary = journal.BeginWriterPermission(ranges);
    REQUIRE(temporary);
    CHECK_FALSE(journal.HasPersistentWriterExclusion(0x4000, 0x8000));
    auto persistent = journal.BeginWriterPermission(ranges, WriterOrigin::NcePersistent);
    REQUIRE(persistent);
    CHECK(journal.HasPersistentWriterExclusion(0x4000, 0x18000));
    CHECK_FALSE(journal.HasPersistentWriterExclusion(0x5000, 0x8000));
    CHECK_FALSE(persistent->MarkReady());
    unsigned commits{};
    CHECK(journal.TryPublishLeased(0, write, [&]() noexcept {
        ++commits;
        return true;
    }).status == PublishStatus::LeaseUnavailable);
    CHECK(commits == 0);
    REQUIRE(journal.DrainThrough(persistent->Watermark(), [](const Write& pending) {
        return pending.bytes[0] == 7;
    }) == DrainResult::Complete);
    REQUIRE(persistent->MarkReady());
    CHECK(journal.HasPersistentWriterExclusion(0x4000, 0x8000));
    persistent.reset();
    CHECK_FALSE(journal.HasPersistentWriterExclusion(0x4000, 0x8000));
    CHECK_FALSE(journal.GrantLease(0x4000, 0x8000, 1));
    temporary.reset();
    REQUIRE(journal.GrantLease(0x4000, 0x8000, 1));
}

TEST_CASE("Host grant precheck recognizes persistent device ranges without matching physical identity",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const std::array ranges{TransitionRange{TransitionSpace::Device, 0x8008, 4}};
    auto persistent = journal.BeginWriterPermission(ranges, WriterOrigin::LegacyPersistent);
    REQUIRE(persistent);
    CHECK(journal.HasPersistentWriterExclusion(0x4000, 0x8000));
    CHECK_FALSE(journal.HasPersistentWriterExclusion(0x4000, 0x9000));
    persistent.reset();
    CHECK_FALSE(journal.HasPersistentWriterExclusion(0x4000, 0x8000));
}

TEST_CASE("Writer revocation and transition retirement do not transfer exclusion counts to new leases",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto first = journal.BeginWriterPermission(ranges);
    auto second = journal.BeginWriterPermission(ranges);
    REQUIRE(first);
    REQUIRE(second);
    SECTION("Ready revokes the lease while another permission still references the page") {
        REQUIRE(first->MarkReady());
    }
    SECTION("A transition retires the closed lease while both permissions remain alive") {
        const auto closure = journal.CloseLease(*lease);
        REQUIRE(closure);
        REQUIRE(first->MarkReady());
        REQUIRE(journal.FinishClosure(*closure));
    }
    CHECK_FALSE(journal.GrantLease(0x4000, 0x18000, 2));
    first.reset();
    CHECK_FALSE(journal.GrantLease(0x4000, 0x18000, 2));
    second.reset();
    const auto next = journal.GrantLease(0x4000, 0x18000, 2);
    REQUIRE(next);
    REQUIRE(journal.FindLease(0x4000, 0x18000, 2));
    auto write = ByteWrite(9);
    write.device_address = next->device_page;
    write.mapping_generation = next->mapping_generation;
    write.lease_generation = next->generation;
    CHECK(journal.TryPublishLeased(0, write, []() noexcept { return true; }).status ==
          PublishStatus::Published);
}

TEST_CASE("Many unrelated persistent exclusions do not reject an existing lease",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    std::vector<std::unique_ptr<Journal::WriterPermission>> pins;
    for (u64 page = 0; page < 512; ++page) {
        const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x10000 + page * 4096, 4096}};
        pins.push_back(journal.BeginWriterPermission(ranges));
        REQUIRE(pins.back());
        REQUIRE(pins.back()->MarkReady());
    }
    REQUIRE(journal.FindLease(0x4000, 0x8000, 1));
    auto write = ByteWrite(9);
    write.lease_generation = lease->generation;
    CHECK(journal.TryPublishLeased(0, write, []() noexcept { return true; }).status == PublishStatus::Published);
    CHECK_FALSE(journal.GrantLease(0x10000, 0x20000, 1));
    pins.clear();
    REQUIRE(journal.GrantLease(0x10000, 0x20000, 1));
}

TEST_CASE("Writer reservation includes a CPU commit already in progress",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    auto write = ByteWrite(7);
    write.lease_generation = lease->generation;
    std::promise<void> entered;
    std::promise<void> release;
    auto released = release.get_future();
    Publication result{PublishStatus::Rejected};
    std::jthread producer{[&] {
        result = journal.TryPublishLeased(0, write, [&]() noexcept {
            entered.set_value();
            released.wait();
            return true;
        });
    }};
    entered.get_future().wait();
    std::promise<void> attempting;
    std::unique_ptr<Journal::WriterPermission> writer;
    std::jthread legacy{[&] {
        const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
        attempting.set_value();
        writer = journal.BeginWriterPermission(ranges);
    }};
    attempting.get_future().wait();
    release.set_value();
    producer.join();
    legacy.join();
    REQUIRE(result.status == PublishStatus::Published);
    REQUIRE(writer);
    CHECK(writer->Watermark() == result.sequence);
    CHECK_FALSE(writer->MarkReady());
    CHECK_FALSE(journal.FindLease(0x4000, 0x8000, 1));
    CHECK(journal.TryPublishLeased(0, write, []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
}

TEST_CASE("Full history does not require enqueuing the legacy fallback",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 1};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    auto readback = journal.BeginReadback();
    auto write = ByteWrite(7);
    write.lease_generation = lease->generation;
    u8 backing{};
    REQUIRE(journal.TryPublishLeased(0, write, [&]() noexcept {
        backing = 7;
        return true;
    }).sequence == 1);
    CHECK(journal.TryPublishLeased(0, write, []() noexcept { return true; }).status == PublishStatus::Full);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto writer = journal.BeginWriterPermission(ranges);
    REQUIRE(writer);
    REQUIRE(journal.DrainThrough(writer->Watermark(), [](const Write&) { return true; }) == DrainResult::Complete);
    REQUIRE(writer->MarkReady());
    backing = 9;
    // An old readback still needs the original B history after the permission revokes its lease.
    journal.WriteReadback(*readback, 0x4000, 1, [&](u64, u64) { backing = 0; });
    CHECK(backing == 9);
    CHECK(journal.ReclaimableThrough() == 0);
    readback.reset();
    CHECK(journal.ReclaimableThrough() == 1);
}

TEST_CASE("Durable exclusions survive global suspension and keep the journal alive",
          "[deferred_writes][deferred_writers]") {
    auto journal = std::make_shared<Journal>(1, 2);
    std::weak_ptr<Journal> lifetime = journal;
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto pin = BeginSharedWriterPermission(journal, ranges);
    REQUIRE(pin);
    REQUIRE(pin->MarkReady());
    {
        auto suspension = BeginSharedSuspension(journal);
        bool prepared{};
        REQUIRE(suspension->TryPrepareIdle(prepared) == SuspensionResult::Ready);
    }
    CHECK_FALSE(journal->GrantLease(0x4000, 0x8000, 1));
    CHECK(journal->GrantLease(0x5000, 0x9000, 1));
    journal.reset();
    CHECK_FALSE(lifetime.expired());
    pin.reset();
    CHECK(lifetime.expired());
}

TEST_CASE("Writer readiness does not steal a concurrent transition closure",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 2};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    const auto closure = journal.CloseLease(*lease);
    REQUIRE(closure);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto writer = journal.BeginWriterPermission(ranges);
    REQUIRE(writer);
    REQUIRE(writer->MarkReady());
    CHECK(journal.ClosureReady(*closure));
    CHECK(journal.FinishClosure(*closure));
    CHECK_FALSE(journal.GrantLease(0x4000, 0x8000, 1));
    writer.reset();
    CHECK(journal.GrantLease(0x4000, 0x8000, 1));
}

TEST_CASE("Producer recovery adopts an abandoned epoch without opening admission early",
          "[deferred_writes][deferred_suspension][deferred_recovery]") {
    auto journal = std::make_shared<Journal>(1, 4);
    REQUIRE_FALSE(AdoptSharedAbandonedSuspension(journal));
    REQUIRE(Publish(*journal, 0, 7).sequence == 1);
    auto readback = journal->BeginReadback();
    auto original = BeginSharedSuspension(journal);
    const auto epoch = original->Epoch();
    CHECK_FALSE(AdoptSharedAbandonedSuspension(journal));
    original.reset();
    auto adopted = AdoptSharedAbandonedSuspension(journal);
    REQUIRE(adopted);
    CHECK(adopted->Epoch() == epoch);
    CHECK(adopted->Watermark() == 1);
    CHECK_FALSE(adopted->CreatedEpoch());
    CHECK_FALSE(AdoptSharedAbandonedSuspension(journal));
    CHECK(Publish(*journal, 0, 9).status == PublishStatus::Suspended);
    bool prepared{};
    CHECK(adopted->TryPrepareIdle(prepared) == SuspensionResult::DrainPending);
    REQUIRE(adopted->TryPrepare([&](Cut cut) {
        if (journal->DrainThrough(cut, [](const Write&) { return true; }) != DrainResult::Complete)
            return false;
        readback.reset();
        return true;
    }) == SuspensionResult::Ready);
    CHECK(Publish(*journal, 0, 9).status == PublishStatus::Suspended);
    adopted.reset();
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
    CHECK(Publish(*journal, 0, 9).sequence == 2);
}

TEST_CASE("Only one concurrent producer adopts the abandoned suspension",
          "[deferred_writes][deferred_suspension][deferred_recovery]") {
    auto journal = std::make_shared<Journal>(2, 4);
    { auto original = BeginSharedSuspension(journal); }
    std::shared_ptr<Suspension> first, second;
    std::atomic<bool> start{};
    std::jthread a{[&] {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        first = AdoptSharedAbandonedSuspension(journal);
    }};
    std::jthread b{[&] {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        second = AdoptSharedAbandonedSuspension(journal);
    }};
    start.store(true, std::memory_order_release);
    a.join();
    b.join();
    REQUIRE(static_cast<bool>(first) != static_cast<bool>(second));
    auto& adopted = first ? first : second;
    auto participant = BeginSharedSuspension(journal);
    CHECK(participant->Epoch() == adopted->Epoch());
    bool prepared{};
    REQUIRE(adopted->TryPrepareIdle(prepared) == SuspensionResult::Ready);
    adopted.reset();
    CHECK(journal->GetSuspensionState() == SuspensionState::SuspendedReady);
    participant.reset();
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
    CHECK_FALSE(AdoptSharedAbandonedSuspension(journal));
}

TEST_CASE("Recovery does not create an epoch after another participant already resumed",
          "[deferred_writes][deferred_suspension][deferred_recovery]") {
    auto journal = std::make_shared<Journal>(1, 2);
    { auto original = BeginSharedSuspension(journal); }
    auto participant = BeginSharedSuspension(journal);
    const auto epoch = participant->Epoch();
    bool prepared{};
    REQUIRE(participant->TryPrepareIdle(prepared) == SuspensionResult::Ready);
    participant.reset();
    REQUIRE_FALSE(AdoptSharedAbandonedSuspension(journal));
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
    auto next = BeginSharedSuspension(journal);
    CHECK(next->Epoch() == epoch + 1);
    REQUIRE(next->TryPrepareIdle(prepared) == SuspensionResult::Ready);
}

TEST_CASE("Fresh sparse identity excludes grants and CPU commits until installation completes",
          "[deferred_writes][deferred_fresh_identity]") {
    Journal journal{2, 4};
    const std::array physical{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto identity = journal.TryReserveFreshIdentity(physical);
    REQUIRE(identity);
    CHECK_FALSE(journal.GrantLease(0x4000, 0xc000, 1)); // Includes aliases.
    bool committed{};
    CHECK(journal.TryPublish(0, ByteWrite(7), [&]() noexcept {
        committed = true;
        return true;
    }).status == PublishStatus::LeaseUnavailable);
    CHECK_FALSE(committed);
    const std::array device{TransitionRange{TransitionSpace::Device, 0x8000, 4096}};
    REQUIRE(identity->TryExtend(device));
    CHECK_FALSE(journal.GrantLease(0xa000, 0x8000, 1));
    CHECK_FALSE(journal.TryReserveFreshIdentity(device));
    CHECK(journal.GrantLease(0x10000, 0x20000, 1));
    identity.reset();
    CHECK(journal.GrantLease(0x4000, 0x8000, 2));
}

TEST_CASE("Fresh identity rejects accepted history even after application and without a lease",
          "[deferred_writes][deferred_fresh_identity]") {
    Journal journal{1, 4};
    const std::array physical{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto readback = journal.BeginReadback();
    REQUIRE(Publish(journal, 0, 7).status == PublishStatus::Published);
    CHECK_FALSE(journal.TryReserveFreshIdentity(physical));
    REQUIRE(journal.DrainThrough(1, [](const Write&) { return true; }) == DrainResult::Complete);
    CHECK_FALSE(journal.TryReserveFreshIdentity(physical));
    CHECK_FALSE(journal.RetireThrough(1));
    readback.reset();
    REQUIRE(journal.RetireThrough(1));
    CHECK(journal.TryReserveFreshIdentity(physical));
    REQUIRE(journal.GrantLease(0x4000, 0x8000, 1));
    CHECK_FALSE(journal.TryReserveFreshIdentity(physical));
}

TEST_CASE("Fresh identity may finish a readback during global draining without waiting",
          "[deferred_writes][deferred_fresh_identity][deferred_suspension]") {
    Journal journal{1, 4};
    auto readback = journal.BeginReadback();
    auto suspension = journal.BeginSuspension();
    bool prepared{};
    CHECK(suspension.TryPrepareIdle(prepared) == SuspensionResult::ReadbacksPending);
    const std::array physical{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto identity = journal.TryReserveFreshIdentity(physical);
    REQUIRE(identity);
    const std::array device{TransitionRange{TransitionSpace::Device, 0x8000, 4096}};
    REQUIRE(identity->TryExtend(device));
    // The caller separately proved these are new pinned pages and an unbacked device range.
    readback.reset();
    REQUIRE(suspension.TryPrepareIdle(prepared) == SuspensionResult::Ready);
    suspension.Reset();
    CHECK_FALSE(journal.GrantLease(0x4000, 0x8000, 1));
    identity.reset();
    CHECK(journal.GrantLease(0x4000, 0x8000, 1));
}

TEST_CASE("Fresh extension rejects an occupied target without partially reserving it",
          "[deferred_writes][deferred_fresh_identity]") {
    Journal journal{1, 4};
    const std::array physical{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    auto identity = journal.TryReserveFreshIdentity(physical);
    REQUIRE(identity);
    REQUIRE(journal.GrantLease(0x10000, 0x20000, 1));
    const std::array additions{TransitionRange{TransitionSpace::Device, 0x8000, 4096},
                               TransitionRange{TransitionSpace::Device, 0x20000, 4096}};
    CHECK_FALSE(identity->TryExtend(additions));
    CHECK(journal.GrantLease(0x30000, 0x8000, 1));
    CHECK_FALSE(journal.GrantLease(0x4000, 0x9000, 1));
    const std::array invalid{TransitionRange{TransitionSpace::Physical, ~u64{}, 2}};
    CHECK_FALSE(identity->TryExtend(invalid));
}

TEST_CASE("Fresh installation serializes with a concurrent lease grant",
          "[deferred_writes][deferred_fresh_identity]") {
    for (unsigned attempt = 0; attempt < 16; ++attempt) {
        Journal journal{1, 2};
        const std::array physical{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
        std::atomic<bool> start{};
        std::optional<Journal::Lease> lease;
        std::thread producer{[&] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            lease = journal.GrantLease(0x4000, 0x8000, 1);
        }};
        start.store(true, std::memory_order_release);
        auto identity = journal.TryReserveFreshIdentity(physical);
        producer.join();
        CHECK(static_cast<bool>(identity) != lease.has_value());
    }
}

TEST_CASE("Mapping callbacks retain a ready token through frontend and backend queues",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    auto journal = std::make_shared<Journal>(2, 4);
    REQUIRE(Publish(*journal, 0, 7).sequence == 1);
    auto readback = journal->BeginReadback();
    auto transition = BeginSharedSuspension(journal);
    REQUIRE(transition->TryPrepare([&](Cut cut) {
        const bool applied = journal->DrainThrough(cut, [](const Write&) { return true; }) ==
                             DrainResult::Complete;
        readback.reset();
        return applied;
    }) == SuspensionResult::Ready);
    unsigned mutations{};
    std::function<void()> frontend = RetainReadySuspension(transition, [&] {
        CHECK(journal->GetSuspensionState() == SuspensionState::SuspendedReady);
        CHECK(Publish(*journal, 1, 9).status == PublishStatus::Suspended);
        ++mutations;
    });
    transition.reset(); // The ioctl/IPC caller has returned.
    auto backend = frontend;
    frontend = {};
    CHECK(Publish(*journal, 1, 9).status == PublishStatus::Suspended);
    backend();
    CHECK(mutations == 1);
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
    backend = {};
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
    CHECK(Publish(*journal, 1, 9).sequence == 2);
}

TEST_CASE("Every mapping callback copy pins the epoch until it is released",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    auto journal = std::make_shared<Journal>(1, 2);
    auto transition = BeginSharedSuspension(journal);
    bool prepared{};
    REQUIRE(transition->TryPrepareIdle(prepared) == SuspensionResult::Ready);
    auto first = RetainReadySuspension(transition, [] {});
    auto second = first;
    transition.reset();
    first();
    first = {};
    CHECK(journal->GetSuspensionState() == SuspensionState::SuspendedReady);
    CHECK(Publish(*journal, 0, 1).status == PublishStatus::Suspended);
    second();
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
    second = {};
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
}

TEST_CASE("Cancelled or throwing mapping callbacks release tokens without pretending to mutate",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    auto journal = std::make_shared<Journal>(1, 2);
    auto transition = BeginSharedSuspension(journal);
    bool prepared{}, mutation{};
    REQUIRE(transition->TryPrepareIdle(prepared) == SuspensionResult::Ready);
    auto cancelled = RetainReadySuspension(transition, [&] { mutation = true; });
    auto failing = RetainReadySuspension(transition, [] { throw std::runtime_error("mapping failed"); });
    transition.reset();
    cancelled = {};
    CHECK_FALSE(mutation);
    CHECK_THROWS_AS(failing(), std::runtime_error);
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
    failing = {};
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
    unsigned legacy_calls{};
    auto legacy = RetainReadySuspension({}, [&] { ++legacy_calls; });
    legacy();
    CHECK(legacy_calls == 1);
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
}

TEST_CASE("Idle suspension preparation never authorizes unapplied writes or live readbacks",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    Journal journal{1, 2};
    auto readback = journal.BeginReadback();
    REQUIRE(Publish(journal, 0, 7).sequence == 1);
    auto token = journal.BeginSuspension();
    bool prepared{true};
    CHECK(token.TryPrepareIdle(prepared) == SuspensionResult::DrainPending);
    CHECK_FALSE(prepared);
    REQUIRE(journal.DrainThrough(1, [](const Write&) { return true; }) == DrainResult::Complete);
    CHECK(token.TryPrepareIdle(prepared) == SuspensionResult::ReadbacksPending);
    CHECK_FALSE(prepared);
    CHECK(journal.HasReadbacks());
    readback.reset();
    CHECK_FALSE(journal.HasReadbacks());
    CHECK(token.TryPrepareIdle(prepared) == SuspensionResult::Ready);
    CHECK(prepared);
    CHECK(token.TryPrepareIdle(prepared) == SuspensionResult::Ready);
    CHECK_FALSE(prepared);
    CHECK(Publish(journal, 0, 9).status == PublishStatus::Suspended);
    token.Reset();
    CHECK(Publish(journal, 0, 9).status == PublishStatus::Published);
}

TEST_CASE("Concurrent idle preparation attributes exactly one ready epoch",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    Journal journal{2, 2};
    auto first = journal.BeginSuspension(11);
    auto second = journal.BeginSuspension(22);
    REQUIRE(first.CreatedEpoch());
    REQUIRE_FALSE(second.CreatedEpoch());
    CHECK(first.Origin() == 11);
    CHECK(second.Origin() == 11);
    CHECK(first.StartedNs() == 0);
    CHECK(second.StartedNs() == 0);
    std::promise<void> start;
    auto ready = start.get_future().share();
    const auto prepare = [&](const Suspension& token) {
        ready.wait();
        bool prepared{};
        const auto result = token.TryPrepareIdle(prepared);
        return std::pair{result, prepared};
    };
    auto a = std::async(std::launch::async, [&] { return prepare(first); });
    auto b = std::async(std::launch::async, [&] { return prepare(second); });
    start.set_value();
    const auto x = a.get();
    const auto y = b.get();
    CHECK(x.first == SuspensionResult::Ready);
    CHECK(y.first == SuspensionResult::Ready);
    CHECK(x.second != y.second);
    auto fork = first.Fork();
    CHECK_FALSE(fork.CreatedEpoch());
    CHECK(fork.Origin() == 11);
    CHECK(Publish(journal, 0, 1).status == PublishStatus::Suspended);
}

TEST_CASE("Idle callers cannot steal an active preparation or change abandoned epoch attribution",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    Journal journal{1, 2};
    u64 epoch{};
    {
        auto abandoned = journal.BeginSuspension(31);
        epoch = abandoned.Epoch();
    }
    auto owner = journal.BeginSuspension(32);
    REQUIRE_FALSE(owner.CreatedEpoch());
    REQUIRE(owner.Epoch() == epoch);
    CHECK(owner.Origin() == 31);
    auto peer = owner.Fork();
    CHECK(owner.TryPrepare([&](Cut) {
        bool prepared{};
        CHECK(peer.TryPrepareIdle(prepared) == SuspensionResult::Busy);
        CHECK_FALSE(prepared);
        CHECK_FALSE(peer.IsReady());
        return true;
    }) == SuspensionResult::Ready);
    owner.Reset();
    peer.Reset();
    auto next = journal.BeginSuspension(33);
    CHECK(next.CreatedEpoch());
    CHECK(next.Epoch() > epoch);
    CHECK(next.Origin() == 33);
}

TEST_CASE("Deferred callbacks retain their ready token and journal past caller destruction",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    REQUIRE_FALSE(BeginSharedSuspension({}));
    auto journal = std::make_shared<Journal>(1, 2);
    std::weak_ptr<Journal> lifetime = journal;
    auto token = BeginSharedSuspension(journal, 4);
    bool prepared{};
    REQUIRE(token->TryPrepareIdle(prepared) == SuspensionResult::Ready);
    std::function<bool()> deferred = [token] { return token->IsReady(); };
    token.reset();
    journal.reset();
    CHECK_FALSE(lifetime.expired());
    CHECK(deferred());
    auto live = lifetime.lock();
    REQUIRE(live);
    CHECK(Publish(*live, 0, 7).status == PublishStatus::Suspended);
    live.reset();
    deferred = {};
    CHECK(lifetime.expired());
}

TEST_CASE("IPC releases suspension during waits and prepares again for deferred cleanup and errors",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    Journal journal{1, 2};
    bool prepared{};
    {
        auto user_buffer_lock = journal.BeginSuspension();
        REQUIRE(user_buffer_lock.TryPrepareIdle(prepared) == SuspensionResult::Ready);
        CHECK(Publish(journal, 0, 1).status == PublishStatus::Suspended);
    }
    // The client may remain blocked indefinitely; this interval must allow other cores to publish.
    CHECK(Publish(journal, 0, 2).sequence == 1);
    auto readback = journal.BeginReadback();
    unsigned drains{};
    try {
        auto cleanup = journal.BeginSuspension();
        REQUIRE(cleanup.TryPrepare([&](Cut cut) {
            ++drains;
            const bool applied = journal.DrainThrough(cut, [](const Write&) { return true; }) ==
                                 DrainResult::Complete;
            readback.reset();
            return applied;
        }) == SuspensionResult::Ready);
        std::mutex inherited_lock;
        std::scoped_lock lock{inherited_lock};
        auto last_reference_finalize = journal.BeginSuspension();
        CHECK(last_reference_finalize.TryPrepare([&](Cut) { ++drains; return false; }) ==
              SuspensionResult::Ready);
        CHECK(Publish(journal, 0, 3).status == PublishStatus::Suspended);
        throw std::runtime_error("IPC error after mapping mutation");
    } catch (const std::runtime_error&) {
    }
    CHECK(drains == 1);
    CHECK(journal.GetSuspensionState() == SuspensionState::Running);
    CHECK(Publish(journal, 0, 3).sequence == 2);
}

TEST_CASE("Nonreturning exits release before switching away and workers prepare their own prefix",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    Journal journal{1, 2};
    auto exiting = journal.BeginSuspension();
    bool prepared{};
    REQUIRE(exiting.TryPrepareIdle(prepared) == SuspensionResult::Ready);
    const auto epoch = exiting.Epoch();
    exiting.Reset(); // Explicitly before leaving a fiber whose stack will not unwind.
    REQUIRE(Publish(journal, 0, 9).sequence == 1);
    auto worker = journal.BeginSuspension();
    CHECK(worker.Epoch() > epoch);
    CHECK(worker.TryPrepareIdle(prepared) == SuspensionResult::DrainPending);
    CHECK(worker.TryPrepare([&](Cut cut) {
        return journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete;
    }) == SuspensionResult::Ready);
    CHECK(Publish(journal, 0, 0).status == PublishStatus::Suspended);
}

TEST_CASE("SVC transition policy covers mapping mutations and excludes blocking IPC and exits",
          "[deferred_writes][deferred_boundaries]") {
    using Kernel::Svc::SvcId;
    using Kernel::Svc::NeedsDeferredTransition;
    for (const auto id : {SvcId::SetHeapSize, SvcId::SetMemoryPermission, SvcId::SetMemoryAttribute,
                         SvcId::MapMemory, SvcId::UnmapMemory, SvcId::MapPhysicalMemory,
                         SvcId::UnmapPhysicalMemory, SvcId::MapSharedMemory, SvcId::UnmapSharedMemory,
                         SvcId::CreateTransferMemory, SvcId::MapTransferMemory, SvcId::UnmapTransferMemory,
                         SvcId::CreateCodeMemory, SvcId::ControlCodeMemory, SvcId::SetProcessMemoryPermission,
                         SvcId::MapProcessMemory, SvcId::UnmapProcessMemory, SvcId::MapProcessCodeMemory,
                         SvcId::UnmapProcessCodeMemory, SvcId::MapInsecureMemory, SvcId::UnmapInsecureMemory,
                         SvcId::CreateThread, SvcId::CloseHandle}) {
        INFO(static_cast<u32>(id));
        CHECK(NeedsDeferredTransition(static_cast<u32>(id)));
    }
    CHECK_FALSE(NeedsDeferredTransition(0x80));
    for (const auto id : {SvcId::SendSyncRequest, SvcId::SendSyncRequestLight,
                         SvcId::SendSyncRequestWithUserBuffer, SvcId::SendAsyncRequestWithUserBuffer,
                         SvcId::ReplyAndReceive, SvcId::ReplyAndReceiveWithUserBuffer,
                         SvcId::ReplyAndReceiveLight, SvcId::ExitThread, SvcId::ExitProcess,
                         SvcId::SleepThread, SvcId::WaitSynchronization, SvcId::SignalProcessWideKey}) {
        INFO(static_cast<u32>(id));
        CHECK_FALSE(NeedsDeferredTransition(static_cast<u32>(id)));
    }
}

TEST_CASE("HLE IPC keeps publication open and final references prepare only after the session lock",
          "[deferred_writes][deferred_boundaries]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    auto write = ByteWrite(7);
    write.lease_generation = lease->generation;
    bool session_locked{}, closed_under_lock{}, closed{}, prepared{};
    const auto finish_reference = [&] {
        closed_under_lock = session_locked;
        auto finalizer = journal.BeginSuspension();
        const auto result = finalizer.TryPrepare([&](Cut cut) {
            return journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete;
        });
        prepared = result == SuspensionResult::Ready;
        closed = true;
    };
    {
        SCOPE_EXIT { finish_reference(); };
        session_locked = true;
        SCOPE_EXIT { session_locked = false; };
        // Receive/reply itself does not close admission or revoke the lease.
        CHECK(journal.GetSuspensionState() == SuspensionState::Running);
        auto producer = std::async(std::launch::async, [&] {
            return journal.TryPublishLeased(0, write, []() noexcept { return true; });
        });
        CHECK(producer.get().status == PublishStatus::Published);
        CHECK_FALSE(closed);
    }
    CHECK(closed);
    CHECK(prepared);
    CHECK_FALSE(closed_under_lock);
    CHECK(journal.AppliedThrough() == 1);
    CHECK(journal.GetSuspensionState() == SuspensionState::Running);
}

TEST_CASE("Detached HLE async reply retains its request through preparation unlock and signal",
          "[deferred_writes][deferred_boundaries]") {
    Journal journal{1, 2};
    auto readback = journal.BeginReadback();
    REQUIRE(Publish(journal, 0, 7).sequence == 1);
    bool session_locked{}, request_alive{true}, user_buffer_locked{true};
    bool event_signaled{}, mutation_ready{}, prepared_without_lock{};
    std::vector<unsigned> order;
    {
        SCOPE_EXIT { order.push_back(4); request_alive = false; };
        SCOPE_EXIT {
            auto transition = journal.BeginSuspension();
            prepared_without_lock = !session_locked;
            mutation_ready = transition.TryPrepare([&](Cut cut) {
                order.push_back(1);
                const bool applied = journal.DrainThrough(cut, [](const Write&) { return true; }) ==
                                     DrainResult::Complete;
                readback.reset();
                return applied;
            }) == SuspensionResult::Ready;
            CHECK(request_alive);
            CHECK(user_buffer_locked);
            CHECK(mutation_ready);
            CHECK_FALSE(event_signaled);
            CHECK(Publish(journal, 0, 9).status == PublishStatus::Suspended);
            user_buffer_locked = false;
            order.push_back(2);
            event_signaled = true;
            order.push_back(3);
        };
        session_locked = true;
        SCOPE_EXIT { session_locked = false; order.push_back(0); };
        // Removing m_current_request transfers its reference; it must survive the later drain.
        CHECK(request_alive);
        CHECK_FALSE(event_signaled);
    }
    CHECK(prepared_without_lock);
    CHECK_FALSE(request_alive);
    CHECK_FALSE(user_buffer_locked);
    CHECK(event_signaled);
    CHECK(order == std::vector<unsigned>{0, 1, 2, 3, 4});
    CHECK(journal.GetSuspensionState() == SuspensionState::Running);
}

TEST_CASE("Signal owner references close outside the scheduler lock also on error",
          "[deferred_writes][deferred_boundaries]") {
    Journal journal{1, 2};
    REQUIRE(Publish(journal, 0, 1).sequence == 1);
    bool scheduler_locked{}, closed_under_lock{}, finalized{}, ready{};
    unsigned references = 2, closes{};
    try {
        std::vector<unsigned*> owners;
        SCOPE_EXIT {
            for (auto* owner : owners) {
                if (!owner) continue;
                ++closes;
                closed_under_lock |= scheduler_locked;
                if (--*owner == 0) {
                    auto token = journal.BeginSuspension();
                    ready = token.TryPrepare([&](Cut cut) {
                        return journal.DrainThrough(cut, [](const Write&) { return true; }) ==
                               DrainResult::Complete;
                    }) == SuspensionResult::Ready;
                    finalized = true;
                }
            }
        };
        scheduler_locked = true;
        SCOPE_EXIT { scheduler_locked = false; };
        owners.push_back(&references);
        owners.push_back(nullptr);
        owners.push_back(&references);
        throw std::runtime_error("signal after acquiring the second owner reference");
    } catch (const std::runtime_error&) {
    }
    CHECK(closes == 2);
    CHECK(references == 0);
    CHECK_FALSE(closed_under_lock);
    CHECK(finalized);
    CHECK(ready);
    CHECK(journal.AppliedThrough() == 1);
}

TEST_CASE("Deferred writes preserve same-byte order across cores and finite cuts",
          "[video_core][deferred_writes]") {
    Journal journal{2, 4};
    u8 cpu{};
    u8 gpu{};
    auto publish = [&](std::size_t core, u8 value) {
        return journal.TryPublish(core, ByteWrite(value), [&]() noexcept {
            cpu = value;
            return true;
        });
    };
    REQUIRE(publish(1, 7).sequence == 1);
    const Cut first_submit = journal.CaptureCut();
    REQUIRE(publish(0, 0).sequence == 2);
    const Cut second_submit = journal.CaptureCut();
    REQUIRE(publish(1, 9).sequence == 3);
    REQUIRE(cpu == 9);

    std::vector<Cut> seen;
    auto sink = [&](const Write& write) {
        seen.push_back(write.sequence);
        gpu = write.bytes[0];
        return true;
    };
    REQUIRE(journal.DrainThrough(first_submit, sink) == DrainResult::Complete);
    REQUIRE(gpu == 7);
    REQUIRE(journal.DrainThrough(second_submit, sink) == DrainResult::Complete);
    REQUIRE(gpu == 0);
    REQUIRE(journal.DrainThrough(journal.CaptureCut(), sink) == DrainResult::Complete);
    REQUIRE(gpu == cpu);
    REQUIRE(seen == std::vector<Cut>{1, 2, 3});
    REQUIRE(journal.DrainThrough(first_submit, sink) == DrainResult::Complete);
    REQUIRE(seen.size() == 3);
}

TEST_CASE("Deferred write consumer does not absorb publications made while applying its cut",
          "[video_core][deferred_writes]") {
    Journal journal{2, 4};
    REQUIRE(Publish(journal, 0, 1).status == PublishStatus::Published);
    const Cut cut = journal.CaptureCut();
    std::future<Publication> publication;
    std::future_status while_applying{std::future_status::timeout};
    unsigned calls{};
    const auto drained = journal.DrainThrough(cut, [&](const Write&) {
        if (++calls == 1) {
            publication = std::async(std::launch::async, [&] { return Publish(journal, 1, 2); });
            while_applying = publication.wait_for(std::chrono::seconds{2});
        }
        return true;
    });
    // Join outside the sink even if a regression holds the publication gate there.
    const auto later = publication.get();
    REQUIRE(drained == DrainResult::Complete);
    REQUIRE(while_applying == std::future_status::ready);
    REQUIRE(calls == 1);
    REQUIRE(later.status == PublishStatus::Published);
    REQUIRE(journal.AppliedThrough() == 1);
    REQUIRE(journal.CaptureCut() == 2);
}

TEST_CASE("Deferred write capacity is reclaimed only after history retirement",
          "[video_core][deferred_writes]") {
    Journal journal{1, 2};
    REQUIRE(Publish(journal, 0, 1).sequence == 1);
    REQUIRE(Publish(journal, 0, 2).sequence == 2);
    u8 cpu = 2;
    u64 pc = 0x100;
    const auto attempt = [&] {
        return journal.TryPublish(0, ByteWrite(3), [&]() noexcept {
            cpu = 3;
            pc += 4;
            return true;
        });
    };
    REQUIRE(attempt().status == PublishStatus::Full);
    REQUIRE(cpu == 2);
    REQUIRE(pc == 0x100);
    std::vector<u8> seen;
    auto sink = [&](const Write& write) {
        seen.push_back(write.bytes[0]);
        return true;
    };
    REQUIRE_FALSE(journal.RetireThrough(1));
    REQUIRE(journal.DrainThrough(2, sink) == DrainResult::Complete);
    REQUIRE(attempt().status == PublishStatus::Full);
    REQUIRE(cpu == 2);
    REQUIRE(pc == 0x100);
    REQUIRE(journal.RetireThrough(1));
    REQUIRE(attempt().sequence == 3);
    REQUIRE(pc == 0x104);
    REQUIRE(Publish(journal, 0, 4).status == PublishStatus::Full);
    REQUIRE(journal.DrainThrough(3, sink) == DrainResult::Complete);
    REQUIRE(journal.RetireThrough(3));
    REQUIRE(Publish(journal, 0, 4).sequence == 4);
    REQUIRE(journal.DrainThrough(4, sink) == DrainResult::Complete);
    REQUIRE(seen == std::vector<u8>{1, 2, 3, 4});
}

TEST_CASE("Deferred write rejection leaves no global sequence hole",
          "[video_core][deferred_writes]") {
    Journal journal{2, 2};
    REQUIRE(journal.TryPublish(0, ByteWrite(7), []() noexcept { return false; }).status ==
            PublishStatus::Rejected);
    REQUIRE(journal.CaptureCut() == 0);
    REQUIRE(Publish(journal, 1, 8).sequence == 1);
    REQUIRE(journal.DrainThrough(1, [](const Write& write) {
                return write.sequence == 1 && write.bytes[0] == 8;
            }) == DrainResult::Complete);
}

TEST_CASE("Invalid deferred write descriptors never invoke CPU commit",
          "[video_core][deferred_writes]") {
    Journal journal{1, 4};
    bool committed{};
    auto commit = [&]() noexcept {
        committed = true;
        return true;
    };
    REQUIRE(journal.TryPublish(1, ByteWrite(0), commit).status == PublishStatus::InvalidWrite);
    for (unsigned kind = 0; kind < 7; ++kind) {
        auto write = ByteWrite(0);
        switch (kind) {
        case 0:
            write.size = 0;
            break;
        case 1:
            write.size = 33;
            break;
        case 2:
            write.mapping_generation = 0;
            break;
        case 3:
            write.lease_generation = 0;
            break;
        case 4:
            write.physical_address = 0x4fff;
            write.size = 2;
            break;
        case 5:
            write.device_address = 0x8fff;
            write.size = 2;
            break;
        case 6:
            write.payload = Payload::Zero;
            write.size = 2049;
            break;
        }
        REQUIRE(journal.TryPublish(0, write, commit).status == PublishStatus::InvalidWrite);
    }
    REQUIRE_FALSE(committed);
    REQUIRE(journal.CaptureCut() == 0);
    auto zero = ByteWrite(0xff);
    zero.payload = Payload::Zero;
    zero.size = 64;
    REQUIRE(journal.TryPublish(0, zero, commit).status == PublishStatus::Published);
    REQUIRE(journal.DrainThrough(1, [](const Write& write) {
                return write.payload == Payload::Zero && write.size == 64;
            }) == DrainResult::Complete);
}

TEST_CASE("Deferred write sink failure preserves the unapplied record for retry",
          "[video_core][deferred_writes]") {
    Journal journal{2, 4};
    REQUIRE(Publish(journal, 0, 1).sequence == 1);
    REQUIRE(Publish(journal, 1, 2).sequence == 2);
    unsigned calls{};
    REQUIRE(journal.DrainThrough(3, [&](const Write&) {
                ++calls;
                return true;
            }) == DrainResult::InvalidCut);
    REQUIRE(calls == 0);
    REQUIRE(journal.DrainThrough(2, [](const Write& write) { return write.sequence == 1; }) ==
            DrainResult::ApplyFailed);
    REQUIRE(journal.AppliedThrough() == 1);
    REQUIRE_FALSE(journal.RetireThrough(2));
    REQUIRE(journal.DrainThrough(2, [&](const Write& write) {
                ++calls;
                return write.sequence == 2 && write.bytes[0] == 2;
            }) == DrainResult::Complete);
    REQUIRE(calls == 1);
}

TEST_CASE("Acquire cut includes a CPU commit observed before descriptor publication",
          "[video_core][deferred_writes]") {
    Journal journal{1, 4};
    std::atomic<u8> backing{};
    std::promise<void> visible;
    auto visible_future = visible.get_future();
    std::promise<void> release;
    auto release_future = release.get_future();
    Publication result{PublishStatus::Rejected};
    std::thread producer([&] {
        result = journal.TryPublish(0, ByteWrite(7), [&]() noexcept {
            backing.store(7, std::memory_order_release);
            visible.set_value();
            // Test-only pause models preemption after CPU commit, before publication.
            release_future.wait();
            return true;
        });
    });
    visible_future.wait();
    const u8 observed = backing.load(std::memory_order_acquire);
    std::promise<void> reader_started;
    auto started = reader_started.get_future();
    auto acquire = std::async(std::launch::async, [&] {
        reader_started.set_value();
        return journal.CaptureCut();
    });
    started.wait();
    const auto before_release = acquire.wait_for(std::chrono::milliseconds{20});
    release.set_value();
    producer.join();
    const Cut cut = acquire.get();
    REQUIRE(observed == 7);
    REQUIRE(before_release == std::future_status::timeout);
    REQUIRE(result.status == PublishStatus::Published);
    REQUIRE(cut == 1);
}

TEST_CASE("Concurrent cores serialize CPU commits in the same order as GPU application",
          "[video_core][deferred_writes]") {
    Journal journal{2, 128};
    std::atomic<u8> backing{};
    std::atomic<bool> failed{};
    auto produce = [&](std::size_t core) {
        for (unsigned i = 0; i < 100; ++i) {
            const u8 value = static_cast<u8>(core * 100 + i);
            const auto result = journal.TryPublish(core, ByteWrite(value), [&]() noexcept {
                backing.store(value, std::memory_order_relaxed);
                return true;
            });
            if (result.status != PublishStatus::Published) {
                failed.store(true);
            }
        }
    };
    std::thread first(produce, 0);
    std::thread second(produce, 1);
    first.join();
    second.join();
    REQUIRE_FALSE(failed.load());
    REQUIRE(journal.CaptureCut() == 200);
    Cut expected{};
    u8 gpu{};
    REQUIRE(journal.DrainThrough(200, [&](const Write& write) {
                gpu = write.bytes[0];
                return write.sequence == ++expected;
            }) == DrainResult::Complete);
    REQUIRE(expected == 200);
    REQUIRE(gpu == backing.load());
}

TEST_CASE("A waiting deferred write sink does not hold the CPU publication gate",
          "[video_core][deferred_writes]") {
    Journal journal{2, 4};
    REQUIRE(Publish(journal, 0, 1).sequence == 1);
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    std::promise<void> release;
    auto release_future = release.get_future();
    auto consumer = std::async(std::launch::async, [&] {
        return journal.DrainThrough(1, [&](const Write&) {
            entered.set_value();
            // Models a GPU sink stalled in Finish, with consumer serialization held.
            release_future.wait();
            return true;
        });
    });
    entered_future.wait();
    auto producer = std::async(std::launch::async, [&] { return Publish(journal, 1, 2); });
    const auto while_waiting = producer.wait_for(std::chrono::seconds{2});
    // Release even on failure so a regression cannot leave the test waiting on a future.
    release.set_value();
    const auto consumed = consumer.get();
    const auto published = producer.get();
    REQUIRE(while_waiting == std::future_status::ready);
    REQUIRE(published.status == PublishStatus::Published);
    REQUIRE(published.sequence == 2);
    REQUIRE(consumed == DrainResult::Complete);
    REQUIRE(journal.AppliedThrough() == 1);
}

TEST_CASE("A full core ring does not consume sequence numbers or block another core",
          "[video_core][deferred_writes]") {
    Journal journal{2, 1};
    REQUIRE(Publish(journal, 0, 1).sequence == 1);
    REQUIRE(Publish(journal, 0, 2).status == PublishStatus::Full);
    REQUIRE(Publish(journal, 1, 3).sequence == 2);
    std::vector<u8> seen;
    REQUIRE(journal.DrainThrough(2, [&](const Write& write) {
                seen.push_back(write.bytes[0]);
                return true;
            }) == DrainResult::Complete);
    REQUIRE(seen == std::vector<u8>{1, 3});
}

TEST_CASE("Readback watermark protects later CPU bytes even after GPU application",
          "[video_core][deferred_writes]") {
    Journal journal{2, 4};
    std::array<u8, 4> backing{};
    const std::array<u8, 4> staging{7, 7, 7, 7};
    auto readback = journal.BeginReadback();
    auto write = ByteWrite(0);
    write.physical_address += 1;
    // Deliberately different device alias, same physical byte. A diff would miss this store.
    write.device_address = 0xa001;
    REQUIRE(journal.TryPublish(1, write, [&]() noexcept {
                backing[1] = 0;
                return true;
            }).status == PublishStatus::Published);
    REQUIRE(journal.DrainThrough(1, [](const Write&) { return true; }) == DrainResult::Complete);
    REQUIRE_FALSE(journal.RetireThrough(1));
    journal.WriteReadback(*readback, 0x4000, staging.size(), [&](u64 offset, u64 size) {
        std::memcpy(backing.data() + offset, staging.data() + offset, size);
    });
    REQUIRE(backing == std::array<u8, 4>{7, 0, 7, 7});
    REQUIRE(journal.AppliedThrough() == 1);
    REQUIRE(journal.ReclaimableThrough() == 0);
    readback.reset();
    REQUIRE(journal.RetireThrough(1));
}

TEST_CASE("Readback uses applied prefix rather than latest CPU publication",
          "[video_core][deferred_writes]") {
    Journal journal{1, 4};
    REQUIRE(Publish(journal, 0, 3).sequence == 1);
    auto old = journal.BeginReadback();
    REQUIRE(old->Watermark() == 0);
    REQUIRE(journal.DrainThrough(1, [](const Write&) { return true; }) == DrainResult::Complete);
    auto current = journal.BeginReadback();
    REQUIRE(current->Watermark() == 1);
    u8 backing = 3;
    journal.WriteReadback(*old, 0x4000, 1, [&](u64, u64) { backing = 7; });
    REQUIRE(backing == 3);
    // A GPU write recorded after CPU patch 1 is newer and may legitimately replace it.
    journal.WriteReadback(*current, 0x4000, 1, [&](u64, u64) { backing = 9; });
    REQUIRE(backing == 9);
    current.reset();
    REQUIRE(journal.ReclaimableThrough() == 0);
    old.reset();
    REQUIRE(journal.ReclaimableThrough() == 1);
}

TEST_CASE("Live readback history saturates before a producer can touch backing",
          "[video_core][deferred_writes]") {
    Journal journal{1, 2};
    auto oldest = journal.BeginReadback();
    REQUIRE(Publish(journal, 0, 1).sequence == 1);
    REQUIRE(journal.DrainThrough(1, [](const Write&) { return true; }) == DrainResult::Complete);
    auto newer = journal.BeginReadback();
    REQUIRE(Publish(journal, 0, 2).sequence == 2);
    REQUIRE(journal.DrainThrough(2, [](const Write&) { return true; }) == DrainResult::Complete);
    bool committed{};
    const auto attempt = [&] {
        return journal.TryPublish(0, ByteWrite(3), [&]() noexcept {
            committed = true;
            return true;
        });
    };
    REQUIRE_FALSE(journal.RetireThrough(2));
    REQUIRE(attempt().status == PublishStatus::Full);
    REQUIRE_FALSE(committed);
    oldest.reset();
    REQUIRE(journal.ReclaimableThrough() == 1);
    REQUIRE(journal.RetireThrough(1));
    REQUIRE(attempt().sequence == 3);
    REQUIRE_FALSE(journal.RetireThrough(2));
    newer.reset();
    REQUIRE(journal.RetireThrough(2));
}

TEST_CASE("Readback masks overlapping cores and implicit zeros at page boundaries",
          "[video_core][deferred_writes]") {
    Journal journal{2, 4};
    auto token = journal.BeginReadback();
    auto zero = ByteWrite(0xff);
    zero.physical_address = 0x4fc0;
    zero.device_address = 0x8fc0;
    zero.payload = Payload::Zero;
    zero.size = 64;
    REQUIRE(journal.TryPublish(0, zero, []() noexcept { return true; }).sequence == 1);
    auto byte = ByteWrite(9);
    byte.physical_address = 0x4fff;
    REQUIRE(journal.TryPublish(1, byte, []() noexcept { return true; }).sequence == 2);
    std::vector<std::pair<u64, u64>> spans;
    spans.reserve(2);
    journal.WriteReadback(*token, 0x4f00, 256, [&](u64 offset, u64 size) {
        spans.emplace_back(offset, size);
    });
    REQUIRE(spans == std::vector<std::pair<u64, u64>>{{0, 192}});
    spans.clear();
    journal.WriteReadback(*token, 0x5000, 4, [&](u64 offset, u64 size) {
        spans.emplace_back(offset, size);
    });
    REQUIRE(spans == std::vector<std::pair<u64, u64>>{{0, 4}});
}

TEST_CASE("CPU commit cannot slip between readback history lookup and backing write",
          "[video_core][deferred_writes]") {
    Journal journal{1, 4};
    auto token = journal.BeginReadback();
    std::atomic<u8> backing{};
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    std::promise<void> release;
    auto release_future = release.get_future();
    auto reader = std::async(std::launch::async, [&] {
        journal.WriteReadback(*token, 0x4000, 1, [&](u64, u64) {
            entered.set_value();
            // Test-only pause models preemption inside the protected writeback.
            release_future.wait();
            backing.store(7);
        });
    });
    entered_future.wait();
    std::promise<void> started;
    auto started_future = started.get_future();
    auto writer = std::async(std::launch::async, [&] {
        started.set_value();
        return journal.TryPublish(0, ByteWrite(9), [&]() noexcept {
            backing.store(9);
            return true;
        });
    });
    started_future.wait();
    const auto before_release = writer.wait_for(std::chrono::milliseconds{20});
    release.set_value();
    reader.get();
    const auto result = writer.get();
    REQUIRE(before_release == std::future_status::timeout);
    REQUIRE(result.status == PublishStatus::Published);
    REQUIRE(backing.load() == 9);
}

TEST_CASE("Physical history index survives collisions partial retirement and ring wrap",
          "[video_core][deferred_writes]") {
    Journal journal{2, 4};
    for (u64 round = 0; round < 24; ++round) {
        auto token = journal.BeginReadback();
        const Cut start = journal.CaptureCut();
        for (u64 i = 0; i < 8; ++i) {
            auto write = ByteWrite(0);
            // Three colliding buckets, starting at the last hash slot to exercise wrap.
            write.physical_address = ((3 + 16 * (i % 3)) << 12) + i;
            REQUIRE(journal.TryPublish(i % 2, write, []() noexcept { return true; }).sequence ==
                    start + i + 1);
        }
        for (u64 page = 0; page < 4; ++page) {
            std::array<u8, 8> actual{};
            journal.WriteReadback(*token, (3 + 16 * page) << 12, actual.size(),
                [&](u64 offset, u64 size) {
                    std::fill_n(actual.begin() + offset, size, 1);
                });
            for (u64 i = 0; i < 8; ++i) {
                REQUIRE(actual[i] == (i % 3 == page ? 0 : 1));
            }
        }
        REQUIRE(journal.DrainThrough(start + 8, [](const Write&) { return true; }) ==
                DrainResult::Complete);
        token.reset();
        REQUIRE(journal.RetireThrough(start + 5));
        auto newer = journal.BeginReadback();
        auto write = ByteWrite(0);
        write.physical_address = 3 << 12;
        REQUIRE(journal.TryPublish(0, write, []() noexcept { return true; }).sequence == start + 9);
        std::array<u8, 8> actual{};
        journal.WriteReadback(*newer, write.physical_address, actual.size(),
            [&](u64 offset, u64 size) { std::fill_n(actual.begin() + offset, size, 1); });
        REQUIRE(actual == std::array<u8, 8>{0, 1, 1, 1, 1, 1, 1, 1});
        REQUIRE(journal.DrainThrough(start + 9, [](const Write&) { return true; }) ==
                DrainResult::Complete);
        newer.reset();
        REQUIRE(journal.RetireThrough(start + 9));
    }
}

TEST_CASE("Lease closure requires its finite prefix before a mapping transaction can finish",
          "[video_core][deferred_writes][deferred_leases]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 41);
    REQUIRE(lease);
    auto write = ByteWrite(7);
    write.mapping_generation = lease->mapping_generation;
    write.lease_generation = lease->generation;
    u8 backing{};
    REQUIRE(journal.TryPublishLeased(0, write, [&]() noexcept {
                backing = 7;
                return true;
            }).status == PublishStatus::Published);
    const auto close = journal.CloseLease(*lease);
    REQUIRE(close);
    REQUIRE(close->cut == 1);
    REQUIRE_FALSE(journal.ClosureReady(*close));
    REQUIRE_FALSE(journal.FinishClosure(*close));
    REQUIRE_FALSE(journal.CloseLease(*lease));
    REQUIRE_FALSE(journal.GrantLease(0x4000, 0xa000, 42));
    REQUIRE(journal.TryPublishLeased(0, write, [&]() noexcept {
                backing = 9;
                return true;
            }).status == PublishStatus::LeaseUnavailable);
    REQUIRE(backing == 7);
    u8 old_buffer{};
    REQUIRE(journal.DrainThrough(close->cut, [&](const Write& entry) {
                old_buffer = entry.bytes[0];
                return true;
            }) == DrainResult::Complete);
    REQUIRE(old_buffer == 7);
    REQUIRE(journal.ClosureReady(*close));
    // The real mapping transaction belongs between ClosureReady and FinishClosure.
    REQUIRE(journal.FinishClosure(*close));
    const auto replacement = journal.GrantLease(0x4000, 0xa000, 42);
    REQUIRE(replacement);
    REQUIRE(replacement->generation > lease->generation);
    REQUIRE_FALSE(journal.FinishClosure(*close));
    REQUIRE(journal.TryPublishLeased(0, write, []() noexcept { return true; }).status ==
            PublishStatus::LeaseUnavailable);
    write.device_address = 0xa000;
    write.mapping_generation = replacement->mapping_generation;
    write.lease_generation = replacement->generation;
    REQUIRE(journal.TryPublishLeased(0, write, []() noexcept { return true; }).sequence == 2);
}

TEST_CASE("Lease generations reject ABA aliases and stale mapping identities before commit",
          "[video_core][deferred_writes][deferred_leases]") {
    Journal journal{1, 4};
    const auto first = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(first);
    const auto close = journal.CloseLease(*first);
    REQUIRE(close);
    REQUIRE(journal.FinishClosure(*close));
    const auto second = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(second);
    REQUIRE(second->generation != first->generation);
    bool committed{};
    auto write = ByteWrite(1);
    write.lease_generation = first->generation;
    auto commit = [&]() noexcept { committed = true; return true; };
    REQUIRE(journal.TryPublishLeased(0, write, commit).status == PublishStatus::LeaseUnavailable);
    write.lease_generation = second->generation;
    write.mapping_generation = 2;
    REQUIRE(journal.TryPublishLeased(0, write, commit).status == PublishStatus::LeaseUnavailable);
    write.mapping_generation = 1;
    write.device_address = 0x9000;
    REQUIRE(journal.TryPublishLeased(0, write, commit).status == PublishStatus::LeaseUnavailable);
    REQUIRE_FALSE(committed);
    REQUIRE(journal.CaptureCut() == 0);
}

namespace {
std::unique_ptr<Journal::TransitionReservation> ReservePage(Journal& journal, u64 generation = 1) {
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    const std::array stamps{MappingStamp{1, generation}};
    return journal.TryReserveTransition(ranges, stamps);
}

Write LeasedByte(const Journal::Lease& lease, u8 value = 7) {
    auto write = ByteWrite(value);
    write.physical_address = lease.physical_page;
    write.device_address = lease.device_page;
    write.mapping_generation = lease.mapping_generation;
    write.lease_generation = lease.generation;
    return write;
}
}

TEST_CASE("Transition reservation closes a set atomically and blocks grants in empty destinations",
          "[deferred_writes][deferred_transitions]") {
    Journal journal{1, 4};
    const auto first = journal.GrantLease(0x4000, 0x8000, 1);
    const auto other = journal.GrantLease(0x6000, 0xa000, 1);
    REQUIRE(first);
    REQUIRE(other);
    const std::array ranges{
        TransitionRange{TransitionSpace::Physical, 0x4001, 1},
        TransitionRange{TransitionSpace::Device, 0xa004, 4},
        TransitionRange{TransitionSpace::Physical, 0x10000, 4096},
        TransitionRange{TransitionSpace::Device, 0x20000, 4096}};
    const std::array stamps{MappingStamp{1, 1}, MappingStamp{2, 4}};
    auto reservation = journal.TryReserveTransition(ranges, stamps);
    REQUIRE(reservation);
    REQUIRE(reservation->ClosedLeases().size() == 2);
    CHECK_FALSE(journal.GrantLease(0x10000, 0x30000, 1));
    CHECK_FALSE(journal.GrantLease(0x30000, 0x20000, 1));
    CHECK_FALSE(journal.TryReserveTransition(ranges, stamps));
    CHECK_FALSE(journal.FinishClosure({*first, reservation->Watermark()}));
    CHECK(journal.TryPublishLeased(0, LeasedByte(*first), []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
    std::mutex outer;
    std::unique_lock locks{outer};
    bool mutated{};
    CHECK(reservation->Run(locks, [](Cut) { return true; },
        [&](std::span<const MappingStamp> captured) noexcept {
            return std::ranges::equal(captured, stamps);
        }, [&]() noexcept { mutated = true; }) == TransitionResult::Complete);
    CHECK(mutated);
    CHECK(locks.owns_lock());
    CHECK(journal.GrantLease(0x10000, 0x20000, 2));
    const auto fresh = journal.GrantLease(0x4000, 0x8000, 2);
    REQUIRE(fresh);
    CHECK(fresh->generation != first->generation);
}

TEST_CASE("Conflicting multi-page reservation never partly closes an earlier lease",
          "[deferred_writes][deferred_transitions]") {
    Journal journal{1, 4};
    const auto a = journal.GrantLease(0x4000, 0x8000, 1);
    const auto b = journal.GrantLease(0x6000, 0xa000, 1);
    REQUIRE(a);
    REQUIRE(b);
    const auto closure = journal.CloseLease(*b);
    REQUIRE(closure);
    const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 0x3000}};
    const std::array stamps{MappingStamp{1, 1}};
    CHECK_FALSE(journal.TryReserveTransition(ranges, stamps));
    CHECK(journal.TryPublishLeased(0, LeasedByte(*a), []() noexcept { return true; }).status ==
          PublishStatus::Published);
    CHECK(journal.FinishClosure(*closure));
}

TEST_CASE("Transition cut includes a concurrent CPU commit already in progress",
          "[deferred_writes][deferred_transitions]") {
    Journal journal{1, 4};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    std::promise<void> entered, release, reserving;
    auto entered_future = entered.get_future();
    auto release_future = release.get_future();
    auto reserving_future = reserving.get_future();
    auto writer = std::async(std::launch::async, [&] {
        return journal.TryPublishLeased(0, LeasedByte(*lease), [&]() noexcept {
            entered.set_value();
            release_future.wait(); // Test-only preemption inside commit.
            return true;
        });
    });
    entered_future.wait();
    auto closer = std::async(std::launch::async, [&] {
        reserving.set_value();
        return ReservePage(journal);
    });
    reserving_future.wait();
    const auto before_release = closer.wait_for(std::chrono::milliseconds{20});
    release.set_value();
    const auto published = writer.get();
    auto reservation = closer.get();
    CHECK(before_release == std::future_status::timeout);
    REQUIRE(published.status == PublishStatus::Published);
    REQUIRE(reservation);
    CHECK(reservation->Watermark() == published.sequence);
    CHECK_FALSE(reservation->ClosureReady());
    CHECK(journal.TryPublishLeased(0, LeasedByte(*lease), []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
}

TEST_CASE("Transition drain drops caller locks and does not absorb later publications",
          "[deferred_writes][deferred_transitions]") {
    Journal journal{2, 4};
    const auto affected = journal.GrantLease(0x4000, 0x8000, 1);
    const auto unrelated = journal.GrantLease(0x6000, 0xa000, 1);
    REQUIRE(affected);
    REQUIRE(unrelated);
    REQUIRE(journal.TryPublishLeased(0, LeasedByte(*affected), []() noexcept { return true; }).sequence == 1);
    std::mutex outer;
    std::unique_lock locks{outer};
    auto reservation = ReservePage(journal);
    REQUIRE(reservation);
    Publication later{PublishStatus::Rejected};
    bool got_outer{}, mutated{};
    const auto result = reservation->Run(locks, [&](Cut cut) {
        auto worker = std::async(std::launch::async, [&] {
            std::unique_lock worker_lock{outer, std::try_to_lock};
            got_outer = worker_lock.owns_lock();
            if (!got_outer) return false;
            return journal.DrainThrough(cut, [&](const Write&) {
                later = journal.TryPublishLeased(1, LeasedByte(*unrelated),
                                                []() noexcept { return true; });
                return true;
            }) == DrainResult::Complete;
        });
        return worker.get();
    }, [](auto) noexcept { return true; }, [&]() noexcept { mutated = true; });
    CHECK(result == TransitionResult::Complete);
    CHECK(got_outer);
    CHECK(mutated);
    CHECK(locks.owns_lock());
    CHECK(later.sequence == 2);
    CHECK(journal.AppliedThrough() == 1);
    CHECK(journal.CaptureCut() == 2);
}

TEST_CASE("Transition revalidates epochs across ABA and retries without reopening old leases",
          "[deferred_writes][deferred_transitions]") {
    Journal journal{1, 4};
    const auto old = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(old);
    std::mutex outer;
    std::unique_lock locks{outer};
    u64 epoch = 1;
    u64 address = 0x8000;
    bool mutated{};
    auto first = ReservePage(journal, epoch);
    REQUIRE(first);
    const auto validate = [&](std::span<const MappingStamp> stamps) noexcept {
        return stamps[0].generation == epoch;
    };
    const auto result = first->Run(locks, [&](Cut) {
        auto remapper = std::async(std::launch::async, [&] {
            std::scoped_lock lock{outer};
            address = 0xa000; ++epoch;
            address = 0x8000; ++epoch;
        });
        remapper.get();
        return true;
    }, validate, [&]() noexcept { mutated = true; });
    CHECK(result == TransitionResult::MappingChanged);
    CHECK(address == 0x8000);
    CHECK_FALSE(mutated);
    first.reset();
    CHECK_FALSE(journal.GrantLease(0x4000, 0x8000, epoch));
    auto retry = ReservePage(journal, epoch);
    REQUIRE(retry);
    REQUIRE(retry->ClosedLeases().size() == 1);
    CHECK(retry->Run(locks, [](Cut) { return true; }, validate,
                     [&]() noexcept { mutated = true; }) == TransitionResult::Complete);
    CHECK(mutated);
    const auto replacement = journal.GrantLease(0x4000, 0x8000, epoch);
    REQUIRE(replacement);
    CHECK(replacement->generation > old->generation);
    CHECK(journal.TryPublishLeased(0, LeasedByte(*old), []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
}

TEST_CASE("Transition cancellation retains pending writes and live readbacks veto mutation",
          "[deferred_writes][deferred_transitions]") {
    Journal journal{1, 1};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    auto readback = journal.BeginReadback();
    REQUIRE(journal.TryPublishLeased(0, LeasedByte(*lease), []() noexcept { return true; }).sequence == 1);
    { auto aborted = ReservePage(journal); REQUIRE(aborted); }
    CHECK_FALSE(journal.GrantLease(0x4000, 0x8000, 1));
    auto retry = ReservePage(journal);
    REQUIRE(retry);
    std::mutex outer;
    std::unique_lock locks{outer};
    bool mutated{};
    unsigned applications{};
    const auto drain = [&](Cut cut) {
        return journal.DrainThrough(cut, [&](const Write&) { ++applications; return true; }) ==
               DrainResult::Complete;
    };
    CHECK(retry->Run(locks, drain, [](auto) noexcept { return true; },
        [&]() noexcept { mutated = true; }) == TransitionResult::ReadbacksPending);
    CHECK_FALSE(mutated);
    CHECK_FALSE(journal.RetireThrough(1));
    CHECK(locks.owns_lock());
    readback.reset();
    CHECK(retry->Run(locks, drain, [](auto) noexcept { return true; },
        [&]() noexcept { mutated = true; }) == TransitionResult::Complete);
    CHECK(mutated);
    CHECK(applications == 1);
    CHECK(journal.RetireThrough(1));
}

TEST_CASE("Full history can be retired during transition drain without caller locks or CPU commit",
          "[deferred_writes][deferred_transitions]") {
    Journal journal{1, 1};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    REQUIRE(journal.TryPublishLeased(0, LeasedByte(*lease), []() noexcept { return true; }).sequence == 1);
    u64 pc = 100;
    CHECK(journal.TryPublishLeased(0, LeasedByte(*lease), [&]() noexcept { ++pc; return true; }).status ==
          PublishStatus::Full);
    CHECK(pc == 100);
    std::mutex outer;
    std::unique_lock locks{outer};
    auto reservation = ReservePage(journal);
    REQUIRE(reservation);
    const auto result = reservation->Run(locks, [&](Cut cut) {
        return std::async(std::launch::async, [&] {
            std::unique_lock worker_lock{outer, std::try_to_lock};
            if (!worker_lock.owns_lock()) return false;
            return journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete &&
                   journal.RetireThrough(cut);
        }).get();
    }, [](auto) noexcept { return true; }, []() noexcept {});
    CHECK(result == TransitionResult::Complete);
    CHECK(locks.owns_lock());
    const auto fresh = journal.GrantLease(0x4000, 0x8000, 2);
    REQUIRE(fresh);
    CHECK(journal.TryPublishLeased(0, LeasedByte(*fresh), [&]() noexcept { ++pc; return true; }).sequence == 2);
    CHECK(pc == 101);
}

TEST_CASE("Failed transition drain restores outer locks and does not mutate",
          "[deferred_writes][deferred_transitions]") {
    Journal journal{1, 1};
    auto reservation = ReservePage(journal);
    REQUIRE(reservation);
    std::mutex outer;
    std::unique_lock locks{outer};
    bool mutated{};
    CHECK(reservation->Run(locks, [](Cut) { return false; }, [](auto) noexcept { return true; },
        [&]() noexcept { mutated = true; }) == TransitionResult::DrainPending);
    CHECK(locks.owns_lock());
    CHECK_FALSE(mutated);
    CHECK_THROWS_AS(reservation->Run(locks, [](Cut) -> bool { throw std::runtime_error("drain"); },
        [](auto) noexcept { return true; }, [&]() noexcept { mutated = true; }), std::runtime_error);
    CHECK(locks.owns_lock());
    CHECK_FALSE(mutated);
}

TEST_CASE("Global suspension rejects commits and grants until the last ready participant leaves",
          "[deferred_writes][deferred_suspension]") {
    Journal journal{2, 1};
    const auto old = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(old);
    REQUIRE(journal.TryPublishLeased(0, LeasedByte(*old), []() noexcept { return true; }).sequence == 1);
    auto outer = journal.BeginSuspension();
    auto nested = outer.Fork();
    CHECK(outer.Epoch() == nested.Epoch());
    CHECK(outer.Watermark() == 1);
    CHECK(journal.GetSuspensionState() == SuspensionState::Draining);
    bool committed{};
    const auto commit = [&]() noexcept { committed = true; return true; };
    CHECK(journal.TryPublishLeased(0, LeasedByte(*old), commit).status == PublishStatus::Suspended);
    CHECK(journal.TryPublish(1, ByteWrite(9), commit).status == PublishStatus::Suspended);
    CHECK_FALSE(committed);
    CHECK_FALSE(journal.GrantLease(0x6000, 0xa000, 1));
    // A callback reporting success cannot bypass the applied-prefix check.
    CHECK(outer.TryPrepare([](Cut) { return true; }) == SuspensionResult::DrainPending);
    CHECK_FALSE(outer.IsReady());
    unsigned applications{};
    CHECK(outer.TryPrepare([&](Cut cut) {
        return journal.DrainThrough(cut, [&](const Write&) { ++applications; return true; }) ==
               DrainResult::Complete;
    }) == SuspensionResult::Ready);
    CHECK(applications == 1);
    CHECK(nested.IsReady());
    CHECK_FALSE(journal.GrantLease(0x4000, 0x8000, 2));
    CHECK(journal.TryPublishLeased(1, LeasedByte(*old), commit).status == PublishStatus::Suspended);
    auto late = journal.BeginSuspension();
    CHECK(late.Epoch() == nested.Epoch());
    CHECK(late.IsReady());
    outer.Reset();
    nested.Reset();
    CHECK(journal.GetSuspensionState() == SuspensionState::SuspendedReady);
    late.Reset();
    CHECK(journal.GetSuspensionState() == SuspensionState::Running);
    const auto fresh = journal.GrantLease(0x4000, 0x8000, 1); // Same mapping, new lease identity.
    REQUIRE(fresh);
    CHECK(fresh->generation > old->generation);
    CHECK(journal.TryPublishLeased(0, LeasedByte(*old), commit).status == PublishStatus::LeaseUnavailable);
    CHECK_FALSE(committed);
    // Capacity one: Ready retired the previous history as well as the old lease.
    CHECK(journal.TryPublishLeased(0, LeasedByte(*fresh), commit).sequence == 2);
    CHECK(committed);
    auto next = journal.BeginSuspension();
    CHECK(next.Epoch() > late.Epoch());
    CHECK(next.Watermark() == 2);
}

TEST_CASE("Two cores share one suspension barrier and fallback waits for its preparation",
          "[deferred_writes][deferred_suspension]") {
    Journal journal{2, 2};
    REQUIRE(Publish(journal, 0, 7).sequence == 1);
    auto first = journal.BeginSuspension();
    std::promise<void> draining, release_drain, second_arrived;
    auto draining_future = draining.get_future();
    auto release_future = release_drain.get_future();
    auto arrived_future = second_arrived.get_future();
    std::atomic<unsigned> callbacks{};
    auto prepare = std::async(std::launch::async, [&] {
        return first.TryPrepare([&](Cut cut) {
            ++callbacks;
            draining.set_value();
            release_future.wait();
            return journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete;
        });
    });
    draining_future.wait();
    struct Observation {
        u64 epoch{};
        Cut cut{};
        SuspensionResult preparation{};
        bool early_ready{};
        bool woke_ready{};
        unsigned fallback_calls{};
    };
    auto other_core = std::async(std::launch::async, [&] {
        auto second = journal.BeginSuspension();
        Observation result{second.Epoch(), second.Watermark()};
        result.preparation = second.TryPrepare([&](Cut) { ++callbacks; return true; });
        result.early_ready = second.WaitForReady(std::chrono::milliseconds{0});
        if (result.early_ready) ++result.fallback_calls;
        second_arrived.set_value();
        result.woke_ready = second.WaitForReady(std::chrono::seconds{2});
        if (result.woke_ready) ++result.fallback_calls; // Retain second throughout fallback.
        return result;
    });
    arrived_future.wait();
    const auto phase_before = journal.GetSuspensionState();
    release_drain.set_value();
    const auto preparation = prepare.get();
    const auto observed = other_core.get();
    CHECK(phase_before == SuspensionState::Draining);
    CHECK(preparation == SuspensionResult::Ready);
    CHECK(observed.epoch == first.Epoch());
    CHECK(observed.cut == first.Watermark());
    CHECK(observed.preparation == SuspensionResult::Busy);
    CHECK_FALSE(observed.early_ready);
    CHECK(observed.woke_ready);
    CHECK(observed.fallback_calls == 1);
    CHECK(callbacks == 1);
    CHECK(first.WaitForReady(std::chrono::milliseconds{0})); // Already-ready predicate.
}

TEST_CASE("Suspension captures a CPU commit already inside the publication gate",
          "[deferred_writes][deferred_suspension]") {
    Journal journal{1, 2};
    std::promise<void> entered, release, suspending;
    auto entered_future = entered.get_future();
    auto release_future = release.get_future();
    auto suspending_future = suspending.get_future();
    auto writer = std::async(std::launch::async, [&] {
        return journal.TryPublish(0, ByteWrite(7), [&]() noexcept {
            entered.set_value();
            release_future.wait(); // Test-only preemption between backing commit and publication.
            return true;
        });
    });
    entered_future.wait();
    auto closer = std::async(std::launch::async, [&] {
        suspending.set_value();
        return journal.BeginSuspension();
    });
    suspending_future.wait();
    const auto before_release = closer.wait_for(std::chrono::milliseconds{20});
    release.set_value();
    const auto published = writer.get();
    auto token = closer.get();
    CHECK(before_release == std::future_status::timeout);
    REQUIRE(published.status == PublishStatus::Published);
    CHECK(token.Watermark() == published.sequence);
    CHECK(Publish(journal, 0, 9).status == PublishStatus::Suspended);
}

TEST_CASE("A deferred callback retains suspension after its simulated SVC returns with an error",
          "[deferred_writes][deferred_suspension]") {
    Journal journal{1, 2};
    std::function<bool()> callback;
    auto simulated_svc = [&]() {
        auto token = journal.BeginSuspension();
        if (token.TryPrepare([](Cut) { return true; }) != SuspensionResult::Ready) return false;
        auto deferred = std::make_shared<Suspension>(token.Fork());
        callback = [deferred, &journal] {
            return deferred->IsReady() && Publish(journal, 0, 7).status == PublishStatus::Suspended;
        };
        return false; // Error after scheduling: local RAII must not resume the callback's epoch.
    };
    CHECK_FALSE(simulated_svc());
    CHECK(journal.GetSuspensionState() == SuspensionState::SuspendedReady);
    auto callback_copy = callback; // std::function copies share ownership of its participant.
    CHECK(callback());
    callback = {};
    CHECK(journal.GetSuspensionState() == SuspensionState::SuspendedReady);
    CHECK(callback_copy());
    callback_copy = {};
    CHECK(journal.GetSuspensionState() == SuspensionState::Running);
    CHECK(Publish(journal, 0, 9).status == PublishStatus::Published);
    auto throwing_svc = [&] {
        auto token = journal.BeginSuspension();
        if (token.TryPrepare([&](Cut cut) {
                return journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete;
            }) != SuspensionResult::Ready) throw std::logic_error("not ready");
        throw std::runtime_error("simulated SVC error");
    };
    CHECK_THROWS_AS(throwing_svc(), std::runtime_error);
    CHECK(journal.GetSuspensionState() == SuspensionState::Running);
}

TEST_CASE("Failed or abandoned suspension remains closed and can be adopted without losing writes",
          "[deferred_writes][deferred_suspension]") {
    Journal journal{1, 1};
    REQUIRE(Publish(journal, 0, 7).sequence == 1);
    u64 epoch{};
    {
        auto token = journal.BeginSuspension();
        epoch = token.Epoch();
        CHECK(token.TryPrepare([](Cut) { return false; }) == SuspensionResult::DrainPending);
        CHECK_THROWS_AS(token.TryPrepare([](Cut) -> bool { throw std::runtime_error("drain"); }),
                        std::runtime_error);
    }
    CHECK(journal.GetSuspensionState() == SuspensionState::Draining);
    CHECK(Publish(journal, 0, 9).status == PublishStatus::Suspended);
    auto adopted = journal.BeginSuspension();
    CHECK(adopted.Epoch() == epoch);
    CHECK(adopted.Watermark() == 1);
    u8 applied_byte{};
    CHECK(adopted.TryPrepare([&](Cut cut) {
        return journal.DrainThrough(cut, [&](const Write& write) {
            applied_byte = write.bytes[0]; return true;
        }) == DrainResult::Complete;
    }) == SuspensionResult::Ready);
    CHECK(applied_byte == 7);
    adopted.Reset();
    CHECK(Publish(journal, 0, 9).sequence == 2);
}

TEST_CASE("Preparation pins its epoch even when the drain callback releases the original token",
          "[deferred_writes][deferred_suspension]") {
    Journal journal{1, 1};
    auto original = journal.BeginSuspension();
    Suspension deferred;
    SuspensionState during{};
    CHECK(original.TryPrepare([&](Cut) {
        original.Reset();
        during = journal.GetSuspensionState();
        deferred = journal.BeginSuspension();
        return true;
    }) == SuspensionResult::Ready);
    CHECK_FALSE(original);
    CHECK(during == SuspensionState::Draining);
    CHECK(deferred.IsReady());
    auto moved = std::move(deferred);
    CHECK_FALSE(deferred.IsReady());
    CHECK_FALSE(deferred.Fork());
    CHECK(deferred.TryPrepare([](Cut) { return true; }) == SuspensionResult::InvalidToken);
    CHECK(moved.IsReady());
    moved.Reset();
    CHECK(journal.GetSuspensionState() == SuspensionState::Running);
}

TEST_CASE("Live and newly recorded readbacks veto suspension readiness and retain protected history",
          "[deferred_writes][deferred_suspension]") {
    Journal journal{1, 1};
    auto readback = journal.BeginReadback();
    u8 backing{};
    REQUIRE(journal.TryPublish(0, ByteWrite(7), [&]() noexcept { backing = 7; return true; }).sequence == 1);
    auto token = journal.BeginSuspension();
    const auto drain = [&](Cut cut) {
        return journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete;
    };
    CHECK(token.TryPrepare(drain) == SuspensionResult::ReadbacksPending);
    CHECK_FALSE(token.IsReady());
    CHECK_FALSE(journal.RetireThrough(1));
    journal.WriteReadback(*readback, 0x4000, 1, [&](u64, u64) { backing = 0; });
    CHECK(backing == 7);
    readback.reset();
    CHECK(token.TryPrepare([&](Cut cut) {
        if (!drain(cut)) return false;
        readback = journal.BeginReadback(); // Created after W applied, before readiness check.
        return true;
    }) == SuspensionResult::ReadbacksPending);
    readback.reset();
    CHECK(token.TryPrepare(drain) == SuspensionResult::Ready);
    // New readbacks are allowed in legacy mode; future B writes still protect their bytes.
    auto legacy_readback = journal.BeginReadback();
    token.Reset();
    REQUIRE(journal.TryPublish(0, ByteWrite(9), [&]() noexcept { backing = 9; return true; }).sequence == 2);
    journal.WriteReadback(*legacy_readback, 0x4000, 1, [&](u64, u64) { backing = 0; });
    CHECK(backing == 9);
}

TEST_CASE("Range transitions can finish during global draining without waiting on that barrier",
          "[deferred_writes][deferred_suspension]") {
    Journal journal{1, 2};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    REQUIRE(journal.TryPublishLeased(0, LeasedByte(*lease), []() noexcept { return true; }).sequence == 1);
    auto suspension = journal.BeginSuspension();
    auto range = ReservePage(journal);
    REQUIRE(range);
    const auto drain = [&](Cut cut) {
        return journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete;
    };
    CHECK(suspension.TryPrepare(drain) == SuspensionResult::TransitionsPending);
    bool mutated{};
    std::mutex outer;
    std::unique_lock locks{outer};
    CHECK(range->Run(locks, drain, [](auto) noexcept { return true; },
                     [&]() noexcept { mutated = true; }) == TransitionResult::Complete);
    locks.unlock();
    CHECK(mutated);
    CHECK(suspension.TryPrepare(drain) == SuspensionResult::Ready);
    suspension.Reset();
    const auto fresh = journal.GrantLease(0x4000, 0x8000, 2);
    REQUIRE(fresh);
    const auto closure = journal.CloseLease(*fresh);
    REQUIRE(closure);
    auto second = journal.BeginSuspension();
    CHECK(second.TryPrepare(drain) == SuspensionResult::TransitionsPending);
    CHECK(journal.FinishClosure(*closure));
    CHECK(second.TryPrepare(drain) == SuspensionResult::Ready);
}

TEST_CASE("Global suspension retires abandoned range identities instead of reopening them",
          "[deferred_writes][deferred_suspension]") {
    Journal journal{1, 1};
    const auto old = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(old);
    { auto abandoned = ReservePage(journal); REQUIRE(abandoned); }
    auto token = journal.BeginSuspension();
    CHECK(token.TryPrepare([](Cut) { return true; }) == SuspensionResult::Ready);
    token.Reset();
    const auto fresh = journal.GrantLease(0x4000, 0x8000, 2);
    REQUIRE(fresh);
    CHECK(fresh->generation > old->generation);
    CHECK(journal.TryPublishLeased(0, LeasedByte(*old), []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
}

TEST_CASE("Range closure excludes a concurrent CPU publication through a simulated DMA or join",
          "[deferred_writes][deferred_suspension][deferred_transitions]") {
    for (const bool is_join : {false, true}) {
        CAPTURE(is_join);
        Journal journal{2, 2};
        const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
        const auto unrelated = journal.GrantLease(0x6000, 0xa000, 1);
        REQUIRE(lease);
        REQUIRE(unrelated);
        u8 backing = 7, gpu{}, joined_gpu{};
        REQUIRE(journal.TryPublishLeased(0, LeasedByte(*lease), []() noexcept { return true; }).sequence == 1);
        auto range = ReservePage(journal);
        REQUIRE(range);
        Publication raced{PublishStatus::Rejected}, other{PublishStatus::Rejected};
        bool committed{}, mutated{};
        std::mutex outer;
        std::unique_lock locks{outer};
        CHECK(range->Run(locks, [&](Cut cut) {
            if (journal.DrainThrough(cut, [&](const Write& write) {
                    gpu = write.bytes[0]; return true;
                }) != DrainResult::Complete) return false;
            std::async(std::launch::async, [&] {
                raced = journal.TryPublishLeased(1, LeasedByte(*lease, 9), [&]() noexcept {
                    committed = true; backing = 9; return true;
                });
                other = journal.TryPublishLeased(1, LeasedByte(*unrelated), []() noexcept { return true; });
            }).get();
            return true;
        }, [](auto) noexcept { return true; }, [&]() noexcept {
            if (is_join) joined_gpu = gpu;
            else backing = gpu;
            mutated = true;
        }) == TransitionResult::Complete);
        CHECK(mutated);
        CHECK_FALSE(committed);
        CHECK(raced.status == PublishStatus::LeaseUnavailable);
        CHECK(other.sequence == 2);
        CHECK(journal.AppliedThrough() == 1);
        CHECK(backing == 7);
        CHECK(gpu == 7);
        if (is_join) CHECK(joined_gpu == 7);
        const auto fresh = journal.GrantLease(0x4000, is_join ? 0xc000 : 0x8000, 2);
        REQUIRE(fresh);
        CHECK(journal.TryPublishLeased(0, LeasedByte(*fresh, 9), [&]() noexcept {
            backing = 9; return true;
        }).status == PublishStatus::Published);
    }
}

TEST_CASE("Nested consumers expand one reservation while preserving live readback history",
          "[deferred_writes][deferred_consumer]") {
    Journal journal{2, 2};
    auto readback = journal.BeginReadback();
    const auto a = journal.GrantLease(0x4000, 0x8000, 1);
    const auto b = journal.GrantLease(0x5000, 0x9000, 1);
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(journal.TryPublishLeased(0, LeasedByte(*a, 7), []() noexcept { return true; }).sequence == 1);
    auto root = journal.TryBeginConsumer();
    const std::array outer{TransitionRange{TransitionSpace::Physical, 0x4000, 0x1000}};
    REQUIRE(root->TryExtend(outer));
    REQUIRE(journal.DrainThrough(root->Watermark(), [](const Write&) { return true; }) ==
            DrainResult::Complete);
    REQUIRE(journal.TryPublishLeased(1, LeasedByte(*b, 9), []() noexcept { return true; }).sequence == 2);
    // FindBuffer/Join share the DMA root, including a partial overlap and a new page.
    auto& child = *root;
    const std::array expanded{TransitionRange{TransitionSpace::Physical, 0x4800, 0x1800}};
    REQUIRE(child.TryExtend(expanded));
    CHECK(root->ClosedLeases().size() == 2);
    CHECK(root->Watermark() == 2);
    CHECK_FALSE(root->Complete());
    CHECK(journal.TryPublishLeased(0, LeasedByte(*a), []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
    CHECK(journal.TryPublishLeased(1, LeasedByte(*b), []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
    REQUIRE(journal.DrainThrough(root->Watermark(), [](const Write&) { return true; }) ==
            DrainResult::Complete);
    REQUIRE(root->Complete()); // Readback is deliberately still alive.
    bool overwritten{};
    journal.WriteReadback(*readback, 0x4000, 1, [&](u64, u64) { overwritten = true; });
    journal.WriteReadback(*readback, 0x5000, 1, [&](u64, u64) { overwritten = true; });
    CHECK_FALSE(overwritten);
    CHECK_FALSE(journal.RetireThrough(2));
    const auto replacement = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(replacement);
    CHECK(replacement->generation > a->generation);
    readback.reset();
    CHECK(journal.RetireThrough(2));
}

TEST_CASE("Consumer expansion rejects foreign ranges and closures without partial closure",
          "[deferred_writes][deferred_consumer]") {
    Journal journal{1, 4};
    const auto a = journal.GrantLease(0x4000, 0x14000, 1);
    const auto b = journal.GrantLease(0x6000, 0x16000, 1);
    const auto c = journal.GrantLease(0x8000, 0x18000, 1);
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(c);
    auto root = journal.TryBeginConsumer();
    const std::array own{TransitionRange{TransitionSpace::Physical, 0x4000, 0x1000}};
    REQUIRE(root->TryExtend(own));
    const std::array foreign_range{TransitionRange{TransitionSpace::Physical, 0x8000, 0x1000}};
    const std::array stamps{MappingStamp{1, 1}};
    auto foreign = journal.TryReserveTransition(foreign_range, stamps);
    REQUIRE(foreign);
    const std::array conflict{TransitionRange{TransitionSpace::Physical, 0x6000, 0x3000}};
    CHECK_FALSE(root->TryExtend(conflict));
    CHECK(root->Watermark() == 0);
    CHECK(root->ClosedLeases().size() == 1);
    CHECK(journal.TryPublishLeased(0, LeasedByte(*b), []() noexcept { return true; }).sequence == 1);
    foreign.reset(); // Abandoned closures may be adopted, but active legacy closures may not.
    const auto closed_b = journal.CloseLease(*b);
    REQUIRE(closed_b);
    CHECK_FALSE(root->TryExtend(conflict));
    REQUIRE(journal.DrainThrough(1, [](const Write&) { return true; }) == DrainResult::Complete);
    REQUIRE(journal.FinishClosure(*closed_b));
    REQUIRE(root->TryExtend(conflict));
    CHECK(root->ClosedLeases().size() == 2);
    CHECK(root->Watermark() == 1);
    CHECK(root->Complete());
}

TEST_CASE("Consumer abort leaves identities adoptable and preserves accepted writes",
          "[deferred_writes][deferred_consumer]") {
    Journal journal{1, 1};
    auto readback = journal.BeginReadback();
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    REQUIRE(journal.TryPublishLeased(0, LeasedByte(*lease, 7), []() noexcept { return true; }).sequence == 1);
    const std::array ranges{TransitionRange{TransitionSpace::Device, 0x8000, 0x1000}};
    {
        auto first = journal.TryBeginConsumer();
        REQUIRE(first->TryExtend(ranges));
    }
    CHECK_FALSE(journal.GrantLease(0x4000, 0x8000, 2));
    auto adopted = journal.TryBeginConsumer();
    REQUIRE(adopted->TryExtend(ranges));
    CHECK_FALSE(adopted->Complete());
    u8 applied{};
    REQUIRE(journal.DrainThrough(adopted->Watermark(), [&](const Write& write) {
        applied = write.bytes[0];
        return true;
    }) == DrainResult::Complete);
    REQUIRE(adopted->Complete());
    CHECK(applied == 7);
    bool overwritten{};
    journal.WriteReadback(*readback, 0x4000, 1, [&](u64, u64) { overwritten = true; });
    CHECK_FALSE(overwritten);
    CHECK_FALSE(adopted->TryExtend(ranges));
    CHECK_FALSE(adopted->Complete());
}

TEST_CASE("Consumer extension includes a CPU commit already in the publication gate",
          "[deferred_writes][deferred_consumer]") {
    Journal journal{1, 1};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    auto consumer = journal.TryBeginConsumer();
    std::promise<void> entered, release, extending;
    auto entered_future = entered.get_future();
    auto release_future = release.get_future();
    auto extending_future = extending.get_future();
    auto writer = std::async(std::launch::async, [&] {
        return journal.TryPublishLeased(0, LeasedByte(*lease), [&]() noexcept {
            entered.set_value();
            release_future.wait();
            return true;
        });
    });
    entered_future.wait();
    auto closer = std::async(std::launch::async, [&] {
        const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 0x1000}};
        extending.set_value();
        return consumer->TryExtend(ranges);
    });
    extending_future.wait();
    const auto status = closer.wait_for(std::chrono::milliseconds{20});
    release.set_value();
    const auto published = writer.get();
    const bool extended = closer.get();
    CHECK(status == std::future_status::timeout);
    REQUIRE(extended);
    REQUIRE(published.status == PublishStatus::Published);
    CHECK(consumer->Watermark() == published.sequence);
    CHECK_FALSE(consumer->ClosureReady());
}

TEST_CASE("Consumer reservations protect empty ranges and reject invalid expansions",
          "[deferred_writes][deferred_consumer]") {
    Journal journal{1, 1};
    auto consumer = journal.TryBeginConsumer();
    const std::array ranges{TransitionRange{TransitionSpace::Device, 0x8000, 0x1000}};
    REQUIRE(consumer->TryExtend(ranges));
    CHECK_FALSE(journal.GrantLease(0x4000, 0x8000, 1));
    auto other = journal.TryBeginConsumer();
    CHECK_FALSE(other->TryExtend(ranges));
    const std::array stamps{MappingStamp{1, 1}};
    CHECK_FALSE(journal.TryReserveTransition(ranges, stamps));
    const std::array invalid{TransitionRange{TransitionSpace::Physical, ~u64{0}, 2}};
    CHECK_FALSE(consumer->TryExtend(invalid));
    CHECK_FALSE(consumer->TryExtend({}));
    REQUIRE(consumer->TryExtend(ranges));
    auto readback = journal.BeginReadback();
    CHECK(consumer->Complete());
    CHECK(journal.GrantLease(0x4000, 0x8000, 1));
}

TEST_CASE("Command page validation sees CPU publications before drain and after history retirement",
          "[deferred_writes][deferred_command_pages]") {
    Journal journal{2, 1};
    REQUIRE(Publish(journal, 0, 7).sequence == 1);
    unsigned visits{};
    CHECK_FALSE(journal.VisitIfCpuPageUnchanged(0x4000, 0, [&] { ++visits; }));
    CHECK(journal.DrainThrough(1, [](const Write&) { return true; }) == DrainResult::Complete);
    CHECK(journal.VisitIfCpuPageUnchanged(0x4000, 1, [&] { ++visits; }));
    CHECK(journal.RetireThrough(1));
    auto alias_write = ByteWrite(7); // Same value, different device alias of the same physical page.
    alias_write.device_address = 0xa000;
    REQUIRE(journal.TryPublish(1, alias_write, []() noexcept { return true; }).sequence == 2);
    CHECK_FALSE(journal.VisitIfCpuPageUnchanged(0x4008, 1, [&] { ++visits; }));
    CHECK(journal.DrainThrough(2, [](const Write&) { return true; }) == DrainResult::Complete);
    CHECK(journal.RetireThrough(2));
    CHECK_FALSE(journal.VisitIfCpuPageUnchanged(0x4000, 1, [&] { ++visits; }));
    CHECK(journal.VisitIfCpuPageUnchanged(0x4000, 2, [&] { ++visits; }));
    auto suspension = journal.BeginSuspension();
    CHECK(suspension.TryPrepare([](Cut) { return true; }) == SuspensionResult::Ready);
    suspension.Reset();
    CHECK_FALSE(journal.VisitIfCpuPageUnchanged(0x4000, 1, [&] { ++visits; }));
    CHECK(visits == 2);
}

TEST_CASE("Rejected CPU commits leave command page snapshots valid and future cuts cannot validate",
          "[deferred_writes][deferred_command_pages]") {
    Journal journal{1, 1};
    CHECK(journal.TryPublish(0, ByteWrite(7), []() noexcept { return false; }).status ==
          PublishStatus::Rejected);
    unsigned visits{};
    CHECK(journal.VisitIfCpuPageUnchanged(0x4000, 0, [&] { ++visits; }));
    CHECK_FALSE(journal.VisitIfCpuPageUnchanged(0x4000, 1, [&] { ++visits; }));
    auto suspension = journal.BeginSuspension();
    CHECK(journal.TryPublish(0, ByteWrite(9), []() noexcept { return true; }).status ==
          PublishStatus::Suspended);
    CHECK(journal.VisitIfCpuPageUnchanged(0x4000, 0, [&] { ++visits; }));
    CHECK(visits == 2);
}

TEST_CASE("A CPU publication cannot interleave with validation and copying of a command page",
          "[deferred_writes][deferred_command_pages]") {
    Journal journal{1, 2};
    std::promise<void> copying, release_copy, publishing;
    auto copying_future = copying.get_future();
    auto release_future = release_copy.get_future();
    auto publishing_future = publishing.get_future();
    u8 backing = 1, observed{};
    auto reader = std::async(std::launch::async, [&] {
        return journal.VisitIfCpuPageUnchanged(0x4000, 0, [&] {
            copying.set_value();
            release_future.wait(); // Test-only preemption inside the final copy.
            observed = backing;
        });
    });
    copying_future.wait();
    auto writer = std::async(std::launch::async, [&] {
        publishing.set_value();
        return journal.TryPublish(0, ByteWrite(9), [&]() noexcept { backing = 9; return true; });
    });
    publishing_future.wait();
    const auto blocked = writer.wait_for(std::chrono::milliseconds{20});
    release_copy.set_value();
    const bool copied = reader.get();
    const auto publication = writer.get();
    CHECK(blocked == std::future_status::timeout);
    CHECK(copied);
    CHECK(observed == 1);
    CHECK(publication.status == PublishStatus::Published);
    CHECK(backing == 9);
    CHECK_FALSE(journal.VisitIfCpuPageUnchanged(0x4000, 0, [] {}));
}

namespace {
struct ConsumerMemory {
    std::map<DAddr, PAddr> pages;
    u8 unused{};
    template <typename Visit>
    bool ApplyOpOnMappedMemory(DAddr address, u64 size, Visit&& visit) {
        if (size > 4096 - (address & 4095)) return false;
        const auto it = pages.find(address & ~DAddr{4095});
        if (it == pages.end()) return false;
        visit(&unused, it->second + (address & 4095));
        return true;
    }
};
}

TEST_CASE("Consumer upload scope closes every physical segment and aliases through nested reads",
          "[deferred_writes][deferred_consumer]") {
    Journal journal{2, 4};
    ConsumerMemory memory{{{0x8000, 0x4000}, {0x9000, 0x6000}}};
    const auto a = journal.GrantLease(0x4000, 0xc000, 1);
    const auto b = journal.GrantLease(0x6000, 0xe000, 1);
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(journal.TryPublishLeased(0, LeasedByte(*a), []() noexcept { return true; }).sequence == 1);
    std::vector<TransitionRange> first, all;
    AddConsumerDeviceRange(first, memory, 0x8000, 4096);
    AddConsumerDeviceRange(all, memory, 0x8000, 3 * 4096); // Last segment has no physical backing.
    const auto apply = [&](Cut cut) {
        REQUIRE(journal.DrainThrough(cut, [](const Write&) { return true; }) == DrainResult::Complete);
    };
    unsigned committed{};
    const auto commit = [&]() noexcept { ++committed; return true; };
    {
        ConsumerScope root{&journal, first, apply};
        CHECK(journal.AppliedThrough() == 1);
        CHECK(journal.TryPublishLeased(0, LeasedByte(*a), commit).status == PublishStatus::LeaseUnavailable);
        CHECK(journal.TryPublishLeased(1, LeasedByte(*b), commit).sequence == 2);
        {
            ConsumerScope child{&journal, all, apply};
            auto publisher = std::async(std::launch::async, [&] {
                return journal.TryPublishLeased(1, LeasedByte(*b), commit);
            });
            CHECK(publisher.get().status == PublishStatus::LeaseUnavailable);
            CHECK(journal.AppliedThrough() == 2);
            CHECK_FALSE(journal.GrantLease(0x10000, 0xa000, 1));
        }
        // The child does not release its newly added physical page before the outer upload ends.
        CHECK_FALSE(journal.GrantLease(0x6000, 0xe000, 1));
        CHECK(journal.TryPublishLeased(1, LeasedByte(*b), commit).status == PublishStatus::LeaseUnavailable);
    }
    CHECK(committed == 1);
    CHECK(journal.GrantLease(0x4000, 0xc000, 2));
    CHECK(journal.GrantLease(0x6000, 0xe000, 2));
}

TEST_CASE("Deferred upload reserves at its backing read and skips images without mapped segments",
          "[deferred_writes][deferred_consumer]") {
    Journal journal{1, 2};
    ConsumerMemory memory{{{0x8000, 0x4000}}};
    const auto lease = journal.GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    u8 backing{}, applied{}, uploaded{};
    bool drained{};
    const auto apply = [&](Cut cut) {
        drained = true;
        REQUIRE(journal.DrainThrough(cut, [&](const Write& write) {
            applied = write.bytes[0]; return true;
        }) == DrainResult::Complete);
    };
    {
        ConsumerScope unmapped{&journal, std::span<const TransitionRange>{}, apply};
        CHECK_FALSE(drained);
    }
    auto queued_upload = [&] {
        std::vector<TransitionRange> ranges;
        AddConsumerDeviceRange(ranges, memory, 0x8000, 4096);
        ConsumerScope scope{&journal, ranges, apply};
        uploaded = backing;
        auto concurrent = std::async(std::launch::async, [&] {
            return journal.TryPublishLeased(0, LeasedByte(*lease, 9), [&]() noexcept {
                backing = 9; return true;
            });
        });
        CHECK(concurrent.get().status == PublishStatus::LeaseUnavailable);
    };
    REQUIRE(journal.TryPublishLeased(0, LeasedByte(*lease, 7), [&]() noexcept {
        backing = 7; return true;
    }).sequence == 1);
    queued_upload();
    CHECK(drained);
    CHECK(applied == 7);
    CHECK(uploaded == 7);
    CHECK(backing == 7);
}

TEST_CASE("Lease grants report the rejection observed under the publication gate",
          "[deferred_writes][deferred_admission]") {
    Journal journal{1, 2};
    LeaseGrantStatus status = LeaseGrantStatus::Granted;
    const auto attempt = [&] { return journal.GrantLease(0x4000, 0x8000, 1, &status); };
    SECTION("invalid operands") {
        CHECK_FALSE(journal.GrantLease(0x4001, 0x8000, 1, &status));
        CHECK(status == LeaseGrantStatus::Invalid);
    }
    SECTION("existing and closing are distinct") {
        const auto lease = attempt();
        REQUIRE(lease);
        CHECK(status == LeaseGrantStatus::Granted);
        CHECK_FALSE(attempt());
        CHECK(status == LeaseGrantStatus::Existing);
        REQUIRE(journal.CloseLease(*lease));
        CHECK_FALSE(attempt());
        CHECK(status == LeaseGrantStatus::Closing);
    }
    SECTION("transition without an existing lease") {
        auto reservation = ReservePage(journal);
        REQUIRE(reservation);
        CHECK_FALSE(attempt());
        CHECK(status == LeaseGrantStatus::Transition);
    }
    SECTION("fresh identity") {
        const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
        auto identity = journal.TryReserveFreshIdentity(ranges);
        REQUIRE(identity);
        CHECK_FALSE(attempt());
        CHECK(status == LeaseGrantStatus::FreshIdentity);
    }
    SECTION("captured or legacy writer") {
        const std::array ranges{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
        auto writer = journal.BeginWriterPermission(ranges);
        REQUIRE(writer);
        CHECK_FALSE(attempt());
        CHECK(status == LeaseGrantStatus::WriterExcluded);
    }
}

TEST_CASE("Queued lease grant observes a suspension begun after its CPU preflight",
          "[deferred_writes][deferred_admission][deferred_suspension]") {
    Journal journal{1, 2};
    REQUIRE(journal.GetSuspensionState() == SuspensionState::Running);
    std::promise<void> queued, execute;
    auto queued_future = queued.get_future();
    auto execute_future = execute.get_future();
    auto owner_callback = std::async(std::launch::async, [&] {
        queued.set_value();
        execute_future.wait();
        LeaseGrantStatus status{};
        const auto lease = journal.GrantLease(0x4000, 0x8000, 1, &status);
        return std::pair{lease.has_value(), status};
    });
    queued_future.wait();
    auto suspension = journal.BeginSuspension();
    execute.set_value();
    const auto [granted, status] = owner_callback.get();
    CHECK_FALSE(granted);
    CHECK(status == LeaseGrantStatus::Suspended);
    REQUIRE(suspension.TryPrepare([](Cut) { return true; }) == SuspensionResult::Ready);
    LeaseGrantStatus ready_status{};
    CHECK_FALSE(journal.GrantLease(0x4000, 0x8000, 1, &ready_status));
    CHECK(ready_status == LeaseGrantStatus::Suspended);
    suspension.Reset();
    CHECK(journal.GrantLease(0x4000, 0x8000, 1, &ready_status));
    CHECK(ready_status == LeaseGrantStatus::Granted);
}

TEST_CASE("Live writer origins count unique physical pages and release their own coverage",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const std::array first_ranges{TransitionRange{TransitionSpace::Physical, 0x4008, 0x2000}};
    const std::array overlap_ranges{TransitionRange{TransitionSpace::Physical, 0x5000, 0x1000}};
    const std::array device_ranges{TransitionRange{TransitionSpace::Device, 0x8000, 0x1000}};
    auto first = journal.BeginWriterPermission(first_ranges, WriterOrigin::NcePersistent);
    auto overlap = journal.BeginWriterPermission(overlap_ranges, WriterOrigin::NcePersistent);
    auto kernel = journal.BeginWriterPermission(overlap_ranges, WriterOrigin::KernelJitPersistent);
    auto legacy = journal.BeginWriterPermission(overlap_ranges, WriterOrigin::LegacyPersistent);
    auto temporary = journal.BeginWriterPermission(device_ranges);
    REQUIRE(first);
    REQUIRE(overlap);
    REQUIRE(kernel);
    REQUIRE(legacy);
    REQUIRE(temporary);
    const auto state = journal.GetRetentionSnapshot();
    const auto& nce = state.writers_by_origin[static_cast<size_t>(WriterOrigin::NcePersistent)];
    CHECK(state.writer_permissions == 5);
    CHECK(nce.permissions == 2);
    CHECK(nce.physical_pages_known == 3);
    CHECK(nce.device_only_permissions == 0);
    CHECK(state.writers_by_origin[static_cast<size_t>(WriterOrigin::KernelJitPersistent)].physical_pages_known == 1);
    CHECK(state.writers_by_origin[static_cast<size_t>(WriterOrigin::LegacyPersistent)].physical_pages_known == 1);
    const auto& transient = state.writers_by_origin[static_cast<size_t>(WriterOrigin::Temporary)];
    CHECK(transient.permissions == 1);
    CHECK(transient.physical_pages_known == 0);
    CHECK(transient.device_only_permissions == 1);
    first.reset();
    const auto reduced = journal.GetRetentionSnapshot();
    CHECK(reduced.writers_by_origin[static_cast<size_t>(WriterOrigin::NcePersistent)].physical_pages_known == 1);
    CHECK(reduced.writers_by_origin[static_cast<size_t>(WriterOrigin::KernelJitPersistent)].physical_pages_known == 1);
    overlap.reset();
    CHECK(journal.GetRetentionSnapshot().writers_by_origin[static_cast<size_t>(WriterOrigin::NcePersistent)].permissions == 0);
}
TEST_CASE("Persistent writer page index deduplicates ranges and retains overlapping owners",
          "[deferred_writes][deferred_writers]") {
    Journal journal{1, 4};
    const std::array ranges{
        TransitionRange{TransitionSpace::Physical, 0x4fff, 2},
        TransitionRange{TransitionSpace::Physical, 0x4000, 0x1000},
        TransitionRange{TransitionSpace::Device, 0x8000, 0x1000}};
    const std::array overlap_ranges{TransitionRange{TransitionSpace::Physical, 0x5000, 1}};
    auto first = journal.BeginWriterPermission(ranges, WriterOrigin::NcePersistent);
    auto second = journal.BeginWriterPermission(overlap_ranges, WriterOrigin::LegacyPersistent);
    REQUIRE(first);
    REQUIRE(second);
    CHECK(journal.HasPersistentWriterExclusion(0x4000, 0));
    CHECK(journal.HasPersistentWriterExclusion(0x5000, 0));
    CHECK(journal.HasPersistentWriterExclusion(0, 0x8000));
    first.reset();
    CHECK_FALSE(journal.HasPersistentWriterExclusion(0x4000, 0));
    CHECK(journal.HasPersistentWriterExclusion(0x5000, 0));
    CHECK_FALSE(journal.HasPersistentWriterExclusion(0, 0x8000));
    second.reset();
    CHECK_FALSE(journal.HasPersistentWriterExclusion(0x5000, 0));
}
TEST_CASE("Live writer inspection intersects every origin and preserves first NCE reason",
          "[deferred_writes][deferred_writers]") {
    namespace Profile = Common::GpuTransitionProfile;
    const bool previous = Profile::enabled.exchange(true);
    {
        Journal journal{1, 4};
        const std::array physical{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
        const std::array device{TransitionRange{TransitionSpace::Device, 0x8000, u64{1} << 34}};
        auto nce = journal.BeginWriterPermission(physical, WriterOrigin::NcePersistent,
                                                Profile::DeferredFallbackReason::UnsupportedInstruction);
        auto kernel = journal.BeginWriterPermission(physical, WriterOrigin::KernelJitPersistent);
        auto legacy = journal.BeginWriterPermission(physical, WriterOrigin::LegacyPersistent);
        auto temporary = journal.BeginWriterPermission(device);
        CHECK(nce->FirstFallbackReason() == Profile::DeferredFallbackReason::UnsupportedInstruction);
        CHECK(journal.InspectWriterExclusions(0x4000).origin_mask == 14);
        const auto all = journal.InspectWriterExclusions(0x4000, 0x8000);
        CHECK(all.complete);
        CHECK(all.origin_mask == 15);
        CHECK(all.first_reason_mask == (u64{1} << static_cast<size_t>(Profile::DeferredFallbackReason::UnsupportedInstruction)));
        CHECK(journal.InspectWriterExclusions(0x7000, 0x100008000).origin_mask == 1);
        temporary.reset();
        CHECK(journal.InspectWriterExclusions(0x7000, 0x8000).origin_mask == 0);
        nce.reset();
        CHECK(journal.InspectWriterExclusions(0x4000).origin_mask == 12);
        CHECK(journal.InspectWriterExclusions(0x4000).first_reason_mask == 0);
        Profile::enabled.store(false);
    }
    Profile::enabled.store(previous);
}

TEST_CASE("Live writer diagnostic index handles overlapping ranges unknown reason and profile toggles",
          "[deferred_writes][deferred_writers]") {
    namespace Profile = Common::GpuTransitionProfile;
    const bool previous = Profile::enabled.exchange(true);
    {
        Journal journal{1, 4};
        const std::array overlap{
            TransitionRange{TransitionSpace::Physical, 0x4000, 8192},
            TransitionRange{TransitionSpace::Physical, 0x5000, 4096}};
        auto first = journal.BeginWriterPermission(overlap, WriterOrigin::NcePersistent);
        const std::array single{TransitionRange{TransitionSpace::Physical, 0x5000, 4096}};
        auto second = journal.BeginWriterPermission(single, WriterOrigin::NcePersistent,
                                                   Profile::DeferredFallbackReason::PageCrossing);
        CHECK(first->FirstFallbackReason() == Profile::DeferredFallbackReason::Count);
        CHECK((journal.InspectWriterExclusions(0x5000).first_reason_mask &
               (u64{1} << static_cast<size_t>(Profile::DeferredFallbackReason::Count))) != 0);
        second.reset();
        CHECK(journal.InspectWriterExclusions(0x5000).origin_mask == 2);
        first.reset();
        CHECK(journal.InspectWriterExclusions(0x5000).origin_mask == 0);
        Profile::enabled.store(false);
        auto unindexed = journal.BeginWriterPermission(single);
        Profile::enabled.store(true);
        CHECK_FALSE(journal.InspectWriterExclusions(0x5000).complete);
        unindexed.reset();
        CHECK(journal.InspectWriterExclusions(0x5000).complete);
    }
    Profile::enabled.store(previous);
}

TEST_CASE("Diagnostic writer inspection and snapshot skip a busy publication gate",
          "[deferred_writes][deferred_writers]") {
    namespace Profile = Common::GpuTransitionProfile;
    Journal journal{1, 4};
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    PublishStatus writer_status{PublishStatus::Rejected};
    std::jthread writer([&] {
        writer_status = journal.TryPublish(0, ByteWrite(7), [&]() noexcept {
            entered.set_value();
            released.wait();
            return true;
        }).status;
    });
    entered.get_future().wait();
    Profile::TakeDiagnosticDrops();
    auto query = std::async(std::launch::async, [&] {
        return std::pair{journal.InspectWriterExclusions(0x4000), journal.GetRetentionSnapshot()};
    });
    const auto status = query.wait_for(std::chrono::seconds{1});
    release.set_value();
    writer.join();
    CHECK(status == std::future_status::ready);
    CHECK(writer_status == PublishStatus::Published);
    const auto result = query.get();
    CHECK_FALSE(result.first.complete);
    CHECK_FALSE(result.second.diagnostic_complete);
    const auto dropped = Profile::TakeDiagnosticDrops();
    CHECK(dropped[static_cast<size_t>(Profile::DiagnosticSite::JournalInspection)] == 1);
    CHECK(dropped[static_cast<size_t>(Profile::DiagnosticSite::JournalSnapshot)] == 1);
    CHECK(journal.GetRetentionSnapshot().diagnostic_complete);
}