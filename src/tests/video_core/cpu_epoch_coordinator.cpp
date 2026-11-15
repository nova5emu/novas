// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <future>
#include <latch>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "video_core/cpu_epoch_coordinator.h"
#include "video_core/sync_request_queue.h"

namespace {
using namespace VideoCommon::CpuEpochs;
using Platform = Common::CpuEpochMemoryBarrier;
struct Fixture {
    bool supported{true}, fail_run{}, valid{true}, can_reprotect{true};
    unsigned runs{}, protects{}, drains{}, captures{}, uploads{}, progress{};
    u64 generation{1};
    CloseResult drain_result{CloseResult::Complete};
    std::shared_ptr<int> owner{std::make_shared<int>(0)};
    Coordinator coordinator;
    Coordinator::Dispatcher dispatch_override;
    static Platform::CallResult Call(void* context, Platform::Command command) noexcept {
        auto& self = *static_cast<Fixture*>(context);
        if (command == Platform::Command::Query) {
            return {self.supported ? 24L : 0L, 0};
        }
        if (command == Platform::Command::PrivateExpedited && ++self.runs > 1 && self.fail_run) {
            return {-1, EIO};
        }
        return {0, 0};
    }
    explicit Fixture(bool available = true)
        : supported{available}, coordinator{16 * 4096, true, Call, this} {
        coordinator.Bind(
            {[this](const Identity& id) { return valid && id.generation == generation; },
             [this](const Identity&) {
                 ++protects;
                 return can_reprotect;
             },
             [this](const Identity&) {
                 ++drains;
                 return drain_result;
             },
             [this](const Identity&, std::span<u8> data) {
                 ++captures;
                 std::ranges::fill(data, 0x5a);
                 return true;
             },
             [this](const Identity&, std::span<const u8> data) {
                 ++uploads;
                 return std::ranges::all_of(data, [](u8 byte) { return byte == 0x5a; });
             }},
            [this](CloseReason reason) {
                return dispatch_override ? dispatch_override(reason)
                                         : coordinator.CloseAllOnOwner(reason);
            },
            [this] { ++progress; });
    }
    Identity Id(PAddr physical = 4096) const {
        return {.cpu_owner = owner.get(),
                .device_owner = owner.get(),
                .address = 0x100000 + physical,
                .physical = physical,
                .device = 0x200000 + physical,
                .generation = generation,
                .lifetime = owner};
    }
    std::shared_ptr<Coordinator::Page> Open(PAddr physical = 4096) {
        auto page = coordinator.BeginOpening(Id(physical));
        REQUIRE(page);
        REQUIRE(coordinator.CompleteOpening(page, [] { return true; }));
        return page;
    }
};
} // namespace

TEST_CASE("CPU epoch coordinator rejects permanent exclusions and unavailable platforms",
          "[cpu_epoch]") {
    Fixture f;
    f.coordinator.Exclude(4096, 4096, Exclusion::Uniform);
    f.coordinator.Exclude(4096, 4096, Exclusion::Texture);
    CHECK(f.coordinator.Exclusions(4096) == 5);
    CHECK_FALSE(f.coordinator.BeginOpening(f.Id()));
    Fixture unavailable{false};
    CHECK_FALSE(unavailable.coordinator.Available());
    CHECK_FALSE(unavailable.coordinator.BeginOpening(unavailable.Id()));
}

TEST_CASE("CPU epoch mutation closes before action and prevents opening", "[cpu_epoch]") {
    Fixture f;
    f.Open();
    {
        auto mutation = f.coordinator.BeforeMutation(CloseReason::Mapping);
        CHECK(f.uploads == 1);
        CHECK(f.coordinator.ActivePages() == 0);
        CHECK_FALSE(f.coordinator.BeginOpening(f.Id(8192)));
    }
    CHECK(f.coordinator.BeginOpening(f.Id(8192)));
}

TEST_CASE("CPU epoch close waits for actual writer but not queued tickets", "[cpu_epoch]") {
    Fixture f;
    auto page = f.Open();
    const auto ticket = page->barrier.ObserveContinuation();
    auto writer = f.coordinator.TryEnterWriter(page, ticket.epoch);
    REQUIRE(writer.disposition == WriteDisposition::Granted);
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Pending);
    CHECK(f.captures == 0);
    CHECK(f.uploads == 0);
    CHECK(f.coordinator.TryEnterWriter(page, page->epoch).disposition == WriteDisposition::Defer);
    writer.writer.Release();
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
    CHECK(f.captures == 1);
    CHECK(f.uploads == 1);
    CHECK(f.coordinator.ActivePages() == 0);
    CHECK(f.coordinator.TryEnterWriter(page, ticket.epoch).disposition == WriteDisposition::Stale);
}

TEST_CASE("CPU epoch physical reuse cannot revive stale shared page ticket", "[cpu_epoch]") {
    Fixture f;
    auto previous = f.Open();
    auto ticket = previous->barrier.ObserveContinuation();
    REQUIRE(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
    ++f.generation;
    auto current = f.Open();
    CHECK(f.coordinator.FindPage(4097) == current);
    CHECK(f.coordinator.TryEnterWriter(previous, ticket.epoch).disposition ==
          WriteDisposition::Stale);
    CHECK(f.coordinator.TryEnterWriter(current, current->epoch).disposition ==
          WriteDisposition::Granted);
}

TEST_CASE("CPU epoch stale mapping fails closed before mutation", "[cpu_epoch]") {
    Fixture f;
    auto page = f.Open();
    ++f.generation;
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Mapping) == CloseResult::Failed);
    CHECK(page->barrier.Inspect().state == PageState::Closing);
    CHECK(f.protects == 1);
    CHECK(f.captures == 0);
    CHECK(f.uploads == 0);
    bool action = false;
    CHECK_THROWS_AS(([&] {
                        auto mutation = f.coordinator.BeforeMutation(CloseReason::Mapping);
                        action = true;
                    }()),
                    std::runtime_error);
    CHECK_FALSE(action);
}

TEST_CASE("CPU epoch native permissions stay revoked through closing", "[cpu_epoch]") {
    Fixture f;
    auto page = f.Open();
    bool writable = false;
    auto inspect = [&] {
        f.coordinator.WithAliasProtection(f.owner.get(), f.Id().address, 4096,
                                          [&](bool allowed) { writable = allowed; });
    };
    inspect();
    CHECK(writable);
    f.drain_result = CloseResult::Pending;
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Pending);
    inspect();
    CHECK_FALSE(writable);
    f.drain_result = CloseResult::Complete;
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
    inspect();
    CHECK(writable);
}

TEST_CASE("CPU epoch platform execution failure never captures", "[cpu_epoch]") {
    Fixture f;
    auto page = f.Open();
    f.fail_run = true;
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Failed);
    CHECK(page->barrier.Inspect().state == PageState::Closing);
    CHECK_FALSE(f.coordinator.Available());
    CHECK(f.captures == 0);
    CHECK(f.uploads == 0);
    CHECK_FALSE(f.coordinator.BeginOpening(f.Id(8192)));
}

TEST_CASE("CPU epoch drain permits captured exclusions without recursive closure", "[cpu_epoch]") {
    Fixture f;
    bool reentrant_mutation_rejected = false;
    f.coordinator.Bind({[](const Identity&) { return true; }, [](const Identity&) { return true; },
                        [&](const Identity& id) {
                            f.coordinator.Exclude(id.physical, 4096, Exclusion::CapturedWriter);
                            try {
                                auto mutation = f.coordinator.BeforeMutation(CloseReason::Mapping);
                            } catch (const std::runtime_error&) {
                                reentrant_mutation_rejected = true;
                            }
                            return CloseResult::Complete;
                        },
                        [](const Identity&, std::span<u8>) { return true; },
                        [](const Identity&, std::span<const u8>) { return true; }},
                       [&](CloseReason reason) { return f.coordinator.CloseAllOnOwner(reason); });
    f.Open();
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
    CHECK(reentrant_mutation_rejected);
    CHECK(f.coordinator.Exclusions(4096) == static_cast<u8>(Exclusion::CapturedWriter));
    CHECK_FALSE(f.coordinator.BeginOpening(f.Id()));
    // A later, unrelated closure must not inherit the nested drain context.
    f.Open(8192);
    CHECK_NOTHROW(f.coordinator.BeforeMutation(CloseReason::Mapping));
}

TEST_CASE("CPU epoch concurrent host writer releases before capture", "[cpu_epoch]") {
    Fixture f;
    auto page = f.Open();
    std::latch admitted{1}, release{1};
    bool granted = false;
    std::thread writer{[&] {
        auto attempt = f.coordinator.TryEnterWriter(page, page->epoch);
        granted = attempt.disposition == WriteDisposition::Granted;
        admitted.count_down();
        release.wait();
    }};
    admitted.wait();
    const auto while_writing = f.coordinator.CloseAllOnOwner(CloseReason::Consumer);
    const unsigned premature_captures = f.captures;
    release.count_down();
    writer.join();
    CHECK(granted);
    CHECK(while_writing == CloseResult::Pending);
    CHECK(premature_captures == 0);
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
    CHECK(f.captures == 1);
}

TEST_CASE("CPU epoch pending close retains queue fence until writer progress", "[cpu_epoch]") {
    using Queue = VideoCommon::SyncRequestQueue;
    Fixture f;
    auto page = f.Open();
    auto writer = f.coordinator.TryEnterWriter(page, page->epoch);
    REQUIRE(writer.disposition == WriteDisposition::Granted);
    Queue queue;
    const auto fence = queue.Push(
        [&] {
            return f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Pending
                       ? Queue::Status::Pending
                       : Queue::Status::Complete;
        },
        Queue::Ordering::Independent);
    auto request = queue.Pop();
    REQUIRE(request);
    auto status = request->action();
    queue.Resolve(std::move(*request), status);
    CHECK_FALSE(queue.IsComplete(fence));
    CHECK(queue.CompletedThrough() < fence);
    const auto before_release = f.progress;
    writer.writer.Release();
    CHECK(f.progress > before_release);
    request = queue.Pop();
    REQUIRE(request);
    status = request->action();
    queue.Resolve(std::move(*request), status);
    CHECK(queue.IsComplete(fence));
    CHECK(queue.CompletedThrough() == fence);
    CHECK(f.captures == 1);
}

TEST_CASE("CPU epoch failed protection preflight never grants native writes", "[cpu_epoch]") {
    Fixture f;
    f.can_reprotect = false;
    auto page = f.coordinator.BeginOpening(f.Id());
    REQUIRE(page);
    bool granted = false;
    CHECK_FALSE(f.coordinator.CompleteOpening(page, [&] {
        granted = true;
        return true;
    }));
    CHECK_FALSE(granted);
    CHECK(f.coordinator.ActivePages() == 0);
    CHECK(f.captures == 0);
    CHECK(f.uploads == 0);
    CHECK_NOTHROW(f.coordinator.BeforeMutation(CloseReason::Mapping));
}
TEST_CASE("CPU epoch opening materializes before RW and preserves native changes on close",
          "[cpu_epoch]") {
    Fixture f;
    std::array<u8, 4096> backing{}, uploaded{};
    std::vector<unsigned> sequence;
    bool writable = false, baseline_valid = false;
    std::shared_ptr<Coordinator::Page> page;
    f.coordinator.Bind({[](const Identity&) { return true; },
                        [&](const Identity&) {
                            writable = false;
                            sequence.push_back(1);
                            return true;
                        },
                        [](const Identity&) { return CloseResult::Complete; },
                        [&](const Identity&, std::span<u8> data) {
                            sequence.push_back(3);
                            std::ranges::copy(backing, data.begin());
                            return true;
                        },
                        [&](const Identity&, std::span<const u8> data) {
                            std::ranges::copy(data, uploaded.begin());
                            return !writable;
                        }},
                       [&](CloseReason reason) { return f.coordinator.CloseAllOnOwner(reason); });
    u64 downloaded{};
    const bool opened = f.coordinator.PrepareAndOpen(
        f.Id(),
        [&] {
            sequence.push_back(2);
            page = f.coordinator.FindPage(4096);
            backing.fill(0x3a);
            downloaded = backing.size();
            return !writable;
        },
        [&] {
            sequence.push_back(4);
            baseline_valid =
                page && page->baseline_captured &&
                std::ranges::all_of(page->opening_snapshot, [](u8 value) { return value == 0x3a; });
            writable = true;
            return baseline_valid;
        },
        downloaded);
    REQUIRE(opened);
    REQUIRE(page);
    CHECK(baseline_valid);
    CHECK(downloaded == 4096);
    CHECK(writable);
    const auto materialized = std::ranges::find(sequence, 2);
    CHECK(std::ranges::find(sequence, 1) < materialized);
    CHECK(materialized < std::ranges::find(sequence, 3));
    CHECK(std::ranges::find(sequence, 3) < std::ranges::find(sequence, 4));
    backing[7] = 0xcc;
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
    CHECK_FALSE(writable);
    CHECK(uploaded[7] == 0xcc);
    CHECK(uploaded[8] == 0x3a);
    CHECK(page->opening_snapshot[7] == 0x3a);
}

TEST_CASE("CPU epoch failed materialization cancels pin without granting RW", "[cpu_epoch]") {
    Fixture f;
    bool grant_called = false;
    u64 downloaded{};
    CHECK_FALSE(f.coordinator.PrepareAndOpen(
        f.Id(), [] { return false; },
        [&] {
            grant_called = true;
            return true;
        },
        downloaded));
    CHECK_FALSE(grant_called);
    CHECK(f.captures == 0);
    CHECK(f.coordinator.ActivePages() == 0);
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
}

TEST_CASE("CPU epoch mapping mutation during materialization prevents RW", "[cpu_epoch]") {
    Fixture f;
    std::latch start_mutation{1}, mutation_registered{1}, finish_dispatch{1};
    bool mutation_performed = false, grant_called = false;
    f.dispatch_override = [&](CloseReason reason) {
        mutation_registered.count_down();
        finish_dispatch.wait();
        return f.coordinator.CloseAllOnOwner(reason);
    };
    std::thread mutation{[&] {
        start_mutation.wait();
        auto guard = f.coordinator.BeforeMutation(CloseReason::Mapping);
        mutation_performed = true;
        ++f.generation;
    }};
    u64 downloaded{};
    const bool opened = f.coordinator.PrepareAndOpen(
        f.Id(),
        [&] {
            start_mutation.count_down();
            mutation_registered.wait();
            return true;
        },
        [&] {
            grant_called = true;
            return true;
        },
        downloaded);
    finish_dispatch.count_down();
    mutation.join();
    CHECK_FALSE(opened);
    CHECK_FALSE(grant_called);
    CHECK(mutation_performed);
    CHECK(f.coordinator.ActivePages() == 0);
}

TEST_CASE("CPU epoch acquires writer after publishing Opening and before materialization",
          "[cpu_epoch]") {
    Fixture f;
    bool acquired = false, opening_visible = false, materialized = false, granted = false;
    u64 downloaded{};
    const bool opened = f.coordinator.PrepareAndOpen(
        f.Id(),
        [&] {
            materialized = acquired && opening_visible;
            return materialized;
        },
        [&] {
            granted = materialized;
            return granted;
        },
        downloaded,
        [&](Identity& identity) {
            const auto pinned = f.coordinator.FindPage(identity.physical);
            opening_visible = pinned && &pinned->identity == &identity &&
                              pinned->barrier.Inspect().state == PageState::Opening;
            acquired = true;
            return true;
        });
    CHECK(opened);
    CHECK(opening_visible);
    CHECK(materialized);
    CHECK(granted);
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
}

TEST_CASE("CPU epoch failed writer acquisition cancels Opening without RW", "[cpu_epoch]") {
    Fixture f;
    bool opening_visible = false, materialized = false, granted = false;
    u64 downloaded{};
    const bool opened = f.coordinator.PrepareAndOpen(
        f.Id(),
        [&] {
            materialized = true;
            return true;
        },
        [&] {
            granted = true;
            return true;
        },
        downloaded,
        [&](Identity& identity) {
            const auto pinned = f.coordinator.FindPage(identity.physical);
            opening_visible = pinned && pinned->barrier.Inspect().state == PageState::Opening;
            return false;
        });
    CHECK_FALSE(opened);
    CHECK(opening_visible);
    CHECK_FALSE(materialized);
    CHECK_FALSE(granted);
    CHECK(f.protects == 0);
    CHECK(f.captures == 0);
    CHECK(f.coordinator.ActivePages() == 0);
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
}

TEST_CASE("CPU epoch disjoint readback exclusion does not request a closure", "[cpu_epoch]") {
    Fixture f;
    bool dispatched = false;
    f.dispatch_override = [&](CloseReason reason) {
        dispatched = true;
        return f.coordinator.CloseAllOnOwner(reason);
    };
    const auto progress_before = f.progress;
    u64 downloaded = 0;
    CHECK(f.coordinator.PrepareAndOpen(
        f.Id(),
        [&] {
            f.coordinator.Exclude(8192, 4096, Exclusion::CapturedWriter);
            return true;
        },
        [] { return true; }, downloaded));
    CHECK_FALSE(dispatched);
    CHECK(f.coordinator.ActivePages() == 1);
    CHECK(f.progress == progress_before);
    CHECK(f.coordinator.Exclusions(8192) == static_cast<u8>(Exclusion::CapturedWriter));
    // A two-byte write crossing the boundary includes the open page's final byte.
    f.coordinator.Exclude(8191, 2, Exclusion::CapturedWriter);
    CHECK(dispatched);
    CHECK(f.coordinator.ActivePages() == 0);
}
TEST_CASE("CPU epoch pins identity before requiring the ready writer proof", "[cpu_epoch]") {
    Fixture f;
    bool ready = false, granted = false;
    f.coordinator.Bind({[&](const Identity&) { return ready; },
                        [](const Identity&) { return true; },
                        [](const Identity&) { return CloseResult::Complete; },
                        [](const Identity&, std::span<u8>) { return true; },
                        [](const Identity&, std::span<const u8>) { return true; },
                        [](const Identity&) { return true; }},
                       [&](CloseReason reason) { return f.coordinator.CloseAllOnOwner(reason); });
    u64 downloaded = 0;
    CHECK(f.coordinator.PrepareAndOpen(
        f.Id(), [&] { return ready; },
        [&] {
            granted = ready;
            return granted;
        },
        downloaded,
        [&](Identity& pinned) {
            CHECK_FALSE(ready);
            const auto page = f.coordinator.FindPage(pinned.physical);
            CHECK(page);
            if (!page)
                return false;
            CHECK(page->barrier.Inspect().state == PageState::Opening);
            ready = true;
            return true;
        }));
    CHECK(granted);
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
}
TEST_CASE("CPU epoch device hole veto preserves unrelated pages and owner scopes", "[cpu_epoch]") {
    Fixture f;
    f.Open();
    const DAddr hole = 0x400000;
    f.coordinator.ExcludeDevice(f.owner.get(), hole, 4096, Exclusion::OtherConsumer);
    CHECK(f.coordinator.ActivePages() == 1);
    CHECK(f.uploads == 0);
    CHECK(f.coordinator.Exclusions(8192) == 0);
    auto rejected = f.Id(8192);
    rejected.device = hole;
    CHECK_FALSE(f.coordinator.BeginOpening(rejected));
    f.Open(8192);
    CHECK(f.coordinator.CloseAllOnOwner(CloseReason::Consumer) == CloseResult::Complete);
    int other_owner{};
    auto independent = f.Id(12288);
    independent.device = hole;
    independent.device_owner = &other_owner;
    CHECK(f.coordinator.BeginOpening(independent));
}

TEST_CASE("CPU epoch mapped device veto propagates only intersecting physical pages",
          "[cpu_epoch]") {
    Fixture f;
    f.coordinator.ExcludeDevice(f.owner.get(), 0x400ff0, 32, Exclusion::Texture);
    f.coordinator.ExcludeDevice(f.owner.get(), 0x401010, 16, Exclusion::Texture);
    {
        auto mutation = f.coordinator.BeforeMutation(CloseReason::Mapping);
        f.coordinator.ExcludeMappedDevice(f.owner.get(), 0x400000, 0x5000, 8192);
    }
    CHECK(f.coordinator.Exclusions(0x4000) == 0);
    CHECK(f.coordinator.Exclusions(0x5000) == static_cast<u8>(Exclusion::Texture));
    CHECK(f.coordinator.Exclusions(0x6000) == static_cast<u8>(Exclusion::Texture));
    CHECK(f.coordinator.Exclusions(0x7000) == 0);
    int other_owner{};
    {
        auto mutation = f.coordinator.BeforeMutation(CloseReason::Mapping);
        f.coordinator.ExcludeMappedDevice(&other_owner, 0x400000, 0x7000, 8192);
    }
    CHECK(f.coordinator.Exclusions(0x7000) == 0);
    CHECK(f.coordinator.BeginOpening(f.Id(0x7000)));
}

TEST_CASE("CPU epoch device veto closes only overlapping live owner", "[cpu_epoch]") {
    Fixture f;
    f.Open();
    int other_owner{};
    f.coordinator.ExcludeDevice(&other_owner, f.Id().device, 4096, Exclusion::CapturedWriter);
    CHECK(f.coordinator.ActivePages() == 1);
    f.coordinator.ExcludeDevice(f.owner.get(), f.Id().device + 1, 1, Exclusion::CapturedWriter);
    CHECK(f.coordinator.ActivePages() == 0);
    CHECK(f.uploads == 1);
    CHECK_FALSE(f.coordinator.BeginOpening(f.Id()));
}

TEST_CASE("CPU epoch concurrent device veto rejects pending opening", "[cpu_epoch]") {
    Fixture f;
    bool grant_called{};
    u64 downloaded{};
    const bool opened = f.coordinator.PrepareAndOpen(
        f.Id(),
        [&] {
            try {
                f.coordinator.ExcludeDevice(f.owner.get(), f.Id().device, 1,
                                            Exclusion::OtherConsumer);
            } catch (const std::runtime_error&) {
                // The owner never waits for its own unfinished Opening.
            }
            return true;
        },
        [&] {
            grant_called = true;
            return true;
        },
        downloaded);
    CHECK_FALSE(opened);
    CHECK_FALSE(grant_called);
    CHECK(f.coordinator.ActivePages() == 0);
}

TEST_CASE("CPU epoch invalid device veto fails closed", "[cpu_epoch]") {
    Fixture f;
    SECTION("overflow") {
        f.coordinator.ExcludeDevice(f.owner.get(), ~u64{0} - 1, 2, Exclusion::OtherConsumer);
    }
    SECTION("missing owner") {
        f.coordinator.ExcludeDevice(nullptr, 0x400000, 4096, Exclusion::OtherConsumer);
    }
    CHECK(f.coordinator.Exclusions(4096) == 0xff);
    CHECK_FALSE(f.coordinator.BeginOpening(f.Id()));
}

TEST_CASE("CPU epoch exclusion statistics count physical pages once per reason", "[cpu_epoch]") {
    Fixture f;
    f.coordinator.Exclude(4096 + 32, 4096, Exclusion::Texture);
    f.coordinator.Exclude(4096, 8192, Exclusion::Texture);
    f.coordinator.Exclude(8192, 1, Exclusion::CapturedWriter);
    const auto stats = f.coordinator.GetExclusionStats();
    CHECK(stats.pages[2] == 2);
    CHECK(stats.pages[3] == 1);
    CHECK(stats.calls[2] == 1);
    CHECK(stats.bytes[2] == 4096);
    CHECK(stats.invalid_physical == 0);
    f.coordinator.Exclude(~u64{0}, 2, Exclusion::OtherConsumer);
    const auto invalid = f.coordinator.GetExclusionStats();
    CHECK(invalid.pages[4] == 0);
    CHECK(invalid.invalid_physical == 1);
}

TEST_CASE("CPU epoch completed device veto cache preserves owner reason and session",
          "[cpu_epoch]") {
    DeviceExclusionCache cache;
    int owner{}, other{};
    CHECK_FALSE(cache.Contains(1, &owner, 0x1000, 16, Exclusion::Texture));
    cache.Remember(1, &owner, 0x1000, 16, Exclusion::Texture);
    // Every byte in this page is ineligible after the first permanent veto.
    CHECK(cache.Contains(1, &owner, 0x1ff0, 16, Exclusion::Texture));
    CHECK_FALSE(cache.Contains(1, &owner, 0x1ff0, 17, Exclusion::Texture));
    CHECK_FALSE(cache.Contains(1, &other, 0x1000, 16, Exclusion::Texture));
    CHECK_FALSE(cache.Contains(2, &owner, 0x1000, 16, Exclusion::Texture));
    CHECK_FALSE(cache.Contains(1, &owner, 0x1000, 16, Exclusion::Uniform));
    cache.Remember(1, &owner, 0x1000, 16, Exclusion::Uniform);
    CHECK(cache.Contains(1, &owner, 0x1000, 16, Exclusion::Texture));
    CHECK(cache.Contains(1, &owner, 0x1000, 16, Exclusion::Uniform));
    cache.Remember(1, &owner, 0x41000, 16, Exclusion::Texture);
    CHECK_FALSE(cache.Contains(1, &owner, 0x1000, 16, Exclusion::Texture));
    cache.Remember(1, &owner, ~u64{0}, 2, Exclusion::Texture);
    CHECK_FALSE(cache.Contains(1, &owner, ~u64{0}, 2, Exclusion::Texture));
    CHECK_FALSE(cache.Contains(1, &owner, 0x41000, 0, Exclusion::Texture));
}

TEST_CASE("CPU epoch cached device veto remains authoritative across physical remapping",
          "[cpu_epoch]") {
    Fixture f;
    DeviceExclusionCache cache;
    const auto id = f.Id();
    f.coordinator.ExcludeDevice(id.device_owner, id.device, 1, Exclusion::Texture);
    cache.Remember(f.coordinator.InstanceId(), id.device_owner, id.device, 1, Exclusion::Texture);
    CHECK(cache.Contains(f.coordinator.InstanceId(), id.device_owner, id.device + 10, 1,
                         Exclusion::Texture));
    CHECK_FALSE(f.coordinator.BeginOpening(id));
    auto remapped = f.Id(8192);
    remapped.device = id.device;
    {
        auto mutation = f.coordinator.BeforeMutation(CloseReason::Mapping);
        f.coordinator.ExcludeMappedDevice(id.device_owner, id.device, remapped.physical, 4096);
    }
    CHECK_FALSE(f.coordinator.BeginOpening(remapped));
    CHECK(f.coordinator.Exclusions(remapped.physical) != 0);
}

TEST_CASE("CPU epoch diagnostic drops a sample while authoritative registry is locked", "[cpu_epoch]") {
    Fixture f;
    std::promise<std::optional<u8>> result;
    auto future = result.get_future();
    std::thread query;
    bool completed_under_lock = false;
    f.coordinator.WithAliasProtection(f.owner.get(), f.Id().address, 4096, [&](bool) {
        query = std::thread([&] { result.set_value(f.coordinator.TryExclusions(4096)); });
        completed_under_lock = future.wait_for(std::chrono::seconds{2}) == std::future_status::ready;
    });
    query.join();
    CHECK(completed_under_lock);
    CHECK_FALSE(future.get().has_value());
    CHECK(f.coordinator.TryExclusions(4096) == 0);
}
