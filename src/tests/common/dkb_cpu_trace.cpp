// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <future>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "common/dkb_cpu_trace.h"
#include "common/dkb_frame_trace.h"
#include "common/guest_thread_profile.h"
#include "common/guest_handoff_profile.h"

namespace {
using namespace Common::DkbCpuTrace;
using namespace std::chrono_literals;

std::optional<u64> BurnCpu() {
    const auto start = Common::CurrentThreadCpuTimeNs();
    if (!start) {
        return std::nullopt;
    }
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    auto end = start;
    while (end && *end - *start < 1'000'000 && std::chrono::steady_clock::now() < deadline) {
        end = Common::CurrentThreadCpuTimeNs();
    }
    return end ? std::optional<u64>{*end - *start} : std::nullopt;
}
}

TEST_CASE("Thread CPU clocks report monotonic execution counters", "[common][dkb_cpu_trace]") {
    Common::ThreadCpuClock disabled{false};
    REQUIRE_FALSE(disabled.Read());
    const auto start = Common::CurrentThreadCpuTimeNs();
#if defined(_WIN32) || defined(__linux__)
    REQUIRE(start);
    Common::ThreadCpuClock captured;
    const auto captured_time = captured.Read();
    REQUIRE(captured_time);
    REQUIRE(*captured_time >= *start);
    const auto busy = BurnCpu();
    REQUIRE(busy);
    REQUIRE(*busy > 0);
#else
    REQUIRE_FALSE(start);
    Common::ThreadCpuClock unsupported;
    REQUIRE_FALSE(unsupported.Read());
#endif
}

TEST_CASE("Thread CPU deltas reject missing or replaced counters", "[common][dkb_cpu_trace]") {
    const ThreadSample before{.generation = 3, .active = true, .cpu_ns = 100};
    ThreadSample after{.generation = 3, .active = true, .cpu_ns = 145};
    REQUIRE(CpuTimeDelta(before, after) == 45U);
    REQUIRE(CpuTimeDelta(before, before) == 0U);
    after.generation = 4;
    REQUIRE_FALSE(CpuTimeDelta(before, after));
    after = before;
    after.cpu_ns = 99;
    REQUIRE_FALSE(CpuTimeDelta(before, after));
    after.cpu_ns.reset();
    REQUIRE_FALSE(CpuTimeDelta(before, after));
    REQUIRE_FALSE(CpuTimeDelta(after, before));
    after = before;
    after.active = false;
    REQUIRE_FALSE(CpuTimeDelta(before, after));
    REQUIRE_FALSE(CpuTimeDelta(after, before));
}

TEST_CASE("Thread CPU registrations are opt-in and track lifetime", "[common][dkb_cpu_trace]") {
    constexpr auto role = Role::GuestSingleCore;
    constexpr auto index = static_cast<size_t>(role);
    const bool enabled = Common::DkbFrameTrace::Enabled();
    ThreadSample first;
    {
        Registration registration{role};
        first = ReadSnapshot().threads[index];
        REQUIRE(first.active == enabled);
        {
            Registration duplicate{role};
            REQUIRE(ReadSnapshot().threads[index].generation == first.generation);
        }
        REQUIRE(ReadSnapshot().threads[index].active == enabled);
    }
    REQUIRE_FALSE(ReadSnapshot().threads[index].active);
    REQUIRE_FALSE(ReadSnapshot().threads[index].cpu_ns);
    REQUIRE_FALSE(CpuTimeDelta(first, ReadSnapshot().threads[index]));
    {
        Registration replacement{role};
        const auto second = ReadSnapshot().threads[index];
        if (enabled) {
            REQUIRE(second.generation > first.generation);
        }
        REQUIRE_FALSE(CpuTimeDelta(first, second));
    }
}

TEST_CASE("Thread CPU snapshots read the registered thread not the caller", "[common][dkb_cpu_trace]") {
    constexpr auto index = static_cast<size_t>(Role::GpuCommand);
    std::promise<std::optional<u64>> ready;
    std::promise<std::optional<u64>> ended;
    std::promise<void> stop;
    auto stop_future = stop.get_future();
    std::jthread worker{[&] {
        Registration registration{Role::GpuCommand};
        ready.set_value(Common::CurrentThreadCpuTimeNs());
        stop_future.wait();
        ended.set_value(Common::CurrentThreadCpuTimeNs());
    }};
    const auto owner_before = ready.get_future().get();
    const auto before = ReadSnapshot();
    const auto busy = BurnCpu();
    const auto after = ReadSnapshot();
    stop.set_value();
    worker.join();
    const auto owner_after = ended.get_future().get();
    REQUIRE(after.wall_time >= before.wall_time);
    REQUIRE_FALSE(ReadSnapshot().threads[index].active);
#if defined(_WIN32) || defined(__linux__)
    if (Common::DkbFrameTrace::Enabled()) {
        const auto delta = CpuTimeDelta(before.threads[index], after.threads[index]);
        REQUIRE(busy);
        REQUIRE(*busy > 0);
        REQUIRE(delta);
        REQUIRE(owner_before);
        REQUIRE(owner_after);
        REQUIRE(*before.threads[index].cpu_ns >= *owner_before);
        REQUIRE(*after.threads[index].cpu_ns <= *owner_after);
    }
#else
    REQUIRE_FALSE(busy);
    REQUIRE_FALSE(owner_before);
    REQUIRE_FALSE(owner_after);
    REQUIRE_FALSE(after.threads[index].cpu_ns);
#endif
}

TEST_CASE("Thread CPU snapshots serialize against unregister", "[common][dkb_cpu_trace]") {
    std::atomic<u32> unavailable{};
    std::promise<void> ready;
    std::jthread reader{[&](std::stop_token stop) {
        ready.set_value();
        while (!stop.stop_requested()) {
            const auto sample = ReadSnapshot().threads[static_cast<size_t>(Role::GpuBack)];
            if (sample.active && !sample.cpu_ns) {
                unavailable.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }};
    ready.get_future().wait();
    for (size_t i = 0; i < 100; ++i) {
        Registration registration{Role::GpuBack};
        std::this_thread::yield();
    }
    reader.request_stop();
    reader.join();
#if defined(_WIN32) || defined(__linux__)
    REQUIRE(unavailable.load() == 0);
#endif
    REQUIRE_FALSE(ReadSnapshot().threads[static_cast<size_t>(Role::GpuBack)].active);
}

TEST_CASE("Guest thread timelines charge open waits through each reporting boundary",
          "[common][guest_thread_profile]") {
    using Common::GuestThreadProfile::Timeline;
    Timeline timeline;
    timeline.cursor_ns = 100;
    timeline.state = 1;
    CHECK(timeline.Advance(150) == 50);
    CHECK(timeline.blocked_ns == 50);
    timeline.ClearWindow();
    CHECK(timeline.cursor_ns == 150);
    CHECK(timeline.Advance(140) == 0);
    CHECK(timeline.cursor_ns == 150);
    CHECK(timeline.Advance(230) == 80);
    CHECK(timeline.blocked_ns == 80);
    timeline.state = 2;
    CHECK(timeline.Advance(280) == 0);
    CHECK(timeline.runnable_ns == 50);
    timeline.scheduled = true;
    CHECK(timeline.Advance(400) == 0);
    CHECK(timeline.scheduled_wall_ns == 120);
    CHECK(timeline.blocked_ns == 80);
    CHECK(timeline.runnable_ns == 50);
}

TEST_CASE("Guest thread timelines distinguish dormant and terminated time from scheduled wall time",
          "[common][guest_thread_profile]") {
    using Common::GuestThreadProfile::Timeline;
    Timeline timeline;
    timeline.cursor_ns = 100;
    timeline.state = 4;
    CHECK(timeline.Advance(180) == 80);
    timeline.state = 3;
    CHECK(timeline.Advance(250) == 0);
    CHECK(timeline.blocked_ns == 80);
    timeline.state = 1;
    timeline.scheduled = true;
    timeline.Advance(300);
    CHECK(timeline.scheduled_wall_ns == 50);
    CHECK(timeline.blocked_ns == 80);
}

TEST_CASE("Guest thread CPU accounting preserves live baselines across windows and rejects clock gaps",
          "[common][guest_thread_profile]") {
    using Common::GuestThreadProfile::Timeline;
    Timeline timeline;
    timeline.cpu_baseline = 1000;
    CHECK(timeline.ChargeCpu(1100));
    CHECK(timeline.host_cpu_ns == 100);
    timeline.ClearWindow();
    CHECK(timeline.cpu_baseline == 1100);
    CHECK(timeline.ChargeCpu(1170));
    CHECK(timeline.host_cpu_ns == 70);
    CHECK_FALSE(timeline.ChargeCpu(std::nullopt));
    CHECK_FALSE(timeline.cpu_baseline);
    CHECK_FALSE(timeline.ChargeCpu(2000));
    CHECK(timeline.host_cpu_ns == 70);
    CHECK(timeline.ChargeCpu(2010));
    CHECK(timeline.host_cpu_ns == 80);
    CHECK_FALSE(timeline.ChargeCpu(10));
    CHECK(timeline.host_cpu_ns == 80);
    timeline.cpu_baseline = 7000;
    CHECK(timeline.ChargeCpu(7030));
    CHECK(timeline.host_cpu_ns == 110);
}

TEST_CASE("Authoritative role ownership survives an omitted scheduler registry update",
          "[common][guest_thread_profile]") {
    using namespace Common::GuestThreadProfile;
    RoleOwner owner;
    const u64 original = owner.Publish(64, 100, 1000, 3);
    Timeline stale;
    stale.cursor_ns = 100;
    stale.state = 1;
    stale.scheduled = true;
    stale.cpu_baseline = 1000;
    owner.Publish(0, 200, 1050, 3);
    owner.Publish(72, 300, 1100, 3);
    const auto current = owner.Read();
    REQUIRE(current);
    CHECK(current->tid == 72);
    CHECK(current->sequence > original);
    CHECK(current->wall_ns == 300);
    CHECK(current->clock_generation == 3);
    CHECK(current->cpu_ns == 1100);
    stale.ReconcileOwnership(current->wall_ns, 500, current->tid == 64);
    CHECK_FALSE(stale.scheduled);
    CHECK_FALSE(stale.wall_complete);
    CHECK_FALSE(stale.cpu_baseline);
    CHECK(stale.unknown_wall_ns == 200);
    CHECK(stale.Advance(500) == 200);
    CHECK(stale.scheduled_wall_ns == 0);
    CHECK(stale.blocked_ns == 200);
    stale.ClearWindow();
    stale.Advance(5500);
    CHECK(stale.scheduled_wall_ns == 0);
    CHECK(stale.blocked_ns == 5000);
    CHECK(stale.host_cpu_ns == 0);
}

TEST_CASE("Recovered owner gaps preserve uncertainty instead of inventing CPU or wall categories",
          "[common][guest_thread_profile]") {
    using namespace Common::GuestThreadProfile;
    Timeline timeline;
    timeline.cursor_ns = 100;
    timeline.state = 2;
    timeline.Advance(200);
    timeline.ReconcileOwnership(300, 500, true);
    CHECK(timeline.unknown_wall_ns == 200);
    CHECK(timeline.runnable_ns == 0);
    CHECK_FALSE(timeline.wall_complete);
    timeline.Advance(500);
    CHECK(timeline.scheduled_wall_ns == 200);
    CHECK_FALSE(timeline.ChargeCpu(9000));
    CHECK(timeline.host_cpu_ns == 0);
    timeline.ReconcileOwnership(250, 600, false);
    CHECK(timeline.cursor_ns == 500);
    CHECK(timeline.unknown_wall_ns == 400);
    CHECK(timeline.scheduled_wall_ns == 0);
    CHECK_FALSE(timeline.cpu_baseline);
}
TEST_CASE("Guest handoff timeline preserves first stages and rejects reversed time", "[common][guest_handoff_profile]") {
    using Common::GuestHandoffProfile::Timeline;
    Timeline t;
    t.pending = true;
    t.Stamp(0, 10);
    t.Stamp(1, 20);
    t.Stamp(1, 30);
    REQUIRE(t.ns[1] == 20);
    t.Stamp(6, 100);
    REQUIRE(t.Ordered());
    REQUIRE(t.ns[2] == 0);
    t.Stamp(3, 15);
    REQUIRE_FALSE(t.Ordered());
    t.Cancel();
    REQUIRE_FALSE(t.pending);
    REQUIRE(t.ns[6] == 0);
    t.Stamp(0, 100);
    t.Stamp(6, 200);
    t.invalid = true;
    REQUIRE_FALSE(t.Ordered());
}

TEST_CASE("Guest handoff does not mix core selections or aborted NCE entries", "[common][guest_handoff_profile]") {
    using Common::GuestHandoffProfile::Timeline;
    Timeline t;
    t.Stamp(0, 10);
    t.Stage(2, 1, 20);
    t.Stage(3, 1, 30);
    t.Stage(5, 1, 40);
    t.Stage(5, 1, 60);
    t.Stamp(6, 70);
    REQUIRE(t.ns[5] == 60);
    REQUIRE(t.Ordered());
    t.Stage(2, 2, 80);
    REQUIRE_FALSE(t.Ordered());
    t.Cancel();
    REQUIRE(t.chosen_core == ~0U);
    REQUIRE_FALSE(t.invalid);
    t.Stage(2, 1, 100);
    t.Stage(4, 2, 110);
    REQUIRE_FALSE(t.Ordered());
}

TEST_CASE("Guest handoff ring bounds loss and preserves order across wrap", "[common][guest_handoff_profile]") {
    Common::GuestHandoffProfile::CompletionRing<u64, 2> ring;
    u64 value = 0;
    REQUIRE_FALSE(ring.Pop(value));
    REQUIRE(ring.Push(1));
    REQUIRE(ring.Push(2));
    REQUIRE_FALSE(ring.Push(3));
    REQUIRE(ring.Available() == 2);
    REQUIRE(ring.Pop(value));
    REQUIRE(value == 1);
    REQUIRE(ring.Push(4));
    REQUIRE(ring.Pop(value));
    REQUIRE(value == 2);
    REQUIRE(ring.Pop(value));
    REQUIRE(value == 4);
    REQUIRE_FALSE(ring.Pop(value));
    ring.Push(5);
    ring.Reset();
    REQUIRE_FALSE(ring.Pop(value));
}

TEST_CASE("Guest handoff captured entry timestamp excludes later host processing", "[common][guest_handoff_profile]") {
    Common::GuestHandoffProfile::Timeline t;
    t.Stamp(0, 100);
    t.Stage(2, 1, 200);
    t.Stage(5, 1, 300);
    t.Stamp(6, 900);
    t.Complete(400);
    REQUIRE(t.ns[6] == 400);
    REQUIRE(t.ns[6] - t.ns[0] == 300);
    REQUIRE(t.Ordered());
    t.Stage(2, 2, 950);
    REQUIRE(t.migrated);
    REQUIRE_FALSE(t.Ordered());
    t.Cancel();
    REQUIRE_FALSE(t.migrated);
    t.Complete(0);
    REQUIRE_FALSE(t.Ordered());
}
