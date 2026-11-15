// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <latch>
#include <thread>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "video_core/cpu_epoch_barrier.h"

namespace {
using namespace VideoCommon::CpuEpochs;
Epoch Open(PageBarrier& barrier) {
    const auto epoch = barrier.BeginOpen();
    REQUIRE(epoch);
    REQUIRE(barrier.CompleteOpen(*epoch));
    return *epoch;
}
bool PrepareCapture(PageBarrier& barrier, Epoch epoch) {
    return barrier.ConfirmReprotected(epoch) && barrier.ConfirmNativeQuiescent(epoch) &&
           barrier.TryClaimDrain(epoch) && barrier.ConfirmJournalDrained(epoch) &&
           barrier.TryClaimCapture(epoch);
}
} // namespace

TEST_CASE("CPU epoch barrier requires every closing proof and exclusive capture",
          "[video_core][cpu_epoch_barrier]") {
    for (u32 proofs = 0; proofs < 8; ++proofs) {
        PageBarrier barrier;
        const auto epoch = Open(barrier);
        REQUIRE(barrier.BeginClose(epoch, 0));
        REQUIRE_FALSE(barrier.CompleteClose(epoch, true));
        if (proofs & 1) {
            CHECK(barrier.ConfirmReprotected(epoch));
        }
        if (proofs & 2) {
            CHECK(barrier.ConfirmNativeQuiescent(epoch) == static_cast<bool>(proofs & 1));
        }
        if (proofs & 4) {
            const bool claimed = barrier.TryClaimDrain(epoch);
            CHECK(claimed == ((proofs & 3) == 3));
            CHECK(barrier.ConfirmJournalDrained(epoch) == claimed);
        }
        REQUIRE(barrier.TryClaimCapture(epoch) == (proofs == 7));
        REQUIRE_FALSE(barrier.TryClaimCapture(epoch));
        REQUIRE(barrier.Inspect().state == PageState::Closing);
    }
}

TEST_CASE("CPU epoch host writer blocks drain and capture until its actual commit ends",
          "[video_core][cpu_epoch_barrier]") {
    PageBarrier barrier;
    const auto epoch = Open(barrier);
    auto attempt = barrier.TryEnterWriter(epoch);
    REQUIRE(attempt.disposition == WriteDisposition::Granted);
    REQUIRE(attempt.writer);
    std::latch entered{1}, release{1};
    std::atomic<u32> value{};
    std::thread writer([guard = std::move(attempt.writer), &entered, &release, &value]() mutable {
        entered.count_down();
        release.wait();
        value.store(42, std::memory_order_relaxed);
        guard.Release();
    });
    entered.wait();
    CHECK(barrier.BeginClose(epoch, 0));
    CHECK(barrier.ConfirmReprotected(epoch));
    CHECK(barrier.ConfirmNativeQuiescent(epoch));
    CHECK(barrier.Inspect().writers == 1);
    CHECK_FALSE(barrier.TryClaimDrain(epoch));
    CHECK_FALSE(barrier.ConfirmJournalDrained(epoch));
    CHECK_FALSE(barrier.TryClaimCapture(epoch));
    CHECK(barrier.TryEnterWriter(epoch).disposition == WriteDisposition::Defer);
    release.count_down();
    writer.join();
    REQUIRE(barrier.Inspect().writers == 0);
    REQUIRE(barrier.TryClaimDrain(epoch));
    REQUIRE_FALSE(barrier.TryClaimDrain(epoch));
    REQUIRE(barrier.ConfirmJournalDrained(epoch));
    REQUIRE(barrier.TryClaimCapture(epoch));
    REQUIRE(value.load(std::memory_order_relaxed) == 42);
    REQUIRE(barrier.CompleteClose(epoch, true));
}

TEST_CASE("CPU epoch queued GPU continuation retains a ticket without blocking closure",
          "[video_core][cpu_epoch_barrier]") {
    PageBarrier barrier;
    const auto epoch = Open(barrier);
    std::latch queued{1}, callback{1};
    PageBarrier::Continuation ticket{};
    WriteDisposition result = WriteDisposition::Granted;
    std::thread waiting([&] {
        ticket = barrier.ObserveContinuation();
        queued.count_down();
        callback.wait();
        auto attempt = barrier.TryEnterWriter(ticket.epoch);
        result = attempt.disposition;
    });
    queued.wait();
    CHECK(ticket.disposition == WriteDisposition::Granted);
    CHECK(barrier.Inspect().writers == 0);
    CHECK(barrier.BeginClose(epoch, 0));
    const bool captured = PrepareCapture(barrier, epoch);
    CHECK(captured);
    callback.count_down();
    waiting.join();
    REQUIRE(result == WriteDisposition::Defer);
    REQUIRE(barrier.Inspect().writers == 0);
    REQUIRE(barrier.CompleteClose(epoch, true));
    REQUIRE(barrier.TryEnterWriter(ticket.epoch).disposition == WriteDisposition::GpuOwned);
}

TEST_CASE("CPU epoch native acknowledgements cannot satisfy a later epoch",
          "[video_core][cpu_epoch_barrier]") {
    PageBarrier barrier;
    const auto old = Open(barrier);
    REQUIRE(barrier.BeginClose(old, (1U << 0) | (1U << 2)));
    REQUIRE_FALSE(barrier.AcknowledgeNativeCore(old, 0));
    REQUIRE_FALSE(barrier.ConfirmNativeQuiescent(old));
    REQUIRE(barrier.ConfirmReprotected(old));
    REQUIRE_FALSE(barrier.AcknowledgeNativeCore(old, 32));
    REQUIRE(barrier.AcknowledgeNativeCore(old, 2));
    REQUIRE_FALSE(barrier.ConfirmNativeQuiescent(old));
    REQUIRE(barrier.AcknowledgeNativeCore(old, 0));
    REQUIRE(barrier.ConfirmNativeQuiescent(old));
    REQUIRE(barrier.TryClaimDrain(old));
    REQUIRE(barrier.ConfirmJournalDrained(old));
    REQUIRE(barrier.TryClaimCapture(old));
    REQUIRE(barrier.CompleteClose(old, true));
    const auto current = Open(barrier);
    REQUIRE(current.generation != old.generation);
    REQUIRE(barrier.TryEnterWriter(old).disposition == WriteDisposition::Stale);
    REQUIRE_FALSE(barrier.CompleteOpen(old));
    REQUIRE_FALSE(barrier.BeginClose(old, 0));
    REQUIRE_FALSE(barrier.ConfirmReprotected(old));
    REQUIRE_FALSE(barrier.AcknowledgeNativeCore(old, 0));
    REQUIRE_FALSE(barrier.ConfirmNativeQuiescent(old));
    REQUIRE_FALSE(barrier.TryClaimDrain(old));
    REQUIRE_FALSE(barrier.ConfirmJournalDrained(old));
    REQUIRE_FALSE(barrier.TryClaimCapture(old));
    REQUIRE_FALSE(barrier.CompleteClose(old, true));
    REQUIRE(barrier.BeginClose(current, 1));
    REQUIRE(barrier.ConfirmReprotected(current));
    REQUIRE_FALSE(barrier.AcknowledgeNativeCore(old, 0));
    REQUIRE_FALSE(barrier.ConfirmNativeQuiescent(current));
    REQUIRE(barrier.AcknowledgeNativeCore(current, 0));
    REQUIRE(barrier.ConfirmNativeQuiescent(current));
}

TEST_CASE("CPU epoch consumer close rejects opening while actor cancellation remains available",
          "[video_core][cpu_epoch_barrier]") {
    PageBarrier barrier;
    const auto epoch = barrier.BeginOpen();
    REQUIRE(epoch);
    REQUIRE_FALSE(barrier.BeginOpen());
    REQUIRE(barrier.ObserveContinuation().disposition == WriteDisposition::Defer);
    REQUIRE_FALSE(barrier.BeginClose(*epoch, 0));
    REQUIRE(barrier.CancelOpen(*epoch, 0));
    REQUIRE_FALSE(barrier.CompleteOpen(*epoch));
    REQUIRE(PrepareCapture(barrier, *epoch));
    REQUIRE(barrier.CompleteClose(*epoch, true));
}

TEST_CASE("CPU epoch failed upload retains a single claimed closing operation",
          "[video_core][cpu_epoch_barrier]") {
    PageBarrier barrier;
    const auto epoch = Open(barrier);
    REQUIRE(barrier.BeginClose(epoch, 0));
    REQUIRE(PrepareCapture(barrier, epoch));
    REQUIRE_FALSE(barrier.CompleteClose(epoch, false));
    REQUIRE(barrier.Inspect().state == PageState::Closing);
    REQUIRE(barrier.Inspect().capture_claimed);
    REQUIRE_FALSE(barrier.TryClaimCapture(epoch));
    REQUIRE_FALSE(barrier.BeginOpen());
    REQUIRE(barrier.TryEnterWriter(epoch).disposition == WriteDisposition::Defer);
    REQUIRE(barrier.CompleteClose(epoch, true));
    REQUIRE_FALSE(barrier.CompleteClose(epoch, true));
}

TEST_CASE("CPU epoch movable writer guard releases each admission exactly once",
          "[video_core][cpu_epoch_barrier]") {
    PageBarrier barrier;
    const auto epoch = Open(barrier);
    auto first = barrier.TryEnterWriter(epoch);
    auto second = barrier.TryEnterWriter(epoch);
    REQUIRE(first.writer);
    REQUIRE(second.writer);
    REQUIRE(barrier.Inspect().writers == 2);
    PageBarrier::Writer moved{std::move(first.writer)};
    REQUIRE_FALSE(first.writer);
    REQUIRE(moved);
    REQUIRE(barrier.Inspect().writers == 2);
    moved = std::move(second.writer);
    REQUIRE_FALSE(second.writer);
    REQUIRE(barrier.Inspect().writers == 1);
    moved.Release();
    moved.Release();
    REQUIRE(barrier.Inspect().writers == 0);
    REQUIRE(barrier.BeginClose(epoch, 0));
    REQUIRE(PrepareCapture(barrier, epoch));
}

TEST_CASE("CPU epoch close versus writer admission has bounded concurrent outcomes",
          "[video_core][cpu_epoch_barrier]") {
    for (u32 round = 0; round < 64; ++round) {
        PageBarrier barrier;
        const auto epoch = Open(barrier);
        std::barrier phase{3};
        WriteDisposition disposition = WriteDisposition::Stale;
        bool closed = false;
        std::thread writer([&] {
            phase.arrive_and_wait();
            auto attempt = barrier.TryEnterWriter(epoch);
            disposition = attempt.disposition;
            phase.arrive_and_wait();
            phase.arrive_and_wait();
        });
        std::thread closer([&] {
            phase.arrive_and_wait();
            closed = barrier.BeginClose(epoch, 0);
            phase.arrive_and_wait();
            phase.arrive_and_wait();
        });
        phase.arrive_and_wait();
        phase.arrive_and_wait();
        if (!closed) {
            closed = barrier.BeginClose(epoch, 0);
        }
        CHECK(closed);
        CHECK((disposition == WriteDisposition::Granted || disposition == WriteDisposition::Defer ||
               disposition == WriteDisposition::Contended));
        CHECK(barrier.Inspect().writers == (disposition == WriteDisposition::Granted ? 1U : 0U));
        CHECK(barrier.ConfirmReprotected(epoch));
        CHECK(barrier.ConfirmNativeQuiescent(epoch));
        if (disposition == WriteDisposition::Granted) {
            CHECK_FALSE(barrier.TryClaimDrain(epoch));
        }
        phase.arrive_and_wait();
        writer.join();
        closer.join();
        REQUIRE(barrier.Inspect().writers == 0);
        REQUIRE(barrier.TryClaimDrain(epoch));
        REQUIRE(barrier.ConfirmJournalDrained(epoch));
        REQUIRE(barrier.TryClaimCapture(epoch));
        REQUIRE(barrier.CompleteClose(epoch, true));
    }
}

TEST_CASE("CPU epoch native acknowledgements can be rescheduled after concurrent attempts",
          "[video_core][cpu_epoch_barrier]") {
    PageBarrier barrier;
    const auto epoch = Open(barrier);
    REQUIRE(barrier.BeginClose(epoch, 0xff));
    REQUIRE(barrier.ConfirmReprotected(epoch));
    std::barrier phase{9};
    std::array<bool, 8> acknowledged{};
    std::array<std::thread, 8> cores;
    for (u32 core = 0; core < cores.size(); ++core) {
        cores[core] = std::thread([&, core] {
            phase.arrive_and_wait();
            acknowledged[core] = barrier.AcknowledgeNativeCore(epoch, core);
        });
    }
    phase.arrive_and_wait();
    for (auto& core : cores) {
        core.join();
    }
    for (u32 core = 0; core < acknowledged.size(); ++core) {
        if (!acknowledged[core]) {
            REQUIRE(barrier.AcknowledgeNativeCore(epoch, core));
        }
    }
    REQUIRE(barrier.ConfirmNativeQuiescent(epoch));
    REQUIRE(barrier.TryClaimDrain(epoch));
    REQUIRE(barrier.ConfirmJournalDrained(epoch));
    REQUIRE(barrier.TryClaimCapture(epoch));
}

TEST_CASE("CPU epoch capture observes plain backing bytes after the last writer releases",
          "[video_core][cpu_epoch_barrier]") {
    PageBarrier barrier;
    const auto epoch = Open(barrier);
    auto attempt = barrier.TryEnterWriter(epoch);
    REQUIRE(attempt.writer);
    std::array<u8, 4096> page{};
    std::latch write_now{1};
    std::thread writer([guard = std::move(attempt.writer), &write_now, &page]() mutable {
        write_now.wait();
        page.fill(0x5a);
        guard.Release();
    });
    CHECK(barrier.BeginClose(epoch, 0));
    CHECK(barrier.ConfirmReprotected(epoch));
    CHECK(barrier.ConfirmNativeQuiescent(epoch));
    CHECK_FALSE(barrier.TryClaimDrain(epoch));
    write_now.count_down();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    bool drained = barrier.TryClaimDrain(epoch);
    while (!drained && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
        drained = barrier.TryClaimDrain(epoch);
    }
    const bool captured = drained && barrier.ConfirmJournalDrained(epoch) &&
                          barrier.TryClaimCapture(epoch);
    if (captured) {
        CHECK(std::all_of(page.begin(), page.end(), [](u8 value) { return value == 0x5a; }));
    }
    writer.join();
    REQUIRE(captured);
    REQUIRE(barrier.CompleteClose(epoch, true));
}
