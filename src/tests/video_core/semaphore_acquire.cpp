// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "video_core/engines/puller.h"
#include "video_core/nx2.h"
#include "video_core/semaphore_acquire.h"

TEST_CASE("SEM_EXECUTE normalizes a distinct opcode domain and rejects unknown acquires",
          "[video_core][deferred_writes][semaphore_acquire]") {
    using namespace Tegra::SemaphoreAcquire;
    REQUIRE(NormalizeExecute(0) == Equal);
    REQUIRE(NormalizeExecute(2) == Gequal);
    REQUIRE(NormalizeExecute(3) == SignedDifferenceGequal);
    REQUIRE(NormalizeExecute(4) == MaskNonzero);
    REQUIRE(NormalizeExecute(5) == MaskZero);
    REQUIRE_FALSE(NormalizeExecute(1)); // RELEASE, handled separately by the puller.
    REQUIRE_FALSE(NormalizeExecute(6));
    REQUIRE_FALSE(NormalizeExecute(7));
    REQUIRE_FALSE(Matches(6, 0, 0));
}

TEST_CASE("SEM_EXECUTE polling and release latching agree including signed wrap",
          "[video_core][deferred_writes][semaphore_acquire]") {
    struct Case { u32 op, value, current; bool expected; };
    const std::array cases{
        Case{0, 7, 6, false}, Case{0, 7, 7, true},
        Case{2, 7, 6, false}, Case{2, 7, 8, true},
        Case{2, 0xffffffff, 0, false}, Case{3, 0xffffffff, 0, true},
        Case{3, 0, 0xffffffff, false}, Case{3, 0, 0x80000000, false},
        Case{3, 0x80000000, 0, false}, Case{3, 0, 0x7fffffff, true},
        Case{4, 0x10, 0x20, false}, Case{4, 0x10, 0x30, true},
        Case{5, 0x10, 0x30, false}, Case{5, 0x10, 0x20, true},
    };
    auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
    int owner{};
    SCOPE_EXIT { waiters.Remove(&owner); };
    for (const auto& item : cases) {
        Tegra::Engines::Puller::PendingAcquire acquire{
            .address = 0x1000,
            .value = item.value,
            .operation = *Tegra::SemaphoreAcquire::NormalizeExecute(item.op),
            .since = {},
            .reported = false,
        };
        REQUIRE(acquire.Matches(item.current) == item.expected);
        waiters.Add(&owner, acquire.address, acquire.value, acquire.operation);
        waiters.OnWrite(acquire.address + 4, item.current);
        REQUIRE_FALSE(waiters.Latched(&owner));
        waiters.OnWrite(acquire.address, item.current);
        REQUIRE(waiters.Latched(&owner) == item.expected);
        if (item.expected) {
            waiters.OnWrite(acquire.address, ~item.current);
            REQUIRE(waiters.Latched(&owner));
        }
        waiters.Remove(&owner);
    }
}
