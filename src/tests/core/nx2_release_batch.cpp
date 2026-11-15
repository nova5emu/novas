// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_release_batch.h"

TEST_CASE("Sparse backing claims reject replacement regions and smaller block lists", "[nx2][sparse]") {
    const Service::Nx2::SparseBackingClaim claim{0x80000000, 7, 12};
    CHECK(claim.Matches(12, 8));
    CHECK_FALSE(claim.Matches(13, 8));
    CHECK_FALSE(claim.Matches(13, 64));
    CHECK_FALSE(claim.Matches(12, 7));
    CHECK_FALSE(claim.Matches(12, 0));
}

TEST_CASE("Sparse release completion follows every removal in its batch", "[nx2][sparse]") {
    Service::Nx2::ReleaseBatch batch;
    std::vector<std::function<void()>> queue;
    auto first = batch.Completion();
    unsigned removed{};
    queue.emplace_back([&] { ++removed; });
    auto second = batch.Completion();
    queue.emplace_back([&] { ++removed; });
    REQUIRE(first == second);
    CHECK_FALSE(first->load());
    batch.Seal([&](auto completion) { queue.emplace_back(std::move(completion)); });
    REQUIRE(queue.size() == 3);
    queue[0]();
    CHECK(removed == 1);
    CHECK_FALSE(first->load());
    queue[1]();
    CHECK(removed == 2);
    CHECK_FALSE(second->load());
    queue[2]();
    CHECK(first->load());
    CHECK(second->load());
}

TEST_CASE("Empty and abandoned sparse batches cannot authorize a release", "[nx2][sparse]") {
    Service::Nx2::ReleaseBatch empty;
    unsigned queued{};
    empty.Seal([&](auto) { ++queued; });
    CHECK(queued == 0);
    std::shared_ptr<std::atomic<bool>> abandoned;
    {
        Service::Nx2::ReleaseBatch batch;
        abandoned = batch.Completion();
    }
    CHECK_FALSE(abandoned->load());
    bool touched{};
    CHECK_FALSE(Service::Nx2::TryReleaseBacking(abandoned, [&] { touched = true; return true; },
                                               [&] { touched = true; }));
    CHECK_FALSE(touched);
}

TEST_CASE("Sparse backing survives an unmap failure and releases once on retry", "[nx2][sparse]") {
    struct Pending {
        std::unique_ptr<unsigned> backing;
        std::shared_ptr<std::atomic<bool>> done;
    };
    std::vector<Pending> pending;
    auto done = std::make_shared<std::atomic<bool>>(false);
    pending.push_back({std::make_unique<unsigned>(17), done});
    unsigned attempts{}, released{}, live_bytes{4096};
    bool can_unmap{};
    const auto collect = [&] {
        std::erase_if(pending, [&](Pending& entry) {
            return Service::Nx2::TryReleaseBacking(entry.done, [&] {
                ++attempts;
                return can_unmap;
            }, [&] {
                ++released;
                live_bytes -= 4096;
            });
        });
    };
    collect();
    CHECK(attempts == 0);
    done->store(true, std::memory_order_release);
    collect();
    REQUIRE(pending.size() == 1);
    CHECK(*pending.front().backing == 17);
    CHECK(released == 0);
    CHECK(live_bytes == 4096);
    can_unmap = true;
    collect();
    CHECK(pending.empty());
    CHECK(attempts == 2);
    CHECK(released == 1);
    CHECK(live_bytes == 0);
    collect();
    CHECK(released == 1);
}
