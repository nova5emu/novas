// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <chrono>
#include <condition_variable>
#include <latch>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "video_core/sync_request_queue.h"

namespace {
using Queue = VideoCommon::SyncRequestQueue;
using Status = Queue::Status;
using Ordering = Queue::Ordering;
// The runtime owner also caps a round at its initial queue size.
void Round(Queue& queue) {
    const auto count = queue.PendingCount();
    for (size_t i = 0; i < count; ++i) {
        auto request = queue.Pop();
        REQUIRE(request);
        const auto status = queue.CanRun(*request) ? request->action() : Status::Pending;
        queue.Resolve(std::move(*request), status);
    }
}
} // namespace

TEST_CASE("GPU sync request queue completes normal ordered FIFO fences",
          "[video_core][sync_request_queue]") {
    Queue queue;
    std::vector<u32> order;
    REQUIRE(queue.CompletedThrough() == 0);
    REQUIRE(queue.IsComplete(0));
    REQUIRE_FALSE(queue.IsComplete(1));
    REQUIRE_FALSE(queue.Pop());
    for (u32 i = 1; i <= 4; ++i) {
        REQUIRE(queue.Push(
                    [&, i] {
                        order.push_back(i);
                        return Status::Complete;
                    },
                    Ordering::AfterPrior) == i);
    }
    REQUIRE(queue.PendingCount() == 4);
    Round(queue);
    REQUIRE(order == std::vector<u32>{1, 2, 3, 4});
    REQUIRE(queue.CompletedThrough() == 4);
    REQUIRE(queue.PendingCount() == 0);
    REQUIRE_FALSE(queue.Pop());
    for (u64 fence = 1; fence <= 4; ++fence) {
        REQUIRE(queue.IsComplete(fence));
    }
    REQUIRE_FALSE(queue.IsComplete(5));
}

TEST_CASE("Pending GPU sync requests retain their own fence across bounded retries",
          "[video_core][sync_request_queue]") {
    Queue queue;
    u32 pending_calls = 0, independent_calls = 0, ordered_calls = 0;
    const auto pending =
        queue.Push([&] { return ++pending_calls <= 2 ? Status::Pending : Status::Complete; },
                   Ordering::Independent);
    const auto independent = queue.Push(
        [&] {
            ++independent_calls;
            return Status::Complete;
        },
        Ordering::Independent);
    const auto ordered = queue.Push(
        [&] {
            ++ordered_calls;
            return Status::Complete;
        },
        Ordering::AfterPrior);
    Round(queue);
    REQUIRE(pending_calls == 1);
    REQUIRE(independent_calls == 1);
    REQUIRE(ordered_calls == 0);
    REQUIRE_FALSE(queue.IsComplete(pending));
    REQUIRE(queue.IsComplete(independent));
    REQUIRE_FALSE(queue.IsComplete(ordered));
    REQUIRE(queue.CompletedThrough() == 0);
    REQUIRE(queue.PendingCount() == 2);
    Round(queue);
    REQUIRE(pending_calls == 2);
    REQUIRE(ordered_calls == 0);
    REQUIRE_FALSE(queue.IsComplete(pending));
    REQUIRE(queue.CompletedThrough() == 0);
    Round(queue);
    REQUIRE(pending_calls == 3);
    REQUIRE(independent_calls == 1);
    REQUIRE(ordered_calls == 1);
    REQUIRE(queue.IsComplete(pending));
    REQUIRE(queue.IsComplete(ordered));
    REQUIRE(queue.CompletedThrough() == ordered);
    REQUIRE(queue.PendingCount() == 0);
}

TEST_CASE("GPU ordered idle request cannot overtake an incomplete prefix",
          "[video_core][sync_request_queue]") {
    Queue queue;
    bool ready = false, idle = false, independent = false;
    const auto first = queue.Push([&] { return ready ? Status::Complete : Status::Pending; },
                                  Ordering::Independent);
    const auto wait_idle = queue.Push(
        [&] {
            idle = true;
            return Status::Complete;
        },
        Ordering::AfterPrior);
    const auto later = queue.Push(
        [&] {
            independent = true;
            return Status::Complete;
        },
        Ordering::Independent);
    Round(queue);
    REQUIRE(independent);
    REQUIRE_FALSE(idle);
    REQUIRE(queue.IsComplete(later));
    REQUIRE_FALSE(queue.IsComplete(wait_idle));
    REQUIRE_FALSE(queue.IsComplete(first));
    REQUIRE(queue.CompletedThrough() == 0);
    ready = true;
    Round(queue);
    REQUIRE(idle);
    REQUIRE(queue.IsComplete(wait_idle));
    REQUIRE(queue.CompletedThrough() == later);
}

TEST_CASE("GPU completed intervals bridge holes without advancing through missing fences",
          "[video_core][sync_request_queue]") {
    Queue queue;
    std::vector<Queue::Request> requests;
    for (u32 i = 0; i < 9; ++i) {
        queue.Push([] { return Status::Complete; }, Ordering::Independent);
    }
    while (auto request = queue.Pop()) {
        requests.push_back(std::move(*request));
    }
    for (u32 fence : {3, 5, 4, 7, 9, 8, 6}) {
        queue.Resolve(std::move(requests[fence - 1]), Status::Complete);
        REQUIRE(queue.IsComplete(fence));
        REQUIRE_FALSE(queue.IsComplete(1));
        REQUIRE_FALSE(queue.IsComplete(2));
        REQUIRE(queue.CompletedThrough() == 0);
    }
    for (u64 fence = 3; fence <= 9; ++fence) {
        REQUIRE(queue.IsComplete(fence));
    }
    queue.Resolve(std::move(requests[0]), Status::Complete);
    REQUIRE(queue.CompletedThrough() == 1);
    REQUIRE_FALSE(queue.IsComplete(2));
    queue.Resolve(std::move(requests[1]), Status::Complete);
    REQUIRE(queue.CompletedThrough() == 9);
    REQUIRE_FALSE(queue.IsComplete(10));
}

TEST_CASE("GPU out-of-order completions match a per-fence reference model",
          "[video_core][sync_request_queue]") {
    Queue queue;
    constexpr u64 count = 32;
    std::vector<Queue::Request> requests;
    std::vector<bool> complete(count + 1, false);
    complete[0] = true;
    for (u64 i = 0; i < count; ++i) {
        queue.Push([] { return Status::Complete; }, Ordering::Independent);
    }
    while (auto request = queue.Pop()) {
        requests.push_back(std::move(*request));
    }
    std::vector<u64> sequence;
    for (u64 i = 0; i < count / 2; ++i) {
        sequence.push_back(count - 1 - i * 2);
    }
    for (u64 fence = 2; fence <= count; fence += 2) {
        sequence.push_back(fence);
    }
    for (const auto fence : sequence) {
        queue.Resolve(std::move(requests[fence - 1]), Status::Complete);
        complete[fence] = true;
        u64 prefix = 0;
        while (prefix < count && complete[prefix + 1]) {
            ++prefix;
        }
        CHECK(queue.CompletedThrough() == prefix);
        for (u64 i = 0; i <= count; ++i) {
            CHECK(queue.IsComplete(i) == complete[i]);
        }
    }
    REQUIRE(queue.CompletedThrough() == count);
}

TEST_CASE("GPU callback outside request mutex permits concurrent enqueue and notification",
          "[video_core][sync_request_queue]") {
    Queue queue;
    std::mutex request_mutex;
    std::condition_variable request_cv;
    std::latch callback_entered{1}, enqueued{1};
    bool notified = false;
    u64 second = 0;
    const auto first = queue.Push(
        [&] {
            callback_entered.count_down();
            enqueued.wait();
            return Status::Complete;
        },
        Ordering::AfterPrior);
    std::optional<Queue::Request> request;
    {
        std::scoped_lock lock{request_mutex};
        request = queue.Pop();
    }
    REQUIRE(request);
    std::thread producer([&] {
        callback_entered.wait();
        {
            std::scoped_lock lock{request_mutex};
            second = queue.Push([] { return Status::Complete; }, Ordering::Independent);
            notified = true;
        }
        request_cv.notify_one();
        enqueued.count_down();
    });
    // No failing assertion before the blocked producer is released.
    const auto status = request->action();
    {
        std::unique_lock lock{request_mutex};
        request_cv.wait(lock, [&] { return notified; });
        queue.Resolve(std::move(*request), status);
    }
    producer.join();
    REQUIRE(second == 2);
    REQUIRE(queue.IsComplete(first));
    REQUIRE_FALSE(queue.IsComplete(second));
    REQUIRE(queue.CompletedThrough() == first);
    REQUIRE(queue.PendingCount() == 1);
    Round(queue);
    REQUIRE(queue.IsComplete(second));
    REQUIRE(queue.CompletedThrough() == second);
}

TEST_CASE("Concurrent GPU ticks run the next callback while the first waits for it",
          "[video_core][sync_request_queue]") {
    Queue queue;
    std::mutex mutex;
    std::condition_variable cv;
    std::latch first_entered{1};
    bool second_finished = false, first_observed_completion = false;
    bool second_allowed = false, second_completed_before_first = false;
    u64 second = 0;
    const auto first = queue.Push(
        [&] {
            std::unique_lock lock{mutex};
            first_entered.count_down();
            first_observed_completion = cv.wait_for(lock, std::chrono::seconds{2}, [&] {
                return second_finished && queue.IsComplete(second);
            });
            return Status::Complete;
        },
        Ordering::AfterPrior);
    second = queue.Push(
        [&] {
            std::scoped_lock lock{mutex};
            second_finished = true;
            return Status::Complete;
        },
        Ordering::AfterPrior);
    std::thread tick([&] {
        std::optional<Queue::Request> request;
        {
            std::scoped_lock lock{mutex};
            request = queue.Pop();
        }
        const auto status = request->action();
        std::scoped_lock lock{mutex};
        queue.Resolve(std::move(*request), status);
    });
    first_entered.wait();
    std::optional<Queue::Request> request;
    {
        std::scoped_lock lock{mutex};
        request = queue.Pop();
        second_allowed = queue.CanRun(*request);
    }
    const auto status = second_allowed ? request->action() : Status::Pending;
    {
        std::scoped_lock lock{mutex};
        queue.Resolve(std::move(*request), status);
        second_completed_before_first = queue.IsComplete(second) && !queue.IsComplete(first);
    }
    cv.notify_all();
    tick.join();
    REQUIRE(second_allowed);
    REQUIRE(first_observed_completion);
    REQUIRE(second_completed_before_first);
    REQUIRE(queue.CompletedThrough() == second);
    REQUIRE(queue.PendingCount() == 0);
}

TEST_CASE("Nested GPU tick can complete a callback needed by its caller",
          "[video_core][sync_request_queue]") {
    Queue queue;
    bool nested_completed = false;
    queue.Push(
        [&] {
            const auto nested = queue.Push(
                [&] {
                    nested_completed = true;
                    return Status::Complete;
                },
                Ordering::AfterPrior);
            Round(queue);
            REQUIRE(queue.IsComplete(nested));
            REQUIRE(queue.CompletedThrough() == 0);
            return Status::Complete;
        },
        Ordering::AfterPrior);
    Round(queue);
    REQUIRE(nested_completed);
    REQUIRE(queue.CompletedThrough() == 2);
}

TEST_CASE("A yielded GPU request remains an ordering barrier during its retry",
          "[video_core][sync_request_queue]") {
    Queue queue;
    queue.Push([] { return Status::Complete; }, Ordering::Independent);
    auto retry = queue.Pop();
    queue.Resolve(std::move(*retry), Status::Pending);
    retry = queue.Pop();
    const auto later = queue.Push([] { return Status::Complete; }, Ordering::AfterPrior);
    auto request = queue.Pop();
    REQUIRE(queue.CanRun(*retry));
    REQUIRE_FALSE(queue.CanRun(*request));
    queue.Resolve(std::move(*retry), Status::Complete);
    REQUIRE(queue.CanRun(*request));
    queue.Resolve(std::move(*request), Status::Complete);
    REQUIRE(queue.IsComplete(later));
}
