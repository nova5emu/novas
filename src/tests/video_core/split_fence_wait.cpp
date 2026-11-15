// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "tests/core/system_test_context.h"
#include "video_core/deferred_write_journal.h"
#include "video_core/fence_manager.h"
#include "video_core/gpu.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace {

struct Cache {
    std::mutex mutex;
    bool has_work{true};
    std::atomic<u32> pops{};
    std::function<void()> on_pop;
    bool HasUncommittedFlushes() const { return has_work; }
    bool HasAsyncFlushWork() const { return has_work; }
    bool ShouldWaitAsyncFlushes() const { return has_work; }
    bool HasPendingFenceWork() const { return has_work; }
    void AccumulateFlushes() {}
    void CommitAsyncFlushes() {}
    void PopAsyncFlushes() {
        if (on_pop) {
            on_pop();
        }
        ++pops;
    }
};

struct Fence : VideoCommon::FenceBase {
    Fence(bool stubbed, u64 tick_) : FenceBase{stubbed}, tick{tick_} {}
    u64 tick;
};

template <bool Async>
struct Traits {
    using FenceType = std::shared_ptr<Fence>;
    using TextureCacheType = Cache;
    using BufferCacheType = Cache;
    using QueryCacheType = Cache;
    static constexpr bool HAS_ASYNC_CHECK = Async;
};

struct Submission {
    std::mutex mutex;
    std::condition_variable cv;
    u64 submitted{};
    std::atomic<bool> timed_out{};
    bool wrong_owner{};
    const std::thread::id owner = std::this_thread::get_id();
};

template <bool Async>
class TestFenceManager final : public VideoCommon::FenceManager<Traits<Async>> {
public:
    TestFenceManager(VideoCore::RasterizerInterface& rasterizer, Tegra::GPU& gpu, Cache& texture,
                 Cache& buffer, Cache& query, Submission& submission_)
        : VideoCommon::FenceManager<Traits<Async>>{rasterizer, gpu, texture, buffer, query},
          submission{submission_} {}

    u64 queued{};
    u32 submissions{};

protected:
    std::shared_ptr<Fence> CreateFence(bool stubbed) override {
        return std::make_shared<Fence>(stubbed, queued + 1);
    }

    void QueueFence(std::shared_ptr<Fence>& fence) override {
        queued = fence->tick;
    }

    bool IsFenceSignaled(std::shared_ptr<Fence>& fence) const override {
        std::scoped_lock lock{submission.mutex};
        return submission.submitted >= fence->tick;
    }

    void WaitFence(std::shared_ptr<Fence>& fence) override {
        std::unique_lock lock{submission.mutex};
        if (!submission.cv.wait_for(lock, std::chrono::seconds{2},
                                    [&] { return submission.submitted >= fence->tick; })) {
            submission.timed_out.store(true);
        }
    }

    void SubmitDeferredWork(const char*) override {
        std::scoped_lock lock{submission.mutex};
        submission.wrong_owner |= std::this_thread::get_id() != submission.owner;
        submission.submitted = queued;
        ++submissions;
        submission.cv.notify_all();
    }

private:
    Submission& submission;
};

using FenceManager = TestFenceManager<true>;

} // namespace

TEST_CASE("Delayed fence callbacks submit before the owner waits", "[video_core][split_fence_wait]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Cache texture, buffer, query;
    Submission submission;
    const auto previous = Settings::values.gpu_fence_behavior.GetValue();
    SCOPE_EXIT { Settings::values.gpu_fence_behavior.SetValue(previous); };
    Settings::values.gpu_fence_behavior.SetValue(Settings::GpuFenceBehavior::Accurate);
    std::vector<u32> callbacks;
    {
        FenceManager manager{rasterizer, gpu, texture, buffer, query, submission};
        for (u32 i = 0; i < 128; ++i) {
            manager.SyncOperation([&, i] { callbacks.push_back(i * 2); });
            manager.SignalFence([&, i] { callbacks.push_back(i * 2 + 1); });
            REQUIRE(manager.submissions == i + 1);
        }
        manager.WaitPendingFences(true);
        REQUIRE(manager.submissions == 130);
        REQUIRE_FALSE(submission.timed_out.load());
        REQUIRE_FALSE(submission.wrong_owner);
    }
    REQUIRE(callbacks.size() == 256);
    for (u32 i = 0; i < callbacks.size(); ++i) {
        REQUIRE(callbacks[i] == i);
    }
}

TEST_CASE("Immediate fence waits still submit outstanding work", "[video_core][split_fence_wait]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Cache texture, buffer, query;
    Submission submission;
    const auto previous = Settings::values.gpu_fence_behavior.GetValue();
    const auto previous_accuracy = Settings::values.current_gpu_accuracy;
    SCOPE_EXIT {
        Settings::values.gpu_fence_behavior.SetValue(previous);
        Settings::values.current_gpu_accuracy = previous_accuracy;
    };
    Settings::values.gpu_fence_behavior.SetValue(Settings::GpuFenceBehavior::Default);
    Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::Low;
    std::atomic<bool> callback{};
    {
        FenceManager manager{rasterizer, gpu, texture, buffer, query, submission};
        manager.SyncOperation([&] { callback.store(true); });
        manager.SignalFence([] {});
        REQUIRE(manager.submissions == 0);
        manager.WaitPendingFences(false);
        REQUIRE(manager.submissions == 0);
        manager.WaitPendingFences(true);
        REQUIRE(manager.submissions == 1);
        Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::High;
        manager.WaitPendingFences(true);
        REQUIRE(manager.submissions == 3);
        REQUIRE_FALSE(submission.timed_out.load());
        REQUIRE_FALSE(submission.wrong_owner);
    }
    REQUIRE(callback.load());
}

TEST_CASE("Deferred transitions wait for readback retirement at low accuracy",
          "[video_core][split_fence_wait][deferred_suspension]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Cache texture, buffer, query;
    Submission submission;
    const auto previous = Settings::values.gpu_fence_behavior.GetValue();
    const auto previous_accuracy = Settings::values.current_gpu_accuracy;
    SCOPE_EXIT {
        Settings::values.gpu_fence_behavior.SetValue(previous);
        Settings::values.current_gpu_accuracy = previous_accuracy;
    };
    Settings::values.gpu_fence_behavior.SetValue(Settings::GpuFenceBehavior::Default);
    Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::Low;

    const auto exercise = [&]<bool Async>() {
        using namespace VideoCommon::DeferredWrites;
        Journal journal{1, 1};
        auto readback = journal.BeginReadback();
        auto suspension = journal.BeginSuspension();
        buffer.on_pop = [&] { readback.reset(); };
        TestFenceManager<Async> manager{rasterizer, gpu, texture, buffer, query, submission};
        bool immediate_callback{};
        manager.SignalFence([&] { immediate_callback = true; });
        REQUIRE(immediate_callback);
        REQUIRE(manager.submissions == 0);
        REQUIRE(suspension.TryPrepare([](Cut) { return true; }) ==
                SuspensionResult::ReadbacksPending);

        manager.WaitPendingReadbacks();
        REQUIRE(manager.submissions == 1);
        REQUIRE(buffer.pops.load() == 2);
        REQUIRE_FALSE(readback);
        REQUIRE(suspension.TryPrepare([](Cut) { return true; }) == SuspensionResult::Ready);
        REQUIRE_FALSE(submission.timed_out.load());
        REQUIRE_FALSE(submission.wrong_owner);
        buffer.on_pop = {};
    };
    SECTION("Asynchronous fences submit on the scheduler owner before waiting") {
        exercise.template operator()<true>();
    }
    SECTION("Non-asynchronous fences release their cache pops on the owner") {
        exercise.template operator()<false>();
    }
}

TEST_CASE("Readback completion marker cannot be skipped as an empty fence",
          "[video_core][split_fence_wait][deferred_suspension]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Cache texture, buffer, query;
    texture.has_work = false;
    buffer.has_work = false;
    query.has_work = false;
    Submission submission;
    const auto previous = Settings::values.gpu_fence_behavior.GetValue();
    const auto previous_accuracy = Settings::values.current_gpu_accuracy;
    SCOPE_EXIT {
        Settings::values.gpu_fence_behavior.SetValue(previous);
        Settings::values.current_gpu_accuracy = previous_accuracy;
    };
    Settings::values.gpu_fence_behavior.SetValue(Settings::GpuFenceBehavior::Default);
    Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::Low;
    const auto exercise = [&]<bool Async>() {
        TestFenceManager<Async> manager{rasterizer, gpu, texture, buffer, query, submission};
        manager.SignalFence([] {});
        REQUIRE(manager.queued == 0);
        manager.WaitPendingReadbacks();
        REQUIRE(manager.queued == 1);
        REQUIRE(manager.submissions == 1);
        REQUIRE(buffer.pops.load() == 0);
        REQUIRE(query.pops.load() == 1);
        REQUIRE_FALSE(submission.timed_out.load());
    };
    SECTION("Asynchronous") {
        exercise.template operator()<true>();
    }
    SECTION("Non-asynchronous") {
        exercise.template operator()<false>();
    }
}

TEST_CASE("Idle promotion releases operations without another guest fence",
          "[video_core][split_fence_wait]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Cache texture, buffer, query;
    Submission submission;
    const auto previous = Settings::values.gpu_fence_behavior.GetValue();
    const auto previous_accuracy = Settings::values.current_gpu_accuracy;
    SCOPE_EXIT {
        Settings::values.gpu_fence_behavior.SetValue(previous);
        Settings::values.current_gpu_accuracy = previous_accuracy;
    };
    Settings::values.gpu_fence_behavior.SetValue(Settings::GpuFenceBehavior::Default);
    Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::Low;
    SECTION("Only a CPU callback remains") {
        texture.has_work = buffer.has_work = query.has_work = false;
    }
    SECTION("Downloads also require a submitted fence") {}

    std::mutex mutex;
    std::condition_variable cv;
    u32 completed{};
    FenceManager manager{rasterizer, gpu, texture, buffer, query, submission};
    REQUIRE_FALSE(manager.CommitPendingOperations());
    REQUIRE(manager.queued == 0);
    REQUIRE(manager.submissions == 0);

    // A producer can append again after the owner has moved the previous batch.
    REQUIRE(manager.SyncOperation([&] {
        manager.SyncOperation([&] {
            std::scoped_lock lock{mutex};
            completed = 2;
            cv.notify_all();
        });
        std::scoped_lock lock{mutex};
        completed = 1;
        cv.notify_all();
    }));
    REQUIRE(manager.GetFenceProgress().uncommitted_operations == 1);
    REQUIRE(manager.CommitPendingOperations());
    {
        std::unique_lock lock{mutex};
        REQUIRE(cv.wait_for(lock, std::chrono::seconds{2}, [&] { return completed == 1; }));
    }
    REQUIRE(manager.HasPendingOperations());
    REQUIRE(manager.GetFenceProgress().uncommitted_operations == 1);
    REQUIRE(manager.CommitPendingOperations());
    {
        std::unique_lock lock{mutex};
        REQUIRE(cv.wait_for(lock, std::chrono::seconds{2}, [&] { return completed == 2; }));
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    const auto retired = [&] {
        const auto progress = manager.GetFenceProgress();
        return progress.completed_operations == 2 && progress.uncommitted_operations == 0 &&
               progress.queued_fences == 0 && progress.queued_operations == 0 &&
               progress.retiring_fences == 0 && progress.retiring_operations == 0;
    };
    while (!retired() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const auto progress = manager.GetFenceProgress();
    REQUIRE(progress.supported);
    REQUIRE(progress.uncommitted_operations == 0);
    REQUIRE(progress.queued_fences == 0);
    REQUIRE(progress.queued_operations == 0);
    REQUIRE(progress.retiring_fences == 0);
    REQUIRE(progress.retiring_operations == 0);
    REQUIRE(progress.idle_promotions == 2);
    REQUIRE(progress.completed_operations == 2);
    REQUIRE_FALSE(manager.CommitPendingOperations());
    REQUIRE(manager.queued == 2);
    REQUIRE(manager.submissions == 2);
    REQUIRE_FALSE(submission.timed_out.load());
    REQUIRE_FALSE(submission.wrong_owner);
}

TEST_CASE("Unsupported rasterizers cannot report deferred readbacks resolved",
          "[video_core][split_fence_wait][deferred_suspension]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    REQUIRE_FALSE(rasterizer.ResolveDeferredWriteReadbacks());
}
