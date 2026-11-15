// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/deferred_transition.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_page_buffer.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "tests/core/system_test_context.h"

TEST_CASE("Queued mapping restores its ready context for nested work and resource destruction",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    using namespace VideoCommon::DeferredWrites;
    ScopedSystemTestContext settings;
    Core::System system;
    auto journal = std::make_shared<Journal>(1, 4);
    auto token = BeginSharedSuspension(journal);
    bool prepared{};
    REQUIRE(token->TryPrepareIdle(prepared) == SuspensionResult::Ready);
    std::function<void()> queued;
    std::function<void()> nested;
    bool destroyed_ready{}, called_ready{}, worker_initially_empty{};
    {
        Core::DeferredTransitionScope scope{system, token};
        auto resource = std::shared_ptr<int>{new int{}, [&](int* pointer) {
            const auto current = Core::DeferredTransitionScope::Current(system);
            destroyed_ready = current && current->IsReady();
            delete pointer;
        }};
        queued = Core::RetainDeferredTransition(system, {}, [&, resource] {
            const auto current = Core::DeferredTransitionScope::Current(system);
            called_ready = current && current->IsReady();
            nested = Core::RetainDeferredTransition(system, {}, [] {});
        });
    }
    token.reset();
    CHECK_FALSE(Core::DeferredTransitionScope::Current(system));
    std::thread consumer{[&] {
        worker_initially_empty = !Core::DeferredTransitionScope::Current(system);
        queued();
    }};
    consumer.join();
    CHECK(worker_initially_empty);
    CHECK(called_ready);
    CHECK(destroyed_ready);
    CHECK(journal->GetSuspensionState() == SuspensionState::SuspendedReady);
    nested();
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
    CHECK_FALSE(Core::DeferredTransitionScope::Current(system));
}

TEST_CASE("Mapping callback exception restores TLS and releases its ready participant",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    using namespace VideoCommon::DeferredWrites;
    ScopedSystemTestContext settings;
    Core::System system;
    auto journal = std::make_shared<Journal>(1, 2);
    auto token = BeginSharedSuspension(journal);
    bool prepared{};
    REQUIRE(token->TryPrepareIdle(prepared) == SuspensionResult::Ready);
    auto callback = Core::RetainDeferredTransition(system, token, [] {
        throw std::runtime_error{"mapping failed"};
    });
    token.reset();
    CHECK_THROWS_AS(callback(), std::runtime_error);
    CHECK_FALSE(Core::DeferredTransitionScope::Current(system));
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
}

TEST_CASE("Cancelling a mapping callback destroys its resources inside a ready context",
          "[deferred_writes][deferred_suspension][deferred_boundaries]") {
    using namespace VideoCommon::DeferredWrites;
    ScopedSystemTestContext settings;
    Core::System system;
    auto journal = std::make_shared<Journal>(1, 2);
    auto token = BeginSharedSuspension(journal);
    bool prepared{};
    REQUIRE(token->TryPrepareIdle(prepared) == SuspensionResult::Ready);
    bool destroyed_ready{}, ran{};
    auto resource = std::shared_ptr<int>{new int{}, [&](int* pointer) {
        const auto context = Core::DeferredTransitionScope::Current(system);
        destroyed_ready = context && context->IsReady();
        delete pointer;
    }};
    auto callback = Core::RetainDeferredTransition(system, token, [&, resource] { ran = true; });
    resource.reset();
    token.reset();
    std::thread cancel{[&] { callback = {}; }};
    cancel.join();
    CHECK_FALSE(ran);
    CHECK(destroyed_ready);
    CHECK_FALSE(Core::DeferredTransitionScope::Current(system));
    CHECK(journal->GetSuspensionState() == SuspensionState::Running);
}

TEST_CASE("Fresh sparse contexts nest without authorizing a global transition",
          "[deferred_writes][deferred_fresh_identity]") {
    using namespace VideoCommon::DeferredWrites;
    ScopedSystemTestContext settings;
    Core::System system;
    Journal journal{1, 2};
    const std::array physical{TransitionRange{TransitionSpace::Physical, 0x4000, 4096}};
    std::shared_ptr<Journal::FreshIdentity> identity = journal.TryReserveFreshIdentity(physical);
    REQUIRE(identity);
    {
        Core::FreshDeferredIdentityScope outer{system, identity};
        {
            Core::FreshDeferredIdentityScope inner{system, identity};
            CHECK(Core::FreshDeferredIdentityScope::Current(system) == identity);
            CHECK_FALSE(Core::DeferredTransitionScope::Current(system));
        }
        CHECK(Core::FreshDeferredIdentityScope::Current(system) == identity);
    }
    CHECK_FALSE(Core::FreshDeferredIdentityScope::Current(system));
}

TEST_CASE("Kernel page-buffer exclusion covers allocation before journal and later reuse",
          "[deferred_writes][deferred_writer_permission][deferred_boundaries]") {
    using namespace VideoCommon::DeferredWrites;
    ScopedSystemTestContext settings;
    Core::System system;
    auto journal = std::make_shared<Journal>(1, 2);
    CHECK_FALSE(Core::ExcludeDeferredKernelPageBuffers(system, journal));
    system.Initialize();
    auto& kernel = system.Kernel();
    std::optional<Kernel::KThread> thread{std::in_place, kernel};
    REQUIRE(Kernel::KThread::InitializeDummyThread(system, &*thread, nullptr).IsSuccess());
    kernel.SetCurrentEmuThread(&*thread);
    kernel.Initialize();
    SCOPE_EXIT {
        system.CoreTiming().ClearPendingEvents();
        kernel.Shutdown();
        kernel.SetCurrentEmuThread(nullptr);
        kernel.UnregisterKernelObject(&*thread);
        thread.reset();
    };
    const auto& slab = kernel.SlabHeap<Kernel::KPageBuffer>();
    const auto first_page = system.DeviceMemory().GetRawPhysicalAddr(
        reinterpret_cast<const void*>(slab.GetSlabHeapAddress()));
    const auto pool_size = slab.GetSlabHeapSize() * slab.GetObjectSize();
    auto* existing = Kernel::KPageBuffer::Allocate(kernel);
    REQUIRE(existing);

    auto permission = Core::ExcludeDeferredKernelPageBuffers(system, journal);
    REQUIRE(permission);
    REQUIRE(permission->Ready());
    auto* later = Kernel::KPageBuffer::Allocate(kernel);
    REQUIRE(later);
    const auto existing_physical = system.DeviceMemory().GetRawPhysicalAddr(existing);
    const auto later_physical = system.DeviceMemory().GetRawPhysicalAddr(later);
    CHECK_FALSE(journal->GrantLease(existing_physical, 0x80000, 1));
    CHECK_FALSE(journal->GrantLease(existing_physical, 0x90000, 2));
    CHECK_FALSE(journal->GrantLease(later_physical, 0xa0000, 3));
    CHECK(journal->GrantLease(first_page + pool_size, 0xb0000, 4));

    Kernel::KPageBuffer::Free(kernel, existing);
    auto* reused = Kernel::KPageBuffer::Allocate(kernel);
    REQUIRE(reused == existing);
    CHECK_FALSE(journal->GrantLease(existing_physical, 0xc0000, 5));
    Kernel::KPageBuffer::Free(kernel, reused);
    Kernel::KPageBuffer::Free(kernel, later);
    CHECK_FALSE(journal->GrantLease(later_physical, 0xd0000, 6));

    permission.reset();
    CHECK(journal->GrantLease(existing_physical, 0xe0000, 7));
    CHECK(journal->GrantLease(later_physical, 0xf0000, 8));
}
