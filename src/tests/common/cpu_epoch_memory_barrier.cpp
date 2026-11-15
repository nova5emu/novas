// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <cerrno>
#include <initializer_list>
#include <string_view>
#if defined(__linux__)
#include <unistd.h>
#endif
#include <catch2/catch_test_macros.hpp>
#include "common/cpu_epoch_memory_barrier.h"
#include "common/heap_tracker.h"
#include "common/host_memory.h"

namespace {
using Barrier = Common::CpuEpochMemoryBarrier;
using Command = Barrier::Command;
using Failure = Barrier::Failure;
struct Step {
    Command command;
    Barrier::CallResult result;
};
struct Fake {
    std::array<Step, 8> steps{};
    size_t count{}, called{};
    bool unexpected{};
    Fake(std::initializer_list<Step> input) {
        for (const auto& step : input) {
            steps[count++] = step;
        }
    }
    static Barrier::CallResult Call(void* context, Command command) noexcept {
        auto& fake = *static_cast<Fake*>(context);
        if (fake.called >= fake.count) {
            fake.unexpected = true;
            return {-1, EINVAL};
        }
        const auto& step = fake.steps[fake.called++];
        fake.unexpected |= step.command != command;
        return step.result;
    }
};
constexpr long Capabilities = static_cast<long>(Command::PrivateExpedited) |
                              static_cast<long>(Command::RegisterPrivateExpedited);
} // namespace

TEST_CASE("CPU epoch membarrier registers and self-tests once before reusable execution",
          "[common][cpu_epoch_memory_barrier]") {
    Fake fake{{Command::Query, {Capabilities, 0}},
              {Command::RegisterPrivateExpedited, {0, 0}},
              {Command::PrivateExpedited, {0, 0}},
              {Command::PrivateExpedited, {0, 0}},
              {Command::PrivateExpedited, {-1, EPERM}}};
    Barrier barrier{Fake::Call, &fake};
    REQUIRE_FALSE(barrier.Available());
    REQUIRE_FALSE(barrier.Run());
    REQUIRE(fake.called == 0);
    REQUIRE(barrier.LastFailure() == Failure::NotInitialized);
    REQUIRE(barrier.Initialize());
    REQUIRE(fake.called == 3);
    REQUIRE(barrier.QueryMask() == Capabilities);
    REQUIRE(barrier.LastFailure() == Failure::None);
    REQUIRE(barrier.Initialize());
    REQUIRE(fake.called == 3);
    REQUIRE(barrier.Run());
    REQUIRE(fake.called == 4);
    REQUIRE_FALSE(barrier.Run());
    REQUIRE_FALSE(barrier.Available());
    REQUIRE(barrier.LastFailure() == Failure::ExecuteFailed);
    REQUIRE(barrier.LastError() == EPERM);
    REQUIRE_FALSE(barrier.Run());
    REQUIRE_FALSE(barrier.Initialize());
    REQUIRE(fake.called == 5);
    REQUIRE_FALSE(fake.unexpected);
}

TEST_CASE("CPU epoch membarrier rejects incomplete capabilities without registration",
          "[common][cpu_epoch_memory_barrier]") {
    for (const auto mask : {0L, static_cast<long>(Command::PrivateExpedited),
                            static_cast<long>(Command::RegisterPrivateExpedited)}) {
        Fake fake{{Command::Query, {mask, 0}}};
        Barrier barrier{Fake::Call, &fake};
        REQUIRE_FALSE(barrier.Initialize());
        REQUIRE(barrier.LastFailure() == Failure::CommandUnavailable);
        REQUIRE(barrier.QueryMask() == mask);
        REQUIRE_FALSE(barrier.Run());
        REQUIRE(fake.called == 1);
        REQUIRE_FALSE(fake.unexpected);
    }
}

TEST_CASE("CPU epoch membarrier startup errors remain unavailable with explicit failure reasons",
          "[common][cpu_epoch_memory_barrier]") {
    Fake query{{Command::Query, {-1, ENOSYS}}};
    Barrier query_barrier{Fake::Call, &query};
    REQUIRE_FALSE(query_barrier.Initialize());
    REQUIRE(query_barrier.LastFailure() == Failure::QueryFailed);
    REQUIRE(query_barrier.LastError() == ENOSYS);
    Fake registration{{Command::Query, {Capabilities, 0}},
                      {Command::RegisterPrivateExpedited, {-1, EPERM}}};
    Barrier registration_barrier{Fake::Call, &registration};
    REQUIRE_FALSE(registration_barrier.Initialize());
    REQUIRE(registration_barrier.LastFailure() == Failure::RegistrationFailed);
    REQUIRE(registration_barrier.LastError() == EPERM);
    REQUIRE(registration.called == 2);
    Fake execution{{Command::Query, {Capabilities, 0}},
                   {Command::RegisterPrivateExpedited, {0, 0}},
                   {Command::PrivateExpedited, {-1, EINVAL}}};
    Barrier execution_barrier{Fake::Call, &execution};
    REQUIRE_FALSE(execution_barrier.Initialize());
    REQUIRE(execution_barrier.LastFailure() == Failure::SelfTestFailed);
    REQUIRE(execution_barrier.LastError() == EINVAL);
    REQUIRE_FALSE(execution_barrier.Available());
    REQUIRE(std::string_view{Barrier::FailureName(execution_barrier.LastFailure())} ==
            "self_test_failed");
    REQUIRE_FALSE(query.unexpected);
    REQUIRE_FALSE(registration.unexpected);
    REQUIRE_FALSE(execution.unexpected);
}

TEST_CASE("CPU epoch reprotection is conservative on unsupported platforms and invalid ranges",
          "[common][cpu_epoch_memory_barrier]") {
    Common::HostMemory memory{0x10000, 0x200000};
    REQUIRE_FALSE(memory.TryReprotectCpuEpoch(1));
    REQUIRE_FALSE(memory.TryReprotectCpuEpoch(0, 0));
    REQUIRE_FALSE(memory.TryReprotectCpuEpoch(0, 0x2000));
    REQUIRE_FALSE(memory.TryReprotectCpuEpoch(memory.VirtualSize()));
#ifdef _WIN32
    memory.Map(0x1000, 0, 0x1000, Common::MemoryPermission::ReadWrite, false);
    REQUIRE_FALSE(memory.TryReprotectCpuEpoch(0x1000));
    Common::CpuEpochMemoryBarrier barrier;
    REQUIRE_FALSE(barrier.Initialize());
    REQUIRE(barrier.LastFailure() == Failure::UnsupportedPlatform);
#else
#ifdef __linux__
    memory.Map(0x1000, 0, 0x1000, Common::MemoryPermission::ReadWrite, false);
    if (sysconf(_SC_PAGESIZE) != 0x1000 || !memory.VirtualBasePointer()) {
        SKIP("CPU epoch reprotection requires a working 4 KiB fastmem arena");
    }
    volatile u8* const native = memory.VirtualBasePointer() + 0x1000;
    native[0] = 42;
    REQUIRE(memory.TryReprotectCpuEpoch(0x1000));
    REQUIRE(native[0] == 42);
    memory.BackingBasePointer()[0] = 43;
    REQUIRE(native[0] == 43);
#endif
#endif
}
