// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <cerrno>
#include "common/cpu_epoch_memory_barrier.h"
#if defined(__linux__)
#include <linux/membarrier.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Common {
CpuEpochMemoryBarrier::CpuEpochMemoryBarrier(SyscallFn syscall, void* context_) noexcept
    : syscall_fn{syscall ? syscall : NativeCall}, context{context_}, native_backend{!syscall} {}
bool CpuEpochMemoryBarrier::Fail(Failure reason, int code) noexcept {
    failure.store(reason, std::memory_order_relaxed);
    error.store(code, std::memory_order_relaxed);
    available.store(false, std::memory_order_release);
    return false;
}
CpuEpochMemoryBarrier::CallResult CpuEpochMemoryBarrier::NativeCall(void*,
                                                                    Command command) noexcept {
#if defined(__linux__) && defined(SYS_membarrier)
    static_assert(static_cast<int>(Command::PrivateExpedited) == MEMBARRIER_CMD_PRIVATE_EXPEDITED);
    static_assert(static_cast<int>(Command::RegisterPrivateExpedited) ==
                  MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED);
    const auto result = ::syscall(SYS_membarrier, static_cast<int>(command), 0, 0);
    return {result, result < 0 ? errno : 0};
#else
    static_cast<void>(command);
    return {-1, ENOSYS};
#endif
}
bool CpuEpochMemoryBarrier::Initialize() noexcept {
    if (initialized) {
        return Available();
    }
    initialized = true;
#if defined(__linux__) && defined(SYS_membarrier)
    constexpr bool native_supported = true;
#else
    constexpr bool native_supported = false;
#endif
    if (native_backend && !native_supported) {
        return Fail(Failure::UnsupportedPlatform, ENOSYS);
    }
    const auto queried = syscall_fn(context, Command::Query);
    if (queried.result < 0) {
        return Fail(Failure::QueryFailed, queried.error);
    }
    query_mask = queried.result;
    constexpr int required = static_cast<int>(Command::PrivateExpedited) |
                             static_cast<int>(Command::RegisterPrivateExpedited);
    if ((query_mask & required) != required) {
        return Fail(Failure::CommandUnavailable, 0);
    }
    const auto registered = syscall_fn(context, Command::RegisterPrivateExpedited);
    if (registered.result != 0) {
        return Fail(Failure::RegistrationFailed, registered.error);
    }
    const auto tested = syscall_fn(context, Command::PrivateExpedited);
    if (tested.result != 0) {
        return Fail(Failure::SelfTestFailed, tested.error);
    }
    failure.store(Failure::None, std::memory_order_relaxed);
    error.store(0, std::memory_order_relaxed);
    available.store(true, std::memory_order_release);
    return true;
}
bool CpuEpochMemoryBarrier::Run() noexcept {
    if (!Available()) {
        return false;
    }
    const auto executed = syscall_fn(context, Command::PrivateExpedited);
    if (executed.result != 0) {
        return Fail(Failure::ExecuteFailed, executed.error);
    }
    return Available();
}
const char* CpuEpochMemoryBarrier::FailureName(Failure value) noexcept {
    switch (value) {
    case Failure::None:
        return "none";
    case Failure::UnsupportedPlatform:
        return "unsupported_platform";
    case Failure::QueryFailed:
        return "query_failed";
    case Failure::CommandUnavailable:
        return "command_unavailable";
    case Failure::RegistrationFailed:
        return "registration_failed";
    case Failure::SelfTestFailed:
        return "self_test_failed";
    case Failure::ExecuteFailed:
        return "execute_failed";
    case Failure::NotInitialized:
        return "not_initialized";
    }
    return "unknown";
}
} // namespace Common
