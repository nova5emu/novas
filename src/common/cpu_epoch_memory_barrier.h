// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include "common/common_types.h"

namespace Common {
class CpuEpochMemoryBarrier {
public:
    enum class Command : int {
        Query = 0,
        PrivateExpedited = 1 << 3,
        RegisterPrivateExpedited = 1 << 4
    };
    enum class Failure {
        None,
        UnsupportedPlatform,
        QueryFailed,
        CommandUnavailable,
        RegistrationFailed,
        SelfTestFailed,
        ExecuteFailed,
        NotInitialized
    };
    struct CallResult {
        long result{};
        int error{};
    };
    using SyscallFn = CallResult (*)(void*, Command) noexcept;
    explicit CpuEpochMemoryBarrier(SyscallFn syscall = nullptr, void* context = nullptr) noexcept;
    // Startup only, before concurrent Run calls. Registration and its self-test run once.
    bool Initialize() noexcept;
    bool Run() noexcept;
    bool Available() const noexcept {
        return available.load(std::memory_order_acquire);
    }
    Failure LastFailure() const noexcept {
        return failure.load(std::memory_order_relaxed);
    }
    int LastError() const noexcept {
        return error.load(std::memory_order_relaxed);
    }
    long QueryMask() const noexcept {
        return query_mask;
    }
    static const char* FailureName(Failure failure) noexcept;

private:
    bool Fail(Failure reason, int code) noexcept;
    static CallResult NativeCall(void*, Command command) noexcept;
    SyscallFn syscall_fn;
    void* context;
    const bool native_backend;
    bool initialized{};
    long query_mask{};
    std::atomic<bool> available{};
    std::atomic<Failure> failure{Failure::NotInitialized};
    std::atomic<int> error{};
};
} // namespace Common
