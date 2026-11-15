// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>

#include "core/arm/arm_interface.h"
#include "core/arm/nce/guest_context.h"
#include "common/gpu_transition_profile.h"
#ifdef HAS_NCE
#include "core/arm/nce/deferred_publication_budget.h"
#endif

namespace Core::Memory {
class Memory;
}

namespace Core {

class System;

class ArmNce final : public ArmInterface {
public:
    ArmNce(System& system, bool uses_wall_clock, std::size_t core_index);
    ~ArmNce() override;

    void Initialize() override;

    Architecture GetArchitecture() const override {
        return Architecture::AArch64;
    }

    HaltReason RunThread(Kernel::KThread* thread) override;
    HaltReason StepThread(Kernel::KThread* thread) override;
#ifdef HAS_NCE
    void CompleteHostOperations(Kernel::KThread* thread) override;
#endif

    void GetContext(Kernel::Svc::ThreadContext& ctx) const override;
    void SetContext(const Kernel::Svc::ThreadContext& ctx) override;
    void SetTpidrroEl0(u64 value) override;

    void GetSvcArguments(std::span<uint64_t, 8> args) const override;
    void SetSvcArguments(std::span<const uint64_t, 8> args) override;
    u32 GetSvcNumber() const override;

    void SignalInterrupt(Kernel::KThread* thread) override;
    void ClearInstructionCache() override;
    void InvalidateCacheRange(u64 addr, std::size_t size) override;

    void LockThread(Kernel::KThread* thread) override;
    void UnlockThread(Kernel::KThread* thread) override;

protected:
    const Kernel::DebugWatchpoint* HaltedWatchpoint() const override {
        return nullptr;
    }

    void RewindBreakpointInstruction() override {}

private:
    // Assembly definitions.
    static HaltReason ReturnToRunCodeByTrampoline(void* tpidr, GuestContext* ctx,
                                                  u64 trampoline_addr);
    static HaltReason ReturnToRunCodeByExceptionLevelChange(int tid, void* tpidr);

    static void ReturnToRunCodeByExceptionLevelChangeSignalHandler(int sig, void* info,
                                                                   void* raw_context);
    static void BreakFromRunCodeSignalHandler(int sig, void* info, void* raw_context);
    static void GuestAlignmentFaultSignalHandler(int sig, void* info, void* raw_context);
    static void GuestAccessFaultSignalHandler(int sig, void* info, void* raw_context);

    static void LockThreadParameters(void* tpidr);
    static void UnlockThreadParameters(void* tpidr);

private:
    // C++ implementation functions for assembly definitions.
    static void* RestoreGuestContext(void* raw_context);
    static void SaveGuestContext(GuestContext* ctx, void* raw_context);
    static bool HandleFailedGuestFault(GuestContext* ctx, void* info, void* raw_context);
    static bool HandleGuestAlignmentFault(GuestContext* ctx, void* info, void* raw_context);
    static bool HandleGuestAccessFault(GuestContext* ctx, void* info, void* raw_context);
    static void HandleHostAlignmentFault(int sig, void* info, void* raw_context);
    static void HandleHostAccessFault(int sig, void* info, void* raw_context);

#ifdef HAS_NCE
    static bool DeferExactStore(GuestContext* ctx, void* info, void* raw_context);
    void CompleteExactStore(Kernel::KThread* thread);
    void RetryNativeAccess(Kernel::KThread* thread, u64 fault_address,
                           Common::GpuTransitionProfile::DeferredFallbackReason reason);
    void ArmNativeRetry(Kernel::KThread* thread, u64 fault_address, bool alignment_failure);
    bool ConsumeNativeRetry(u64 pc, u64 fault_address, bool alignment_failure);

    struct PendingStore {
        u64 fault_address{};
        u64 backing_page{};
        bool fallback_only{};
        bool alignment{};
        Common::GpuTransitionProfile::DeferredFallbackReason fallback_reason{};
    };
    PendingStore m_pending_store{};
    std::atomic<bool> m_store_pending{};
    std::atomic<bool> m_retry_store{};
    Kernel::KThread* m_retry_store_thread{};
    u64 m_retry_store_pc{};
    u64 m_retry_store_address{};
    bool m_retry_alignment_failure{};
    Kernel::KThread* m_store_resume_thread{};
    u64 m_store_host_start_ns{};
    u64 m_store_host_done_ns{};
    bool m_exact_stores_enabled{};
    bool m_deferred_fallbacks_enabled{};
    bool m_deferred_runtime_logged{};
    DeferredPublicationBudget m_publication_budget;
#endif

public:
    Core::System& m_system;

    // Members set on initialization.
    std::size_t m_core_index{};
    pid_t m_thread_id{-1};

    // Core context.
    GuestContext m_guest_ctx{};
    Kernel::KThread* m_running_thread{};

    // Stack for signal processing.
    std::unique_ptr<u8[]> m_stack{};
};

} // namespace Core
