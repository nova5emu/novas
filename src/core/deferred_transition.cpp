// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "core/core.h"
#include "core/deferred_transition.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_page_buffer.h"
#include "core/hle/kernel/kernel.h"

namespace Core {
using VideoCommon::DeferredWrites::Suspension;

thread_local DeferredTransitionScope* DeferredTransitionScope::current{};
thread_local FreshDeferredIdentityScope* FreshDeferredIdentityScope::current{};

DeferredTransitionScope::DeferredTransitionScope(
    System& system_, Common::GpuTransitionProfile::SuspensionSite site)
    : system{system_},
      epoch_mutation{system_.DeviceMemory().CpuEpochs().BeforeMutation(
          VideoCommon::CpuEpochs::CloseReason::Mapping)},
      token{Current(system_) ? Current(system_) : system_.BeginDeferredTransition(site)} {
    if (!token) return;
    ASSERT_MSG(token->IsReady(), "A transition context must already be ready");
    if (auto* journal = system.DeferredWriteJournal()) {
        ASSERT(journal->OwnsSuspension(*token));
    }
    previous = current;
    current = this;
}

DeferredTransitionScope::DeferredTransitionScope(System& system_,
                                                std::shared_ptr<Suspension> token_)
    : system{system_}, token{std::move(token_)} {
    if (!token) return;
    ASSERT_MSG(token->IsReady(), "A transition context must already be ready");
    if (auto* journal = system.DeferredWriteJournal()) {
        ASSERT(journal->OwnsSuspension(*token));
    }
    previous = current;
    current = this;
}

DeferredTransitionScope::~DeferredTransitionScope() {
    if (!token) return;
    ASSERT(current == this);
    current = previous;
}

std::shared_ptr<Suspension> DeferredTransitionScope::Current(System& system_) {
    for (auto* scope = current; scope; scope = scope->previous) {
        if (&scope->system == &system_) return scope->token;
    }
    return {};
}

std::shared_ptr<Suspension> DeferredTransitionScope::RequireReady(System& system_) {
    if (!system_.DeferredWriteJournal()) return {};
    auto result = Current(system_);
    ASSERT_MSG(result && result->IsReady(), "Mapping mutation lacks its outer ready transition");
    return result;
}

FreshDeferredIdentityScope::FreshDeferredIdentityScope(System& system_,
                                                     std::shared_ptr<Identity> identity_)
    : system{system_}, identity{std::move(identity_)} {
    if (!identity) return;
    previous = current;
    current = this;
}

FreshDeferredIdentityScope::~FreshDeferredIdentityScope() {
    if (!identity) return;
    ASSERT(current == this);
    current = previous;
}

std::shared_ptr<FreshDeferredIdentityScope::Identity> FreshDeferredIdentityScope::Current(
    System& system_) {
    for (auto* scope = current; scope; scope = scope->previous) {
        if (&scope->system == &system_) return scope->identity;
    }
    return {};
}

std::shared_ptr<VideoCommon::DeferredWrites::Journal::WriterPermission>
ExcludeDeferredKernelPageBuffers(
    System& system, const std::shared_ptr<VideoCommon::DeferredWrites::Journal>& journal) {
    if (!journal) return {};
    // Cache-only fixtures can construct a GPU before the kernel's slab heaps exist.
    if (!system.Kernel().GetSystemResourceLimit()) return {};
    const auto& slab = system.Kernel().SlabHeap<Kernel::KPageBuffer>();
    if (slab.GetObjectSize() == 0) return {};
    const auto bytes = slab.GetSlabHeapSize() * slab.GetObjectSize();
    if (bytes == 0) return {};

    // The fixed pool includes TLS created before the GPU/journal exists, future TLS and IPC
    // page buffers. Exclude physical identity so all device aliases remain ineligible too.
    const auto physical = system.DeviceMemory().GetRawPhysicalAddr(
        reinterpret_cast<const void*>(slab.GetSlabHeapAddress()));
    const std::array ranges{VideoCommon::DeferredWrites::TransitionRange{
        VideoCommon::DeferredWrites::TransitionSpace::Physical, physical, bytes}};
    system.DeviceMemory().CpuEpochs().Exclude(
        physical, bytes, VideoCommon::CpuEpochs::Exclusion::CapturedWriter);
    auto permission = VideoCommon::DeferredWrites::BeginSharedWriterPermission(journal, ranges);
    if (!permission || !permission->MarkReady()) {
        UNREACHABLE_MSG("Kernel page-buffer exclusion must precede deferred producers");
    }
    return permission;
}

std::function<void()> RetainDeferredTransition(System& system, std::shared_ptr<Suspension> token,
                                              std::function<void()>&& operation) {
    if (!token) token = DeferredTransitionScope::Current(system);
    if (!token) return std::move(operation);
    struct ContextOperation {
        System& owner;
        std::shared_ptr<Suspension> participant;
        std::function<void()> action;

        ContextOperation(System& system_, std::shared_ptr<Suspension> participant_,
                         std::function<void()>&& action_)
            : owner{system_}, participant{std::move(participant_)}, action{std::move(action_)} {}
        ContextOperation(const ContextOperation&) = default;
        ContextOperation(ContextOperation&&) noexcept = default;
        ~ContextOperation() {
            if (!action) return;
            // Queue cancellation also destroys resources that can unmap memory.
            DeferredTransitionScope context{owner, participant};
            action = {};
        }
        void operator()() {
            DeferredTransitionScope context{owner, participant};
            auto run_once = std::move(action);
            run_once();
        }
    };
    auto action = ContextOperation{system, token, std::move(operation)};
    return VideoCommon::DeferredWrites::RetainReadySuspension(std::move(token), std::move(action));
}
} // namespace Core
