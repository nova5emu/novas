// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <memory>
#include "common/gpu_transition_profile.h"
#include "video_core/deferred_write_journal.h"
#include "video_core/cpu_epoch_coordinator.h"

namespace Core {
class System;

// Roots prepare before taking transaction locks; callbacks install an already-ready token.
class DeferredTransitionScope {
public:
    DeferredTransitionScope(System& system, Common::GpuTransitionProfile::SuspensionSite site);
    DeferredTransitionScope(System& system,
                            std::shared_ptr<VideoCommon::DeferredWrites::Suspension> token);
    ~DeferredTransitionScope();
    DeferredTransitionScope(const DeferredTransitionScope&) = delete;
    DeferredTransitionScope& operator=(const DeferredTransitionScope&) = delete;

    static std::shared_ptr<VideoCommon::DeferredWrites::Suspension> Current(System& system);
    static std::shared_ptr<VideoCommon::DeferredWrites::Suspension> RequireReady(System& system);
    auto GetToken() const { return token; }

private:
    System& system;
    VideoCommon::CpuEpochs::Coordinator::Mutation epoch_mutation;
    std::shared_ptr<VideoCommon::DeferredWrites::Suspension> token;
    DeferredTransitionScope* previous{};
    static thread_local DeferredTransitionScope* current;
};

class FreshDeferredIdentityScope {
public:
    using Identity = VideoCommon::DeferredWrites::Journal::FreshIdentity;
    FreshDeferredIdentityScope(System& system, std::shared_ptr<Identity> identity);
    ~FreshDeferredIdentityScope();
    FreshDeferredIdentityScope(const FreshDeferredIdentityScope&) = delete;
    FreshDeferredIdentityScope& operator=(const FreshDeferredIdentityScope&) = delete;
    static std::shared_ptr<Identity> Current(System& system);

private:
    System& system;
    std::shared_ptr<Identity> identity;
    FreshDeferredIdentityScope* previous{};
    static thread_local FreshDeferredIdentityScope* current;
};

std::function<void()> RetainDeferredTransition(
    System& system, std::shared_ptr<VideoCommon::DeferredWrites::Suspension> token,
    std::function<void()>&& operation);

std::shared_ptr<VideoCommon::DeferredWrites::Journal::WriterPermission>
ExcludeDeferredKernelPageBuffers(
    System& system, const std::shared_ptr<VideoCommon::DeferredWrites::Journal>& journal);
} // namespace Core
