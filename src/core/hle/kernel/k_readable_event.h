// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/kernel/k_synchronization_object.h"
#include "core/hle/kernel/slab_helpers.h"
#include "core/hle/result.h"

namespace Kernel {

class KernelCore;
class KEvent;

class KReadableEvent : public KSynchronizationObject {
    KERNEL_AUTOOBJECT_TRAITS(KReadableEvent, KSynchronizationObject);

public:
    explicit KReadableEvent(KernelCore& kernel);
    ~KReadableEvent() override;

    void Initialize(KernelCore& kernel, KEvent* parent);

    KEvent* GetParent() const {
        return m_parent;
    }

    void SetWaitProfile(const char* reason, u64 detail = 0, u64 target = 0) {
        m_profile_detail.store(detail, std::memory_order_relaxed);
        m_profile_target.store(target, std::memory_order_relaxed);
        m_profile_reason.store(reason, std::memory_order_release);
    }
    const char* GetWaitProfileReason() const {
        return m_profile_reason.load(std::memory_order_acquire);
    }
    u64 GetWaitProfileDetail() const {
        return m_profile_detail.load(std::memory_order_relaxed);
    }
    u64 GetWaitProfileTarget() const {
        return m_profile_target.load(std::memory_order_relaxed);
    }

    Result Signal(KernelCore& kernel);
    Result Clear(KernelCore& kernel);
    bool IsSignaled(KernelCore& kernel) const override;
    void Destroy(KernelCore& kernel) override;
    Result Reset(KernelCore& kernel);

private:
    bool m_is_signaled{};
    std::atomic<const char*> m_profile_reason{};
    std::atomic<u64> m_profile_detail{}, m_profile_target{};
    KEvent* m_parent{};
};

} // namespace Kernel
