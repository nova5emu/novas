// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <optional>
#include "common/common_types.h"
#include "common/dkb_cpu_trace.h"

namespace Common::GuestThreadProfile {

struct ThreadInfo {
    u64 pid{}, tid{}, entry{};
    u32 priority{};
    bool main_thread{};
};

// States supplied by the kernel adapter: initialized, waiting, runnable, terminated, dormant.
struct Timeline {
    u64 cursor_ns{}, scheduled_wall_ns{}, runnable_ns{}, blocked_ns{}, host_cpu_ns{}, unknown_wall_ns{};
    u32 state{};
    bool scheduled{};
    bool wall_complete{true};
    std::optional<u64> cpu_baseline;

    u64 Advance(u64 now_ns);
    bool ChargeCpu(std::optional<u64> now_ns);
    void ReconcileOwnership(u64 known_event_ns, u64 now_ns, bool current_owner);
    void ClearWindow();
};

struct OwnerSample {
    u64 sequence{}, tid{}, wall_ns{}, clock_generation{};
    std::optional<u64> cpu_ns;
};

// One publishing host thread per role. Readers make a bounded, nonblocking attempt.
class RoleOwner {
public:
    u64 Publish(u64 owner_tid, u64 now, std::optional<u64> cpu, u64 generation);
    std::optional<OwnerSample> Read() const;

private:
    std::atomic<u64> sequence{}, tid{}, wall_ns{}, cpu_ns{}, clock_generation{};
    std::atomic<bool> cpu_valid{};
};

void Configure();
bool Enabled();
void OnState(const ThreadInfo& info, u32 state, u8 svc);
void OnSwitch(const ThreadInfo& info, bool scheduled, DkbCpuTrace::Role role);
void SetWait(const ThreadInfo& info, const char* reason, u64 object = 0, u64 detail = 0,
             u64 target = 0, u8 svc = 0);
void RecordWake(const ThreadInfo& waker, const ThreadInfo& waiter, u64 object, u8 svc);
void Report();

} // namespace Common::GuestThreadProfile
