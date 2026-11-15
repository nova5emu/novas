// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/guest_thread_profile.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "common/logging.h"
#include "common/thread_cpu_clock.h"

namespace Common::GuestThreadProfile {

u64 Timeline::Advance(u64 now_ns) {
    if (now_ns < cursor_ns) {
        return 0;
    }
    const u64 elapsed = now_ns - cursor_ns;
    cursor_ns = now_ns;
    if (scheduled) {
        scheduled_wall_ns += elapsed;
    } else if (state == 2) {
        runnable_ns += elapsed;
    } else if (state == 1 || state == 4) {
        blocked_ns += elapsed;
        return elapsed;
    }
    return 0;
}

bool Timeline::ChargeCpu(std::optional<u64> now_ns) {
    const bool valid = cpu_baseline && now_ns && *now_ns >= *cpu_baseline;
    if (valid) {
        host_cpu_ns += *now_ns - *cpu_baseline;
    }
    cpu_baseline = now_ns;
    return valid;
}

void Timeline::ClearWindow() {
    scheduled_wall_ns = runnable_ns = blocked_ns = host_cpu_ns = unknown_wall_ns = 0;
    wall_complete = true;
}

void Timeline::ReconcileOwnership(u64 known_event_ns, u64 now_ns, bool current_owner) {
    const u64 cut = (std::max)(cursor_ns, (std::min)(known_event_ns, now_ns));
    unknown_wall_ns += scheduled_wall_ns + runnable_ns + blocked_ns + cut - cursor_ns;
    scheduled_wall_ns = runnable_ns = blocked_ns = 0;
    cursor_ns = cut;
    scheduled = current_owner;
    wall_complete = false;
    cpu_baseline.reset();
}

u64 RoleOwner::Publish(u64 owner_tid, u64 now, std::optional<u64> cpu, u64 generation) {
    sequence.fetch_add(1, std::memory_order_acq_rel);
    tid.store(owner_tid, std::memory_order_relaxed);
    wall_ns.store(now, std::memory_order_relaxed);
    cpu_ns.store(cpu.value_or(0), std::memory_order_relaxed);
    cpu_valid.store(cpu.has_value(), std::memory_order_relaxed);
    clock_generation.store(generation, std::memory_order_relaxed);
    return sequence.fetch_add(1, std::memory_order_release) + 1;
}

std::optional<OwnerSample> RoleOwner::Read() const {
    const u64 before = sequence.load(std::memory_order_acquire);
    if (before & 1) {
        return std::nullopt;
    }
    OwnerSample sample{before, tid.load(std::memory_order_relaxed), wall_ns.load(std::memory_order_relaxed),
                       clock_generation.load(std::memory_order_relaxed), std::nullopt};
    const u64 cpu = cpu_ns.load(std::memory_order_relaxed);
    if (cpu_valid.load(std::memory_order_relaxed)) {
        sample.cpu_ns = cpu;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (sequence.load(std::memory_order_relaxed) != before) {
        return std::nullopt;
    }
    return sample;
}

namespace {
constexpr size_t ThreadLimit = 4096;
constexpr size_t WaitLimit = 64;
constexpr u64 ReportIntervalNs = 5'000'000'000;

struct WaitKey {
    std::string reason{"unspecified"};
    u64 object{}, detail{}, target{};
    u8 svc{};
    bool operator<(const WaitKey& other) const {
        return std::tie(reason, object, detail, target, svc) <
               std::tie(other.reason, other.object, other.detail, other.target, other.svc);
    }
};
struct Entry {
    ThreadInfo info;
    Timeline timeline;
    WaitKey wait;
    std::optional<WaitKey> suspended_wait;
    std::map<WaitKey, u64> waits;
    u64 wait_overflow_ns{}, role_epoch{}, clock_generation{};
    u64 state_epoch{std::numeric_limits<u64>::max()};
    DkbCpuTrace::Role role{DkbCpuTrace::Role::Count};
    bool cpu_complete{true};
    bool owner_known{true};
};
struct ThreadReport {
    ThreadInfo info;
    Timeline timeline;
    std::map<WaitKey, u64> waits;
    u64 wait_overflow_ns{};
    bool cpu_complete{};
    bool owner_known{};
    u64 state_epoch{};
};
struct CpuProbe {
    std::optional<DkbCpuTrace::ThreadSample> clock;
    u64 owner_sequence{};
};
using WakeKey = std::tuple<u64, u64, u64, u64, u64, u8>;
std::mutex registry_mutex;
std::map<std::pair<u64, u64>, Entry> threads;
std::array<RoleOwner, DkbCpuTrace::RoleCount> role_owners;
std::array<std::atomic<u64>, DkbCpuTrace::RoleCount> role_clock_generations;
std::map<WakeKey, u64> wakes;
std::atomic<u64> wake_drops{}, wake_overflow{};
std::atomic<u64> dropped_events{}, dropped_reports{}, capacity_drops{};
std::atomic<u64> lost_event_epoch{};
std::atomic<u64> next_report_ns{};
std::atomic<bool> configured_enabled{}, reset_pending{};
std::atomic<u64> session{};
u64 window_start_ns{}, epoch{}, window_state_epoch{};

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void Advance(Entry& entry, u64 now) {
    const u64 blocked = entry.timeline.Advance(now);
    if (!blocked) {
        return;
    }
    const auto found = entry.waits.find(entry.wait);
    if (found != entry.waits.end()) {
        found->second += blocked;
    } else if (entry.waits.size() < WaitLimit) {
        entry.waits.emplace(entry.wait, blocked);
    } else {
        entry.wait_overflow_ns += blocked;
    }
}

Entry* FindOrRegister(const ThreadInfo& info, u64 now) {
    const auto key = std::pair{info.pid, info.tid};
    auto found = threads.find(key);
    if (found == threads.end()) {
        if (threads.size() >= ThreadLimit) {
            capacity_drops.fetch_add(1, std::memory_order_relaxed);
            lost_event_epoch.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        found = threads.try_emplace(key).first;
        found->second.info = info;
        found->second.timeline.cursor_ns = now;
    }
    found->second.info.priority = info.priority;
    found->second.info.main_thread |= info.main_thread;
    if (!found->second.info.entry) {
        found->second.info.entry = info.entry;
    }
    if (!window_start_ns) {
        window_start_ns = now;
        window_state_epoch = lost_event_epoch.load(std::memory_order_relaxed);
        next_report_ns.store(now + ReportIntervalNs, std::memory_order_relaxed);
    }
    return &found->second;
}

std::unique_lock<std::mutex> LockEvent() {
    std::unique_lock lock{registry_mutex, std::try_to_lock};
    if (!lock) {
        dropped_events.fetch_add(1, std::memory_order_relaxed);
        lost_event_epoch.fetch_add(1, std::memory_order_relaxed);
    } else if (reset_pending.exchange(false, std::memory_order_relaxed)) {
        threads.clear();
        wakes.clear();
        window_start_ns = epoch = 0;
        next_report_ns.store(0, std::memory_order_relaxed);
    }
    return lock;
}
} // namespace

void Configure() {
    configured_enabled.store(false, std::memory_order_relaxed);
    const auto flag = [](const char* name) {
        const char* value = std::getenv(name);
        return value && value[0] == '1' && value[1] == '\0';
    };
    const bool active = flag("EDEN_NX2_GPU_PROFILE") || flag("EDEN_DKB_FRAME_TRACE");
    session.fetch_add(1, std::memory_order_relaxed);
    next_report_ns.store(0, std::memory_order_relaxed);
    dropped_events.store(0, std::memory_order_relaxed);
    dropped_reports.store(0, std::memory_order_relaxed);
    capacity_drops.store(0, std::memory_order_relaxed);
    const u64 reset_ns = NowNs();
    for (size_t i = 0; i < role_clock_generations.size(); ++i) {
        role_clock_generations[i].store(0, std::memory_order_relaxed);
        role_owners[i].Publish(0, reset_ns, std::nullopt, 0);
    }
    wake_drops.store(0, std::memory_order_relaxed);
    wake_overflow.store(0, std::memory_order_relaxed);
    std::unique_lock lock{registry_mutex, std::try_to_lock};
    if (lock) {
        threads.clear();
        wakes.clear();
        window_start_ns = epoch = 0;
        reset_pending.store(false, std::memory_order_relaxed);
    } else {
        reset_pending.store(true, std::memory_order_relaxed);
    }
    configured_enabled.store(active, std::memory_order_relaxed);
}

bool Enabled() {
    return configured_enabled.load(std::memory_order_relaxed);
}

void OnState(const ThreadInfo& info, u32 state, u8 svc) {
    if (!Enabled()) {
        return;
    }
    auto lock = LockEvent();
    if (!lock) {
        return;
    }
    const u64 now = NowNs();
    auto* entry = FindOrRegister(info, now);
    if (!entry) {
        return;
    }
    Advance(*entry, now);
    entry->state_epoch = lost_event_epoch.load(std::memory_order_relaxed);
    const u32 previous_state = entry->timeline.state;
    if (state == previous_state) {
        return;
    }
    entry->timeline.state = state;
    if (state == 4) {
        if (previous_state == 1) {
            entry->suspended_wait = entry->wait;
        } else {
            entry->suspended_wait.reset();
        }
        entry->wait = {"dormant", 0, 0, 0, svc};
    } else if (state == 1) {
        if (previous_state == 4 && entry->suspended_wait) {
            entry->wait = std::move(*entry->suspended_wait);
        } else {
            entry->wait = {"unspecified", 0, 0, 0, svc};
        }
        entry->suspended_wait.reset();
    } else {
        entry->wait = {"unspecified", 0, 0, 0, svc};
        entry->suspended_wait.reset();
    }
}

void OnSwitch(const ThreadInfo& info, bool scheduled, DkbCpuTrace::Role role) {
    if (!Enabled()) {
        return;
    }
    const auto role_index = static_cast<size_t>(role);
    if (role_index >= role_owners.size()) {
        dropped_events.fetch_add(1, std::memory_order_relaxed);
        lost_event_epoch.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto cpu_now = CurrentThreadCpuTimeNs();
    u64 generation = role_clock_generations[role_index].load(std::memory_order_relaxed);
    if (scheduled && !generation) {
        const auto sample = DkbCpuTrace::TryReadThreadSample(role);
        if (sample && sample->active) {
            generation = sample->generation;
            role_clock_generations[role_index].store(generation, std::memory_order_relaxed);
        }
    }
    const u64 event_ns = NowNs();
    const u64 role_epoch = role_owners[role_index].Publish(scheduled ? info.tid : 0,
                                                         event_ns, cpu_now, generation);
    auto lock = LockEvent();
    if (!lock) {
        return;
    }
    const u64 now = NowNs();
    auto* entry = FindOrRegister(info, now);
    if (!entry) {
        return;
    }
    if (entry->timeline.scheduled &&
        (entry->role != role || entry->role_epoch + 2 != role_epoch)) {
        entry->timeline.ReconcileOwnership(event_ns, now, scheduled);
        entry->waits.clear();
        entry->wait_overflow_ns = 0;
        entry->cpu_complete = false;
    }
    Advance(*entry, now);
    const bool reliable_role = role != DkbCpuTrace::Role::GuestSingleCore;
    if (entry->timeline.scheduled && reliable_role) {
        if (entry->role != role || entry->role_epoch + 2 != role_epoch ||
            !entry->timeline.ChargeCpu(cpu_now)) {
            entry->cpu_complete = false;
        }
    }
    entry->timeline.scheduled = scheduled;
    if (scheduled && entry->timeline.state == 0) {
        entry->timeline.state = 2;
    }
    entry->role = role;
    entry->role_epoch = role_epoch;
    entry->owner_known = true;
    entry->timeline.cpu_baseline = scheduled && reliable_role ? cpu_now : std::nullopt;
    if (!reliable_role) {
        entry->cpu_complete = false;
    }
    if (scheduled) {
        entry->clock_generation = generation;
        if (!cpu_now) {
            entry->cpu_complete = false;
        }
    }
}

void SetWait(const ThreadInfo& info, const char* reason, u64 object, u64 detail, u64 target, u8 svc) {
    if (!Enabled()) {
        return;
    }
    auto lock = LockEvent();
    if (!lock) {
        return;
    }
    const u64 now = NowNs();
    auto* entry = FindOrRegister(info, now);
    if (!entry) {
        return;
    }
    Advance(*entry, now);
    WaitKey wait{reason ? std::string{reason} : "unspecified", object, detail, target, svc};
    if (entry->timeline.state == 4) {
        entry->suspended_wait = std::move(wait);
    } else {
        entry->wait = std::move(wait);
    }
}

void RecordWake(const ThreadInfo& waker, const ThreadInfo& waiter, u64 object, u8 svc) {
    if (!Enabled()) {
        return;
    }
    auto lock = LockEvent();
    if (!lock) {
        wake_drops.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const WakeKey key{waker.pid, waker.tid, waiter.pid, waiter.tid, object, svc};
    const auto found = wakes.find(key);
    if (found != wakes.end()) {
        ++found->second;
    } else if (wakes.size() < ThreadLimit) {
        wakes.emplace(key, 1);
    } else {
        wake_overflow.fetch_add(1, std::memory_order_relaxed);
    }
}

void Report() {
    if (!Enabled()) {
        return;
    }
    const u64 check_now = NowNs();
    const u64 deadline = next_report_ns.load(std::memory_order_relaxed);
    if (!deadline || check_now < deadline) {
        return;
    }
    std::array<CpuProbe, DkbCpuTrace::RoleCount> probes;
    for (size_t i = 0; i < probes.size(); ++i) {
        const auto before = role_owners[i].Read();
        if (!before || !before->tid) {
            continue;
        }
        const auto clock = DkbCpuTrace::TryReadThreadSample(static_cast<DkbCpuTrace::Role>(i));
        const auto after = role_owners[i].Read();
        if (after && before->sequence == after->sequence) {
            probes[i] = {clock, after->sequence};
            if (clock && clock->active) {
                role_clock_generations[i].store(clock->generation, std::memory_order_relaxed);
            }
        }
    }
    std::vector<ThreadReport> reports;
    std::map<WakeKey, u64> wake_reports;
    u64 window_ns{}, report_epoch{}, report_session{}, report_state_epoch{};
    bool window_with_loss{};
    {
        std::unique_lock lock{registry_mutex, std::try_to_lock};
        if (!lock) {
            dropped_reports.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (reset_pending.load(std::memory_order_relaxed) || !window_start_ns ||
            next_report_ns.load(std::memory_order_relaxed) != deadline) {
            return;
        }
        std::array<std::optional<OwnerSample>, DkbCpuTrace::RoleCount> owners;
        for (size_t i = 0; i < owners.size(); ++i) {
            owners[i] = role_owners[i].Read();
        }
        const u64 now = NowNs();
        window_ns = now - window_start_ns;
        report_epoch = ++epoch;
        report_session = session.load(std::memory_order_relaxed);
        reports.reserve(threads.size());
        for (auto it = threads.begin(); it != threads.end();) {
            auto& entry = it->second;
            std::optional<size_t> actual_role;
            for (size_t i = 0; i < owners.size(); ++i) {
                if (owners[i] && owners[i]->tid && owners[i]->tid == entry.info.tid) {
                    actual_role = i;
                    break;
                }
            }
            const bool same_owner = actual_role && entry.timeline.scheduled &&
                static_cast<size_t>(entry.role) == *actual_role &&
                entry.role_epoch == owners[*actual_role]->sequence;
            if (actual_role && !same_owner) {
                const auto& owner = *owners[*actual_role];
                const bool exact_start = owner.wall_ns >= entry.timeline.cursor_ns;
                entry.timeline.ReconcileOwnership(owner.wall_ns, now, true);
                entry.waits.clear();
                entry.wait_overflow_ns = 0;
                entry.timeline.cpu_baseline = exact_start ? owner.cpu_ns : std::nullopt;
                entry.role = static_cast<DkbCpuTrace::Role>(*actual_role);
                entry.role_epoch = owner.sequence;
                entry.clock_generation = owner.clock_generation;
                entry.owner_known = true;
                entry.cpu_complete = false;
            } else if (!actual_role && entry.timeline.scheduled) {
                const size_t previous_role = static_cast<size_t>(entry.role);
                const auto previous_owner = previous_role < owners.size() ? owners[previous_role] : std::nullopt;
                entry.timeline.ReconcileOwnership(previous_owner ? previous_owner->wall_ns : now, now, false);
                entry.waits.clear();
                entry.wait_overflow_ns = 0;
                entry.owner_known = previous_owner.has_value();
                entry.cpu_complete = false;
            }
            // Virtual-core handoffs in single-core mode lack a matching SwitchThread event.
            if (entry.role == DkbCpuTrace::Role::GuestSingleCore) {
                entry.timeline.ReconcileOwnership(now, now, false);
                entry.waits.clear();
                entry.wait_overflow_ns = 0;
                entry.owner_known = false;
                entry.cpu_complete = false;
            }
            Advance(entry, now);
            if (entry.timeline.scheduled) {
                const auto role = static_cast<size_t>(entry.role);
                const auto sample = role < probes.size() ? probes[role].clock : std::nullopt;
                const bool owner_matches = role < owners.size() && owners[role] &&
                    owners[role]->tid == entry.info.tid && owners[role]->sequence == entry.role_epoch &&
                    probes[role].owner_sequence == entry.role_epoch;
                if (!owner_matches || !sample || !sample->active || !sample->cpu_ns) {
                    entry.cpu_complete = false;
                    entry.timeline.cpu_baseline.reset();
                } else if (!entry.clock_generation || sample->generation != entry.clock_generation ||
                           !entry.timeline.cpu_baseline) {
                    entry.cpu_complete = false;
                    entry.timeline.cpu_baseline = sample->cpu_ns;
                    entry.clock_generation = sample->generation;
                } else if (!entry.timeline.ChargeCpu(sample->cpu_ns)) {
                    entry.cpu_complete = false;
                }
            }
            reports.push_back({entry.info, entry.timeline, std::move(entry.waits),
                               entry.wait_overflow_ns, entry.cpu_complete, entry.owner_known, entry.state_epoch});
            entry.timeline.ClearWindow();
            entry.waits.clear();
            entry.wait_overflow_ns = 0;
            entry.cpu_complete = true;
            if (entry.timeline.state == 3 && !entry.timeline.scheduled) {
                it = threads.erase(it);
            } else {
                ++it;
            }
        }
        wake_reports.swap(wakes);
        report_state_epoch = lost_event_epoch.load(std::memory_order_relaxed);
        window_with_loss = report_state_epoch != window_state_epoch;
        window_state_epoch = report_state_epoch;
        window_start_ns = now;
        next_report_ns.store(now + ReportIntervalNs, std::memory_order_relaxed);
    }
    LOG_INFO(HW_GPU, "GUEST_THREADS session={} epoch={} window_wall_ms={:.3f} threads={} dropped_events={} "
                     "dropped_reports={} capacity_drops={} wake_drops={} wake_overflow={} sampling=best_effort thread_limit={} "
                     "wait_limit_per_thread={} scheduled_wall_includes_host_wait=1 lost_event_epoch={} window_with_loss={}",
             report_session, report_epoch, window_ns / 1e6, reports.size(),
             dropped_events.exchange(0, std::memory_order_relaxed),
             dropped_reports.exchange(0, std::memory_order_relaxed),
             capacity_drops.exchange(0, std::memory_order_relaxed),
             wake_drops.exchange(0, std::memory_order_relaxed), wake_overflow.exchange(0, std::memory_order_relaxed),
             ThreadLimit, WaitLimit, report_state_epoch, window_with_loss);
    for (const auto& entry : reports) {
        LOG_INFO(HW_GPU, "GUEST_THREAD session={} epoch={} pid={} tid={} entry={:#x} priority={} main_thread={} "
                         "state={} scheduled={} scheduled_known={} scheduled_wall_ms={:.3f} runnable_ms={:.3f} blocked_ms={:.3f} "
                         "unknown_wall_ms={:.3f} wall_complete={} host_cpu_ms={} host_cpu_observed_ms={:.3f} "
                         "host_cpu_window_percent={} host_cpu_complete={} "
                         "wait_overflow_ms={:.3f}",
                 report_session, report_epoch, entry.info.pid, entry.info.tid, entry.info.entry, entry.info.priority,
                 entry.info.main_thread, entry.timeline.state, entry.timeline.scheduled, entry.owner_known,
                 entry.timeline.scheduled_wall_ns / 1e6, entry.timeline.runnable_ns / 1e6,
                 entry.timeline.blocked_ns / 1e6, entry.timeline.unknown_wall_ns / 1e6,
                 entry.timeline.wall_complete && !window_with_loss && entry.state_epoch == report_state_epoch,
                 entry.cpu_complete ? fmt::format("{:.3f}", entry.timeline.host_cpu_ns / 1e6) : "unavailable",
                 entry.timeline.host_cpu_ns / 1e6,
                 entry.cpu_complete && window_ns ? fmt::format("{:.3f}", 100.0 * entry.timeline.host_cpu_ns / window_ns) : "unavailable",
                 entry.cpu_complete, entry.wait_overflow_ns / 1e6);
        for (const auto& [wait, ns] : entry.waits) {
            LOG_INFO(HW_GPU, "GUEST_THREAD_WAIT session={} epoch={} pid={} tid={} reason={} object={:#x} detail={:#x} "
                             "target={:#x} svc={:#x} blocked_ms={:.3f}",
                     report_session, report_epoch, entry.info.pid, entry.info.tid, wait.reason, wait.object,
                     wait.detail, wait.target, wait.svc, ns / 1e6);
        }
    }
    for (const auto& [key, count] : wake_reports) {
        const auto& [waker_pid, waker_tid, waiter_pid, waiter_tid, object, svc] = key;
        LOG_INFO(HW_GPU, "GUEST_THREAD_WAKE session={} epoch={} pid_waker={} tid_waker={} pid_waiter={} "
                         "tid_waiter={} object={:#x} svc={:#x} count={}",
                 report_session, report_epoch, waker_pid, waker_tid, waiter_pid, waiter_tid, object, svc, count);
    }
}

} // namespace Common::GuestThreadProfile
