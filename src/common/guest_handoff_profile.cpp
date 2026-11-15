// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <map>
#include <set>
#include <tuple>
#include <vector>
#include "common/guest_handoff_profile.h"
#include "common/logging.h"

namespace Common::GuestHandoffProfile {
namespace {
constexpr size_t CoreCount = 6, Capacity = 8192, ActivityCapacity = 4096;
std::atomic<bool> enabled{};
std::atomic<u64> session{1}, frame{}, drops{}, ring_drops{}, activity_drops{};
std::atomic<u64> wake_calls{}, wake_accepted{}, wake_different_tid{}, wake_unknown_target{},
    wake_pid0{}, cancelled{}, wake_guest_different_tid{};
u64 Now() {
#ifdef __ANDROID__
    timespec stamp{};
    if (clock_gettime(CLOCK_MONOTONIC, &stamp) != 0) {
        return 0;
    }
    return static_cast<u64>(stamp.tv_sec) * 1'000'000'000ULL + stamp.tv_nsec;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}
struct Record {
    Timeline times;
    GuestThreadProfile::ThreadInfo info;
    u64 from_pid{}, from_tid{}, object{}, frame{}, entry_frame{}, virtual_affinity{},
        physical_affinity{};
    u32 core{};
    WakeKind kind{};
    u8 svc{};
    bool signal_endpoint{};
};
using Ring = CompletionRing<Record, Capacity>;
std::array<Ring, CoreCount> rings;
struct Activity {
    std::atomic<u64> tid{};
    GuestThreadProfile::ThreadInfo info{};
    u64 virtual_affinity{}, physical_affinity{};
    std::atomic<bool> initialized{};
    std::array<std::atomic<u64>, CoreCount> enters{};
};
std::array<Activity, ActivityCapacity> activities;
struct IrqState {
    std::atomic<bool> waiting{};
    std::atomic<u64> exit_request{}, handled_request{}, coalesced{};
    std::array<std::atomic<u64>, 10> exit_hist{}, handled_hist{};
    std::atomic<u64> idle_requests{}, running_requests{};
};
std::array<IrqState, CoreCount> irqs;
size_t Bucket(u64 value) {
    constexpr std::array<u64, 9> limits{10'000,    50'000,    100'000,   250'000,   500'000,
                                        1'000'000, 2'000'000, 5'000'000, 10'000'000};
    return std::lower_bound(limits.begin(), limits.end(), value) - limits.begin();
}
std::atomic_flag reporting = ATOMIC_FLAG_INIT;
u64 last_report{}, last_frame{};
u32 ActivitySlot(const GuestThreadProfile::ThreadInfo& info, u64 va, u64 pa) {
    for (size_t probe = 0; probe < 16; ++probe) {
        const auto index = (info.tid + probe) % ActivityCapacity;
        auto& slot = activities[index];
        u64 empty = 0;
        if (info.tid && slot.tid.compare_exchange_strong(empty, info.tid)) {
            slot.info = info;
            slot.virtual_affinity = va;
            slot.physical_affinity = pa;
            slot.initialized.store(true, std::memory_order_release);
            return static_cast<u32>(index);
        }
        if (empty == info.tid && slot.initialized.load(std::memory_order_acquire)) {
            return static_cast<u32>(index);
        }
    }
    activity_drops.fetch_add(1, std::memory_order_relaxed);
    return ~0U;
}
struct Histogram {
    std::array<u64, 10> bins{};
    u64 count{}, sum{}, maximum{}, missing{};
    void Add(u64 value) {
        constexpr std::array<u64, 9> limits{10'000,    50'000,    100'000,   250'000,   500'000,
                                            1'000'000, 2'000'000, 5'000'000, 10'000'000};
        ++bins[std::lower_bound(limits.begin(), limits.end(), value) - limits.begin()];
        ++count;
        sum += value;
        maximum = std::max(maximum, value);
    }
};
} // namespace
void Timeline::Cancel() {
    *this = {};
}
void Timeline::Stamp(size_t stage, u64 now) {
    if (stage < ns.size() && !ns[stage]) {
        ns[stage] = now;
    }
}
void Timeline::Stage(size_t stage, u32 core, u64 now) {
    if (stage == 2) {
        if (chosen_core != ~0U && chosen_core != core) {
            invalid = true;
            migrated = true;
        } else {
            chosen_core = core;
        }
    } else if (stage >= 3 && chosen_core != ~0U && core != chosen_core) {
        invalid = true;
        migrated = true;
    }
    if (stage == 5) {
        ns[5] = now;
        ns[6] = 0;
    } else {
        Stamp(stage, now);
    }
}
void Timeline::Complete(u64 entry_ns) {
    ns[6] = entry_ns;
    if (!entry_ns) {
        invalid = true;
    }
}
bool Timeline::Ordered() const {
    u64 previous = 0;
    for (const auto value : ns) {
        if (value && value < previous) {
            return false;
        }
        if (value) {
            previous = value;
        }
    }
    return !invalid;
}
bool Enabled() {
    return enabled.load(std::memory_order_relaxed);
}
void Configure() {
    enabled.store(false, std::memory_order_relaxed);
    session.fetch_add(1, std::memory_order_relaxed);
    frame = 0;
    wake_calls = 0;
    wake_accepted = 0;
    wake_different_tid = 0;
    wake_unknown_target = 0;
    wake_pid0 = 0;
    cancelled = 0;
    wake_guest_different_tid = 0;
    drops = 0;
    ring_drops = 0;
    activity_drops = 0;
    for (auto& ring : rings) {
        ring.Reset();
    }
    for (auto& slot : activities) {
        slot.tid = 0;
        slot.initialized = false;
        for (auto& count : slot.enters) {
            count = 0;
        }
    }
    for (auto& irq : irqs) {
        irq.waiting = false;
        irq.exit_request = 0;
        irq.handled_request = 0;
        irq.coalesced = 0;
        irq.idle_requests = 0;
        irq.running_requests = 0;
        for (auto& value : irq.exit_hist) {
            value = 0;
        }
        for (auto& value : irq.handled_hist) {
            value = 0;
        }
    }
    last_frame = 0;
    last_report = Now();
    const auto on = [](const char* name) {
        const char* value = std::getenv(name);
        return value && value[0] == '1' && value[1] == '\0';
    };
    enabled.store(on("EDEN_NX2_GPU_PROFILE") || on("EDEN_DKB_FRAME_TRACE"));
}
static_assert(std::atomic<u64>::is_always_lock_free);
u64 CurrentFrame() noexcept {
    return frame.load(std::memory_order_relaxed);
}
void Frame() {
    if (Enabled()) {
        frame.fetch_add(1, std::memory_order_relaxed);
    }
}
bool ThreadState::Lock() {
    if (gate.test_and_set(std::memory_order_acquire)) {
        lost_epoch.fetch_add(1, std::memory_order_relaxed);
        drops.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const auto current_session = GuestHandoffProfile::session.load(std::memory_order_relaxed);
    if (session != current_session) {
        session = current_session;
        timeline.Cancel();
        pending = false;
        activity_slot = ~0U;
        activity_session.store(current_session, std::memory_order_release);
        observed_loss = lost_epoch.load(std::memory_order_relaxed);
    }
    const auto lost = lost_epoch.load(std::memory_order_relaxed);
    if (lost != observed_loss) {
        timeline.invalid = true;
        observed_loss = lost;
    }
    return true;
}
bool ThreadState::HasPending() const {
    return Enabled() && pending.load(std::memory_order_acquire);
}
void ThreadState::Unlock() {
    gate.clear(std::memory_order_release);
}
void ThreadState::Wake(u64 pid, u64 tid, u64 obj, WakeKind why, u8 service) {
    if (!Enabled()) {
        return;
    }
    wake_calls.fetch_add(1, std::memory_order_relaxed);
    if (!Lock()) {
        return;
    }
    if (!timeline.pending) {
        wake_accepted.fetch_add(1, std::memory_order_relaxed);
        const auto target = target_tid.load(std::memory_order_relaxed);
        if (!target) {
            wake_unknown_target.fetch_add(1, std::memory_order_relaxed);
        } else if (target != tid) {
            wake_different_tid.fetch_add(1, std::memory_order_relaxed);
            if (pid && target_pid.load(std::memory_order_relaxed)) {
                wake_guest_different_tid.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (!pid) {
            wake_pid0.fetch_add(1, std::memory_order_relaxed);
        }
        timeline.Cancel();
        timeline.pending = true;
        timeline.Stamp(0, Now());
        from_pid = pid;
        from_tid = tid;
        object = obj;
        kind = why;
        svc = service;
        frame = GuestHandoffProfile::frame.load(std::memory_order_relaxed);
        pending.store(true, std::memory_order_release);
    }
    Unlock();
}
void ThreadState::Runnable(bool runnable) {
    if (!Enabled() || !Lock()) {
        return;
    }
    if (!runnable) {
        if (timeline.pending) {
            cancelled.fetch_add(1, std::memory_order_relaxed);
        }
        timeline.Cancel();
        pending.store(false, std::memory_order_release);
    } else if (timeline.pending) {
        timeline.Stamp(1, Now());
    }
    Unlock();
}
void ThreadState::Stage(size_t stage, u32 core) {
    if (!Enabled() || !pending.load(std::memory_order_acquire) || !Lock()) {
        return;
    }
    if (timeline.pending) {
        timeline.Stage(stage, core, Now());
    }
    Unlock();
}
void ThreadState::Chosen(u32 core) {
    Stage(2, core);
}
void ThreadState::CoreService(u32 core) {
    Stage(3, core);
}
void ThreadState::Switch(u32 core) {
    Stage(4, core);
}
void ThreadState::NceRun(u32 core) {
    Stage(5, core);
}
void ThreadState::Enter(const GuestThreadProfile::ThreadInfo& info, u32 core, u64 va, u64 pa,
                        u64 entry_ns, u64 entry_frame) {
    if (!Enabled() || core >= CoreCount) {
        return;
    }
    target_tid.store(info.tid, std::memory_order_relaxed);
    target_pid.store(info.pid, std::memory_order_relaxed);
    bool counted = false;
    const auto slot = activity_slot.load(std::memory_order_acquire);
    if (activity_session.load(std::memory_order_acquire) ==
            GuestHandoffProfile::session.load(std::memory_order_relaxed) &&
        slot != ~0U) {
        activities[slot].enters[core].fetch_add(1, std::memory_order_relaxed);
        counted = true;
        if (!pending.load(std::memory_order_acquire)) {
            return;
        }
    } else if (!pending.load(std::memory_order_acquire)) {
        if (!Lock()) {
            return;
        }
        if (activity_slot == ~0U) {
            activity_slot = ActivitySlot(info, va, pa);
        }
        if (activity_slot != ~0U) {
            activities[activity_slot].enters[core].fetch_add(1, std::memory_order_relaxed);
        }
        Unlock();
        return;
    }
    if (!Lock()) {
        return;
    }
    if (activity_slot == ~0U) {
        activity_slot = ActivitySlot(info, va, pa);
    }
    if (!counted && activity_slot != ~0U) {
        activities[activity_slot].enters[core].fetch_add(1, std::memory_order_relaxed);
    }
    if (timeline.pending) {
        timeline.Stage(6, core, entry_ns ? entry_ns : Now());
        timeline.Complete(entry_ns ? entry_ns : timeline.ns[6]);
        timeline.invalid |= observed_loss != lost_epoch.load(std::memory_order_relaxed);
        Record record{timeline,
                      info,
                      from_pid,
                      from_tid,
                      object,
                      frame,
                      entry_frame == ~u64{0} ? CurrentFrame() : entry_frame,
                      va,
                      pa,
                      core,
                      kind,
                      svc,
                      entry_ns != 0};
        if (!rings[core].Push(record)) {
            ring_drops.fetch_add(1, std::memory_order_relaxed);
        }
        timeline.Cancel();
        pending.store(false, std::memory_order_release);
    }
    Unlock();
}
void CoreIdle(u32 core, bool waiting) {
    if (!Enabled() || core >= CoreCount) {
        return;
    }
    auto& irq = irqs[core];
    irq.waiting.store(waiting, std::memory_order_relaxed);
    if (!waiting) {
        const auto request = irq.exit_request.exchange(0, std::memory_order_relaxed);
        const auto now = Now();
        if (request && now >= request) {
            irq.exit_hist[Bucket(now - request)].fetch_add(1, std::memory_order_relaxed);
        }
    }
}
void CoreInterrupt(u32 core) {
    if (!Enabled() || core >= CoreCount) {
        return;
    }
    auto& irq = irqs[core];
    const auto now = Now();
    const bool idle = irq.waiting.load(std::memory_order_relaxed);
    (idle ? irq.idle_requests : irq.running_requests).fetch_add(1, std::memory_order_relaxed);
    if (idle && irq.exit_request.exchange(now, std::memory_order_relaxed)) {
        irq.coalesced.fetch_add(1, std::memory_order_relaxed);
    }
    if (irq.handled_request.exchange(now, std::memory_order_relaxed)) {
        irq.coalesced.fetch_add(1, std::memory_order_relaxed);
    }
}
void CoreInterruptHandled(u32 core) {
    if (!Enabled() || core >= CoreCount) {
        return;
    }
    auto& irq = irqs[core];
    const auto request = irq.handled_request.exchange(0, std::memory_order_relaxed);
    const auto now = Now();
    if (request && now >= request) {
        irq.handled_hist[Bucket(now - request)].fetch_add(1, std::memory_order_relaxed);
    }
}
void Report() {
    if (!Enabled() || reporting.test_and_set(std::memory_order_acquire)) {
        return;
    }
    const u64 now = Now();
    if (now - last_report < 5'000'000'000ULL) {
        reporting.clear(std::memory_order_release);
        return;
    }
    const auto elapsed = now - last_report;
    last_report = now;
    std::array<std::array<Histogram, 7>, 2> hist{};
    using Key = std::tuple<u64, u64, u64, u64, u64, u8, u8, bool>;
    struct Edge {
        u64 count{}, sum{}, maximum{}, cores{}, incomplete{}, frames{};
    };
    std::map<Key, Edge> edges;
    std::vector<Record> chronological;
    std::set<u64> wake_frames;
    struct FrameCounts {
        u64 completions{}, guest_handoffs{}, guest_valid_timing{};
    };
    std::map<std::pair<u64, u64>, FrameCounts> frame_edges;
    const auto current_frame = GuestHandoffProfile::frame.load(std::memory_order_relaxed);
    const auto elapsed_frames = current_frame - last_frame;
    last_frame = current_frame;
    u64 invalid = 0, completed = 0, aggregation_drops = 0, migrated = 0;
    for (auto& ring : rings) {
        const auto available = ring.Available();
        size_t sampled = 0;
        Record record;
        for (u64 n = 0; n < available && ring.Pop(record); ++n) {
            ++completed;
            const auto frame_pair = std::pair{record.frame, record.entry_frame};
            if (frame_edges.size() < 4096 || frame_edges.contains(frame_pair)) {
                auto& counts = frame_edges[frame_pair];
                ++counts.completions;
                const bool guest_handoff = record.from_pid && record.info.pid &&
                                           record.from_tid != record.info.tid;
                counts.guest_handoffs += guest_handoff;
                counts.guest_valid_timing += guest_handoff && record.times.Ordered();
            } else {
                ++aggregation_drops;
            }
            if (record.times.migrated) {
                ++migrated;
            }
            if (!record.times.Ordered()) {
                ++invalid;
                continue;
            }
            for (size_t stage = 0; stage < 7; ++stage) {
                const size_t begin = stage ? stage - 1 : 0;
                const size_t end = stage ? stage : 6;
                const auto a = record.times.ns[begin], b = record.times.ns[end];
                if (a && b && b >= a) {
                    hist[record.signal_endpoint][stage].Add(b - a);
                } else {
                    ++hist[record.signal_endpoint][stage].missing;
                }
            }
            if (record.from_tid != record.info.tid && record.frame && wake_frames.size() < 4096) {
                wake_frames.insert(record.frame);
            }
            const Key key{record.from_pid, record.from_tid,       record.info.pid,
                          record.info.tid, record.object,         static_cast<u8>(record.kind),
                          record.svc,      record.signal_endpoint};
            if (!edges.contains(key) && edges.size() >= 4096) {
                ++aggregation_drops;
                continue;
            }
            auto& edge = edges[key];
            const auto duration = record.times.ns[6] - record.times.ns[0];
            ++edge.count;
            edge.sum += duration;
            edge.maximum = std::max(edge.maximum, duration);
            edge.cores |= 1ULL << record.core;
            if (std::any_of(record.times.ns.begin(), record.times.ns.end(),
                            [](u64 t) { return !t; })) {
                ++edge.incomplete;
            }
            edge.frames += record.frame != record.entry_frame;
            if (sampled < 8 && record.from_tid != record.info.tid) {
                chronological.push_back(record);
                ++sampled;
            }
        }
    }
    LOG_INFO(Common,
             "GUEST_HANDOFF window_ms={} completed={} invalid={} migrated={} dropped_events={} "
             "dropped_ring={} dropped_activity={} dropped_aggregation={} frames={} "
             "nontrivial_wake_frames={} frame_epoch={} "
             "boundary=per_endpoint_not_guest_first_instruction timing=observed_not_critical_path",
             elapsed / 1'000'000, completed, invalid, migrated, drops.exchange(0),
             ring_drops.exchange(0), activity_drops.exchange(0), aggregation_drops, elapsed_frames,
             wake_frames.size(), current_frame);
    LOG_INFO(
        Common,
        "GUEST_HANDOFF_WAKES calls={} accepted={} different_tid_known={} target_unknown={} "
        "waker_pid0={} cancelled={} guest_different_tid_known={} guest_filter=both_pid_nonzero "
        "counts_include_pending=true repeat_wakes_keep_first=true",
        wake_calls.exchange(0), wake_accepted.exchange(0), wake_different_tid.exchange(0),
        wake_unknown_target.exchange(0), wake_pid0.exchange(0), cancelled.exchange(0),
        wake_guest_different_tid.exchange(0));
    for (const auto& [frames, counts] : frame_edges) {
        LOG_INFO(Common,
                 "GUEST_HANDOFF_FRAME wake_frame={} entry_frame={} completions={} "
                 "guest_handoffs={} guest_valid_timing={} completed_only=true "
                 "association=adjacent_atomic_snapshot",
                 frames.first, frames.second, counts.completions, counts.guest_handoffs,
                 counts.guest_valid_timing);
    }
    constexpr std::array<const char*, 7> names{
        "wake_enter",         "signal_ready",  "ready_chosen", "chosen_hostservice",
        "hostservice_switch", "switch_ncerun", "ncerun_entry"};
    for (size_t endpoint = 0; endpoint < hist.size(); ++endpoint) {
        for (size_t i = 0; i < hist[endpoint].size(); ++i) {
            const auto& h = hist[endpoint][i];
            LOG_INFO(Common,
                     "GUEST_HANDOFF_HIST endpoint={} stage={} count={} missing={} avg_us={} "
                     "max_us={} le10={} "
                     "le50={} le100={} le250={} le500={} le1000={} le2000={} le5000={} le10000={} "
                     "gt10000={} bins=exclusive",
                     endpoint ? "signal_context_restored" : "cpp_before_trampoline", names[i],
                     h.count, h.missing, h.count ? h.sum / h.count / 1000 : 0, h.maximum / 1000,
                     h.bins[0], h.bins[1], h.bins[2], h.bins[3], h.bins[4], h.bins[5], h.bins[6],
                     h.bins[7], h.bins[8], h.bins[9]);
        }
    }
    for (const auto& [key, e] : edges) {
        const auto& [fp, ft, tp, tt, obj, kind, svc, signal_endpoint] = key;
        LOG_INFO(
            Common,
            "GUEST_HANDOFF_EDGE from_pid={} from_tid={} pid={} tid={} object={:#x} kind={} svc={} "
            "endpoint={} count={} core_mask={:#x} avg_us={} max_us={} incomplete={} "
            "frame_changed={}",
            fp, ft, tp, tt, obj, kind, svc,
            signal_endpoint ? "signal_context_restored" : "cpp_before_trampoline", e.count, e.cores,
            e.sum / e.count / 1000, e.maximum / 1000, e.incomplete, e.frames);
    }
    std::map<u64, u64> process_cores;
    for (auto& slot : activities) {
        if (!slot.initialized.load(std::memory_order_acquire)) {
            continue;
        }
        for (size_t core = 0; core < CoreCount; ++core) {
            const auto count = slot.enters[core].exchange(0, std::memory_order_relaxed);
            if (count) {
                process_cores[slot.info.pid] |= u64{1} << core;
                LOG_INFO(Common,
                         "GUEST_HANDOFF_ACTIVE pid={} tid={} entry={:#x} core={} enters={} "
                         "virtual_affinity={:#x} physical_affinity={:#x} metadata=first_execution",
                         slot.info.pid, slot.info.tid, slot.info.entry, core, count,
                         slot.virtual_affinity, slot.physical_affinity);
            }
        }
    }
    for (const auto& [pid, core_mask] : process_cores) {
        LOG_INFO(Common,
                 "GUEST_HANDOFF_CORES pid={} active_core_mask={:#x} active_cores={} "
                 "configured_cores={} activity=observed_guest_entries",
                 pid, core_mask, std::popcount(core_mask), CoreCount);
    }
    std::sort(chronological.begin(), chronological.end(),
              [](const auto& a, const auto& b) { return a.times.ns[6] < b.times.ns[6]; });
    for (const auto& r : chronological) {
        LOG_INFO(Common,
                 "GUEST_HANDOFF_CHAIN label=chronological_sample from_tid={} tid={} core={} "
                 "endpoint={} wake_frame={} entry_frame={} wake_ns={} ready_ns={} chosen_ns={} "
                 "hostservice_ns={} switch_ns={} "
                 "ncerun_ns={} entry_ns={}",
                 r.from_tid, r.info.tid, r.core,
                 r.signal_endpoint ? "signal_context_restored" : "cpp_before_trampoline", r.frame,
                 r.entry_frame, r.times.ns[0], r.times.ns[1], r.times.ns[2], r.times.ns[3],
                 r.times.ns[4], r.times.ns[5], r.times.ns[6]);
    }
    for (size_t core = 0; core < CoreCount; ++core) {
        auto& irq = irqs[core];
        std::array<u64, 10> exits{}, handled{};
        for (size_t i = 0; i < 10; ++i) {
            exits[i] = irq.exit_hist[i].exchange(0, std::memory_order_relaxed);
            handled[i] = irq.handled_hist[i].exchange(0, std::memory_order_relaxed);
        }
        LOG_INFO(
            Common,
            "GUEST_HANDOFF_IRQ core={} idle_requests={} running_requests={} coalesced_markers={} "
            "classification=approximate latest_request=true "
            "idle_exit_bins={},{},{},{},{},{},{},{},{},{} "
            "handled_bins={},{},{},{},{},{},{},{},{},{}",
            core, irq.idle_requests.exchange(0), irq.running_requests.exchange(0),
            irq.coalesced.exchange(0), exits[0], exits[1], exits[2], exits[3], exits[4], exits[5],
            exits[6], exits[7], exits[8], exits[9], handled[0], handled[1], handled[2], handled[3],
            handled[4], handled[5], handled[6], handled[7], handled[8], handled[9]);
    }
    reporting.clear(std::memory_order_release);
}
} // namespace Common::GuestHandoffProfile
