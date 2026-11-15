// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "common/logging.h"
#include "video_core/compute_release_trace.h"

namespace VideoCommon::ComputeReleaseTrace {
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t MaxDispatches = 128;
constexpr size_t MaxReleases = 256;
constexpr size_t MaxRanges = 64;
constexpr u32 MaxLinesPerSecond = 128;
constexpr u32 MaxRangeLinesPerSecond = 32;
constexpr auto Lifetime = std::chrono::seconds{10};

u32 ReadSetting(const char* name, u32 fallback, u32 maximum) {
    const char* text = std::getenv(name);
    if (!text || !*text || *text == '-') return fallback;
    char* end{};
    const auto value = std::strtoull(text, &end, 10);
    if (*end != '\0') return fallback;
    return static_cast<u32>((std::min)(value, static_cast<unsigned long long>(maximum)));
}

struct Config {
    bool enabled{};
    u32 start_seconds{};
    u32 dispatches_per_second{};
    Clock::time_point start = Clock::now();
};

const Config& Settings() {
    static const Config config = [] {
        const char* value = std::getenv("EDEN_NX2_COMPUTE_RELEASE_TRACE");
        return Config{
            value && value[0] == '1' && value[1] == '\0',
            ReadSetting("EDEN_NX2_COMPUTE_RELEASE_TRACE_START_SECONDS", 0, 86400),
            ReadSetting("EDEN_NX2_COMPUTE_RELEASE_TRACE_MAX_DISPATCHES_PER_SECOND", 8, 64),
        };
    }();
    return config;
}

struct Metadata {
    u64 id{};
    u64 asid{};
    s32 channel{};
    u64 qmd{};
    u64 program{};
    u64 hash{};
    u64 tick{};
};

struct Dispatch {
    Metadata meta;
    Clock::time_point born;
    u64 latest_release{};
    std::vector<u64> releases;
    u32 ranges{};
    bool cpu_read{};
    bool cpu_write{};
    bool observed{};
};

struct ReleaseRecord {
    u64 id{};
    Metadata meta;
    Clock::time_point born;
    u64 gpu{};
    u64 device{};
    u32 payload{};
    bool published{};
    bool discarded{};
    bool acquired{};
    bool observed{};
    bool cpu_read{};
    bool cpu_write{};
};

struct Interval {
    u64 begin{};
    u64 end{};
    u64 prefix_end{};
    u64 dispatch{};
    u64 gpu{};
    u32 binding{};
    bool written{};
    bool readable{};
    std::string kind;
};

struct Registry {
    std::mutex mutex;
    std::array<Dispatch, MaxDispatches> dispatches;
    std::array<ReleaseRecord, MaxReleases> releases;
    std::vector<Interval> intervals;
    u64 next_dispatch{};
    u64 next_release{};
    u64 next_event{};
    u64 second = (std::numeric_limits<u64>::max)();
    Clock::time_point next_sample;
    u32 selected{};
    u32 emitted{};
    u32 range_lines{};
    u64 sampled_out{};
    u64 suppressed{};
    u64 ranges_dropped{};
    u64 range_lines_suppressed{};
    u64 lookup_truncated{};
    std::atomic<u64> contended{};
};

Registry& State() {
    static Registry state;
    return state;
}

using Lines = std::vector<std::string>;

void Output(const Lines& lines) {
    for (const auto& line : lines) LOG_INFO(HW_GPU, "COMPUTE_RELEASE_TRACE {}", line);
}

bool Active(Clock::time_point now) {
    const auto& config = Settings();
    return config.enabled && now - config.start >= std::chrono::seconds{config.start_seconds};
}

void Window(Registry& state, Clock::time_point now, Lines& lines) {
    const u64 second = std::chrono::duration_cast<std::chrono::seconds>(now - Settings().start).count();
    if (state.second != (std::numeric_limits<u64>::max)() && second <= state.second) return;
    const auto contended = state.contended.exchange(0, std::memory_order_relaxed);
    if (state.second != (std::numeric_limits<u64>::max)()) {
        lines.push_back(fmt::format(
            "quota second={} selected={} sampled_out={} emitted={} suppressed={} "
            "contended={} ranges_dropped={} range_lines_suppressed={} lookup_truncated={} coverage=sampled-hooks-only "
            "negative_is_not_proof=1",
            state.second, state.selected, state.sampled_out, state.emitted, state.suppressed,
            contended, state.ranges_dropped, state.range_lines_suppressed, state.lookup_truncated));
    } else {
        lines.push_back(fmt::format(
            "enabled start_s={} dispatches_per_s={} dispatch_slots={} release_slots={} "
            "ranges_per_dispatch={} lifetime_s=10 event_lines_per_s={} range_lines_per_s={} cpu_lookup_cap=256 "
            "coverage=sampled-hooks-only ranges=bound-overapproximation negative_is_not_proof=1",
            Settings().start_seconds, Settings().dispatches_per_second, MaxDispatches,
            MaxReleases, MaxRanges, MaxLinesPerSecond, MaxRangeLinesPerSecond));
    }
    state.second = second;
    state.selected = state.emitted = state.range_lines = 0;
    state.sampled_out = state.suppressed = state.ranges_dropped = state.lookup_truncated = 0;
    state.range_lines_suppressed = 0;
}

template <typename MakeLine>
void Emit(Registry& state, Lines& lines, MakeLine&& make_line) {
    if (state.emitted >= MaxLinesPerSecond) {
        ++state.suppressed;
        return;
    }
    ++state.emitted;
    lines.push_back(make_line(++state.next_event));
}

std::string Describe(const Metadata& meta) {
    return fmt::format("dispatch={} as={} channel={} qmd={:#x} program={:#x} shader={:016x} tick={}",
                       meta.id, meta.asid, meta.channel, meta.qmd, meta.program, meta.hash, meta.tick);
}

Dispatch* FindDispatch(Registry& state, u64 id, Clock::time_point now) {
    if (!id) return nullptr;
    auto& record = state.dispatches[id % MaxDispatches];
    return record.meta.id == id && now >= record.born && now - record.born < Lifetime ? &record : nullptr;
}

ReleaseRecord* FindRelease(Registry& state, u64 id, Clock::time_point now) {
    if (!id) return nullptr;
    auto& record = state.releases[id % MaxReleases];
    return record.id == id && now >= record.born && now - record.born < Lifetime ? &record : nullptr;
}

void Reindex(Registry& state) {
    u64 end{};
    for (auto& interval : state.intervals) {
        end = (std::max)(end, interval.end);
        interval.prefix_end = end;
    }
}

bool ValidRange(u64 begin, u64 size) {
    return size != 0 && begin <= (std::numeric_limits<u64>::max)() - size;
}

} // namespace

bool Enabled() {
    return Settings().enabled;
}

u64 Begin(u64 asid, s32 channel, u64 qmd, u64 program) {
    if (!Enabled()) return 0;
    const auto now = Clock::now();
    if (!Active(now)) return 0;
    auto& state = State();
    Lines lines;
    u64 id{};
    {
        std::scoped_lock lock{state.mutex};
        Window(state, now, lines);
        if (state.selected >= Settings().dispatches_per_second || now < state.next_sample) {
            ++state.sampled_out;
        } else {
            ++state.selected;
            state.next_sample = now + std::chrono::nanoseconds{1'000'000'000 / Settings().dispatches_per_second};
            id = ++state.next_dispatch;
            auto& record = state.dispatches[id % MaxDispatches];
            const u64 replaced = record.meta.id;
            record = {.meta = {id, asid, channel, qmd, program}, .born = now};
            std::erase_if(state.intervals, [&](const Interval& interval) {
                return interval.dispatch == replaced || !FindDispatch(state, interval.dispatch, now);
            });
            Reindex(state);
            Emit(state, lines, [&](u64 event) {
                return fmt::format("event={} begin {}", event, Describe(record.meta));
            });
        }
    }
    Output(lines);
    return id;
}

void Shader(u64 dispatch, u64 hash, u64 tick) {
    if (!dispatch || !Enabled()) return;
    const auto now = Clock::now();
    auto& state = State();
    Lines lines;
    {
        std::scoped_lock lock{state.mutex};
        Window(state, now, lines);
        if (auto* record = FindDispatch(state, dispatch, now)) {
            record->meta.hash = hash;
            record->meta.tick = tick;
            Emit(state, lines, [&](u64 event) {
                return fmt::format("event={} shader {}", event, Describe(record->meta));
            });
        }
    }
    Output(lines);
}

void Range(u64 dispatch, u64 gpu, u64 device, u64 size, const char* kind, u32 binding,
           bool written, bool readable) {
    if (!dispatch || !Enabled() || !ValidRange(device, size)) return;
    const auto now = Clock::now();
    auto& state = State();
    Lines lines;
    {
        std::scoped_lock lock{state.mutex};
        Window(state, now, lines);
        if (auto* record = FindDispatch(state, dispatch, now)) {
            if (record->ranges >= MaxRanges) {
                ++state.ranges_dropped;
            } else {
                ++record->ranges;
                Interval interval{device, device + size, 0, dispatch, gpu, binding, written,
                                  readable, kind ? kind : "unknown"};
                const auto position = std::lower_bound(state.intervals.begin(), state.intervals.end(),
                    device, [](const Interval& entry, u64 address) { return entry.begin < address; });
                state.intervals.insert(position, std::move(interval));
                Reindex(state);
                if (state.range_lines < MaxRangeLinesPerSecond) {
                    ++state.range_lines;
                    Emit(state, lines, [&](u64 event) {
                        return fmt::format("event={} range {} kind={} binding={} gpu={:#x} device={:#x} "
                                           "size={:#x} written={} readable={} extent=bound-overapproximation",
                                           event, Describe(record->meta), kind ? kind : "unknown", binding,
                                           gpu, device, size, written, readable);
                    });
                } else {
                    ++state.range_lines_suppressed;
                }
            }
        }
    }
    Output(lines);
}

u64 Release(u64 dispatch, u64 gpu, u64 device, u32 payload, u32 control, const char* source) {
    if (!dispatch || !Enabled()) return 0;
    const auto now = Clock::now();
    auto& state = State();
    Lines lines;
    u64 id{};
    {
        std::scoped_lock lock{state.mutex};
        Window(state, now, lines);
        if (auto* record = FindDispatch(state, dispatch, now)) {
            id = ++state.next_release;
            state.releases[id % MaxReleases] = {id, record->meta, now, gpu, device, payload};
            record->latest_release = id;
            std::erase_if(record->releases, [&](u64 previous) { return !FindRelease(state, previous, now); });
            record->releases.push_back(id);
            Emit(state, lines, [&](u64 event) {
                return fmt::format("event={} release={} {} gpu={:#x} device={:#x} payload={:#x} "
                                   "control={:#x} source={}", event, id, Describe(record->meta), gpu,
                                   device, payload, control, source ? source : "unknown");
            });
        }
    }
    Output(lines);
    return id;
}

void Published(u64 release, u32 value, const char* site) {
    if (!release || !Enabled()) return;
    const auto now = Clock::now();
    auto& state = State();
    Lines lines;
    {
        std::scoped_lock lock{state.mutex};
        Window(state, now, lines);
        if (auto* record = FindRelease(state, release, now)) {
            record->published = true;
            Emit(state, lines, [&](u64 event) {
                return fmt::format("event={} published release={} {} device={:#x} value={:#x} site={}",
                                   event, release, Describe(record->meta), record->device, value,
                                   site ? site : "unknown");
            });
        }
    }
    Output(lines);
}

void Discarded(u64 release, const char* reason) {
    if (!release || !Enabled()) return;
    const auto now = Clock::now();
    auto& state = State();
    Lines lines;
    {
        std::scoped_lock lock{state.mutex};
        Window(state, now, lines);
        if (auto* record = FindRelease(state, release, now)) {
            record->discarded = true;
            Emit(state, lines, [&](u64 event) {
                return fmt::format("event={} discarded release={} {} reason={}", event, release,
                                   Describe(record->meta), reason ? reason : "unknown");
            });
        }
    }
    Output(lines);
}

void ObserveCpu(u64 device, u64 size, bool write, u64 cpu_va, u32 core, const char* site) {
    if (!Enabled() || !ValidRange(device, size)) return;
    const auto now = Clock::now();
    if (!Active(now)) return;
    auto& state = State();
    Lines lines;
    {
        std::unique_lock lock{state.mutex, std::try_to_lock};
        if (!lock.owns_lock()) {
            state.contended.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Window(state, now, lines);
        const u64 end = device + size;
        auto position = std::lower_bound(state.intervals.begin(), state.intervals.end(), end,
            [](const Interval& interval, u64 address) { return interval.begin < address; });
        size_t inspected{};
        size_t releases_inspected{};
        std::array<bool, MaxDispatches> visited{};
        while (position != state.intervals.begin()) {
            --position;
            if (position->prefix_end <= device) break;
            if (++inspected > 256) {
                ++state.lookup_truncated;
                break;
            }
            if (position->end <= device || (write ? !position->readable : !position->written)) continue;
            auto* record = FindDispatch(state, position->dispatch, now);
            if (!record || visited[record->meta.id % MaxDispatches]) continue;
            visited[record->meta.id % MaxDispatches] = true;
            const auto observe = [&](ReleaseRecord* release) {
                bool& seen = release ? (write ? release->cpu_write : release->cpu_read)
                                     : (write ? record->cpu_write : record->cpu_read);
                if (seen) return;
                seen = true;
                const bool first = !record->observed;
                const bool first_release = release && !release->observed;
                record->observed = true;
                if (release) release->observed = true;
                Emit(state, lines, [&](u64 event) {
                    return fmt::format("event={} observed={} {} release={} published={} discarded={} age_us={} "
                        "first_dispatch_observation={} first_release_observation={} candidate=address-overlap "
                        "cpu_va={:#x} core={} device={:#x} size={:#x} kind={} binding={} "
                        "bound_gpu={:#x} bound_device={:#x} bound_size={:#x} site={}",
                        event, write ? "cpu-write-input" : "cpu-read-output", Describe(record->meta),
                        release ? release->id : 0, release && release->published, release && release->discarded,
                        release ? std::chrono::duration_cast<std::chrono::microseconds>(now - release->born).count() : 0,
                        first, first_release, cpu_va, core, device, size, position->kind, position->binding,
                        position->gpu, position->begin, position->end - position->begin, site ? site : "unknown");
                });
            };
            for (const u64 id : record->releases) {
                if (++releases_inspected > 256) break;
                if (auto* release = FindRelease(state, id, now)) observe(release);
            }
            if (releases_inspected > 256) {
                ++state.lookup_truncated;
                break;
            }
            // A pre-release observation has no release attribution. Later accesses can still
            // become the first observed access for each subsequently issued release.
            if (record->latest_release == 0) observe(nullptr);
        }
    }
    Output(lines);
}

void ObserveAcquire(u64 device, u64 gpu, u64 asid, s32 channel, u32 value, const char* site,
                    bool value_known) {
    if (!Enabled()) return;
    const auto now = Clock::now();
    if (!Active(now)) return;
    auto& state = State();
    Lines lines;
    {
        std::unique_lock lock{state.mutex, std::try_to_lock};
        if (!lock.owns_lock()) {
            state.contended.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Window(state, now, lines);
        for (auto& release : state.releases) {
            if (!release.id || now < release.born || now - release.born >= Lifetime || release.acquired ||
                (device ? release.device != device : release.meta.asid != asid || release.gpu != gpu)) {
                continue;
            }
            release.acquired = true;
            const bool first_release = !release.observed;
            release.observed = true;
            auto* record = FindDispatch(state, release.meta.id, now);
            const bool first = record && !record->observed;
            if (record) record->observed = true;
            Emit(state, lines, [&](u64 event) {
                return fmt::format("event={} observed=acquire-observation release={} {} consumer_as={} "
                    "consumer_channel={} gpu={:#x} device={:#x} observed_value={:#x} "
                    "observed_value_available={} payload={:#x} "
                    "published={} discarded={} age_us={} first_dispatch_observation={} first_release_observation={} "
                    "candidate=address-overlap accepted=unknown first_observed_poll=1 site={}",
                    event, release.id, Describe(release.meta), asid, channel, gpu, device, value, value_known,
                    release.payload, release.published, release.discarded,
                    std::chrono::duration_cast<std::chrono::microseconds>(now - release.born).count(),
                    first, first_release,
                    site ? site : "unknown");
            });
        }
    }
    Output(lines);
}

} // namespace VideoCommon::ComputeReleaseTrace
