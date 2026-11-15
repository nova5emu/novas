// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <source_location>
#include <type_traits>

#include "common/common_types.h"
#include "common/logging.h"

namespace Vulkan::HangTrace {

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_HANG_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

inline bool ResourcesEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_HANG_TRACE_RESOURCES");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled && Enabled();
}

template <typename T>
inline u64 Handle(T value) {
    if constexpr (std::is_pointer_v<T>) {
        return static_cast<u64>(reinterpret_cast<uintptr_t>(value));
    } else {
        return static_cast<u64>(value);
    }
}

struct Entry {
    const char* kind{};
    u64 sequence{}, tick{}, time_ns{};
    std::array<u64, 8> data{};
    const char* file{};
    const char* function{};
    u32 line{};
};

template <size_t N>
struct Ring {
    // Never wait for another thread, including the thread being diagnosed.
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    std::atomic<u64> dropped{};
    std::array<Entry, N> entries{};
    u64 next{};

    void Push(const Entry& entry) {
        if (busy.test_and_set(std::memory_order_acquire)) {
            dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        entries[next++ % N] = entry;
        busy.clear(std::memory_order_release);
    }

    void Dump(u64 tick, bool filter) {
        if (busy.test_and_set(std::memory_order_acquire)) {
            LOG_ERROR(Render_Vulkan, "HANG ring busy; snapshot unavailable");
            return;
        }
        const auto copy = entries;
        const u64 end = next;
        busy.clear(std::memory_order_release);
        LOG_ERROR(Render_Vulkan, "HANG ring records={} overwritten={} contention_drops={}",
                  end, end > N ? end - N : 0, dropped.load(std::memory_order_relaxed));
        for (u64 i = end > N ? end - N : 0; i < end; ++i) {
            const auto& e = copy[i % N];
            if (!e.kind || (filter && e.tick != tick)) {
                continue;
            }
            LOG_ERROR(Render_Vulkan,
                      "HANG seq={} ns={} tick={} {} data={:016x},{:016x},{:016x},{:016x},"
                      "{:016x},{:016x},{:016x},{:016x} site={}:{} function={}",
                      e.sequence, e.time_ns, e.tick, e.kind, e.data[0], e.data[1], e.data[2],
                      e.data[3], e.data[4], e.data[5], e.data[6], e.data[7],
                      e.file ? e.file : "", e.line, e.function ? e.function : "");
        }
    }
};

struct State {
    std::array<Ring<256>, 64> work;
    Ring<2048> events;
    std::atomic<u64> sequence{}, latest_tick{};
    std::atomic<bool> reported{};
    std::atomic<bool> persistent_reported{};
    std::atomic_flag dumping = ATOMIC_FLAG_INIT;
};

inline State& Get() {
    static State state;
    return state;
}

struct Binding {
    u64 pipeline{}, hash{}, shader{};
};
struct Context {
    u64 tick{};
    Binding graphics, compute;
};
inline thread_local Context context;

inline auto& ResourceRings() {
    static std::array<Ring<1024>, 64> rings;
    return rings;
}

inline Entry Make(const char* kind, u64 tick, std::array<u64, 8> data) {
    return {kind, Get().sequence.fetch_add(1, std::memory_order_relaxed), tick,
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count()), data};
}

inline u64 Event(const char* kind, u64 tick = 0, std::array<u64, 8> data = {},
                 std::source_location location = std::source_location::current()) {
    if (!Enabled()) {
        return ~u64{0};
    }
    auto entry = Make(kind, tick, data);
    entry.file = location.file_name();
    entry.function = location.function_name();
    entry.line = location.line();
    Get().events.Push(entry);
    return entry.sequence;
}

// Outside the worker this is only a recording-progress snapshot, not ownership of a tick.
inline u64 QueueTick() {
    return context.tick ? context.tick : Get().latest_tick.load(std::memory_order_relaxed);
}

inline void Resource(const char* kind, std::array<u64, 8> data) {
    if (ResourcesEnabled() && context.tick != 0) {
        ResourceRings()[context.tick % 64].Push(Make(kind, context.tick, data));
    }
}

struct BufferAllocation {
    u64 generation{}, size{}, memory{}, offset{}, allocation_size{}, properties{};
    const u8* upload{};
};
inline std::mutex allocation_mutex;
inline std::unordered_map<u64, BufferAllocation> allocations;
inline u64 allocation_generation{};

inline void RegisterBuffer(u64 buffer, BufferAllocation info) {
    if (!ResourcesEnabled()) {
        return;
    }
    std::scoped_lock lock{allocation_mutex};
    info.generation = ++allocation_generation;
    allocations[buffer] = info;
}

inline void ForgetBuffer(u64 buffer) {
    if (ResourcesEnabled()) {
        std::scoped_lock lock{allocation_mutex};
        allocations.erase(buffer);
    }
}

inline BufferAllocation FindBuffer(u64 buffer) {
    std::unique_lock lock{allocation_mutex, std::try_to_lock};
    if (lock.owns_lock()) {
        if (const auto it = allocations.find(buffer); it != allocations.end()) {
            return it->second;
        }
    }
    return {};
}

inline std::array<u64, 3> SampleUpload(u64 buffer, u64 offset, u64 range) {
    std::array<u64, 3> result{};
    std::unique_lock lock{allocation_mutex, std::try_to_lock};
    if (!lock.owns_lock()) {
        return result;
    }
    const auto it = allocations.find(buffer);
    if (it == allocations.end() || !it->second.upload) {
        return result;
    }
    const auto& info = it->second;
    constexpr std::array<u64, 2> positions{0xc0, 0x150};
    for (size_t i = 0; i < positions.size(); ++i) {
        const u64 end = positions[i] + sizeof(u64);
        if (range >= end && offset <= info.size && end <= info.size - offset) {
            std::memcpy(&result[i + 1], info.upload + offset + positions[i], sizeof(u64));
            result[0] |= u64{1} << i;
        }
    }
    return result;
}

inline void BeginTick(u64 tick) {
    if (Enabled()) {
        context = {tick, {}, {}};
        Get().latest_tick.store(tick, std::memory_order_relaxed);
    }
}

inline void Bind(bool compute, u64 pipeline, u64 hash = 0, u64 shader = 0) {
    if (Enabled()) {
        (compute ? context.compute : context.graphics) = {pipeline, hash, shader};
    }
}

inline u64 Work(const char* kind, bool compute, u64 command_buffer,
                 u64 x = 0, u64 y = 0, u64 z = 0, u64 indirect_buffer = 0,
                 std::source_location location = std::source_location::current()) {
    if (!Enabled() || context.tick == 0) {
        return ~u64{0};
    }
    const auto& b = compute ? context.compute : context.graphics;
    auto entry = Make(kind, context.tick,
        {command_buffer, b.pipeline, b.hash, b.shader, x, y, z, indirect_buffer});
    entry.file = location.file_name();
    entry.function = location.function_name();
    entry.line = location.line();
    Get().work[context.tick % 64].Push(entry);
    return entry.sequence;
}

inline std::atomic<void (*)(u64, u64)> dump_gpu_markers{};

inline void OnStall(u64 gpu, u64 wait, bool persistent = false) {
    if (!Enabled()) {
        return;
    }
    auto& state = Get();
    auto& reported = persistent ? state.persistent_reported : state.reported;
    if (reported.load(std::memory_order_relaxed) ||
        state.dumping.test_and_set(std::memory_order_acquire)) {
        return;
    }
    if (reported.exchange(true, std::memory_order_relaxed)) {
        state.dumping.clear(std::memory_order_release);
        return;
    }
    const u64 latest = Get().latest_tick.load(std::memory_order_relaxed);
    LOG_ERROR(Render_Vulkan, "HANG gpu={} wait={} latest_recording_tick={} retained_ticks=64 persistent={} "
              "work_data=cmdbuf,pipeline,pipeline_hash,compute_shader,x_or_offset,y,z,indirect_buffer",
              gpu, wait, latest, persistent);
    Get().events.Dump(gpu, false);
    if (const auto dump = dump_gpu_markers.load(std::memory_order_acquire)) {
        dump(gpu, latest);
    }
    const u64 first = (std::max)(gpu + 1, latest > 63 ? latest - 63 : u64{1});
    if (first > gpu + 1) {
        LOG_ERROR(Render_Vulkan, "HANG first incomplete tick {} evicted; retained from {}", gpu + 1, first);
    }
    // Inspect the oldest unretired ticks; later submissions may already have made GPU progress.
    for (u64 tick = first; tick <= latest && tick <= gpu + 4; ++tick) {
        Get().work[tick % 64].Dump(tick, true);
        if (ResourcesEnabled()) {
            ResourceRings()[tick % 64].Dump(tick, true);
        }
    }
    state.dumping.clear(std::memory_order_release);
}

} // namespace Vulkan::HangTrace
