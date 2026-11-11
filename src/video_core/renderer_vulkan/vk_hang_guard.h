// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Phone GPUs (Adreno) can hang for good on a compute shader that runs fine on PC; the driver or Android then
// kills the app with no error. The hang guard remembers which compute shaders were dispatched for the first
// time in the submissions the GPU never finished, writes them to gpu_hang_shaders.txt in the config folder
// and, from the next boot on, skips their dispatches (logged once each). Android only.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_set>

#include "common/common_types.h"
#include "common/fs/path_util.h"
#include "common/logging.h"

namespace Vulkan::HangGuard {

#ifdef __ANDROID__

struct State {
    std::mutex mutex;
    bool loaded = false;
    bool reported = false;
    std::unordered_set<u64> blocked;
    std::unordered_set<u64> seen;
    std::unordered_set<u64> skip_logged;
    struct Recent {
        u64 hash;
        u64 tick;
    };
    std::array<Recent, 64> recent{};
    size_t recent_next = 0;
    std::string driver_key;
};

inline State& GetState() {
    static State state;
    return state;
}

inline std::string ListPath() {
    const auto dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir);
    const std::string& key = GetState().driver_key;
    if (key.empty()) {
        return Common::FS::PathToUTF8String(dir / "gpu_hang_shaders.txt");
    }
    const auto keyed = dir / ("gpu_hang_shaders-" + key + ".txt");
    // The first list had no driver in its name; it came from Qualcomm's own driver (id 8).
    std::error_code error;
    const auto legacy = dir / "gpu_hang_shaders.txt";
    if (key.starts_with("8-") && !std::filesystem::exists(keyed, error) &&
        std::filesystem::exists(legacy, error)) {
        std::filesystem::rename(legacy, keyed, error);
    }
    return Common::FS::PathToUTF8String(keyed);
}

/// Each driver keeps its own list (set when the Vulkan device is created, before any dispatch).
inline void SetDriverKey(std::string key) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    if (state.driver_key != key) {
        state.driver_key = std::move(key);
        state.loaded = false;
        state.blocked.clear();
    }
}

inline void LoadLocked(State& state) {
    if (state.loaded) {
        return;
    }
    state.loaded = true;
    if (FILE* file = std::fopen(ListPath().c_str(), "r")) {
        char line[64];
        while (std::fgets(line, sizeof(line), file)) {
            const u64 hash = std::strtoull(line, nullptr, 16);
            if (hash != 0) {
                state.blocked.insert(hash);
            }
        }
        std::fclose(file);
    }
    if (!state.blocked.empty()) {
        LOG_WARNING(Render_Vulkan, "Hang guard: {} compute shader(s) that hung the GPU before are skipped ({})",
                    state.blocked.size(), ListPath());
    }
}

/// Whether a compute shader hung the GPU in an earlier session (its disk preload is skipped too).
inline bool IsBlocked(u64 shader_hash) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    LoadLocked(state);
    return state.blocked.contains(shader_hash);
}

/// Call before recording a compute dispatch. Returns false when the dispatch must be skipped.
inline bool BeforeDispatch(u64 shader_hash, u64 tick) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    LoadLocked(state);
    if (state.blocked.contains(shader_hash)) {
        if (state.skip_logged.insert(shader_hash).second) {
            LOG_WARNING(Render_Vulkan, "Hang guard: skipping compute shader {:016x}", shader_hash);
        }
        return false;
    }
    if (state.seen.insert(shader_hash).second) {
        state.recent[state.recent_next % state.recent.size()] = {shader_hash, tick};
        ++state.recent_next;
    }
    return true;
}

/// Call when the GPU has not reached wait_tick for several seconds (it is stuck at gpu_tick).
inline void OnStall(u64 gpu_tick, u64 wait_tick) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    if (state.reported) {
        return;
    }
    state.reported = true;
    LoadLocked(state);
    std::string list;
    const size_t count = (std::min)(state.recent_next, state.recent.size());
    for (size_t i = 0; i < count; ++i) {
        const auto& entry = state.recent[i];
        // Submissions after gpu_tick never finished: a shader first dispatched there is the suspect.
        if (entry.tick > gpu_tick && entry.tick <= wait_tick && !state.blocked.contains(entry.hash)) {
            state.blocked.insert(entry.hash);
            list += fmt::format("{:016x}\n", entry.hash);
        }
    }
    if (list.empty()) {
        LOG_ERROR(Render_Vulkan, "Hang guard: GPU stuck at tick {}, no new compute shader in ticks {}..{}",
                  gpu_tick, gpu_tick + 1, wait_tick);
        return;
    }
    if (FILE* file = std::fopen(ListPath().c_str(), "a")) {
        std::fputs(list.c_str(), file);
        std::fclose(file);
    }
    LOG_ERROR(Render_Vulkan, "Hang guard: GPU stuck at tick {}; compute shaders first used after it, skipped "
              "from the next boot on:\n{}", gpu_tick, list);
}

#else

inline void SetDriverKey(std::string) {}
inline bool IsBlocked(u64) {
    return false;
}
inline bool BeforeDispatch(u64, u64) {
    return true;
}
inline void OnStall(u64, u64) {}

#endif

} // namespace Vulkan::HangGuard
