// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/adpf.h"

#ifdef __ANDROID__

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include "common/logging.h"

namespace Common::ADPF {

namespace {

constexpr std::chrono::nanoseconds DEFAULT_TARGET = std::chrono::nanoseconds{16'666'667};

struct AHintManager;
struct AHintSession;

using PFN_GetManager = AHintManager* (*)();
using PFN_CreateSession = AHintSession* (*)(AHintManager*, const s32*, size_t, s64);
using PFN_CloseSession = void (*)(AHintSession*);
using PFN_UpdateTarget = int (*)(AHintSession*, s64);
using PFN_ReportActual = int (*)(AHintSession*, s64);
using PFN_SetThreads = int (*)(AHintSession*, const pid_t*, size_t);
using PFN_SetPowerEfficiency = int (*)(AHintSession*, bool);

struct Api {
    PFN_GetManager get_manager = nullptr;
    PFN_CreateSession create_session = nullptr;
    PFN_CloseSession close_session = nullptr;
    PFN_UpdateTarget update_target = nullptr;
    PFN_ReportActual report_actual = nullptr;
    PFN_SetThreads set_threads = nullptr;
    PFN_SetPowerEfficiency set_power_efficiency = nullptr;
    AHintManager* manager = nullptr;
    bool usable = false;
};

const Api& Resolve() {
    static const Api api = [] {
        Api resolved;
        void* library = dlopen("libandroid.so", RTLD_NOW);
        if (library == nullptr) {
            LOG_INFO(Common, "libandroid.so unavailable, ADPF is disabled");
            return resolved;
        }
        const auto load = [library](const char* name) { return dlsym(library, name); };

        resolved.get_manager = reinterpret_cast<PFN_GetManager>(load("APerformanceHint_getManager"));
        resolved.create_session =
            reinterpret_cast<PFN_CreateSession>(load("APerformanceHint_createSession"));
        resolved.close_session =
            reinterpret_cast<PFN_CloseSession>(load("APerformanceHint_closeSession"));
        resolved.update_target =
            reinterpret_cast<PFN_UpdateTarget>(load("APerformanceHint_updateTargetWorkDuration"));
        resolved.report_actual =
            reinterpret_cast<PFN_ReportActual>(load("APerformanceHint_reportActualWorkDuration"));
        resolved.set_threads =
            reinterpret_cast<PFN_SetThreads>(load("APerformanceHint_setThreads"));
        resolved.set_power_efficiency = reinterpret_cast<PFN_SetPowerEfficiency>(
            load("APerformanceHint_setPreferPowerEfficiency"));

        if (resolved.get_manager == nullptr || resolved.create_session == nullptr ||
            resolved.close_session == nullptr || resolved.update_target == nullptr ||
            resolved.report_actual == nullptr) {
            LOG_INFO(Common, "Performance hint API not exported, ADPF is disabled");
            return resolved;
        }

        resolved.manager = resolved.get_manager();
        if (resolved.manager == nullptr) {
            LOG_INFO(Common, "Device does not provide a performance hint manager");
            return resolved;
        }

        resolved.usable = true;
        LOG_INFO(Common, "ADPF available, setThreads {}, power efficiency {}",
                 resolved.set_threads != nullptr ? "yes" : "no",
                 resolved.set_power_efficiency != nullptr ? "yes" : "no");
        return resolved;
    }();
    return api;
}

struct SessionState {
    AHintSession* handle = nullptr;
    std::vector<pid_t> threads;
    bool unsupported = false;
    std::chrono::steady_clock::time_point last_frame{};
    int last_report_result = -1;
};

std::mutex g_mutex;
std::array<SessionState, 2> g_sessions;
std::atomic<bool> g_enabled{false};
std::atomic<s64> g_target_ns{DEFAULT_TARGET.count()};

const char* SessionName(Session session) {
    return session == Session::Render ? "render" : "background";
}

SessionState& StateOf(Session session) {
    return g_sessions[static_cast<size_t>(session)];
}

bool IsBackgroundUsable(const Api& api) {
    return api.set_power_efficiency != nullptr;
}

void CloseLocked(SessionState& state) {
    if (state.handle != nullptr) {
        Resolve().close_session(state.handle);
        state.handle = nullptr;
    }
    state.last_frame = {};
    state.last_report_result = -1;
}

AHintSession* CreateSessionFor(Session session, const std::vector<pid_t>& threads) {
    const Api& api = Resolve();
    s64 target = session == Session::Render ? g_target_ns.load(std::memory_order_relaxed) : 0;

    std::vector<s32> ids;
    ids.reserve(threads.size());
    for (const pid_t tid : threads) {
        ids.push_back(static_cast<s32>(tid));
    }

    AHintSession* handle = api.create_session(api.manager, ids.data(), ids.size(), target);
    if (handle == nullptr && target == 0) {
        target = DEFAULT_TARGET.count();
        handle = api.create_session(api.manager, ids.data(), ids.size(), target);
    }
    std::string preference_result = "unavailable";
    if (handle != nullptr && api.set_power_efficiency != nullptr) {
        preference_result = std::to_string(
            api.set_power_efficiency(handle, session == Session::Background));
    }
    LOG_INFO(Common,
             "ADPF session_create kind={} tids=[{}] target_ns={} result={} "
             "prefer_power_efficiency={} preference_result={}",
             SessionName(session), fmt::join(threads, ","), target,
             handle != nullptr ? "created" : "failed",
             session == Session::Background, preference_result);
    return handle;
}

bool SyncLocked(Session session, SessionState& state) {
    if (state.threads.empty()) {
        CloseLocked(state);
        return false;
    }

    const Api& api = Resolve();
    if (state.handle != nullptr && api.set_threads != nullptr) {
        std::vector<pid_t> ids = state.threads;
        const int result = api.set_threads(state.handle, ids.data(), ids.size());
        LOG_INFO(Common, "ADPF session_threads kind={} tids=[{}] result={}", SessionName(session),
                 fmt::join(ids, ","), result);
        if (result == 0) {
            return true;
        }
    }

    AHintSession* const replacement = CreateSessionFor(session, state.threads);
    if (replacement == nullptr) {
        if (state.handle == nullptr) {
            state.unsupported = true;
        }
        LOG_WARNING(Common, "Could not open a performance hint session for {} threads, falling back",
                    state.threads.size());
        CloseLocked(state);
        return false;
    }
    CloseLocked(state);
    state.handle = replacement;
    return true;
}

} // Anonymous namespace

void Configure() {
    const char* value = std::getenv("EDEN_NX2_ADPF");
    const bool enabled = value == nullptr || std::string_view{value} != "0";
    std::scoped_lock lock{g_mutex};
    for (SessionState& state : g_sessions) {
        CloseLocked(state);
        state.threads.clear();
        state.unsupported = false;
    }
    g_target_ns.store(DEFAULT_TARGET.count(), std::memory_order_relaxed);
    g_enabled.store(enabled, std::memory_order_release);
    LOG_INFO(Common,
             "ADPF config enabled={} target_ns={} frame_source=renderer_frame_end "
             "actual_duration_clamped=false app_restart_required=1",
             enabled, DEFAULT_TARGET.count());
    if (enabled) {
        Resolve();
    }
}

bool IsSessionSupported(Session session) {
    if (!g_enabled.load(std::memory_order_acquire)) {
        return false;
    }
    const Api& api = Resolve();
    if (!api.usable) {
        return false;
    }
    if (session == Session::Background && !IsBackgroundUsable(api)) {
        return false;
    }
    std::scoped_lock lock{g_mutex};
    return g_enabled.load(std::memory_order_relaxed) && !StateOf(session).unsupported;
}

bool AddCurrentThread(Session session) {
    if (!IsSessionSupported(session)) {
        return false;
    }

    const pid_t tid = gettid();
    std::scoped_lock lock{g_mutex};
    if (!g_enabled.load(std::memory_order_relaxed)) {
        return false;
    }

    for (size_t i = 0; i < g_sessions.size(); ++i) {
        SessionState& state = g_sessions[i];
        if (static_cast<size_t>(session) == i) {
            continue;
        }
        const auto it = std::find(state.threads.begin(), state.threads.end(), tid);
        if (it != state.threads.end()) {
            state.threads.erase(it);
            SyncLocked(static_cast<Session>(i), state);
        }
    }

    SessionState& state = StateOf(session);
    const bool added =
        std::find(state.threads.begin(), state.threads.end(), tid) == state.threads.end();
    if (added) {
        state.threads.push_back(tid);
    }
    if (!SyncLocked(session, state)) {
        if (added) {
            std::erase(state.threads, tid);
        }
        return false;
    }
    char name[16]{};
    pthread_getname_np(pthread_self(), name, sizeof(name));
    LOG_INFO(Common, "ADPF thread kind={} tid={} name={} enrolled=true", SessionName(session), tid,
             name);
    return true;
}

void RemoveCurrentThread() {
    if (!g_enabled.load(std::memory_order_acquire) || !Resolve().usable) {
        return;
    }
    const pid_t tid = gettid();
    std::scoped_lock lock{g_mutex};
    for (size_t i = 0; i < g_sessions.size(); ++i) {
        SessionState& state = g_sessions[i];
        if (std::erase(state.threads, tid) != 0) {
            SyncLocked(static_cast<Session>(i), state);
        }
    }
}

void SetTargetWorkDuration(std::chrono::nanoseconds target) {
    if (!g_enabled.load(std::memory_order_acquire)) {
        return;
    }
    const Api& api = Resolve();
    if (!api.usable || target.count() <= 0) {
        return;
    }
    if (g_target_ns.exchange(target.count(), std::memory_order_relaxed) == target.count()) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    SessionState& state = StateOf(Session::Render);
    if (state.handle != nullptr) {
        api.update_target(state.handle, target.count());
    }
}

void ReportFrameInterval() {
    if (!g_enabled.load(std::memory_order_acquire)) {
        return;
    }
    const Api& api = Resolve();
    if (!api.usable) {
        return;
    }

    std::scoped_lock lock{g_mutex};
    SessionState& state = StateOf(Session::Render);
    if (!g_enabled.load(std::memory_order_relaxed) || state.handle == nullptr) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto previous = state.last_frame;
    state.last_frame = now;
    if (previous.time_since_epoch().count() == 0) {
        return;
    }

    const s64 actual = std::chrono::duration_cast<std::chrono::nanoseconds>(now - previous).count();
    if (actual <= 0) {
        return;
    }

    const int result = api.report_actual(state.handle, actual);
    if (result != state.last_report_result) {
        LOG_INFO(Common, "ADPF report kind=render actual_ns={} target_ns={} result={}", actual,
                 g_target_ns.load(std::memory_order_relaxed), result);
        state.last_report_result = result;
    }
}

void ResetFrameInterval() {
    if (!g_enabled.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    StateOf(Session::Render).last_frame = {};
}

void Shutdown() {
    g_enabled.store(false, std::memory_order_release);
    std::scoped_lock lock{g_mutex};
    for (SessionState& state : g_sessions) {
        CloseLocked(state);
        state.threads.clear();
        state.unsupported = false;
    }
}

} // namespace Common::ADPF

#else

namespace Common::ADPF {

void Configure() {}

bool IsSessionSupported(Session) {
    return false;
}

bool AddCurrentThread(Session) {
    return false;
}

void RemoveCurrentThread() {}

void SetTargetWorkDuration(std::chrono::nanoseconds) {}

void ReportFrameInterval() {}

void ResetFrameInterval() {}

void Shutdown() {}

} // namespace Common::ADPF

#endif
