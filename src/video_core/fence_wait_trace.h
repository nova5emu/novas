// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <source_location>
#include <string_view>
#include <thread>

#include "common/dkb_frame_trace.h"
#include "common/logging.h"

namespace VideoCommon::FenceWaitTrace {

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_QUERY_WAIT_TRACE");
        return Common::DkbFrameTrace::Enabled() || (value && std::string_view{value} == "1");
    }();
    return enabled;
}

enum class Op { PendingFences, PendingReadbacks, QueryResults };

// Phases are wall time on the caller thread, not GPU execution time.
class Scope {
    using Clock = std::chrono::steady_clock;

    struct Entry {
        Op op{};
        std::source_location caller;
        bool force{};
        u64 calls{}, submitted{}, waited{}, before_submit{}, before_wait{};
        double total{}, prepare{}, submit{}, wait{}, maximum{};
    };

    struct Window {
        std::array<Entry, 64> entries{};
        size_t size{};
        u64 dropped{};
        Clock::time_point start{};
    };

public:
    explicit Scope(Op operation, bool forced = true,
                   std::source_location location = std::source_location::current())
        : enabled{Enabled()}, force{forced}, op{operation}, caller{location} {
        if (enabled) {
            start = Clock::now();
        }
    }

    void Prepared(bool callback_complete) {
        if (enabled) {
            prepared = Clock::now();
            callback_before_submit = callback_complete;
            has_submit = true;
        }
    }

    void Submitted(bool callback_complete) {
        if (enabled) {
            submitted = Clock::now();
            callback_before_wait = callback_complete;
            has_wait = true;
        }
    }

    ~Scope() {
        if (!enabled) {
            return;
        }
        const auto finish = Clock::now();
        const auto milliseconds = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        static thread_local Window window;
        if (window.start == Clock::time_point{}) {
            window.start = start;
        }
        size_t index{};
        for (; index < window.size; ++index) {
            const auto& entry = window.entries[index];
            if (entry.op == op && entry.force == force && entry.caller.line() == caller.line() &&
                std::string_view{entry.caller.file_name()} == caller.file_name()) {
                break;
            }
        }
        if (index == window.entries.size()) {
            ++window.dropped;
        } else {
            if (index == window.size) {
                window.entries[index] = {.op = op, .caller = caller, .force = force};
                ++window.size;
            }
            auto& entry = window.entries[index];
            const double total = milliseconds(start, finish);
            ++entry.calls;
            entry.total += total;
            entry.maximum = std::max(entry.maximum, total);
            entry.prepare += milliseconds(start, has_submit ? prepared : finish);
            entry.submit += has_submit ? milliseconds(prepared, has_wait ? submitted : finish) : 0;
            entry.wait += has_wait ? milliseconds(submitted, finish) : 0;
            entry.submitted += has_submit;
            entry.waited += has_wait;
            entry.before_submit += has_submit && callback_before_submit;
            entry.before_wait += has_wait && callback_before_wait;
        }
        if (finish - window.start < std::chrono::seconds{10}) {
            return;
        }
        static constexpr std::array names{"pending-fences", "pending-readbacks", "query-results"};
        const auto thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
        const double elapsed = milliseconds(window.start, finish);
        for (size_t row = 0; row < window.size; ++row) {
            const auto& entry = window.entries[row];
            LOG_INFO(Render_Vulkan,
                     "DKB query_wait thread={:x} window_ms={:.3f} op={} force={} caller={}:{} "
                     "calls={} total_ms={:.3f} max_ms={:.3f} prepare_ms={:.3f} submit_ms={:.3f} "
                     "cv_ms={:.3f} submit_calls={} wait_calls={} callback_before_submit={} "
                     "callback_before_wait={} dropped_calls={}",
                     thread, elapsed, names[static_cast<size_t>(entry.op)], entry.force,
                     entry.caller.file_name(), entry.caller.line(), entry.calls, entry.total,
                     entry.maximum, entry.prepare, entry.submit, entry.wait, entry.submitted,
                     entry.waited, entry.before_submit, entry.before_wait, window.dropped);
        }
        window.size = 0;
        window.dropped = 0;
        window.start = finish;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    bool enabled{}, force{}, has_submit{}, has_wait{};
    bool callback_before_submit{}, callback_before_wait{};
    Op op;
    std::source_location caller;
    Clock::time_point start{}, prepared{}, submitted{};
};

} // namespace VideoCommon::FenceWaitTrace
