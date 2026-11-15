// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/caller_stacks.h"

#ifdef __ANDROID__
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <unwind.h>
#include <fmt/format.h>
#include "common/logging.h"
#include "common/gpu_transition_profile.h"
#endif

namespace Common::CallerStacks {

#ifdef __ANDROID__
namespace {
constexpr std::size_t MaxFrames = 14;
using Stack = std::array<std::uintptr_t, MaxFrames>;

struct Unwind {
    Stack* stack;
    std::size_t count;
};

_Unwind_Reason_Code Collect(_Unwind_Context* context, void* arg) {
    auto* state = static_cast<Unwind*>(arg);
    const std::uintptr_t pc = _Unwind_GetIP(context);
    if (pc == 0) {
        return _URC_END_OF_STACK;
    }
    (*state->stack)[state->count++] = pc;
    return state->count >= MaxFrames ? _URC_END_OF_STACK : _URC_NO_REASON;
}

std::uintptr_t LibraryBase() {
    static const std::uintptr_t base = [] {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&LibraryBase), &info) != 0) {
            return reinterpret_cast<std::uintptr_t>(info.dli_fbase);
        }
        return std::uintptr_t{0};
    }();
    return base;
}

std::mutex g_mutex;
std::map<std::string, std::map<Stack, std::size_t>> g_stacks;
std::map<std::string, std::size_t> g_calls;
auto g_window = std::chrono::steady_clock::now();

void ReportLocked() {
    const auto now = std::chrono::steady_clock::now();
    if (now - g_window < std::chrono::seconds{10}) {
        return;
    }
    g_window = now;
    const std::uintptr_t base = LibraryBase();
    for (auto& [site, stacks] : g_stacks) {
        std::vector<std::pair<std::size_t, const Stack*>> sorted;
        for (const auto& [stack, count] : stacks) {
            sorted.emplace_back(count, &stack);
        }
        std::sort(sorted.rbegin(), sorted.rend());
        LOG_INFO(Render_Vulkan, "Caller stacks of {}: {} calls in 10 s, library base {:#x}", site,
                 g_calls[site], base);
        for (std::size_t i = 0; i < sorted.size() && i < 4; ++i) {
            std::string frames;
            for (const std::uintptr_t pc : *sorted[i].second) {
                if (pc == 0) {
                    break;
                }
                frames += pc >= base && base != 0 ? fmt::format(" {:x}", pc - base)
                                                  : fmt::format(" !{:x}", pc);
            }
            LOG_INFO(Render_Vulkan, "  {} samples:{}", sorted[i].first, frames);
        }
    }
    g_stacks.clear();
    g_calls.clear();
}
} // namespace

void Sample(const char* site, std::size_t every) {
    static std::atomic<std::size_t> counter{};
    const bool take = counter.fetch_add(1, std::memory_order_relaxed) % every == 0;
    Stack stack{};
    if (take) {
        Unwind state{&stack, 0};
        _Unwind_Backtrace(Collect, &state);
    }
    auto lock = Common::GpuTransitionProfile::TryDiagnosticLock(
        g_mutex, Common::GpuTransitionProfile::DiagnosticSite::CallerStacks);
    if (!lock) {
        return;
    }
    ++g_calls[site];
    if (take) {
        ++g_stacks[site][stack];
    }
    ReportLocked();
}
#else
void Sample(const char*, std::size_t) {}
#endif

} // namespace Common::CallerStacks
