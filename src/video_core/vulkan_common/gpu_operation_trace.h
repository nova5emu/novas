// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <source_location>
#include <type_traits>
#include <utility>
#include <vulkan/vulkan_core.h>

#include "common/common_types.h"

namespace VideoCommon::GpuOperationTrace {

enum class Kind : u32 {
    Barrier,
    Copy,
    Blit,
    Clear,
    Resolve,
    RenderPassBegin,
    RenderPassEnd,
    Draw,
    Dispatch,
    Count,
};

struct Info {
    Kind kind{};
    const char* name{};
    std::source_location caller;
    u32 src_stage{};
    u32 dst_stage{};
    u32 memory_barriers{};
    u32 buffer_barriers{};
    u32 image_barriers{};
    u32 regions{};
    u64 bytes{};
};

inline constexpr u32 InvalidToken = ~u32{0};

struct Sink {
    u32 (*begin)(void*, VkCommandBuffer, const Info&){};
    void (*end)(void*, VkCommandBuffer, u32){};
    void* context{};
};

inline thread_local Sink* active{};

enum class Purpose {
    Generic,
    StagingUpload,
};

inline thread_local Purpose purpose{};

class PurposeScope {
public:
    explicit PurposeScope(Purpose value) noexcept : enabled{active != nullptr} {
        if (enabled) {
            previous = std::exchange(purpose, value);
        }
    }
    ~PurposeScope() {
        if (enabled) {
            purpose = previous;
        }
    }

    PurposeScope(const PurposeScope&) = delete;
    PurposeScope& operator=(const PurposeScope&) = delete;

private:
    Purpose previous{};
    bool enabled{};
};

// The worker owns the sink. Metadata is collected only while tracing is active.
class Scope {
public:
    template <typename MakeInfo>
        requires std::is_invocable_r_v<Info, MakeInfo>
    Scope(VkCommandBuffer command_buffer, MakeInfo&& make_info) noexcept
        : sink{active}, command{command_buffer} {
        if (sink && sink->begin && sink->end) {
            token = sink->begin(sink->context, command, std::forward<MakeInfo>(make_info)());
        }
    }

    Scope(VkCommandBuffer command_buffer, const Info& info) noexcept
        : Scope{command_buffer, [&info] { return info; }} {}

    ~Scope() {
        if (token != InvalidToken) {
            sink->end(sink->context, command, token);
        }
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Sink* sink{};
    VkCommandBuffer command{};
    u32 token{InvalidToken};
};

} // namespace VideoCommon::GpuOperationTrace
