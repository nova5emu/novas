// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <limits>
#include <mutex>
#include <vector>
#include <vulkan/vulkan_core.h>
#include "video_core/vulkan_common/vulkan_hang_trace.h"

namespace Vulkan::HangTrace {

// Slots are reused only after the submission's existing fence/timeline has retired.
// The CPU never clears words that may still be written by the GPU.
struct GpuMarkers {
    static constexpr size_t SlotCount = 8192;
    static constexpr VkDeviceSize BufferSize = SlotCount * 2 * sizeof(u32);
    struct Slot {
        u64 seq{}, tick{}, command_buffer{};
        u32 token{};
    };
    VkBuffer buffer{};
    volatile const u32* mapped{};
    PFN_vkCmdWriteBufferMarkerAMD write{};
    std::atomic<u64> retired{};
    std::atomic<u64> dropped{};
    std::mutex mutex;
    std::array<Slot, SlotCount> slots{};
    size_t next{};
    u32 token{};

    void Retire(u64 tick) {
        u64 old = retired.load(std::memory_order_relaxed);
        while (old < tick && !retired.compare_exchange_weak(old, tick,
                                                          std::memory_order_relaxed)) {}
    }

    void Dump(u64 gpu, u64 latest) {
        std::unique_lock lock{mutex, std::try_to_lock};
        if (!lock.owns_lock()) {
            LOG_ERROR(Render_Vulkan, "HANG AMD markers busy; snapshot unavailable");
            return;
        }
        u64 max_after = 0, first_open = ~u64{0};
        bool any_after = false;
        for (size_t i = 0; i < SlotCount; ++i) {
            const auto& s = slots[i];
            if (!s.token) {
                continue;
            }
            // Live crash snapshot, not a synchronized readback of rendering results.
            const u32 before = mapped[2 * i];
            const u32 after = mapped[2 * i + 1];
            if (after == s.token) {
                max_after = (std::max)(max_after, s.seq);
                any_after = true;
            } else if (before == s.token) {
                first_open = (std::min)(first_open, s.seq);
            }
            if (s.tick > gpu && s.tick <= gpu + 4 && s.tick <= latest) {
                LOG_ERROR(Render_Vulkan,
                          "HANG AMD marker seq={} tick={} cmd={:x} slot={} token={} before={} after={} raw={},{}",
                          s.seq, s.tick, s.command_buffer, i, s.token,
                          before == s.token, after == s.token, before, after);
            }
        }
        LOG_ERROR(Render_Vulkan,
                  "HANG AMD summary any_after={} max_after_seq={} first_open_seq={} dropped={} "
                  "(CPU seq order; not a GPU completion frontier; UINT64_MAX means no open marker)",
                  any_after, max_after, first_open, dropped.load(std::memory_order_relaxed));
    }
};

inline std::mutex marker_registry_mutex;
inline std::vector<GpuMarkers*> marker_registry;

inline void DumpGpuMarkers(u64 gpu, u64 latest) {
    std::unique_lock lock{marker_registry_mutex, std::try_to_lock};
    if (!lock.owns_lock()) {
        LOG_ERROR(Render_Vulkan, "HANG AMD marker registry busy; snapshot unavailable");
        return;
    }
    for (auto* markers : marker_registry) {
        markers->Dump(gpu, latest);
    }
}

class MarkerScope {
public:
    MarkerScope(GpuMarkers* state_, VkCommandBuffer command_, u64 sequence)
        : command{command_} {
        if (!state_ || sequence == ~u64{0}) {
            return;
        }
        std::unique_lock lock{state_->mutex, std::try_to_lock};
        if (!lock.owns_lock()) {
            ++state_->dropped;
            return;
        }
        index = state_->next;
        auto& slot = state_->slots[index];
        if ((slot.token && slot.tick > state_->retired.load(std::memory_order_relaxed)) ||
            state_->token == std::numeric_limits<u32>::max()) {
            ++state_->dropped;
            return;
        }
        state = state_;
        token = ++state->token;
        slot = {sequence, context.tick, Handle(command), token};
        state->next = (index + 1) % GpuMarkers::SlotCount;
        state->write(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, state->buffer,
                     index * 8, token);
    }

    ~MarkerScope() {
        if (state) {
            state->write(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, state->buffer,
                         index * 8 + 4, token);
        }
    }
    MarkerScope(const MarkerScope&) = delete;
    MarkerScope& operator=(const MarkerScope&) = delete;

private:
    GpuMarkers* state{};
    VkCommandBuffer command{};
    size_t index{};
    u32 token{};
};

} // namespace Vulkan::HangTrace
