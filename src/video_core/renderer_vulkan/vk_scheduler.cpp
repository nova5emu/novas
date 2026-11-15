// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <vector>
#include <string_view>
#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/vk_query_cache.h"

#include "common/logging.h"
#include "common/gpu_transition_profile.h"
#include "common/settings.h"
#include "common/scope_exit.h"
#include "common/dkb_cpu_trace.h"
#include "common/thread.h"
#include "video_core/cpu_readback_trace.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_command_pool.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

namespace {
bool OperationTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_OP_TRACE");
        return Common::DkbFrameTrace::Enabled() && value && std::string_view{value} == "1";
    }();
    return enabled;
}

std::atomic<u64> g_profile_submits{};
std::atomic<u64> g_profile_renderpasses{};
std::atomic<u64> g_profile_finishes{};
std::atomic<u64> g_profile_finish_ns{};
std::atomic<u64> g_profile_window_start{};
std::mutex g_profile_end_mutex;
std::map<std::string, u64> g_profile_ends;
std::map<std::string, std::pair<u64, u64>> g_profile_finish_sites; ///< site -> (count, ns)

bool SubmitTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_SUBMIT_TRACE");
        return Common::DkbFrameTrace::Enabled() || (value && std::string_view{value} == "1");
    }();
    return enabled;
}

bool SchedulerCountersEnabled() {
    return VideoCommon::Nx2Profile::Enabled() || SubmitTraceEnabled();
}

enum SubmitEvent : size_t {
    FlushCall,
    FinishCall,
    DeferAccepted,
    DeferRejected,
    DeferredFlush,
    DeferredNoop,
    WaitSubmittedCall,
    SubmitExecutionCall,
    QueueSubmit,
    SubmitWithWaiter,
    SubmitEventCount,
};

// Reasons are static strings at the call sites, as are source_location's strings.
using SubmitSite = std::tuple<std::string_view, std::string_view, u32, std::string_view>;
using SubmitCounts = std::array<u64, SubmitEventCount>;
std::mutex g_submit_trace_mutex;
std::map<SubmitSite, SubmitCounts> g_submit_trace_sites;

void CountSubmitEvent(SubmitEvent event, std::string_view reason,
                     std::source_location location) {
    if (!SubmitTraceEnabled()) {
        return;
    }
    std::scoped_lock lock{g_submit_trace_mutex};
    ++g_submit_trace_sites[{reason, location.file_name(), location.line(),
                           location.function_name()}][event];
}

std::string_view SourceBasename(std::string_view file) {
    const auto slash = file.find_last_of("/\\");
    return slash == std::string_view::npos ? file : file.substr(slash + 1);
}

void ReportSubmitReasons(double seconds) {
    if (!SubmitTraceEnabled()) {
        return;
    }
    std::map<SubmitSite, SubmitCounts> sites;
    {
        std::scoped_lock lock{g_submit_trace_mutex};
        sites.swap(g_submit_trace_sites);
    }
    for (const auto& [site, counts] : sites) {
        const auto& [reason, file, line, function] = site;
        const auto rate = [&](SubmitEvent event) {
            return static_cast<double>(counts[event]) / seconds;
        };
        LOG_INFO(Render_Vulkan,
                 "Submit causes window_s={:.3f} reason={} caller={}:{} function=\"{}\" "
                 "submits/s={:.2f} queue_submits/s={:.2f} flush_calls/s={:.2f} "
                 "finish_calls/s={:.2f} deferred_requests/s={:.2f} immediate_requests/s={:.2f} "
                 "deferred_flushes/s={:.2f} deferred_noops/s={:.2f} "
                 "wait_requests/s={:.2f} submits_with_waiter/s={:.2f}",
                 seconds, reason, SourceBasename(file), line, function,
                 rate(SubmitExecutionCall), rate(QueueSubmit), rate(FlushCall), rate(FinishCall),
                 rate(DeferAccepted), rate(DeferRejected), rate(DeferredFlush), rate(DeferredNoop),
                 rate(WaitSubmittedCall), rate(SubmitWithWaiter));
    }
}

// NovaS: where the CPU waits for the GPU to go idle (a phone ran ~100 a frame against a PC's 2).
void CountFinish(const char* file, u32 line, u64 ns) {
    std::string_view name{file};
    if (const auto slash = name.find_last_of("/\\"); slash != std::string_view::npos) {
        name.remove_prefix(slash + 1);
    }
    std::scoped_lock lock{g_profile_end_mutex};
    auto& site = g_profile_finish_sites[fmt::format("{}:{}", name, line)];
    ++site.first;
    site.second += ns;
}

void CountRenderPassEnd(const char* file, u32 line) {
    if (!SchedulerCountersEnabled()) {
        return;
    }
    std::string_view name{file};
    if (const auto slash = name.find_last_of('/'); slash != std::string_view::npos) {
        name.remove_prefix(slash + 1);
    }
    std::scoped_lock lock{g_profile_end_mutex};
    ++g_profile_ends[fmt::format("{}:{}", name, line)];
}

void ReportSubmitCounts() {
    const u64 now = VideoCommon::Nx2Profile::Now();
    u64 start = g_profile_window_start.load(std::memory_order_relaxed);
    if (start == 0) {
        g_profile_window_start.compare_exchange_strong(start, now);
        return;
    }
    if (now - start < 10'000'000'000ULL ||
        !g_profile_window_start.compare_exchange_strong(start, now)) {
        return;
    }
    const double seconds = static_cast<double>(now - start) / 1e9;
    ReportSubmitReasons(seconds);
    LOG_INFO(Render_Vulkan,
             "Scheduler per second: {:.0f} submits, {:.0f} render passes, {:.0f} finishes "
             "waiting {:.1f} ms",
             static_cast<double>(g_profile_submits.exchange(0)) / seconds,
             static_cast<double>(g_profile_renderpasses.exchange(0)) / seconds,
             static_cast<double>(g_profile_finishes.exchange(0)) / seconds,
             static_cast<double>(g_profile_finish_ns.exchange(0)) / 1e6 / seconds);
    std::vector<std::pair<u64, std::string>> ends;
    {
        std::scoped_lock lock{g_profile_end_mutex};
        for (const auto& [site, count] : g_profile_ends) {
            ends.emplace_back(count, site);
        }
        g_profile_ends.clear();
    }
    std::sort(ends.rbegin(), ends.rend());
    for (size_t i = 0; i < ends.size() && i < 16; ++i) {
        LOG_INFO(Render_Vulkan, "Render pass ends {:.0f}/s at {}",
                 static_cast<double>(ends[i].first) / seconds, ends[i].second);
    }
    std::vector<std::pair<std::pair<u64, u64>, std::string>> finishes;
    {
        std::scoped_lock lock{g_profile_end_mutex};
        for (const auto& [site, value] : g_profile_finish_sites) {
            finishes.emplace_back(value, site);
        }
        g_profile_finish_sites.clear();
    }
    std::sort(finishes.rbegin(), finishes.rend());
    for (size_t i = 0; i < finishes.size() && i < 8; ++i) {
        LOG_INFO(Render_Vulkan, "GPU finishes {:.0f}/s ({:.1f} ms/s) at {}",
                 static_cast<double>(finishes[i].first.first) / seconds,
                 static_cast<double>(finishes[i].first.second) / 1e6 / seconds,
                 finishes[i].second);
    }
}
}

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf,
                                         vk::CommandBuffer upload_cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf, upload_cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Device& device_, StateTracker& state_tracker_)
    : device{device_}, state_tracker{state_tracker_},
      master_semaphore{std::make_unique<MasterSemaphore>(device)},
      command_pool{std::make_unique<CommandPool>(*master_semaphore, device)} {

    if (Common::DkbFrameTrace::Enabled()) {
        const auto physical = device.GetPhysical();
        trace_timestamp_bits =
            physical.GetQueueFamilyProperties()[device.GetGraphicsFamily()].timestampValidBits;
        trace_timestamp_period = physical.GetProperties().limits.timestampPeriod;
        if (trace_timestamp_bits != 0 && trace_timestamp_bits <= 64 && trace_timestamp_period > 0) {
            try {
                trace_query_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = TraceMaxSubmissions * 2,
                    .pipelineStatistics = 0,
                });
                trace_submissions.resize(TraceMaxSubmissions);
                trace_work_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = TraceMaxWork * 2,
                    .pipelineStatistics = 0,
                });
                trace_work_items.resize(TraceMaxWork);
                if (OperationTraceEnabled()) {
                    trace_operation_pool = device.GetLogical().CreateQueryPool({
                        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                        .queryType = VK_QUERY_TYPE_TIMESTAMP,
                        .queryCount = TraceMaxOperations * 2,
                    });
                    trace_operations.resize(TraceMaxOperations);
                    LOG_INFO(Render_Vulkan,
                             "DKB gpu_ops enabled max_operations={} completion intervals; "
                             "timestamps perturb timing, category sums may overlap",
                             TraceMaxOperations);
                }
            } catch (const vk::Exception& error) {
                LOG_WARNING(Render_Vulkan, "DKB gpu_timestamp unavailable: {}", error.what());
            }
        } else {
            LOG_INFO(Render_Vulkan, "DKB gpu_timestamp unsupported: valid_bits={} period_ns={}",
                     trace_timestamp_bits, trace_timestamp_period);
        }
    }

    AcquireNewChunk();
    AllocateWorkerCommandBuffer();
    worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
}

Scheduler::~Scheduler() {
    if (trace_query_pool) {
        WaitWorker();
        worker_thread.request_stop();
        worker_thread.join();
        if (trace_gpu_tick != 0) {
            master_semaphore->Wait(trace_gpu_tick);
        }
    }
}

u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                     std::source_location location) {
    return FlushWithReason("caller", signal_semaphore, wait_semaphore, location);
}

u64 Scheduler::FlushWithReason(std::string_view reason, VkSemaphore signal_semaphore,
                              VkSemaphore wait_semaphore, std::source_location location) {
    CountSubmitEvent(FlushCall, reason, location);
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Flush");
    }
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore, reason, location);
    AllocateNewContext();
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    return signal_value;
}

bool Scheduler::DeferFlush(std::string_view reason, std::source_location location) {
    // Fences come one per compute release: ~11,500 a second in Hyrule Warriors: Age of
    // Imprisonment fights, each a queue submission (~22,000 a second with the others), which
    // kept the Vulkan worker busy submitting. Batched within 500 us: ~10-15% more fps there.
    // Command list copies after dispatches were ~9,000 more a second, of which the front waited
    // for about 5%.
    static const s64 interval_us = [] {
        const char* value = std::getenv("EDEN_NX2_FENCE_FLUSH_US");
        if (value != nullptr) {
            return static_cast<s64>(std::atoll(value));
        }
        return s64{500};
    }();
    if (interval_us <= 0 || !VideoCommon::GpuSplit::OnBackThread() ||
        std::chrono::steady_clock::now() - last_flush >= std::chrono::microseconds{interval_us}) {
        CountSubmitEvent(DeferRejected, reason, location);
        return false;
    }
    CountSubmitEvent(DeferAccepted, reason, location);
    flush_pending = true;
    return true;
}

void Scheduler::FlushDeferred(std::string_view reason, std::source_location location) {
    if (SubmitTraceEnabled() && reason == "idle-hook" &&
        trace_submit_wait_tick.load(std::memory_order_relaxed) >= CurrentTick()) {
        reason = "idle-hook-with-waiter";
    }
    CountSubmitEvent(flush_pending ? DeferredFlush : DeferredNoop, reason, location);
    if (flush_pending) {
        FlushWithReason(reason, {}, {}, location);
    }
}

bool Scheduler::WaitSubmitted(u64 tick, std::source_location location) {
    if (IsSubmitted(tick)) {
        return true;
    }
    if (SubmitTraceEnabled()) {
        CountSubmitEvent(WaitSubmittedCall, "wait-submitted", location);
        u64 requested = trace_submit_wait_tick.load(std::memory_order_relaxed);
        while (requested < tick && !trace_submit_wait_tick.compare_exchange_weak(
                   requested, tick, std::memory_order_relaxed)) {
        }
    }
    auto& split = VideoCommon::GpuSplit::Get();
    split.RequestIdleHook();
    // No time limit: the back can spend seconds on one entry (a large shader's translation),
    // and a fence reported as submitted before it is would release its downloads too early.
    const auto start = std::chrono::steady_clock::now();
    auto next_log = start + std::chrono::seconds{2};
    for (u32 spins = 0; !IsSubmitted(tick); ++spins) {
        if (!split.BackRunning()) {
            return IsSubmitted(tick);
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_log) {
            next_log = now + std::chrono::seconds{2};
            LOG_WARNING(Render_Vulkan, "Waited {} s for the GPU thread to submit tick {}",
                        std::chrono::duration_cast<std::chrono::seconds>(now - start).count(),
                        tick);
            split.RequestIdleHook();
        }
        if (spins < 64) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
    }
    return true;
}

void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                       std::source_location location) {
    CountSubmitEvent(FinishCall, "finish", location);
    Common::GpuTransitionProfile::WatchdogGpuScope watchdog{
        Common::GpuTransitionProfile::WatchdogPhase::GpuFinish};
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Finish");
    }
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    const bool profile = SchedulerCountersEnabled();
    const u64 start = profile ? VideoCommon::Nx2Profile::Now() : 0;
    {
        VideoCommon::CpuReadbackTrace::Scope trace{VideoCommon::CpuReadbackTrace::FinishSubmit};
        SubmitExecution(signal_semaphore, wait_semaphore, "finish", location);
    }
    {
        VideoCommon::CpuReadbackTrace::Scope trace{VideoCommon::CpuReadbackTrace::FinishWait};
        Wait(presubmit_tick);
    }
    AllocateNewContext();
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    if (profile) {
        const u64 elapsed = VideoCommon::Nx2Profile::Now() - start;
        g_profile_finishes.fetch_add(1, std::memory_order_relaxed);
        g_profile_finish_ns.fetch_add(elapsed, std::memory_order_relaxed);
        CountFinish(location.file_name(), location.line(), elapsed);
    }
}

void Scheduler::WaitWorker() {
    DispatchWork();

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    std::scoped_lock el{execution_mutex};
}

void Scheduler::MarkFrameEnd(u64 frame) {
    if (Common::DkbFrameTrace::Enabled()) {
        Record([this, frame](vk::CommandBuffer) { trace_frame_end = frame; });
    }
}

void Scheduler::DispatchWork() {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::DispatchWork");
    }
    if (chunk && !chunk->Empty()) {
        // Waking the worker is a system call (~2% of the GPU thread at one dispatch per 8
        // draws): skip it while the worker is still busy, it pops the queue before waiting.
        bool wake;
        {
            std::scoped_lock ql{queue_mutex};
            work_queue.push(std::move(chunk));
            wake = worker_waiting;
        }
        if (wake) {
            event_cv.notify_all();
        }
        AcquireNewChunk();
    }
}

void Scheduler::BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                                    const VkClearValue* clear_values, u32 clear_value_count) {
    if (SchedulerCountersEnabled()) {
        g_profile_renderpasses.fetch_add(1, std::memory_order_relaxed);
    }
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    state.renderpass = renderpass;
    state.framebuffer = framebuffer_handle;
    state.render_area = render_area;

    if (GPU::Logging::IsActive() && Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string render_pass_info =
            fmt::format("renderArea={}x{}, numImages={}", render_area.width, render_area.height,
                        framebuffer->NumImages());
        GPU::Logging::GPULogger::GetInstance().LogRenderPassBegin(render_pass_info);
    }

    std::array<VkClearValue, 9> values{};
    for (u32 i = 0; i < clear_value_count && i < values.size(); ++i) {
        values[i] = clear_values[i];
    }
    const bool after_compute = std::exchange(compute_writes_pending, false);
    Record([renderpass, framebuffer_handle, render_area, values, clear_value_count,
            after_compute](vk::CommandBuffer cmdbuf) {
        const VkRenderPassBeginInfo renderpass_bi{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = renderpass,
            .framebuffer = framebuffer_handle,
            .renderArea =
                {
                    .offset = {.x = 0, .y = 0},
                    .extent = render_area,
                },
            .clearValueCount = clear_value_count,
            .pClearValues = clear_value_count != 0 ? values.data() : nullptr,
        };
        if (after_compute) {
            // Draws read compute output as indirect arguments, vertices and resources; nothing
            // else orders those reads after the dispatch.
            static constexpr VkMemoryBarrier COMPUTE_TO_GRAPHICS{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, COMPUTE_TO_GRAPHICS);
        }
        cmdbuf.BeginRenderPass(renderpass_bi, VK_SUBPASS_CONTENTS_INLINE);
    });
    num_renderpass_images = framebuffer->NumImages();
    renderpass_images = framebuffer->Images();
    renderpass_image_ranges = framebuffer->ImageRanges();
    framebuffer->MarkResolveShadowsUpToDate();
}

void Scheduler::RealizeDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    const DeferredClear dc = deferred_clear;
    deferred_clear = {};

    std::array<VkClearValue, 9> clear_values{};
    u32 count = 0;
    const RenderPassKey& base = dc.framebuffer->RenderPassKeyBase();
    for (u32 slot = 0; slot < 8; ++slot) {
        if (base.color_formats[slot] == VideoCore::Surface::PixelFormat::Invalid) {
            continue;
        }
        clear_values[count++] = dc.color_values[slot];
    }
    if (base.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
        clear_values[count++] = dc.depth_stencil_value;
    }
    const u32 color_discard_mask =
        dc.framebuffer->DiscardsMsaaColor() ? dc.color_clear_mask : 0u;
    const bool depth_stencil_discard =
        dc.depth_stencil && dc.framebuffer->DiscardsMsaaDepthStencil();
    const VkRenderPass renderpass = dc.framebuffer->RenderPassVariant(
        dc.color_clear_mask, dc.depth_stencil, color_discard_mask, depth_stencil_discard);
    EndRenderPass();
    BeginRenderPassImpl(dc.framebuffer, renderpass, clear_values.data(), count);
}

bool Scheduler::DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot,
                                const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.color_clear_mask |= 1u << rt_slot;
    deferred_clear.color_values[rt_slot] = value;
    return true;
}

bool Scheduler::DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.depth_stencil = true;
    deferred_clear.depth_stencil_value = value;
    return true;
}

void Scheduler::FlushDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    RealizeDeferredClear();
    EndRenderPass();
}

void Scheduler::RequestRenderpass(const Framebuffer* framebuffer, std::source_location location) {
    if (deferred_clear.framebuffer == framebuffer) {
        RealizeDeferredClear();
        return;
    }
    const VkRenderPass renderpass = framebuffer->RenderPass();
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    if (renderpass == state.renderpass && framebuffer_handle == state.framebuffer &&
        render_area.width == state.render_area.width &&
        render_area.height == state.render_area.height) {
        return;
    }
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    // Ends any active pass and realizes a deferred clear
    EndRenderPass();
    BeginRenderPassImpl(framebuffer, renderpass, nullptr, 0);
}

void Scheduler::RequestOutsideRenderPassOperationContext(std::source_location location) {
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    EndRenderPass();
}

bool Scheduler::UpdateGraphicsPipeline(GraphicsPipeline* pipeline) {
    if (state.graphics_pipeline == pipeline) {
        if (pipeline && pipeline->UsesExtendedDynamicState() &&
            state.needs_state_enable_refresh) {
            state_tracker.InvalidateStateEnableFlag();
            state.needs_state_enable_refresh = false;
        }
        return false;
    }

    state.graphics_pipeline = pipeline;

    if (!pipeline) {
        return true;
    }

    if (!pipeline->UsesExtendedDynamicState()) {
        state.needs_state_enable_refresh = true;
    } else if (state.needs_state_enable_refresh) {
        state_tracker.InvalidateStateEnableFlag();
        state.needs_state_enable_refresh = false;
    }

    return true;
}

bool Scheduler::UpdateRescaling(bool is_rescaling, bool mesh_layout) {
    if (state.rescaling_defined && is_rescaling == state.is_rescaling &&
        mesh_layout == state.mesh_layout) {
        return false;
    }
    state.rescaling_defined = true;
    state.is_rescaling = is_rescaling;
    state.mesh_layout = mesh_layout;
    return true;
}

bool Scheduler::UpdateDescriptorBufferChunk(u32 descriptor_chunk) {
    if (state.descriptor_buffer_bound && descriptor_chunk == state.descriptor_buffer_chunk) {
        return false;
    }
    state.descriptor_buffer_bound = true;
    state.descriptor_buffer_chunk = descriptor_chunk;
    return true;
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    VideoCommon::GpuOperationTrace::Sink operation_sink{
        .begin = [](void* context, VkCommandBuffer command,
                    const VideoCommon::GpuOperationTrace::Info& info) {
            return static_cast<Scheduler*>(context)->BeginTraceOperation(command, info);
        },
        .end = [](void* context, VkCommandBuffer command, u32 token) {
            static_cast<Scheduler*>(context)->EndTraceOperation(command, token);
        },
        .context = this,
    };
    SCOPE_EXIT { VideoCommon::GpuOperationTrace::active = nullptr; };
    Common::SetCurrentThreadName("VulkanWorker");
    const Common::DkbCpuTrace::Registration trace_thread{Common::DkbCpuTrace::Role::VulkanWorker};
    HangTrace::BeginTick(1);
    // The first pair was begun by the constructor, before the worker had a trace tick.
    current_cmdbuf.TraceBoundary("main-begin");
    current_upload_cmdbuf.TraceBoundary("upload-begin");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            event_cv.wait(lk, stop_token, [&] {
                worker_waiting = !TryPopQueue(work);
                return !worker_waiting;
            });
            worker_waiting = false;

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            void(std::exchange(lk, std::unique_lock{execution_mutex}));

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            HangTrace::Event("chunk-execute chunk,has_submit", HangTrace::context.tick,
                {HangTrace::Handle(work.get()), has_submit ? 1ULL : 0ULL});
            if (trace_operation_pool && trace_gpu_recording) {
                VideoCommon::GpuOperationTrace::active = &operation_sink;
            }
            work->ExecuteAll(current_cmdbuf, current_upload_cmdbuf);
            if (trace_operation_pool) {
                VideoCommon::GpuOperationTrace::active = nullptr;
            }

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffer();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffer() {
    current_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    current_cmdbuf.TraceBoundary("main-begin");
    current_upload_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_upload_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    current_upload_cmdbuf.TraceBoundary("upload-begin");
    if (trace_query_pool && !trace_gpu_failed) {
        TraceGpuSubmission();
    }
}

void Scheduler::TraceShader(u64 hash, u32 stage) {
    if (hash == 0) {
        return;
    }
    for (u32 i = 0; i < trace_work.num_shaders; ++i) {
        if (trace_work.shaders[i].hash == hash && trace_work.shaders[i].stage == stage) {
            return;
        }
    }
    if (trace_work.num_shaders == trace_work.shaders.size()) {
        ++trace_work.untracked_uses;
        return;
    }
    trace_work.shaders[trace_work.num_shaders++] = {hash, stage};
}

void Scheduler::TraceGraphicsPipeline(const std::array<u64, 6>& hashes) {
    if (!trace_query_pool) {
        return;
    }
    Record([this, hashes](vk::CommandBuffer) {
        if (!trace_gpu_recording) {
            return;
        }
        ++trace_work.draw_attempts;
        for (u32 stage = 0; stage < hashes.size(); ++stage) {
            TraceShader(hashes[stage], stage);
        }
    });
}

void Scheduler::TraceComputePipeline(u64 hash) {
    if (!trace_query_pool) {
        return;
    }
    Record([this, hash](vk::CommandBuffer) {
        if (!trace_gpu_recording) {
            return;
        }
        ++trace_work.dispatch_attempts;
        TraceShader(hash, 6);
    });
}

void Scheduler::ReportWorkTimes() {
    const u32 count = trace_work_count;
    trace_work_count = 0;
    if (count == 0) {
        return;
    }
    struct Result {
        u64 timestamp;
        u64 available;
    };
    std::vector<Result> results(count * 2);
    const auto result = device.GetLogical().GetQueryResults(
        *trace_work_pool, 0, count * 2, results.size() * sizeof(Result), results.data(),
        sizeof(Result), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS) {
        return;
    }
    struct Totals {
        double total_ms{};
        double max_ms{};
        u32 count{};
        std::array<u32, 3> max_size{};
    };
    std::map<std::pair<u64, u64>, Totals> by_shaders;
    double total_ms{};
    for (u32 i = 0; i < count; ++i) {
        if (results[i * 2].available == 0 || results[i * 2 + 1].available == 0) {
            continue;
        }
        const double ms = static_cast<double>(Common::DkbFrameTrace::TimestampDelta(
                              results[i * 2].timestamp, results[i * 2 + 1].timestamp,
                              trace_timestamp_bits)) *
                          trace_timestamp_period / 1e6;
        const WorkTrace& item = trace_work_items[i];
        Totals& totals = by_shaders[{item.hash, item.fragment_hash}];
        totals.total_ms += ms;
        ++totals.count;
        if (ms > totals.max_ms) {
            totals.max_ms = ms;
            totals.max_size = item.size;
        }
        total_ms += ms;
    }
    std::vector<std::pair<std::pair<u64, u64>, Totals>> sorted(by_shaders.begin(),
                                                               by_shaders.end());
    std::ranges::sort(sorted, std::greater<>{},
                      [](const auto& entry) { return entry.second.total_ms; });
    if (sorted.size() > 8) {
        sorted.resize(8);
    }
    std::string slowest;
    for (const auto& [shaders, totals] : sorted) {
        const auto& [hash, fragment_hash] = shaders;
        slowest += fragment_hash == 0
                       ? fmt::format(" s6:{:016x}", hash)
                       : fmt::format(" s1:{:016x}+s5:{:016x}", hash, fragment_hash);
        slowest += fmt::format("={:.3f}ms/{}x(max {:.3f}ms size {}x{}x{})", totals.total_ms,
                               totals.count, totals.max_ms, totals.max_size[0],
                               totals.max_size[1], totals.max_size[2]);
    }
    LOG_INFO(Render_Vulkan,
             "DKB gpu_work renderer_frame={} timed={} timed_sum_ms={:.3f} slowest:{}",
             trace_sample_frame, count, total_ms, slowest);
}

u32 Scheduler::BeginTraceOperation(VkCommandBuffer command,
                                  const VideoCommon::GpuOperationTrace::Info& info) {
    using namespace VideoCommon::GpuOperationTrace;
    if (!trace_gpu_recording || !trace_operation_pool || trace_gpu_failed) {
        return InvalidToken;
    }
    const auto kind = static_cast<size_t>(info.kind);
    ++trace_operation_seen[kind];
    ++trace_work.operations;
    trace_work.non_barrier_operations += info.kind != Kind::Barrier;
    if (info.kind == Kind::Barrier) {
        const bool src_all = (info.src_stage & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) != 0;
        const bool dst_all = (info.dst_stage & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) != 0;
        trace_barrier_all_commands += src_all || dst_all;
        trace_barrier_full += src_all && dst_all;
        trace_memory_barriers += info.memory_barriers;
        trace_buffer_barriers += info.buffer_barriers;
        trace_image_barriers += info.image_barriers;
    }
    const bool staging = purpose == Purpose::StagingUpload && info.kind == Kind::Copy;
    trace_staging_uploads += staging;
    trace_staging_bytes += staging ? info.bytes : 0;
    if (trace_operation_count == TraceMaxOperations) {
        ++trace_operation_dropped[kind];
        return InvalidToken;
    }
    const u32 token = trace_operation_count++;
    trace_operations[token] = {info, trace_gpu_count, command == *current_upload_cmdbuf, staging};
    vk::CommandBuffer{command, device.GetDispatchLoader()}.WriteTimestamp(
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_operation_pool, token * 2);
    return token;
}

void Scheduler::EndTraceOperation(VkCommandBuffer command, u32 token) {
    vk::CommandBuffer{command, device.GetDispatchLoader()}.WriteTimestamp(
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_operation_pool, token * 2 + 1);
}

double Scheduler::TraceSignedMilliseconds(u64 begin, u64 end) const {
    return Common::DkbFrameTrace::SignedTimestampDelta(begin, end, trace_timestamp_bits) *
           trace_timestamp_period / 1e6;
}

bool Scheduler::ReportOperationTimes(
    u64 origin, double span_ms, double submission_sum_ms,
    const std::vector<std::pair<double, double>>& submission_intervals) {
    if (!trace_operation_pool) {
        return true;
    }
    struct Result {
        u64 timestamp, available;
    };
    std::vector<Result> results(trace_operation_count * 2);
    if (!results.empty()) {
        const auto result = device.GetLogical().GetQueryResults(
            *trace_operation_pool, 0, static_cast<u32>(results.size()), results.size() * sizeof(Result),
            results.data(), sizeof(Result),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_NOT_READY) {
            return false;
        }
        if (result != VK_SUCCESS) {
            LOG_WARNING(Render_Vulkan, "DKB gpu_ops query failure={}", static_cast<int>(result));
            trace_gpu_failed = true;
            return false;
        }
        if (std::ranges::any_of(results, [](const Result& entry) { return entry.available == 0; })) {
            return false;
        }
    }
    static constexpr std::array names{"barrier", "copy", "blit", "clear", "resolve", "rp_begin",
                                      "rp_end", "draw", "dispatch", "staging_upload"};
    struct Total {
        double sum{}, max{};
        u64 count{}, bytes{}, regions{};
        u32 worst_submission{};
    };
    // The staging class is a subset of copies, not an additional component of GPU time.
    using Site = std::tuple<size_t, std::string_view, std::string_view, u32, u32, u32, bool>;
    std::map<Site, Total> sites;
    std::array<double, TraceOperationKinds + 1> sums{};
    std::array<u64, TraceOperationKinds + 1> timed{};
    std::vector<std::pair<double, double>> intervals;
    u32 invalid{};
    for (u32 i = 0; i < trace_operation_count; ++i) {
        const auto& op = trace_operations[i];
        const auto& info = op.info;
        const double begin = TraceSignedMilliseconds(origin, results[i * 2].timestamp);
        const double duration = TraceSignedMilliseconds(results[i * 2].timestamp,
                                                        results[i * 2 + 1].timestamp);
        if (duration < 0.0) {
            ++invalid;
            continue;
        }
        intervals.emplace_back(begin, begin + duration);
        const auto accumulate = [&](size_t kind) {
            sums[kind] += duration;
            ++timed[kind];
            auto& total = sites[{kind, info.name, info.caller.file_name(), info.caller.line(),
                                 info.src_stage, info.dst_stage, op.upload_stream}];
            total.sum += duration;
            if (duration >= total.max) {
                total.max = duration;
                total.worst_submission = op.submission;
            }
            ++total.count;
            total.bytes += info.bytes;
            total.regions += info.regions;
        };
        accumulate(static_cast<size_t>(info.kind));
        if (op.staging_upload) {
            accumulate(TraceOperationKinds);
        }
    }
    std::string breakdown;
    double interval_sum{};
    for (size_t kind = 0; kind < names.size(); ++kind) {
        const bool staging = kind == TraceOperationKinds;
        if (!staging) {
            interval_sum += sums[kind];
        }
        breakdown += fmt::format(" {}_ms={:.3f} {}_count={} {}_timed={} {}_dropped={}",
                                 names[kind], sums[kind], names[kind],
                                 staging ? trace_staging_uploads : trace_operation_seen[kind],
                                 names[kind], timed[kind], names[kind],
                                 staging ? trace_staging_uploads - timed[kind]
                                         : trace_operation_dropped[kind]);
    }
    const double operation_coverage = Common::DkbFrameTrace::IntervalUnionMilliseconds(intervals);
    const double submission_coverage =
        Common::DkbFrameTrace::IntervalUnionMilliseconds(submission_intervals);
    intervals.insert(intervals.end(), submission_intervals.begin(), submission_intervals.end());
    const double combined_coverage = Common::DkbFrameTrace::IntervalUnionMilliseconds(std::move(intervals));
    const double covered_inside = std::clamp(operation_coverage + submission_coverage - combined_coverage,
                                             0.0, submission_coverage);
    LOG_INFO(Render_Vulkan,
             "DKB gpu_ops renderer_frame={} span_ms={:.3f} submission_sum_ms={:.3f} "
             "submission_union_ms={:.3f} uncovered_span_ms={:.3f} "
             "unattributed_inside_submissions_ms={:.3f} "
             "interval_sum_ms={:.3f} coverage_union_ms={:.3f} dropped_submissions={} "
             "invalid_pairs={} max_operations={} barrier_all_commands={} barrier_full={} "
             "memory_barriers={} buffer_barriers={} image_barriers={} "
             "staging_known_bytes={}{}",
             trace_sample_frame, span_ms, submission_sum_ms, submission_coverage,
             span_ms < 0.0 ? -1.0 : std::max(0.0, span_ms - submission_coverage),
             std::max(0.0, submission_coverage - covered_inside), interval_sum, operation_coverage,
             trace_gpu_skipped,
             invalid, TraceMaxOperations, trace_barrier_all_commands, trace_barrier_full,
             trace_memory_barriers, trace_buffer_barriers, trace_image_barriers, trace_staging_bytes,
             breakdown);
    std::vector<std::pair<Site, Total>> ranked(sites.begin(), sites.end());
    std::ranges::sort(ranked, [](const auto& lhs, const auto& rhs) {
        return lhs.second.sum > rhs.second.sum;
    });
    std::array<u32, TraceOperationKinds + 1> ranks{};
    for (const auto& [site, total] : ranked) {
        const auto& [kind, name, file, line, src, dst, upload] = site;
        if (ranks[kind] == 8) {
            continue;
        }
        LOG_INFO(Render_Vulkan,
                 "DKB gpu_op_top renderer_frame={} class={} rank={} op={} caller={}:{} "
                 "stream={} src_stage=0x{:x} dst_stage=0x{:x} sum_ms={:.3f} max_ms={:.3f} "
                 "count={} regions={} known_bytes={} worst_tick={}",
                 trace_sample_frame, names[kind], ++ranks[kind], name, file, line,
                 upload ? "upload" : "main", src, dst, total.sum, total.max, total.count,
                 total.regions, total.bytes, trace_submissions[total.worst_submission].tick);
    }
    trace_operation_count = 0;
    trace_operation_seen = {};
    trace_operation_dropped = {};
    trace_barrier_all_commands = trace_barrier_full = trace_memory_barriers = 0;
    trace_buffer_barriers = trace_image_barriers = trace_staging_uploads = trace_staging_bytes = 0;
    return true;
}

void Scheduler::TraceGpuSubmission() {
    if (trace_gpu_pending) {
        master_semaphore->Refresh();
        if (!master_semaphore->IsFree(trace_gpu_tick)) {
            return;
        }
        struct Result {
            u64 timestamp;
            u64 available;
        };
        std::array<Result, TraceMaxSubmissions * 2> results{};
        const auto result = device.GetLogical().GetQueryResults(
            *trace_query_pool, 0, trace_gpu_count * 2, trace_gpu_count * 2 * sizeof(Result),
            results.data(), sizeof(Result),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_NOT_READY) {
            return;
        }
        if (result != VK_SUCCESS) {
            LOG_WARNING(Render_Vulkan, "DKB gpu_timestamp disabled: query result {}",
                        static_cast<int>(result));
            trace_gpu_failed = true;
            return;
        }
        double total_ms{};
        std::array<std::pair<double, u32>, 4> slowest{};
        for (u32 i = 0; i < trace_gpu_count * 2; i += 2) {
            if (results[i].available == 0 || results[i + 1].available == 0) {
                return;
            }
            const auto ticks = Common::DkbFrameTrace::TimestampDelta(
                results[i].timestamp, results[i + 1].timestamp, trace_timestamp_bits);
            const double duration_ms = static_cast<double>(ticks) * trace_timestamp_period / 1e6;
            total_ms += duration_ms;
            if (duration_ms >= 10.0 && duration_ms > slowest.back().first) {
                slowest.back() = {duration_ms, i / 2};
                std::ranges::sort(slowest, std::greater<>{});
            }
        }
        const double span_ms = trace_gpu_skipped == 0
            ? static_cast<double>(Common::DkbFrameTrace::TimestampDelta(
                  results[0].timestamp, results[trace_gpu_count * 2 - 1].timestamp,
                  trace_timestamp_bits)) * trace_timestamp_period / 1e6
            : -1.0;
        std::vector<std::pair<double, double>> submission_intervals;
        if (trace_operation_pool) {
            for (u32 i = 0; i < trace_gpu_count; ++i) {
                submission_intervals.emplace_back(
                    TraceSignedMilliseconds(results[0].timestamp, results[i * 2].timestamp),
                    TraceSignedMilliseconds(results[0].timestamp, results[i * 2 + 1].timestamp));
            }
        }
        if (!ReportOperationTimes(results[0].timestamp, span_ms, total_ms, submission_intervals)) {
            return;
        }
        LOG_INFO(Render_Vulkan,
                 "DKB gpu_frame renderer_frame={} last_submission_tick={} "
                 "gpu_frame_span_ms={:.3f} gpu_submission_sum_ms={:.3f} "
                 "measured_submissions={} dropped_submissions={} "
                 "valid_bits={} period_ns={}",
                 trace_sample_frame, trace_gpu_tick, span_ms, total_ms, trace_gpu_count,
                 trace_gpu_skipped, trace_timestamp_bits, trace_timestamp_period);
        if (trace_operation_pool) {
            for (std::string_view reason : {"query-report-fence", "query-report-delay"}) {
                u32 submissions{}, no_non_barrier{}, operations{};
                double gpu_ms{};
                for (u32 i = 0; i < trace_gpu_count; ++i) {
                    const auto& submitted = trace_submissions[i];
                    if (submitted.reason != reason) {
                        continue;
                    }
                    ++submissions;
                    no_non_barrier += submitted.non_barrier_operations == 0;
                    operations += submitted.operations;
                    gpu_ms += TraceSignedMilliseconds(results[i * 2].timestamp,
                                                       results[i * 2 + 1].timestamp);
                }
                LOG_INFO(Render_Vulkan,
                         "DKB gpu_query_submit renderer_frame={} reason={} submits={} "
                         "without_non_barrier_ops={} operations={} interval_sum_ms={:.3f}",
                         trace_sample_frame, reason, submissions, no_non_barrier, operations, gpu_ms);
            }
            std::vector<std::pair<double, u32>> gaps;
            double overlap_ms{};
            for (u32 i = 0; i < trace_gpu_count; ++i) {
                if (i != 0) {
                    const double gap = TraceSignedMilliseconds(results[i * 2 - 1].timestamp,
                                                               results[i * 2].timestamp);
                    overlap_ms += std::max(0.0, -gap);
                    gaps.emplace_back(gap, i);
                }
            }
            const double covered =
                Common::DkbFrameTrace::IntervalUnionMilliseconds(std::move(submission_intervals));
            LOG_INFO(Render_Vulkan,
                     "DKB gpu_submit_gaps renderer_frame={} submission_union_ms={:.3f} "
                     "uncovered_span_ms={:.3f} adjacent_overlap_ms={:.3f} "
                     "boundaries={} dropped_submissions={}",
                     trace_sample_frame, covered, span_ms < 0.0 ? -1.0 : std::max(0.0, span_ms - covered),
                     overlap_ms, gaps.size(), trace_gpu_skipped);
            std::ranges::sort(gaps, std::greater<>{});
            for (size_t rank = 0; rank < std::min<size_t>(8, gaps.size()); ++rank) {
                const auto [gap, index] = gaps[rank];
                const auto& next = trace_submissions[index];
                const auto& previous = trace_submissions[index - 1];
                const auto ms = [](auto begin, auto end) {
                    return std::chrono::duration<double, std::milli>(end - begin).count();
                };
                LOG_INFO(Render_Vulkan,
                         "DKB gpu_gap_top renderer_frame={} rank={} gpu_gap_ms={:.3f} "
                         "previous_tick={} next_tick={} reason={} caller={}:{} "
                         "previous_reason={} previous_caller={}:{} wait_semaphore={} "
                         "request_to_worker_ms={:.3f} worker_to_submit_ms={:.3f} "
                         "submit_call_ms={:.3f} host_submit_gap_ms={:.3f}",
                         trace_sample_frame, rank + 1, gap, previous.tick, next.tick, next.reason,
                         next.caller.file_name(), next.caller.line(), previous.reason,
                         previous.caller.file_name(), previous.caller.line(), next.waits_semaphore,
                         ms(next.requested, next.worker), ms(next.worker, next.submit_begin),
                         ms(next.submit_begin, next.submit_end), ms(previous.submit_end, next.submit_begin));
            }
        }
        for (const auto& [duration_ms, index] : slowest) {
            if (duration_ms < 10.0) {
                continue;
            }
            const auto& work = trace_submissions[index];
            std::string shaders;
            for (u32 i = 0; i < work.num_shaders; ++i) {
                const auto& shader = work.shaders[i];
                shaders += fmt::format("{}s{}:{:016x}", shaders.empty() ? "" : ",",
                                       shader.stage, shader.hash);
            }
            LOG_INFO(Render_Vulkan,
                     "DKB gpu_submission renderer_frame={} tick={} gpu_ms={:.3f} "
                     "draw_attempts={} dispatch_attempts={} shaders={} untracked_uses={}",
                     trace_sample_frame, work.tick, duration_ms, work.draw_attempts,
                     work.dispatch_attempts, shaders, work.untracked_uses);
        }
        ReportWorkTimes();
        trace_gpu_tick = 0;
        trace_gpu_count = 0;
        trace_gpu_skipped = 0;
        trace_gpu_pending = false;
        trace_gpu_rate.next = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    }
    if (!trace_gpu_sampling || trace_gpu_count == TraceMaxSubmissions) {
        return;
    }
    current_upload_cmdbuf.ResetQueryPool(*trace_query_pool, trace_gpu_count * 2, 2);
    if (trace_gpu_count == 0 && trace_work_pool) {
        // Draws are timed inside render passes, where queries cannot be reset.
        current_upload_cmdbuf.ResetQueryPool(*trace_work_pool, 0, TraceMaxWork * 2);
    }
    if (trace_gpu_count == 0 && trace_operation_pool) {
        current_upload_cmdbuf.ResetQueryPool(*trace_operation_pool, 0, TraceMaxOperations * 2);
    }
    current_upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *trace_query_pool,
                                        trace_gpu_count * 2);
    trace_gpu_recording = true;
}

u64 Scheduler::SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                              std::string_view reason, std::source_location location) {
    const auto requested = trace_operation_pool ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
    CountSubmitEvent(SubmitExecutionCall, reason, location);
    if (SchedulerCountersEnabled()) {
        g_profile_submits.fetch_add(1, std::memory_order_relaxed);
        ReportSubmitCounts();
    }
    EndPendingOperations();
    InvalidateState();

    const u64 signal_value = master_semaphore->NextTick();
    if (SubmitTraceEnabled() &&
        trace_submit_wait_tick.load(std::memory_order_relaxed) >= signal_value) {
        CountSubmitEvent(SubmitWithWaiter, reason, location);
    }
    HangTrace::Event("tick-reserved", signal_value);
    RecordWithUploadBuffer([signal_semaphore, wait_semaphore, signal_value, reason, location, requested,
                            this](vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
        const auto worker = trace_operation_pool ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
        SubmissionTrace* sampled_submission{};
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WRITE_BARRIER);
        upload_cmdbuf.TraceBoundary("upload-end");
        upload_cmdbuf.End();
        if (trace_gpu_recording) {
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_query_pool,
                                 trace_gpu_count * 2 + 1);
            trace_gpu_tick = signal_value;
            trace_work.tick = signal_value;
            trace_work.reason = reason;
            trace_work.caller = location;
            trace_work.requested = requested;
            trace_work.worker = worker;
            trace_work.waits_semaphore = wait_semaphore != VK_NULL_HANDLE;
            trace_submissions[trace_gpu_count] = trace_work;
            sampled_submission = &trace_submissions[trace_gpu_count];
            ++trace_gpu_count;
            trace_gpu_recording = false;
        } else if (trace_gpu_sampling) {
            ++trace_gpu_skipped;
        }
        if (trace_query_pool) {
            trace_work = {};
        }
        cmdbuf.TraceBoundary("main-end");
        cmdbuf.End();

        if (on_submit) {
            on_submit();
        }

        std::scoped_lock lock{submit_mutex};
        if (sampled_submission && trace_operation_pool) {
            sampled_submission->submit_begin = std::chrono::steady_clock::now();
        }
        const VkResult result = master_semaphore->SubmitQueue(
            cmdbuf, upload_cmdbuf, signal_semaphore, wait_semaphore, signal_value);
        if (sampled_submission && trace_operation_pool) {
            sampled_submission->submit_end = std::chrono::steady_clock::now();
        }
        switch (result) {
        case VK_SUCCESS:
            CountSubmitEvent(QueueSubmit, reason, location);
            // Log successful queue submission
            if (GPU::Logging::IsActive() &&
                Settings::values.gpu_log_vulkan_calls.GetValue()) {
                GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                    "vkQueueSubmit", "", VK_SUCCESS);
            }
            break;
        case VK_ERROR_DEVICE_LOST:
            device.ReportLoss();
            [[fallthrough]];
        default:
            vk::Check(result);
            break;
        }
        HangTrace::BeginTick(signal_value + 1);
        if (trace_frame_end != 0) {
            if (trace_gpu_sampling) {
                trace_gpu_pending = true;
                trace_gpu_sampling = false;
            }
            if (trace_query_pool && !trace_gpu_failed && !trace_gpu_pending &&
                trace_gpu_rate.Ready(std::chrono::steady_clock::now())) {
                trace_sample_frame = trace_frame_end + 1;
                trace_gpu_sampling = true;
            }
            trace_frame_end = 0;
        }
    });
    HangTrace::Event("chunk-enqueue", signal_value);
    chunk->MarkSubmit();
    DispatchWork();
    return signal_value;
}

void Scheduler::AllocateNewContext() {
    // Enable counters once again. These are disabled when a command buffer is finished.
}

void Scheduler::InvalidateState() {
    ++recording_epoch;
    state.graphics_pipeline = nullptr;
    state.rescaling_defined = false;
    state.descriptor_buffer_bound = false;
    state_tracker.InvalidateCommandBufferState();
}

void Scheduler::EndPendingOperations() {
    // Pause, don't reset: a submission can land between a guest's counter reset and its report,
    // and resetting here would drop the samples counted so far (the report then reads too low).
    query_cache->CounterClose(VideoCommon::QueryType::ZPassPixelCount64);
    EndRenderPass();
}

void Scheduler::EndRenderPass()
    {
        RealizeDeferredClear();
        if (!state.renderpass) {
            return;
        }
        ++recording_epoch;
        fragment_barrier.Reset();
        render_pass_has_stores = false;

        query_cache->CounterClose(VideoCommon::QueryType::StreamingByteCount);

        // Log render pass end
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogRenderPassEnd();
        }

        query_cache->CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, false);
        query_cache->NotifySegment(false);

        Record([num_images = num_renderpass_images,
                       images = renderpass_images,
                       ranges = renderpass_image_ranges,
                       has_transform_feedback = device.IsExtTransformFeedbackSupported()](
                          vk::CommandBuffer cmdbuf) {
            std::array<VkImageMemoryBarrier, 9> barriers;
            for (size_t i = 0; i < num_images; ++i) {
                const VkImageSubresourceRange& range = ranges[i];
                const bool is_color = (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
                const bool is_depth_stencil = (range.aspectMask
                                              & (VK_IMAGE_ASPECT_DEPTH_BIT
                                                 | VK_IMAGE_ASPECT_STENCIL_BIT)) !=0;

                VkAccessFlags src_access = 0;

                if (is_color)
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                else if (is_depth_stencil)
                    src_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                else
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                  | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                barriers[i] = VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = src_access,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = images[i],
                        .subresourceRange = range,
                };
            }
            cmdbuf.EndRenderPass();
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   0, nullptr, nullptr, vk::Span(barriers.data(), num_images));
            if (has_transform_feedback) {
                static constexpr VkMemoryBarrier XFB_OUTPUT_BARRIER{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                    .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, XFB_OUTPUT_BARRIER);
            }
        });

        state.renderpass = VkRenderPass{};
        num_renderpass_images = 0;
    }


void Scheduler::AcquireNewChunk() {
    std::scoped_lock rl{reserve_mutex};

    if (chunk_reserve.empty()) {
        // If we don't have anything reserved, we need to make a new chunk.
        chunk = std::make_unique<CommandChunk>();
    } else {
        // Otherwise, we can just take from the reserve.
        chunk = std::move(chunk_reserve.back());
        chunk_reserve.pop_back();
    }
}

} // namespace Vulkan
