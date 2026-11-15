// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef __ANDROID__
#include <unistd.h>
#endif

#include "common/adpf.h"
#include "common/assert.h"
#include "common/gpu_transition_profile.h"
#include "common/guest_thread_profile.h"
#include "common/guest_handoff_profile.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/deferred_write_config.h"
#include "core/memory.h"
#include "core/core_timing.h"
#include "core/deferred_transition.h"
#include "core/gpu_dirty_memory_manager.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "core/hardware_properties.h"
#include "core/hle/service/nvdrv/nvdata.h"
#include "core/perf_stats.h"
#include "video_core/cdma_pusher.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/scheduler.h"
#include "video_core/cpu_readback_trace.h"
#include "video_core/compute_release_trace.h"
#include "video_core/nce_page_usage_profile.h"
#include "video_core/deferred_consumer.h"
#include "video_core/deferred_write_journal.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu.h"
#include "video_core/nce_page_usage_profile.h"
#include "video_core/gpu_write_trace.h"
#include "video_core/gpu_split.h"
#include "video_core/pending_maps.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_timestamp.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/renderer_base.h"
#include "video_core/shader_notify.h"
#include "video_core/sync_request_queue.h"

#ifdef HAS_NCE
#include <algorithm>
#include "common/page_table.h"
#include "core/arm/nce/nce_store_backing.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#endif

namespace Tegra {

struct GPU::Impl {
    explicit Impl(Core::System& system_, bool is_async_, bool use_nvdec_)
        : system{system_}
        , gpu_thread{system_}
        , use_nvdec{use_nvdec_}
        , shader_notify()
        , is_async{is_async_}
    {
#ifdef __ANDROID__
        cpu_epoch_host_page_bytes = sysconf(_SC_PAGESIZE);
#endif
        VideoCommon::GpuWriteTrace::Configure();
        Common::GpuTransitionProfile::Configure();
        Common::GuestThreadProfile::Configure();
        Common::GuestHandoffProfile::Configure();
        VideoCommon::NcePageUsageProfile::Configure();
#ifdef HAS_NCE
        if (VideoCommon::NcePageUsageProfile::CpuEpochAuditEnabled()) {
            LOG_INFO(HW_GPU,
                     "CPU_EPOCH_RUNTIME cpu_mapping_registry={} "
                     "native_quiescence=mprotect_membarrier storage_selection=properties "
                     "native_only=true uniform_enabled=false gpu_split_required=true",
                     system.DeviceMemory().CpuMappings().Enabled());
        }
#endif
        const char* value = std::getenv("EDEN_NX2_DEFERRED_WRITES");
        if (value && value[0] == '1' && value[1] == '\0') {
            deferred_writes = std::make_shared<VideoCommon::DeferredWrites::Journal>(
                Core::Hardware::NUM_CPU_CORES);
            deferred_kernel_pages = Core::ExcludeDeferredKernelPageBuffers(system, deferred_writes);
        }
    }

    ~Impl() {
        NotifyShutdown();
        system.DeviceMemory().CpuEpochs().Unbind();
        // Rasterizer teardown flushes commands and invalidates the bound engine's dirty flags.
        // Keep channels, their scheduler references, and the graphics context alive until then.
        if (renderer) {
            renderer.reset();
            // Channel-owned address spaces can release their SMMU reservations afterwards.
            // Their final unmaps must not call the destroyed cache's interface.
            system.Host1x().MemoryManager().BindInterface(nullptr);
        }
    }

    std::shared_ptr<Control::ChannelState> CreateChannel(s32 channel_id) {
        auto channel_state = std::make_shared<Tegra::Control::ChannelState>(channel_id);
        channels.emplace(channel_id, channel_state);
        scheduler.DeclareChannel(channel_state);
        return channel_state;
    }

    void BindChannel(s32 channel_id) {
        if (bound_channel != channel_id) {
            auto it = channels.find(channel_id);
            ASSERT(it != channels.end());
            bound_channel = channel_id;
            current_channel = it->second.get();
            renderer->ReadRasterizer()->BindChannel(*current_channel);
        }
    }

    std::shared_ptr<Control::ChannelState> AllocateChannel() {
        return CreateChannel(new_channel_id++);
    }

    void InitChannel(Control::ChannelState& to_init, u64 program_id) {
        to_init.Init(system, program_id);
        to_init.BindRasterizer(renderer->ReadRasterizer());
        renderer->ReadRasterizer()->InitializeChannel(to_init);
    }

    void InitAddressSpace(Tegra::MemoryManager& memory_manager) {
        memory_manager.BindRasterizer(renderer->ReadRasterizer());
    }

    void ReleaseChannel(Control::ChannelState& to_release) {
        UNIMPLEMENTED();
    }

    /// Binds a renderer to the GPU.
    void BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer_) {
        renderer = std::move(renderer_);
        system.Host1x().memory_manager.BindInterface(renderer->ReadRasterizer());
        system.Host1x().gmmu_manager.BindRasterizer(renderer->ReadRasterizer());
        BindCpuEpochBackend();
    }

    void BindCpuEpochBackend() {
        using namespace VideoCommon::CpuEpochs;
        auto& epochs = system.DeviceMemory().CpuEpochs();
        if (!epochs.Requested()) {
            return;
        }
        const auto validate_identity = [](const Identity& id) {
            return id.memory && id.memory->ValidateCpuEpochIdentity(
                id.cpu_owner, id.address, id.physical, id.generation);
        };
        epochs.Bind(
            {[this, validate_identity](const Identity& id) {
                 return deferred_writes && id.writer && id.writer->Ready() && validate_identity(id);
             },
             [](const Identity& id) {
                 return id.memory && id.memory->ReprotectCpuEpoch(
                     id.cpu_owner, id.address, id.physical, id.generation);
             },
             [this](const Identity&) {
                 if (!VideoCommon::GpuSplit::OwnsScheduler()) {
                     return CloseResult::Failed;
                 }
                 if (deferred_writes) {
                     ApplyDeferredWriteCut(deferred_writes->CaptureCut());
                 }
                 return renderer->ReadRasterizer()->ResolveDeferredWriteReadbacks() ?
                     CloseResult::Complete : CloseResult::Failed;
             },
             [](const Identity& id, std::span<u8> data) {
                 return id.memory && id.memory->CaptureCpuEpoch(
                     id.cpu_owner, id.address, id.physical, id.generation, data);
             },
             [this](const Identity& id, std::span<const u8> data) {
                 return renderer->ReadRasterizer()->UploadCpuEpochPage(id.device, id.physical, data);
             }, validate_identity},
            [this](CloseReason reason) {
                CloseResult result = CloseResult::Failed;
                auto close = [this, reason, &result] {
                    result = system.DeviceMemory().CpuEpochs().CloseAllOnOwner(reason);
                };
                if (VideoCommon::GpuSplit::OwnsScheduler()) {
                    close();
                } else if (VideoCommon::GpuSplit::OnGpuThread()) {
                    auto& split = VideoCommon::GpuSplit::Get();
                    split.Defer(close);
                    split.Drain(VideoCommon::DrainSite::Map);
                } else if (VideoCommon::GpuSplit::HasGpuThread() &&
                           !shutting_down.load(std::memory_order_acquire)) {
                    const u64 fence = RequestSyncOperation([&] {
                        close();
                        return result == CloseResult::Pending ?
                            VideoCommon::SyncRequestQueue::Status::Pending :
                            VideoCommon::SyncRequestQueue::Status::Complete;
                    });
                    gpu_thread.TickGPU(is_async);
                    WaitForSyncOperation(fence);
                }
                // The callback yields on Pending; writer release requests another GPU tick.
                return result;
            },
            [this] {
                if (VideoCommon::GpuSplit::HasGpuThread() &&
                    !shutting_down.load(std::memory_order_acquire)) {
                    gpu_thread.TickGPU(true);
                }
            });
        const bool ownership_capable = epochs.Available() && deferred_writes &&
            Core::DeferredWritesConfig::ProducerRequested() && cpu_epoch_host_page_bytes == 4096;
        LOG_INFO(HW_GPU, "CPU_EPOCH_BACKEND bound=true mode={} ownership_capable={} host_page_bytes={} "
                         "host_pending=retry owner_pending=fail_closed upload=existing_buffer_snapshot",
                 ownership_capable ? "native_storage_prototype" : "audit",
                 ownership_capable, cpu_epoch_host_page_bytes);
    }
    /// Flush all current written commands into the host GPU for execution.
    void FlushCommands() {
        renderer->ReadRasterizer()->FlushCommands();
    }

    /// Synchronizes CPU writes with Host GPU memory.
    void InvalidateGPUCache() {
        std::vector<std::pair<DAddr, u64>> writes;
        std::function<void(PAddr, size_t)> callback_writes(
            [&writes](PAddr address, size_t size) { writes.emplace_back(address, size); });
        system.GatherGPUDirtyMemory(callback_writes);
        if (!writes.empty()) {
            renderer->ReadRasterizer()->OnCacheInvalidations(writes);
        }
    }

    void ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
        for (auto& manager : system.GetGPUDirtyMemoryManager()) {
            manager.ForEachPending(callback);
        }
    }

    /// Signal the ending of command list.
    void OnCommandListEnd() {
        renderer->ReadRasterizer()->ReleaseFences(false);
        Settings::UpdateGPUAccuracy();
    }

    /// Request a host GPU memory flush from the CPU.
    template <typename Func>
    [[nodiscard]] u64 RequestSyncOperation(Func&& action) {
        std::unique_lock lck{sync_request_mutex};
        using Queue = VideoCommon::SyncRequestQueue;
        // Status-returning actions opt into independent retry and revalidate their dependencies.
        constexpr bool retryable = std::is_same_v<std::invoke_result_t<Func&>, Queue::Status>;
        auto operation = [action = std::forward<Func>(action)]() mutable {
            if constexpr (retryable) {
                return action();
            } else {
                action();
                return Queue::Status::Complete;
            }
        };
        return sync_requests.Push(std::move(operation), retryable ? Queue::Ordering::Independent :
                                                                  Queue::Ordering::AfterPrior);
    }

    /// Obtains current flush request fence id.
    [[nodiscard]] u64 CurrentSyncRequestFence() const {
        // Independent requests may complete beyond this contiguous prefix.
        return current_sync_fence.load(std::memory_order_acquire);
    }

    void WaitForSyncOperation(const u64 fence) {
        std::unique_lock lck{sync_request_mutex};
        sync_request_cv.wait(lck, [this, fence] {
            return CurrentSyncRequestFence() >= fence || sync_requests.IsComplete(fence);
        });
    }

    void WaitForIdle() {
        const u64 fence = RequestSyncOperation([] {});
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(fence);
    }

    /// Tick pending requests within the GPU.
    void TickWork() {
        namespace Profile = Common::GpuTransitionProfile;
        Profile::WatchdogGpuScope work{Profile::WatchdogPhase::GpuTickWork};
        std::unique_lock lck{sync_request_mutex, std::defer_lock};
        {
            Profile::WatchdogGpuScope waiting{Profile::WatchdogPhase::GpuLockAcquire};
            lck.lock();
        }
        // Retry a round only when completion can unblock earlier ordered requests.
        bool made_progress = true;
        while (made_progress && sync_requests.PendingCount() != 0) {
            made_progress = false;
            const size_t attempts = sync_requests.PendingCount();
            for (size_t i = 0; i < attempts; ++i) {
                auto request = sync_requests.Pop();
                if (!request) break;
                auto status = VideoCommon::SyncRequestQueue::Status::Pending;
                const bool can_run = sync_requests.CanRun(*request);
                lck.unlock();
                if (can_run) {
                    Profile::WatchdogGpuScope callback{Profile::WatchdogPhase::GpuCallback};
                    auto& epochs = system.DeviceMemory().CpuEpochs();
                    if (epochs.Requested() && epochs.ActivePages()) {
                        const auto closed = epochs.CloseAllOnOwner(VideoCommon::CpuEpochs::CloseReason::Consumer);
                        if (closed == VideoCommon::CpuEpochs::CloseResult::Failed) {
                            LOG_CRITICAL(HW_GPU, "CPU_EPOCH_BOUNDARY failed site=gpu_callback");
                            throw std::runtime_error("CPU epoch closure failed before GPU callback");
                        }
                        if (closed == VideoCommon::CpuEpochs::CloseResult::Complete) status = request->action();
                    } else {
                        status = request->action();
                    }
                }
                {
                    Profile::WatchdogGpuScope waiting{Profile::WatchdogPhase::GpuLockAcquire};
                    lck.lock();
                }
                sync_requests.Resolve(std::move(*request), status);
                made_progress |= status == VideoCommon::SyncRequestQueue::Status::Complete;
                current_sync_fence.store(sync_requests.CompletedThrough(), std::memory_order_release);
                sync_request_cv.notify_all();
            }
        }
        lck.unlock();
        system.DeviceMemory().CpuEpochs().Report();
    }
    [[nodiscard]] u64 GetTicks() const {
        return ReadGpuTimestamp(system.CoreTiming(), g_nx2_ampere.load(std::memory_order_relaxed),
                                Settings::values.gpu_clock.GetValue());
    }

    void RendererFrameEndNotify() {
        system.GetPerfStats().EndGameFrame();
        Common::GuestHandoffProfile::Frame();
        Common::ADPF::ReportFrameInterval();
    }

    /// Performs any additional setup necessary in order to begin GPU emulation.
    /// This can be used to launch any necessary threads and register any necessary
    /// core timing events.
    void Start() {
        Settings::UpdateGPUAccuracy();
        gpu_thread.StartThread(*renderer, renderer->Context(), scheduler);
    }

    void NotifyShutdown() {
        gpu_thread.NotifyShutdown();
        std::unique_lock lk{sync_mutex};
        shutting_down.store(true, std::memory_order::relaxed);
        sync_cv.notify_all();
    }

    /// Obtain the CPU Context
    void ObtainContext() {
        if (!cpu_context) {
            cpu_context = renderer->GetRenderWindow().CreateSharedContext();
        }
        cpu_context->MakeCurrent();
    }

    /// Release the CPU Context
    void ReleaseContext() {
        cpu_context->DoneCurrent();
    }

    /// Push GPU command entries to be processed
    void PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
        gpu_thread.SubmitList(channel, std::move(entries), is_async);
    }

    void RunInOrder(std::function<void()>&& func) {
        gpu_thread.Run(std::move(func), is_async);
    }

    void MapInOrder(GPUVAddr gpu_addr, u64 size, std::function<void()>&& func) {
        const u64 id = pending_maps.Add(gpu_addr, size);
        gpu_thread.Run(
            [this, id, func = std::move(func)] {
                func();
                pending_maps.Remove(id);
            },
            is_async, true);
    }

    bool IsMapPending(GPUVAddr gpu_addr, u64 size) const {
        return pending_maps.Overlaps(gpu_addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    void FlushRegion(DAddr addr, u64 size) {
        gpu_thread.FlushRegion(addr, size, is_async);
    }

    VideoCore::RasterizerDownloadArea OnCPURead(DAddr addr, u64 size) {
        VideoCommon::ComputeReleaseTrace::ObserveCpu(addr, size, false, 0,
            static_cast<u32>(system.GetCurrentHostThreadID()), "OnCPURead-intent");
        namespace Trace = VideoCommon::CpuReadbackTrace;
        const bool trace = Trace::Enabled();
        const auto begin = trace ? Trace::Clock::now() : Trace::Clock::time_point{};
        Trace::Sample sample{};
        auto raster_area = renderer->ReadRasterizer()->GetFlushArea(addr, size);
        const auto lookup_end = trace ? Trace::Clock::now() : Trace::Clock::time_point{};
        if (trace) {
            sample[Trace::Lookup] = Trace::Milliseconds(begin, lookup_end);
        }
        if (raster_area.preemtive) {
            if (trace) {
                sample[Trace::Total] = sample[Trace::Lookup];
                Trace::Report(sample, false);
            }
            return raster_area;
        }
        raster_area.preemtive = true;
        // A guest CPU read of GPU-written memory (reactive flushing): count them, they decide
        // whether buffer data must be copied back at every fence (EDEN_NX2_BUFFER_DOWNLOADS).
        if (static std::atomic<u64> reads{}; (reads.fetch_add(1, std::memory_order_relaxed) & 63) == 0) {
            LOG_INFO(HW_GPU, "CPU read of GPU-written memory [{:#x}, {:#x}) ({} so far)",
                     raster_area.start_address, raster_area.end_address,
                     reads.load(std::memory_order_relaxed));
        }
        Trace::Clock::time_point flush_end{};
        const u64 fence = RequestSyncOperation([this, raster_area, trace, lookup_end,
                                                &sample, &flush_end]() {
            Trace::ActiveSample active{trace ? &sample : nullptr};
            const auto start = trace ? Trace::Clock::now() : Trace::Clock::time_point{};
            // Apply collected guest writes before downloading GPU data over guest memory.
            InvalidateGPUCache();
            const auto invalidated = trace ? Trace::Clock::now() : Trace::Clock::time_point{};
            renderer->ReadRasterizer()->FlushRegion(
                raster_area.start_address, raster_area.end_address - raster_area.start_address);
            if (trace) {
                flush_end = Trace::Clock::now();
                sample[Trace::Queue] = Trace::Milliseconds(lookup_end, start);
                sample[Trace::Invalidate] = Trace::Milliseconds(start, invalidated);
                sample[Trace::Flush] = Trace::Milliseconds(invalidated, flush_end);
            }
        });
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(fence);
        if (trace) {
            const auto end = Trace::Clock::now();
            sample[Trace::Resume] = Trace::Milliseconds(flush_end, end);
            sample[Trace::Total] = Trace::Milliseconds(begin, end);
            Trace::Report(sample, true);
        }
        return raster_area;
    }

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
        if (preserve_gpu_writes) {
            VideoCore::RasterizerInterface* rasterizer = renderer->ReadRasterizer();
            if (!rasterizer->TryCacheInvalidation(addr, size)) {
                const auto fault = (VideoCommon::GpuWriteTrace::enabled ||
                                    VideoCommon::GpuWriteTrace::profile_footprints)
                    ? VideoCommon::GpuWriteTrace::fault : VideoCommon::GpuWriteTrace::Fault::None;
                const u64 fence = RequestSyncOperation([rasterizer, addr, size, fault] {
                    namespace Trace = VideoCommon::GpuWriteTrace;
                    if (fault != Trace::Fault::None) {
                        Trace::Cpu(fault == Trace::Fault::Store ? Trace::Site::NceStorePage
                                                              : Trace::Site::NceReadPage,
                                   addr, size, false);
                    }
                    rasterizer->FlushRegion(addr, size, VideoCommon::CacheType::BufferCache);
                    rasterizer->OnCacheInvalidation(addr, size);
                });
                gpu_thread.TickGPU(is_async);
                WaitForSyncOperation(fence);
            }
            return;
        }
        gpu_thread.InvalidateRegion(addr, size);
    }

    bool OnCPUWrite(DAddr addr, u64 size) {
        return renderer->ReadRasterizer()->OnCPUWrite(addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed and invalidated
    void FlushAndInvalidateRegion(DAddr addr, u64 size) {
        gpu_thread.FlushAndInvalidateRegion(addr, size, is_async);
    }

    void RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers, std::vector<Service::Nvidia::NvFence>&& fences) {
        const size_t num_fences{fences.size()};
        size_t current_request_counter{};
        if (num_fences != 0) {
            std::unique_lock<std::mutex> lk(request_swap_mutex);
            if (free_swap_counters.empty()) {
                current_request_counter = request_swap_counters.size();
                request_swap_counters.emplace_back(num_fences);
            } else {
                current_request_counter = free_swap_counters.front();
                request_swap_counters[current_request_counter] = num_fences;
                free_swap_counters.pop_front();
            }
        }
        pending_composite_fence = RequestSyncOperation(
            [this, current_request_counter, num_fences, composite_layers = std::move(layers),
             composite_fences = std::move(fences)] {
                if (num_fences == 0) {
                    renderer->Composite(composite_layers);
                    return;
                }
                auto& syncpoint_manager = system.Host1x().GetSyncpointManager();
                const auto executer = [this, current_request_counter, composite_layers]() {
                    {
                        std::unique_lock<std::mutex> lk(request_swap_mutex);
                        if (--request_swap_counters[current_request_counter] != 0) {
                            return;
                        }
                        free_swap_counters.push_back(current_request_counter);
                    }
                    renderer->Composite(composite_layers);
                };
                for (size_t i = 0; i < num_fences; i++) {
                    syncpoint_manager.RegisterGuestAction(composite_fences[i].id,
                                                          composite_fences[i].value, executer);
                }
            });
        gpu_thread.TickGPU(is_async);
    }

    void WaitForComposite() {
        const u64 fence = pending_composite_fence;
        if (fence == 0) {
            return;
        }
        pending_composite_fence = 0;
        if (shutting_down.load(std::memory_order_relaxed)) {
            return;
        }
        WaitForSyncOperation(fence);
    }

    std::vector<u8> GetAppletCaptureBuffer() {
        std::vector<u8> out;

        const auto wait_fence =
            RequestSyncOperation([&] { out = renderer->GetAppletCaptureBuffer(); });
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(wait_fence);

        return out;
    }

    void ApplyDeferredWriteCut(u64 cut) {
        using namespace VideoCommon::DeferredWrites;
        if (!deferred_writes || !VideoCommon::GpuSplit::OwnsScheduler()) {
            UNREACHABLE_MSG("Deferred write cut outside its scheduler owner");
        }
        const auto result = deferred_writes->DrainThrough(cut, [this](const Write& write) {
            return renderer && renderer->ReadRasterizer()->ApplyDeferredWrite(write);
        });
        if (result != DrainResult::Complete) {
            UNREACHABLE_MSG("Accepted deferred write lost its buffer or mapping identity");
        }
        const bool retired = deferred_writes->RetireThrough(deferred_writes->ReclaimableThrough());
        ASSERT(retired);
    }

    long cpu_epoch_host_page_bytes{4096};
    Core::System& system;
    std::shared_ptr<VideoCommon::DeferredWrites::Journal> deferred_writes;
    std::shared_ptr<VideoCommon::DeferredWrites::Journal::WriterPermission> deferred_kernel_pages;

    // Destruction of thread must be done before all (non trivial)
    // previous members has been destroyed
    VideoCommon::GPUThread::ThreadManager gpu_thread;

    std::unique_ptr<VideoCore::RendererBase> renderer;
    const bool use_nvdec;

    s32 new_channel_id{1};
    /// Shader build notifier
    VideoCore::ShaderNotify shader_notify;
    /// When true, we are about to shut down emulation session, so terminate outstanding tasks
    std::atomic_bool shutting_down{};

    std::array<std::atomic<u32>, Service::Nvidia::MaxSyncPoints> syncpoints{};

    std::array<std::list<u32>, Service::Nvidia::MaxSyncPoints> syncpt_interrupts;

    std::mutex sync_mutex;
    std::mutex device_mutex;

    std::condition_variable sync_cv;

    VideoCommon::SyncRequestQueue sync_requests;
    std::atomic<u64> current_sync_fence{};

    VideoCommon::PendingMaps pending_maps;
    std::mutex sync_request_mutex;
    std::condition_variable sync_request_cv;

    const bool is_async;

    std::unique_ptr<Core::Frontend::GraphicsContext> cpu_context;

    Tegra::Control::Scheduler scheduler;
    ::Common::unordered_map<s32, std::shared_ptr<Tegra::Control::ChannelState>> channels;
    Tegra::Control::ChannelState* current_channel;
    s32 bound_channel{-1};

    std::deque<size_t> free_swap_counters;
    std::deque<size_t> request_swap_counters;
    std::mutex request_swap_mutex;
    u64 pending_composite_fence{};
};

GPU::GPU(Core::System& system, bool is_async, bool use_nvdec)
    : impl{std::make_unique<Impl>(system, is_async, use_nvdec)}
{}

GPU::~GPU() = default;

std::shared_ptr<Control::ChannelState> GPU::AllocateChannel() {
    return impl->AllocateChannel();
}

void GPU::InitChannel(Control::ChannelState& to_init, u64 program_id) {
    impl->InitChannel(to_init, program_id);
}

void GPU::BindChannel(s32 channel_id) {
    impl->BindChannel(channel_id);
}

void GPU::ReleaseChannel(Control::ChannelState& to_release) {
    impl->ReleaseChannel(to_release);
}

void GPU::InitAddressSpace(Tegra::MemoryManager& memory_manager) {
    impl->InitAddressSpace(memory_manager);
}

void GPU::BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer) {
    impl->BindRenderer(std::move(renderer));
}

void GPU::FlushCommands() {
    impl->FlushCommands();
}

void GPU::InvalidateGPUCache() {
    impl->InvalidateGPUCache();
}

bool GPU::HasPendingCPUWrites(DAddr addr, u64 size) {
    for (auto& manager : impl->system.GetGPUDirtyMemoryManager()) {
        if (manager.Overlaps(addr, size)) {
            return true;
        }
    }
    return false;
}

void GPU::ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
    impl->ForEachPendingCpuWrite(callback);
}

void GPU::OnCommandListEnd() {
    impl->OnCommandListEnd();
}

u64 GPU::RequestFlush(DAddr addr, std::size_t size) {
    return impl->RequestSyncOperation([this, addr, size]() {
        impl->renderer->ReadRasterizer()->FlushRegion(addr, size);
    });
}

u64 GPU::RunOnGpuThread(std::function<void()>&& action) {
    return impl->RequestSyncOperation(std::move(action));
}

u64 GPU::CurrentSyncRequestFence() const {
    return impl->CurrentSyncRequestFence();
}

void GPU::WaitForSyncOperation(u64 fence) {
    return impl->WaitForSyncOperation(fence);
}

std::shared_ptr<VideoCommon::DeferredWrites::Journal::WriterPermission>
GPU::BeginDeferredQueryWrite(DAddr address, u64 size) {
    if (!DeferredWriteJournal()) return {};
    std::vector<VideoCommon::DeferredWrites::TransitionRange> ranges;
    VideoCommon::DeferredWrites::AddConsumerDeviceRange(
        ranges, impl->system.Host1x().MemoryManager(), address, size);
    return BeginDeferredWriterPermission(ranges);
}

u64 GPU::CaptureDeferredWriteCut() const {
    return impl->deferred_writes ? impl->deferred_writes->CaptureCut() : 0;
}

VideoCommon::DeferredWrites::Journal* GPU::DeferredWriteJournal() {
    return impl->deferred_writes ? &*impl->deferred_writes : nullptr;
}

void GPU::OrderDeferredWriteCut(u64 cut, Common::GpuTransitionProfile::DeferredDrainSite site) {
    if (cut == 0) {
        return;
    }
    Common::GpuTransitionProfile::RecordDeferredDrain(site);
    auto& split = VideoCommon::GpuSplit::Get();
    if (split.Active() && !VideoCommon::GpuSplit::OwnsScheduler()) {
        split.Defer([state = impl.get(), cut] { state->ApplyDeferredWriteCut(cut); });
    } else {
        impl->ApplyDeferredWriteCut(cut);
    }
}

void GPU::OnDeferredWriteAcquire() {
    OrderDeferredWriteCut(CaptureDeferredWriteCut(),
                         Common::GpuTransitionProfile::DeferredDrainSite::Acquire);
}

bool GPU::DrainDeferredWriteCutWithoutLocks(u64 cut) {
    auto* journal = DeferredWriteJournal();
    if (!journal) return cut == 0;
    if (cut > journal->CaptureCut()) return false;
    if (journal->AppliedThrough() >= cut) return true;
    Common::GpuTransitionProfile::RecordDeferredDrain(
        Common::GpuTransitionProfile::DeferredDrainSite::Transition);
    if (VideoCommon::GpuSplit::OwnsScheduler()) {
        impl->ApplyDeferredWriteCut(cut);
    } else if (VideoCommon::GpuSplit::OnGpuThread()) {
        auto& split = VideoCommon::GpuSplit::Get();
        ASSERT(split.Active());
        split.Defer([state = impl.get(), cut] { state->ApplyDeferredWriteCut(cut); });
        split.Drain(VideoCommon::DrainSite::Map);
    } else {
        if (!VideoCommon::GpuSplit::HasGpuThread() ||
            impl->shutting_down.load(std::memory_order_acquire)) return false;
        const u64 fence = impl->RequestSyncOperation(
            [state = impl.get(), cut] { state->ApplyDeferredWriteCut(cut); });
        impl->gpu_thread.TickGPU(impl->is_async);
        impl->WaitForSyncOperation(fence);
    }
    return journal->AppliedThrough() >= cut;
}

std::unique_ptr<VideoCommon::DeferredWrites::Suspension> GPU::SuspendDeferredWrites() {
    auto* journal = DeferredWriteJournal();
    if (!journal) return {};
    return std::make_unique<VideoCommon::DeferredWrites::Suspension>(journal->BeginSuspension());
}

std::shared_ptr<VideoCommon::DeferredWrites::Journal::WriterPermission>
GPU::BeginDeferredWriterPermission(
    std::span<const VideoCommon::DeferredWrites::TransitionRange> ranges) {
    return PrepareDeferredWriterPermission(ReserveDeferredWriterPermission(ranges));
}

std::shared_ptr<VideoCommon::DeferredWrites::Journal::WriterPermission>
GPU::ReserveDeferredWriterPermission(
    std::span<const VideoCommon::DeferredWrites::TransitionRange> ranges,
    VideoCommon::DeferredWrites::WriterOrigin origin,
    Common::GpuTransitionProfile::DeferredFallbackReason first_reason) {
    using namespace VideoCommon::DeferredWrites;
    auto& epochs = impl->system.DeviceMemory().CpuEpochs();
    if (epochs.Requested()) {
        for (const auto& range : ranges) {
            if (range.size == 0) {
                continue;
            }
            if (range.space == TransitionSpace::Physical) {
                epochs.Exclude(range.address, range.size, VideoCommon::CpuEpochs::Exclusion::CapturedWriter);
                continue;
            }
            impl->system.Host1x().MemoryManager().ExcludeCpuEpochRange(
                range.address, range.size, VideoCommon::CpuEpochs::Exclusion::CapturedWriter);
        }
    }
    auto journal = impl->deferred_writes;
    if (!journal) return {};
    auto permission = BeginSharedWriterPermission(journal, ranges, origin, first_reason);
    if (!permission) UNREACHABLE_MSG("Invalid legacy writer range");
    return permission;
}

std::shared_ptr<VideoCommon::DeferredWrites::Journal::WriterPermission>
GPU::PrepareDeferredWriterPermission(
    std::shared_ptr<VideoCommon::DeferredWrites::Journal::WriterPermission> permission) {
    using namespace VideoCommon::DeferredWrites;
    if (!permission) return {};
    auto journal = impl->deferred_writes;
    ASSERT(journal);

    // Owner-side query callbacks participate in completing the existing GPU work. A host
    // fallback must instead join a suspension already in progress before entering old code.
    std::shared_ptr<Suspension> suspension;
    if (!VideoCommon::GpuSplit::OwnsScheduler() &&
        journal->GetSuspensionState() != SuspensionState::Running) {
        suspension = BeginDeferredTransition(Common::GpuTransitionProfile::SuspensionSite::LegacyWriter);
    }
    if (!permission->MarkReady()) {
        if (!DrainDeferredWriteCutWithoutLocks(permission->Watermark()) || !permission->MarkReady()) {
            UNREACHABLE_MSG("Legacy writer cannot commit before its deferred prefix");
        }
    }
    if (!suspension) return permission;
    // Keep the joined Ready epoch alive through the actual legacy commit, not just the drain.
    auto* pointer = permission.get();
    return std::shared_ptr<Journal::WriterPermission>(pointer,
        [permission = std::move(permission), suspension = std::move(suspension)](auto*) {});
}

std::shared_ptr<VideoCommon::DeferredWrites::Suspension> GPU::BeginDeferredTransition(
    Common::GpuTransitionProfile::SuspensionSite site, bool can_wait) {
    using namespace VideoCommon::DeferredWrites;
    namespace Profile = Common::GpuTransitionProfile;
    auto journal = impl->deferred_writes;
    if (!journal) return {};
    const u64 begin = Profile::SuspensionNowNs();
    auto token = BeginSharedSuspension(journal, static_cast<u64>(site), begin != 0);
    return PrepareDeferredTransition(std::move(token), site, can_wait);
}

bool GPU::RecoverAbandonedDeferredWritesWithoutLocks() {
    namespace Profile = Common::GpuTransitionProfile;
    auto token = VideoCommon::DeferredWrites::AdoptSharedAbandonedSuspension(impl->deferred_writes);
    if (!token) return false;
    auto ready = PrepareDeferredTransition(std::move(token), Profile::SuspensionSite::ProducerRecovery,
                                           true);
    Profile::RecordDeferredRecovery(ready->IsReady());
    return true;
}

std::shared_ptr<VideoCommon::DeferredWrites::Suspension> GPU::PrepareDeferredTransition(
    std::shared_ptr<VideoCommon::DeferredWrites::Suspension> token,
    Common::GpuTransitionProfile::SuspensionSite site, bool can_wait) {
    using namespace VideoCommon::DeferredWrites;
    namespace Profile = Common::GpuTransitionProfile;
    auto journal = impl->deferred_writes;
    ASSERT(journal && token && journal->OwnsSuspension(*token));
    const u64 begin = Profile::SuspensionNowNs();
    Profile::RecordSuspensionParticipation(site, token->CreatedEpoch());
    bool prepared{};
    auto result = token->TryPrepareIdle(prepared);
    const auto record_epoch_ready = [&] {
        if (prepared && result == SuspensionResult::Ready) {
            Profile::RecordSuspensionEpochReady(static_cast<Profile::SuspensionSite>(token->Origin()),
                                                token->StartedNs());
        }
    };
    record_epoch_ready();
    if (result != SuspensionResult::Ready) {
        ASSERT_MSG(can_wait, "Deferred transition requires an outer suspension before kernel locks");
        Profile::RecordSuspension(site, Profile::SuspensionEvent::Pending);
        prepared = false;
        // Claim preparation on the scheduler owner, not on a host thread waiting for that
        // owner. Complete the readbacks and the Ready transition in the same callback.
        const auto prepare = [&] {
            result = token->TryPrepare([&](Cut cut) {
                prepared = true;
                if (journal->AppliedThrough() < cut) {
                    Profile::RecordDeferredDrain(Profile::DeferredDrainSite::Transition);
                    impl->ApplyDeferredWriteCut(cut);
                }
                if (journal->HasReadbacks()) {
                    return impl->renderer->ReadRasterizer()->ResolveDeferredWriteReadbacks();
                }
                return true;
            });
            record_epoch_ready();
        };
        if (VideoCommon::GpuSplit::OwnsScheduler()) {
            prepare();
        } else if (VideoCommon::GpuSplit::OnGpuThread()) {
            auto& split = VideoCommon::GpuSplit::Get();
            ASSERT(split.Active());
            split.Defer(prepare);
            split.Drain(VideoCommon::DrainSite::Map);
        } else {
            if (!VideoCommon::GpuSplit::HasGpuThread() ||
                impl->shutting_down.load(std::memory_order_acquire)) {
                Profile::RecordSuspension(site, Profile::SuspensionEvent::Failure);
                UNREACHABLE_MSG("Deferred transition needs a live scheduler owner");
            }
            const u64 fence = impl->RequestSyncOperation(prepare);
            impl->gpu_thread.TickGPU(impl->is_async);
            impl->WaitForSyncOperation(fence);
        }
    }
    if (result != SuspensionResult::Ready) {
        Profile::RecordSuspension(site, Profile::SuspensionEvent::Failure);
        UNREACHABLE_MSG("Deferred transition cannot mutate before Ready: {}", static_cast<int>(result));
    }
    Profile::RecordSuspensionReady(site, prepared, begin);
    return token;
}

VideoCommon::DeferredWrites::SuspensionResult GPU::PrepareDeferredWriteSuspensionWithoutLocks(
    VideoCommon::DeferredWrites::Suspension& token,
    const std::function<bool()>& resolve_readbacks) {
    using VideoCommon::DeferredWrites::SuspensionResult;
    auto* journal = DeferredWriteJournal();
    if (!journal || !journal->OwnsSuspension(token) || !resolve_readbacks) {
        return SuspensionResult::InvalidToken;
    }
    return token.TryPrepare([&](u64 cut) {
        return DrainDeferredWriteCutWithoutLocks(cut) && resolve_readbacks();
    });
}

#ifdef HAS_NCE
Core::NceExactStore::Result GPU::PublishNceStore(
    size_t core, DAddr addr, std::span<const u8> bytes, const Core::NceExactStore::Backing& backing,
    u64& publication_ns, bool exact_only) {
    using Core::NceExactStore::Result;
    using namespace VideoCommon::DeferredWrites;
    namespace Profile = Common::GpuTransitionProfile;
    using CpuStatus = Core::DeferredCpuMappings::Status;
    publication_ns = 0;
    auto* journal = DeferredWriteJournal();
    if (!journal) return Result::Disabled;
    if (journal->GetSuspensionState() != SuspensionState::Running) {
        // Adoption and preparation are host operations, before any mapping/cache/gate lock.
        const bool recovered = RecoverAbandonedDeferredWritesWithoutLocks();
        (void)recovered;
    }
    if (const auto valid = Core::NceExactStore::ValidateBackingStore(addr, bytes, backing);
        valid != Result::Success) return valid;
    if (!backing.page_table || !backing.pointer) return Result::MappingChanged;
    auto& device = impl->system.Host1x().MemoryManager();
    if (!device.IsPhysicalPointer(backing.pointer)) return Result::MappingChanged;
    const PAddr physical = device.GetRawPhysicalAddr(backing.pointer);
    const PAddr physical_page = physical & ~PAddr{4095};
    const DAddr device_page = addr & ~DAddr{4095};
    const auto cpu_result = [](CpuStatus status) {
        switch (status) {
        case CpuStatus::Eligible: return Result::Success;
        case CpuStatus::Alias: return Result::IneligibleAlias;
        case CpuStatus::Shared: return Result::IneligibleShared;
        case CpuStatus::Transfer: return Result::IneligibleTransfer;
        case CpuStatus::Code: return Result::IneligibleCode;
        case CpuStatus::CapturedWriter: return Result::IneligibleCaptured;
        case CpuStatus::NotPrivateHeap: return Result::IneligibleCpuState;
        case CpuStatus::Disabled: return Result::Disabled;
        default: return Result::MappingChanged;
        }
    };
    const auto cached_pointer_matches = [&] {
        const u64 page = backing.address >> 12;
        if (page >= backing.page_table->entries.size()) return false;
        const auto [pointer, type, block] = backing.page_table->entries[page].PointerTypeBlock(true);
        return type == Common::PageType::RasterizerCachedMemory && pointer &&
               reinterpret_cast<u8*>(pointer + backing.address) == backing.pointer;
    };

    // A temporary global transition must not permanently exclude an otherwise usable page.
    // Join only until Ready, not Running: the transaction owner may need this core to progress.
    const auto suspended_exact = [&]() -> Result {
        if (!VideoCommon::GpuSplit::HasGpuThread() ||
            impl->shutting_down.load(std::memory_order_acquire)) return Result::WrongThread;
        std::shared_ptr<Suspension> ready;
        {
            Profile::DeferredProducerScope timing{Profile::DeferredProducerPhase::SuspensionReady};
            ready = BeginDeferredTransition(Profile::SuspensionSite::LegacyWriter);
        }
        ASSERT(ready && ready->IsReady());
        Profile::DeferredProducerScope timing{Profile::DeferredProducerPhase::SuspendedExact};
        Result exact_result = Result::BackendUnsupported;
        const u64 fence = impl->RequestSyncOperation([&] {
            const auto status = impl->system.DeviceMemory().CpuMappings().WithPage(
                backing.page_table, backing.address, physical_page, [&](const auto&) {
                    if (!cached_pointer_matches()) { exact_result = Result::MappingChanged; return; }
                    // CPU identity stays stable through the exact legacy commit. The rasterizer
                    // revalidates device mapping, CPU cleanliness, queries and texture conflicts.
                    auto exact_backing = backing;
                    exact_backing.allow_clean_page = true;
                    exact_result = impl->renderer->ReadRasterizer()->WriteNceStore(addr, bytes, exact_backing);
                });
            if (status != CpuStatus::Eligible) exact_result = cpu_result(status);
        });
        impl->gpu_thread.TickGPU(impl->is_async);
        impl->WaitForSyncOperation(fence);
        if (Profile::enabled.load(std::memory_order_relaxed))
            Profile::deferred_counters.RecordSuspendedExact(exact_result == Result::Success);
        return exact_result;
    };
    if (exact_only || journal->GetSuspensionState() != SuspensionState::Running) return suspended_exact();

    Result result = Result::LeaseUnavailable;
    bool need_grant{};
    const auto publish = [&](bool check_persistent_exclusion) {
        const u64 attempt_begin = Core::NceExactStore::NowNs();
        Profile::DeferredProducerScope timing{Profile::DeferredProducerPhase::PublishAttempt};
        const auto record_admission = [](Profile::DeferredProducerPhase phase, u64 begin) {
            if (begin) Profile::deferred_counters.RecordPhase(phase, Profile::SuspensionNowNs() - begin);
        };
        const u64 cpu_begin = Profile::SuspensionNowNs();
        bool cpu_entered{};
        need_grant = false;
        const auto status = impl->system.DeviceMemory().CpuMappings().WithPage(
            backing.page_table, backing.address, physical_page, [&](const auto& snapshot) {
                cpu_entered = true;
                record_admission(Profile::DeferredProducerPhase::PublishCpuAdmission, cpu_begin);
                if (!cached_pointer_matches()) { result = Result::MappingChanged; return; }
                const u64 device_begin = Profile::SuspensionNowNs();
                bool device_entered{};
                bool mapped = device.ApplyOpOnSingleMapping(backing.pointer, bytes.size(), [&](DAddr current) {
                    device_entered = true;
                    record_admission(Profile::DeferredProducerPhase::PublishDeviceAdmission, device_begin);
                    if (current != addr) { result = Result::MappingChanged; return; }
                    std::optional<Journal::Lease> lease;
                    LeaseLookupStatus lookup_status{};
                    const bool diagnostics = Profile::enabled.load(std::memory_order_relaxed);
                    {
                        Profile::DeferredProducerScope lookup{Profile::DeferredProducerPhase::PublishLeaseLookup};
                        lease = journal->FindLease(physical_page, device_page, snapshot.generation,
                                                   diagnostics ? &lookup_status : nullptr);
                    }
                    if (!lease) {
                        Profile::RecordLeaseDiagnostic(
                            check_persistent_exclusion ? "lookup_initial" : "lookup_after_grant",
                            lease_lookup_status_names[static_cast<size_t>(lookup_status)], physical_page);
                        if (check_persistent_exclusion &&
                            journal->HasPersistentWriterExclusion(physical_page, device_page)) {
                            result = Result::LeaseWriterExcluded;
                            if (Profile::enabled.load(std::memory_order_relaxed)) {
                                Profile::deferred_counters.RecordPhase(
                                    Profile::DeferredProducerPhase::LeaseGrantAvoided, 0);
                            }
                            return;
                        }
                        result = Result::LeaseUnavailable;
                        need_grant = true;
                        return;
                    }
                    Write write;
                    write.physical_address = physical;
                    write.device_address = addr;
                    write.mapping_generation = snapshot.generation;
                    write.lease_generation = lease->generation;
                    write.size = static_cast<u32>(bytes.size());
                    write.payload = backing.zero_block ? Payload::Zero : Payload::Bytes;
                    if (!backing.zero_block) std::copy(bytes.begin(), bytes.end(), write.bytes.begin());
                    LeaseLookupStatus commit_status{};
                    const auto publication = [&] {
                        Profile::DeferredProducerScope commit{Profile::DeferredProducerPhase::PublishCommit};
                        return journal->TryPublishLeased(core, write, [&]() noexcept {
                            return Core::NceExactStore::CommitBackingStore(addr, bytes, backing);
                        }, diagnostics ? &commit_status : nullptr);
                    }();
                    if (publication.status == PublishStatus::LeaseUnavailable) {
                        Profile::RecordLeaseDiagnostic(
                            check_persistent_exclusion ? "commit_initial" : "commit_after_grant",
                            lease_lookup_status_names[static_cast<size_t>(commit_status)], physical_page);
                    }
                    switch (publication.status) {
                    case PublishStatus::Published: result = Result::Success; break;
                    case PublishStatus::Full: result = Result::QueueFull; break;
                    case PublishStatus::Suspended: result = Result::Suspended; break;
                    case PublishStatus::LeaseUnavailable: result = Result::LeaseUnavailable; break;
                    case PublishStatus::Rejected: result = Result::CommitRejected; break;
                    default: result = Result::InvalidOperands; break;
                    }
                });
                if (!device_entered)
                    record_admission(Profile::DeferredProducerPhase::PublishDeviceAdmission, device_begin);
                if (!mapped) result = Result::AliasedMapping;
            });
        if (!cpu_entered) record_admission(Profile::DeferredProducerPhase::PublishCpuAdmission, cpu_begin);
        if (status != CpuStatus::Eligible) result = cpu_result(status);
        publication_ns += Core::NceExactStore::NowNs() - attempt_begin;
    };
    publish(true);
    if (result == Result::Suspended) return suspended_exact();
    if (!need_grant || result != Result::LeaseUnavailable) return result;
    if (!VideoCommon::GpuSplit::HasGpuThread() ||
        impl->shutting_down.load(std::memory_order_acquire)) return Result::WrongThread;
    if (journal->GetSuspensionState() != SuspensionState::Running) return suspended_exact();

    // Only first admission needs the owner. No lock held while waiting for this callback.
    {
        Profile::WatchdogCoreScope watchdog{static_cast<u32>(core), Profile::WatchdogPhase::GrantQueued};
        Profile::DeferredProducerScope timing{Profile::DeferredProducerPhase::LeaseGrant};
        const u64 queued = Profile::SuspensionNowNs();
        u64 callback_begin{}, callback_end{};
        const u64 fence = impl->RequestSyncOperation([&] {
            callback_begin = Profile::SuspensionNowNs();
            const auto status = impl->system.DeviceMemory().CpuMappings().WithPage(
                backing.page_table, backing.address, physical_page, [&](const auto& snapshot) {
                    if (!cached_pointer_matches()) { result = Result::MappingChanged; return; }
                    if (journal->FindLease(physical_page, device_page, snapshot.generation)) {
                        result = Result::Success;
                        return;
                    }
                    Journal::Lease lease{};
                    result = impl->renderer->ReadRasterizer()->GrantDeferredWriteLease(
                        device_page, physical_page, snapshot.generation, lease);
                });
            if (status != CpuStatus::Eligible) result = cpu_result(status);
            callback_end = Profile::SuspensionNowNs();
        });
        impl->gpu_thread.TickGPU(impl->is_async);
        impl->WaitForSyncOperation(fence);
        if (queued) {
            Profile::deferred_counters.RecordPhase(Profile::DeferredProducerPhase::LeaseGrantQueue,
                                                   callback_begin - queued);
            Profile::deferred_counters.RecordPhase(Profile::DeferredProducerPhase::LeaseGrantCallback,
                                                   callback_end - callback_begin);
            Profile::deferred_counters.RecordPhase(Profile::DeferredProducerPhase::LeaseGrantReturn,
                                                   Profile::SuspensionNowNs() - callback_end);
        }
    }
    if (result != Result::Success) {
        Profile::RecordLeaseDiagnostic("grant_result",
            Profile::deferred_fallback_names[static_cast<size_t>(result)], physical_page);
    }
    if (result == Result::Suspended) return suspended_exact();
    if (result != Result::Success) return result;
    // Mapping and admission may have changed during the round trip; never commit on that hint.
    publish(false);
    if (result == Result::Suspended) return suspended_exact();
    return result;
}

Core::NceExactStore::Result GPU::WriteNceStore(DAddr addr, std::span<const u8> bytes,
                                            const Core::NceExactStore::Backing& backing) {
    using Core::NceExactStore::Result;
    const size_t max_size = backing.zero_block ? Core::NceExactStore::DcZvaZeros.size() : 32;
    if (bytes.empty() || bytes.size() > max_size) {
        return Result::InvalidOperands;
    }
    if (backing.zero_block && !std::all_of(bytes.begin(), bytes.end(), [](u8 b) { return b == 0; })) {
        return Result::InvalidOperands;
    }
    if (!VideoCommon::GpuSplit::HasGpuThread() ||
        impl->shutting_down.load(std::memory_order_acquire)) {
        return Result::WrongThread;
    }
    const u64 begin = Core::NceExactStore::NowNs();
    u64 callback_begin{}, callback_end{};
    std::array<u8, 32> data{};
    if (!backing.zero_block) {
        std::copy(bytes.begin(), bytes.end(), data.begin());
    }
    Result result = Result::BackendUnsupported;
    const u64 fence = impl->RequestSyncOperation(
        [this, addr, data, size = bytes.size(), backing, &result, &callback_begin, &callback_end] {
            callback_begin = Core::NceExactStore::NowNs();
            VideoCommon::GpuWriteTrace::Cpu(VideoCommon::GpuWriteTrace::Site::NceExact, addr, size);
            const std::span<const u8> payload = backing.zero_block ?
                std::span{Core::NceExactStore::DcZvaZeros}.first(size) :
                std::span{data}.first(size);
            result = impl->renderer->ReadRasterizer()->WriteNceStore(addr, payload, backing);
            callback_end = Core::NceExactStore::NowNs();
            return VideoCommon::SyncRequestQueue::Status::Complete;
        });
    impl->gpu_thread.TickGPU(impl->is_async);
    // Completion of the host callback, not a Vulkan fence or a Scheduler::Finish.
    impl->WaitForSyncOperation(fence);
    const u64 end = Core::NceExactStore::NowNs();
    Core::NceExactStore::timing_gpu_queue.Add(callback_begin - begin);
    Core::NceExactStore::timing_gpu_callback.Add(callback_end - callback_begin);
    Core::NceExactStore::timing_gpu_wake.Add(end - callback_end);
    Core::NceExactStore::timing_gpu_round_trip.Add(end - begin);
    return result;
}
#endif

VideoCommon::CpuEpochs::Coordinator::Mutation GPU::BeforeCpuEpochConsumption() {
    return impl->system.DeviceMemory().CpuEpochs().BeforeMutation(
        VideoCommon::CpuEpochs::CloseReason::Consumer);
}
bool GPU::TryOpenCpuEpoch(VideoCommon::CpuEpochs::Identity identity) {
    using namespace VideoCommon::CpuEpochs;
    auto& epochs = impl->system.DeviceMemory().CpuEpochs();
    if (!epochs.Available() || !impl->deferred_writes || impl->cpu_epoch_host_page_bytes != 4096 ||
        identity.device_owner != &impl->system.Host1x().MemoryManager() ||
        impl->shutting_down.load(std::memory_order_acquire) ||
        !Core::DeferredWritesConfig::ProducerRequested() ||
        !VideoCommon::GpuSplit::Get().BackRunning()) {
        epochs.NoteRejected(Rejection::Runtime);
        return false;
    }
    if (!VideoCommon::NcePageUsageProfile::IsCpuEpochStorageCandidate(identity.physical)) {
        epochs.NoteRejected(Rejection::Candidate);
        return false;
    }
    bool opened = false;
    const auto fence = impl->RequestSyncOperation([this, identity = std::move(identity), &opened]() mutable {
        auto& owner_epochs = impl->system.DeviceMemory().CpuEpochs();
        if (!VideoCommon::GpuSplit::OwnsScheduler() || owner_epochs.ActivePages()) {
            owner_epochs.NoteRejected();
            return VideoCommon::SyncRequestQueue::Status::Complete;
        }
        u64 downloaded = 0;
        opened = owner_epochs.PrepareAndOpen(
            identity,
            [&] {
                return (!impl->deferred_writes->HasReadbacks() ||
                        impl->renderer->ReadRasterizer()->ResolveDeferredWriteReadbacks()) &&
                       impl->renderer->ReadRasterizer()->PrepareCpuEpochPage(
                           identity.device, identity.physical, downloaded);
            },
            [&] {
                return identity.memory->GrantCpuEpoch(identity.cpu_owner, identity.address,
                                                      identity.physical, identity.generation);
            },
            downloaded,
            [&](Identity& pinned) {
                using namespace VideoCommon::DeferredWrites;
                const std::array ranges{
                    TransitionRange{TransitionSpace::Physical, pinned.physical, Coordinator::PageSize},
                    TransitionRange{TransitionSpace::Device, pinned.device, Coordinator::PageSize}};
                pinned.writer = BeginSharedWriterPermission(impl->deferred_writes, ranges);
                if (!pinned.writer)
                    return false;
                impl->ApplyDeferredWriteCut(pinned.writer->Watermark());
                return pinned.writer->MarkReady();
            });
        return VideoCommon::SyncRequestQueue::Status::Complete;
    });
    impl->gpu_thread.TickGPU(impl->is_async);
    impl->WaitForSyncOperation(fence);
    return opened;
}
void GPU::TickWork() {
    impl->TickWork();
}

/// Gets a mutable reference to the Host1x interface
Host1x::Host1x& GPU::Host1x() {
    return impl->system.Host1x();
}

/// Gets an immutable reference to the Host1x interface.
const Host1x::Host1x& GPU::Host1x() const {
    return impl->system.Host1x();
}

Engines::Maxwell3D& GPU::Maxwell3D() {
    return impl->current_channel->payload->maxwell_3d;
}

const Engines::Maxwell3D& GPU::Maxwell3D() const {
    return impl->current_channel->payload->maxwell_3d;
}

Engines::KeplerCompute& GPU::KeplerCompute() {
    return impl->current_channel->payload->kepler_compute;
}

const Engines::KeplerCompute& GPU::KeplerCompute() const {
    return impl->current_channel->payload->kepler_compute;
}

Tegra::DmaPusher& GPU::DmaPusher() {
    return impl->current_channel->payload->dma_pusher;
}

const Tegra::DmaPusher& GPU::DmaPusher() const {
    return impl->current_channel->payload->dma_pusher;
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

const VideoCore::RendererBase& GPU::Renderer() const {
    return *impl->renderer;
}

VideoCore::ShaderNotify& GPU::ShaderNotify() {
    return impl->shader_notify;
}

const VideoCore::ShaderNotify& GPU::ShaderNotify() const {
    return impl->shader_notify;
}

void GPU::RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers,
                           std::vector<Service::Nvidia::NvFence>&& fences) {
    impl->RequestComposite(std::move(layers), std::move(fences));
}

void GPU::WaitForComposite() {
    impl->WaitForComposite();
}

std::vector<u8> GPU::GetAppletCaptureBuffer() {
    return impl->GetAppletCaptureBuffer();
}

u64 GPU::GetTicks() const {
    return impl->GetTicks();
}

bool GPU::IsAsync() const {
    return impl->is_async;
}

bool GPU::HasSuspendedChannels() const {
    return impl->scheduler.HasPendingChannelsAnyThread();
}

bool GPU::UseNvdec() const {
    return impl->use_nvdec;
}

void GPU::RendererFrameEndNotify() {
    impl->RendererFrameEndNotify();
}

void GPU::Start() {
    impl->Start();
}

void GPU::NotifyShutdown() {
    impl->NotifyShutdown();
}

void GPU::WaitForIdle() {
    impl->WaitForIdle();
}

void GPU::ObtainContext() {
    impl->ObtainContext();
}

void GPU::ReleaseContext() {
    impl->ReleaseContext();
}

void GPU::PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
    impl->PushGPUEntries(channel, std::move(entries));
}

VideoCore::RasterizerDownloadArea GPU::OnCPURead(PAddr addr, u64 size) {
    return impl->OnCPURead(addr, size);
}

void GPU::FlushRegion(DAddr addr, u64 size) {
    impl->FlushRegion(addr, size);
}

void GPU::RunInOrder(std::function<void()>&& func,
                     std::shared_ptr<VideoCommon::DeferredWrites::Suspension> transition) {
    impl->RunInOrder(Core::RetainDeferredTransition(
        impl->system, std::move(transition), std::move(func)));
}

void GPU::MapInOrder(GPUVAddr gpu_addr, u64 size, std::function<void()>&& func,
                     std::shared_ptr<VideoCommon::DeferredWrites::Suspension> transition) {
    impl->MapInOrder(gpu_addr, size, Core::RetainDeferredTransition(
        impl->system, std::move(transition), std::move(func)));
}

bool GPU::IsMapPending(GPUVAddr gpu_addr, u64 size) const {
    return impl->IsMapPending(gpu_addr, size);
}

void GPU::InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
    impl->InvalidateRegion(addr, size, preserve_gpu_writes);
}

bool GPU::OnCPUWrite(DAddr addr, u64 size) {
    return impl->OnCPUWrite(addr, size);
}

void GPU::FlushAndInvalidateRegion(DAddr addr, u64 size) {
    impl->FlushAndInvalidateRegion(addr, size);
}

} // namespace Tegra
