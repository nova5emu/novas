// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/service/nvdrv/core/container.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/control/channel_state.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/semaphore_acquire.h"

namespace Tegra::Engines {
struct PullerTestAccess {
    static void StartAcquire(DmaPusher& pusher, u32 execute_operation, u32 value) {
        pusher.puller.StartSemaphoreAcquire(pusher, Puller::PendingAcquire{
            .address = 0x100000,
            .value = value,
            .operation = *SemaphoreAcquire::NormalizeExecute(execute_operation),
            .since = std::chrono::steady_clock::now(),
            .reported = false,
        });
    }
};
} // namespace Tegra::Engines

namespace {

struct Runtime;

struct Buffer : VideoCommon::BufferBase {
    Buffer(Runtime&, VideoCommon::NullBufferParams params) : BufferBase{params} {}
    Buffer(Runtime&, DAddr address, u32 size, bool mapped)
        : BufferBase{address, size}, bytes(size), host_mapped{mapped} {}

    void MarkUsage(u64 offset, u64 size) {
        REQUIRE(offset + size <= bytes.size());
    }
    std::span<u8> Mapped() {
        return host_mapped ? std::span{bytes} : std::span<u8>{};
    }
    u64 LastUsageTick() const {
        return tick;
    }

    std::vector<u8> bytes;
    bool host_mapped{};
    u64 tick{};
};

struct Staging {
    Buffer* buffer;
    u64 offset;
    std::span<u8> mapped_span;
};

struct Runtime {
    bool CanReportMemoryUsage() const {
        return false;
    }
    u64 GetDeviceLocalMemory() const {
        return 0;
    }
    u32 GetStorageBufferAlignment() const {
        return alignment;
    }
    u32 GetUniformBufferAlignment() const {
        return 1;
    }
    bool IsFree(u64) const {
        return list_copy_ready;
    }
    void WaitSubmittedTick(u64) {
        if (before_list_wait) before_list_wait();
        Finish();
        list_copy_ready = true;
    }
    bool CanReorderUpload(const Buffer&, std::span<VideoCommon::BufferCopy>) const {
        return false;
    }
    Staging DownloadStagingBuffer(u64 size, bool = false) {
        if (staging) retired_staging.push_back(std::move(staging));
        staging = std::make_unique<Buffer>(*this, 0, size + 128, false);
        ++allocations;
        return {staging.get(), 128, std::span{staging->bytes}.subspan(128)};
    }
    Staging UploadStagingBuffer(u64 size) {
        return DownloadStagingBuffer(size);
    }
    void CopyBuffer(Buffer* dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src.bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst->bytes.size());
            pending.push_back([dst, &src, copy] {
                std::memcpy(dst->bytes.data() + copy.dst_offset, src.bytes.data() + copy.src_offset,
                            copy.size);
            });
            ++downloads;
        }
    }
    void CopyBuffer(Buffer& dst, Buffer* src, std::span<const VideoCommon::BufferCopy> copies, bool,
                    bool) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src->bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst.bytes.size());
            std::memcpy(dst.bytes.data() + copy.dst_offset, src->bytes.data() + copy.src_offset,
                        copy.size);
            ++uploads;
            upload_bytes += copy.size;
        }
    }
    void CopyBuffer(Buffer& dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool) {
        if (before_gpu_copy) before_gpu_copy(dst, src);
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src.bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst.bytes.size());
            std::memcpy(dst.bytes.data() + copy.dst_offset, src.bytes.data() + copy.src_offset,
                        copy.size);
        }
    }
    void ClearBuffer(Buffer& buffer, u32 offset, size_t size, u32 value) {
        if (before_clear) before_clear(buffer);
        REQUIRE(offset + size <= buffer.bytes.size());
        std::memset(buffer.bytes.data() + offset, static_cast<int>(value), size);
    }
    void OnBufferDeleted(Buffer& buffer) {
        if (before_delete) before_delete(buffer);
    }
    void TickFrame(Common::SlotVector<Buffer>&) {}
    u64 GetDeviceMemoryUsage() const {
        return 0;
    }
    void FreeDeferredStagingBuffer(Staging&) {}
    void PreCopyBarrier() {}
    void PostCopyBarrier() {}
    u64 CurrentTick() const { return 1; }
    void Finish() {
        if (before_completion) {
            before_completion();
        }
        for (auto& copy : pending) {
            copy();
        }
        pending.clear();
        ++finishes;
    }
    void Wait(u64 tick) {
        if (before_completion) {
            before_completion();
        }
        waited_tick = tick;
        ++waits;
    }
    void FinishForHostRead() {
        ++host_read_barriers;
        Finish();
    }
    void BindUniformBuffer(Buffer&, u32, u32) {}
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        stream.resize(size);
        return stream;
    }

    u32 alignment{256};
    u32 allocations{}, downloads{}, uploads{}, finishes{}, waits{};
    u32 host_read_barriers{};
    u64 upload_bytes{}, waited_tick{};
    std::unique_ptr<Buffer> staging;
    std::vector<std::unique_ptr<Buffer>> retired_staging;
    std::vector<u8> stream;
    std::vector<std::function<void()>> pending;
    std::function<void()> before_completion;
    bool list_copy_ready{true};
    std::function<void()> before_list_wait;
    std::function<void(Buffer&, Buffer&)> before_gpu_copy;
    std::function<void(Buffer&)> before_clear;
    std::function<void(Buffer&)> before_delete;
};

struct Tracker : VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager> {
    using MemoryTrackerBase::MemoryTrackerBase;
    void MarkRegionAsCpuModified(DAddr address, u64 size) {
        ++cpu_marks;
        MemoryTrackerBase::MarkRegionAsCpuModified(address, size);
    }
    u32 cpu_marks{};
};

struct Policy {
    using Runtime = ::Runtime;
    using Buffer = ::Buffer;
    using Async_Buffer = Staging;
    using MemoryTracker = Tracker;
    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using Cache = VideoCommon::BufferCache<Policy>;

} // namespace

namespace VideoCommon {

struct BufferCacheTestAccess {
    static Binding Bind(Cache& cache, Tegra::MemoryManager& memory, GPUVAddr descriptor,
                        bool written = false, bool native = false, bool table = false) {
        cache.gpu_memory = &memory;
        return cache.StorageBufferBinding(descriptor, 0, written, native, table);
    }
    static BufferId Add(Cache& cache, DAddr address, u32 size, bool mapped = false) {
        const auto id = cache.slot_buffers.insert(cache.runtime, address, size, mapped);
        cache.Register(id);
        cache.memory_tracker.ForEachUploadRange(address, size, [](auto, auto) {});
        return id;
    }
    static Buffer& Get(Cache& cache, BufferId id) {
        return cache.slot_buffers[id];
    }
    static void Dirty(Cache& cache, DAddr address, u64 size) {
        cache.memory_tracker.MarkRegionAsGpuModified(address, size);
        cache.gpu_modified_ranges.Add(address, size);
        cache.uncommitted_gpu_modified_ranges.Add(address, size);
        cache.committed_gpu_modified_ranges.emplace_back().Add(address, size);
        cache.pending_download_ranges.emplace_back().Add(address, size);
    }
    static bool Pending(Cache& cache, DAddr address, u64 size) {
        if (cache.gpu_modified_ranges.Intersects(address, size) ||
            cache.uncommitted_gpu_modified_ranges.Intersects(address, size)) {
            return true;
        }
        for (const auto& ranges : cache.committed_gpu_modified_ranges) {
            if (ranges.Intersects(address, size))
                return true;
        }
        for (const auto& ranges : cache.pending_download_ranges) {
            if (ranges.Intersects(address, size))
                return true;
        }
        return false;
    }
    static void DrawWrite(Cache& cache, DAddr page) {
        std::scoped_lock lock{cache.nx2_list_mutex};
        auto& entry = cache.nx2_list_pages[page];
        entry.draw_written = true;
        ++entry.writes;
    }
    static void QueueListCopy(Cache& cache, BufferId id, DAddr page) {
        std::scoped_lock lock{cache.nx2_list_mutex};
        ++cache.nx2_list_pages[page].writes;
        cache.nx2_list_copy_requests.emplace_back(id, page);
    }
    static void AttachChannel(Cache& cache, BufferCacheChannelInfo& channel) {
        cache.channel_state = &channel;
        cache.maxwell3d = &channel.maxwell3d;
        cache.gpu_memory = &channel.gpu_memory;
    }
    static BufferId Find(Cache& cache, DAddr address, u32 size) {
        return cache.FindBuffer(address, size, false);
    }
    static void Delete(Cache& cache, BufferId id) {
        cache.DeleteBuffer(id);
    }
    static bool Contains(Cache& cache, BufferId id) {
        return cache.slot_buffers.contains(id);
    }
    static void ApplyExact(Cache& cache, const DeferredWrites::Write& write) {
        REQUIRE(cache.ApplyDeferredWrite(write));
    }
    static bool CpuDirty(Cache& cache, DAddr address, u64 size) {
        return cache.memory_tracker.IsRegionCpuModified(address, size);
    }
    static void MarkCpuDirty(Cache& cache, DAddr address, u64 size) {
        cache.memory_tracker.MarkRegionAsCpuModified(address, size);
    }
    static void Download(Cache& cache, std::span<const std::pair<DAddr, u64>> ranges) {
        cache.DownloadRangesMemory(ranges);
    }
    static void DownloadBuffers(Cache& cache, std::span<const BufferId> ids) {
        cache.DownloadBuffersMemory(ids);
    }
    static void GpuWrite(Cache& cache, BufferId id, DAddr address, u32 size) {
        cache.MarkWrittenBuffer(id, address, size);
        // Request a fence copy independently of the game's async-download policy.
        cache.uncommitted_gpu_modified_ranges.Add(address, size);
    }
    static void Uniform(Cache& cache, BufferCacheChannelInfo& channel, BufferId id, DAddr address,
                        u32 size) {
        cache.channel_state = &channel;
        cache.maxwell3d = &channel.maxwell3d;
        cache.gpu_memory = &channel.gpu_memory;
        channel.uniform_buffers[0][0] = {.device_addr = address, .size = size, .buffer_id = id};
        cache.BindHostGraphicsUniformBuffer(0, 0, 0, true);
    }
    static void NextFrame(Cache& cache) {
        cache.native_indirect_uniforms_this_frame.clear();
    }
    static u32 CpuMarks(Cache& cache) {
        return cache.memory_tracker.cpu_marks;
    }
    static size_t Deferred(Cache& cache) {
        std::scoped_lock lock{cache.deferred_page_writes_mutex};
        return cache.deferred_page_writes.size();
    }
    static void Defer(Cache& cache, DAddr address, u64 size) {
        cache.DeferPageWrite(address, size);
    }
    static void ApplyDeferred(Cache& cache) {
        cache.ApplyDeferredPageWrites();
    }
    static std::vector<std::pair<DAddr, u64>> DeferredRanges(Cache& cache) {
        std::scoped_lock lock{cache.deferred_page_writes_mutex};
        return cache.deferred_page_writes;
    }
};

} // namespace VideoCommon

namespace {

using Access = VideoCommon::BufferCacheTestAccess;

struct Fixture {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    std::optional<Tegra::MemoryManager> memory;
    Runtime runtime;
    std::unique_ptr<Cache> cache;
    std::unique_ptr<Kernel::KProcess> process;
    Core::Asid asid{};
    u8* bytes{};

    Fixture() {
        system.Initialize();
        process = std::make_unique<Kernel::KProcess>(system.Kernel());
        auto& table = process->GetPageTable().GetImpl();
        table.Resize(24, 12);
        bytes = physical.GetPointerFromRaw<u8>(0x10000);
        std::memset(bytes, 0, 0x40000);
        for (u64 page = 0; page < 64; ++page) {
            table.entries.GetAndFault(0x10 + page)
                .Store(false, Common::PageType::Memory, 1,
                       reinterpret_cast<uintptr_t>(bytes) - 0x10000);
        }
        process->GetMemory().SetCurrentPageTable(*process);
        asid = device.RegisterProcess(&process->GetMemory());
        device.BindInterface(&rasterizer);
        device.Map(0x10000, 0x10000, 0x40000, asid);
        ResetMemory();
        cache = std::make_unique<Cache>(device, runtime);
    }
    ~Fixture() {
        cache.reset();
        memory.reset();
        device.Unmap(0x10000, 0x40000);
        device.UnregisterProcess(asid);
        system.Kernel().UnregisterKernelObject(process.get());
    }
    void ResetMemory() {
        memory.reset();
        memory.emplace(system, device, 32, 1ULL << 31, 16);
        memory->BindRasterizer(&rasterizer);
        memory->Map(0x100000, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);
    }
    void Descriptor(GPUVAddr address, u64 size) {
        std::memcpy(bytes, &address, sizeof(address));
        std::memcpy(bytes + 8, &size, sizeof(size));
    }
    VideoCommon::Binding Bind(bool written = false, bool native = false, bool table = false) {
        return Access::Bind(*cache, *memory, 0x100000, written, native, table);
    }
};

} // namespace

TEST_CASE("Deferred admission permits clean GPU ownership but requires a complete CPU-clean buffer",
          "[buffer_ports][deferred_writes][deferred_admission]") {
    using Reason = Common::GpuTransitionProfile::DeferredFallbackReason;
    Fixture f;
    CHECK(f.cache->ValidateDeferredWritePage(0x20000) == Reason::NoGpuBuffer);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    CHECK(f.cache->ValidateDeferredWritePage(0x20000) == Reason::Success);
    Access::Dirty(*f.cache, 0x20080, 16);
    CHECK(f.cache->ValidateDeferredWritePage(0x20000) == Reason::Success);
    CHECK(f.cache->ValidateDeferredWritePage(0x20001) == Reason::InvalidOperands);
    SECTION("CPU-dirty neighbor makes the entire page ineligible") {
        Access::MarkCpuDirty(*f.cache, 0x20f00, 4);
        CHECK(f.cache->ValidateDeferredWritePage(0x20000) == Reason::CpuModified);
    }
    SECTION("Readback before the queued grant leaves the clean buffer eligible") {
        f.cache->DownloadMemory(0x20000, 0x1000);
        CHECK_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
        CHECK(f.cache->ValidateDeferredWritePage(0x20000) == Reason::Success);
        CHECK(f.runtime.finishes == 1);
        return;
    }
    SECTION("Queued old page writes must finish before granting") {
        Access::Defer(*f.cache, 0x20004, 4);
        CHECK(f.cache->ValidateDeferredWritePage(0x20000) == Reason::PendingWrites);
        CHECK(Access::Deferred(*f.cache) == 1);
    }
    CHECK(f.runtime.uploads == 0);
    CHECK(f.runtime.finishes == 0);
    CHECK(Access::Get(*f.cache, id).IsInBounds(0x20000, 4096));
}

TEST_CASE("Deferred admission cannot use a buffer containing only part of a CPU page",
          "[buffer_ports][deferred_writes][deferred_admission]") {
    using Reason = Common::GpuTransitionProfile::DeferredFallbackReason;
    Fixture f;
    Access::Add(*f.cache, 0x20080, 128);
    Access::Dirty(*f.cache, 0x20080, 16);
    CHECK(f.cache->ValidateDeferredWritePage(0x20000) == Reason::NoGpuBufferBounds);
    CHECK(f.runtime.uploads == 0);
    CHECK(f.runtime.finishes == 0);
}

TEST_CASE("Fresh device mapping validates the entire target before permission or mutation",
          "[buffer_ports][deferred_boundaries]") {
    Fixture f;
    constexpr DAddr target = 0x80000;
    u32 permits{};
    REQUIRE_FALSE(f.device.TryMapUnbacked(target, 0x10000, 0x2000, f.asid, true, [&] {
        ++permits;
        return false;
    }));
    REQUIRE(permits == 1);
    REQUIRE(f.device.GetPointer<u8>(target) == nullptr);
    REQUIRE(f.device.GetPointer<u8>(target + 0x1000) == nullptr);

    f.device.Map(target + 0x1000, 0x21000, 0x1000, f.asid);
    SCOPE_EXIT { f.device.Unmap(target, 0x2000); };
    REQUIRE_FALSE(f.device.TryMapUnbacked(target, 0x10000, 0x2000, f.asid, true, [&] {
        ++permits;
        return true;
    }));
    REQUIRE(permits == 1);
    REQUIRE(f.device.GetPointer<u8>(target) == nullptr);
    REQUIRE(f.device.GetPointer<u8>(target + 0x1000) == f.bytes + 0x11000);

    f.device.Unmap(target + 0x1000, 0x1000);
    REQUIRE(f.device.TryMapUnbacked(target, 0x10000, 0x2000, f.asid, true, [&] {
        ++permits;
        return true;
    }));
    REQUIRE(permits == 2);
    REQUIRE(f.device.GetPointer<u8>(target) == f.bytes);
    REQUIRE(f.device.GetPointer<u8>(target + 0x1000) == f.bytes + 0x1000);
    REQUIRE_FALSE(f.device.TryMapUnbacked(target + 1, 0x10000, 0x1000, f.asid, false,
                                         [] { return true; }));
}

TEST_CASE("Concurrent fresh device mappings cannot both receive permission for the same range",
          "[buffer_ports][deferred_boundaries]") {
    Fixture f;
    constexpr DAddr target = 0x80000;
    SCOPE_EXIT { f.device.Unmap(target, 0x1000); };
    std::barrier start{2};
    std::atomic<u32> permits{};
    bool first{}, second{};
    std::jthread a{[&] {
        start.arrive_and_wait();
        first = f.device.TryMapUnbacked(target, 0x10000, 0x1000, f.asid, true, [&] {
            ++permits;
            return true;
        });
    }};
    std::jthread b{[&] {
        start.arrive_and_wait();
        second = f.device.TryMapUnbacked(target, 0x20000, 0x1000, f.asid, true, [&] {
            ++permits;
            return true;
        });
    }};
    a.join();
    b.join();
    REQUIRE(first != second);
    REQUIRE(permits.load() == 1);
    REQUIRE(f.device.GetPointer<u8>(target) == f.bytes + (first ? 0 : 0x10000));
}

TEST_CASE("Physical device mappings validate all spans before permission or publication",
          "[buffer_ports][deferred_boundaries][physical_backing]") {
    ScopedSystemTestContext context;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    constexpr DAddr target = 0x80000;
    DAddr destination = target;
    std::vector<Core::DeviceMemoryPhysicalSpan> spans{{0x10000, 0x1000}, {0x20000, 0x1000}};
    u32 permits{};
    SECTION("Empty list") { spans.clear(); }
    SECTION("Zero size") { spans.back().size = 0; }
    SECTION("Unaligned physical offset") { ++spans.back().offset; }
    SECTION("Unaligned physical size") { ++spans.back().size; }
    SECTION("Outside actual DRAM") { spans.back().offset = physical.buffer.BackingSize(); }
    SECTION("Physical range crosses DRAM end") {
        spans.back() = {physical.buffer.BackingSize() - 0x1000, 0x2000};
    }
    SECTION("Overflowing physical offset") { spans.back().offset = ~u64{0} - 0xfff; }
    SECTION("Unaligned device address") { ++destination; }
    SECTION("Device zero is the reverse-map sentinel") { destination = 0; }
    SECTION("Device range crosses address space end") {
        destination = (1ULL << Tegra::MaxwellDeviceMemoryManager::AS_BITS) - 0x1000;
    }
    REQUIRE_FALSE(device.TryMapPhysicalUnbacked(destination, spans, [&] {
        ++permits;
        return true;
    }));
    REQUIRE(permits == 0);
    REQUIRE(device.GetPointer<u8>(target) == nullptr);
    REQUIRE(device.GetPointer<u8>(target + 0x1000) == nullptr);
}

TEST_CASE("Physical device mapping rejects occupied pages and denied permission without partial writes",
          "[buffer_ports][deferred_boundaries][physical_backing]") {
    ScopedSystemTestContext context;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    constexpr DAddr target = 0x80000;
    const std::array spans{Core::DeviceMemoryPhysicalSpan{0x10000, 0x2000}};
    SCOPE_EXIT { device.Unmap(target, 0x2000); };
    u32 permits{};
    REQUIRE_FALSE(device.TryMapPhysicalUnbacked(target, spans, [&] {
        ++permits;
        return false;
    }));
    REQUIRE(permits == 1);
    REQUIRE(device.GetPointer<u8>(target) == nullptr);
    REQUIRE(device.GetPointer<u8>(target + 0x1000) == nullptr);
    const std::array occupied{Core::DeviceMemoryPhysicalSpan{0x30000, 0x1000}};
    REQUIRE(device.TryMapPhysicalUnbacked(target + 0x1000, occupied, [] { return true; }));
    REQUIRE_FALSE(device.TryMapPhysicalUnbacked(target, spans, [&] {
        ++permits;
        return true;
    }));
    REQUIRE(permits == 1);
    REQUIRE(device.GetPointer<u8>(target) == nullptr);
    REQUIRE(device.GetPointer<u8>(target + 0x1000) == physical.GetPointerFromRaw<u8>(0x30000));
}

TEST_CASE("Physical backing is immediately visible without a process and preserves fragmented aliases",
          "[buffer_ports][deferred_boundaries][physical_backing]") {
    ScopedSystemTestContext context;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    constexpr DAddr target = 0x80000;
    constexpr DAddr alias = 0x90000;
    const std::array spans{Core::DeviceMemoryPhysicalSpan{0x10000, 0x1000},
                           Core::DeviceMemoryPhysicalSpan{0x11000, 0x1000},
                           Core::DeviceMemoryPhysicalSpan{0x30000, 0x1000}};
    SCOPE_EXIT {
        device.Unmap(alias, 0x1000);
        device.Unmap(target, 0x3000);
    };
    REQUIRE(device.TryMapPhysicalUnbacked(target, spans, [] { return true; }));
    REQUIRE(device.GetSpan(target, 0x2000) == physical.GetPointerFromRaw<u8>(0x10000));
    REQUIRE(device.GetSpan(target, 0x3000) == nullptr);
    const std::array<u8, 4> pattern{0x12, 0x34, 0x56, 0x78};
    device.WriteBlockUnsafe(target + 0x1ffe, pattern.data(), pattern.size());
    REQUIRE(physical.GetPointerFromRaw<u8>(0x11ffe)[0] == pattern[0]);
    REQUIRE(physical.GetPointerFromRaw<u8>(0x11ffe)[1] == pattern[1]);
    REQUIRE(physical.GetPointerFromRaw<u8>(0x30000)[0] == pattern[2]);
    REQUIRE(physical.GetPointerFromRaw<u8>(0x30000)[1] == pattern[3]);
    std::array<u8, 4> readback{};
    device.ReadBlockUnsafe(target + 0x1ffe, readback.data(), readback.size());
    REQUIRE(readback == pattern);
    // These calls must not dereference a process/ASID for physical-only pages.
    device.UpdatePagesCachedCount(target, 0x3000, 1);
    device.UpdatePagesCachedCount(target, 0x3000, -1);

    const std::array alias_span{Core::DeviceMemoryPhysicalSpan{0x10000, 0x1000}};
    REQUIRE(device.TryMapPhysicalUnbacked(alias, alias_span, [] { return true; }));
    Common::ScratchBuffer<u32> scratch;
    std::vector<DAddr> aliases;
    device.ApplyOpOnPAddr(0x10000, scratch, [&](DAddr address) { aliases.push_back(address); });
    std::sort(aliases.begin(), aliases.end());
    const std::vector<DAddr> expected{target, alias};
    REQUIRE(aliases == expected);
    device.Unmap(alias, 0x1000);
    aliases.clear();
    device.ApplyOpOnPAddr(0x10000, scratch, [&](DAddr address) { aliases.push_back(address); });
    REQUIRE(aliases == std::vector<DAddr>{target});

    device.Unmap(target + 0x1000, 0x1000);
    REQUIRE(device.GetSpan(target, 0x2000) == nullptr);
    const std::array replacement{Core::DeviceMemoryPhysicalSpan{0x40000, 0x1000}};
    REQUIRE(device.TryMapPhysicalUnbacked(target + 0x1000, replacement, [] { return true; }));
    REQUIRE(device.GetPointer<u8>(target + 0x1000) == physical.GetPointerFromRaw<u8>(0x40000));
    REQUIRE(device.GetSpan(target, 0x2000) == nullptr);
}

TEST_CASE("Concurrent physical backing claims grant permission to only one producer",
          "[buffer_ports][deferred_boundaries][physical_backing]") {
    ScopedSystemTestContext context;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    constexpr DAddr target = 0x80000;
    SCOPE_EXIT { device.Unmap(target, 0x1000); };
    std::barrier start{2};
    std::atomic<u32> permits{};
    bool first{}, second{};
    const auto claim = [&](u64 offset) {
        const std::array spans{Core::DeviceMemoryPhysicalSpan{offset, 0x1000}};
        start.arrive_and_wait();
        return device.TryMapPhysicalUnbacked(target, spans, [&] {
            ++permits;
            return true;
        });
    };
    std::jthread a{[&] { first = claim(0x10000); }};
    std::jthread b{[&] { second = claim(0x20000); }};
    a.join();
    b.join();
    REQUIRE(first != second);
    REQUIRE(permits.load() == 1);
    REQUIRE(device.GetPointer<u8>(target) == physical.GetPointerFromRaw<u8>(first ? 0x10000 : 0x20000));
}

TEST_CASE("Nvdrv session id reuse preserves queued and channel mapping owners",
          "[buffer_ports][deferred_boundaries]") {
    Fixture f;
    const bool execute_callback = GENERATE(false, true);
    Kernel::KAutoObject::Create(f.process.get());
    // Host1x is too large for the test thread's stack.
    auto host1x_owner = std::make_unique<Tegra::Host1x::Host1x>(f.system);
    auto& host1x = *host1x_owner;
    host1x.MemoryManager().BindInterface(&f.rasterizer);
    auto container_owner = std::make_unique<Service::Nvidia::NvCore::Container>(host1x);
    auto& container = *container_owner;
    const auto first_id = container.OpenSession(f.process.get());
    auto first = container.GetSessionLifetime(first_id);
    REQUIRE(first);
    const auto first_asid = first->asid;
    REQUIRE(f.process->GetReferenceCount() == 2);

    std::weak_ptr old_owner = first;
    std::function<void()> queued = [owner = first] { REQUIRE(owner->process != nullptr); };
    Tegra::Control::ChannelState channel{0};
    channel.memory_manager_owner = std::shared_ptr<Tegra::MemoryManager>(
        new Tegra::MemoryManager(f.system, host1x.MemoryManager(), 32, 1ULL << 31, 16),
        [owner = first](Tegra::MemoryManager* memory) mutable {
            delete memory;
            owner.reset();
        });
    channel.memory_manager = channel.memory_manager_owner.get();
    first.reset();
    container.CloseSession(first_id);
    REQUIRE(container.GetSession(first_id) == nullptr);
    REQUIRE_FALSE(old_owner.expired());

    const auto second_id = container.OpenSession(f.process.get());
    auto second = container.GetSessionLifetime(second_id);
    REQUIRE(second_id.id == first_id.id);
    REQUIRE(second->asid.id != first_asid.id);
    REQUIRE(f.process->GetReferenceCount() == 3);
    if (execute_callback) {
        queued();
    }
    queued = {};
    REQUIRE_FALSE(old_owner.expired());
    channel.memory_manager_owner.reset();
    channel.memory_manager = nullptr;
    REQUIRE(old_owner.expired());
    REQUIRE(f.process->GetReferenceCount() == 2);

    const auto recycled = host1x.MemoryManager().RegisterProcess(&f.process->GetMemory());
    REQUIRE(recycled.id == first_asid.id);
    host1x.MemoryManager().UnregisterProcess(recycled);
    second.reset();
    container.CloseSession(second_id);
    REQUIRE(f.process->GetReferenceCount() == 1);
}

TEST_CASE("Device mappings can retire after the rasterizer has been detached",
          "[buffer_ports][deferred_boundaries]") {
    Fixture f;
    constexpr DAddr target = 0x80000;
    f.device.Map(target, 0x10000, 0x1000, f.asid);
    REQUIRE(f.device.GetPointer<u8>(target) == f.bytes);
    f.device.BindInterface(nullptr);
    SCOPE_EXIT { f.device.BindInterface(&f.rasterizer); };
    f.device.Unmap(target, 0x1000);
    REQUIRE(f.device.GetPointer<u8>(target) == nullptr);
}

TEST_CASE("Storage binding cache tracks map changes and descriptor dependencies",
          "[buffer_ports]") {
    Fixture f;
    constexpr GPUVAddr base = 0x200000;
    f.memory->Map(base, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.Descriptor(base + 0x123, 0x456);
    for (int i = 0; i < 3; ++i) {
        const auto binding = f.Bind(true);
        REQUIRE(binding.device_addr == 0x20100);
        REQUIRE(binding.gpu_addr == base + 0x100);
        REQUIRE(binding.size == 0x479);
        REQUIRE(binding.buffer_id == VideoCommon::BufferId{});
        REQUIRE(binding.segment_count == 0);
    }
    REQUIRE(f.Bind().size == 0xf00);
    f.Descriptor(base + 0x123, 0x1456);
    REQUIRE(f.Bind(true).size == 0x1479);
    f.runtime.alignment = 4096;
    REQUIRE(f.Bind(true).device_addr == 0x20000);
    REQUIRE(f.Bind(true).size == 0x1579);
    REQUIRE(f.Bind(true, true).size == 0x10000);
    f.memory->Map(base, 0x30000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(f.Bind(true).device_addr == 0x30000);
    f.memory->MapSparse(base, 0x10000, true);
    REQUIRE(f.Bind().size == 0);
    f.memory->Unmap(base, 0x10000);
    REQUIRE(f.Bind().size == 0);
    f.memory->Map(base, 0x20000, 0x2000, Tegra::PTEKind::PITCH, false);
    REQUIRE(f.Bind(true, true).size == 0x2000);
    f.memory->Map(base + 0x1000, 0x40000, 0x1000, Tegra::PTEKind::PITCH, false);
    REQUIRE(f.Bind(true, true).size == 0x1000);
    const auto old_owner = f.memory->GetID();
    f.ResetMemory();
    f.memory->Map(base, 0x40000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(f.memory->GetID() != old_owner);
    REQUIRE(f.Bind(true).device_addr == 0x40000);
    f.Descriptor(0, 0x123);
    REQUIRE(f.Bind().size == 0);
}

TEST_CASE("Storage binding cache preserves native and table size policies", "[buffer_ports]") {
    Fixture f;
    constexpr GPUVAddr base = 0x4000000;
    f.memory->Map(base, 0x10000000, 80 << 20, Tegra::PTEKind::PITCH, true);
    f.Descriptor(base, 256);
    REQUIRE(f.Bind().size == 4096);
    REQUIRE(f.Bind(false, true).size == 8 << 20);
    REQUIRE(f.Bind(false, false, true).size == 64 << 20);
    REQUIRE(f.Bind(true, false, true).size == 256);
    f.Descriptor(base, (1ULL << 32) | 256);
    REQUIRE(f.Bind().size == 8 << 20);
    REQUIRE(f.Bind(false, false, true).size == 64 << 20);
    f.Descriptor(base + (79 << 20), 0);
    REQUIRE(f.Bind(false, false, true).size == 1 << 20);
}

TEST_CASE("Storage binding cache does not reuse a destroyed address space", "[buffer_ports]") {
    Fixture f;
    f.memory->Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.Descriptor(0x200000, 0x1000);
    REQUIRE(f.Bind(true).device_addr == 0x20000);
    const auto serial = f.memory->MapSerial();
    const auto owner = f.memory->GetID();
    const auto* address = &*f.memory;
    f.ResetMemory();
    f.memory->Map(0x200000, 0x30000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(&*f.memory == address);
    REQUIRE(f.memory->MapSerial() == serial);
    REQUIRE(f.memory->GetID() != owner);
    REQUIRE(f.Bind(true).device_addr == 0x30000);
}

TEST_CASE("Batch downloads clamp buffers, deduplicate pages and finish before publication",
          "[buffer_ports]") {
    Fixture f;
    const auto first = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto second = Access::Add(*f.cache, 0x30000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, first).bytes, 0x12);
    std::ranges::fill(Access::Get(*f.cache, second).bytes, 0x34);
    Access::Dirty(*f.cache, 0x2f000, 0x2000);
    Access::Dirty(*f.cache, 0x32000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.bytes[0x1f000] == 0);
        REQUIRE(f.bytes[0x20000] == 0);
    };
    const std::array<std::pair<DAddr, u64>, 3> ranges{{
        {0x2f008, 0x1000},
        {0x2f000, 0x2000},
        {0x20000, 0},
    }};
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.allocations == 1);
    REQUIRE(f.runtime.downloads == 2);
    REQUIRE(f.bytes[0x1f000] == 0x12);
    REQUIRE(f.bytes[0x1ffff] == 0x12);
    REQUIRE(f.bytes[0x20000] == 0x34);
    REQUIRE(f.bytes[0x20fff] == 0x34);
    REQUIRE(f.bytes[0x22000] == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x2f000, 0x2000));
    REQUIRE(Access::Pending(*f.cache, 0x32000, 0x1000));
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.finishes == 1);
}

TEST_CASE("Mapped and mixed batches make device writes host visible in one completion",
          "[buffer_ports]") {
    Fixture f;
    const auto first = Access::Add(*f.cache, 0x20000, 0x10000, true);
    const auto second = Access::Add(*f.cache, 0x30000, 0x10000, true);
    const auto clean = Access::Add(*f.cache, 0x40000, 0x10000, true);
    auto& a = Access::Get(*f.cache, first);
    auto& b = Access::Get(*f.cache, second);
    a.tick = 8;
    b.tick = 19;
    Access::Get(*f.cache, clean).tick = 99;
    const std::array<std::pair<DAddr, u64>, 3> ranges{{
        {0x20000, 0x1000},
        {0x30000, 0x1000},
        {0x40000, 0x1000},
    }};
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::Dirty(*f.cache, 0x30000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        REQUIRE(f.bytes[0x20000] == 0);
        std::ranges::fill(a.bytes, 0x56);
        std::ranges::fill(b.bytes, 0x78);
    };
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.host_read_barriers == 1);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.allocations == 0);
    REQUIRE(f.bytes[0x10000] == 0x56);
    REQUIRE(f.bytes[0x20000] == 0x78);
    b.host_mapped = false;
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::Dirty(*f.cache, 0x30000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 2);
        REQUIRE(f.bytes[0x10000] == 0x56);
        REQUIRE(f.bytes[0x20000] == 0x78);
        std::ranges::fill(a.bytes, 0x9a);
        std::ranges::fill(b.bytes, 0xbc);
    };
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.finishes == 2);
    REQUIRE(f.runtime.downloads == 1);
    REQUIRE(f.bytes[0x10000] == 0x9a);
    REQUIRE(f.bytes[0x20000] == 0xbc);
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.host_read_barriers == 2);
    REQUIRE(f.runtime.finishes == 2);
}

TEST_CASE("Single mapped download completes host visibility before guest publication",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, true);
    auto& buffer = Access::Get(*f.cache, id);
    buffer.tick = 7;
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        std::ranges::fill(buffer.bytes, 0xcd);
    };
    f.cache->DownloadMemory(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.allocations == 0);
    REQUIRE(f.bytes[0x10000] == 0xcd);
    f.cache->DownloadMemory(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 1);
}

TEST_CASE("Command page batch publishes only completed matching write generations",
          "[buffer_ports]") {
    Fixture f;
    const bool mapped = GENERATE(false, true);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, mapped);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x42);
    Access::Dirty(*f.cache, 0x20000, 0x3000);
    Access::DrawWrite(*f.cache, 0x20000);
    Access::DrawWrite(*f.cache, 0x21000);
    Access::DrawWrite(*f.cache, 0x22000);
    std::array<u8, 4> out{};
    using Read = Cache::Nx2ListRead;
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        REQUIRE(f.cache->Nx2ReadListCopy(0x20000, out, true) == Read::Unknown);
        Access::DrawWrite(*f.cache, 0x22000);
    };
    const std::array<DAddr, 1> requested{0x20000};
    f.cache->Nx2FlushListPages(requested);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.downloads == (mapped ? 0u : 3u));
    REQUIRE(f.cache->Nx2ReadListCopy(0x20000, out, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x22000, out, true) == Read::Unknown);
    REQUIRE(f.bytes[0x12000] == 0x42);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, false) == Read::NeedsSync);
    Access::DrawWrite(*f.cache, 0x21000);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, true) == Read::Unknown);
}

TEST_CASE("Command page readers remain ordered during batch completion", "[buffer_ports]") {
    struct {
        Core::DeviceMemory physical;
        Tegra::MaxwellDeviceMemoryManager device{physical};
        Runtime runtime;
        std::unique_ptr<Cache> cache = std::make_unique<Cache>(device, runtime);
    } f;
    Access::Add(*f.cache, 0x20000, 0x10000);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::DrawWrite(*f.cache, 0x20000);
    std::barrier rendezvous{2};
    Cache::Nx2ListRead before{}, after{};
    std::jthread reader{[&] {
        std::array<u8, 4> out{};
        rendezvous.arrive_and_wait();
        before = f.cache->Nx2ReadListCopy(0x20000, out, true);
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
        after = f.cache->Nx2ReadListCopy(0x20000, out, true);
    }};
    f.runtime.before_completion = [&] {
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
    };
    const std::array<DAddr, 1> requested{0x20000};
    f.cache->Nx2FlushListPages(requested);
    rendezvous.arrive_and_wait();
    reader.join();
    REQUIRE(before == Cache::Nx2ListRead::Unknown);
    REQUIRE(after == Cache::Nx2ListRead::Synced);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.downloads == 1);
}

TEST_CASE("Native uniform banks upload directly without overwriting GPU-owned bytes",
          "[buffer_ports]") {
    Fixture f;
    f.memory->EnableNativeTiling();
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x10000;
    channel.uniform_buffer_sizes = &sizes;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::memset(f.bytes + 0x10000, 0x5a, 0x10000);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x10000);
    REQUIRE(Access::CpuMarks(*f.cache) == 0);
    REQUIRE(Access::Get(*f.cache, id).bytes[0xffff] == 0x5a);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.uploads == 1);
    Access::NextFrame(*f.cache);
    Access::Dirty(*f.cache, 0x21000, 0x1000);
    std::memset(f.bytes + 0x10000, 0xa5, 0x10000);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.uploads == 1);
    REQUIRE(Access::Get(*f.cache, id).bytes[0x1000] == 0x5a);
}

TEST_CASE("Native uniform uploads stop at holes and keep unaligned bank boundaries",
          "[buffer_ports]") {
    Fixture f;
    f.memory->EnableNativeTiling();
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x10000;
    channel.uniform_buffer_sizes = &sizes;
    const auto id = Access::Add(*f.cache, 0x20000, 0x20000);
    std::memset(f.bytes + 0x10000, 0x6b, 0x20000);
    f.device.Unmap(0x22000, 0x1000);
    Access::Uniform(*f.cache, channel, id, 0x20123, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x1edd);
    const auto& data = Access::Get(*f.cache, id).bytes;
    REQUIRE(data[0x122] == 0);
    REQUIRE(data[0x123] == 0x6b);
    REQUIRE(data[0x1fff] == 0x6b);
    REQUIRE(data[0x2000] == 0);
    REQUIRE(data[0x3000] == 0);
    REQUIRE(Access::CpuMarks(*f.cache) == 0);
    f.device.Map(0x22000, 0x22000, 0x1000, f.asid);
    Access::NextFrame(*f.cache);
    Access::Uniform(*f.cache, channel, id, 0x20123, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x11edd);
    REQUIRE(data[0x10122] == 0x6b);
    REQUIRE(data[0x10123] == 0);
}

TEST_CASE("Downloads keep CPU writes collected and not yet applied", "[buffer_ports]") {
    Fixture f;
    const auto staged = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto mapped = Access::Add(*f.cache, 0x30000, 0x10000, true);
    std::ranges::fill(Access::Get(*f.cache, staged).bytes, 0x12);
    std::ranges::fill(Access::Get(*f.cache, mapped).bytes, 0x34);
    f.cache->SetPendingCpuWrites([](const std::function<void(DAddr, size_t)>& callback) {
        callback(0x20100, 0x10);
        callback(0x30ff8, 0x10);
    });
    std::memset(f.bytes + 0x10100, 0xee, 0x10);
    std::memset(f.bytes + 0x20ff8, 0xee, 0x10);
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    Access::Dirty(*f.cache, 0x30000, 0x2000);
    const std::array<std::pair<DAddr, u64>, 1> ranges{{{0x20000, 0x2000}}};
    Access::Download(*f.cache, ranges);
    REQUIRE(f.bytes[0x100ff] == 0x12);
    REQUIRE(std::all_of(f.bytes + 0x10100, f.bytes + 0x10110, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x10110] == 0x12);
    REQUIRE(f.bytes[0x11fff] == 0x12);
    f.cache->DownloadMemory(0x30000, 0x2000);
    REQUIRE(f.bytes[0x20ff7] == 0x34);
    REQUIRE(std::all_of(f.bytes + 0x20ff8, f.bytes + 0x21008, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x21008] == 0x34);
}

namespace {

/// Runs work on a thread marked as the GPU thread (the scheduler's owner), one call at a time.
class OwnerThread {
public:
    OwnerThread() : thread{[this](std::stop_token stop) { Run(stop); }} {
        std::unique_lock lock{mutex};
        ready.wait(lock, [&] { return marked; });
    }
    ~OwnerThread() {
        {
            std::scoped_lock lock{mutex};
            thread.request_stop();
        }
        wake.notify_all();
        thread.join();
    }
    /// Runs `work` on the owner and rethrows what it threw (a failed REQUIRE included).
    void Do(std::function<void()> work) {
        std::unique_lock lock{mutex};
        job = std::move(work);
        wake.notify_all();
        ready.wait(lock, [&] { return !job; });
        if (auto thrown = std::exchange(error, nullptr)) {
            std::rethrow_exception(thrown);
        }
    }

private:
    void Run(std::stop_token stop) {
        VideoCommon::GpuSplit::MarkGpuThread();
        SCOPE_EXIT {
            VideoCommon::GpuSplit::UnmarkGpuThread();
        };
        {
            std::scoped_lock lock{mutex};
            marked = true;
        }
        ready.notify_all();
        std::unique_lock lock{mutex};
        while (!stop.stop_requested()) {
            wake.wait(lock, [&] { return job || stop.stop_requested(); });
            if (job) {
                try {
                    job();
                } catch (...) {
                    error = std::current_exception();
                }
                job = nullptr;
                ready.notify_all();
            }
        }
    }

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable ready;
    std::function<void()> job;
    std::exception_ptr error;
    bool marked{};
    std::jthread thread;
};

struct DeferredWriteFixture : Fixture {
    DeferredWriteFixture() {
        id = Access::Add(*cache, 0x20000, 0x10000);
        std::ranges::fill(Access::Get(*cache, id).bytes, 0x12);
        Access::Dirty(*cache, 0x20000, 0x1000);
        // The guest CPU writes 16 bytes in the middle of the GPU-written page.
        std::memset(bytes + 0x10100, 0xee, 0x10);
    }
    bool GuestHasGpuBytes() const {
        return bytes[0x10000] == 0x12 && bytes[0x100ff] == 0x12 && bytes[0x10110] == 0x12 &&
               bytes[0x10fff] == 0x12;
    }
    bool GuestKeepsCpuBytes() const {
        return std::all_of(bytes + 0x10100, bytes + 0x10110, [](u8 b) { return b == 0xee; });
    }
    VideoCommon::BufferId id;
};

} // namespace

TEST_CASE("Without a GPU thread a partial page write keeps the old behaviour",
          "[buffer_ports][deferred_page_writes]") {
    DeferredWriteFixture f;
    REQUIRE_FALSE(VideoCommon::GpuSplit::HasGpuThread());
    const u32 marks = Access::CpuMarks(*f.cache);
    f.cache->WriteMemory(0x20100, 0x10);
    CHECK(Access::Deferred(*f.cache) == 0);
    CHECK(Access::CpuMarks(*f.cache) == marks + 1);
    CHECK(f.runtime.downloads == 0);
}

TEST_CASE("A non-owner partial page write waits for the owner, which keeps the GPU bytes",
          "[buffer_ports][deferred_page_writes]") {
    DeferredWriteFixture f;
    OwnerThread owner;
    REQUIRE(VideoCommon::GpuSplit::HasGpuThread());
    REQUIRE_FALSE(VideoCommon::GpuSplit::OwnsScheduler());
    const u32 marks = Access::CpuMarks(*f.cache);

    f.cache->WriteMemory(0x20100, 0x10);
    CHECK(Access::Deferred(*f.cache) == 1);
    CHECK(Access::CpuMarks(*f.cache) == marks);
    CHECK(f.runtime.downloads == 0);
    // The CPU's bytes leave the GPU-written set at once; the rest of the page stays in it.
    CHECK_FALSE(Access::Pending(*f.cache, 0x20100, 0x10));
    CHECK(f.cache->IsRegionGpuModified(0x20000, 0x100));
    CHECK(f.cache->IsRegionGpuModified(0x20110, 0x100));

    // Another non-owner call doesn't apply it either.
    f.cache->DownloadMemory(0x20000, 0x1000);
    CHECK(Access::Deferred(*f.cache) == 1);

    SECTION("a download on the owner applies it first") {
        owner.Do([&] { f.cache->DownloadMemory(0x20000, 0x1000); });
    }
    SECTION("the owner's frame tick applies it") {
        owner.Do([&] { f.cache->TickFrame(); });
    }
    SECTION("an owner write elsewhere on the page applies it first") {
        std::memset(f.bytes + 0x10800, 0xdd, 4);
        owner.Do([&] { f.cache->WriteMemory(0x20800, 4); });
        CHECK(f.bytes[0x10800] == 0xdd);
    }
    SECTION("an owner write on a page without GPU data applies it first") {
        owner.Do([&] { f.cache->WriteMemory(0x40000, 4); });
    }
    CHECK(Access::Deferred(*f.cache) == 0);
    CHECK(f.runtime.downloads >= 1);
    CHECK(Access::CpuMarks(*f.cache) > marks);
    CHECK(f.GuestHasGpuBytes());
    CHECK(f.GuestKeepsCpuBytes());
}

TEST_CASE("An owner partial page write downloads the GPU bytes at once",
          "[buffer_ports][deferred_page_writes]") {
    DeferredWriteFixture f;
    OwnerThread owner;
    const u32 marks = Access::CpuMarks(*f.cache);
    owner.Do([&] { f.cache->WriteMemory(0x20100, 0x10); });
    CHECK(Access::Deferred(*f.cache) == 0);
    CHECK(Access::CpuMarks(*f.cache) == marks + 1);
    CHECK(f.GuestHasGpuBytes());
    CHECK(f.GuestKeepsCpuBytes());
}

TEST_CASE("Deferred page writes keep their order and skip pages already downloaded",
          "[buffer_ports][deferred_page_writes]") {
    DeferredWriteFixture f;
    OwnerThread owner;
    Access::Dirty(*f.cache, 0x23000, 0x1000);
    std::memset(f.bytes + 0x13000, 0x77, 8);
    f.cache->WriteMemory(0x20100, 0x10);
    f.cache->WriteMemory(0x23000, 8);
    // A write on a page without GPU data is applied at once.
    const u32 marks = Access::CpuMarks(*f.cache);
    f.cache->WriteMemory(0x25000, 8);
    CHECK(Access::CpuMarks(*f.cache) == marks + 1);
    CHECK(Access::Deferred(*f.cache) == 2);

    owner.Do([&] { f.cache->DownloadMemory(0x20000, 0x10000); });
    CHECK(Access::Deferred(*f.cache) == 0);
    CHECK(Access::CpuMarks(*f.cache) == marks + 3);
    CHECK(f.GuestHasGpuBytes());
    CHECK(f.GuestKeepsCpuBytes());
    CHECK(f.bytes[0x13000] == 0x77);
    CHECK(f.bytes[0x13007] == 0x77);
    CHECK(f.bytes[0x13008] == 0x12);
    CHECK_FALSE(Access::Pending(*f.cache, 0x20000, 0x10000));
}

TEST_CASE("Unmaps never wait for the owner", "[buffer_ports][deferred_page_writes]") {
    DeferredWriteFixture f;
    OwnerThread owner;
    const u32 marks = Access::CpuMarks(*f.cache);
    f.cache->WriteMemory(0x20000, 0x1000, false);
    CHECK(Access::Deferred(*f.cache) == 0);
    CHECK(Access::CpuMarks(*f.cache) == marks + 1);
    CHECK(f.runtime.downloads == 0);
}

TEST_CASE("The GPU thread mark follows the thread's lifetime",
          "[buffer_ports][deferred_page_writes]") {
    REQUIRE_FALSE(VideoCommon::GpuSplit::HasGpuThread());
    {
        OwnerThread first;
        CHECK(VideoCommon::GpuSplit::HasGpuThread());
        bool owns = false;
        first.Do([&] { owns = VideoCommon::GpuSplit::OwnsScheduler(); });
        CHECK(owns);
        CHECK_FALSE(VideoCommon::GpuSplit::OwnsScheduler());
    }
    CHECK_FALSE(VideoCommon::GpuSplit::HasGpuThread());
}

TEST_CASE("Deferred page writes merge only where they overlap or touch",
          "[buffer_ports][deferred_page_writes]") {
    using Ranges = std::vector<std::pair<DAddr, u64>>;
    DeferredWriteFixture f;
    OwnerThread owner;
    std::memset(f.bytes + 0x10100, 0xee, 0x20);
    std::memset(f.bytes + 0x10400, 0xee, 4);

    for (int i = 0; i < 3; ++i) {
        f.cache->WriteMemory(0x20100, 0x10);
    }
    CHECK(Access::DeferredRanges(*f.cache) == Ranges{{0x20100, 0x10}});
    f.cache->WriteMemory(0x20108, 0x10);
    CHECK(Access::DeferredRanges(*f.cache) == Ranges{{0x20100, 0x18}});
    f.cache->WriteMemory(0x20118, 8);
    CHECK(Access::DeferredRanges(*f.cache) == Ranges{{0x20100, 0x20}});
    f.cache->WriteMemory(0x200f8, 8);
    CHECK(Access::DeferredRanges(*f.cache) == Ranges{{0x200f8, 0x28}});
    // A gap holds GPU-written bytes the guest copy lacks, so a range across it can't merge.
    f.cache->WriteMemory(0x20400, 4);
    CHECK(Access::DeferredRanges(*f.cache) == Ranges{{0x200f8, 0x28}, {0x20400, 4}});

    owner.Do([&] { f.cache->DownloadMemory(0x20000, 0x1000); });
    CHECK(Access::Deferred(*f.cache) == 0);
    CHECK(f.bytes[0x10000] == 0x12);
    CHECK(f.bytes[0x10200] == 0x12);
    CHECK(f.bytes[0x10404] == 0x12);
    CHECK(f.bytes[0x10fff] == 0x12);
    CHECK(std::all_of(f.bytes + 0x10100, f.bytes + 0x10120, [](u8 b) { return b == 0xee; }));
    CHECK(std::all_of(f.bytes + 0x10400, f.bytes + 0x10404, [](u8 b) { return b == 0xee; }));
}

TEST_CASE("Deferred page writes queued from several threads are each applied once",
          "[buffer_ports][deferred_page_writes]") {
    DeferredWriteFixture f;
    OwnerThread owner;
    constexpr int Threads = 4;
    constexpr int WritesPerThread = 64;
    const u32 marks = Access::CpuMarks(*f.cache);
    std::atomic<bool> done{};
    {
        std::vector<std::jthread> writers;
        for (int t = 0; t < Threads; ++t) {
            writers.emplace_back([&, t] {
                for (int i = 0; i < WritesPerThread; ++i) {
                    // 8 bytes every 16: no two writes touch, so none merge.
                    Access::Defer(*f.cache, 0x20000 + (i * Threads + t) * 0x10, 8);
                }
            });
        }
        std::jthread drainer{[&] {
            while (!done.load(std::memory_order_acquire)) {
                owner.Do([&] { Access::ApplyDeferred(*f.cache); });
            }
        }};
        for (auto& writer : writers) {
            writer.join();
        }
        done.store(true, std::memory_order_release);
    }
    owner.Do([&] { Access::ApplyDeferred(*f.cache); });
    CHECK(Access::Deferred(*f.cache) == 0);
    CHECK(Access::CpuMarks(*f.cache) == marks + Threads * WritesPerThread);
}

TEST_CASE("A failure on the owner thread reaches the test and leaves the owner usable",
          "[buffer_ports][deferred_page_writes]") {
    OwnerThread owner;
    CHECK_THROWS_AS(owner.Do([] { throw std::runtime_error{"owner job failed"}; }),
                    std::runtime_error);
    bool ran = false;
    owner.Do([&] { ran = VideoCommon::GpuSplit::OwnsScheduler(); });
    CHECK(ran);
}

TEST_CASE("Deferred CPU store during Finish survives single batch and GC readbacks",
          "[buffer_ports][deferred_writes]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    f.cache->SetDeferredWriteJournal(&journal);
    const bool mapped = GENERATE(false, true);
    const unsigned mode = GENERATE(0u, 1u, 2u);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, mapped);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 7);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.device.ApplyOpOnMappedMemory(0x20008, 1, [&](u8* pointer, PAddr physical) {
            Write write;
            write.physical_address = physical;
            write.device_address = 0x20008;
            write.mapping_generation = 1;
            write.lease_generation = 1;
            write.size = 1;
            const auto result = journal.TryPublish(0, write, [&]() noexcept {
                *pointer = 0;
                return true;
            });
            REQUIRE(result.status == PublishStatus::Published);
        }));
    };
    if (mode == 0) {
        f.cache->DownloadMemory(0x20000, 0x1000);
    } else if (mode == 1) {
        const std::array<std::pair<DAddr, u64>, 1> ranges{{{0x20000, 0x1000}}};
        Access::Download(*f.cache, ranges);
    } else {
        const std::array ids{id};
        Access::DownloadBuffers(*f.cache, ids);
    }
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.bytes[0x10007] == 7);
    REQUIRE(f.bytes[0x10008] == 0);
    REQUIRE(f.bytes[0x10009] == 7);
    REQUIRE(journal.CaptureCut() == 1);
    REQUIRE(journal.AppliedThrough() == 0);
}

TEST_CASE("Async readback preserves CPU history and invalidation by a later GPU write",
          "[buffer_ports][deferred_writes]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    f.cache->SetDeferredWriteJournal(&journal);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 7);
    Access::GpuWrite(*f.cache, id, 0x20000, 0x1000);
    f.cache->CommitAsyncFlushes();
    f.runtime.Finish();
    REQUIRE(f.bytes[0x10007] == 0);

    Write write;
    write.physical_address = 0x20008;
    write.device_address = 0x20008;
    write.mapping_generation = 1;
    write.lease_generation = 1;
    write.size = 1;
    REQUIRE(journal.TryPublish(0, write, [&]() noexcept {
                f.bytes[0x10008] = 0;
                return true;
            }).status == PublishStatus::Published);
    REQUIRE(journal.DrainThrough(1, [&](const Write&) {
                buffer.bytes[8] = 0;
                return true;
            }) == DrainResult::Complete);
    REQUIRE_FALSE(journal.RetireThrough(1));

    buffer.bytes[16] = 9;
    Access::GpuWrite(*f.cache, id, 0x20010, 1);
    f.cache->PopAsyncFlushes();
    REQUIRE(f.bytes[0x10007] == 7);
    REQUIRE(f.bytes[0x10008] == 0);
    REQUIRE(f.bytes[0x10009] == 7);
    REQUIRE(f.bytes[0x10010] == 0);
    REQUIRE(f.cache->IsRegionGpuModified(0x20010, 1));
    REQUIRE_FALSE(f.cache->IsRegionGpuModified(0x20009, 1));
    REQUIRE(journal.ReclaimableThrough() == 1);
    REQUIRE(journal.RetireThrough(1));
    // A fresh readback must finally deliver the newer GPU byte.
    f.cache->DownloadMemory(0x20010, 1);
    REQUIRE(f.bytes[0x10010] == 9);
}

TEST_CASE("Normalized SEM_EXECUTE acquires suspend the channel until their condition is met",
          "[buffer_ports][deferred_writes][semaphore_acquire]") {
    Fixture f;
    const u32 operation = GENERATE(0u, 2u, 3u, 4u, 5u);
    const u32 unmet = operation == 4 ? 0x20 : operation == 5 ? 0x10 : 0;
    const u32 met = operation == 5 ? 0 : 0x10;
    Tegra::Control::ChannelState channel{0};
    Tegra::DmaPusher pusher{f.system, *f.memory, channel};
    pusher.BindRasterizer(&f.rasterizer);
    auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
    SCOPE_EXIT { waiters.Remove(&pusher.puller); };
    std::memcpy(f.bytes, &unmet, sizeof(unmet));
    Tegra::Engines::PullerTestAccess::StartAcquire(pusher, operation, 0x10);
    REQUIRE(pusher.puller.HasPendingAcquire());
    REQUIRE(pusher.suspensions.load() == 1);
    REQUIRE_FALSE(pusher.puller.CanResumeAcquire(pusher));
    REQUIRE(pusher.suspensions.load() == 1);
    std::memcpy(f.bytes, &met, sizeof(met));
    REQUIRE(pusher.puller.CanResumeAcquire(pusher));
    // A poll cannot consume the acquire or its cut; ResumeAcquire does that on the GPU thread.
    REQUIRE(pusher.puller.HasPendingAcquire());
    REQUIRE(pusher.suspensions.load() == 1);
}

namespace {
VideoCommon::DeferredWrites::Publication PublishListByte(
    VideoCommon::DeferredWrites::Journal& journal, Fixture& fixture, DAddr device, u8 value) {
    using namespace VideoCommon::DeferredWrites;
    Publication result{PublishStatus::InvalidWrite};
    fixture.device.ApplyOpOnMappedMemory(device, 1, [&](u8* pointer, PAddr physical) {
        Write write;
        write.physical_address = physical;
        write.device_address = device;
        write.mapping_generation = 1;
        write.lease_generation = 1;
        write.size = 1;
        write.bytes[0] = value;
        result = journal.TryPublish(0, write, [&]() noexcept {
            *pointer = value;
            return true;
        });
    });
    return result;
}
} // namespace

TEST_CASE("Command list copies reject CPU publications before application and after history retirement",
          "[buffer_ports][deferred_writes][deferred_list_copy]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 8};
    Fixture f;
    const bool alias = GENERATE(false, true);
    f.cache->SetDeferredWriteJournal(&journal);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x42);
    Access::QueueListCopy(*f.cache, id, 0x20000);
    REQUIRE(f.cache->Nx2CopyListPages(1));
    f.runtime.Finish();
    std::array<u8, 4> out{};
    using Read = Cache::Nx2ListRead;
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Read::Copied);
    REQUIRE(out[0] == 0x42);
    // An unrelated physical page does not invalidate this copy.
    REQUIRE(PublishListByte(journal, f, 0x30008, 5).status == PublishStatus::Published);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Read::Copied);
    if (alias) f.device.Map(0x70000, 0x20000, 0x1000, f.asid);
    SCOPE_EXIT { if (alias) f.device.Unmap(0x70000, 0x1000); };
    REQUIRE(PublishListByte(journal, f, alias ? 0x70008 : 0x20008, 9).status == PublishStatus::Published);
    out.fill(0xaa);
    REQUIRE(journal.AppliedThrough() == 0);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Read::Unknown);
    REQUIRE(std::ranges::all_of(out, [](u8 byte) { return byte == 0xaa; }));
    REQUIRE(journal.DrainThrough(journal.CaptureCut(), [&](const Write& write) {
        if (write.physical_address == 0x20008) Access::Get(*f.cache, id).bytes[8] = write.bytes[0];
        return true;
    }) == DrainResult::Complete);
    REQUIRE(journal.RetireThrough(journal.CaptureCut()));
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Read::Unknown);
    Access::QueueListCopy(*f.cache, id, 0x20000);
    REQUIRE(f.cache->Nx2CopyListPages(2));
    f.runtime.Finish();
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Read::Copied);
    REQUIRE(out[0] == 9);
    REQUIRE(out[1] == 0x42);
}

TEST_CASE("Command list wait rechecks CPU publications before exposing staging bytes",
          "[buffer_ports][deferred_writes][deferred_list_copy]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    f.cache->SetDeferredWriteJournal(&journal);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x42);
    Access::QueueListCopy(*f.cache, id, 0x20000);
    REQUIRE(f.cache->Nx2CopyListPages(1));
    f.runtime.list_copy_ready = false;
    Publication publication{PublishStatus::Rejected};
    f.runtime.before_list_wait = [&] {
        std::jthread writer{[&] { publication = PublishListByte(journal, f, 0x20008, 9); }};
        writer.join();
    };
    std::array<u8, 4> out{0xaa, 0xaa, 0xaa, 0xaa};
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Cache::Nx2ListRead::Unknown);
    REQUIRE(publication.status == PublishStatus::Published);
    REQUIRE(journal.AppliedThrough() == 0);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(std::ranges::all_of(out, [](u8 byte) { return byte == 0xaa; }));
}

TEST_CASE("Command list synced status is invalidated by CPU publication without a GPU write",
          "[buffer_ports][deferred_writes][deferred_list_copy]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    f.cache->SetDeferredWriteJournal(&journal);
    const bool draw_written = GENERATE(false, true);
    if (draw_written) Access::DrawWrite(*f.cache, 0x20000);
    f.cache->Nx2NoteListSynced(0x20000);
    std::array<u8, 4> out{};
    using Read = Cache::Nx2ListRead;
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Read::Synced);
    REQUIRE(PublishListByte(journal, f, 0x20008, 9).status == PublishStatus::Published);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Read::Unknown);
    REQUIRE(journal.DrainThrough(1, [](const Write&) { return true; }) == DrainResult::Complete);
    REQUIRE(journal.RetireThrough(1));
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Read::Unknown);
    f.cache->Nx2NoteListSynced(0x20000);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Read::Synced);
}

TEST_CASE("Command page batch cannot validate synced status over a concurrent CPU publication",
          "[buffer_ports][deferred_writes][deferred_list_copy]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    f.cache->SetDeferredWriteJournal(&journal);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, true);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x42);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::DrawWrite(*f.cache, 0x20000);
    f.runtime.before_completion = [&] {
        REQUIRE(PublishListByte(journal, f, 0x20008, 9).status == PublishStatus::Published);
    };
    const std::array<DAddr, 1> pages{0x20000};
    f.cache->Nx2FlushListPages(pages);
    std::array<u8, 4> out{};
    REQUIRE(f.bytes[0x10008] == 9);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Cache::Nx2ListRead::Unknown);
    REQUIRE(journal.AppliedThrough() == 0);
    f.runtime.before_completion = {};
}

TEST_CASE("Command list copy checks the current physical page identity",
          "[buffer_ports][deferred_writes][deferred_list_copy]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    f.cache->SetDeferredWriteJournal(&journal);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x42);
    Access::QueueListCopy(*f.cache, id, 0x20000);
    REQUIRE(f.cache->Nx2CopyListPages(1));
    f.runtime.Finish();
    std::array<u8, 4> out{};
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Cache::Nx2ListRead::Copied);
    f.device.Unmap(0x20000, 0x1000);
    f.device.Map(0x20000, 0x30000, 0x1000, f.asid);
    out.fill(0xaa);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20008, out, true) == Cache::Nx2ListRead::Unknown);
    REQUIRE(std::ranges::all_of(out, [](u8 byte) { return byte == 0xaa; }));
}

namespace {
VideoCommon::DeferredWrites::Publication PublishConsumerByte(
    VideoCommon::DeferredWrites::Journal& journal, Fixture& fixture,
    const VideoCommon::DeferredWrites::Journal::Lease& lease, u8 value) {
    using namespace VideoCommon::DeferredWrites;
    Publication result{PublishStatus::InvalidWrite};
    fixture.device.ApplyOpOnMappedMemory(lease.device_page + 8, 1, [&](u8* pointer, PAddr physical) {
        Write write;
        write.physical_address = physical;
        write.device_address = lease.device_page + 8;
        write.mapping_generation = lease.mapping_generation;
        write.lease_generation = lease.generation;
        write.size = 1;
        write.bytes[0] = value;
        result = journal.TryPublishLeased(0, write, [&]() noexcept {
            *pointer = value;
            return true;
        });
    });
    return result;
}

void ConnectConsumerJournal(Fixture& fixture, VideoCommon::DeferredWrites::Journal& journal) {
    fixture.cache->SetDeferredWriteJournal(&journal, [&](VideoCommon::DeferredWrites::Cut cut) {
        const auto result = journal.DrainThrough(cut, [&](const VideoCommon::DeferredWrites::Write& write) {
            Access::ApplyExact(*fixture.cache, write);
            return true;
        });
        REQUIRE(result == VideoCommon::DeferredWrites::DrainResult::Complete);
    });
}
} // namespace

TEST_CASE("Deferred sink applies recorded payload without rereading or rewriting newer CPU bytes",
          "[buffer_ports][deferred_writes][deferred_sink]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 8};
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x42);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    const auto lease = journal.GrantLease(0x20000, 0x20000, 1);
    REQUIRE(lease);
    REQUIRE(PublishConsumerByte(journal, f, *lease, 7).sequence == 1);
    REQUIRE(PublishConsumerByte(journal, f, *lease, 9).sequence == 2);
    const auto drain = [&](Cut cut) {
        return journal.DrainThrough(cut, [&](const Write& write) {
            return f.cache->ApplyDeferredWrite(write);
        });
    };
    REQUIRE(drain(1) == DrainResult::Complete);
    CHECK(buffer.bytes[8] == 7);
    CHECK(f.bytes[0x10008] == 9);
    CHECK(buffer.bytes[7] == 0x42);
    CHECK(buffer.bytes[9] == 0x42);
    CHECK(f.runtime.upload_bytes == 1);
    CHECK(f.runtime.finishes == 0);
    CHECK_FALSE(Access::CpuDirty(*f.cache, 0x20000, 0x1000));
    CHECK_FALSE(Access::Pending(*f.cache, 0x20008, 1));
    CHECK(Access::Pending(*f.cache, 0x20007, 1));
    REQUIRE(drain(2) == DrainResult::Complete);
    CHECK(buffer.bytes[8] == 9);
    CHECK(f.runtime.upload_bytes == 2);
    CHECK(journal.RetireThrough(2));
}

TEST_CASE("Deferred sink protects a newer CPU commit from an already copied async readback",
          "[buffer_ports][deferred_writes][deferred_sink]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 8};
    Fixture f;
    ConnectConsumerJournal(f, journal);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x42);
    Access::GpuWrite(*f.cache, id, 0x20000, 0x1000);
    f.cache->CommitAsyncFlushes();
    f.runtime.Finish();
    REQUIRE(journal.HasReadbacks());
    const auto lease = journal.GrantLease(0x20000, 0x20000, 1);
    REQUIRE(lease);
    REQUIRE(PublishConsumerByte(journal, f, *lease, 7).sequence == 1);
    REQUIRE(PublishConsumerByte(journal, f, *lease, 9).sequence == 2);
    const auto drain = [&](Cut cut) {
        return journal.DrainThrough(cut, [&](const Write& write) {
            return f.cache->ApplyDeferredWrite(write);
        });
    };
    REQUIRE(drain(1) == DrainResult::Complete);
    CHECK_FALSE(journal.RetireThrough(1));
    f.cache->PopAsyncFlushes();
    CHECK(f.bytes[0x10007] == 0x42);
    CHECK(f.bytes[0x10008] == 9);
    CHECK(f.bytes[0x10009] == 0x42);
    CHECK(buffer.bytes[8] == 7);
    REQUIRE(drain(2) == DrainResult::Complete);
    CHECK(buffer.bytes[8] == 9);
    CHECK_FALSE(Access::CpuDirty(*f.cache, 0x20000, 0x1000));
    CHECK(f.runtime.upload_bytes == 2);
    CHECK(f.runtime.finishes == 1);
}

TEST_CASE("Deferred zero payload uploads only its exact bytes and does not require remaining GPU dirt",
          "[buffer_ports][deferred_writes][deferred_sink]") {
    using namespace VideoCommon::DeferredWrites;
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x42);
    std::memset(f.bytes + 0x10000, 0x33, 0x1000);
    Write write;
    write.physical_address = 0x20020;
    write.device_address = 0x20020;
    write.mapping_generation = 1;
    write.lease_generation = 1;
    write.payload = Payload::Zero;
    write.size = 128;
    REQUIRE_FALSE(f.cache->IsRegionGpuModified(0x20000, 0x1000));
    REQUIRE(f.cache->ApplyDeferredWrite(write));
    CHECK(buffer.bytes[31] == 0x42);
    CHECK(buffer.bytes[160] == 0x42);
    CHECK(std::all_of(buffer.bytes.begin() + 32, buffer.bytes.begin() + 160,
                      [](u8 byte) { return byte == 0; }));
    CHECK(f.bytes[0x10020] == 0x33);
    CHECK(f.runtime.upload_bytes == 128);
    CHECK(f.runtime.finishes == 0);
    CHECK_FALSE(Access::CpuDirty(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Deferred sink rejects invalid identities and dirty pages before changing buffer state",
          "[buffer_ports][deferred_writes][deferred_sink]") {
    using namespace VideoCommon::DeferredWrites;
    Fixture f;
    Write write;
    write.physical_address = 0x20008;
    write.device_address = 0x20008;
    write.mapping_generation = 1;
    write.lease_generation = 1;
    write.size = 1;
    write.bytes[0] = 7;
    CHECK_FALSE(f.cache->ApplyDeferredWrite(write));
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x42);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    SECTION("Changed physical identity") {
        write.physical_address = 0x30008;
    }
    SECTION("CPU dirty page cannot trigger a whole-page upload") {
        Access::MarkCpuDirty(*f.cache, 0x20000, 0x1000);
    }
    SECTION("Store crosses its device page") {
        write.device_address = 0x20fff;
        write.physical_address = 0x20fff;
        write.size = 2;
    }
    SECTION("Payload exceeds its fixed record") {
        write.size = 33;
    }
    SECTION("Unknown payload") {
        write.payload = static_cast<Payload>(2);
    }
    CHECK_FALSE(f.cache->ApplyDeferredWrite(write));
    CHECK(f.runtime.uploads == 0);
    CHECK(f.runtime.upload_bytes == 0);
    CHECK(f.runtime.finishes == 0);
    CHECK(buffer.bytes[8] == 0x42);
    CHECK(Access::Pending(*f.cache, 0x20008, 1));
}

TEST_CASE("DMA reservation spans nested joins and the scoped guest-memory writeback",
          "[buffer_ports][deferred_writes][deferred_consumer]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 8};
    Fixture f;
    const u64 size = GENERATE(u64{16}, u64{0x10000});
    f.memory->Map(0x200000, 0x20000, 0x30000, Tegra::PTEKind::PITCH, true);
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    Access::AttachChannel(*f.cache, channel);
    ConnectConsumerJournal(f, journal);
    Access::Add(*f.cache, 0x10000, 0x10000);
    const auto source_id = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto destination_id = Access::Add(*f.cache, 0x30000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, source_id).bytes, 0x42);
    std::ranges::fill(Access::Get(*f.cache, destination_id).bytes, 0x24);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    const auto source = journal.GrantLease(0x20000, 0x20000, 1);
    const auto destination = journal.GrantLease(0x30000, 0x30000, 1);
    const auto unrelated = journal.GrantLease(0x10000, 0x10000, 1);
    REQUIRE(source);
    REQUIRE(destination);
    REQUIRE(unrelated);
    auto old_readback = journal.BeginReadback();
    REQUIRE(PublishConsumerByte(journal, f, *source, 7).sequence == 1);
    REQUIRE(PublishConsumerByte(journal, f, *destination, 3).sequence == 2);
    unsigned copies{};
    bool posted_unrelated{};
    f.runtime.before_gpu_copy = [&](Buffer&, Buffer&) {
        REQUIRE(journal.AppliedThrough() >= 2);
        Publication source_race{PublishStatus::Rejected}, destination_race{PublishStatus::Rejected};
        Publication outside{PublishStatus::Rejected};
        std::jthread writer{[&] {
            source_race = PublishConsumerByte(journal, f, *source, 9);
            destination_race = PublishConsumerByte(journal, f, *destination, 9);
            if (!posted_unrelated) outside = PublishConsumerByte(journal, f, *unrelated, 5);
        }};
        writer.join();
        REQUIRE(source_race.status == PublishStatus::LeaseUnavailable);
        REQUIRE(destination_race.status == PublishStatus::LeaseUnavailable);
        REQUIRE_FALSE(journal.GrantLease(0x20000, 0x20000, 2));
        REQUIRE_FALSE(journal.GrantLease(0x30000, 0x30000, 2));
        if (!posted_unrelated) {
            REQUIRE(outside.status == PublishStatus::Published);
            posted_unrelated = true;
        }
        ++copies;
    };
    REQUIRE(f.cache->DMACopy(0x200008, 0x210008, size));
    REQUIRE(copies >= 1);
    if (size > 16) REQUIRE(copies > 1); // FindBuffer joined, then DeleteBuffer nested in the scope.
    REQUIRE(f.bytes[0x20008] == 7); // DeviceGuestMemoryScoped's destructor has run.
    const auto final_id = Access::Find(*f.cache, 0x30008, 1);
    const auto& final_buffer = Access::Get(*f.cache, final_id);
    REQUIRE(final_buffer.bytes[final_buffer.Offset(0x30008)] == 7);
    REQUIRE_FALSE(journal.RetireThrough(2)); // Consumer completion preserves live readback history.
    REQUIRE(journal.GrantLease(0x20000, 0x20000, 2));
    REQUIRE(journal.GrantLease(0x30000, 0x30000, 2));
    f.runtime.before_gpu_copy = {};
}

TEST_CASE("DMA clear applies accepted bytes before clearing and excludes a concurrent publisher",
          "[buffer_ports][deferred_writes][deferred_consumer]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    f.memory->Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    Access::AttachChannel(*f.cache, channel);
    ConnectConsumerJournal(f, journal);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto lease = journal.GrantLease(0x20000, 0x20000, 1);
    REQUIRE(lease);
    REQUIRE(PublishConsumerByte(journal, f, *lease, 7).sequence == 1);
    unsigned clears{};
    f.runtime.before_clear = [&](Buffer& buffer) {
        REQUIRE(journal.AppliedThrough() == 1);
        REQUIRE(buffer.bytes[8] == 7);
        Publication raced{PublishStatus::Rejected};
        std::jthread writer{[&] { raced = PublishConsumerByte(journal, f, *lease, 9); }};
        writer.join();
        REQUIRE(raced.status == PublishStatus::LeaseUnavailable);
        REQUIRE_FALSE(journal.GrantLease(0x20000, 0x20000, 2));
        ++clears;
    };
    REQUIRE(f.cache->DMAClear(0x200000, 4, 0x11111111));
    REQUIRE(clears == 1);
    REQUIRE(Access::Get(*f.cache, id).bytes[8] == 0x11);
    REQUIRE(f.bytes[0x10008] == 7); // Clear is GPU-side, not a new CPU store.
    REQUIRE(journal.GrantLease(0x20000, 0x20000, 2));
    f.runtime.before_clear = {};
}

TEST_CASE("Joining buffers closes their full ranges before copying and keeps deletion nested",
          "[buffer_ports][deferred_writes][deferred_consumer]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    Access::AttachChannel(*f.cache, channel);
    ConnectConsumerJournal(f, journal);
    const auto left_id = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto right_id = Access::Add(*f.cache, 0x30000, 0x10000);
    const auto left = journal.GrantLease(0x20000, 0x20000, 1);
    const auto right = journal.GrantLease(0x30000, 0x30000, 1);
    REQUIRE(left);
    REQUIRE(right);
    auto old_readback = journal.BeginReadback();
    REQUIRE(PublishConsumerByte(journal, f, *left, 7).sequence == 1);
    REQUIRE(PublishConsumerByte(journal, f, *right, 8).sequence == 2);
    unsigned copies{}, deletions{};
    f.runtime.before_gpu_copy = [&](Buffer&, Buffer&) {
        REQUIRE(journal.AppliedThrough() == 2);
        Publication raced{PublishStatus::Rejected};
        std::jthread writer{[&] { raced = PublishConsumerByte(journal, f, *right, 9); }};
        writer.join();
        REQUIRE(raced.status == PublishStatus::LeaseUnavailable);
        REQUIRE_FALSE(journal.GrantLease(0x30000, 0x30000, 2));
        ++copies;
    };
    f.runtime.before_delete = [&](Buffer& buffer) {
        REQUIRE_FALSE(journal.GrantLease(buffer.CpuAddr(), buffer.CpuAddr(), 2));
        ++deletions;
    };
    const auto joined_id = Access::Find(*f.cache, 0x20008, 0x10008);
    REQUIRE(copies == 2);
    REQUIRE(deletions == 2);
    REQUIRE_FALSE(Access::Contains(*f.cache, left_id));
    REQUIRE_FALSE(Access::Contains(*f.cache, right_id));
    const auto& joined = Access::Get(*f.cache, joined_id);
    REQUIRE(joined.bytes[8] == 7);
    REQUIRE(joined.bytes[0x10008] == 8);
    REQUIRE_FALSE(journal.RetireThrough(2));
    REQUIRE(journal.GrantLease(0x20000, 0x20000, 2));
    REQUIRE(journal.GrantLease(0x30000, 0x30000, 2));
    f.runtime.before_gpu_copy = {};
    f.runtime.before_delete = {};
}

TEST_CASE("Standalone buffer deletion drains its lease before invalidation without retiring history",
          "[buffer_ports][deferred_writes][deferred_consumer]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    Access::AttachChannel(*f.cache, channel);
    ConnectConsumerJournal(f, journal);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto lease = journal.GrantLease(0x20000, 0x20000, 1);
    REQUIRE(lease);
    auto old_readback = journal.BeginReadback();
    REQUIRE(PublishConsumerByte(journal, f, *lease, 7).sequence == 1);
    bool observed{};
    f.runtime.before_delete = [&](Buffer& buffer) {
        REQUIRE(journal.AppliedThrough() == 1);
        REQUIRE(buffer.bytes[8] == 7);
        Publication raced{PublishStatus::Rejected};
        std::jthread writer{[&] { raced = PublishConsumerByte(journal, f, *lease, 9); }};
        writer.join();
        REQUIRE(raced.status == PublishStatus::LeaseUnavailable);
        REQUIRE_FALSE(journal.GrantLease(0x20000, 0x20000, 2));
        observed = true;
    };
    Access::Delete(*f.cache, id);
    REQUIRE(observed);
    REQUIRE_FALSE(Access::Contains(*f.cache, id));
    REQUIRE_FALSE(journal.RetireThrough(1));
    REQUIRE(journal.GrantLease(0x20000, 0x20000, 2));
    f.runtime.before_delete = {};
}

TEST_CASE("A real pending async readback survives a buffer join without overwriting newer CPU or GPU bytes",
          "[buffer_ports][deferred_writes][deferred_consumer]") {
    using namespace VideoCommon::DeferredWrites;
    Journal journal{1, 4};
    Fixture f;
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    Access::AttachChannel(*f.cache, channel);
    ConnectConsumerJournal(f, journal);
    const auto old_id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, old_id).bytes, 7);
    Access::GpuWrite(*f.cache, old_id, 0x20000, 0x1000);
    f.cache->CommitAsyncFlushes();
    f.runtime.Finish(); // Copy completed, but its staging-to-backing readback is still pending.
    REQUIRE(f.bytes[0x10007] == 0);
    const auto lease = journal.GrantLease(0x20000, 0x20000, 1);
    REQUIRE(lease);
    REQUIRE(PublishConsumerByte(journal, f, *lease, 9).sequence == 1);
    const auto joined_id = Access::Find(*f.cache, 0x20008, 0x10008);
    REQUIRE_FALSE(Access::Contains(*f.cache, old_id));
    REQUIRE(journal.AppliedThrough() == 1);
    REQUIRE_FALSE(journal.RetireThrough(1));
    auto& joined = Access::Get(*f.cache, joined_id);
    REQUIRE(joined.bytes[8] == 9);
    joined.bytes[16] = 11;
    Access::GpuWrite(*f.cache, joined_id, 0x20010, 1);
    f.cache->PopAsyncFlushes();
    REQUIRE(f.bytes[0x10007] == 7);
    REQUIRE(f.bytes[0x10008] == 9);
    REQUIRE(f.bytes[0x10009] == 7);
    REQUIRE(f.bytes[0x10010] == 0);
    REQUIRE(f.cache->IsRegionGpuModified(0x20010, 1));
    REQUIRE(journal.RetireThrough(1));
}
