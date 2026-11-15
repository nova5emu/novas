// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>
#include "common/cpu_epoch_memory_barrier.h"
#include "common/range_sets.h"
#include "video_core/cpu_epoch_barrier.h"
#include "video_core/deferred_write_journal.h"

namespace Core::Memory {
class Memory;
}

namespace VideoCommon::CpuEpochs {

enum class Exclusion : u8 {
    Uniform = 1,
    Index = 2,
    Texture = 4,
    CapturedWriter = 8,
    OtherConsumer = 16,
    Mapping = 32,
};
enum class CloseReason { Consumer, Mapping, Protection, CapturedWriter, Shutdown };
enum class CloseResult { Complete, Pending, Failed };
enum class Rejection { Runtime, Candidate, Admission, Materialize, Grant, Count };

// Per-thread cache of completed permanent vetoes. Mapping changes preserve device vetoes.
class DeviceExclusionCache {
public:
    bool Contains(u64 instance, const void* owner, DAddr address, u64 size,
                  Exclusion reason) const {
        const auto& entry = entries[(address >> 12) % entries.size()];
        return size && size - 1 <= ~u64{0} - address && entry.instance == instance &&
               entry.owner == owner && entry.first == (address >> 12) &&
               entry.last == ((address + size - 1) >> 12) &&
               (entry.reasons & static_cast<u8>(reason)) == static_cast<u8>(reason);
    }
    void Remember(u64 instance, const void* owner, DAddr address, u64 size, Exclusion reason) {
        if (!size || size - 1 > ~u64{0} - address)
            return;
        auto& entry = entries[(address >> 12) % entries.size()];
        const auto first = address >> 12, last = (address + size - 1) >> 12;
        const u8 previous = entry.instance == instance && entry.owner == owner &&
                                    entry.first == first && entry.last == last
                                ? entry.reasons
                                : 0;
        entry = {instance, owner, first, last, static_cast<u8>(previous | static_cast<u8>(reason))};
    }

private:
    struct Entry {
        u64 instance{};
        const void* owner{};
        u64 first{}, last{};
        u8 reasons{};
    };
    std::array<Entry, 64> entries{};
};

struct Identity {
    Core::Memory::Memory* memory{};
    const void* cpu_owner{};
    const void* device_owner{};
    VAddr address{};
    PAddr physical{};
    DAddr device{};
    u64 generation{};
    std::shared_ptr<void> lifetime;
    std::shared_ptr<DeferredWrites::Journal::WriterPermission> writer;
};

class Coordinator {
public:
    static constexpr u64 PageSize = 4096;
    struct Backend {
        std::function<bool(const Identity&)> validate;
        std::function<bool(const Identity&)> reprotect;
        std::function<CloseResult(const Identity&)> drain;
        std::function<bool(const Identity&, std::span<u8>)> capture;
        std::function<bool(const Identity&, std::span<const u8>)> upload;
        std::function<bool(const Identity&)> validate_opening{};
    };
    using Dispatcher = std::function<CloseResult(CloseReason)>;
    struct Page {
        Identity identity;
        PageBarrier barrier;
        Epoch epoch;
        u64 close_started_ns{};
        std::array<u8, PageSize> snapshot{}, opening_snapshot{};
        bool protected_stores{}, captured{}, baseline_captured{}, was_open{};
    };
    class Mutation {
    public:
        Mutation() = default;
        ~Mutation();
        Mutation(Mutation&& other) noexcept;
        Mutation& operator=(Mutation&& other) noexcept;
        Mutation(const Mutation&) = delete;
        Mutation& operator=(const Mutation&) = delete;

    private:
        friend class Coordinator;
        explicit Mutation(Coordinator* owner_) : owner{owner_} {}
        void Release();
        Coordinator* owner{};
    };

    Coordinator(u64 physical_size, bool requested,
                Common::CpuEpochMemoryBarrier::SyscallFn syscall = nullptr,
                void* context = nullptr);
    ~Coordinator();
    u64 InstanceId() const {
        return instance_id;
    }
    struct ExclusionStats {
        std::array<u64, 6> pages{}, calls{}, bytes{};
        u64 invalid_physical{}, invalid_device{};
    };
    ExclusionStats GetExclusionStats() const;
    bool Requested() const {
        return requested;
    }
    bool Available() const {
        return requested && memory_barrier.Available();
    }
    const Common::CpuEpochMemoryBarrier& Platform() const {
        return memory_barrier;
    }
    void Bind(Backend backend, Dispatcher dispatcher, std::function<void()> progress = {});
    void Unbind();
    Mutation BeforeMutation(CloseReason reason);
    void Exclude(PAddr physical, u64 size, Exclusion reason);
    void ExcludeDevice(const void* owner, DAddr address, u64 size, Exclusion reason);
    // Mapping mutation is already admitted; no active pages or GPU waits under mapping_guard.
    void ExcludeMappedDevice(const void* owner, DAddr address, PAddr physical, u64 size);
    u8 Exclusions(PAddr physical) const;
    std::optional<u8> TryExclusions(PAddr physical) const;
    std::shared_ptr<Page> BeginOpening(Identity identity);
    bool PrepareAndOpen(Identity identity, const std::function<bool()>& materialize,
                        const std::function<bool()>& grant_write, u64& downloaded_bytes,
                        const std::function<bool(Identity&)>& acquire_writer = {});
    void NoteRejected(Rejection reason = Rejection::Admission);
    bool CompleteOpening(const std::shared_ptr<Page>& page,
                         const std::function<bool()>& grant_write);
    std::shared_ptr<Page> FindPage(PAddr physical) const;
    PageBarrier::WriteAttempt TryEnterWriter(const std::shared_ptr<Page>& page, Epoch epoch);
    CloseResult CloseAllOnOwner(CloseReason reason);
    size_t ActivePages() const;
    void Report();
    // Serializes ordinary permission changes with opening/closing admission.
    void WithAliasProtection(const void* owner, VAddr address, u64 size,
                             const std::function<void(bool allow_native_access)>& operation);

private:
    CloseResult ClosePage(const std::shared_ptr<Page>& page);
    const bool requested;
    const u64 instance_id;
    Common::CpuEpochMemoryBarrier memory_barrier;
    mutable std::mutex mutex;
    std::vector<u8> exclusions;
    std::map<PAddr, std::shared_ptr<Page>> pages;
    std::atomic<size_t> active_pages{};
    ExclusionStats exclusion_stats;
    std::map<const void*, std::array<Common::RangeSet<DAddr>, 6>> device_exclusions;
    Backend backend;
    Dispatcher dispatcher;
    std::function<void()> progress;
    size_t mutations{};
    bool exclusions_complete{true};
    std::array<u64, 5> close_reasons{};
    std::array<u64, static_cast<size_t>(Rejection::Count)> rejection_reasons{};
    u64 estimated_traps_avoided{}, estimated_changed_blocks{}, opened{}, closed{}, excluded_pages{},
        failures{}, rejected{}, downloaded_bytes{}, uploaded_bytes{}, close_ns{};
    std::atomic<u64> last_report{};
};

} // namespace VideoCommon::CpuEpochs
