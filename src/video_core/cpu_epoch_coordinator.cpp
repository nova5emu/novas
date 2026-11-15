// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>
#include "common/logging.h"
#include "common/range_sets.inc"
#include "common/scope_exit.h"
#include "video_core/cpu_epoch_coordinator.h"

namespace VideoCommon::CpuEpochs {
namespace {
std::atomic<u64> next_instance_id{};
thread_local Coordinator* closing_owner{};
thread_local Coordinator* draining_owner{};
u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

Coordinator::Coordinator(u64 physical_size, bool requested_,
                         Common::CpuEpochMemoryBarrier::SyscallFn syscall, void* context)
    : requested{requested_},
      instance_id{next_instance_id.fetch_add(1, std::memory_order_relaxed) + 1},
      memory_barrier{syscall, context}, exclusions(requested ? physical_size / PageSize : 0),
      last_report{NowNs()} {
    if (requested) {
        memory_barrier.Initialize();
        LOG_INFO(HW_GPU,
                 "CPU_EPOCH_PLATFORM available={} query_mask={:#x} "
                 "failure={} errno={}",
                 Available(), memory_barrier.QueryMask(),
                 Common::CpuEpochMemoryBarrier::FailureName(memory_barrier.LastFailure()),
                 memory_barrier.LastError());
    }
}
Coordinator::~Coordinator() = default;
Coordinator::ExclusionStats Coordinator::GetExclusionStats() const {
    std::scoped_lock lock{mutex};
    return exclusion_stats;
}
Coordinator::Mutation::~Mutation() {
    Release();
}
Coordinator::Mutation::Mutation(Mutation&& other) noexcept
    : owner{std::exchange(other.owner, nullptr)} {}
Coordinator::Mutation& Coordinator::Mutation::operator=(Mutation&& other) noexcept {
    if (this != &other) {
        Release();
        owner = std::exchange(other.owner, nullptr);
    }
    return *this;
}
void Coordinator::Mutation::Release() {
    if (auto* coordinator = std::exchange(owner, nullptr)) {
        std::scoped_lock lock{coordinator->mutex};
        --coordinator->mutations;
    }
}
void Coordinator::Bind(Backend backend_, Dispatcher dispatcher_, std::function<void()> progress_) {
    std::scoped_lock lock{mutex};
    if (!pages.empty())
        throw std::logic_error("CPU epoch backend still has live pages");
    backend = std::move(backend_);
    dispatcher = std::move(dispatcher_);
    progress = std::move(progress_);
}
void Coordinator::Unbind() {
    std::scoped_lock lock{mutex};
    if (!pages.empty())
        throw std::logic_error("CPU epoch shutdown requires closed pages");
    backend = {};
    dispatcher = {};
    progress = {};
}
Coordinator::Mutation Coordinator::BeforeMutation(CloseReason reason) {
    if (!requested)
        return {};
    Dispatcher dispatch;
    bool needs_close;
    {
        std::scoped_lock lock{mutex};
        ++mutations;
        needs_close = !pages.empty();
        dispatch = dispatcher;
    }
    Mutation guard{this};
    if (needs_close && closing_owner == this && draining_owner == this &&
        reason == CloseReason::CapturedWriter)
        return guard;
    if (needs_close && closing_owner == this) {
        throw std::runtime_error("Memory mutation reentered CPU epoch closure");
    }
    if (needs_close && (!dispatch || dispatch(reason) != CloseResult::Complete)) {
        throw std::runtime_error("CPU epoch closure failed before memory mutation");
    }
    return guard;
}
void Coordinator::Exclude(PAddr physical, u64 size, Exclusion reason) {
    if (!requested || !size)
        return;
    static thread_local DeviceExclusionCache excluded;
    if (excluded.Contains(instance_id, this, physical, size, reason))
        return;
    bool overlap = false;
    bool valid = false;
    {
        std::scoped_lock lock{mutex};
        for (size_t bit = 0; bit < exclusion_stats.calls.size(); ++bit) {
            if (static_cast<u8>(reason) & (1U << bit)) {
                ++exclusion_stats.calls[bit];
                exclusion_stats.bytes[bit] += size;
            }
        }
        if (size - 1 > std::numeric_limits<u64>::max() - physical ||
            ((physical + size - 1) / PageSize) >= exclusions.size()) {
            exclusions_complete = false;
            ++exclusion_stats.invalid_physical;
            overlap = !pages.empty();
        } else {
            valid = true;
            const auto last = (physical + size - 1) / PageSize;
            for (u64 page = physical / PageSize; page <= last; ++page) {
                if (!exclusions[page])
                    ++excluded_pages;
                const auto added = static_cast<u8>(reason) & ~exclusions[page];
                for (size_t bit = 0; bit < exclusion_stats.pages.size(); ++bit) {
                    exclusion_stats.pages[bit] += (added >> bit) & 1U;
                }
                exclusions[page] |= static_cast<u8>(reason);
                overlap |= pages.contains(page * PageSize);
            }
        }
    }
    if (overlap) {
        auto mutation =
            BeforeMutation(reason == Exclusion::CapturedWriter ? CloseReason::CapturedWriter
                                                               : CloseReason::Consumer);
    }
    if (valid)
        excluded.Remember(instance_id, this, physical, size, reason);
}
void Coordinator::ExcludeDevice(const void* owner, DAddr address, u64 size, Exclusion reason) {
    if (!requested || !size) {
        return;
    }
    bool overlap{};
    {
        std::scoped_lock lock{mutex};
        if (!owner || size > std::numeric_limits<u64>::max() - address) {
            exclusions_complete = false;
            ++exclusion_stats.invalid_device;
            overlap = !pages.empty();
        } else {
            auto& ranges = device_exclusions[owner];
            for (size_t bit = 0; bit < ranges.size(); ++bit) {
                if (static_cast<u8>(reason) & (1U << bit))
                    ranges[bit].Add(address, size);
            }
            for (const auto& [physical, page] : pages) {
                if (page->identity.device_owner == owner &&
                    (page->identity.device <= address ? address - page->identity.device < PageSize
                                                      : page->identity.device - address < size)) {
                    overlap = true;
                    break;
                }
            }
        }
    }
    if (overlap) {
        auto mutation =
            BeforeMutation(reason == Exclusion::CapturedWriter ? CloseReason::CapturedWriter
                                                               : CloseReason::Consumer);
    }
}

void Coordinator::ExcludeMappedDevice(const void* owner, DAddr address, PAddr physical, u64 size) {
    if (!requested || !size) {
        return;
    }
    struct Intersection {
        PAddr physical;
        u64 size;
        Exclusion reason;
    };
    std::vector<Intersection> intersections;
    bool invalid{};
    {
        std::scoped_lock lock{mutex};
        invalid = !owner || size > std::numeric_limits<u64>::max() - address ||
                  size > std::numeric_limits<u64>::max() - physical;
        if (invalid) {
            exclusions_complete = false;
        } else if (const auto it = device_exclusions.find(owner); it != device_exclusions.end()) {
            for (size_t bit = 0; bit < it->second.size(); ++bit) {
                it->second[bit].ForEachInRange(address, size, [&](DAddr start, DAddr end) {
                    intersections.push_back({physical + (start - address), end - start,
                                             static_cast<Exclusion>(1U << bit)});
                });
            }
        }
    }
    if (invalid) {
        auto mutation = BeforeMutation(CloseReason::Mapping);
        return;
    }
    for (const auto& [start, length, reason] : intersections) {
        Exclude(start, length, reason);
    }
}
void Coordinator::NoteRejected(Rejection reason) {
    if (!requested)
        return;
    std::scoped_lock lock{mutex};
    ++rejected;
    ++rejection_reasons[static_cast<size_t>(reason)];
}
bool Coordinator::PrepareAndOpen(Identity identity, const std::function<bool()>& materialize,
                                 const std::function<bool()>& grant_write, u64& downloaded,
                                 const std::function<bool(Identity&)>& acquire_writer) {
    auto page = BeginOpening(std::move(identity));
    if (!page) {
        NoteRejected();
        return false;
    }
    bool success = false;
    SCOPE_EXIT {
        std::function<void()> notify;
        if (!success) {
            std::scoped_lock lock{mutex};
            const auto it = pages.find(page->identity.physical);
            if (it != pages.end() && it->second == page &&
                page->barrier.Inspect().state == PageState::Opening) {
                pages.erase(it);
                active_pages.store(pages.size(), std::memory_order_release);
                if (mutations)
                    notify = progress;
            }
        }
        if (notify)
            notify();
    };
    if (acquire_writer && !acquire_writer(page->identity)) {
        NoteRejected(Rejection::Admission);
        return false;
    }
    if (!backend.validate(page->identity)) {
        NoteRejected(Rejection::Admission);
        return false;
    }
    if (!backend.reprotect(page->identity) || !memory_barrier.Run() || !materialize()) {
        NoteRejected(Rejection::Materialize);
        return false;
    }
    if (!backend.capture(page->identity, page->opening_snapshot)) {
        NoteRejected(Rejection::Materialize);
        return false;
    }
    page->baseline_captured = true;
    success = CompleteOpening(page, grant_write);
    if (success) {
        std::scoped_lock lock{mutex};
        downloaded_bytes += downloaded;
    }
    return success;
}
u8 Coordinator::Exclusions(PAddr physical) const {
    std::scoped_lock lock{mutex};
    if (!requested || !exclusions_complete || physical / PageSize >= exclusions.size())
        return 0xff;
    return exclusions[physical / PageSize];
}
std::optional<u8> Coordinator::TryExclusions(PAddr physical) const {
    std::unique_lock lock{mutex, std::try_to_lock};
    if (!lock) return std::nullopt;
    if (!requested || !exclusions_complete || physical / PageSize >= exclusions.size()) return u8{0xff};
    return exclusions[physical / PageSize];
}
std::shared_ptr<Coordinator::Page> Coordinator::BeginOpening(Identity identity) {
    std::scoped_lock lock{mutex};
    if (!Available() || mutations || !exclusions_complete || pages.size() >= 256 ||
        !identity.device_owner || identity.device > std::numeric_limits<u64>::max() - PageSize ||
        (identity.physical & (PageSize - 1)) || (identity.address & (PageSize - 1)) ||
        (identity.device & (PageSize - 1)) || !identity.generation || !identity.lifetime ||
        identity.physical / PageSize >= exclusions.size() ||
        exclusions[identity.physical / PageSize] || pages.contains(identity.physical) ||
        !backend.validate || !backend.reprotect || !backend.drain || !backend.capture ||
        !backend.upload)
        return {};
    if (const auto veto = device_exclusions.find(identity.device_owner);
        veto != device_exclusions.end() &&
        std::any_of(veto->second.begin(), veto->second.end(), [&](const auto& ranges) {
            return ranges.Intersects(identity.device, PageSize);
        })) {
        return {};
    }
    for (const auto& [physical, page] : pages) {
        if (page->barrier.Inspect().state != PageState::Open)
            return {};
    }
    const auto& validate_identity =
        backend.validate_opening ? backend.validate_opening : backend.validate;
    if (!validate_identity(identity))
        return {};
    auto page = std::make_shared<Page>();
    page->identity = std::move(identity);
    page->epoch = *page->barrier.BeginOpen();
    pages.emplace(page->identity.physical, page);
    active_pages.store(pages.size(), std::memory_order_release);
    return page;
}
bool Coordinator::CompleteOpening(const std::shared_ptr<Page>& page,
                                  const std::function<bool()>& grant_write) {
    std::function<void()> notify;
    SCOPE_EXIT {
        if (notify)
            notify();
    };
    std::scoped_lock lock{mutex};
    if (mutations)
        notify = progress;
    const auto it = pages.find(page->identity.physical);
    if (it == pages.end() || it->second != page ||
        page->barrier.Inspect().state != PageState::Opening)
        return false;
    const auto device_veto = device_exclusions.find(page->identity.device_owner);
    const bool device_excluded =
        device_veto != device_exclusions.end() &&
        std::any_of(
            device_veto->second.begin(), device_veto->second.end(),
            [&](const auto& ranges) { return ranges.Intersects(page->identity.device, PageSize); });
    if (mutations || device_excluded || exclusions[page->identity.physical / PageSize] ||
        !backend.validate(page->identity) || !backend.reprotect(page->identity)) {
        pages.erase(it);
        active_pages.store(pages.size(), std::memory_order_release);
        ++rejected;
        ++rejection_reasons[static_cast<size_t>(Rejection::Admission)];
        return false;
    }
    // Publish the materialized backing before granting native access.
    std::atomic_thread_fence(std::memory_order_release);
    try {
        if (!grant_write()) {
            page->barrier.CancelOpen(page->epoch, 0);
            ++rejected;
            ++rejection_reasons[static_cast<size_t>(Rejection::Grant)];
            return false;
        }
    } catch (...) {
        page->barrier.CancelOpen(page->epoch, 0);
        throw;
    }
    if (!page->barrier.CompleteOpen(page->epoch)) {
        page->barrier.CancelOpen(page->epoch, 0);
        return false;
    }
    page->was_open = true;
    ++opened;
    return true;
}
std::shared_ptr<Coordinator::Page> Coordinator::FindPage(PAddr physical) const {
    std::scoped_lock lock{mutex};
    const auto it = pages.find(physical & ~(PageSize - 1));
    return it == pages.end() ? nullptr : it->second;
}
PageBarrier::WriteAttempt Coordinator::TryEnterWriter(const std::shared_ptr<Page>& page,
                                                      Epoch epoch) {
    if (!page)
        return {WriteDisposition::GpuOwned, {}};
    std::scoped_lock lock{mutex};
    const auto it = pages.find(page->identity.physical);
    if (it == pages.end() || it->second != page)
        return {WriteDisposition::Stale, {}};
    auto attempt = it->second->barrier.TryEnterWriter(epoch);
    if (attempt.writer) {
        attempt.writer.NotifyOnRelease([page, notify = progress] {
            static_cast<void>(page);
            if (notify)
                notify();
        });
    }
    return attempt;
}
CloseResult Coordinator::ClosePage(const std::shared_ptr<Page>& page) {
    const auto epoch = page->epoch;
    auto state = page->barrier.Inspect();
    if (state.state == PageState::Opening)
        return CloseResult::Pending;
    if (state.state == PageState::GpuOwned)
        return CloseResult::Complete;
    if (!page->close_started_ns)
        page->close_started_ns = NowNs();
    if (!backend.validate(page->identity))
        return CloseResult::Failed;
    if (state.state == PageState::Open && !page->barrier.BeginClose(epoch, 0))
        return CloseResult::Pending;
    state = page->barrier.Inspect();
    if (!state.reprotected) {
        if (!backend.reprotect(page->identity))
            return CloseResult::Failed;
        if (!page->barrier.ConfirmReprotected(epoch))
            return CloseResult::Pending;
    }
    if (!page->protected_stores) {
        if (!memory_barrier.Run())
            return CloseResult::Failed;
        page->protected_stores = true;
    }
    if (!page->barrier.ConfirmNativeQuiescent(epoch))
        return CloseResult::Pending;
    state = page->barrier.Inspect();
    if (!state.drain_claimed && !page->barrier.TryClaimDrain(epoch))
        return CloseResult::Pending;
    if (!state.journal_drained) {
        auto* previous = draining_owner;
        draining_owner = this;
        SCOPE_EXIT {
            draining_owner = previous;
        };
        const auto drained = backend.drain(page->identity);
        if (drained != CloseResult::Complete)
            return drained;
        if (!page->barrier.ConfirmJournalDrained(epoch))
            return CloseResult::Pending;
    }
    if (!page->barrier.Inspect().capture_claimed && !page->barrier.TryClaimCapture(epoch))
        return CloseResult::Pending;
    if (!page->captured) {
        if (!backend.validate(page->identity) || !backend.capture(page->identity, page->snapshot))
            return CloseResult::Failed;
        page->captured = true;
    }
    if (!backend.validate(page->identity) || !backend.upload(page->identity, page->snapshot))
        return CloseResult::Failed;
    return page->barrier.CompleteClose(epoch, true) ? CloseResult::Complete : CloseResult::Pending;
}
CloseResult Coordinator::CloseAllOnOwner(CloseReason reason) {
    if (closing_owner == this)
        return CloseResult::Pending;
    auto* previous_owner = closing_owner;
    closing_owner = this;
    SCOPE_EXIT {
        closing_owner = previous_owner;
    };
    std::vector<std::shared_ptr<Page>> closing;
    {
        std::scoped_lock lock{mutex};
        for (const auto& [physical, page] : pages) {
            if (page->barrier.Inspect().state == PageState::Open)
                page->barrier.BeginClose(page->epoch, 0);
            closing.push_back(page);
        }
    }
    CloseResult result = CloseResult::Complete;
    for (const auto& page : closing) {
        const auto status = ClosePage(page);
        std::scoped_lock lock{mutex};
        if (status == CloseResult::Failed) {
            ++failures;
            return status;
        }
        if (status == CloseResult::Pending) {
            result = status;
            continue;
        }
        if (page->was_open) {
            ++closed;
            close_ns += NowNs() - page->close_started_ns;
            ++close_reasons[static_cast<size_t>(reason)];
        }
        if (page->baseline_captured) {
            u64 changed_blocks = 0;
            for (size_t offset = 0; offset < PageSize; offset += 16) {
                changed_blocks += !std::equal(page->snapshot.begin() + offset,
                                              page->snapshot.begin() + offset + 16,
                                              page->opening_snapshot.begin() + offset);
            }
            estimated_changed_blocks += changed_blocks;
            estimated_traps_avoided += changed_blocks ? changed_blocks - 1 : 0;
        }
        uploaded_bytes += PageSize;
        pages.erase(page->identity.physical);
        active_pages.store(pages.size(), std::memory_order_release);
    }
    return result;
}
size_t Coordinator::ActivePages() const {
    return active_pages.load(std::memory_order_acquire);
}
void Coordinator::WithAliasProtection(const void* owner, VAddr address, u64 size,
                                      const std::function<void(bool)>& operation) {
    if (!requested) {
        operation(true);
        return;
    }
    std::scoped_lock lock{mutex};
    if (!size || size - 1 > std::numeric_limits<u64>::max() - address) {
        operation(false);
        return;
    }
    bool may_write = true;
    for (const auto& [physical, page] : pages) {
        const auto& identity = page->identity;
        if (identity.cpu_owner == owner && address <= identity.address + PageSize - 1 &&
            identity.address <= address + size - 1 &&
            page->barrier.Inspect().state != PageState::Open)
            may_write = false;
    }
    operation(may_write);
}
void Coordinator::Report() {
    if (!requested)
        return;
    const auto now = NowNs();
    if (now - last_report.load(std::memory_order_relaxed) < 10'000'000'000ULL)
        return;
    std::scoped_lock lock{mutex};
    if (now - last_report < 10'000'000'000ULL)
        return;
    last_report = now;
    constexpr std::array names{"uniform",         "index",          "texture",
                               "captured_writer", "other_consumer", "mapping"};
    for (size_t i = 0; i < names.size(); ++i) {
        LOG_INFO(HW_GPU,
                 "CPU_EPOCH_EXCLUSIONS reason={} bit={:#x} unique_pages={} registrations={} "
                 "range_bytes={} counters=cumulative cache_hits_omitted=true "
                 "pages_overlap_across_reasons=true",
                 names[i], 1U << i, exclusion_stats.pages[i], exclusion_stats.calls[i],
                 exclusion_stats.bytes[i]);
    }
    LOG_INFO(HW_GPU,
             "CPU_EPOCH_EXCLUSIONS invalid_physical={} invalid_device={} registry_complete={}",
             exclusion_stats.invalid_physical, exclusion_stats.invalid_device, exclusions_complete);
    LOG_INFO(HW_GPU,
             "CPU_EPOCH_COORDINATOR counters=cumulative available={} registry_complete={} "
             "active_pages={} opened={} "
             "closed={} "
             "excluded_pages={} failures={} rejected={} downloaded_bytes={} uploaded_bytes={} "
             "close_ns={} "
             "traps_avoided_estimate={} estimate_model=changed_16B_blocks_minus_first "
             "changed_blocks_16={} close_consumer={} close_mapping={} close_protection={} "
             "close_writer={} close_shutdown={} close_avg_us={} "
             "reject_runtime={} reject_candidate={} reject_admission={} reject_materialize={} "
             "reject_grant={} platform_failure={} platform_errno={}",
             Available(), exclusions_complete, pages.size(), opened, closed, excluded_pages,
             failures, rejected, downloaded_bytes, uploaded_bytes, close_ns,
             estimated_traps_avoided, estimated_changed_blocks, close_reasons[0], close_reasons[1],
             close_reasons[2], close_reasons[3], close_reasons[4],
             closed ? close_ns / closed / 1000 : 0, rejection_reasons[0], rejection_reasons[1],
             rejection_reasons[2], rejection_reasons[3], rejection_reasons[4],
             Common::CpuEpochMemoryBarrier::FailureName(memory_barrier.LastFailure()),
             memory_barrier.LastError());
}

} // namespace VideoCommon::CpuEpochs
