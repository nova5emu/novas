// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>
#include <new>
#include <boost/icl/interval_map.hpp>

#include "common/assert.h"
#include "common/common_types.h"
#include "common/gpu_transition_profile.h"
#include "common/scope_exit.h"

namespace VideoCommon::DeferredWrites {

using Cut = u64;

enum class Payload : u8 { Bytes, Zero };

enum class LeaseLookupStatus : size_t {
    Found, Suspended, Missing, Closing, DeviceMismatch, MappingMismatch, WriterExcluded,
    GenerationMismatch, Count,
};
inline constexpr std::array lease_lookup_status_names{
    "found", "suspended", "missing", "closing", "device_mismatch", "mapping_mismatch",
    "writer_excluded", "lease_generation_mismatch",
};
static_assert(lease_lookup_status_names.size() == static_cast<size_t>(LeaseLookupStatus::Count));

struct Write {
    Cut sequence{};
    PAddr physical_address{};
    DAddr device_address{};
    u64 mapping_generation{};
    u64 lease_generation{};
    u32 size{};
    Payload payload{Payload::Bytes};
    std::array<u8, 32> bytes{};
};

enum class PublishStatus { Published, Full, InvalidWrite, Rejected, SequenceExhausted, LeaseUnavailable, Suspended };
struct Publication {
    PublishStatus status;
    Cut sequence{};
};
enum class DrainResult { Complete, InvalidCut, ApplyFailed };
enum class LeaseGrantStatus {
    Granted, Invalid, Suspended, Existing, Closing, Transition, FreshIdentity,
    WriterExcluded, GenerationExhausted,
};

enum class TransitionSpace { Physical, Device };
struct TransitionRange {
    TransitionSpace space;
    u64 address;
    u64 size;
};
// Caller-owned, nonwrapping epochs. CPU virtual ranges must first be resolved under its locks.
struct MappingStamp {
    u64 domain;
    u64 generation;
    auto operator<=>(const MappingStamp&) const = default;
};
enum class TransitionResult { Complete, DrainPending, ReadbacksPending, MappingChanged };

enum class SuspensionState { Running, Draining, SuspendedReady };
enum class SuspensionResult { Ready, Busy, DrainPending, ReadbacksPending, TransitionsPending, InvalidToken };

class Journal;

// Transfer ownership to deferred callbacks, or Fork a participant for nested work. A ready
// token must stay alive throughout the mutation/fallback; a transient state check is not a pin.
class Suspension {
public:
    Suspension() = default;
    ~Suspension();
    Suspension(const Suspension&) = delete;
    Suspension& operator=(const Suspension&) = delete;
    Suspension(Suspension&& other) noexcept;
    Suspension& operator=(Suspension&& other) noexcept;

    explicit operator bool() const { return owner != nullptr; }
    [[nodiscard]] Cut Watermark() const { return cut; }
    [[nodiscard]] u64 Epoch() const { return epoch; }
    [[nodiscard]] bool CreatedEpoch() const { return created_epoch; }
    [[nodiscard]] u64 Origin() const { return origin; }
    [[nodiscard]] u64 StartedNs() const { return started_ns; }
    [[nodiscard]] Suspension Fork() const;
    [[nodiscard]] bool IsReady() const;
    // No callback or preparer claim escapes the publication gate on the idle fast path.
    [[nodiscard]] SuspensionResult TryPrepareIdle(bool& prepared) const;
    // Host threads only, without kernel/cache/mapping locks. Timeout never authorizes fallback.
    [[nodiscard]] bool WaitForReady(std::chrono::milliseconds timeout) const;
    void Reset();

    // One participant prepares; others get Busy and may wait/retry without outer locks.
    // drain(W) applies W AND arranges completion of old readbacks. No journal gate is held.
    template <typename Drain>
    [[nodiscard]] SuspensionResult TryPrepare(Drain&& drain) const;

private:
    friend class Journal;
    Suspension(Journal& journal, u64 generation, Cut watermark, u64 source, u64 start,
               bool created = false)
        : owner{&journal}, epoch{generation}, cut{watermark}, origin{source},
          started_ns{start}, created_epoch{created} {}
    Journal* owner{};
    u64 epoch{};
    Cut cut{};
    u64 origin{};
    u64 started_ns{};
    bool created_epoch{};
};

// Portable ordering and CPU readback history; mapping/lease eligibility is external.
// Producers, cut capture and slot reclamation share a short publication gate. The consumer
// invokes its sink WITHOUT that gate; a GPU wait in the sink cannot block CPU publication.
enum class WriterOrigin : size_t { Temporary, NcePersistent, KernelJitPersistent, LegacyPersistent, Count };
inline constexpr std::array writer_origin_names{
    "temporary", "nce_fallback_persistent", "kernel_jit", "legacy_writer",
};

struct WriterProfileCounts {
    std::array<u32, static_cast<size_t>(WriterOrigin::Count)> origins{};
    std::array<u32, static_cast<size_t>(Common::GpuTransitionProfile::DeferredFallbackReason::Count) + 1> reasons{};
    WriterProfileCounts& operator+=(const WriterProfileCounts& other) {
        for (size_t i = 0; i < origins.size(); ++i) origins[i] += other.origins[i];
        for (size_t i = 0; i < reasons.size(); ++i) reasons[i] += other.reasons[i];
        return *this;
    }
    WriterProfileCounts& operator-=(const WriterProfileCounts& other) {
        for (size_t i = 0; i < origins.size(); ++i) origins[i] -= other.origins[i];
        for (size_t i = 0; i < reasons.size(); ++i) reasons[i] -= other.reasons[i];
        return *this;
    }
    bool operator==(const WriterProfileCounts&) const = default;
};
static_assert(static_cast<size_t>(Common::GpuTransitionProfile::DeferredFallbackReason::Count) < 64);
class Journal {
public:
    struct WriterOriginSnapshot {
        u64 permissions{}, physical_pages_known{}, device_only_permissions{};
    };
    struct RetentionSnapshot {
        std::array<WriterOriginSnapshot, static_cast<size_t>(WriterOrigin::Count)> writers_by_origin{};
        u64 fixed_bytes{}, live_records{}, pending_records{}, leases{}, writer_permissions{}, readbacks{};
        u64 suspension_participants{};
        SuspensionState state{};
        bool diagnostic_complete{};
    };
    [[nodiscard]] RetentionSnapshot GetRetentionSnapshot() const {
        auto lock = Common::GpuTransitionProfile::TryDiagnosticLock(
            publication_gate, Common::GpuTransitionProfile::DiagnosticSite::JournalSnapshot);
        if (!lock) {
            return {};
        }
        RetentionSnapshot result{};
        result.diagnostic_complete = true;
        result.fixed_bytes = sizeof(cpu_page_publications) + page_index.capacity() * sizeof(PageBucket);
        for (const auto& ring : rings) {
            result.fixed_bytes += ring.entries.capacity() * sizeof(Entry);
            result.live_records += ring.published - ring.retired;
            result.pending_records += ring.published - ring.consumed;
        }
        result.leases = leases.size();
        std::array<std::vector<std::pair<u64, u64>>, static_cast<size_t>(WriterOrigin::Count)> pages;
        for (auto* pin = writer_permissions; pin; pin = pin->next) {
            ++result.writer_permissions;
            const auto origin = static_cast<size_t>(pin->origin);
            auto& counts = result.writers_by_origin[origin];
            ++counts.permissions;
            bool physical{};
            for (const auto& range : pin->ranges) {
                if (range.space != TransitionSpace::Physical) continue;
                physical = true;
                pages[origin].emplace_back(range.address >> 12,
                                          (range.address + range.size - 1) >> 12);
            }
            if (!physical) ++counts.device_only_permissions;
        }
        for (auto* readback = readbacks; readback; readback = readback->next) ++result.readbacks;
        result.suspension_participants = suspension_participants;
        result.state = suspension_state;
        lock.unlock();
        for (size_t origin = 0; origin < pages.size(); ++origin) {
            auto& intervals = pages[origin];
            std::sort(intervals.begin(), intervals.end());
            u64 next{};
            for (const auto& [first, last] : intervals) {
                const u64 begin = (std::max)(first, next);
                if (begin <= last) result.writers_by_origin[origin].physical_pages_known += last - begin + 1;
                next = (std::max)(next, last + 1);
            }
        }

        return result;
    }

    [[nodiscard]] Suspension BeginSuspension(u64 origin = 0, bool measure = false) {
        std::scoped_lock lock{publication_gate};
        const bool created = suspension_state == SuspensionState::Running;
        if (created) {
            if (suspension_epoch == (std::numeric_limits<u64>::max)()) {
                UNREACHABLE_MSG("Deferred write suspension epoch exhausted");
            }
            ++suspension_epoch;
            suspension_cut = published;
            suspension_origin = origin;
            suspension_started_ns = measure ? static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()) : 0;
            suspension_state = SuspensionState::Draining;
        }
        AddSuspensionParticipantLocked();
        return Suspension{*this, suspension_epoch, suspension_cut, suspension_origin,
                          suspension_started_ns, created};
    }

    [[nodiscard]] SuspensionState GetSuspensionState() const {
        std::scoped_lock lock{publication_gate};
        return suspension_state;
    }

    [[nodiscard]] bool HasReadbacks() const {
        std::scoped_lock lock{publication_gate};
        return readbacks != nullptr;
    }

    [[nodiscard]] bool OwnsSuspension(const Suspension& token) const {
        std::scoped_lock lock{publication_gate};
        return ValidSuspensionLocked(token);
    }

    // Adopt only an abandoned epoch. Never turn a racing resume into a new suspension.
    [[nodiscard]] std::optional<Suspension> TryAdoptAbandonedSuspension() {
        std::scoped_lock lock{publication_gate};
        if (suspension_state != SuspensionState::Draining || suspension_participants ||
            suspension_preparing) {
            return std::nullopt;
        }
        AddSuspensionParticipantLocked();
        return Suspension{*this, suspension_epoch, suspension_cut, suspension_origin,
                          suspension_started_ns};
    }

    struct Lease {
        PAddr physical_page{};
        DAddr device_page{};
        u64 mapping_generation{};
        u64 generation{};
    };
    struct Closure {
        Lease lease;
        Cut cut{};
    };

    class TransitionReservation;
    class ConsumerContext;

    // Admission exclusion, not a mapping pin or a replacement for cache invalidation.
    // Prepare before the legacy writer's locks; keep it through the actual commit (or job).
    class WriterPermission {
    public:
        ~WriterPermission() {
            std::scoped_lock lock{owner.publication_gate};
            if (!linked) return;
            if (profile_indexed) {
                if (owner.writer_profile_valid) owner.UpdateWriterProfile(*this, false);
            } else {
                --owner.unindexed_profile_writers;
            }
            for (const auto& page : persistent_pages) {
                const auto it = owner.persistent_writer_pages.find(page);
                ASSERT(it != owner.persistent_writer_pages.end() && it->second != 0);
                if (--it->second == 0) owner.persistent_writer_pages.erase(it);
            }
            for (auto& [physical, state] : owner.leases) {
                if (!Touches(ranges, state.lease)) continue;
                ASSERT(state.writer_exclusions != 0);
                --state.writer_exclusions;
            }
            if (previous) previous->next = next;
            else owner.writer_permissions = next;
            if (next) next->previous = previous;
        }
        WriterPermission(const WriterPermission&) = delete;
        WriterPermission& operator=(const WriterPermission&) = delete;

        [[nodiscard]] Common::GpuTransitionProfile::DeferredFallbackReason FirstFallbackReason() const { return first_reason; }
        [[nodiscard]] Cut Watermark() const { return cut; }
        [[nodiscard]] bool Ready() const {
            std::scoped_lock lock{owner.publication_gate};
            return ready;
        }
        // Applying B must finish before the old path can commit. Its existing readback
        // invalidation/preservation and mapping revalidation are still mandatory.
        [[nodiscard]] bool MarkReady() {
            std::map<PAddr, LeaseState> retired_leases;
            std::scoped_lock lock{owner.publication_gate};
            if (owner.AppliedThrough() < cut) return false;
            if (ready) return true;
            for (auto it = owner.leases.begin(); it != owner.leases.end();) {
                // A concurrent transition owns its closure and will revoke that lease itself.
                if (!it->second.closing && Touches(ranges, it->second.lease)) {
                    retired_leases.insert(owner.leases.extract(it++));
                } else {
                    ++it;
                }
            }
            ready = true;
            return true;
        }

    private:
        friend class Journal;
        explicit WriterPermission(Journal& journal) : owner{journal} {}
        Journal& owner;
        std::vector<TransitionRange> ranges;
        WriterOrigin origin{WriterOrigin::Temporary};
        Common::GpuTransitionProfile::DeferredFallbackReason first_reason{Common::GpuTransitionProfile::DeferredFallbackReason::Count};
        bool profile_indexed{};
        std::vector<std::pair<TransitionSpace, u64>> persistent_pages;
        Cut cut{};
        bool ready{};
        bool linked{};
        WriterPermission* previous{};
        WriterPermission* next{};
    };

    // Cancellation before Ready has no write effects: the old leases remain valid.
    // Overlapping writers may share exclusion; their mutual ordering remains the caller's.
    [[nodiscard]] std::unique_ptr<WriterPermission> BeginWriterPermission(
        std::span<const TransitionRange> ranges, WriterOrigin origin = WriterOrigin::Temporary,
        Common::GpuTransitionProfile::DeferredFallbackReason first_reason = Common::GpuTransitionProfile::DeferredFallbackReason::Count) {
        if (ranges.empty() || origin >= WriterOrigin::Count || first_reason > Common::GpuTransitionProfile::DeferredFallbackReason::Count) return {};
        for (const auto& range : ranges) {
            if ((range.space != TransitionSpace::Physical && range.space != TransitionSpace::Device) ||
                !range.size || range.size - 1 > (std::numeric_limits<u64>::max)() - range.address)
                return {};
        }
        auto permission = std::unique_ptr<WriterPermission>{new WriterPermission{*this}};
        permission->ranges.assign(ranges.begin(), ranges.end());
        permission->origin = origin;
        permission->first_reason = first_reason;
        if (origin != WriterOrigin::Temporary) {
            for (const auto& range : ranges) {
                const u64 last = (range.address + range.size - 1) >> 12;
                for (u64 page = range.address >> 12; page <= last; ++page)
                    permission->persistent_pages.emplace_back(range.space, page);
            }
            auto& pages = permission->persistent_pages;
            std::sort(pages.begin(), pages.end());
            pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
        }
        std::map<std::pair<TransitionSpace, u64>, u64> prepared_pages;
        for (const auto& page : permission->persistent_pages) prepared_pages.emplace(page, 1);
        std::scoped_lock lock{publication_gate};
        while (!prepared_pages.empty()) {
            auto node = prepared_pages.extract(prepared_pages.begin());
            auto inserted = persistent_writer_pages.insert(std::move(node));
            if (!inserted.inserted) ++inserted.position->second;
        }
        if (Common::GpuTransitionProfile::enabled.load(std::memory_order_relaxed) && writer_profile_valid) {
            permission->profile_indexed = UpdateWriterProfile(*permission, true);
        }
        if (!permission->profile_indexed) ++unindexed_profile_writers;
        for (auto& [physical, state] : leases) {
            if (Touches(ranges, state.lease)) {
                permission->cut = (std::max)(permission->cut, state.last_sequence);
                ++state.writer_exclusions;
            }
        }
        permission->next = writer_permissions;
        if (writer_permissions) writer_permissions->previous = permission.get();
        writer_permissions = permission.get();
        permission->linked = true;
        return permission;
    }

    [[nodiscard]] bool HasPersistentWriterExclusion(PAddr physical_page, DAddr device_page) const {
        std::scoped_lock lock{publication_gate};
        return persistent_writer_pages.contains({TransitionSpace::Physical, physical_page >> 12}) ||
               persistent_writer_pages.contains({TransitionSpace::Device, device_page >> 12});
    }
    struct WriterExclusionInspection {
        u64 origin_mask{}, first_reason_mask{};
        bool complete{};
    };
    [[nodiscard]] WriterExclusionInspection InspectWriterExclusions(
        PAddr physical_page, std::optional<DAddr> device_page = std::nullopt) const {
        Common::GpuTransitionProfile::WatchdogInspectionScope diagnostic;
        auto lock = Common::GpuTransitionProfile::TryDiagnosticLock(
            publication_gate, Common::GpuTransitionProfile::DiagnosticSite::JournalInspection);
        if (!lock) {
            return {0, 0, false};
        }
        WriterExclusionInspection result{0, 0, writer_profile_valid && unindexed_profile_writers == 0};
        const auto inspect = [&](TransitionSpace space, u64 page) {
            const auto& index = writer_profile_pages[static_cast<size_t>(space)];
            const auto found = index.find(page >> 12);
            if (found == index.end()) return;
            for (size_t i = 0; i < found->second.origins.size(); ++i)
                if (found->second.origins[i]) result.origin_mask |= u64{1} << i;
            for (size_t i = 0; i < found->second.reasons.size(); ++i)
                if (found->second.reasons[i]) result.first_reason_mask |= u64{1} << i;
        };
        if (writer_profile_valid) {
            inspect(TransitionSpace::Physical, physical_page);
            if (device_page) inspect(TransitionSpace::Device, *device_page);
        }
        return result;
    }
    // Only for newly allocated, pinned physical pages and currently unbacked device ranges.
    // The caller proves prior owners/readbacks were retired at their teardown boundary.
    // This is NOT a replacement/remap permission and never drains while cache locks are held.
    class FreshIdentity {
    public:
        ~FreshIdentity() {
            std::scoped_lock lock{owner.publication_gate};
            if (!linked) return;
            if (previous) previous->next = next;
            else owner.fresh_identities = next;
            if (next) next->previous = previous;
        }
        FreshIdentity(const FreshIdentity&) = delete;
        FreshIdentity& operator=(const FreshIdentity&) = delete;

        // One installation thread extends the permission, under its mapping locks.
        // Other producers/grants observe the whole extension atomically under the commit gate.
        [[nodiscard]] bool TryExtend(std::span<const TransitionRange> additions) {
            if (additions.empty()) return false;
            for (const auto& range : additions) {
                if ((range.space != TransitionSpace::Physical && range.space != TransitionSpace::Device) ||
                    !range.size || range.size - 1 > (std::numeric_limits<u64>::max)() - range.address)
                    return false;
            }
            auto extended = ranges;
            extended.insert(extended.end(), additions.begin(), additions.end());
            std::scoped_lock lock{owner.publication_gate};
            for (const auto& [physical, state] : owner.leases) {
                if (Touches(additions, state.lease)) return false;
            }
            for (const auto& ring : owner.rings) {
                for (u64 i = ring.retired; i < ring.published; ++i) {
                    const auto& write = ring.entries[i % ring.entries.size()].write;
                    if (TouchesWrite(additions, write)) return false;
                }
            }
            for (auto* active = owner.fresh_identities; active; active = active->next) {
                if (active == this) continue;
                for (const auto& a : active->ranges) {
                    for (const auto& b : additions) {
                        if (a.space == b.space && Overlaps(a.address, a.size, b.address, b.size))
                            return false;
                    }
                }
            }
            ranges.swap(extended);
            if (!linked) {
                next = owner.fresh_identities;
                if (next) next->previous = this;
                owner.fresh_identities = this;
                linked = true;
            }
            return true;
        }

    private:
        friend class Journal;
        explicit FreshIdentity(Journal& journal) : owner{journal} {}
        Journal& owner;
        std::vector<TransitionRange> ranges;
        FreshIdentity* previous{};
        FreshIdentity* next{};
        bool linked{};
    };

    [[nodiscard]] std::unique_ptr<FreshIdentity> TryReserveFreshIdentity(
        std::span<const TransitionRange> ranges) {
        auto identity = std::unique_ptr<FreshIdentity>{new FreshIdentity{*this}};
        if (!identity->TryExtend(ranges)) return {};
        return identity;
    }

private:
    struct LeaseState {
        Lease lease;
        Cut cut{};
        Cut last_sequence{};
        // Updated with permission insertion/removal under publication_gate. New grants still
        // inspect the permission list, so no lease can appear inside an existing exclusion.
        u64 writer_exclusions{};
        bool closing{};
        TransitionReservation* reservation{};
        bool abandoned{};
    };
    using LeaseNode = std::map<PAddr, LeaseState>::node_type;

public:

    class TransitionReservation {
    public:
        ~TransitionReservation() { Abort(); }
        TransitionReservation(const TransitionReservation&) = delete;
        TransitionReservation& operator=(const TransitionReservation&) = delete;


        [[nodiscard]] Cut Watermark() const { return cut; }
        [[nodiscard]] std::span<const Lease> ClosedLeases() const { return closed; }
        [[nodiscard]] bool ClosureReady() const {
            return owner && owner->AppliedThrough() >= cut;
        }

        // Abort means no mutation occurred. Keep affected leases closed and their identities
        // alive; a later reservation adopts these abandoned closures. Never silently reopen.
        // Accepted writes/history survive. Destruction never waits for GPU work.
        void Abort() {
            if (owner) owner->ReleaseTransition(*this, false);
        }

        // locks is the COMPLETE caller lock bundle, initially held. drain runs without it and
        // without journal gates; it must also arrange old readback completion. On return locks
        // are held again. Caller exclusion must prevent NEW readbacks during validation/mutation.
        // validate compares the captured epochs (not addresses or a 16-bit page-table block).
        template <typename Locks, typename Drain, typename Validate, typename Mutate>
        [[nodiscard]] TransitionResult Run(Locks& locks, Drain&& drain, Validate&& validate,
                                           Mutate&& mutate) {
            static_assert(std::is_nothrow_invocable_r_v<bool, Validate, std::span<const MappingStamp>>);
            static_assert(std::is_nothrow_invocable_r_v<void, Mutate>);
            ASSERT(owner && locks.owns_lock());
            bool drained{};
            locks.unlock();
            {
                SCOPE_EXIT { locks.lock(); };
                drained = drain(cut);
            }
            if (!drained || !ClosureReady()) return TransitionResult::DrainPending;
            // Tokens currently lack range identities: reject conservatively while ANY is live.
            {
                std::scoped_lock lock{owner->publication_gate};
                if (owner->readbacks) return TransitionResult::ReadbacksPending;
            }
            if (!validate(stamps)) return TransitionResult::MappingChanged;
            mutate(); // No journal gate across kernel/cache callbacks.
            owner->ReleaseTransition(*this, true);
            return TransitionResult::Complete;
        }

    private:
        friend class Journal;
        friend class ConsumerContext;
        explicit TransitionReservation(Journal& journal) : owner{&journal} {}
        Journal* owner;
        std::vector<TransitionRange> ranges;
        std::vector<MappingStamp> stamps;
        std::vector<Lease> closed;
        std::vector<LeaseNode> retired;
        Cut cut{};
        TransitionReservation* previous{};
        TransitionReservation* next{};
        bool linked{};
    };

    // One serialized scheduler owner shares this context through nested cache operations.
    // Physical backing/translations must stay stable; this is not a mapping transition.
    class ConsumerContext {
    public:
        ConsumerContext(const ConsumerContext&) = delete;
        ConsumerContext& operator=(const ConsumerContext&) = delete;

        [[nodiscard]] Cut Watermark() const { return reservation->Watermark(); }
        [[nodiscard]] std::span<const Lease> ClosedLeases() const {
            return reservation->ClosedLeases();
        }
        [[nodiscard]] bool ClosureReady() const { return reservation->ClosureReady(); }

        // Extends the root's union atomically. Children borrow the root and never release it.
        // On success, apply the new watermark before using any of the affected buffers.
        [[nodiscard]] bool TryExtend(std::span<const TransitionRange> ranges) {
            return reservation->owner &&
                   reservation->owner->ExtendConsumer(*reservation, ranges);
        }

        // Call after the outermost mutation. Live readbacks retain all their CPU history.
        [[nodiscard]] bool Complete() {
            if (!ClosureReady()) return false;
            reservation->owner->ReleaseTransition(*reservation, true);
            return true;
        }

        // Conservatively closes identities even if some nested mutations already completed.
        void Abort() { reservation->Abort(); }

    private:
        friend class Journal;
        explicit ConsumerContext(Journal& journal)
            : reservation{new TransitionReservation{journal}} {}
        std::unique_ptr<TransitionReservation> reservation;
    };

    [[nodiscard]] std::unique_ptr<ConsumerContext> TryBeginConsumer() {
        return std::unique_ptr<ConsumerContext>{new ConsumerContext{*this}};
    }

    // Call with stable translations/epochs under the caller's locks. Null means invalid input
    // or a competing closure/reservation: release those locks before waiting and retrying.
    // All affected leases close atomically at ONE cut, with no partial closure on rejection.
    [[nodiscard]] std::unique_ptr<TransitionReservation> TryReserveTransition(
        std::span<const TransitionRange> ranges, std::span<const MappingStamp> stamps) {
        if (ranges.empty() || stamps.empty()) return {};
        for (const auto& range : ranges) {
            if ((range.space != TransitionSpace::Physical && range.space != TransitionSpace::Device) ||
                range.size == 0 || range.size - 1 > (std::numeric_limits<u64>::max)() - range.address)
                return {};
        }
        for (const auto& stamp : stamps) {
            if (!stamp.domain || !stamp.generation) return {};
        }
        auto reservation = std::unique_ptr<TransitionReservation>{new TransitionReservation{*this}};
        reservation->ranges.assign(ranges.begin(), ranges.end());
        reservation->stamps.assign(stamps.begin(), stamps.end());
        for (;;) {
            std::unique_lock lock{publication_gate};
            for (auto* active = transitions; active; active = active->next) {
                for (const auto& a : active->ranges) {
                    for (const auto& b : ranges) {
                        if (a.space == b.space && Overlaps(a.address, a.size, b.address, b.size))
                            return {};
                    }
                }
            }
            size_t affected{};
            for (const auto& [physical, state] : leases) {
                if (!Touches(ranges, state.lease)) continue;
                if (state.closing && !state.abandoned) return {};
                ++affected;
            }
            // Allocation never holds the publication gate. Recheck if grants grew in between.
            if (reservation->closed.capacity() < affected || reservation->retired.capacity() < affected) {
                lock.unlock();
                reservation->closed.reserve(affected);
                reservation->retired.reserve(affected);
                continue;
            }
            for (const auto& [physical, state] : leases) {
                if (!Touches(ranges, state.lease)) continue;
                reservation->closed.push_back(state.lease);
            }
            reservation->cut = published;
            for (const auto& lease : reservation->closed) {
                auto& state = leases.at(lease.physical_page);
                state.closing = true;
                state.cut = published;
                state.reservation = reservation.get();
                state.abandoned = false;
            }
            reservation->next = transitions;
            if (transitions) transitions->previous = reservation.get();
            transitions = reservation.get();
            reservation->linked = true;
            return reservation;
        }
    }

    // The caller has already checked buffer ownership, aliases, CPU permissions and textures.
    // This registry is not the eligibility policy. Grant/regrant only on the scheduler owner.
    [[nodiscard]] std::optional<Lease> GrantLease(
        PAddr physical, DAddr device, u64 mapping, LeaseGrantStatus* status = nullptr) {
        const auto reject = [&](LeaseGrantStatus reason) -> std::optional<Lease> {
            if (status) *status = reason;
            return std::nullopt;
        };
        if ((physical & 4095) || (device & 4095) || mapping == 0) {
            return reject(LeaseGrantStatus::Invalid);
        }
        std::map<PAddr, LeaseState> prepared;
        prepared.emplace(physical, LeaseState{});
        auto node = prepared.extract(physical);
        std::scoped_lock lock{publication_gate};
        if (suspension_state != SuspensionState::Running) return reject(LeaseGrantStatus::Suspended);
        if (const auto existing = leases.find(physical); existing != leases.end()) {
            return reject(existing->second.closing ? LeaseGrantStatus::Closing : LeaseGrantStatus::Existing);
        }
        if (lease_generation == (std::numeric_limits<u64>::max)())
            return reject(LeaseGrantStatus::GenerationExhausted);
        for (auto* reservation = transitions; reservation; reservation = reservation->next) {
            if (Touches(reservation->ranges, Lease{physical, device, mapping, 0}))
                return reject(LeaseGrantStatus::Transition);
        }
        for (auto* identity = fresh_identities; identity; identity = identity->next) {
            if (Touches(identity->ranges, Lease{physical, device, mapping, 0}))
                return reject(LeaseGrantStatus::FreshIdentity);
        }
        if (WriterExcludedLocked(Lease{physical, device, mapping, 0}))
            return reject(LeaseGrantStatus::WriterExcluded);
        const Lease lease{physical, device, mapping, ++lease_generation};
        node.mapped().lease = lease;
        leases.insert(std::move(node));
        if (status) *status = LeaseGrantStatus::Granted;
        return lease;
    }

    // A copy is only a hint: TryPublishLeased revalidates it under the commit gate.
    [[nodiscard]] std::optional<Lease> FindLease(PAddr physical, DAddr device, u64 mapping,
                                               LeaseLookupStatus* status = nullptr) const {
        std::scoped_lock lock{publication_gate};
        if (suspension_state != SuspensionState::Running) {
            if (status) *status = LeaseLookupStatus::Suspended;
            return std::nullopt;
        }
        const auto it = leases.find(physical);
        if (it == leases.end() || it->second.closing ||
            it->second.lease.device_page != device ||
            it->second.lease.mapping_generation != mapping ||
            it->second.writer_exclusions != 0) {
            if (status) {
                if (it == leases.end()) *status = LeaseLookupStatus::Missing;
                else if (it->second.closing) *status = LeaseLookupStatus::Closing;
                else if (it->second.lease.device_page != device) *status = LeaseLookupStatus::DeviceMismatch;
                else if (it->second.lease.mapping_generation != mapping) *status = LeaseLookupStatus::MappingMismatch;
                else *status = LeaseLookupStatus::WriterExcluded;
            }
            return std::nullopt;
        }
        if (status) *status = LeaseLookupStatus::Found;
        return it->second.lease;
    }

    // Stops admission and captures a finite prefix atomically with CPU commits.
    // A second closer must retry outside its locks, not proceed as though closure succeeded.
    [[nodiscard]] std::optional<Closure> CloseLease(const Lease& lease) {
        std::scoped_lock lock{publication_gate};
        auto it = leases.find(lease.physical_page);
        if (it == leases.end() || it->second.lease.generation != lease.generation ||
            it->second.closing) {
            return std::nullopt;
        }
        it->second.closing = true;
        it->second.cut = published;
        return Closure{it->second.lease, published};
    }

    [[nodiscard]] bool ClosureReady(const Closure& closure) const {
        std::scoped_lock lock{publication_gate};
        return ClosureReadyLocked(closure);
    }

    // Caller drained the cut, resolved old readbacks and completed/revalidated the mutation.
    // Closing state prevents regrant throughout that transaction. No implicit reopening.
    [[nodiscard]] bool FinishClosure(const Closure& closure) {
        typename std::map<PAddr, LeaseState>::node_type retired;
        {
            std::scoped_lock lock{publication_gate};
            if (!ClosureReadyLocked(closure) ||
                leases.at(closure.lease.physical_page).reservation ||
                leases.at(closure.lease.physical_page).abandoned) {
                return false;
            }
            retired = leases.extract(closure.lease.physical_page);
        }
        return true;
    }

    template <typename Commit>
    [[nodiscard]] Publication TryPublishLeased(std::size_t core, const Write& write, Commit&& commit,
                                               LeaseLookupStatus* status = nullptr) {
        static_assert(std::is_nothrow_invocable_r_v<bool, Commit>);
        if (core >= rings.size() || !Valid(write)) {
            return RecordPublication(PublishStatus::InvalidWrite);
        }
        std::scoped_lock lock{publication_gate};
        if (suspension_state != SuspensionState::Running) {
            if (status) *status = LeaseLookupStatus::Suspended;
            return RecordPublication(PublishStatus::Suspended);
        }
        const auto it = leases.find(write.physical_address & ~PAddr{4095});
        if (it == leases.end() || it->second.closing || it->second.writer_exclusions != 0 ||
            it->second.lease.generation != write.lease_generation ||
            it->second.lease.mapping_generation != write.mapping_generation ||
            it->second.lease.device_page + (write.physical_address & 4095) != write.device_address) {
            if (status) {
                if (it == leases.end()) *status = LeaseLookupStatus::Missing;
                else if (it->second.closing) *status = LeaseLookupStatus::Closing;
                else if (it->second.writer_exclusions != 0) *status = LeaseLookupStatus::WriterExcluded;
                else if (it->second.lease.generation != write.lease_generation) *status = LeaseLookupStatus::GenerationMismatch;
                else if (it->second.lease.mapping_generation != write.mapping_generation) *status = LeaseLookupStatus::MappingMismatch;
                else *status = LeaseLookupStatus::DeviceMismatch;
            }
            return RecordPublication(PublishStatus::LeaseUnavailable);
        }
        if (status) *status = LeaseLookupStatus::Found;
        const auto result = PublishLocked<true>(core, write, commit);
        if (result.status == PublishStatus::Published) it->second.last_sequence = result.sequence;
        return result;
    }

    class Readback {
    public:
        ~Readback() {
            if (!linked) {
                return;
            }
            std::scoped_lock lock{owner.publication_gate};
            if (previous) {
                previous->next = next;
            } else {
                owner.readbacks = next;
            }
            if (next) {
                next->previous = previous;
            }
        }
        Readback(const Readback&) = delete;
        Readback& operator=(const Readback&) = delete;

        [[nodiscard]] Cut Watermark() const {
            return watermark;
        }

    private:
        friend class Journal;
        explicit Readback(Journal& owner_) : owner{owner_} {}
        Journal& owner;
        Cut watermark{};
        Readback* previous{};
        Readback* next{};
        bool linked{};
    };

    explicit Journal(std::size_t core_count, std::size_t capacity_per_core = 4096)
        : rings(core_count), page_index(core_count * capacity_per_core * 2) {
        if (core_count == 0 || capacity_per_core == 0) {
            UNREACHABLE_MSG("Deferred write journal requires nonempty rings");
        }
        for (auto& ring : rings) {
            ring.entries.resize(capacity_per_core);
        }
    }

    ~Journal() {
        if (suspension_participants || suspension_preparing) {
            UNREACHABLE_MSG("Suspension outlived its deferred write journal");
        }
        if (transitions || fresh_identities || writer_permissions) {
            UNREACHABLE_MSG("Transition reservation outlived its deferred write journal");
        }
        if (readbacks) {
            UNREACHABLE_MSG("Readback outlived its deferred write journal");
        }
    }

    // Scheduler owner, immediately before recording copies (or waiting for a mapped buffer).
    // Allocation precedes the gates. Do not call from a DrainThrough sink.
    [[nodiscard]] std::unique_ptr<Readback> BeginReadback() {
        auto token = std::unique_ptr<Readback>{new Readback{*this}};
        std::scoped_lock consume{consumer_gate};
        std::scoped_lock lock{publication_gate};
        token->watermark = AppliedThrough();
        token->next = readbacks;
        if (readbacks) {
            readbacks->previous = token.get();
        }
        readbacks = token.get();
        token->linked = true;
        return token;
    }

    // Caller holds stable mapping/lease identity and excludes invalidated GPU readbacks.
    // visit(offset, size) writes only surviving bytes AND any associated validity state.
    // It runs under the CPU commit gate: no allocation, GPU wait or journal reentry.
    template <typename Visit>
    void WriteReadback(const Readback& token, PAddr physical, u64 size, Visit&& visit) {
        if (&token.owner != this || size == 0 || size > 4096 - (physical & 4095)) {
            UNREACHABLE_MSG("Invalid deferred write readback page");
        }
        std::scoped_lock lock{publication_gate};
        std::bitset<4096> protected_bytes;
        for (auto* entry = page_index[FindPage(physical >> 12)].head; entry; entry = entry->next) {
            const auto& write = entry->write;
            if (write.sequence <= token.watermark) {
                continue;
            }
            const u64 begin = (std::max)(physical & 4095, write.physical_address & 4095);
            const u64 end = (std::min)((physical & 4095) + size,
                                      (write.physical_address & 4095) + write.size);
            for (u64 byte = begin; byte < end; ++byte) {
                protected_bytes.set(byte - (physical & 4095));
            }
        }
        for (u64 offset = 0; offset < size;) {
            if (protected_bytes.test(offset)) {
                ++offset;
                continue;
            }
            const u64 begin = offset++;
            while (offset < size && !protected_bytes.test(offset)) {
                ++offset;
            }
            visit(begin, offset - begin);
        }
    }

    [[nodiscard]] Cut ReclaimableThrough() const {
        std::scoped_lock lock{publication_gate};
        return ReclaimableThroughLocked();
    }

    // A command-page snapshot is valid only if no CPU commit to its physical page followed
    // the applied prefix used to capture it. Keep this stamp after history/lease retirement.
    // Hash collisions reject extra snapshots; they can never validate an obsolete one.
    // Caller holds stable translation (mapping_guard before this gate). visit must not allocate,
    // wait, acquire cache/mapping locks, or reenter the journal; normally it just copies bytes.
    template <typename Visit>
    [[nodiscard]] bool VisitIfCpuPageUnchanged(PAddr physical, Cut applied_at_copy, Visit&& visit) {
        std::scoped_lock lock{publication_gate};
        if (applied_at_copy > AppliedThrough() ||
            cpu_page_publications[CpuPageBucket(physical)] > applied_at_copy) {
            return false;
        }
        visit();
        return true;
    }

    // Unleased ordering/test entry point; production B producers must use TryPublishLeased.
    // commit validates external preconditions and commits CPU bytes. On false it must have
    // changed nothing. It must not allocate, wait for GPU work or reenter this journal.
    // Capacity is checked before commit; no descriptor/sequence is published on rejection.
    template <typename Commit>
    [[nodiscard]] Publication TryPublish(std::size_t core, const Write& write, Commit&& commit) {
        static_assert(std::is_nothrow_invocable_r_v<bool, Commit>);
        if (core >= rings.size() || !Valid(write)) {
            return RecordPublication(PublishStatus::InvalidWrite);
        }
        std::scoped_lock lock{publication_gate};
        return PublishLocked(core, write, commit);
    }

private:
    static Publication RecordPublication(PublishStatus status, Cut sequence = 0,
                                         u64 bytes = 0, u64 occupancy = 0) {
        using ProfileStatus = Common::GpuTransitionProfile::DeferredPublishStatus;
        ProfileStatus sample{};
        switch (status) {
        case PublishStatus::Published: sample = ProfileStatus::Published; break;
        case PublishStatus::Full: sample = ProfileStatus::Full; break;
        case PublishStatus::InvalidWrite: sample = ProfileStatus::InvalidWrite; break;
        case PublishStatus::Rejected: sample = ProfileStatus::Rejected; break;
        case PublishStatus::SequenceExhausted: sample = ProfileStatus::SequenceExhausted; break;
        case PublishStatus::LeaseUnavailable: sample = ProfileStatus::LeaseUnavailable; break;
        case PublishStatus::Suspended: sample = ProfileStatus::Suspended; break;
        }
        Common::GpuTransitionProfile::RecordDeferredPublish(sample, bytes, occupancy);
        return {status, sequence};
    }

    template <bool Leased = false, typename Commit>
    Publication PublishLocked(std::size_t core, const Write& write, Commit& commit) {
        if (suspension_state != SuspensionState::Running) return RecordPublication(PublishStatus::Suspended);
        // Leased callers checked the per-lease exclusion count under this same gate.
        // FreshIdentity cannot overlap an existing lease and blocks any new grant.
        if constexpr (!Leased) {
            if (WriterExcludedLocked(Lease{write.physical_address & ~PAddr{4095},
                                           write.device_address & ~DAddr{4095},
                                           write.mapping_generation, write.lease_generation})) {
                return RecordPublication(PublishStatus::LeaseUnavailable);
            }
            for (auto* identity = fresh_identities; identity; identity = identity->next) {
                if (TouchesWrite(identity->ranges, write)) return RecordPublication(PublishStatus::LeaseUnavailable);
            }
        }
        auto& ring = rings[core];
        if (ring.published - ring.retired == ring.entries.size()) {
            return RecordPublication(PublishStatus::Full);
        }
        if (published == (std::numeric_limits<Cut>::max)()) {
            return RecordPublication(PublishStatus::SequenceExhausted);
        }
        if (!commit()) {
            return RecordPublication(PublishStatus::Rejected);
        }
        auto& entry = ring.entries[ring.published % ring.entries.size()];
        entry.write = write;
        entry.write.sequence = ++published;
        cpu_page_publications[CpuPageBucket(write.physical_address)] = published;
        Link(entry);
        ++ring.published;
        return RecordPublication(PublishStatus::Published, entry.write.sequence, write.size,
                                 ring.published - ring.retired);
    }

public:

    // Acquiring the same gate includes any CPU commit already visible to the observer,
    // even if that producer had not yet published its descriptor when observed.
    [[nodiscard]] Cut CaptureCut() const {
        std::scoped_lock lock{publication_gate};
        return published;
    }

    [[nodiscard]] Cut AppliedThrough() const {
        return applied.load(std::memory_order_acquire);
    }

    // One serialized consumer, global commit order across all cores, finite prefix only.
    // A false sink result must have no side effects; that record remains unapplied.
    // The sink may publish another write, but must not recursively drain or retire.
    template <typename Apply>
    [[nodiscard]] DrainResult DrainThrough(Cut cut, Apply&& apply) {
        std::scoped_lock consume{consumer_gate};
        if (cut > CaptureCut()) {
            return DrainResult::InvalidCut;
        }
        while (AppliedThrough() < cut) {
            Ring* selected = nullptr;
            Write next;
            {
                std::scoped_lock lock{publication_gate};
                const Cut expected = AppliedThrough() + 1;
                for (auto& ring : rings) {
                    if (ring.consumed != ring.published) {
                        const auto& candidate = ring.entries[ring.consumed % ring.entries.size()].write;
                        if (candidate.sequence == expected) {
                            selected = &ring;
                            next = candidate;
                            break;
                        }
                    }
                }
            }
            if (!selected) {
                UNREACHABLE_MSG("Deferred write prefix has a hole");
            }
            if (!apply(next)) {
                return DrainResult::ApplyFailed;
            }
            {
                std::scoped_lock lock{publication_gate};
                ++selected->consumed;
                applied.store(next.sequence, std::memory_order_release);
            }
            Common::GpuTransitionProfile::RecordDeferredApplied();
        }
        return DrainResult::Complete;
    }

    // Explicit retirement, NOT a side effect of DrainThrough or token destruction.
    // Live tokens veto retirement beyond their applied watermark.
    [[nodiscard]] bool RetireThrough(Cut cut) {
        std::scoped_lock lock{publication_gate};
        if (cut > ReclaimableThroughLocked()) {
            return false;
        }
        RetireThroughLocked(cut);
        return true;
    }

private:
    friend class Suspension;

    static bool TouchesWrite(std::span<const TransitionRange> ranges, const Write& write) {
        for (const auto& range : ranges) {
            const u64 address = range.space == TransitionSpace::Physical
                                    ? write.physical_address : write.device_address;
            if (Overlaps(range.address, range.size, address, write.size)) return true;
        }
        return false;
    }

    void AddSuspensionParticipantLocked() {
        if (suspension_participants == (std::numeric_limits<u64>::max)()) {
            UNREACHABLE_MSG("Deferred write suspension participant count exhausted");
        }
        ++suspension_participants;
    }

    bool ValidSuspensionLocked(const Suspension& token) const {
        return token.owner == this && token.epoch == suspension_epoch &&
               suspension_state != SuspensionState::Running;
    }

    Suspension ForkSuspension(const Suspension& token) {
        std::scoped_lock lock{publication_gate};
        if (!ValidSuspensionLocked(token)) return {};
        AddSuspensionParticipantLocked();
        return Suspension{*this, suspension_epoch, suspension_cut, suspension_origin,
                          suspension_started_ns};
    }

    void ReleaseSuspension(const Suspension& token) {
        std::scoped_lock lock{publication_gate};
        ASSERT(ValidSuspensionLocked(token) && suspension_participants);
        if (--suspension_participants == 0 &&
            suspension_state == SuspensionState::SuspendedReady) {
            ASSERT(!suspension_preparing);
            suspension_state = SuspensionState::Running;
        }
        // Abandoning an unfinished drain keeps admission closed. The next Begin adopts it.
    }

    SuspensionResult ClaimSuspensionPreparation(const Suspension& token) {
        std::scoped_lock lock{publication_gate};
        if (!ValidSuspensionLocked(token)) return SuspensionResult::InvalidToken;
        if (suspension_state == SuspensionState::SuspendedReady) return SuspensionResult::Ready;
        if (suspension_preparing) return SuspensionResult::Busy;
        suspension_preparing = true;
        return SuspensionResult::DrainPending; // This participant owns the preparation attempt.
    }

    void EndSuspensionPreparation() {
        std::scoped_lock lock{publication_gate};
        suspension_preparing = false;
    }

    SuspensionResult CompleteSuspensionPreparation(const Suspension& token,
                                                   bool* idle_prepared = nullptr) {
        // Construct/deallocate map storage outside the commit gate (including MSVC's sentinel).
        std::map<PAddr, LeaseState> retired_leases;
        {
            std::scoped_lock lock{publication_gate};
            if (!ValidSuspensionLocked(token)) return SuspensionResult::InvalidToken;
            if (idle_prepared) {
                if (suspension_state == SuspensionState::SuspendedReady) return SuspensionResult::Ready;
                if (suspension_preparing) return SuspensionResult::Busy;
            } else {
                ASSERT(suspension_preparing && suspension_state == SuspensionState::Draining);
            }
            if (AppliedThrough() < suspension_cut) return SuspensionResult::DrainPending;
            if (readbacks) return SuspensionResult::ReadbacksPending;
            if (transitions) return SuspensionResult::TransitionsPending;
            for (const auto& [physical, state] : leases) {
                if (state.closing && !state.abandoned) return SuspensionResult::TransitionsPending;
            }
            RetireThroughLocked(suspension_cut);
            leases.swap(retired_leases);
            suspension_state = SuspensionState::SuspendedReady;
            if (idle_prepared) *idle_prepared = true;
        }
        suspension_changed.notify_all();
        return SuspensionResult::Ready;
    }

    void RetireThroughLocked(Cut cut) {
        u64 count{};
        for (auto& ring : rings) {
            while (ring.retired != ring.consumed &&
                   ring.entries[ring.retired % ring.entries.size()].write.sequence <= cut) {
                Unlink(ring.entries[ring.retired % ring.entries.size()]);
                ++ring.retired;
                ++count;
            }
        }
        Common::GpuTransitionProfile::RecordDeferredRetired(count);
    }

    static bool Overlaps(u64 a, u64 a_size, u64 b, u64 b_size) {
        return a <= b ? b - a < a_size : a - b < b_size;
    }

    static bool Touches(std::span<const TransitionRange> ranges, const Lease& lease) {
        for (const auto& range : ranges) {
            const u64 page = range.space == TransitionSpace::Physical ? lease.physical_page
                                                                     : lease.device_page;
            if (Overlaps(range.address, range.size, page, 4096)) return true;
        }
        return false;
    }

    bool ExtendConsumer(TransitionReservation& reservation,
                        std::span<const TransitionRange> added_ranges) {
        if (added_ranges.empty()) return false;
        for (const auto& range : added_ranges) {
            if ((range.space != TransitionSpace::Physical && range.space != TransitionSpace::Device) ||
                range.size == 0 || range.size - 1 > (std::numeric_limits<u64>::max)() - range.address)
                return false;
        }
        // Never allocate while publication is gated. A list represents the accumulated union;
        // already covered requests need no additional range entry.
        auto extended = reservation.ranges;
        for (const auto& range : added_ranges) {
            const bool covered = std::ranges::any_of(extended, [&](const auto& previous) {
                return previous.space == range.space && previous.address <= range.address &&
                       range.address - previous.address < previous.size &&
                       range.size <= previous.size - (range.address - previous.address);
            });
            if (!covered) extended.push_back(range);
        }
        for (;;) {
            std::unique_lock lock{publication_gate};
            for (auto* active = transitions; active; active = active->next) {
                if (active == &reservation) continue;
                for (const auto& a : active->ranges) {
                    for (const auto& b : extended) {
                        if (a.space == b.space && Overlaps(a.address, a.size, b.address, b.size))
                            return false;
                    }
                }
            }
            size_t added{};
            for (const auto& [physical, state] : leases) {
                if (!Touches(extended, state.lease) || state.reservation == &reservation) continue;
                if (state.closing && !state.abandoned) return false;
                ++added;
            }
            const size_t needed = reservation.closed.size() + added;
            if (reservation.closed.capacity() < needed || reservation.retired.capacity() < needed) {
                lock.unlock();
                reservation.closed.reserve(needed);
                reservation.retired.reserve(needed);
                continue;
            }
            for (auto& [physical, state] : leases) {
                if (!Touches(extended, state.lease) || state.reservation == &reservation) continue;
                reservation.closed.push_back(state.lease);
                state.closing = true;
                state.reservation = &reservation;
                state.abandoned = false;
            }
            reservation.ranges.swap(extended);
            reservation.cut = published;
            for (const auto& lease : reservation.closed) {
                leases.at(lease.physical_page).cut = published;
            }
            if (!reservation.linked) {
                reservation.next = transitions;
                if (transitions) transitions->previous = &reservation;
                transitions = &reservation;
                reservation.linked = true;
            }
            return true;
        }
    }

    void ReleaseTransition(TransitionReservation& reservation, bool completed) {
        // Deallocate completed lease nodes outside the gate. Abort keeps their old identity.
        static_assert(std::is_nothrow_move_constructible_v<LeaseNode>);
        {
            std::scoped_lock lock{publication_gate};
            if (reservation.linked) {
                for (const auto& lease : reservation.closed) {
                    auto it = leases.find(lease.physical_page);
                    ASSERT(it != leases.end() && it->second.reservation == &reservation);
                    if (completed) {
                        reservation.retired.push_back(leases.extract(it));
                    } else {
                        it->second.reservation = nullptr;
                        it->second.abandoned = true;
                    }
                }
                if (reservation.previous) reservation.previous->next = reservation.next;
                else transitions = reservation.next;
                if (reservation.next) reservation.next->previous = reservation.previous;
                reservation.linked = false;
            }
            reservation.owner = nullptr;
        }
        reservation.retired.clear();
    }

    bool ClosureReadyLocked(const Closure& closure) const {
        const auto it = leases.find(closure.lease.physical_page);
        return it != leases.end() && it->second.closing &&
               it->second.lease.generation == closure.lease.generation &&
               it->second.cut == closure.cut && AppliedThrough() >= closure.cut;
    }

    bool WriterExcludedLocked(const Lease& lease) const {
        for (auto* permission = writer_permissions; permission; permission = permission->next) {
            if (Touches(permission->ranges, lease)) return true;
        }
        return false;
    }

    Cut ReclaimableThroughLocked() const {
        Cut cut = AppliedThrough();
        for (auto* token = readbacks; token; token = token->next) {
            cut = (std::min)(cut, token->watermark);
        }
        return cut;
    }

    static bool Valid(const Write& write) {
        constexpr u64 page_mask = 4095;
        if (write.size == 0 || write.mapping_generation == 0 || write.lease_generation == 0 ||
            (write.physical_address & page_mask) + write.size > page_mask + 1 ||
            (write.device_address & page_mask) + write.size > page_mask + 1) {
            return false;
        }
        switch (write.payload) {
        case Payload::Bytes:
            return write.size <= write.bytes.size();
        case Payload::Zero:
            return write.size <= 2048;
        default:
            return false;
        }
    }

    struct Entry {
        Write write;
        Entry* previous{};
        Entry* next{};
    };

    struct PageBucket {
        PAddr page{};
        Entry* head{};
    };

    std::size_t HashPage(PAddr page) const {
        return (page * 0x9e3779b97f4a7c15ULL) % page_index.size();
    }

    static constexpr std::size_t CpuPageBucket(PAddr physical) {
        // Fold high bits too: distant physical pages must not systematically share low buckets.
        const u64 page = physical >> 12;
        return ((page * 0x9e3779b97f4a7c15ULL) >> 50);
    }

    std::size_t FindPage(PAddr page) const {
        auto slot = HashPage(page);
        while (page_index[slot].head && page_index[slot].page != page) {
            slot = (slot + 1) % page_index.size();
        }
        return slot;
    }

    void Link(Entry& entry) {
        const auto page = entry.write.physical_address >> 12;
        auto& bucket = page_index[FindPage(page)];
        entry.previous = nullptr;
        entry.next = bucket.head;
        if (bucket.head) {
            bucket.head->previous = &entry;
        }
        bucket = {page, &entry};
    }

    void Unlink(Entry& entry) {
        auto hole = FindPage(entry.write.physical_address >> 12);
        if (entry.previous) {
            entry.previous->next = entry.next;
        } else {
            page_index[hole].head = entry.next;
        }
        if (entry.next) {
            entry.next->previous = entry.previous;
        }
        entry.previous = entry.next = nullptr;
        if (page_index[hole].head) {
            return;
        }
        // Backshift deletion prevents tombstones accumulating as rings wrap.
        const auto capacity = page_index.size();
        for (auto scan = (hole + 1) % capacity; page_index[scan].head;
             scan = (scan + 1) % capacity) {
            const auto home = HashPage(page_index[scan].page);
            if ((hole + capacity - home) % capacity < (scan + capacity - home) % capacity) {
                page_index[hole] = page_index[scan];
                hole = scan;
            }
        }
        page_index[hole] = {};
    }

    struct Ring {
        std::vector<Entry> entries;
        u64 published{};
        u64 consumed{};
        u64 retired{};
    };

    std::vector<Ring> rings;
    std::vector<PageBucket> page_index;
    std::array<Cut, 16384> cpu_page_publications{};
    mutable std::mutex publication_gate;
    std::mutex consumer_gate;
    Cut published{};
    std::atomic<Cut> applied{};
    Readback* readbacks{};
    std::map<PAddr, LeaseState> leases;
    u64 lease_generation{};
    TransitionReservation* transitions{};
    FreshIdentity* fresh_identities{};
    WriterPermission* writer_permissions{};
    std::map<std::pair<TransitionSpace, u64>, u64> persistent_writer_pages;
    std::array<boost::icl::interval_map<u64, WriterProfileCounts>, 2> writer_profile_pages;
    u64 unindexed_profile_writers{};
    bool writer_profile_valid{true};

    bool UpdateWriterProfile(const WriterPermission& permission, bool add) {
        WriterProfileCounts counts;
        counts.origins[static_cast<size_t>(permission.origin)] = 1;
        if (permission.origin == WriterOrigin::NcePersistent)
            counts.reasons[static_cast<size_t>(permission.first_reason)] = 1;
        try {
            for (const auto& range : permission.ranges) {
                auto& index = writer_profile_pages[static_cast<size_t>(range.space)];
                const auto interval = boost::icl::interval<u64>::right_open(
                    range.address >> 12, ((range.address + range.size - 1) >> 12) + 1);
                if (add) index.add({interval, counts});
                else index.subtract({interval, counts});
            }
        } catch (const std::bad_alloc&) {
            for (auto& index : writer_profile_pages) index.clear();
            writer_profile_valid = false;
            return false;
        }
        return true;
    }
    std::condition_variable suspension_changed;
    SuspensionState suspension_state{SuspensionState::Running};
    u64 suspension_epoch{};
    u64 suspension_origin{};
    u64 suspension_started_ns{};
    u64 suspension_participants{};
    Cut suspension_cut{};
    bool suspension_preparing{};
};

inline std::shared_ptr<Journal::WriterPermission> BeginSharedWriterPermission(
    std::shared_ptr<Journal> journal, std::span<const TransitionRange> ranges,
    WriterOrigin origin = WriterOrigin::Temporary,
    Common::GpuTransitionProfile::DeferredFallbackReason first_reason = Common::GpuTransitionProfile::DeferredFallbackReason::Count) {
    if (!journal) return {};
    auto permission = journal->BeginWriterPermission(ranges, origin, first_reason);
    if (!permission) return {};
    return std::shared_ptr<Journal::WriterPermission>(permission.release(),
        [journal](Journal::WriterPermission* token) { delete token; });
}

// A callback can retain this handle past renderer teardown. The journal outlives the final
// participant's destructor; copying the handle retains that same participant, not a new epoch.
inline std::shared_ptr<Suspension> BeginSharedSuspension(std::shared_ptr<Journal> journal,
                                                       u64 origin = 0, bool measure = false) {
    if (!journal) return {};
    return std::shared_ptr<Suspension>(new Suspension{journal->BeginSuspension(origin, measure)},
                                      [journal](Suspension* token) { delete token; });
}

inline std::shared_ptr<Suspension> AdoptSharedAbandonedSuspension(std::shared_ptr<Journal> journal) {
    if (!journal) return {};
    auto token = journal->TryAdoptAbandonedSuspension();
    if (!token) return {};
    return std::shared_ptr<Suspension>(new Suspension{std::move(*token)},
                                      [journal](Suspension* value) { delete value; });
}

// The caller prepares BEFORE queueing, without outer locks. Copies across frontend/backend
// queues retain the ready participant; each copy is one-shot. This does not retain objects
// captured by raw pointer. Release on invocation even if the queue keeps its consumed wrapper.
inline std::function<void()> RetainReadySuspension(std::shared_ptr<Suspension> token,
                                                 std::function<void()>&& operation) {
    if (!token) return std::move(operation);
    ASSERT_MSG(token->IsReady(), "Deferred mapping callback requires a prepared suspension");
    return [token = std::move(token), operation = std::move(operation)]() mutable {
        ASSERT(token && token->IsReady());
        auto participant = std::move(token);
        auto run_once = std::move(operation);
        run_once();
    };
}

inline Suspension::~Suspension() { Reset(); }

inline SuspensionResult Suspension::TryPrepareIdle(bool& prepared) const {
    prepared = false;
    auto participant = Fork();
    if (!participant) return SuspensionResult::InvalidToken;
    return participant.owner->CompleteSuspensionPreparation(participant, &prepared);
}

inline Suspension::Suspension(Suspension&& other) noexcept
    : owner{std::exchange(other.owner, nullptr)}, epoch{other.epoch}, cut{other.cut},
      origin{other.origin}, started_ns{other.started_ns}, created_epoch{other.created_epoch} {}

inline Suspension& Suspension::operator=(Suspension&& other) noexcept {
    if (this != &other) {
        Reset();
        owner = std::exchange(other.owner, nullptr);
        epoch = other.epoch;
        cut = other.cut;
        origin = other.origin;
        started_ns = other.started_ns;
        created_epoch = other.created_epoch;
    }
    return *this;
}

inline void Suspension::Reset() {
    if (owner) {
        owner->ReleaseSuspension(*this);
        owner = nullptr;
    }
}

inline Suspension Suspension::Fork() const {
    return owner ? owner->ForkSuspension(*this) : Suspension{};
}

inline bool Suspension::IsReady() const {
    if (!owner) return false;
    std::scoped_lock lock{owner->publication_gate};
    return owner->ValidSuspensionLocked(*this) &&
           owner->suspension_state == SuspensionState::SuspendedReady;
}

inline bool Suspension::WaitForReady(std::chrono::milliseconds timeout) const {
    if (!owner) return false;
    std::unique_lock lock{owner->publication_gate};
    return owner->suspension_changed.wait_for(lock, timeout, [&] {
        return owner->ValidSuspensionLocked(*this) &&
               owner->suspension_state == SuspensionState::SuspendedReady;
    });
}

template <typename Drain>
SuspensionResult Suspension::TryPrepare(Drain&& drain) const {
    // Pin independently: even an error path that releases the original token cannot reopen
    // admission while the drain callback is running. A token object itself is not thread-safe.
    auto participant = Fork();
    if (!participant) return SuspensionResult::InvalidToken;
    auto& journal = *participant.owner;
    const auto claimed = journal.ClaimSuspensionPreparation(participant);
    if (claimed != SuspensionResult::DrainPending) return claimed;
    SCOPE_EXIT { journal.EndSuspensionPreparation(); };
    if (!drain(participant.cut)) return SuspensionResult::DrainPending;
    return journal.CompleteSuspensionPreparation(participant);
}

} // namespace VideoCommon::DeferredWrites
