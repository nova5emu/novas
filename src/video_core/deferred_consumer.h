// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "video_core/deferred_write_journal.h"

namespace VideoCommon::DeferredWrites {

// Capture both identities: closing one DAddr must also exclude producers through its aliases.
// Caller keeps translations stable through the operation; mapping transitions are separate.
template <typename Memory>
void AddConsumerDeviceRange(std::vector<TransitionRange>& ranges, Memory& memory,
                            DAddr address, u64 size) {
    if (size == 0) return;
    ASSERT(size - 1 <= (std::numeric_limits<u64>::max)() - address);
    ranges.push_back({TransitionSpace::Device, address, size});
    for (u64 offset = 0; offset < size;) {
        const u64 length = (std::min)(size - offset, 4096 - ((address + offset) & 4095));
        memory.ApplyOpOnMappedMemory(address + offset, length, [&](u8*, PAddr physical) {
            auto& last = ranges.back();
            if (last.space == TransitionSpace::Physical && last.address + last.size == physical) {
                last.size += length;
            } else {
                ranges.push_back({TransitionSpace::Physical, physical, length});
            }
        });
        offset += length;
    }
}

// Scheduler-owner stack, for operations that preserve physical backing. Nested operations
// extend the outer union, never release its leases. No wait for another owner under cache locks.
class ConsumerScope {
public:
    template <typename Apply>
    ConsumerScope(Journal* journal_, std::span<const TransitionRange> ranges, Apply&& apply)
        : journal{journal_} {
        if (!journal || ranges.empty()) {
            journal = nullptr;
            return;
        }
        exceptions = std::uncaught_exceptions();
        for (auto* scope = current; scope; scope = scope->previous) {
            if (scope->journal == journal) {
                context = scope->context;
                break;
            }
        }
        if (!context) {
            owned = journal->TryBeginConsumer();
            if (!owned) UNREACHABLE_MSG("Cannot begin deferred consumer reservation");
            context = owned.get();
        }
        if (!context->TryExtend(ranges)) {
            UNREACHABLE_MSG("Foreign deferred transition under consumer cache locks");
        }
        // The sink must patch existing leased buffers, never create/join/delete buffers or
        // recursively drain. The production sink remains disconnected until the producer stage.
        if (!context->ClosureReady()) apply(context->Watermark());
        if (!context->ClosureReady()) {
            UNREACHABLE_MSG("Deferred consumer mutation before its write prefix was applied");
        }
        previous = current;
        current = this;
    }

    ~ConsumerScope() {
        if (!journal) return;
        ASSERT(current == this);
        current = previous;
        if (owned && std::uncaught_exceptions() == exceptions && !owned->Complete()) {
            UNREACHABLE_MSG("Deferred consumer reservation ended before its prefix was applied");
        }
        // Exceptions leave the identities closed, via ConsumerContext's abort destructor.
    }

    ConsumerScope(const ConsumerScope&) = delete;
    ConsumerScope& operator=(const ConsumerScope&) = delete;

private:
    inline static thread_local ConsumerScope* current{};
    Journal* journal;
    int exceptions{};
    ConsumerScope* previous{};
    Journal::ConsumerContext* context{};
    std::unique_ptr<Journal::ConsumerContext> owned;
};

} // namespace VideoCommon::DeferredWrites
