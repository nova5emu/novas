// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <span>
#include <vector>
#include "common/alignment.h"
#include "common/slot_vector.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

struct BufferDownloadFragment {
    Common::SlotId buffer_id;
    BufferCopy copy;
};

struct DownloadCopyBatch {
    Common::SlotId buffer_id;
    std::vector<BufferCopy> copies;
};

struct DownloadCopyPlan {
    std::vector<DownloadCopyBatch> batches;
    u64 staging_size{};
    u64 transfer_bytes{};
    u64 copy_regions{};
};

// Only transfer envelopes include holes. Fragments keep their exact source range and size;
// their destination offsets are redirected into the envelope's staging allocation.
inline DownloadCopyPlan PlanDownloadCopies(std::span<BufferDownloadFragment> fragments,
                                           u64 max_gap = 4096) {
    std::ranges::sort(fragments, [](const auto& lhs, const auto& rhs) {
        return lhs.buffer_id != rhs.buffer_id ? lhs.buffer_id < rhs.buffer_id
                                             : lhs.copy.src_offset < rhs.copy.src_offset;
    });
    DownloadCopyPlan plan;
    size_t first{};
    while (first < fragments.size()) {
        const auto buffer_id = fragments[first].buffer_id;
        const u64 begin = fragments[first].copy.src_offset;
        u64 end = begin + fragments[first].copy.size;
        size_t next = first + 1;
        while (next < fragments.size() && fragments[next].buffer_id == buffer_id) {
            const auto& copy = fragments[next].copy;
            if (copy.src_offset > end && copy.src_offset - end > max_gap) break;
            end = (std::max)(end, copy.src_offset + copy.size);
            ++next;
        }
        if (plan.batches.empty() || plan.batches.back().buffer_id != buffer_id) {
            plan.batches.push_back({buffer_id, {}});
        }
        plan.batches.back().copies.push_back({begin, plan.staging_size, end - begin});
        for (size_t i = first; i < next; ++i) {
            fragments[i].copy.dst_offset =
                plan.staging_size + fragments[i].copy.src_offset - begin;
        }
        plan.transfer_bytes += end - begin;
        ++plan.copy_regions;
        plan.staging_size += Common::AlignUp(end - begin, u64{64});
        first = next;
    }
    return plan;
}

} // namespace VideoCommon
