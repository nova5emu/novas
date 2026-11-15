// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include "video_core/sync_request_queue.h"

namespace VideoCommon {

u64 SyncRequestQueue::Push(std::function<Status()> action, Ordering ordering) {
    if (last == std::numeric_limits<u64>::max()) {
        throw std::overflow_error("GPU sync request fence exhausted");
    }
    const u64 fence = last + 1;
    pending.push_back({fence, std::move(action), ordering});
    last = fence;
    return fence;
}
std::optional<SyncRequestQueue::Request> SyncRequestQueue::Pop() {
    if (pending.empty()) {
        return std::nullopt;
    }
    Request request = std::move(pending.front());
    pending.pop_front();
    return request;
}
bool SyncRequestQueue::CanRun(const Request& request) const {
    return request.ordering == Ordering::Independent || deferred.empty() ||
           *deferred.begin() >= request.fence;
}
void SyncRequestQueue::Resolve(Request request, Status status) {
    if (status == Status::Pending) {
        deferred.insert(request.fence);
        pending.push_back(std::move(request));
    } else {
        deferred.erase(request.fence);
        Complete(request.fence);
    }
}
bool SyncRequestQueue::IsComplete(u64 fence) const {
    if (fence <= completed) {
        return true;
    }
    const auto next = completed_ranges.upper_bound(fence);
    return next != completed_ranges.begin() && std::prev(next)->second >= fence;
}
void SyncRequestQueue::Complete(u64 fence) {
    if (fence <= completed) {
        return;
    }
    if (fence == completed + 1) {
        completed = fence;
        const auto next = completed_ranges.begin();
        if (next != completed_ranges.end() && next->first == completed + 1) {
            completed = next->second;
            completed_ranges.erase(next);
        }
        return;
    }
    auto next = completed_ranges.upper_bound(fence);
    u64 begin = fence, end = fence;
    auto previous = completed_ranges.end();
    if (next != completed_ranges.begin()) {
        const auto candidate = std::prev(next);
        if (candidate->second >= fence - 1) {
            previous = candidate;
            begin = candidate->first;
            end = (std::max)(end, candidate->second);
        }
    }
    const bool merge_next = next != completed_ranges.end() &&
                            end != std::numeric_limits<u64>::max() && next->first == end + 1;
    if (merge_next) {
        end = next->second;
    }
    if (previous != completed_ranges.end()) {
        previous->second = end;
    } else {
        completed_ranges.emplace(begin, end);
    }
    if (merge_next) {
        completed_ranges.erase(next);
    }
}

} // namespace VideoCommon
