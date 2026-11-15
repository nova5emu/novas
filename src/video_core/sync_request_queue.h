// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include "common/common_types.h"

namespace VideoCommon {

// Protected by the GPU's request mutex. The owner runs actions outside that mutex.
class SyncRequestQueue {
public:
    enum class Status { Complete, Pending };
    enum class Ordering { AfterPrior, Independent };
    struct Request {
        u64 fence;
        std::function<Status()> action;
        Ordering ordering;
    };

    u64 Push(std::function<Status()> action, Ordering ordering);
    std::optional<Request> Pop();
    bool CanRun(const Request& request) const;
    void Resolve(Request request, Status status);
    bool IsComplete(u64 fence) const;
    u64 CompletedThrough() const { return completed; }
    size_t PendingCount() const { return pending.size(); }

private:
    void Complete(u64 fence);
    std::deque<Request> pending;
    std::map<u64, u64> completed_ranges;
    // Only requests that yielded Pending impose a completion dependency.
    std::set<u64> deferred;
    u64 last{}, completed{};
};

} // namespace VideoCommon
