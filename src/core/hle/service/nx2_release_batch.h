// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <utility>

#include "common/assert.h"
#include "common/common_types.h"

namespace Service::Nx2 {

struct SparseBackingClaim {
    u64 region;
    u64 block;
    u64 generation;

    bool Matches(u64 current_generation, u64 block_count) const {
        return generation == current_generation && block < block_count;
    }
};

// Seal only after emitting every mapping removal covered by this batch.
class ReleaseBatch {
public:
    std::shared_ptr<std::atomic<bool>> Completion() {
        ASSERT(!sealed);
        if (!done) done = std::make_shared<std::atomic<bool>>(false);
        return done;
    }

    template <typename Enqueue>
    void Seal(Enqueue&& enqueue) {
        ASSERT(!sealed);
        if (done) {
            std::forward<Enqueue>(enqueue)([completion = done] {
                completion->store(true, std::memory_order_release);
            });
        }
        sealed = true;
    }

private:
    std::shared_ptr<std::atomic<bool>> done;
    bool sealed{};
};

template <typename Unmap, typename Release>
bool TryReleaseBacking(const std::shared_ptr<std::atomic<bool>>& done,
                       Unmap&& unmap, Release&& release) {
    if (!done || !done->load(std::memory_order_acquire)) return false;
    if (!std::forward<Unmap>(unmap)()) return false;
    std::forward<Release>(release)();
    return true;
}

} // namespace Service::Nx2
