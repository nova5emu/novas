// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <mutex>

#include "core/hle/kernel/k_event.h"

namespace Service {

struct Nx2VideoReadyState {
    void Signal(Kernel::KernelCore& kernel) {
        std::scoped_lock lk{mutex};
        if (event != nullptr && running &&
            (room.load(std::memory_order_relaxed) ||
             output_ready.load(std::memory_order_relaxed))) {
            event->Signal(kernel);
        }
    }

    /// A decoded frame is waiting for the client: wake it now, whatever the input queue level.
    void SignalOutput(Kernel::KernelCore& kernel) {
        output_ready.store(true, std::memory_order_relaxed);
        Signal(kernel);
    }

    void SetRunning(Kernel::KernelCore& kernel, bool value) {
        std::scoped_lock lk{mutex};
        running = value;
        if (event != nullptr && !running) {
            event->Clear(kernel);
        }
    }

    std::mutex mutex;
    Kernel::KEvent* event{};
    std::atomic<bool> room{true};
    /// Outputs are waiting to be fetched. The client waits on this one event for both input room
    /// and finished frames: with only `room` (input queue below its watermark) a movie whose tiny
    /// AV1 frames keep hundreds of inputs queued woke the client only on its ~100 ms timeout,
    /// playing at ~8 fps.
    std::atomic<bool> output_ready{false};

private:
    bool running{true};
};

} // namespace Service
