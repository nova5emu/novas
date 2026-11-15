// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <atomic>
#include "common/guest_thread_profile.h"

namespace Common::GuestHandoffProfile {
enum class WakeKind : u8 { Arbiter, Mutex, Event };
struct Timeline {
    std::array<u64, 7> ns{};
    u32 chosen_core{~0U};
    bool pending{}, invalid{}, migrated{};
    void Cancel();
    void Stamp(size_t stage, u64 now);
    void Stage(size_t stage, u32 core, u64 now);
    void Complete(u64 entry_ns);
    bool Ordered() const;
};
// One producer per core and one GPU-loop consumer. Full queues discard new records.
template <typename T, size_t Capacity>
class CompletionRing {
public:
    bool Push(const T& value) {
        const auto w = write.load(std::memory_order_relaxed);
        if (w - read.load(std::memory_order_acquire) >= Capacity) {
            return false;
        }
        records[w % Capacity] = value;
        write.store(w + 1, std::memory_order_release);
        return true;
    }
    bool Pop(T& value) {
        const auto r = read.load(std::memory_order_relaxed);
        if (r == write.load(std::memory_order_acquire)) {
            return false;
        }
        value = records[r % Capacity];
        read.store(r + 1, std::memory_order_release);
        return true;
    }
    u64 Available() const {
        return write.load(std::memory_order_acquire) - read.load(std::memory_order_relaxed);
    }
    void Reset() {
        read = 0;
        write = 0;
    }

private:
    std::array<T, Capacity> records{};
    std::atomic<u64> write{}, read{};
};
class ThreadState {
public:
    void Wake(u64 from_pid, u64 from_tid, u64 object, WakeKind kind, u8 svc);
    void Runnable(bool runnable);
    void Chosen(u32 core);
    void CoreService(u32 core);
    void Switch(u32 core);
    void NceRun(u32 core);
    void Enter(const GuestThreadProfile::ThreadInfo& info, u32 core, u64 virtual_affinity,
               u64 physical_affinity, u64 entry_ns = 0, u64 entry_frame = ~u64{0});
    bool HasPending() const;

private:
    bool Lock();
    void Unlock();
    void Stage(size_t stage, u32 core);
    std::atomic_flag gate = ATOMIC_FLAG_INIT;
    std::atomic<u64> lost_epoch{};
    std::atomic<bool> pending{};
    u64 session{}, observed_loss{}, from_pid{}, from_tid{}, object{}, frame{};
    WakeKind kind{};
    u8 svc{};
    Timeline timeline{};
    std::atomic<u32> activity_slot{~0U};
    std::atomic<u64> activity_session{}, target_tid{}, target_pid{};
};
void Configure();
bool Enabled();
void Frame();
// Signal-handler leaf: a lock-free atomic load, without allocation or TLS access.
u64 CurrentFrame() noexcept;
void Report();
void CoreIdle(u32 core, bool waiting);
void CoreInterrupt(u32 core);
void CoreInterruptHandled(u32 core);
} // namespace Common::GuestHandoffProfile
