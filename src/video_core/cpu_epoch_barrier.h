// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <functional>
#include <optional>
#include <utility>
#include "common/common_types.h"

namespace VideoCommon::CpuEpochs {

enum class PageState { GpuOwned, Opening, Open, Closing };
enum class WriteDisposition { Granted, Defer, GpuOwned, Stale, Contended, Full };

struct Epoch {
    u32 generation{};
};

// One stable physical identity per barrier. Its owner must outlive every Writer.
// CAS operations return on contention for rescheduling by the coordinator.
class PageBarrier {
public:
    class Writer {
    public:
        Writer() = default;
        ~Writer();
        Writer(Writer&& other) noexcept;
        Writer& operator=(Writer&& other) noexcept;
        Writer(const Writer&) = delete;
        Writer& operator=(const Writer&) = delete;
        explicit operator bool() const noexcept {
            return owner != nullptr;
        }
        void Release() noexcept;
        void NotifyOnRelease(std::function<void()> notify) {
            released = std::move(notify);
        }

    private:
        friend class PageBarrier;
        explicit Writer(PageBarrier* owner_) : owner{owner_} {}
        PageBarrier* owner{};
        std::function<void()> released;
    };

    struct WriteAttempt {
        WriteDisposition disposition;
        Writer writer;
    };
    struct Continuation {
        WriteDisposition disposition;
        Epoch epoch;
    };
    struct Snapshot {
        PageState state;
        Epoch epoch;
        u32 writers;
        bool reprotected, native_quiescent, drain_claimed, journal_drained, capture_claimed;
    };

    std::optional<Epoch> BeginOpen() noexcept;
    bool CompleteOpen(Epoch epoch) noexcept;
    // Only the opening actor cancels, after it has stopped changing writable mappings.
    bool CancelOpen(Epoch epoch, u32 native_core_mask) noexcept;
    // A queued continuation holds only this epoch number, never a Writer while waiting for GPU.
    Continuation ObserveContinuation() const noexcept;
    // A granted Writer spans only the immediate backing commit. Defer leaves the callback
    // for later rescheduling; the normal native retry may change protection during Closing.
    WriteAttempt TryEnterWriter(Epoch epoch) noexcept;
    bool BeginClose(Epoch epoch, u32 native_core_mask) noexcept;

    // These acknowledgements follow the real mapping/NCE/journal operations in the coordinator.
    bool ConfirmReprotected(Epoch epoch) noexcept;
    // A core acknowledges after leaving native stores following reprotection. Host commits
    // use Writer guards separately, including cores currently waiting for a GPU callback.
    bool AcknowledgeNativeCore(Epoch epoch, u32 core) noexcept;
    bool ConfirmNativeQuiescent(Epoch epoch) noexcept;
    // Claim only after admitted writers have left, then drain the fixed journal prefix/readbacks.
    bool TryClaimDrain(Epoch epoch) noexcept;
    bool ConfirmJournalDrained(Epoch epoch) noexcept;
    bool TryClaimCapture(Epoch epoch) noexcept;
    bool CompleteClose(Epoch epoch, bool upload_recorded) noexcept;
    Snapshot Inspect() const noexcept;

private:
    static constexpr u64 WriterMask = (u64{1} << 20) - 1;
    static constexpr u64 StateMask = u64{3} << 20;
    static constexpr u64 Reprotected = u64{1} << 22;
    static constexpr u64 NativeQuiescent = u64{1} << 23;
    static constexpr u64 JournalDrained = u64{1} << 24;
    static constexpr u64 CaptureClaimed = u64{1} << 25;
    static constexpr u64 DrainClaimed = u64{1} << 26;
    static constexpr u64 CapturePrerequisites = Reprotected | NativeQuiescent | JournalDrained;
    static constexpr u64 GenerationMask = ~u64{0} << 32;
    static PageState State(u64 value) noexcept;
    static u64 StateBits(PageState state) noexcept;
    static u32 Generation(u64 value) noexcept;
    static WriteDisposition Disposition(u64 value) noexcept;
    bool Confirm(Epoch epoch, u64 flag, u64 prerequisites = 0) noexcept;
    bool StartClosing(Epoch epoch, PageState expected, u32 native_core_mask) noexcept;
    std::atomic<u64> control{};
    std::atomic<u64> native_pending{};
};

} // namespace VideoCommon::CpuEpochs
