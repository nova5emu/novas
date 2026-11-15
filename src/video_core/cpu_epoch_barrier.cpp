// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <limits>
#include <utility>
#include "video_core/cpu_epoch_barrier.h"

namespace VideoCommon::CpuEpochs {

static_assert(std::atomic<u64>::is_always_lock_free);

PageBarrier::Writer::~Writer() {
    Release();
}
PageBarrier::Writer::Writer(Writer&& other) noexcept
    : owner{std::exchange(other.owner, nullptr)}, released{std::move(other.released)} {}
PageBarrier::Writer& PageBarrier::Writer::operator=(Writer&& other) noexcept {
    if (this != &other) {
        Release();
        owner = std::exchange(other.owner, nullptr);
        released = std::move(other.released);
    }
    return *this;
}
void PageBarrier::Writer::Release() noexcept {
    if (auto* barrier = std::exchange(owner, nullptr)) {
        auto notify = std::move(released);
        barrier->control.fetch_sub(1, std::memory_order_release);
        if (notify) {
            notify();
        }
    }
}
PageState PageBarrier::State(u64 value) noexcept {
    return static_cast<PageState>((value & StateMask) >> 20);
}
u64 PageBarrier::StateBits(PageState state) noexcept {
    return static_cast<u64>(state) << 20;
}
u32 PageBarrier::Generation(u64 value) noexcept {
    return static_cast<u32>(value >> 32);
}
WriteDisposition PageBarrier::Disposition(u64 value) noexcept {
    switch (State(value)) {
    case PageState::Open:
        return WriteDisposition::Granted;
    case PageState::GpuOwned:
        return WriteDisposition::GpuOwned;
    default:
        return WriteDisposition::Defer;
    }
}
std::optional<Epoch> PageBarrier::BeginOpen() noexcept {
    auto value = control.load(std::memory_order_acquire);
    if (State(value) != PageState::GpuOwned || Generation(value) == std::numeric_limits<u32>::max()) {
        return std::nullopt;
    }
    const Epoch epoch{Generation(value) + 1};
    const u64 desired = (u64{epoch.generation} << 32) | StateBits(PageState::Opening);
    if (!control.compare_exchange_strong(value, desired, std::memory_order_acq_rel)) {
        return std::nullopt;
    }
    return epoch;
}
bool PageBarrier::CompleteOpen(Epoch epoch) noexcept {
    auto value = control.load(std::memory_order_acquire);
    if (Generation(value) != epoch.generation || State(value) != PageState::Opening) {
        return false;
    }
    const u64 desired = (value & GenerationMask) | StateBits(PageState::Open);
    return control.compare_exchange_strong(value, desired, std::memory_order_acq_rel);
}
PageBarrier::Continuation PageBarrier::ObserveContinuation() const noexcept {
    const auto value = control.load(std::memory_order_acquire);
    return {Disposition(value), Epoch{Generation(value)}};
}
PageBarrier::WriteAttempt PageBarrier::TryEnterWriter(Epoch epoch) noexcept {
    auto value = control.load(std::memory_order_acquire);
    if (Generation(value) != epoch.generation) {
        return {WriteDisposition::Stale, {}};
    }
    if (const auto disposition = Disposition(value); disposition != WriteDisposition::Granted) {
        return {disposition, {}};
    }
    if ((value & WriterMask) == WriterMask) {
        return {WriteDisposition::Full, {}};
    }
    if (!control.compare_exchange_strong(value, value + 1, std::memory_order_acq_rel)) {
        return {WriteDisposition::Contended, {}};
    }
    return {WriteDisposition::Granted, Writer{this}};
}
bool PageBarrier::BeginClose(Epoch epoch, u32 native_core_mask) noexcept {
    return StartClosing(epoch, PageState::Open, native_core_mask);
}
bool PageBarrier::CancelOpen(Epoch epoch, u32 native_core_mask) noexcept {
    return StartClosing(epoch, PageState::Opening, native_core_mask);
}
bool PageBarrier::StartClosing(Epoch epoch, PageState expected, u32 native_core_mask) noexcept {
    auto value = control.load(std::memory_order_acquire);
    if (Generation(value) != epoch.generation || State(value) != expected) {
        return false;
    }
    const u64 desired = (value & (GenerationMask | WriterMask)) | StateBits(PageState::Closing);
    if (!control.compare_exchange_strong(value, desired, std::memory_order_acq_rel)) {
        return false;
    }
    native_pending.store((u64{epoch.generation} << 32) | native_core_mask, std::memory_order_release);
    return true;
}
bool PageBarrier::Confirm(Epoch epoch, u64 flag, u64 prerequisites) noexcept {
    auto value = control.load(std::memory_order_acquire);
    if (Generation(value) != epoch.generation || State(value) != PageState::Closing ||
        (value & prerequisites) != prerequisites) {
        return false;
    }
    return control.compare_exchange_strong(value, value | flag, std::memory_order_acq_rel);
}
bool PageBarrier::ConfirmReprotected(Epoch epoch) noexcept {
    return Confirm(epoch, Reprotected);
}
bool PageBarrier::AcknowledgeNativeCore(Epoch epoch, u32 core) noexcept {
    const auto value = control.load(std::memory_order_acquire);
    if (core >= 32 || Generation(value) != epoch.generation ||
        State(value) != PageState::Closing || !(value & Reprotected)) {
        return false;
    }
    auto pending = native_pending.load(std::memory_order_acquire);
    if (Generation(pending) != epoch.generation) {
        return false;
    }
    const auto desired = pending & ~(u64{1} << core);
    return native_pending.compare_exchange_strong(pending, desired, std::memory_order_acq_rel);
}
bool PageBarrier::ConfirmNativeQuiescent(Epoch epoch) noexcept {
    const auto pending = native_pending.load(std::memory_order_acquire);
    if (Generation(pending) != epoch.generation || static_cast<u32>(pending) != 0) {
        return false;
    }
    return Confirm(epoch, NativeQuiescent, Reprotected);
}
bool PageBarrier::TryClaimDrain(Epoch epoch) noexcept {
    auto value = control.load(std::memory_order_acquire);
    constexpr u64 prerequisites = Reprotected | NativeQuiescent;
    if (Generation(value) != epoch.generation || State(value) != PageState::Closing ||
        (value & WriterMask) || (value & prerequisites) != prerequisites || (value & DrainClaimed)) {
        return false;
    }
    return control.compare_exchange_strong(value, value | DrainClaimed, std::memory_order_acq_rel);
}
bool PageBarrier::ConfirmJournalDrained(Epoch epoch) noexcept {
    return Confirm(epoch, JournalDrained, DrainClaimed);
}
bool PageBarrier::TryClaimCapture(Epoch epoch) noexcept {
    auto value = control.load(std::memory_order_acquire);
    if (Generation(value) != epoch.generation || State(value) != PageState::Closing ||
        (value & WriterMask) || (value & CapturePrerequisites) != CapturePrerequisites ||
        (value & CaptureClaimed)) {
        return false;
    }
    return control.compare_exchange_strong(value, value | CaptureClaimed, std::memory_order_acq_rel);
}
bool PageBarrier::CompleteClose(Epoch epoch, bool upload_recorded) noexcept {
    if (!upload_recorded) {
        return false;
    }
    auto value = control.load(std::memory_order_acquire);
    if (Generation(value) != epoch.generation || State(value) != PageState::Closing ||
        !(value & CaptureClaimed)) {
        return false;
    }
    return control.compare_exchange_strong(value, value & GenerationMask, std::memory_order_acq_rel);
}
PageBarrier::Snapshot PageBarrier::Inspect() const noexcept {
    const auto value = control.load(std::memory_order_acquire);
    return {State(value), Epoch{Generation(value)}, static_cast<u32>(value & WriterMask),
            (value & Reprotected) != 0, (value & NativeQuiescent) != 0,
            (value & DrainClaimed) != 0,
            (value & JournalDrained) != 0, (value & CaptureClaimed) != 0};
}

} // namespace VideoCommon::CpuEpochs
