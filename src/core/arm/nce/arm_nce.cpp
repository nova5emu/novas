// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cinttypes>
#include <memory>

#include "common/signal_chain.h"
#include "common/guest_thread_profile.h"
#include "common/guest_handoff_profile.h"
#include "core/arm/nce/arm_nce.h"
#include "core/arm/nce/interpreter_visitor.h"
#include "core/arm/nce/nce_fault_stats.h"
#include "core/arm/nce/patcher.h"
#include "core/core.h"
#include "core/memory.h"

#include "core/hle/kernel/k_process.h"

#ifdef HAS_NCE
#include <cstdlib>
#include <optional>

#include "common/page_table.h"
#include "core/arm/nce/nce_fault_policy.h"
#include "core/arm/nce/nce_store_decoder.h"
#include "core/deferred_write_config.h"
#include "core/device_memory.h"
#include "video_core/gpu.h"
#include "video_core/nce_page_usage_profile.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/host1x/host1x.h"
#endif

#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>

namespace Core {

namespace {

struct sigaction g_orig_bus_action;
struct sigaction g_orig_segv_action;

// Verify assembly offsets.
using NativeExecutionParameters = Kernel::KThread::NativeExecutionParameters;
static_assert(offsetof(NativeExecutionParameters, native_context) == TpidrEl0NativeContext);
static_assert(offsetof(NativeExecutionParameters, lock) == TpidrEl0Lock);
static_assert(offsetof(NativeExecutionParameters, magic) == TpidrEl0TlsMagic);

fpsimd_context* GetFloatingPointState(mcontext_t& host_ctx) {
    _aarch64_ctx* header = reinterpret_cast<_aarch64_ctx*>(&host_ctx.__reserved);
    while (header->magic != FPSIMD_MAGIC) {
        header = reinterpret_cast<_aarch64_ctx*>(reinterpret_cast<char*>(header) + header->size);
    }
    return reinterpret_cast<fpsimd_context*>(header);
}

using namespace Common::Literals;
constexpr u32 StackSize = 128_KiB;

#ifdef HAS_NCE
bool ExactStoresEnabled() {
    const char* value = std::getenv("EDEN_NX2_NCE_EXACT_STORES");
    if (value) {
        return value[0] != '0';
    }
#ifdef __ANDROID__
    return true;
#else
    return false;
#endif
}

std::optional<u64> AccessFaultEsr(const mcontext_t& context) {
    const auto* cursor = reinterpret_cast<const u8*>(context.__reserved);
    const auto* const end = cursor + sizeof(context.__reserved);
    while (static_cast<std::size_t>(end - cursor) >= sizeof(_aarch64_ctx)) {
        const auto* header = reinterpret_cast<const _aarch64_ctx*>(cursor);
        if (header->magic == 0 || header->size < sizeof(_aarch64_ctx) ||
            header->size > static_cast<std::size_t>(end - cursor) || (header->size & 15) != 0) {
            break;
        }
        if (header->magic == ESR_MAGIC) {
            if (header->size >= sizeof(esr_context)) {
                return reinterpret_cast<const esr_context*>(header)->esr;
            }
            break;
        }
        cursor += header->size;
    }
    return std::nullopt;
}

u8* CachedBacking(const Common::PageTable& table, u64 address) {
    const u64 page = address >> Memory::YUZU_PAGEBITS;
    if (page >= table.entries.size()) {
        return nullptr;
    }
    const auto raw = table.entries[page].Raw();
    if (static_cast<Common::PageType>(raw.type) != Common::PageType::RasterizerCachedMemory) {
        return nullptr;
    }
    const auto pointer = Common::PageTable::PageEntryData::ExtractPointer(raw, true);
    return pointer ? reinterpret_cast<u8*>(pointer + address) : nullptr;
}
#endif

} // namespace

void* ArmNce::RestoreGuestContext(void* raw_context) {
    // Retrieve the host context.
    auto& host_ctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;

    // Thread-local parameters will be located in x9.
    auto* tpidr = reinterpret_cast<NativeExecutionParameters*>(host_ctx.regs[9]);
    auto* guest_ctx = static_cast<GuestContext*>(tpidr->native_context);

    // Retrieve the host floating point state.
    auto* fpctx = GetFloatingPointState(host_ctx);

    // Save host callee-saved registers.
    std::memcpy(guest_ctx->host_ctx.host_saved_vregs.data(), &fpctx->vregs[8],
                sizeof(guest_ctx->host_ctx.host_saved_vregs));
    std::memcpy(guest_ctx->host_ctx.host_saved_regs.data(), &host_ctx.regs[19],
                sizeof(guest_ctx->host_ctx.host_saved_regs));

    // Save stack pointer.
    guest_ctx->host_ctx.host_sp = host_ctx.sp;

    // Restore all guest state except tpidr_el0.
    host_ctx.sp = guest_ctx->sp;
    host_ctx.pc = guest_ctx->pc;
    host_ctx.pstate = guest_ctx->pstate;
    fpctx->fpcr = guest_ctx->fpcr;
    fpctx->fpsr = guest_ctx->fpsr;
    std::memcpy(host_ctx.regs, guest_ctx->cpu_registers.data(), sizeof(host_ctx.regs));
    std::memcpy(fpctx->vregs, guest_ctx->vector_registers.data(), sizeof(fpctx->vregs));

    // Signal-safe timestamp only. The normal host continuation publishes the sample.
    static_assert(std::atomic<u64>::is_always_lock_free);
    if (tpidr->handoff_entry_ns.load(std::memory_order_relaxed) == 1) {
        timespec now{};
        if (clock_gettime(CLOCK_MONOTONIC, &now) == 0) {
            tpidr->handoff_entry_frame.store(Common::GuestHandoffProfile::CurrentFrame(),
                                            std::memory_order_relaxed);
            tpidr->handoff_entry_ns.store(static_cast<u64>(now.tv_sec) * 1'000'000'000 +
                                            static_cast<u64>(now.tv_nsec),
                                        std::memory_order_release);
        }
    }

    // Return the new thread-local storage pointer.
    return tpidr;
}

void ArmNce::SaveGuestContext(GuestContext* guest_ctx, void* raw_context) {
    // Retrieve the host context.
    auto& host_ctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;

    // Retrieve the host floating point state.
    auto* fpctx = GetFloatingPointState(host_ctx);

    // Save all guest registers except tpidr_el0.
    std::memcpy(guest_ctx->cpu_registers.data(), host_ctx.regs, sizeof(host_ctx.regs));
    std::memcpy(guest_ctx->vector_registers.data(), fpctx->vregs, sizeof(fpctx->vregs));
    guest_ctx->fpsr = fpctx->fpsr;
    guest_ctx->fpcr = fpctx->fpcr;
    guest_ctx->pstate = static_cast<u32>(host_ctx.pstate);
    guest_ctx->pc = host_ctx.pc;
    guest_ctx->sp = host_ctx.sp;

    // Restore stack pointer.
    host_ctx.sp = guest_ctx->host_ctx.host_sp;

    // Restore host callee-saved registers.
    std::memcpy(&host_ctx.regs[19], guest_ctx->host_ctx.host_saved_regs.data(),
                sizeof(guest_ctx->host_ctx.host_saved_regs));
    std::memcpy(&fpctx->vregs[8], guest_ctx->host_ctx.host_saved_vregs.data(),
                sizeof(guest_ctx->host_ctx.host_saved_vregs));

    // Return from the call on exit by setting pc to x30.
    host_ctx.pc = guest_ctx->host_ctx.host_saved_regs[11];

    // Clear esr_el1 and return it.
    host_ctx.regs[0] = guest_ctx->esr_el1.exchange(0);
}

bool ArmNce::HandleFailedGuestFault(GuestContext* guest_ctx, void* raw_info, void* raw_context) {
    auto& host_ctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;
    auto* info = static_cast<siginfo_t*>(raw_info);

    // We can't handle the access, so determine why we crashed.
    const bool is_prefetch_abort = host_ctx.pc == reinterpret_cast<u64>(info->si_addr);

    // For data aborts, skip the instruction and return to guest code.
    // This will allow games to continue in many scenarios where they would otherwise crash.
    if (!is_prefetch_abort) {
        host_ctx.pc += 4;
        return true;
    }

    // This is a prefetch abort.
    guest_ctx->esr_el1.fetch_or(static_cast<u64>(HaltReason::PrefetchAbort));

    // Forcibly mark the context as locked. We are still running.
    // We may race with SignalInterrupt here:
    // - If we lose the race, then SignalInterrupt will send us a signal we are masking,
    //   and it will do nothing when it is unmasked, as we have already left guest code.
    // - If we win the race, then SignalInterrupt will wait for us to unlock first.
    auto& thread_params = guest_ctx->parent->m_running_thread->GetNativeExecutionParameters();
    thread_params.lock.store(SpinLockLocked);

    // Return to host.
    SaveGuestContext(guest_ctx, raw_context);
    return false;
}

bool ArmNce::HandleGuestAlignmentFault(GuestContext* guest_ctx, void* raw_info, void* raw_context) {
    auto& host_ctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;
#ifdef HAS_NCE
    auto& nce = *guest_ctx->parent;
    if (nce.m_deferred_fallbacks_enabled) {
        const u64 fault_address = reinterpret_cast<u64>(static_cast<siginfo_t*>(raw_info)->si_addr);
        if (nce.ConsumeNativeRetry(host_ctx.pc, fault_address, true)) {
            // The host decoder rejected this instruction. Preserve the old failure handling
            // without decoding again in the signal (the guest may have changed the opcode).
            return HandleFailedGuestFault(guest_ctx, raw_info, raw_context);
        }
        nce.m_pending_store = {fault_address, 0, true, true};
        nce.m_store_pending.store(true, std::memory_order_release);
        nce.m_running_thread->GetNativeExecutionParameters().lock.store(SpinLockLocked);
        SaveGuestContext(guest_ctx, raw_context);
        return false;
    }
#endif
    auto* fpctx = GetFloatingPointState(host_ctx);
    auto& memory = guest_ctx->parent->m_running_thread->GetOwnerProcess()->GetMemory();

    // Match and execute an instruction.
    auto next_pc = MatchAndExecuteOneInstruction(memory, &host_ctx, fpctx);
    if (next_pc) {
        host_ctx.pc = *next_pc;
        return true;
    }

    // We couldn't handle the access.
    return HandleFailedGuestFault(guest_ctx, raw_info, raw_context);
}

bool ArmNce::HandleGuestAccessFault(GuestContext* guest_ctx, void* raw_info, void* raw_context) {
    auto* info = static_cast<siginfo_t*>(raw_info);

    // NovaS diagnostics: was it a read or which kind of write (InvalidateNCE counts it when the
    // page is the GPU's cached memory). WnR comes from the kernel's ESR record.
    {
        auto& host_ctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;
        bool is_write = false;
        auto* header = reinterpret_cast<_aarch64_ctx*>(&host_ctx.__reserved);
        for (int i = 0; i < 16 && header->magic != 0; ++i) {
            if (header->magic == ESR_MAGIC) {
                is_write = (reinterpret_cast<esr_context*>(header)->esr & (1ULL << 6)) != 0;
                break;
            }
            header = reinterpret_cast<_aarch64_ctx*>(reinterpret_cast<char*>(header) + header->size);
        }
        const u32 inst = *reinterpret_cast<const u32*>(host_ctx.pc);
        NceFaultStats::last_class = NceFaultStats::Classify(inst, is_write);
    }

#ifdef HAS_NCE
    if (DeferExactStore(guest_ctx, raw_info, raw_context)) {
        return false;
    }
#endif

    // Try to handle an invalid access.
    // TODO: handle accesses which split a page?
    const Common::ProcessAddress addr =
        (reinterpret_cast<u64>(info->si_addr) & ~Memory::YUZU_PAGEMASK);
    auto& memory = guest_ctx->parent->m_running_thread->GetOwnerProcess()->GetMemory();
    if (memory.InvalidateNCE(addr, Memory::YUZU_PAGESIZE)) {
        // We handled the access successfully and are returning to guest code.
        return true;
    }

    // We couldn't handle the access.
    return HandleFailedGuestFault(guest_ctx, raw_info, raw_context);
}

#ifdef HAS_NCE
bool ArmNce::ConsumeNativeRetry(u64 pc, u64 fault_address, bool alignment_failure) {
    if (!m_retry_store.exchange(false, std::memory_order_acquire)) {
        return false;
    }
    if (m_retry_store_thread == m_running_thread && m_retry_store_pc == pc &&
        m_retry_store_address == fault_address && m_retry_alignment_failure == alignment_failure) {
        NceExactStore::retry_bypassed.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    NceExactStore::retry_mismatch.fetch_add(1, std::memory_order_relaxed);
    return false;
}

bool ArmNce::DeferExactStore(GuestContext* ctx, void* raw_info, void* raw_context) {
    using NceExactStore::CountFallback;
    using NceExactStore::Result;
    auto& nce = *ctx->parent;
    const auto& info = *static_cast<const siginfo_t*>(raw_info);
    const auto& context = static_cast<const ucontext_t*>(raw_context)->uc_mcontext;
    const u64 fault_address = reinterpret_cast<u64>(info.si_addr);
    if (nce.ConsumeNativeRetry(context.pc, fault_address, false)) {
        return false;
    }
    const auto& table = nce.m_running_thread->GetOwnerProcess()->GetPageTable().GetImpl();
    const u64 page = fault_address & ~Memory::YUZU_PAGEMASK;
    auto* backing = CachedBacking(table, page);
    if (!backing) {
        return false;
    }
    const auto defer_to_host = [&](bool fallback_only, Result reason = Result::Success) {
        nce.m_pending_store = {fault_address, reinterpret_cast<u64>(backing), fallback_only, false, reason};
        nce.m_store_pending.store(true, std::memory_order_release);
        // Follow the existing fault-to-host protocol, including the interrupt handshake.
        nce.m_running_thread->GetNativeExecutionParameters().lock.store(SpinLockLocked);
        SaveGuestContext(ctx, raw_context);
        return true;
    };
    if (!nce.m_exact_stores_enabled && !nce.m_deferred_fallbacks_enabled) {
        CountFallback(Result::Disabled);
        return false;
    }
    const auto esr = AccessFaultEsr(context);
    if (!esr) {
        CountFallback(Result::MissingEsr);
        return nce.m_deferred_fallbacks_enabled && defer_to_host(true, Result::MissingEsr);
    }
    // Android can report a translation FSC for a cached, protected page. B revalidates its
    // CPU/device identity in the host continuation; the old exact-store path stays unchanged.
    const auto decision = NceFaultPolicy::Classify(info.si_code == SEGV_ACCERR, *esr,
        backing != nullptr, context.pc == fault_address, nce.m_deferred_fallbacks_enabled);
    if (decision == NceFaultPolicy::Decision::Unsupported) {
        using NceExactStore::FaultReason;
        const auto reason = info.si_code != SEGV_ACCERR ? FaultReason::SignalCode :
            ((*esr >> 26) & 0x3f) != 0x24 ? FaultReason::ExceptionClass :
            (*esr & 0x3c) != 0x0c ? FaultReason::FaultStatus : FaultReason::PcEqualsAddress;
        NceExactStore::unsupported_reasons[static_cast<std::size_t>(reason)]
            .fetch_add(1, std::memory_order_relaxed);
        NceExactStore::unsupported_ec[(*esr >> 26) & 0x3f].fetch_add(1, std::memory_order_relaxed);
        NceExactStore::unsupported_fsc[*esr & 0x3f].fetch_add(1, std::memory_order_relaxed);
        NceExactStore::unsupported_wnr[(*esr >> 6) & 1].fetch_add(1, std::memory_order_relaxed);
        NceExactStore::unsupported_si_code[info.si_code == SEGV_ACCERR ? 0 :
                                          info.si_code == SEGV_MAPERR ? 1 : 2]
            .fetch_add(1, std::memory_order_relaxed);
        CountFallback(Result::UnsupportedFault);
        return nce.m_deferred_fallbacks_enabled && defer_to_host(true, Result::UnsupportedFault);
    }
    if (decision == NceFaultPolicy::Decision::Read) {
        CountFallback(Result::Read);
        return nce.m_deferred_fallbacks_enabled && defer_to_host(true, Result::Read);
    }
    // A page the guest keeps writing: one download (a Finish, ~2.6 ms on the phone) leaves it
    // writable for every store that follows, while each emulated store costs ~150 us. Past
    // StoreBudget stores to the same page within a 16 ms window, take the download instead.
    if (!nce.m_deferred_fallbacks_enabled && !NceExactStore::WithinPageBudget(page)) {
        CountFallback(Result::PageBudget);
        return false;
    }

    return defer_to_host(false);
}

void ArmNce::CompleteExactStore(Kernel::KThread* thread) {
    using NceExactStore::Result;
    namespace Profile = Common::GpuTransitionProfile;
    const Profile::WatchdogCoreScope host_phase{m_core_index, Profile::WatchdogPhase::HostContinuation};
    auto& process = *thread->GetOwnerProcess();
    const auto& table = process.GetPageTable().GetImpl();
    const u64 fault_address = m_pending_store.fault_address;
    if (!m_deferred_runtime_logged && Common::GuestThreadProfile::Enabled()) {
        m_deferred_runtime_logged = true;
        LOG_INFO(Core_ARM, "GPU_DEFERRED_RUNTIME core={} producer_active={} exact_stores_active={} journal_present={}",
                 m_core_index, m_deferred_fallbacks_enabled, m_exact_stores_enabled,
                 static_cast<bool>(m_system.DeferredWriteJournal()));
    }
    const bool writer_diagnostics = Common::GpuTransitionProfile::enabled.load(std::memory_order_relaxed);
    if (writer_diagnostics || VideoCommon::NcePageUsageProfile::Enabled()) {
        const Profile::WatchdogCoreScope inspection_phase{m_core_index, Profile::WatchdogPhase::WriterInspection};
        namespace Profile = Common::GpuTransitionProfile;
        bool physical_resolved{};
        bool device_resolved{};
        const u64 page = fault_address >> Memory::YUZU_PAGEBITS;
        if (page < table.entries.size()) {
            const auto pointer = table.entries[page].Pointer(true);
            const auto* backing = reinterpret_cast<const u8*>(pointer + fault_address);
            auto& device_memory = m_system.Host1x().MemoryManager();
            if (pointer && device_memory.IsPhysicalPointer(backing)) {
                physical_resolved = true;
                const PAddr physical_page =
                    device_memory.GetRawPhysicalAddr(backing) & ~PAddr{Memory::YUZU_PAGEMASK};
                VideoCommon::NcePageUsageProfile::Fault(physical_page, m_guest_ctx.pc,
                    thread->GetThreadId(), static_cast<u32>(m_core_index));
                if (writer_diagnostics) {
                    std::optional<DAddr> device_page;
                    device_resolved = device_memory.TryApplyOpOnSingleMapping(backing, 1, [&](DAddr address) {
                        device_page = address & ~DAddr{Memory::YUZU_PAGEMASK};
                    });
                    Profile::RecordLegacyNceFault(physical_page);
                    if (const auto journal = m_system.DeferredWriteJournal()) {
                        const auto writers = journal->InspectWriterExclusions(physical_page, device_page);
                        Profile::RecordWriterNceFault(physical_page, writers.origin_mask,
                                                     writers.first_reason_mask,
                                                     writers.complete && device_resolved);
                    }
                }
            }
        }
        if (!physical_resolved) {
            VideoCommon::NcePageUsageProfile::UnresolvedFault();
        }
        if (writer_diagnostics) {
            Profile::RecordWriterNceCoverage(physical_resolved, device_resolved);
        }
    }
    if (m_deferred_fallbacks_enabled) {
        const Profile::WatchdogCoreScope recovery_phase{m_core_index, Profile::WatchdogPhase::Recovery};
        // Recovery must happen before preparing any legacy fallback, and never from the signal.
        const Common::GpuTransitionProfile::DeferredProducerScope phase{
            Common::GpuTransitionProfile::DeferredProducerPhase::Recovery};
        const bool recovered = m_system.GPU().RecoverAbandonedDeferredWritesWithoutLocks();
        (void)recovered;
    }
    if (m_pending_store.alignment) {
        VideoCommon::NcePageUsageProfile::UnresolvedCpuWrite();
        Common::GpuTransitionProfile::RecordDeferredFallback(Result::Unaligned);
        const auto next_pc = MatchAndExecuteOneInstruction(
            process.GetMemory(), m_guest_ctx.cpu_registers, m_guest_ctx.vector_registers,
            m_guest_ctx.sp, m_guest_ctx.pc, true);
        if (next_pc) {
            m_guest_ctx.pc = *next_pc;
        } else {
            ArmNativeRetry(thread, fault_address, true);
        }
        return;
    }
    if (m_pending_store.fallback_only) {
        VideoCommon::NcePageUsageProfile::UnresolvedCpuWrite();
        Common::GpuTransitionProfile::RecordDeferredFallback(m_pending_store.fallback_reason);
        RetryNativeAccess(thread, fault_address, m_pending_store.fallback_reason);
        return;
    }
    std::optional<PAddr> budget_excluded_page;
    std::optional<Result> budget_rejection;
    std::optional<PAddr> profiled_physical;
    bool native_epoch_opened = false;
    NceExactStore::Store store{};
    const Result result = [&]() -> Result {
        const Profile::WatchdogCoreScope decode_phase{m_core_index, Profile::WatchdogPhase::Decode};
        const u64 pc = m_guest_ctx.pc;
        if ((pc & 3) != 0 || (pc >> Memory::YUZU_PAGEBITS) >= table.entries.size()) {
            NceExactStore::opcode_read_failed.fetch_add(1, std::memory_order_relaxed);
            return Result::UnsupportedInstruction;
        }
        const auto [code, code_type, code_block] =
            table.entries[pc >> Memory::YUZU_PAGEBITS].PointerTypeBlock();
        if (!code || code_type != Common::PageType::Memory) {
            NceExactStore::opcode_read_failed.fetch_add(1, std::memory_order_relaxed);
            return Result::UnsupportedInstruction;
        }
        u32 instruction{};
        std::memcpy(&instruction, reinterpret_cast<const void*>(code + pc), sizeof(instruction));
        if (NceFaultStats::Classify(instruction, true) == NceFaultStats::StoreOther) {
            NceExactStore::RecordOtherOpcode(instruction);
        }
        u64 dczid_el0 = 0x10;
        if ((instruction & 0xffffffe0U) == 0xd50b7420U) {
            // NCE does not patch DCZID_EL0 or DC ZVA: use the same register as native guest code.
            // This read and all opcode accounting happen after leaving the signal handler.
            asm volatile("mrs %0, dczid_el0" : "=r"(dczid_el0));
            NceExactStore::dc_zva_detected.fetch_add(1, std::memory_order_relaxed);
            NceExactStore::dc_zva_dczid[dczid_el0 & 0x1f].fetch_add(1, std::memory_order_relaxed);
        }
        NceExactStore::StoreVisitor visitor{m_guest_ctx.cpu_registers, m_guest_ctx.sp,
                                            m_guest_ctx.vector_registers, dczid_el0};
        if (const auto decoded = visitor.Decode(instruction, fault_address, store);
            decoded != Result::Success) {
            if (decoded == Result::UnsupportedInstruction) {
                Common::GpuTransitionProfile::RecordRejectedNceOpcode(instruction, pc);
            }
            return decoded;
        }

        // Use the same cached-page authority as InvalidateNCE. The producer revalidates this
        // hint under the CPU identity gate before granting a lease or committing bytes.
        auto* backing = CachedBacking(table, store.address);
        if (!backing || reinterpret_cast<u64>(backing) !=
                            m_pending_store.backing_page + (store.address & Memory::YUZU_PAGEMASK)) {
            return Result::MappingChanged;
        }
        auto& device_memory = m_system.Host1x().MemoryManager();
        if (!device_memory.IsPhysicalPointer(backing)) {
            return Result::MappingChanged;
        }
        if (VideoCommon::NcePageUsageProfile::Enabled()) {
            profiled_physical = device_memory.GetRawPhysicalAddr(backing);
        }
        DAddr device_address{};
        if (!device_memory.ApplyOpOnSingleMapping(backing, store.size,
                                                  [&](DAddr address) { device_address = address; })) {
            return Result::AliasedMapping;
        }
        const NceExactStore::Backing mapping{
            &table, store.address, backing,
            table.entries[store.address >> Memory::YUZU_PAGEBITS].Block(),
            store.zero_block && store.size > 16 ? 16 : store.pair ? store.size / 2 : store.size,
            store.pair, store.zero_block};
        const std::span<const u8> bytes =
            store.zero_block ? std::span{NceExactStore::DcZvaZeros}.first(store.size)
                             : std::span<const u8>{store.bytes}.first(store.size);
        if (m_deferred_fallbacks_enabled && m_system.DeviceMemory().CpuEpochs().Requested() &&
            store.size <= Memory::YUZU_PAGESIZE - (store.address & Memory::YUZU_PAGEMASK) &&
            VideoCommon::NcePageUsageProfile::IsCpuEpochStorageCandidate(
                device_memory.GetRawPhysicalAddr(backing) & ~PAddr{4095})) {
            const PAddr physical = device_memory.GetRawPhysicalAddr(backing) & ~PAddr{4095};
            VideoCommon::CpuEpochs::Identity identity{.memory = &process.GetMemory(),
                                                      .cpu_owner = &table,
                .device_owner = &device_memory,
                                                      .address = store.address & ~VAddr{4095},
                                                      .physical = physical,
                                                      .device = device_address & ~DAddr{4095}};
            const auto proof = m_system.DeviceMemory().CpuMappings().WithPage(
                &table, identity.address, physical,
                [&](const auto& snapshot) { identity.generation = snapshot.generation; });
            if (proof == DeferredCpuMappings::Status::Eligible) {
                identity.lifetime = std::make_shared<Kernel::KScopedAutoObject<Kernel::KProcess>>(
                    m_system.Kernel(), &process);
                if (m_system.GPU().TryOpenCpuEpoch(std::move(identity))) {
                    native_epoch_opened = true;
                    return Result::Success;
                }
            }
        }
        if (m_deferred_fallbacks_enabled) {
            const PAddr physical_page = device_memory.GetRawPhysicalAddr(backing) & ~PAddr{4095};
            using BudgetDecision = DeferredPublicationBudget::Decision;
            const auto decision =
                m_publication_budget.BeforeAttempt(physical_page, NceExactStore::NowNs());
            if (const auto event = m_publication_budget.TakeCoreEvent()) {
                Common::GpuTransitionProfile::RecordBudgetCoreEvent(
                    event->activated, event->slow_samples, event->max_ns, m_core_index);
            }
            switch (decision) {
            case BudgetDecision::PublicationCost: budget_rejection = Result::PublicationCost; break;
            case BudgetDecision::CorePublicationCost: budget_rejection = Result::PublicationCoreCost; break;
            case BudgetDecision::FaultDensity: budget_rejection = Result::FaultDensity; break;
            case BudgetDecision::Allow: break;
            }
            u64 publication_ns{};
            if (budget_rejection) {
                Profile::RecordDeferredFallback(*budget_rejection);
                const Profile::WatchdogCoreScope exact_phase{m_core_index, Profile::WatchdogPhase::BudgetExact};
                const Profile::DeferredProducerScope exact_timing{Profile::DeferredProducerPhase::BudgetExact};
                const auto exact = m_system.GPU().PublishNceStore(
                    m_core_index, device_address, bytes, mapping, publication_ns, true);
                if (Profile::enabled.load(std::memory_order_relaxed)) {
                    Profile::deferred_counters.RecordPhase(exact == Result::Success ?
                        Profile::DeferredProducerPhase::BudgetExactSuccess :
                        Profile::DeferredProducerPhase::BudgetExactFailed, 0);
                }
                if (exact != Result::Success) budget_excluded_page = physical_page;
                return exact;
            }
            const Profile::WatchdogCoreScope publish_phase{m_core_index, Profile::WatchdogPhase::Publish};
            const auto published = m_system.GPU().PublishNceStore(
                m_core_index, device_address, bytes, mapping, publication_ns);
            if (published == Result::Success) {
                m_publication_budget.Observe(physical_page, publication_ns, NceExactStore::NowNs());
                if (const auto event = m_publication_budget.TakeCoreEvent()) {
                    Common::GpuTransitionProfile::RecordBudgetCoreEvent(
                        event->activated, event->slow_samples, event->max_ns, m_core_index);
                }
            }
            // The epoch can have retired before this publication reports its transient exclusion.
            if (published != Result::Success &&
                m_system.DeviceMemory().CpuEpochs().Requested() &&
                (m_system.DeviceMemory().CpuEpochs().FindPage(physical_page) ||
                 VideoCommon::NcePageUsageProfile::IsCpuEpochStorageCandidate(physical_page))) {
                auto exact = mapping;
                exact.allow_clean_page = true;
                return m_system.GPU().PublishNceStore(
                    m_core_index, device_address, bytes, exact, publication_ns, true);
            }
            return published;
        }
        return m_system.GPU().WriteNceStore(device_address, bytes, mapping);
    }();

    if (native_epoch_opened) {
        NceFaultStats::counts[store.zero_block ? NceFaultStats::StoreOther :
                             store.simd ? NceFaultStats::StoreSimd :
                             store.pair ? NceFaultStats::StorePair : NceFaultStats::StoreSingle]
            .fetch_add(1, std::memory_order_relaxed);
        // Resume the faulting instruction unchanged under native RW ownership.
        return;
    }
    if (result == Result::Success) {
        if (profiled_physical) {
            VideoCommon::NcePageUsageProfile::CpuWrite(*profiled_physical, store.size);
        }
        if (store.writeback) {
            if (store.base_register == 31) {
                m_guest_ctx.sp = store.writeback_value;
            } else {
                m_guest_ctx.cpu_registers[store.base_register] = store.writeback_value;
            }
        }
        m_guest_ctx.pc += 4;
        auto& counter = store.zero_block ? NceExactStore::emulated_dc_zva : store.simd ?
            (store.pair ? NceExactStore::emulated_simd_pair : NceExactStore::emulated_simd_single) :
            (store.pair ? NceExactStore::emulated_pair : NceExactStore::emulated_single);
        counter.fetch_add(1, std::memory_order_relaxed);
        if (store.zero_block) {
            NceExactStore::emulated_dc_zva_bytes.fetch_add(store.size, std::memory_order_relaxed);
        }
        NceFaultStats::counts[store.zero_block ? NceFaultStats::StoreOther :
                             store.simd ? NceFaultStats::StoreSimd :
                             store.pair ? NceFaultStats::StorePair : NceFaultStats::StoreSingle]
            .fetch_add(1, std::memory_order_relaxed);
        return;
    }

    VideoCommon::NcePageUsageProfile::UnresolvedCpuWrite();
    NceExactStore::CountFallback(result);
    if (m_deferred_fallbacks_enabled) Common::GpuTransitionProfile::RecordDeferredFallback(result);
    RetryNativeAccess(thread, fault_address, result);
    if (budget_excluded_page) {
        Common::GpuTransitionProfile::RecordBudgetExcludedPage(
            *budget_rejection,
            *budget_excluded_page);
    }
}

void ArmNce::RetryNativeAccess(Kernel::KThread* thread, u64 fault_address,
                             Common::GpuTransitionProfile::DeferredFallbackReason reason) {
    const Common::GpuTransitionProfile::WatchdogCoreScope native_phase{
        m_core_index, Common::GpuTransitionProfile::WatchdogPhase::NativeFallback};
    if (m_deferred_fallbacks_enabled) {
        const Common::GpuTransitionProfile::DeferredProducerScope phase{
            Common::GpuTransitionProfile::DeferredProducerPhase::NativeFallback};
        // Host continuation only. Keep both this VA and its physical identities excluded
        // across the retry and subsequent writable accesses, including remaps and aliases.
        thread->GetOwnerProcess()->GetMemory().PrepareNativeFallback(
            fault_address & ~Memory::YUZU_PAGEMASK, Memory::YUZU_PAGESIZE, reason);
    }
    ArmNativeRetry(thread, fault_address, false);
}

void ArmNce::ArmNativeRetry(Kernel::KThread* thread, u64 fault_address, bool alignment_failure) {
    // No guest state was changed. Let the next fault run the original handler exactly once,
    // including its unmapped-memory handling, without sending it around this path again.
    m_retry_store_thread = thread;
    m_retry_store_pc = m_guest_ctx.pc;
    m_retry_store_address = fault_address;
    m_retry_alignment_failure = alignment_failure;
    m_retry_store.store(true, std::memory_order_release);
}
#endif

void ArmNce::HandleHostAlignmentFault(int sig, void* raw_info, void* raw_context) {
    return g_orig_bus_action.sa_sigaction(sig, static_cast<siginfo_t*>(raw_info), raw_context);
}

void ArmNce::HandleHostAccessFault(int sig, void* raw_info, void* raw_context) {
    return g_orig_segv_action.sa_sigaction(sig, static_cast<siginfo_t*>(raw_info), raw_context);
}

void ArmNce::LockThread(Kernel::KThread* thread) {
    auto* thread_params = &thread->GetNativeExecutionParameters();
    LockThreadParameters(thread_params);
}

void ArmNce::UnlockThread(Kernel::KThread* thread) {
    auto* thread_params = &thread->GetNativeExecutionParameters();
    m_guest_ctx.tpidr_el0 = thread_params->tpidr_el0;
    m_guest_ctx.tpidrro_el0 = thread_params->tpidrro_el0;
    thread_params->native_context = nullptr;
    UnlockThreadParameters(thread_params);
}

HaltReason ArmNce::RunThread(Kernel::KThread* thread) {
    thread->ProfileHandoffNceRun(m_core_index);
    // Check if we're already interrupted.
    // If we are, we can just return immediately.
    HaltReason hr = static_cast<HaltReason>(m_guest_ctx.esr_el1.exchange(0));
    if (True(hr)) {
        return hr;
    }

    // Pre-fetch thread context data to improve cache locality
    auto* thread_params = &thread->GetNativeExecutionParameters();
    auto* process = thread->GetOwnerProcess();

    // Move non-critical operations outside the locked section
    const u64 tpidr_el0_cache = m_guest_ctx.tpidr_el0;
    const u64 tpidrro_el0_cache = m_guest_ctx.tpidrro_el0;

    // Critical section begins - minimize operations here
    m_running_thread = thread;
    m_guest_ctx.parent = this;
    thread_params->native_context = &m_guest_ctx;
    thread_params->tpidr_el0 = tpidr_el0_cache;
    thread_params->tpidrro_el0 = tpidrro_el0_cache;

    // Memory barrier to ensure visibility of changes
    std::atomic_thread_fence(std::memory_order_release);
    thread_params->is_running = true;

    // TODO: finding and creating the post handler needs to be locked
    // to deal with dynamic loading of NROs.
    const auto& post_handlers = process->GetPostHandlers();
    const auto it = post_handlers.find(m_guest_ctx.pc);
#ifdef HAS_NCE
    if (m_store_resume_thread) {
        if (m_store_resume_thread == thread) {
            const u64 now = NceExactStore::NowNs();
            NceExactStore::timing_resume.Add(now - m_store_host_done_ns);
            NceExactStore::timing_total.Add(now - m_store_host_start_ns);
        } else {
            NceExactStore::resume_skipped.fetch_add(1, std::memory_order_relaxed);
        }
        m_store_resume_thread = nullptr;
    }
#endif
    const bool signal_entry = it == post_handlers.end() && thread->ProfileHandoffPending();
    if (signal_entry) {
        thread_params->handoff_entry_ns.store(1, std::memory_order_relaxed);
    } else {
        thread->ProfileHandoffEnter(m_core_index);
    }
    if (it != post_handlers.end()) {
        hr = ReturnToRunCodeByTrampoline(thread_params, &m_guest_ctx, it->second);
    } else {
        hr = ReturnToRunCodeByExceptionLevelChange(m_thread_id, thread_params);  // Android: Use "process handle SIGUSR2 -n true -p true -s false" (and SIGURG) in LLDB when debugging
    }

    if (signal_entry) {
        const u64 entry_ns = thread_params->handoff_entry_ns.exchange(0, std::memory_order_acquire);
        if (entry_ns > 1) {
            thread->ProfileHandoffEnter(m_core_index, entry_ns,
                thread_params->handoff_entry_frame.load(std::memory_order_relaxed));
        } else {
            thread->ProfileHandoffCancel();
        }
    }

#ifdef HAS_NCE
    if (m_store_pending.load(std::memory_order_acquire)) {
        m_store_host_start_ns = NceExactStore::NowNs();
    }
#endif
    // Critical section for thread cleanup
    std::atomic_thread_fence(std::memory_order_acquire);

    // Cache values before releasing thread
    const u64 final_tpidr_el0 = thread_params->tpidr_el0;

    // Minimize critical section
    thread_params->is_running = false;
    thread_params->native_context = nullptr;
    m_running_thread = nullptr;

    // Non-critical updates can happen after releasing the thread
    m_guest_ctx.tpidr_el0 = final_tpidr_el0;

    // Return the halt reason.
    return hr;
}

#ifdef HAS_NCE
void ArmNce::CompleteHostOperations(Kernel::KThread* thread) {
    if (!m_store_pending.exchange(false, std::memory_order_acquire)) return;
    CompleteExactStore(thread);
    if (m_system.DebuggerEnabled()) GetContext(thread->GetContext());
    m_store_host_done_ns = NceExactStore::NowNs();
    m_store_resume_thread = thread;
    NceExactStore::timing_host.Add(m_store_host_done_ns - m_store_host_start_ns);
}
#endif

HaltReason ArmNce::StepThread(Kernel::KThread* thread) {
    return HaltReason::StepThread;
}

u32 ArmNce::GetSvcNumber() const {
    return m_guest_ctx.svc;
}

void ArmNce::GetSvcArguments(std::span<uint64_t, 8> args) const {
    for (size_t i = 0; i < 8; i++) {
        args[i] = m_guest_ctx.cpu_registers[i];
    }
}

void ArmNce::SetSvcArguments(std::span<const uint64_t, 8> args) {
    for (size_t i = 0; i < 8; i++) {
        m_guest_ctx.cpu_registers[i] = args[i];
    }
}

ArmNce::ArmNce(System& system, bool uses_wall_clock, std::size_t core_index)
    : ArmInterface{uses_wall_clock}, m_system{system}, m_core_index{core_index} {
    m_guest_ctx.system = &m_system;
#ifdef HAS_NCE
    m_exact_stores_enabled = ExactStoresEnabled();
    m_deferred_fallbacks_enabled = DeferredWritesConfig::ProducerRequested();
#endif
}

ArmNce::~ArmNce() = default;

void ArmNce::Initialize() {
    if (m_thread_id == -1) {
        m_thread_id = gettid();
    }

    // Configure signal stack.
    if (!m_stack) {
        m_stack = std::make_unique<u8[]>(StackSize);

        stack_t ss{};
        ss.ss_sp = m_stack.get();
        ss.ss_size = StackSize;
        sigaltstack(&ss, nullptr);
    }

    // Set up signals.
    static std::once_flag flag;
    std::call_once(flag, [] {
        using HandlerType = decltype(sigaction::sa_sigaction);

        sigset_t signal_mask;
        sigemptyset(&signal_mask);
        sigaddset(&signal_mask, ReturnToRunCodeByExceptionLevelChangeSignal);
        sigaddset(&signal_mask, BreakFromRunCodeSignal);
        sigaddset(&signal_mask, GuestAlignmentFaultSignal);
        sigaddset(&signal_mask, GuestAccessFaultSignal);

        struct sigaction return_to_run_code_action {};
        return_to_run_code_action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        return_to_run_code_action.sa_sigaction = reinterpret_cast<HandlerType>(
            &ArmNce::ReturnToRunCodeByExceptionLevelChangeSignalHandler);
        return_to_run_code_action.sa_mask = signal_mask;
        Common::SigAction(ReturnToRunCodeByExceptionLevelChangeSignal, &return_to_run_code_action,
                          nullptr);

        struct sigaction break_from_run_code_action {};
        break_from_run_code_action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        break_from_run_code_action.sa_sigaction =
            reinterpret_cast<HandlerType>(&ArmNce::BreakFromRunCodeSignalHandler);
        break_from_run_code_action.sa_mask = signal_mask;
        Common::SigAction(BreakFromRunCodeSignal, &break_from_run_code_action, nullptr);

        struct sigaction alignment_fault_action {};
        alignment_fault_action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        alignment_fault_action.sa_sigaction =
            reinterpret_cast<HandlerType>(&ArmNce::GuestAlignmentFaultSignalHandler);
        alignment_fault_action.sa_mask = signal_mask;
        Common::SigAction(GuestAlignmentFaultSignal, &alignment_fault_action, nullptr);

        struct sigaction access_fault_action {};
        access_fault_action.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
        access_fault_action.sa_sigaction =
            reinterpret_cast<HandlerType>(&ArmNce::GuestAccessFaultSignalHandler);
        access_fault_action.sa_mask = signal_mask;
        Common::SigAction(GuestAccessFaultSignal, &access_fault_action, &g_orig_segv_action);
    });
}

void ArmNce::SetTpidrroEl0(u64 value) {
    m_guest_ctx.tpidrro_el0 = value;
}

void ArmNce::GetContext(Kernel::Svc::ThreadContext& ctx) const {
    for (size_t i = 0; i < 29; i++) {
        ctx.r[i] = m_guest_ctx.cpu_registers[i];
    }
    ctx.fp = m_guest_ctx.cpu_registers[29];
    ctx.lr = m_guest_ctx.cpu_registers[30];
    ctx.sp = m_guest_ctx.sp;
    ctx.pc = m_guest_ctx.pc;
    ctx.pstate = m_guest_ctx.pstate;
    ctx.v = m_guest_ctx.vector_registers;
    ctx.fpcr = m_guest_ctx.fpcr;
    ctx.fpsr = m_guest_ctx.fpsr;
    ctx.tpidr = m_guest_ctx.tpidr_el0;
}

void ArmNce::SetContext(const Kernel::Svc::ThreadContext& ctx) {
    for (size_t i = 0; i < 29; i++) {
        m_guest_ctx.cpu_registers[i] = ctx.r[i];
    }
    m_guest_ctx.cpu_registers[29] = ctx.fp;
    m_guest_ctx.cpu_registers[30] = ctx.lr;
    m_guest_ctx.sp = ctx.sp;
    m_guest_ctx.pc = ctx.pc;
    m_guest_ctx.pstate = ctx.pstate;
    m_guest_ctx.vector_registers = ctx.v;
    m_guest_ctx.fpcr = ctx.fpcr;
    m_guest_ctx.fpsr = ctx.fpsr;
    m_guest_ctx.tpidr_el0 = ctx.tpidr;
}

void ArmNce::SignalInterrupt(Kernel::KThread* thread) {
    constexpr u64 break_loop = static_cast<u64>(HaltReason::BreakLoop);
    if ((m_guest_ctx.esr_el1.fetch_or(break_loop) & break_loop) != 0) return;

    auto* params = &thread->GetNativeExecutionParameters();
    LockThreadParameters(params);

    // Ensure visibility of is_running after lock acquire
    std::atomic_thread_fence(std::memory_order_acquire);

    if (params->is_running) {
        // We should signal to the running thread.
        // The running thread will unlock the thread context.
        syscall(SYS_tkill, m_thread_id, BreakFromRunCodeSignal);
    } else {
        // If the thread is no longer running, we have nothing to do.
        UnlockThreadParameters(params);
    }
}

[[maybe_unused]] const std::size_t CACHE_PAGE_SIZE = 4096;

void ArmNce::ClearInstructionCache() {
#ifdef __aarch64__
    // Ensure all previous memory operations complete
    asm volatile("dsb ish\n"
                 "dsb ish\n"
                 "isb" ::: "memory");
#endif
}

void ArmNce::InvalidateCacheRange(u64 addr, std::size_t size) {
    this->ClearInstructionCache();
}

} // namespace Core
