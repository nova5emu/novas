// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "video_core/cpu_epoch_admission.h"
#include "video_core/nce_page_usage_profile.h"

using namespace VideoCommon::NcePageUsageProfile;

TEST_CASE("NCE page profile observes only faulted pages and retains mixed usage",
          "[video_core][nce_page_usage_profile]") {
    const auto t = std::make_unique<Tracker>();
    t->Reset();
    t->UsePhysical(0x1000, 4096, 0x9000, Bit(Use::Uniform));
    REQUIRE_FALSE(t->Get(0x1000));
    t->Fault(0x1001, 0x80, 5, 1);
    t->Fault(0x1fff, 0x84, 6, 2);
    REQUIRE(t->Watched() == 1);
    auto page = t->Get(0x1000);
    REQUIRE(page->faults == 2);
    REQUIRE(page->ever_mask == 0);
    REQUIRE(page->first_pc == 0x80);
    REQUIRE(page->recent_pc == 0x84);
    REQUIRE(page->first_tid == 5);
    REQUIRE(page->core_mask == 6);
    t->UsePhysical(0x1800, 0x801, 0x9800, Bit(Use::Uniform) | Bit(Use::GpuWrite));
    t->UsePhysical(0x1000, 4096, 0x9000, Bit(Use::Texture));
    REQUIRE_FALSE(t->Get(0x2000));
    page = t->Get(0x1000);
    REQUIRE(page->ever_mask == (Bit(Use::Uniform) | Bit(Use::GpuWrite) | Bit(Use::Texture)));
    REQUIRE(page->device_base == 0x9000);
    REQUIRE_FALSE(page->aliases);
    t->UsePhysical(0x1000, 1, 0xa000, Bit(Use::StorageRead));
    REQUIRE(t->Get(0x1000)->aliases);
    page = t->Take(0);
    REQUIRE(page->faults == 2);
    REQUIRE(t->Get(0x1000)->faults == 0);
    REQUIRE(t->Get(0x1000)->window_mask == 0);
    REQUIRE(t->Get(0x1000)->ever_mask != 0);
}

TEST_CASE("NCE page profile intersects large ranges and diagnoses incomplete addresses",
          "[video_core][nce_page_usage_profile]") {
    const auto t = std::make_unique<Tracker>();
    t->Reset();
    t->Fault(0x2000, 1, 2, 0);
    t->Fault(0x9000, 3, 4, 3);
    t->UsePhysical(0x1001, 0x9000, 0x4001, Bit(Use::Vertex));
    REQUIRE(t->Get(0x2000)->device_base == 0x5000);
    REQUIRE(t->Get(0x9000)->device_base == 0xc000);
    t->UsePhysical(0x2001, 1, 0, Bit(Use::Index));
    REQUIRE(t->Get(0x2000)->ever_mask == (Bit(Use::Vertex) | Bit(Use::Index)));
    t->UsePhysical(~u64{0} - 1, 4, 0, Bit(Use::Uniform));
    t->UsePhysical(0x2000, 1, 0x5000, 1U << 20);
    t->UnresolvedFault();
    const auto totals = t->TakeTotals();
    REQUIRE(totals.faults == 3);
    REQUIRE(totals.unresolved == 1);
    REQUIRE(totals.range_overflow == 1);
    REQUIRE(totals.invalid_usage == 1);
    REQUIRE(totals.alias_unknown == 1);
    REQUIRE(t->TakeTotals().faults == 0);
    t->Reset();
    REQUIRE(t->Watched() == 0);
    REQUIRE_FALSE(t->Get(0x2000));
}

TEST_CASE("NCE page profile bounds capacity and probe collisions",
          "[video_core][nce_page_usage_profile]") {
    const auto t = std::make_unique<Tracker>();
    t->Reset();
    for (size_t i = 0; i < Tracker::Capacity + 1; ++i) {
        t->Fault(i * 4096, 1, 1, 0);
    }
    REQUIRE(t->Watched() == Tracker::Capacity);
    REQUIRE(t->TakeTotals().dropped == 1);
    t->Reset();
    for (size_t i = 0; i < Tracker::Probes + 1; ++i) {
        t->Fault(i * Tracker::HashCapacity * 4096, 1, 1, 0);
    }
    REQUIRE(t->Watched() == Tracker::Probes);
    REQUIRE(t->TakeTotals().dropped == 1);
    REQUIRE_FALSE(t->Get(Tracker::Probes * Tracker::HashCapacity * 4096));
}

namespace {
void SetProfileEnv(const char* value) {
#ifdef _WIN32
    _putenv_s("EDEN_NX2_GPU_PROFILE", value);
#else
    setenv("EDEN_NX2_GPU_PROFILE", value, 1);
#endif
}
struct MappingProbe {
    u64 calls{};
    bool available{true};
    template <typename Callback>
    bool TryInspectPhysicalRange(DAddr device, u64 size, Callback&& callback) {
        ++calls;
        if (!available) {
            return false;
        }
        callback(0x1000, device, size);
        return true;
    }
};
} // namespace

TEST_CASE("NCE page profile mapping adapter skips unwatched and unavailable ranges",
          "[video_core][nce_page_usage_profile]") {
    const char* old = std::getenv("EDEN_NX2_GPU_PROFILE");
    const std::string saved = old ? old : "";
    SCOPE_EXIT {
        SetProfileEnv(saved.c_str());
        Configure();
    };
    SetProfileEnv("1");
    Configure();
    MappingProbe memory;
    UseDeviceRange(memory, 0x9000, 4096, Bit(Use::Uniform));
    REQUIRE(memory.calls == 0);
    Fault(0x1000, 1, 1, 0);
    REQUIRE(NeedsMapping());
    UseDeviceRange(memory, 0x9000, 4096, Bit(Use::Uniform));
    REQUIRE(memory.calls == 1);
    memory.available = false;
    UseDeviceRange(memory, 0x9000, 4096, Bit(Use::Vertex));
    REQUIRE(memory.calls == 2);
    UseDeviceRange(memory, 0x9000, 0, Bit(Use::Vertex));
    REQUIRE(memory.calls == 2);
    const auto tracker = std::make_unique<Tracker>();
    tracker->Reset();
    tracker->CoverageGap(8192);
    REQUIRE(tracker->TakeTotals().coverage_gap_bytes == 8192);
}

TEST_CASE("NCE page profile counts concurrent faults without changing first identity",
          "[video_core][nce_page_usage_profile]") {
    const auto tracker = std::make_unique<Tracker>();
    tracker->Reset();
    tracker->Fault(0x1000, 10, 20, 0);
    std::array<std::thread, 4> workers;
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i] {
            for (size_t n = 0; n < 500; ++n) {
                tracker->Fault(0x1000, 30 + i, 40 + i, static_cast<u32>(i));
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    const auto page = tracker->Get(0x1000);
    REQUIRE(tracker->Watched() == 1);
    REQUIRE(page->faults == 2001);
    REQUIRE(page->first_pc == 10);
    REQUIRE(page->first_tid == 20);
    REQUIRE(page->core_mask == 15);
    REQUIRE(tracker->TakeTotals().dropped == 0);
}

TEST_CASE("NCE page profile groups faults between observed GPU buffer reads",
          "[video_core][nce_page_usage_profile]") {
    const auto tracker = std::make_unique<Tracker>();
    tracker->Reset();
    tracker->Fault(0x1000, 1, 2, 0);
    tracker->Fault(0x1000, 1, 2, 0);
    tracker->Fault(0x1000, 1, 2, 0);
    auto page = tracker->Get(0x1000);
    REQUIRE(page->fault_bursts_approx == 1);
    REQUIRE(page->read_use_events == 0);
    tracker->UsePhysical(0x1000, 4096, 0x9000, Bit(Use::Uniform));
    tracker->UsePhysical(0x1000, 4096, 0x9000, Bit(Use::Vertex));
    tracker->Fault(0x1000, 1, 2, 0);
    tracker->Fault(0x1000, 1, 2, 0);
    page = tracker->Get(0x1000);
    REQUIRE(page->faults == 5);
    REQUIRE(page->fault_bursts_approx == 2);
    REQUIRE(page->read_use_events == 2);
    REQUIRE(page->read_use_version == 2);
    tracker->UsePhysical(0x1000, 4096, 0x9000, Bit(Use::GpuWrite));
    tracker->UsePhysical(0x1000, 4096, 0x9000, Bit(Use::Uniform) | Bit(Use::GpuWrite));
    tracker->Fault(0x1000, 1, 2, 0);
    page = tracker->Take(0);
    REQUIRE(page->fault_bursts_approx == 2);
    REQUIRE(page->read_use_events == 2);
    REQUIRE(page->ever_mask & Bit(Use::GpuWrite));
    page = tracker->Get(0x1000);
    REQUIRE(page->read_use_version == 2);
    REQUIRE(page->read_use_events == 0);
    REQUIRE(page->fault_bursts_approx == 0);
    tracker->Fault(0x1000, 1, 2, 0);
    REQUIRE(tracker->Get(0x1000)->fault_bursts_approx == 0);
    tracker->UsePhysical(0x1000, 4096, 0x9000, Bit(Use::Index));
    tracker->Fault(0x1000, 1, 2, 0);
    REQUIRE(tracker->Get(0x1000)->fault_bursts_approx == 1);
    REQUIRE(tracker->Get(0x1000)->epoch_races == 0);
}

TEST_CASE("NCE byte footprints cross words exactly and distinguish declared GPU overlap",
          "[video_core][nce_page_usage_profile]") {
    const auto t = std::make_unique<Tracker>();
    t->Reset();
    t->Fault(0x1000, 1, 2, 0);
    t->CpuWrite(0x103f, 2);
    t->CpuWrite(0x103f, 2);
    t->UsePhysical(0x1040, 4, 0x9040, Bit(Use::GpuWrite));
    const auto p = t->Get(0x1000);
    REQUIRE(p->cpu_bytes == 2);
    REQUIRE(p->gpu_declared_bytes == 4);
    REQUIRE(p->overlap_bytes == 1);
    REQUIRE(p->cpu_mask[0] == (u64{1} << 63));
    REQUIRE(p->cpu_mask[1] == 1);
    REQUIRE(p->gpu_declared_mask[1] == 15);
    const auto totals = t->TakeTotals();
    REQUIRE(totals.cpu_writes == 2);
    REQUIRE(totals.cpu_write_bytes == 4);
    REQUIRE(totals.dropped_cpu_writes == 0);
}

TEST_CASE("NCE byte footprints distinguish disjoint offsets and reset each window",
          "[video_core][nce_page_usage_profile]") {
    const auto t = std::make_unique<Tracker>();
    t->Reset();
    t->Fault(0x1000, 1, 2, 0);
    t->CpuWrite(0x1100, 4);
    t->UsePhysical(0x1200, 16, 0x9200, Bit(Use::GpuWrite));
    auto p = t->Take(0);
    REQUIRE(p->cpu_bytes == 4);
    REQUIRE(p->gpu_declared_bytes == 16);
    REQUIRE(p->overlap_bytes == 0);
    p = t->Get(0x1000);
    REQUIRE(p->cpu_bytes == 0);
    REQUIRE(p->gpu_declared_bytes == 0);
    t->CpuWrite(0x1100, 4);
    t->UsePhysical(0x1101, 2, 0x9101, Bit(Use::GpuWrite));
    p = t->Get(0x1000);
    REQUIRE(p->overlap_bytes == 2);
    REQUIRE(p->gpu_declared_bytes == 2);
    t->Reset();
    t->Fault(0x1000, 1, 2, 0);
    REQUIRE(t->Get(0x1000)->cpu_bytes == 0);
    REQUIRE(t->Get(0x1000)->gpu_declared_bytes == 0);
}

TEST_CASE("NCE byte footprints split pages and expose unknown CPU writes",
          "[video_core][nce_page_usage_profile]") {
    const auto t = std::make_unique<Tracker>();
    t->Reset();
    t->Fault(0x1000, 1, 2, 0);
    t->Fault(0x2000, 1, 2, 0);
    t->CpuWrite(0x1fff, 2);
    t->UsePhysical(0x1ffe, 4, 0x9ffe, Bit(Use::GpuWrite));
    REQUIRE(t->Get(0x1000)->cpu_bytes == 1);
    REQUIRE(t->Get(0x2000)->cpu_bytes == 1);
    REQUIRE(t->Get(0x1000)->gpu_declared_bytes == 2);
    REQUIRE(t->Get(0x2000)->gpu_declared_bytes == 2);
    REQUIRE(t->Get(0x1000)->overlap_bytes == 1);
    REQUIRE(t->Get(0x2000)->overlap_bytes == 1);
    REQUIRE(t->Get(0x1000)->cpu_mask[63] == (u64{1} << 63));
    REQUIRE(t->Get(0x2000)->cpu_mask[0] == 1);
    t->CpuWrite(~u64{0} - 1, 4);
    t->CpuWrite(0x3000, 1);
    t->UnresolvedCpuWrite();
    const auto total = t->TakeTotals();
    REQUIRE(total.cpu_writes == 3);
    REQUIRE(total.cpu_write_bytes == 7);
    REQUIRE(total.unresolved_cpu_writes == 1);
    REQUIRE(total.dropped_cpu_writes == 2);
    REQUIRE(total.range_overflow == 1);
    REQUIRE(t->TakeTotals().cpu_writes == 0);
}

TEST_CASE("NCE binding samples retain the last writer after subsequent read bindings",
          "[video_core][nce_page_usage_profile]") {
    const auto t = std::make_unique<Tracker>();
    t->Reset();
    t->Fault(0x1000, 1, 2, 0);
    const BindingInfo writer{0x9000, 4096, 8, 4, 1, 7, true};
    const BindingInfo reader{0xa000, 16, 9, 1, 2, 3, false};
    t->Binding(0x1000, 4096, writer);
    t->Binding(0x1000, 16, reader);
    const auto p = t->Get(0x1000);
    REQUIRE(p->last_binding.kind == 1);
    REQUIRE(p->last_binding.buffer_id == 9);
    REQUIRE(p->last_writable_binding.kind == 4);
    REQUIRE(p->last_writable_binding.buffer_id == 8);
    REQUIRE(p->last_writable_binding.index == 7);
    REQUIRE(p->last_writable_binding.written);
}

TEST_CASE("NCE page tracker is small and allocates storage only during explicit reset",
          "[video_core][nce_page_usage_profile]") {
    REQUIRE(sizeof(Tracker) < 512);
    Tracker tracker;
    REQUIRE_FALSE(tracker.Initialized());
    REQUIRE(tracker.Watched() == 0);
    tracker.Fault(0x1000, 1, 2, 0);
    tracker.CpuWrite(0x1000, 1);
    tracker.UsePhysical(0x1000, 1, 0x9000, Bit(Use::GpuWrite));
    REQUIRE_FALSE(tracker.Initialized());
    REQUIRE_FALSE(tracker.Get(0x1000));
    tracker.Reset();
    REQUIRE(tracker.Initialized());
    REQUIRE(tracker.TakeTotals().faults == 0);
    tracker.Fault(0x1000, 1, 2, 0);
    tracker.CpuWrite(0x1000, 1);
    tracker.UsePhysical(0x1000, 1, 0x9000, Bit(Use::GpuWrite));
    REQUIRE(tracker.Get(0x1000)->overlap_bytes == 1);
    tracker.Reset(false);
    REQUIRE_FALSE(tracker.Initialized());
    REQUIRE(tracker.Watched() == 0);
    REQUIRE_FALSE(tracker.Get(0x1000));
    REQUIRE(tracker.TakeTotals().cpu_writes == 0);
    tracker.Reset(false);
    REQUIRE_FALSE(tracker.Initialized());
    tracker.Reset(true);
    REQUIRE(tracker.Initialized());
    tracker.Fault(0x1000, 3, 4, 1);
    const auto sample = tracker.Get(0x1000);
    REQUIRE(sample->faults == 1);
    REQUIRE(sample->first_pc == 3);
    REQUIRE(sample->ever_mask == 0);
    REQUIRE(sample->cpu_bytes == 0);
    REQUIRE(sample->gpu_declared_bytes == 0);
    REQUIRE(sample->last_binding.kind == 0);
}

TEST_CASE("CPU epochs select storage by observed properties rather than buffer identity",
          "[nce_page_usage_profile]") {
    PageSample sample{};
    sample.device_known = true;
    sample.device_base = 0x9000;
    sample.ever_mask = Bit(Use::StorageRead) | Bit(Use::GpuWrite);
    sample.last_binding = {0x8000, 8192, 27, 3, 2, 1, false};
    sample.last_writable_binding = {0x9000, 4096, 39, 4, 0, 2, true};
    sample.cpu_bytes = 1000;
    sample.gpu_declared_bytes = 2048;
    CHECK(IsCpuEpochStorageCandidate(sample));
    sample.last_writable_binding.buffer_id = 116;
    sample.last_binding = {0x9800, 2048, 27, 3, 2, 1, false};
    sample.last_writable_binding = {0x9000, 2048, 116, 4, 0, 2, true};
    CHECK(IsCpuEpochStorageCandidate(sample));
    sample.last_binding = {0x8000, 8192, 27, 3, 2, 1, false};
    sample.last_writable_binding = {0x9000, 4096, 116, 4, 0, 2, true};
    CHECK(IsCpuEpochStorageCandidate(sample));
    SECTION("overlap") {
        sample.overlap_bytes = 1;
    }
    SECTION("alias") {
        sample.aliases = true;
    }
    SECTION("uniform use") {
        sample.ever_mask |= Bit(Use::Uniform);
    }
    SECTION("missing device identity") {
        sample.device_known = false;
    }
    SECTION("missing CPU footprint") {
        sample.cpu_bytes = 0;
    }
    SECTION("missing declared GPU footprint") {
        sample.gpu_declared_bytes = 0;
    }
    SECTION("unknown writer role") {
        sample.last_writable_binding.kind = 0;
    }
    SECTION("uniform writer role") {
        sample.last_writable_binding.kind = 2;
    }
    SECTION("multi range writer unsupported") {
        sample.last_writable_binding.kind = 5;
    }
    SECTION("writer not marked writable") {
        sample.last_writable_binding.written = false;
    }
    SECTION("missing buffer") {
        sample.last_writable_binding.buffer_id = 0;
    }
    SECTION("binding covers another page") {
        sample.last_writable_binding.device_address = 0xa000;
    }
    SECTION("binding ends at page boundary") {
        sample.last_binding.size = 4096;
    }
    CHECK_FALSE(IsCpuEpochStorageCandidate(sample));
}

TEST_CASE("CPU epochs admit observed read only storage without a GPU writer",
          "[nce_page_usage_profile]") {
    PageSample sample{};
    sample.device_known = true;
    sample.device_base = 0x9000;
    sample.ever_mask = Bit(Use::StorageRead);
    sample.cpu_bytes = 32;
    sample.last_binding = {0x9000, 4096, 52, 3, 0, 3, false};
    CHECK(IsCpuEpochStorageCandidate(sample));
    sample.last_binding.kind = 4;
    CHECK(IsCpuEpochStorageCandidate(sample));
    SECTION("declared writes contradict read only sample") {
        sample.gpu_declared_bytes = 1;
    }
    SECTION("writable binding contradicts read only sample") {
        sample.last_binding.written = true;
    }
    SECTION("historical writer contradicts read only sample") {
        sample.last_writable_binding.kind = 4;
    }
    SECTION("unknown reader") {
        sample.last_binding.kind = 0;
    }
    SECTION("uniform reader") {
        sample.last_binding.kind = 1;
    }
    SECTION("unknown buffer") {
        sample.last_binding.buffer_id = 0;
    }
    SECTION("unknown size") {
        sample.last_binding.size = 0;
    }
    SECTION("missing CPU bytes") {
        sample.cpu_bytes = 0;
    }
    SECTION("alias") {
        sample.aliases = true;
    }
    SECTION("unknown device") {
        sample.device_known = false;
    }
    CHECK_FALSE(IsCpuEpochStorageCandidate(sample));
}

TEST_CASE("CPU epoch audit recognizes read only storage and actual download requirements",
          "[nce_page_usage_profile]") {
    PageSample sample{};
    sample.device_known = true;
    sample.device_base = 0x9000;
    sample.ever_mask = Bit(Use::StorageRead);
    sample.cpu_bytes = 32;
    sample.last_binding = {0x9800, 2048, 52, 3, 0, 3, false};
    const auto clean = EvaluateCpuEpochAudit(&sample, 0x1000, 4, false, false);
    CHECK(clean.observed_candidate);
    CHECK_FALSE(clean.requires_download);
    CHECK_FALSE(clean.can_open);
    const auto dirty = EvaluateCpuEpochAudit(&sample, 0x1000, 4, false, true);
    CHECK(dirty.observed_candidate);
    CHECK(dirty.requires_download);
    const auto pending = EvaluateCpuEpochAudit(&sample, 0x1000, 4, true, false);
    CHECK(pending.observed_candidate);
    CHECK(pending.reasons & VideoCommon::CpuEpochAdmission::Bit(
                                VideoCommon::CpuEpochAdmission::Reason::PendingReadbacksOrUnknown));
    CHECK_FALSE(EvaluateCpuEpochAudit(&sample, 0x1fff, 2, false, false).observed_candidate);
    CHECK_FALSE(EvaluateCpuEpochAudit(nullptr, 0x1000, 4, false, false).observed_candidate);
}

TEST_CASE("CPU epoch audit rejects current stores in declared GPU write offsets",
          "[nce_page_usage_profile]") {
    PageSample sample{};
    sample.device_known = true;
    sample.device_base = 0x9000;
    sample.ever_mask = Bit(Use::StorageRead) | Bit(Use::GpuWrite);
    sample.cpu_bytes = 32;
    sample.gpu_declared_bytes = 1;
    sample.gpu_declared_mask[0] = u64{1} << 56;
    sample.last_binding = {0x9800, 2048, 52, 3, 0, 3, false};
    sample.last_writable_binding = {0x9000, 2048, 69, 4, 0, 1, true};
    CHECK(EvaluateCpuEpochAudit(&sample, 0x1000, 4, false, false).observed_candidate);
    CHECK_FALSE(EvaluateCpuEpochAudit(&sample, 0x1038, 4, false, false).observed_candidate);
}
