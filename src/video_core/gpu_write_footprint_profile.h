// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <map>
#include "video_core/gpu_write_trace.h"

namespace VideoCommon::GpuWriteTrace {

enum class DownloadPath : size_t { HostRead, Staging, Immediate, AsyncCopy, AsyncRetire, Count };
inline constexpr size_t DownloadPathCount = static_cast<size_t>(DownloadPath::Count);
enum class DownloadPhase : size_t { Planning, CopyRecord, FinishWait, Writeback, Count };
inline constexpr size_t DownloadPhaseCount = static_cast<size_t>(DownloadPhase::Count);
struct DownloadCount {
    u64 operations{}, bytes{}, fragments{}, finish_operations{};
};
struct DownloadMetrics : DownloadCount {
    std::array<u64, DownloadPhaseCount> phase_ns{};
    u64 copy_regions{}, transfer_bytes{}, finish_copy_regions{}, max_copy_regions_per_finish{};
    u64 finish_fragments{}, max_fragments_per_finish{};
};
struct FootprintOwnerKey {
    u64 shader{};
    u32 binding{};
    Origin origin{Origin::Other};
    FootprintDecision decision{FootprintDecision::Unattributed};
    u8 analysis_reason{};
    auto operator<=>(const FootprintOwnerKey&) const = default;
};
struct FootprintOwnerRow {
    FootprintOwnerKey key;
    u64 marks{}, declared_bytes{}, marked_bytes{}, repeated_bytes{}, reuse_unknown_bytes{};
    u64 canceled_readback_bytes{}, max_binding_size{}, conflicts{};
    u64 fully_dirty_marks{};
    DAddr last_address{};
    u64 last_size{};
    std::array<u64, SiteCount> conflict_sites{};
    std::array<DownloadCount, DownloadPathCount> downloads{};
};
struct FootprintProfileSnapshot {
    std::vector<FootprintOwnerRow> rows;
    std::array<DownloadMetrics, DownloadPathCount> downloads{};
    u64 range_overflow{}, owner_overflow{}, live_segments{};
};

class FootprintProfiler {
public:
    // Disjoint intervals keep only the last mark. Limits degrade attribution to unknown.
    explicit FootprintProfiler(size_t max_segments = 65536, size_t max_owners = 4096);
    ~FootprintProfiler();
    void Mark(Origin origin, DAddr address, u64 size, const Attribution* attribution,
              u64 repeated_bytes = ~u64{0}, u64 canceled_readback_bytes = 0);
    void Clear(DAddr address, u64 size);
    void Cpu(Site site, DAddr address, u64 size, bool exact);
    FootprintProfileSnapshot TakeSnapshot();

    struct DownloadSample {
        using Clock = std::chrono::steady_clock;
        FootprintProfiler* profiler{};
        std::map<size_t, DownloadCount> owners;
        u64 bytes{}, fragments{};
        std::array<u64, DownloadPhaseCount> phase_ns{};
        u64 copy_regions{}, transfer_bytes{};
        DownloadPhase active_phase{DownloadPhase::Count};
        Clock::time_point phase_start{};
        void Add(DAddr address, u64 size);
        void AddCopies(u64 regions, u64 transferred_bytes);
        void BeginPhase(DownloadPhase phase);
        void EndPhase();
        // Explicit timestamps permit deterministic phase accounting tests.
        void BeginPhase(DownloadPhase phase, Clock::time_point now);
        void EndPhase(Clock::time_point now);
        void Commit(DownloadPath path, bool finishes = false);
    };

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

void ConfigureFootprintProfiler(bool active);
void ReportFootprintProfiler(double seconds);
FootprintProfiler::DownloadSample MakeFootprintDownload();

} // namespace VideoCommon::GpuWriteTrace
