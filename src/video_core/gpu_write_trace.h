// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <compare>
#include <memory>
#include <mutex>
#include <vector>

#include "common/common_types.h"
#include "common/range_sets.h"
#include "shader_recompiler/ir_opt/storage_write_footprint.h"

namespace VideoCommon::GpuWriteTrace {

enum class Origin : size_t {
    DrawSsbo,
    DispatchSsbo,
    TransformFeedback,
    DrawTexel,
    DispatchTexel,
    DmaCopy,
    DmaClear,
    ImageToBuffer,
    Query,
    Other,
    Count,
};
enum class Site : size_t {
    CpuExact,
    CpuTracker,
    CachedCollected,
    Preserve,
    NceExact,
    NceStorePage,
    NceReadPage,
    Count,
};
enum class Fault : u8 { None, Store, Read };
enum class FootprintDecision : u8 {
    AnalysisUnknown, Applied, IneligibleStores, SplitBinding, InvalidCopyRange, Disabled, Unattributed,
};
enum class SizeSource : u8 { Unknown, Descriptor, NativeMapping, TableMapping, MappingFallback };
enum class Removal : size_t {
    CpuWrite, Download, AsyncDownload, InlineUpload, DmaOverwrite, DmaClear, Discard, CpuClaim,
    Other, Count,
};
inline constexpr size_t RemovalCount = static_cast<size_t>(Removal::Count);
inline thread_local SizeSource last_size_source{SizeSource::Unknown};

struct Attribution {
    using Interval = Shader::Optimization::StorageWriteInterval;
    u64 shader{};
    GPUVAddr gpu_address{};
    u32 binding{}, cbuf{}, cbuf_offset{};
    std::array<u32, 3> grid{}, block{};
    SizeSource size_source{};
    bool indirect{}, split{}, known{};
    u8 reason{};
    u64 begin{}, end{}; // Proven envelope relative to the aligned binding.
    std::vector<Interval> intervals; // Complete potential write set; empty uses the legacy envelope.
    bool applied{}; // Actual cache marking uses intervals, rather than the declared binding.
    FootprintDecision decision{FootprintDecision::AnalysisUnknown};
    auto operator<=>(const Attribution&) const = default;
};

inline bool enabled{}; // Configured before the guest/GPU threads start; not a live toggle.
inline bool apply_footprints{};
inline bool profile_footprints{};
inline thread_local Fault fault{Fault::None};
inline constexpr size_t OriginCount = static_cast<size_t>(Origin::Count);
inline constexpr size_t SiteCount = static_cast<size_t>(Site::Count);

struct MarkStats {
    u64 calls{}, bytes{}, page_touches{}, unique_pages{};
    u64 range_bytes{}, live_bytes{}, operation_calls{}, operation_bytes{};
};
struct ConflictStats {
    u64 hits{}, overlap{}, line64{}, page4k{}, unknown{};
    u64 gap_sum{}, gap_max{};
    u64 possible_overlap{}, outside_possible{}, footprint_unknown{};
};
struct SiteStats {
    u64 calls{}, hits{}, no_live_range{}, multi_origin{};
    u64 model_overlap{}, model_outside{};
};
struct Responsibility {
    struct Interval { DAddr begin{}, end{}; };
    Attribution identity;
    DAddr original{};
    u64 size{}, marks{}, live_bytes{}, fragments{}, launch_changes{};
    DAddr live_begin{}, live_end{}, sample_cpu{}, sample_nearest{};
    std::array<Interval, 4> remaining_intervals{};
    DAddr sample_fragment_begin{}, sample_fragment_end{};
    Site sample_site{};
    std::array<ConflictStats, SiteCount> conflicts{};
    std::array<u64, RemovalCount> removed_bytes{};
    Removal last_removal{Removal::Other};
    DAddr last_removed_address{};
    u64 last_removed_size{};
};
struct ShaderEstimate {
    u64 shader{}, conflicts{}, proven_marks{}, unknown_marks{};
    u64 possible_overlap{}, outside_possible{}, footprint_unknown{};
    u64 empty_pages_alone{}, empty_pages_all_proven{};
    u64 conflict_pages_empty_all_proven{};
};
struct Sample {
    Site site{};
    Origin origin{};
    DAddr cpu{}, marked{};
    u64 size{}, gap{};
    bool exact{};
};
struct Snapshot {
    std::array<MarkStats, OriginCount> marks{};
    std::array<SiteStats, SiteCount> sites{};
    std::array<std::array<ConflictStats, OriginCount>, SiteCount> conflicts{};
    std::array<Sample, 12> samples{};
    size_t sample_count{};
    std::vector<Responsibility> top;
    std::vector<ShaderEstimate> shaders;
    u64 empty_pages_all_proven{}, attribution_overflow{};
};

// Tracks declared ranges, NOT executed shader stores. Several origins can contribute to a
// dirty byte until it is cleared. All public operations are serialized; never use in a signal.
class Tracker {
public:
    Tracker();
    ~Tracker();
    void Mark(Origin origin, DAddr address, u64 size, bool add_range = true,
              const Attribution* attribution = nullptr);
    void AddRange(Origin origin, DAddr address, u64 size);
    void Clear(DAddr address, u64 size, Removal reason = Removal::Other);
    void Operation(Origin origin, u64 size);
    void Cpu(Site site, DAddr address, u64 size, bool exact);
    Snapshot TakeSnapshot();

private:
    std::mutex mutex;
    std::array<Common::RangeSet<DAddr>, OriginCount> live;
    std::array<Common::RangeSet<DAddr>, OriginCount> pages;
    Snapshot counts;
    struct Details;
    std::unique_ptr<Details> details;
};

void Configure();
void RecordFootprintMark(u64 declared_bytes, u64 marked_bytes, u64 intervals, bool applied);
void Report();
void MarkImpl(Origin origin, DAddr address, u64 size, bool add_range,
              const Attribution* attribution);
void AddRangeImpl(Origin origin, DAddr address, u64 size);
void ClearImpl(DAddr address, u64 size, Removal reason);
void OperationImpl(Origin origin, u64 size);
void CpuImpl(Site site, DAddr address, u64 size, bool exact);
void ProfileMark(Origin origin, DAddr address, u64 size, const Attribution* attribution,
                 u64 repeated_bytes = ~u64{0}, u64 canceled_readback_bytes = 0);
void ProfileClear(DAddr address, u64 size);
void ProfileCpu(Site site, DAddr address, u64 size, bool exact);

inline void Mark(Origin origin, DAddr address, u64 size, bool add_range = true,
                 const Attribution* attribution = nullptr, u64 repeated_bytes = ~u64{0},
                 u64 canceled_readback_bytes = 0) {
    if (profile_footprints && fault == Fault::None && add_range) {
        ProfileMark(origin, address, size, attribution, repeated_bytes, canceled_readback_bytes);
    }
    if (enabled && fault == Fault::None) {
        MarkImpl(origin, address, size, add_range, attribution);
    }
}
inline void AddRange(Origin origin, DAddr address, u64 size) {
    if (profile_footprints && fault == Fault::None) ProfileMark(origin, address, size, nullptr);
    if (enabled && fault == Fault::None) {
        AddRangeImpl(origin, address, size);
    }
}
inline void Clear(DAddr address, u64 size, Removal reason = Removal::Other) {
    if (profile_footprints && fault == Fault::None) ProfileClear(address, size);
    if (enabled && fault == Fault::None) {
        ClearImpl(address, size, reason);
    }
}
inline void Operation(Origin origin, u64 size) {
    if (enabled && fault == Fault::None) {
        OperationImpl(origin, size);
    }
}
inline void Cpu(Site site, DAddr address, u64 size, bool exact = true) {
    if (profile_footprints && fault == Fault::None && site != Site::CpuExact &&
        site != Site::CachedCollected) ProfileCpu(site, address, size, exact);
    if (enabled && fault == Fault::None) {
        CpuImpl(site, address, size, exact);
    }
}

} // namespace VideoCommon::GpuWriteTrace
