// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/gpu_write_footprint_profile.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include "common/logging.h"

namespace VideoCommon::GpuWriteTrace {
namespace {
std::unique_ptr<FootprintProfiler> profiler;
constexpr std::array path_names{"host_read", "staging", "immediate", "async_copy", "async_retire"};
constexpr std::array origin_names{"draw_ssbo", "dispatch_ssbo", "transform_feedback", "draw_texel",
    "dispatch_texel", "dma_copy", "dma_clear", "image_to_buffer", "query", "other"};
bool ValidRange(DAddr address, u64 size) {
    return size && address < (u64{1} << 36) && size <= (u64{1} << 36) - address;
}
std::string_view Reason(const FootprintOwnerKey& key) {
    switch (key.decision) {
    case FootprintDecision::AnalysisUnknown:
        return Shader::StorageWriteReasonName(static_cast<Shader::StorageWriteReason>(key.analysis_reason));
    case FootprintDecision::Applied: return "applied";
    case FootprintDecision::IneligibleStores: return "ineligible_stores";
    case FootprintDecision::SplitBinding: return "split_binding";
    case FootprintDecision::InvalidCopyRange: return "invalid_copy_range";
    case FootprintDecision::Disabled: return "switch_disabled";
    case FootprintDecision::Unattributed: return "unattributed";
    }
    return "invalid_reason";
}
} // namespace

struct FootprintProfiler::Impl {
    struct Segment { DAddr end; size_t owner; };
    struct Owner { FootprintOwnerRow row; u64 visited{}; };
    std::mutex mutex;
    std::map<DAddr, Segment> segments;
    std::map<FootprintOwnerKey, size_t> keys;
    std::vector<Owner> owners{1}; // Missing history is always owner zero.
    std::array<DownloadMetrics, DownloadPathCount> downloads{};
    size_t max_segments, max_owners;
    u64 range_overflow{}, owner_overflow{}, epoch{};

    Impl(size_t segments_limit, size_t owners_limit)
        : max_segments{segments_limit}, max_owners{(std::max)(owners_limit, size_t{1})} {}

    auto Split(DAddr address) {
        auto next = segments.lower_bound(address);
        if (next != segments.end() && next->first == address) return next;
        if (next == segments.begin()) return next;
        auto previous = std::prev(next);
        if (previous->second.end <= address) return next;
        const auto tail = previous->second;
        previous->second.end = address;
        return segments.emplace_hint(next, address, tail);
    }
    void Limit() {
        if (segments.size() <= max_segments) return;
        segments.clear();
        ++range_overflow;
    }
    void Assign(DAddr begin, DAddr end, size_t owner) {
        auto next = segments.upper_bound(begin);
        if (next != segments.begin()) {
            const auto previous = std::prev(next);
            if (previous->second.owner == owner && previous->second.end >= end) return;
        }
        const auto last = Split(end);
        const auto first = Split(begin);
        segments.erase(first, last);
        auto added = segments.emplace(begin, Segment{end, owner}).first;
        if (added != segments.begin()) {
            auto previous = std::prev(added);
            if (previous->second.end == begin && previous->second.owner == owner) {
                previous->second.end = end;
                segments.erase(added);
                added = previous;
            }
        }
        next = std::next(added);
        if (next != segments.end() && added->second.end == next->first && next->second.owner == owner) {
            added->second.end = next->second.end;
            segments.erase(next);
        }
        Limit();
    }
    template <class F>
    void Visit(DAddr begin, DAddr end, F&& visit) {
        auto it = segments.upper_bound(begin);
        if (it != segments.begin()) --it;
        DAddr cursor = begin;
        for (; it != segments.end() && it->first < end; ++it) {
            if (it->second.end <= cursor) continue;
            const DAddr lower = (std::max)(cursor, it->first);
            if (cursor < lower) visit(size_t{0}, cursor, lower);
            const DAddr upper = (std::min)(end, it->second.end);
            visit(it->second.owner, lower, upper);
            cursor = upper;
        }
        if (cursor < end) visit(size_t{0}, cursor, end);
    }
    size_t OwnerId(const FootprintOwnerKey& key) {
        if (const auto found = keys.find(key); found != keys.end()) return found->second;
        if (owners.size() >= max_owners) {
            ++owner_overflow;
            return 0;
        }
        const size_t id = owners.size();
        keys.emplace(key, id);
        owners.push_back(Owner{.row = {.key = key}});
        return id;
    }
};

FootprintProfiler::FootprintProfiler(size_t max_segments, size_t max_owners)
    : impl{std::make_unique<Impl>(max_segments, max_owners)} {}
FootprintProfiler::~FootprintProfiler() = default;

void FootprintProfiler::Mark(Origin origin, DAddr address, u64 size, const Attribution* attribution,
                             u64 repeated_bytes, u64 canceled_readback_bytes) {
    if (!ValidRange(address, size)) return;
    std::scoped_lock lock{impl->mutex};
    FootprintOwnerKey key{.origin = origin};
    bool sparse{};
    if (attribution) {
        key.shader = attribution->shader;
        key.binding = attribution->binding;
        key.decision = attribution->decision;
        key.analysis_reason = key.decision == FootprintDecision::AnalysisUnknown ? attribution->reason : 0;
        sparse = attribution->applied && attribution->known && !attribution->split &&
            Shader::Optimization::ValidStorageWriteIntervals(attribution->intervals, size);
    }
    const size_t owner = impl->OwnerId(key);
    auto& row = impl->owners[owner].row;
    ++row.marks;
    row.declared_bytes += size;
    row.max_binding_size = (std::max)(row.max_binding_size, size);
    row.last_address = address;
    row.last_size = size;
    u64 marked{};
    if (sparse) {
        for (const auto& interval : attribution->intervals) {
            impl->Assign(address + interval.begin, address + interval.end, owner);
            marked += interval.end - interval.begin;
        }
    } else {
        impl->Assign(address, address + size, owner);
        marked = size;
    }
    row.marked_bytes += marked;
    if (repeated_bytes == ~u64{0}) row.reuse_unknown_bytes += marked;
    else {
        row.repeated_bytes += repeated_bytes;
        row.fully_dirty_marks += repeated_bytes == marked;
    }
    row.canceled_readback_bytes += canceled_readback_bytes;
}

void FootprintProfiler::Clear(DAddr address, u64 size) {
    if (!ValidRange(address, size)) return;
    std::scoped_lock lock{impl->mutex};
    const auto last = impl->Split(address + size);
    const auto first = impl->Split(address);
    impl->segments.erase(first, last);
    impl->Limit();
}

void FootprintProfiler::Cpu(Site site, DAddr address, u64 size, bool exact) {
    if (!ValidRange(address, size)) return;
    std::scoped_lock lock{impl->mutex};
    const u64 epoch = ++impl->epoch;
    const DAddr begin = address & ~DAddr{4095};
    const DAddr end = (address + size + 4095) & ~DAddr{4095};
    impl->Visit(begin, end, [&](size_t owner, DAddr, DAddr) {
        if (!owner || impl->owners[owner].visited == epoch) return;
        auto& entry = impl->owners[owner];
        entry.visited = epoch;
        ++entry.row.conflicts;
        ++entry.row.conflict_sites[static_cast<size_t>(site)];
    });
    // These are page-conflict associations; no inferred exact byte overlap or sampling weights.
    static_cast<void>(exact);
}

void FootprintProfiler::DownloadSample::Add(DAddr address, u64 size) {
    if (!profiler || !ValidRange(address, size)) return;
    auto& state = *profiler->impl;
    std::scoped_lock lock{state.mutex};
    ++fragments;
    bytes += size;
    state.Visit(address, address + size, [&](size_t owner, DAddr begin, DAddr end) {
        auto& count = owners[owner];
        count.bytes += end - begin;
        ++count.fragments;
    });
}

void FootprintProfiler::DownloadSample::AddCopies(u64 regions, u64 transferred_bytes) {
    if (!profiler) return;
    copy_regions += regions;
    transfer_bytes += transferred_bytes;
}

void FootprintProfiler::DownloadSample::BeginPhase(DownloadPhase phase) {
    if (!profiler) return;
    BeginPhase(phase, Clock::now());
}

void FootprintProfiler::DownloadSample::EndPhase() {
    if (!profiler || active_phase == DownloadPhase::Count) return;
    EndPhase(Clock::now());
}

void FootprintProfiler::DownloadSample::BeginPhase(DownloadPhase phase, Clock::time_point now) {
    if (!profiler) return;
    EndPhase(now);
    active_phase = phase;
    phase_start = now;
}

void FootprintProfiler::DownloadSample::EndPhase(Clock::time_point now) {
    if (!profiler || active_phase == DownloadPhase::Count) return;
    if (now > phase_start) {
        phase_ns[static_cast<size_t>(active_phase)] +=
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - phase_start).count());
    }
    active_phase = DownloadPhase::Count;
}

void FootprintProfiler::DownloadSample::Commit(DownloadPath path, bool finishes) {
    if (!profiler) return;
    EndPhase();
    if (!bytes) {
        phase_ns = {};
        copy_regions = transfer_bytes = 0;
        return;
    }
    auto& state = *profiler->impl;
    std::scoped_lock lock{state.mutex};
    const auto index = static_cast<size_t>(path);
    auto& total = state.downloads[index];
    ++total.operations;
    total.bytes += bytes;
    total.fragments += fragments;
    total.finish_operations += finishes;
    for (size_t i = 0; i < phase_ns.size(); ++i) total.phase_ns[i] += phase_ns[i];
    total.copy_regions += copy_regions;
    total.transfer_bytes += transfer_bytes;
    if (finishes) {
        total.finish_copy_regions += copy_regions;
        total.max_copy_regions_per_finish = (std::max)(total.max_copy_regions_per_finish, copy_regions);
        total.finish_fragments += fragments;
        total.max_fragments_per_finish = (std::max)(total.max_fragments_per_finish, fragments);
    }
    for (const auto& [owner, count] : owners) {
        auto& result = state.owners[owner].row.downloads[index];
        ++result.operations;
        result.bytes += count.bytes;
        result.fragments += count.fragments;
        result.finish_operations += finishes;
    }
    owners.clear();
    bytes = fragments = 0;
    phase_ns = {};
    copy_regions = transfer_bytes = 0;
}

FootprintProfileSnapshot FootprintProfiler::TakeSnapshot() {
    std::scoped_lock lock{impl->mutex};
    FootprintProfileSnapshot result;
    result.downloads = std::exchange(impl->downloads, {});
    result.range_overflow = std::exchange(impl->range_overflow, 0);
    result.owner_overflow = std::exchange(impl->owner_overflow, 0);
    result.live_segments = impl->segments.size();
    for (auto& entry : impl->owners) {
        auto& row = entry.row;
        result.rows.push_back(row);
        row = {.key = row.key, .last_address = row.last_address, .last_size = row.last_size};
    }
    return result;
}

void ConfigureFootprintProfiler(bool active) {
    profiler.reset();
    if (active) profiler = std::make_unique<FootprintProfiler>();
}
void ProfileMark(Origin origin, DAddr address, u64 size, const Attribution* attribution,
                 u64 repeated_bytes, u64 canceled_readback_bytes) {
    profiler->Mark(origin, address, size, attribution, repeated_bytes, canceled_readback_bytes);
}
void ProfileClear(DAddr address, u64 size) { profiler->Clear(address, size); }
void ProfileCpu(Site site, DAddr address, u64 size, bool exact) { profiler->Cpu(site, address, size, exact); }
FootprintProfiler::DownloadSample MakeFootprintDownload() {
    FootprintProfiler::DownloadSample result{
        .profiler = profile_footprints && fault == Fault::None ? profiler.get() : nullptr};
    result.BeginPhase(DownloadPhase::Planning);
    return result;
}

void ReportFootprintProfiler(double seconds) {
    const auto snapshot = profiler->TakeSnapshot();
    LOG_INFO(HW_GPU, "GPU_WRITE_FOOTPRINT profile_window={:.3f}s apply={} "
                    "attribution=last_mark_not_executed_store "
                    "cpu_exact_collected=0 conflicts=selected_page_checks site_counts_not_additive=1 "
                    "owner_ops_are_associations=1 async_retire_is_candidate_bytes=1 "
                    "download_timing=host_steady_clock transfer_bytes_excludes_staging_padding=1 "
                    "copy_record_scope=host_prepare_enqueue finish_wait_scope=runtime_finish_call "
                    "host_read_route=FinishForHostRead_mapped_or_mixed "
                    "live_segments={} range_overflow={} owner_overflow={}",
             seconds, apply_footprints, snapshot.live_segments, snapshot.range_overflow, snapshot.owner_overflow);
    for (size_t path = 0; path < DownloadPathCount; ++path) {
        const auto& count = snapshot.downloads[path];
        const double fragments_per_finish = count.finish_operations
            ? static_cast<double>(count.finish_fragments) / count.finish_operations : 0.0;
        const double copy_regions_per_finish = count.finish_operations
            ? static_cast<double>(count.finish_copy_regions) / count.finish_operations : 0.0;
        LOG_INFO(HW_GPU, "GPU_WRITE_FOOTPRINT download path={} ops={} bytes={} range_fragments={} finish_ops={} "
                        "copy_regions={} transfer_bytes={} finish_copy_regions={} max_copy_regions_per_finish={} "
                        "finish_fragments={} max_fragments_per_finish={} fragments_per_finish={:.3f} "
                        "copy_regions_per_finish={:.3f} "
                        "planning_ms={:.3f} copy_record_ms={:.3f} finish_wait_ms={:.3f} writeback_ms={:.3f}",
                 path_names[path], count.operations, count.bytes, count.fragments, count.finish_operations,
                 count.copy_regions, count.transfer_bytes, count.finish_copy_regions, count.max_copy_regions_per_finish,
                 count.finish_fragments, count.max_fragments_per_finish, fragments_per_finish, copy_regions_per_finish,
                 count.phase_ns[static_cast<size_t>(DownloadPhase::Planning)] / 1e6,
                 count.phase_ns[static_cast<size_t>(DownloadPhase::CopyRecord)] / 1e6,
                 count.phase_ns[static_cast<size_t>(DownloadPhase::FinishWait)] / 1e6,
                 count.phase_ns[static_cast<size_t>(DownloadPhase::Writeback)] / 1e6);
    }
    const auto download_bytes = [](const FootprintOwnerRow& row) {
        u64 total{};
        for (size_t i = 0; i < static_cast<size_t>(DownloadPath::AsyncRetire); ++i) total += row.downloads[i].bytes;
        return total;
    };
    const auto finish_ops = [](const FootprintOwnerRow& row) {
        u64 total{};
        for (const auto& count : row.downloads) total += count.finish_operations;
        return total;
    };
    const auto download_ops = [](const FootprintOwnerRow& row) {
        u64 total{};
        for (size_t i = 0; i < static_cast<size_t>(DownloadPath::AsyncRetire); ++i) {
            total += row.downloads[i].operations;
        }
        return total;
    };
    std::map<FootprintOwnerKey, FootprintOwnerRow> grouped;
    for (const auto& row : snapshot.rows) {
        auto key = row.key;
        key.binding = 0; // Rank shaders/reasons together, rather than splitting their bindings.
        auto& sum = grouped[key];
        if (sum.last_size == 0 || row.max_binding_size > sum.max_binding_size) {
            sum.key = row.key; // Binding/address are a representative sample, not the group key.
            sum.last_address = row.last_address;
            sum.last_size = row.last_size;
        }
        sum.marks += row.marks;
        sum.declared_bytes += row.declared_bytes;
        sum.marked_bytes += row.marked_bytes;
        sum.repeated_bytes += row.repeated_bytes;
        sum.reuse_unknown_bytes += row.reuse_unknown_bytes;
        sum.canceled_readback_bytes += row.canceled_readback_bytes;
        sum.fully_dirty_marks += row.fully_dirty_marks;
        sum.max_binding_size = (std::max)(sum.max_binding_size, row.max_binding_size);
        sum.conflicts += row.conflicts;
        for (size_t i = 0; i < SiteCount; ++i) sum.conflict_sites[i] += row.conflict_sites[i];
        for (size_t i = 0; i < DownloadPathCount; ++i) {
            sum.downloads[i].bytes += row.downloads[i].bytes;
            sum.downloads[i].operations += row.downloads[i].operations;
            sum.downloads[i].fragments += row.downloads[i].fragments;
            sum.downloads[i].finish_operations += row.downloads[i].finish_operations;
        }
    }
    std::vector<FootprintOwnerRow> rows;
    for (const auto& [key, row] : grouped) rows.push_back(row);
    constexpr std::array ranks{"declared_B", "finish_ops", "download_B", "download_ops", "conflicts"};
    for (size_t ranking = 0; ranking < ranks.size(); ++ranking) {
        const auto weight = [&](const FootprintOwnerRow& row) {
            switch (ranking) {
            case 0: return row.declared_bytes;
            case 1: return finish_ops(row);
            case 2: return download_bytes(row);
            case 3: return download_ops(row);
            default: return row.conflicts;
            }
        };
        std::vector<size_t> order(rows.size());
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(order, [&](size_t a, size_t b) {
            const auto x = weight(rows[a]), y = weight(rows[b]);
            return x != y ? x > y : rows[a].key < rows[b].key;
        });
        for (size_t rank = 0; rank < (std::min)(order.size(), size_t{10}); ++rank) {
            const auto& row = rows[order[rank]];
            if (!weight(row)) break;
            const auto& host = row.downloads[static_cast<size_t>(DownloadPath::HostRead)];
            LOG_INFO(HW_GPU, "GPU_WRITE_FOOTPRINT top_by={} rank={} shader={:016x} sample_binding={} "
                            "origin={} reason={} marks={} declared_B={} marked_B={} already_dirty_B={} "
                            "fully_dirty_marks={} reuse_unknown_B={} canceled_readback_B={} "
                            "max_binding_B={} last={:#x}+{:#x} "
                            "conflicts={} tracker_ops={} preserve_ops={} nce_exact_ops={} nce_page_ops={} "
                            "download_B={} download_associations={} finish_associations={} "
                            "host_read_associations={} host_read_B={} "
                            "host_attribution_fragments={}",
                     ranks[ranking], rank + 1, row.key.shader, row.key.binding,
                     origin_names[static_cast<size_t>(row.key.origin)], Reason(row.key), row.marks,
                     row.declared_bytes, row.marked_bytes, row.repeated_bytes, row.fully_dirty_marks, row.reuse_unknown_bytes,
                     row.canceled_readback_bytes, row.max_binding_size, row.last_address, row.last_size,
                     row.conflicts, row.conflict_sites[static_cast<size_t>(Site::CpuTracker)],
                     row.conflict_sites[static_cast<size_t>(Site::Preserve)],
                     row.conflict_sites[static_cast<size_t>(Site::NceExact)],
                     row.conflict_sites[static_cast<size_t>(Site::NceStorePage)] +
                         row.conflict_sites[static_cast<size_t>(Site::NceReadPage)],
                     download_bytes(row), download_ops(row), finish_ops(row), host.operations, host.bytes, host.fragments);
        }
    }
    std::map<std::string_view, FootprintOwnerRow> reasons;
    for (const auto& row : rows) {
        if (row.key.origin != Origin::DispatchSsbo) continue;
        auto& sum = reasons[Reason(row.key)];
        sum.marks += row.marks;
        sum.declared_bytes += row.declared_bytes;
        sum.marked_bytes += row.marked_bytes;
        sum.conflicts += row.conflicts;
        for (size_t i = 0; i < DownloadPathCount; ++i) {
            sum.downloads[i].bytes += row.downloads[i].bytes;
            sum.downloads[i].finish_operations += row.downloads[i].finish_operations;
        }
    }
    for (const auto& [reason, row] : reasons) {
        if (!row.marks && !row.conflicts && !download_bytes(row)) continue;
        LOG_INFO(HW_GPU, "GPU_WRITE_FOOTPRINT reason={} marks={} declared_B={} marked_B={} "
                        "conflict_associations={} download_B={} finish_associations={}",
                 reason, row.marks, row.declared_bytes, row.marked_bytes, row.conflicts,
                 download_bytes(row), finish_ops(row));
    }
}

} // namespace VideoCommon::GpuWriteTrace
