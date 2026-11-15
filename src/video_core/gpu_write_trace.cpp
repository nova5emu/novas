// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/gpu_write_trace.h"
#include "video_core/gpu_write_trace_dump.h"
#include "video_core/gpu_write_footprint_profile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <memory>
#include <map>
#include <unordered_map>
#include <utility>
#include "common/logging.h"
#include "common/range_sets.inc"
#include "shader_recompiler/shader_info.h"

namespace VideoCommon::GpuWriteTrace {
namespace {
constexpr std::array origin_names{
    "draw_ssbo", "dispatch_ssbo", "transform_feedback", "draw_texel", "dispatch_texel",
    "dma_copy", "dma_clear", "image_to_buffer", "query", "other",
};
constexpr std::array site_names{
    "cpu_exact", "cpu_tracker", "cached_collected", "preserve", "nce_exact",
    "nce_store_page", "nce_read_page",
};
constexpr std::array size_names{"unknown", "descriptor_word", "native_mapping_cap",
                                "table_mapping_cap", "mapping_fallback_cap"};
constexpr std::array removal_names{"cpu_write", "download", "async_download", "inline_upload",
                                   "dma_overwrite", "dma_clear", "discard", "cpu_claim", "other"};
std::unique_ptr<Tracker> tracker;
std::chrono::steady_clock::time_point last_report;
struct FootprintCounters {
    std::atomic<u64> calls{}, applied{}, fallback{}, declared_bytes{}, marked_bytes{}, intervals{};
} footprint_counts;

bool SparseApplied(const Attribution& attribution, u64 size) {
    return attribution.applied && attribution.known && !attribution.split &&
           Shader::Optimization::ValidStorageWriteIntervals(attribution.intervals, size);
}

bool Valid(DAddr address, u64 size) {
    return size != 0 && address < (1ULL << 36) && size <= (1ULL << 36) - address;
}
} // namespace

struct Tracker::Details {
    struct Key {
        Attribution identity;
        DAddr address;
        u64 size;
        auto operator<=>(const Key&) const = default;
    };
    struct Record {
        Responsibility stats;
        Common::RangeSet<DAddr> actual, predicted;
        Common::RangeSet<DAddr> conflict_empty_pages;
        u64 visited{};
        bool fully_indexed{};
    };
    static constexpr size_t MaxRecords = 4096;
    static constexpr size_t MaxPageLinks = 1 << 20;
    std::map<Key, size_t> keys;
    std::vector<Record> records;
    std::unordered_map<DAddr, std::vector<size_t>> page_index;
    Common::RangeSet<DAddr> unattributed, predicted;
    u64 epoch{}, overflow{};
    size_t page_links{};

    static Key MakeKey(Attribution identity, DAddr address, u64 size) {
        // Evaluated intervals are part of the key: changed footprints retain earlier GPU history.
        identity.grid = {};
        identity.block = {};
        return {identity, address, size};
    }

    template <typename F>
    void Visit(DAddr address, u64 size, F&& visit) {
        ++epoch;
        const DAddr last = (address + size - 1) >> 12;
        for (DAddr page = address >> 12; page <= last; ++page) {
            const auto found = page_index.find(page);
            if (found == page_index.end()) continue;
            for (size_t index : found->second) {
                auto& record = records[index];
                if (record.visited == epoch) continue;
                record.visited = epoch;
                visit(record);
            }
        }
    }

    void Mark(DAddr address, u64 size, const Attribution* identity) {
        if (!identity) {
            unattributed.Add(address, size);
            predicted.Add(address, size);
            return;
        }
        auto safe = *identity;
        // An incomplete proof never makes the counterfactual cleaner.
        safe.known &= !safe.split && safe.begin < safe.end && safe.end <= size;
        u64 previous_end{};
        for (const auto& range : safe.intervals) {
            safe.known &= range.begin >= previous_end && range.begin < range.end && range.end <= size;
            previous_end = range.end;
        }
        if (!safe.known) safe.intervals.clear();
        safe.applied = SparseApplied(safe, size);
        const Key key = MakeKey(safe, address, size);
        auto found = keys.find(key);
        if (found == keys.end()) {
            const u64 page_count = ((address + size - 1) >> 12) - (address >> 12) + 1;
            if (records.size() == MaxRecords || page_count > MaxPageLinks - page_links) {
                ++overflow;
                Mark(address, size, nullptr);
                return;
            }
            const size_t index = records.size();
            records.emplace_back();
            records.back().stats.identity = safe;
            records.back().stats.original = address;
            records.back().stats.size = size;
            found = keys.emplace(key, index).first;
        }
        auto& r = records[found->second];
        const bool already_full = r.fully_indexed;
        if (r.stats.identity.grid != safe.grid || r.stats.identity.block != safe.block) {
            ++r.stats.launch_changes;
            r.stats.identity.grid = safe.grid;
            r.stats.identity.block = safe.block;
        }
        if (!r.fully_indexed) {
            const u64 page_count = ((address + size - 1) >> 12) - (address >> 12) + 1;
            if (page_count > MaxPageLinks - page_links) {
                ++overflow;
                Mark(address, size, nullptr);
                return;
            }
            for (DAddr p = address >> 12; p <= (address + size - 1) >> 12; ++p) {
                auto& ids = page_index[p];
                if (std::ranges::find(ids, found->second) == ids.end()) {
                    ids.push_back(found->second);
                    ++page_links;
                }
            }
            r.fully_indexed = true;
        }
        ++r.stats.marks;
        if (already_full) return;
        if (safe.applied) {
            for (const auto& range : safe.intervals) {
                r.actual.Add(address + range.begin, range.end - range.begin);
            }
        } else {
            r.actual.Add(address, size);
        }
        if (safe.known && !safe.intervals.empty()) {
            for (const auto& range : safe.intervals) {
                r.predicted.Add(address + range.begin, range.end - range.begin);
                predicted.Add(address + range.begin, range.end - range.begin);
            }
            return;
        }
        const DAddr begin = safe.known ? address + safe.begin : address;
        const u64 length = safe.known ? safe.end - safe.begin : size;
        r.predicted.Add(begin, length);
        predicted.Add(begin, length);
    }

    void Clear(DAddr address, u64 size, Removal reason) {
        Visit(address, size, [&](Record& r) {
            u64 removed{};
            r.actual.ForEachInRange(address, size, [&](DAddr a, DAddr b) { removed += b - a; });
            if (!removed) return;
            r.fully_indexed = false;
            r.stats.removed_bytes[static_cast<size_t>(reason)] += removed;
            r.stats.last_removal = reason;
            r.stats.last_removed_address = address;
            r.stats.last_removed_size = size;
            r.actual.Subtract(address, size);
            r.predicted.Subtract(address, size);
        });
        for (DAddr page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
            const auto found = page_index.find(page);
            if (found == page_index.end()) continue;
            auto& ids = found->second;
            const auto before = ids.size();
            std::erase_if(ids, [&](size_t id) { return !records[id].actual.Intersects(page << 12, 4096); });
            page_links -= before - ids.size();
            if (ids.empty()) page_index.erase(found);
        }
        unattributed.Subtract(address, size);
        predicted.Subtract(address, size);
    }

    void Cpu(Site site, DAddr address, u64 size, bool exact) {
        const DAddr first = address & ~DAddr{4095};
        const DAddr end = address + size;
        const DAddr last = (end + 4095) & ~DAddr{4095};
        Visit(address, size, [&](Record& r) {
            u64 gap = ~u64{};
            DAddr nearest{}, fragment_begin{}, fragment_end{};
            bool same_line{};
            r.actual.ForEachInRange(first, last - first, [&](DAddr a, DAddr b) {
                same_line |= (a >> 6) <= ((end - 1) >> 6) && ((b - 1) >> 6) >= (address >> 6);
                const u64 distance = a >= end ? a - (end - 1) : b <= address ? address - (b - 1) : 0;
                if (distance < gap) {
                    gap = distance;
                    nearest = a >= end ? a : b <= address ? b - 1 : (std::max)(a, address);
                    fragment_begin = a;
                    fragment_end = b;
                }
            });
            if (gap == ~u64{}) return;
            if (site == Site::CpuExact && r.stats.identity.known) {
                for (DAddr page = first; page < last; page += 4096) {
                    if (r.actual.Intersects(page, 4096) && !predicted.Intersects(page, 4096)) {
                        r.conflict_empty_pages.Add(page >> 12, 1);
                    }
                }
            }
            auto& c = r.stats.conflicts[static_cast<size_t>(site)];
            ++c.hits;
            if (!r.stats.identity.known || !exact) {
                ++c.footprint_unknown;
            } else if (r.predicted.Intersects(address, size)) {
                ++c.possible_overlap;
            } else {
                ++c.outside_possible;
            }
            if (!exact) {
                ++c.unknown;
            } else {
                c.gap_sum += gap;
                c.gap_max = (std::max)(c.gap_max, gap);
                if (!gap) ++c.overlap;
                else if (same_line) ++c.line64;
                else ++c.page4k;
            }
            if (c.hits == 1 && (site == Site::CpuExact || !r.stats.sample_fragment_end)) {
                r.stats.sample_site = site;
                r.stats.sample_cpu = address;
                r.stats.sample_nearest = nearest;
                r.stats.sample_fragment_begin = fragment_begin;
                r.stats.sample_fragment_end = fragment_end;
            }
        });
    }

    static void AddPages(Common::RangeSet<DAddr>& pages, const Common::RangeSet<DAddr>& bytes) {
        bytes.ForEach([&](DAddr a, DAddr b) {
            pages.Add(a >> 12, ((b - 1) >> 12) - (a >> 12) + 1);
        });
    }
    static u64 Count(const Common::RangeSet<DAddr>& ranges) {
        u64 count{};
        ranges.ForEach([&](DAddr a, DAddr b) { count += b - a; });
        return count;
    }
    void SnapshotInto(Snapshot& snapshot) {
        Common::RangeSet<DAddr> actual_pages, predicted_pages, all_empty;
        AddPages(actual_pages, unattributed);
        AddPages(predicted_pages, predicted);
        std::map<u64, ShaderEstimate> shaders;
        for (auto& r : records) {
            AddPages(actual_pages, r.actual);
            auto& s = shaders[r.stats.identity.shader];
            s.shader = r.stats.identity.shader;
            const auto& conflict = r.stats.conflicts[static_cast<size_t>(Site::CpuExact)];
            s.conflicts += conflict.hits;
            s.possible_overlap += conflict.possible_overlap;
            s.outside_possible += conflict.outside_possible;
            s.footprint_unknown += conflict.footprint_unknown;
            (r.stats.identity.known ? s.proven_marks : s.unknown_marks) += r.stats.marks;
            auto row = r.stats;
            r.actual.ForEach([&](DAddr a, DAddr b) {
                if (row.fragments < row.remaining_intervals.size()) {
                    row.remaining_intervals[row.fragments] = {a, b};
                }
                if (!row.fragments++) row.live_begin = a;
                row.live_end = b;
                row.live_bytes += b - a;
            });
            snapshot.top.push_back(row);
        }
        actual_pages.ForEach([&](DAddr a, DAddr b) { all_empty.Add(a, b - a); });
        predicted_pages.ForEach([&](DAddr a, DAddr b) { all_empty.Subtract(a, b - a); });
        snapshot.empty_pages_all_proven = Count(all_empty);
        for (const auto& [hash, s] : shaders) snapshot.shaders.push_back(s);
        std::ranges::sort(snapshot.shaders, [](const auto& a, const auto& b) {
            return a.conflicts != b.conflicts ? a.conflicts > b.conflicts : a.shader < b.shader;
        });
        if (snapshot.shaders.size() > 8) snapshot.shaders.resize(8);
        for (auto& s : snapshot.shaders) {
            Common::RangeSet<DAddr> retained, owned, alone, conflict_empty;
            AddPages(retained, unattributed);
            for (const auto& r : records) {
                const bool own = r.stats.identity.shader == s.shader;
                AddPages(retained, own ? r.predicted : r.actual);
                if (own) {
                    AddPages(owned, r.actual);
                    r.conflict_empty_pages.ForEach([&](DAddr a, DAddr b) {
                        conflict_empty.Add(a, b - a);
                    });
                }
            }
            owned.ForEach([&](DAddr a, DAddr b) {
                alone.Add(a, b - a);
                all_empty.ForEachInRange(a, b - a, [&](DAddr c, DAddr d) {
                    s.empty_pages_all_proven += d - c;
                });
            });
            retained.ForEach([&](DAddr a, DAddr b) { alone.Subtract(a, b - a); });
            s.empty_pages_alone = Count(alone);
            s.conflict_pages_empty_all_proven = Count(conflict_empty);
        }
        std::ranges::sort(snapshot.top, [](const auto& a, const auto& b) {
            const u64 ah = a.conflicts[static_cast<size_t>(Site::CpuExact)].hits;
            const u64 bh = b.conflicts[static_cast<size_t>(Site::CpuExact)].hits;
            return ah != bh ? ah > bh : a.marks > b.marks;
        });
        if (snapshot.top.size() > 8) snapshot.top.resize(8);
        snapshot.attribution_overflow = std::exchange(overflow, 0);
        for (auto& r : records) {
            r.stats.marks = 0;
            r.stats.launch_changes = 0;
            r.stats.conflicts = {};
            r.stats.removed_bytes = {};
            r.stats.sample_fragment_end = 0;
            r.conflict_empty_pages.Clear();
        }
        // Keep retired statistics for this report, but no historical entries in future CPU lookups.
        if (std::ranges::any_of(records, [](const auto& r) { return r.actual.Empty(); })) {
            std::erase_if(records, [](const auto& r) { return r.actual.Empty(); });
            keys.clear();
            page_index.clear();
            page_links = 0;
            for (size_t i = 0; i < records.size(); ++i) {
                auto& r = records[i];
                keys.emplace(MakeKey(r.stats.identity, r.stats.original, r.stats.size), i);
                r.actual.ForEach([&](DAddr a, DAddr b) {
                    for (DAddr p = a >> 12; p <= (b - 1) >> 12; ++p) {
                        auto& ids = page_index[p];
                        if (ids.empty() || ids.back() != i) {
                            ids.push_back(i);
                            ++page_links;
                        }
                    }
                });
            }
        }
    }
};

Tracker::Tracker() : details{std::make_unique<Details>()} {}
Tracker::~Tracker() = default;

void Tracker::Mark(Origin origin, DAddr address, u64 size, bool add_range,
                   const Attribution* attribution) {
    if (!Valid(address, size)) {
        return;
    }
    std::scoped_lock lock{mutex};
    const auto index = static_cast<size_t>(origin);
    auto& stat = counts.marks[index];
    ++stat.calls;
    const auto count_range = [&](DAddr start, u64 length) {
        stat.bytes += length;
        const u64 first = start >> 12;
        const u64 page_count = ((start + length - 1) >> 12) - first + 1;
        stat.page_touches += page_count;
        pages[index].Add(first, page_count);
        if (add_range) {
            live[index].Add(start, length);
            stat.range_bytes += length;
        }
    };
    if (attribution && SparseApplied(*attribution, size)) {
        for (const auto& range : attribution->intervals) {
            count_range(address + range.begin, range.end - range.begin);
        }
    } else {
        count_range(address, size);
    }
    if (add_range) {
        details->Mark(address, size, attribution);
    }
}

void Tracker::AddRange(Origin origin, DAddr address, u64 size) {
    if (!Valid(address, size)) {
        return;
    }
    std::scoped_lock lock{mutex};
    const auto index = static_cast<size_t>(origin);
    live[index].Add(address, size);
    details->Mark(address, size, nullptr);
    counts.marks[index].range_bytes += size;
}

void Tracker::Clear(DAddr address, u64 size, Removal reason) {
    if (!Valid(address, size)) {
        return;
    }
    std::scoped_lock lock{mutex};
    details->Clear(address, size, reason);
    for (auto& ranges : live) {
        ranges.Subtract(address, size);
    }
}

void Tracker::Operation(Origin origin, u64 size) {
    std::scoped_lock lock{mutex};
    auto& stat = counts.marks[static_cast<size_t>(origin)];
    ++stat.operation_calls;
    stat.operation_bytes += size;
}

void Tracker::Cpu(Site site, DAddr address, u64 size, bool exact) {
    if (!Valid(address, size)) {
        return;
    }
    std::scoped_lock lock{mutex};
    details->Cpu(site, address, size, exact);
    auto& site_stat = counts.sites[static_cast<size_t>(site)];
    ++site_stat.calls;
    const DAddr end = address + size;
    const DAddr page_begin = address & ~DAddr{4095};
    const DAddr page_end = (end + 4095) & ~DAddr{4095};
    u64 origins = 0;
    for (size_t index = 0; index < OriginCount; ++index) {
        u64 gap = (std::numeric_limits<u64>::max)();
        DAddr nearest{};
        bool same_line = false;
        live[index].ForEachInRange(page_begin, page_end - page_begin,
            [&](DAddr begin, DAddr finish) {
                same_line |= (begin >> 6) <= ((end - 1) >> 6) &&
                             ((finish - 1) >> 6) >= (address >> 6);
                const u64 distance = begin >= end ? begin - (end - 1)
                                     : finish <= address ? address - (finish - 1) : 0;
                if (distance < gap) {
                    gap = distance;
                    nearest = begin >= end ? begin : finish <= address ? finish - 1
                                                                          : (std::max)(begin, address);
                }
            });
        if (gap == (std::numeric_limits<u64>::max)()) {
            continue;
        }
        ++origins;
        auto& stat = counts.conflicts[static_cast<size_t>(site)][index];
        ++stat.hits;
        if (!exact) {
            ++stat.unknown;
        } else {
            stat.gap_sum += gap;
            stat.gap_max = (std::max)(stat.gap_max, gap);
            if (gap == 0) {
                ++stat.overlap;
            } else if (same_line) {
                ++stat.line64;
            } else {
                ++stat.page4k;
            }
        }
        if (stat.hits == 1 && counts.sample_count < counts.samples.size()) {
            counts.samples[counts.sample_count++] =
                Sample{site, static_cast<Origin>(index), address, nearest, size, gap, exact};
        }
    }
    if (origins) {
        ++site_stat.hits;
        if (exact) {
            if (details->predicted.Intersects(address, size)) ++site_stat.model_overlap;
            else ++site_stat.model_outside;
        }
    } else {
        ++site_stat.no_live_range;
    }
    if (origins > 1) {
        ++site_stat.multi_origin;
    }
}

Snapshot Tracker::TakeSnapshot() {
    std::scoped_lock lock{mutex};
    Snapshot result = counts;
    details->SnapshotInto(result);
    counts = {};
    for (size_t i = 0; i < OriginCount; ++i) {
        pages[i].ForEach([&](DAddr begin, DAddr end) { result.marks[i].unique_pages += end - begin; });
        pages[i].Clear();
        live[i].ForEach([&](DAddr begin, DAddr end) { result.marks[i].live_bytes += end - begin; });
    }
    return result;
}

void Configure() {
    const char* value = std::getenv("EDEN_NX2_GPU_WRITE_TRACE");
    enabled = value && value[0] == '1' && value[1] == '\0';
    apply_footprints = Shader::Optimization::StorageWriteFootprintsEnabled();
    profile_footprints = Shader::Optimization::StorageWriteFootprintProfilingEnabled();
    ConfigureFootprintProfiler(profile_footprints);
    footprint_counts.calls = 0;
    footprint_counts.applied = 0;
    footprint_counts.fallback = 0;
    footprint_counts.declared_bytes = 0;
    footprint_counts.marked_bytes = 0;
    footprint_counts.intervals = 0;
    last_report = std::chrono::steady_clock::now();
    if (apply_footprints) {
        LOG_INFO(HW_GPU, "GPU_WRITE_FOOTPRINT enabled=1 backend=vulkan stores=32_64_128 "
                        "unknown=full mixed_pages=preserve");
    }
    tracker.reset();
    ResetComputeDumps(enabled);
    if (!enabled) {
        return;
    }
    tracker = std::make_unique<Tracker>();
}

void RecordFootprintMark(u64 declared_bytes, u64 marked_bytes, u64 intervals, bool applied) {
    footprint_counts.calls.fetch_add(1, std::memory_order_relaxed);
    (applied ? footprint_counts.applied : footprint_counts.fallback)
        .fetch_add(1, std::memory_order_relaxed);
    footprint_counts.declared_bytes.fetch_add(declared_bytes, std::memory_order_relaxed);
    footprint_counts.marked_bytes.fetch_add(marked_bytes, std::memory_order_relaxed);
    footprint_counts.intervals.fetch_add(intervals, std::memory_order_relaxed);
}
void MarkImpl(Origin origin, DAddr address, u64 size, bool add_range,
              const Attribution* attribution) {
    tracker->Mark(origin, address, size, add_range, attribution);
}
void AddRangeImpl(Origin origin, DAddr address, u64 size) {
    tracker->AddRange(origin, address, size);
}
void ClearImpl(DAddr address, u64 size, Removal reason) {
    tracker->Clear(address, size, reason);
}
void OperationImpl(Origin origin, u64 size) {
    tracker->Operation(origin, size);
}
void CpuImpl(Site site, DAddr address, u64 size, bool exact) {
    tracker->Cpu(site, address, size, exact);
}

void Report() {
    if (!enabled && !profile_footprints) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    if (seconds < 10.0) {
        return;
    }
    last_report = now;
    if (apply_footprints) {
        LOG_INFO(HW_GPU, "GPU_WRITE_FOOTPRINT window={:.3f}s bindings={} applied={} fallback={} "
                        "declared_B={} marked_B={} intervals={}", seconds,
                 footprint_counts.calls.exchange(0, std::memory_order_relaxed),
                 footprint_counts.applied.exchange(0, std::memory_order_relaxed),
                 footprint_counts.fallback.exchange(0, std::memory_order_relaxed),
                 footprint_counts.declared_bytes.exchange(0, std::memory_order_relaxed),
                 footprint_counts.marked_bytes.exchange(0, std::memory_order_relaxed),
                 footprint_counts.intervals.exchange(0, std::memory_order_relaxed));
    }
    if (profile_footprints) ReportFootprintProfiler(seconds);
    if (!enabled) return;
    const auto snapshot = tracker->TakeSnapshot();
    LOG_INFO(HW_GPU, "GPU_WRITE_TRACE window={:.3f}s declared_ranges_only={} origins_can_overlap=1 "
                    "cpu_exact=attempt_before_64B_collection cpu_tracker=uncached_page_checks "
                    "nce_page_distance=unknown sites_are_not_additive=1", seconds,
             static_cast<int>(!apply_footprints));
    LOG_INFO(HW_GPU, "GPU_WRITE_TRACE estimate empty_pages_all_proven_now={} "
                    "attribution_overflow={} units=device_4K_pages counterfactual=observed_mark_clear_stream "
                    "unknown_retains_full_range=1 physical_alias_dedup=0 branches=union_all "
                    "identified_origins=vulkan_dispatch_ssbo top=8 ranks=cpu_exact_associations",
             snapshot.empty_pages_all_proven, snapshot.attribution_overflow);
    for (const auto& s : snapshot.shaders) {
        LOG_INFO(HW_GPU, "GPU_WRITE_TRACE shader={:016x} conflict_associations={} proven_marks={} "
                        "unknown_marks={} empty_pages_alone_now={} empty_pages_all_proven_now={} "
                        "conflict_pages_empty_all_proven_window={} potential_overlap={} "
                        "outside_potential={} footprint_unknown={}",
                 s.shader, s.conflicts, s.proven_marks, s.unknown_marks, s.empty_pages_alone,
                 s.empty_pages_all_proven, s.conflict_pages_empty_all_proven,
                 s.possible_overlap, s.outside_possible, s.footprint_unknown);
    }
    size_t rank{};
    for (const auto& row : snapshot.top) {
        const auto& a = row.identity;
        u64 potential_bytes{};
        if (a.known) {
            if (a.intervals.empty()) {
                potential_bytes = a.end - a.begin;
            } else {
                for (const auto& interval : a.intervals) potential_bytes += interval.end - interval.begin;
            }
        }
        LOG_INFO(HW_GPU, "GPU_WRITE_TRACE owner rank={} shader={:016x} binding={} cb={}+{:#x} "
                        "gpu_va={:#x} original={:#x}+{:#x} size_source={} grid_last={}x{}x{} "
                        "block_last={}x{}x{} launch_changes={} indirect={} split={} proof={} analysis_reason={} "
                        "envelope=[{:#x},{:#x}) potential_intervals={} potential_B={} applied={} "
                        "marks={} remaining_B={} fragments={} remaining_bounds=[{:#x},{:#x})",
                 ++rank, a.shader, a.binding, a.cbuf, a.cbuf_offset, a.gpu_address,
                 row.original, row.size, size_names[static_cast<size_t>(a.size_source)],
                 a.grid[0], a.grid[1], a.grid[2], a.block[0], a.block[1], a.block[2],
                 row.launch_changes, a.indirect, a.split, a.known ? "potential_write_set" : "unknown",
                 Shader::StorageWriteReasonName(static_cast<Shader::StorageWriteReason>(a.reason)), a.begin, a.end,
                 a.intervals.size(), potential_bytes, a.applied,
                 row.marks, row.live_bytes, row.fragments, row.live_begin, row.live_end);
        for (size_t i = 0; i < (std::min)(a.intervals.size(), size_t{4}); ++i) {
            LOG_INFO(HW_GPU, "GPU_WRITE_TRACE owner_potential rank={} interval={} "
                            "relative=[{:#x},{:#x}) truncated={}", rank, i,
                     a.intervals[i].begin, a.intervals[i].end, a.intervals.size() > 4);
        }
        for (size_t i = 0; i < (std::min<u64>)(row.fragments, row.remaining_intervals.size()); ++i) {
            const auto& interval = row.remaining_intervals[i];
            LOG_INFO(HW_GPU, "GPU_WRITE_TRACE owner_remaining rank={} interval={} range=[{:#x},{:#x}) "
                            "truncated={}", rank, i, interval.begin, interval.end,
                     row.fragments > row.remaining_intervals.size());
        }
        for (size_t site = 0; site < SiteCount; ++site) {
            const auto& c = row.conflicts[site];
            if (!c.hits) continue;
            LOG_INFO(HW_GPU, "GPU_WRITE_TRACE owner_conflict rank={} site={} hits={} overlap={} "
                            "same64={} same4K={} unknown={} gap_max_B={} potential_overlap={} "
                            "outside_potential={} footprint_unknown={}",
                     rank, site_names[site], c.hits, c.overlap, c.line64, c.page4k, c.unknown, c.gap_max,
                     c.possible_overlap, c.outside_possible, c.footprint_unknown);
        }
        if (row.sample_fragment_end) {
            LOG_INFO(HW_GPU, "GPU_WRITE_TRACE owner_sample rank={} site={} cpu={:#x} nearest={:#x} "
                            "remaining_fragment_in_cpu_pages=[{:#x},{:#x})",
                     rank, site_names[static_cast<size_t>(row.sample_site)], row.sample_cpu,
                     row.sample_nearest, row.sample_fragment_begin, row.sample_fragment_end);
        }
        for (size_t reason = 0; reason < RemovalCount; ++reason) {
            if (!row.removed_bytes[reason]) continue;
            LOG_INFO(HW_GPU, "GPU_WRITE_TRACE owner_retire rank={} reason={} removed_B={}",
                     rank, removal_names[reason], row.removed_bytes[reason]);
        }
        if (row.last_removed_size) {
            LOG_INFO(HW_GPU, "GPU_WRITE_TRACE owner_last_retire rank={} reason={} request={:#x}+{:#x}",
                     rank, removal_names[static_cast<size_t>(row.last_removal)],
                     row.last_removed_address, row.last_removed_size);
        }
    }
    for (size_t i = 0; i < OriginCount; ++i) {
        const auto& m = snapshot.marks[i];
        LOG_INFO(HW_GPU, "GPU_WRITE_TRACE mark={} calls={} mark_Bps={:.1f} page_touches_ps={:.1f} "
                        "unique_4K_pages={} range_Bps={:.1f} live_B={} op_calls={} op_Bps={:.1f}",
                 origin_names[i], m.calls, m.bytes / seconds, m.page_touches / seconds,
                 m.unique_pages, m.range_bytes / seconds, m.live_bytes, m.operation_calls,
                 m.operation_bytes / seconds);
    }
    for (size_t s = 0; s < SiteCount; ++s) {
        const auto& site = snapshot.sites[s];
        if (!site.calls) {
            continue;
        }
        LOG_INFO(HW_GPU, "GPU_WRITE_TRACE cpu={} calls={} hit_calls={} no_live_range={} multi_origin={} "
                        "model_overlap={} model_outside={} (unknown_writers_keep_full_ranges)",
                 site_names[s], site.calls, site.hits, site.no_live_range, site.multi_origin,
                 site.model_overlap, site.model_outside);
        for (size_t o = 0; o < OriginCount; ++o) {
            const auto& c = snapshot.conflicts[s][o];
            if (!c.hits) {
                continue;
            }
            const u64 measured = c.hits - c.unknown;
            LOG_INFO(HW_GPU, "GPU_WRITE_TRACE conflict={} origin={} hits={} overlap={} same64={} "
                            "same4K={} unknown={} gap_avg_B={:.1f} gap_max_B={}",
                     site_names[s], origin_names[o], c.hits, c.overlap, c.line64, c.page4k, c.unknown,
                     measured ? static_cast<double>(c.gap_sum) / measured : 0.0, c.gap_max);
        }
    }
    for (size_t i = 0; i < snapshot.sample_count; ++i) {
        const auto& s = snapshot.samples[i];
        LOG_INFO(HW_GPU, "GPU_WRITE_TRACE sample={} origin={} addr={:#x} size={} nearest_mark={:#x} "
                        "gap_B={} exact={}", site_names[static_cast<size_t>(s.site)],
                 origin_names[static_cast<size_t>(s.origin)], s.cpu, s.size, s.marked, s.gap, s.exact);
    }
    std::array<u64, 8> top_hashes{};
    size_t hash_count{};
    for (const auto& shader : snapshot.shaders) {
        if (hash_count == top_hashes.size()) break;
        top_hashes[hash_count++] = shader.shader;
    }
    DumpTopComputeShaders(std::span<const u64>{top_hashes}.first(hash_count));
}
} // namespace VideoCommon::GpuWriteTrace
