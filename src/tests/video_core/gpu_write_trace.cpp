// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/gpu_write_trace.h"

namespace {
using namespace VideoCommon::GpuWriteTrace;

constexpr size_t Index(Origin value) {
    return static_cast<size_t>(value);
}
constexpr size_t Index(Site value) {
    return static_cast<size_t>(value);
}
} // namespace

TEST_CASE("GPU write trace separates repeated marks from live bytes and unique pages",
          "[gpu_write_trace]") {
    Tracker trace;
    trace.Mark(Origin::DrawSsbo, 0x1ff0, 32);
    trace.Mark(Origin::DrawSsbo, 0x1ff0, 32);
    const auto first = trace.TakeSnapshot().marks[Index(Origin::DrawSsbo)];
    CHECK(first.calls == 2);
    CHECK(first.bytes == 64);
    CHECK(first.range_bytes == 64);
    CHECK(first.page_touches == 4);
    CHECK(first.unique_pages == 2);
    CHECK(first.live_bytes == 32);

    const auto next = trace.TakeSnapshot().marks[Index(Origin::DrawSsbo)];
    CHECK(next.calls == 0);
    CHECK(next.unique_pages == 0);
    CHECK(next.live_bytes == 32);
}

TEST_CASE("GPU write trace distinguishes overlap line sharing and distant bytes",
          "[gpu_write_trace]") {
    Tracker trace;
    trace.Mark(Origin::DrawSsbo, 0x1100, 4);
    trace.Cpu(Site::CpuExact, 0x1102, 4, true); // overlap
    trace.Cpu(Site::CpuExact, 0x1108, 4, true); // nearest byte 0x1103: gap 5
    trace.Cpu(Site::CpuExact, 0x1f00, 4, true); // same page, gap 3581
    trace.Cpu(Site::CpuExact, 0x2000, 4, true); // different page
    const auto snapshot = trace.TakeSnapshot();
    const auto& c = snapshot.conflicts[Index(Site::CpuExact)][Index(Origin::DrawSsbo)];
    CHECK(c.hits == 3);
    CHECK(c.overlap == 1);
    CHECK(c.line64 == 1);
    CHECK(c.page4k == 1);
    CHECK(c.gap_sum == 3586);
    CHECK(c.gap_max == 3581);
    CHECK(snapshot.sites[Index(Site::CpuExact)].no_live_range == 1);
}

TEST_CASE("GPU write trace does not turn a collected line into an exact guest store",
          "[gpu_write_trace]") {
    Tracker trace;
    trace.Mark(Origin::DrawSsbo, 0x103f, 1);
    trace.Cpu(Site::CpuExact, 0x103b, 4, true);
    trace.Cpu(Site::CachedCollected, 0x1000, 64, true);
    trace.Cpu(Site::NceStorePage, 0x1000, 4096, false);
    const auto snapshot = trace.TakeSnapshot();
    const auto& exact = snapshot.conflicts[Index(Site::CpuExact)][Index(Origin::DrawSsbo)];
    CHECK(exact.overlap == 0);
    CHECK(exact.line64 == 1);
    CHECK(exact.gap_max == 1);
    CHECK(snapshot.conflicts[Index(Site::CachedCollected)][Index(Origin::DrawSsbo)].overlap == 1);
    const auto& page = snapshot.conflicts[Index(Site::NceStorePage)][Index(Origin::DrawSsbo)];
    CHECK(page.unknown == 1);
    CHECK(page.overlap == 0);
    CHECK(page.gap_sum == 0);
}

TEST_CASE("GPU write trace removes cleared provenance from every origin",
          "[gpu_write_trace]") {
    Tracker trace;
    trace.Mark(Origin::DrawSsbo, 0x1000, 16);
    trace.Mark(Origin::DispatchSsbo, 0x1008, 16);
    trace.Cpu(Site::CpuExact, 0x100c, 4, true);
    trace.Clear(0x1008, 16);
    trace.Cpu(Site::CpuExact, 0x100c, 4, true);
    const auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.sites[Index(Site::CpuExact)].multi_origin == 1);
    CHECK(snapshot.conflicts[Index(Site::CpuExact)][Index(Origin::DrawSsbo)].hits == 2);
    CHECK(snapshot.conflicts[Index(Site::CpuExact)][Index(Origin::DispatchSsbo)].hits == 1);
    CHECK(snapshot.marks[Index(Origin::DrawSsbo)].live_bytes == 8);
    CHECK(snapshot.marks[Index(Origin::DispatchSsbo)].live_bytes == 0);

    trace.Clear(0x1000, 4096);
    trace.Cpu(Site::CpuExact, 0x1000, 4, true);
    CHECK(trace.TakeSnapshot().sites[Index(Site::CpuExact)].no_live_range == 1);
}

TEST_CASE("GPU write trace keeps DMA tracker range distinct from dirty byte intervals",
          "[gpu_write_trace]") {
    Tracker trace;
    trace.AddRange(Origin::DmaCopy, 0x1010, 4);
    trace.Mark(Origin::DmaCopy, 0x1000, 8192, false);
    trace.Cpu(Site::CpuTracker, 0x2000, 4, true);
    const auto snapshot = trace.TakeSnapshot();
    const auto& mark = snapshot.marks[Index(Origin::DmaCopy)];
    CHECK(mark.bytes == 8192);
    CHECK(mark.range_bytes == 4);
    CHECK(mark.live_bytes == 4);
    CHECK(mark.unique_pages == 2);
    CHECK(snapshot.sites[Index(Site::CpuTracker)].no_live_range == 1);
}

TEST_CASE("GPU write trace clear and query operations do not create dirty ownership",
          "[gpu_write_trace]") {
    Tracker trace;
    trace.Operation(Origin::DmaClear, 4096);
    trace.Operation(Origin::Query, 8);
    trace.Cpu(Site::CpuExact, 0x1000, 4, true);
    const auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.marks[Index(Origin::DmaClear)].operation_bytes == 4096);
    CHECK(snapshot.marks[Index(Origin::Query)].operation_calls == 1);
    CHECK(snapshot.marks[Index(Origin::Query)].bytes == 0);
    CHECK(snapshot.marks[Index(Origin::DmaClear)].live_bytes == 0);
    CHECK(snapshot.sites[Index(Site::CpuExact)].no_live_range == 1);
}

TEST_CASE("GPU write trace clips marks to the pages touched by a CPU access",
          "[gpu_write_trace]") {
    Tracker trace;
    trace.Mark(Origin::TransformFeedback, 0x1fff, 8194);
    trace.Cpu(Site::CpuExact, 0x2ffe, 4, true);
    trace.Cpu(Site::CpuExact, 0x1000, 1, true);
    const auto snapshot = trace.TakeSnapshot();
    const auto& c = snapshot.conflicts[Index(Site::CpuExact)][Index(Origin::TransformFeedback)];
    CHECK(c.hits == 2);
    CHECK(c.overlap == 1);
    CHECK(c.page4k == 1);
    CHECK(c.gap_max == 4095);
}

TEST_CASE("GPU write trace line sharing does not depend on the nearest byte",
          "[gpu_write_trace]") {
    Tracker trace;
    trace.Mark(Origin::DrawSsbo, 0x103f, 1);
    trace.Mark(Origin::DrawSsbo, 0x1042, 1);
    trace.Cpu(Site::CpuExact, 0x1040, 1, true);
    const auto snapshot = trace.TakeSnapshot();
    const auto& c = snapshot.conflicts[Index(Site::CpuExact)][Index(Origin::DrawSsbo)];
    CHECK(c.line64 == 1);
    CHECK(c.page4k == 0);
    CHECK(c.gap_max == 1);
}

TEST_CASE("GPU write attribution retains original range and explains removed bytes",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 0x1234, .binding = 3, .grid = {8, 1, 1},
                      .size_source = SizeSource::Descriptor};
    trace.Mark(Origin::DispatchSsbo, 0x1000, 32, true, &owner);
    trace.Clear(0x1008, 24, Removal::InlineUpload);
    trace.Cpu(Site::CpuExact, 0x1010, 4, true);
    const auto snapshot = trace.TakeSnapshot();
    REQUIRE(snapshot.top.size() == 1);
    const auto& row = snapshot.top[0];
    CHECK(row.identity.shader == 0x1234);
    CHECK(row.identity.binding == 3);
    CHECK(row.original == 0x1000);
    CHECK(row.size == 32);
    CHECK(row.live_bytes == 8);
    CHECK(row.live_end == 0x1008);
    CHECK(row.remaining_intervals[0].begin == 0x1000);
    CHECK(row.remaining_intervals[0].end == 0x1008);
    CHECK(row.removed_bytes[static_cast<size_t>(Removal::InlineUpload)] == 24);
    CHECK(row.last_removal == Removal::InlineUpload);
    CHECK(row.conflicts[Index(Site::CpuExact)].hits == 1);
    CHECK(row.sample_nearest == 0x1007);
    CHECK(row.sample_fragment_end == 0x1008);
}

TEST_CASE("GPU write footprint estimate respects unknown writers on the same page",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 1, .known = true, .begin = 0, .end = 16};
    trace.Mark(Origin::DispatchSsbo, 0x1000, 0x3000, true, &owner);
    trace.Mark(Origin::DrawSsbo, 0x2001, 1); // Unattributed: conservatively keep this page.
    const auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.empty_pages_all_proven == 1);
    REQUIRE(snapshot.shaders.size() == 1);
    CHECK(snapshot.shaders[0].empty_pages_alone == 1);
    CHECK(snapshot.shaders[0].proven_marks == 1);
}

TEST_CASE("GPU write footprint estimate separates a shader alone from cooperative savings",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution a{.shader = 1, .known = true, .begin = 0, .end = 16};
    auto b = a;
    b.shader = 2;
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &a);
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &b);
    trace.Cpu(Site::CpuExact, 0x2000, 4, true);
    trace.Cpu(Site::CpuExact, 0x2004, 4, true);
    auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.empty_pages_all_proven == 1);
    REQUIRE(snapshot.shaders.size() == 2);
    for (const auto& s : snapshot.shaders) {
        CHECK(s.empty_pages_alone == 0);
        CHECK(s.empty_pages_all_proven == 1);
        CHECK(s.conflict_pages_empty_all_proven == 1); // unique page, not store count
    }
    snapshot = trace.TakeSnapshot();
    CHECK(snapshot.empty_pages_all_proven == 1);
    CHECK(snapshot.shaders[0].conflict_pages_empty_all_proven == 0);
}

TEST_CASE("GPU write footprint estimate retains earlier writes and rejects incomplete proofs",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 1, .known = true, .begin = 0, .end = 16};
    SECTION("previous write has a wider footprint") {
        auto old = owner;
        old.end = 8192;
        trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &old);
    }
    SECTION("split binding") { owner.split = true; }
    SECTION("footprint exceeds the resolved binding") { owner.end = 8193; }
    SECTION("unknown footprint") { owner.known = false; }
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    CHECK(trace.TakeSnapshot().empty_pages_all_proven == 0);
}

TEST_CASE("GPU write attribution reindexes cleared pages and keeps last launch dimensions",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 1, .grid = {1, 1, 1}};
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    trace.Clear(0x2000, 4096, Removal::Download);
    trace.Cpu(Site::CpuExact, 0x2000, 4, true);
    owner.grid[0] = 8;
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    trace.Cpu(Site::CpuExact, 0x2000, 4, true);
    auto snapshot = trace.TakeSnapshot();
    REQUIRE(snapshot.top.size() == 1);
    CHECK(snapshot.top[0].identity.grid[0] == 8);
    CHECK(snapshot.top[0].launch_changes == 1);
    CHECK(snapshot.top[0].conflicts[Index(Site::CpuExact)].hits == 1);
    trace.Clear(0x1000, 8192, Removal::CpuWrite);
    trace.TakeSnapshot();
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    trace.Cpu(Site::CpuExact, 0x2000, 4, true);
    snapshot = trace.TakeSnapshot();
    REQUIRE(snapshot.top.size() == 1);
    CHECK(snapshot.top[0].conflicts[Index(Site::CpuExact)].hits == 1);
}

TEST_CASE("GPU write attribution overflow keeps pages dirty and retired capacity is reusable",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.known = true, .begin = 0, .end = 4};
    for (u64 shader = 1; shader <= 4097; ++shader) {
        owner.shader = shader;
        trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    }
    auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.attribution_overflow == 1);
    CHECK(snapshot.empty_pages_all_proven == 0);
    trace.Clear(0x1000, 8192, Removal::Download);
    trace.TakeSnapshot();
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    snapshot = trace.TakeSnapshot();
    CHECK(snapshot.attribution_overflow == 0);
    CHECK(snapshot.empty_pages_all_proven == 1);
}

TEST_CASE("GPU write trace preserves holes in a complete potential store set",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 1, .known = true, .begin = 128, .end = 248,
                      .intervals = {{128, 140}, {144, 156}, {160, 172}, {244, 248}}};
    trace.Mark(Origin::DispatchSsbo, 0x1000, 256, true, &owner);
    trace.Cpu(Site::CpuExact, 0x1000, 128, true);
    trace.Cpu(Site::CpuExact, 0x108c, 4, true); // [140,144): hole between stores
    trace.Cpu(Site::CpuExact, 0x1088, 4, true); // potential write
    auto snapshot = trace.TakeSnapshot();
    REQUIRE(snapshot.top.size() == 1);
    const auto& c = snapshot.top[0].conflicts[Index(Site::CpuExact)];
    CHECK(c.overlap == 3); // All overlap the original declared binding.
    CHECK(c.outside_possible == 2);
    CHECK(c.possible_overlap == 1);
    REQUIRE(snapshot.shaders.size() == 1);
    CHECK(snapshot.shaders[0].outside_possible == 2);
    CHECK(snapshot.shaders[0].possible_overlap == 1);
    CHECK(snapshot.shaders[0].footprint_unknown == 0);
    CHECK(snapshot.sites[Index(Site::CpuExact)].model_outside == 2);
    CHECK(snapshot.empty_pages_all_proven == 0); // Holes do not empty this page.
}

TEST_CASE("GPU write trace sparse footprint changes retain previous dispatch writes",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 1, .grid = {1, 1, 1}, .known = true, .begin = 0, .end = 4,
                      .intervals = {{0, 4}}};
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    owner.grid[0] = 2;
    owner.end = 4100;
    owner.intervals.push_back({4096, 4100});
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    // Going back to the shorter launch does not retract the older write in the second page.
    owner.grid[0] = 1;
    owner.end = 4;
    owner.intervals.pop_back();
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    trace.Cpu(Site::CpuExact, 0x2000, 4, true);
    const auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.empty_pages_all_proven == 0);
    CHECK(snapshot.sites[Index(Site::CpuExact)].model_overlap == 1);
    REQUIRE(snapshot.top.size() == 2);
}

TEST_CASE("GPU write trace sparse outside result respects other unknown writers",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 1, .known = true, .begin = 128, .end = 132,
                      .intervals = {{128, 132}}};
    trace.Mark(Origin::DispatchSsbo, 0x1000, 256, true, &owner);
    trace.Mark(Origin::DrawSsbo, 0x1000, 4);
    trace.Cpu(Site::CpuExact, 0x1000, 4, true);
    const auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.top[0].conflicts[Index(Site::CpuExact)].outside_possible == 1);
    CHECK(snapshot.sites[Index(Site::CpuExact)].model_outside == 0);
    CHECK(snapshot.sites[Index(Site::CpuExact)].model_overlap == 1);
}

TEST_CASE("GPU write trace applied footprints track actual intervals and keep mixed pages",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 1, .known = true, .begin = 128, .end = 4236,
                      .intervals = {{128, 140}, {4224, 4236}}, .applied = true};
    trace.Mark(Origin::DispatchSsbo, 0x1000, 12288, true, &owner);
    trace.Cpu(Site::CpuExact, 0x1000, 4, true); // Hole, but neighbours keep this page GPU-owned.
    trace.Cpu(Site::CpuExact, 0x3000, 4, true); // No write on this page at all.
    const auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.marks[Index(Origin::DispatchSsbo)].live_bytes == 24);
    CHECK(snapshot.marks[Index(Origin::DispatchSsbo)].bytes == 24);
    CHECK(snapshot.marks[Index(Origin::DispatchSsbo)].unique_pages == 2);
    CHECK(snapshot.sites[Index(Site::CpuExact)].hits == 1);
    CHECK(snapshot.sites[Index(Site::CpuExact)].no_live_range == 1);
    CHECK(snapshot.sites[Index(Site::CpuExact)].model_outside == 1);
    REQUIRE(snapshot.top.size() == 1);
    CHECK(snapshot.top[0].live_bytes == 24);
    CHECK(snapshot.top[0].identity.applied);
}

TEST_CASE("GPU write trace applied footprint does not erase older or aliased GPU writes",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 1, .known = true, .begin = 0, .end = 4,
                      .intervals = {{0, 4}}, .applied = true};
    SECTION("older footprint of the same shader") {
        auto old = owner;
        old.begin = 4096;
        old.end = 4100;
        old.intervals = {{4096, 4100}};
        trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &old);
    }
    SECTION("unknown alias in a different binding") {
        Attribution other{.shader = 2, .binding = 1};
        trace.Mark(Origin::DispatchSsbo, 0x2000, 4, true, &other);
    }
    trace.Mark(Origin::DispatchSsbo, 0x1000, 8192, true, &owner);
    trace.Clear(0x1000, 4, Removal::Download);
    trace.Cpu(Site::CpuExact, 0x2000, 4, true);
    const auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.marks[Index(Origin::DispatchSsbo)].live_bytes == 4);
    CHECK(snapshot.sites[Index(Site::CpuExact)].model_overlap == 1);
}

TEST_CASE("GPU write trace malformed applied footprints fail closed",
          "[gpu_write_trace]") {
    Tracker trace;
    Attribution owner{.shader = 1, .known = true, .begin = 0, .end = 4,
                      .intervals = {{0, 4}}, .applied = true};
    SECTION("empty") { owner.intervals.clear(); }
    SECTION("out of bounds") { owner.intervals = {{4096, 4100}}; }
    SECTION("split") { owner.split = true; }
    SECTION("unknown") { owner.known = false; }
    trace.Mark(Origin::DispatchSsbo, 0x1000, 4096, true, &owner);
    const auto snapshot = trace.TakeSnapshot();
    CHECK(snapshot.marks[Index(Origin::DispatchSsbo)].live_bytes == 4096);
    REQUIRE(snapshot.top.size() == 1);
    CHECK_FALSE(snapshot.top[0].identity.applied);
    CHECK(snapshot.top[0].live_bytes == 4096);
}
