// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/gpu_write_footprint_profile.h"

namespace {
using namespace VideoCommon::GpuWriteTrace;
constexpr size_t Host = static_cast<size_t>(DownloadPath::HostRead);
Attribution Owner(u64 shader) {
    return {.shader = shader,
            .reason = static_cast<u8>(Shader::StorageWriteReason::DynamicOrUnsupportedOffset)};
}
u64 Bytes(const FootprintProfileSnapshot& snapshot, u64 shader, DownloadPath path = DownloadPath::HostRead) {
    u64 bytes{};
    for (const auto& row : snapshot.rows) {
        if (row.key.shader == shader) bytes += row.downloads[static_cast<size_t>(path)].bytes;
    }
    return bytes;
}
}

TEST_CASE("Footprint profile attributes overlapping ranges to the latest mark",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    FootprintProfiler profile;
    const auto a = Owner(1), b = Owner(2);
    profile.Mark(Origin::DispatchSsbo, 0x1000, 256, &a);
    profile.Mark(Origin::DispatchSsbo, 0x1040, 64, &b);
    FootprintProfiler::DownloadSample download{.profiler = &profile};
    download.Add(0x1000, 256);
    download.Commit(DownloadPath::HostRead, true);
    const auto snapshot = profile.TakeSnapshot();
    CHECK(Bytes(snapshot, 1) == 192);
    CHECK(Bytes(snapshot, 2) == 64);
    CHECK(snapshot.downloads[Host].operations == 1);
    CHECK(snapshot.downloads[Host].finish_operations == 1);
    CHECK(snapshot.downloads[Host].fragments == 1);
    u64 associations{};
    for (const auto& row : snapshot.rows) associations += row.downloads[Host].finish_operations;
    CHECK(associations == 2); // One Finish has two writer associations, not two Finishes.
}

TEST_CASE("Footprint profile freezes download attribution before retirement and later writes",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    FootprintProfiler profile;
    const auto a = Owner(1), b = Owner(2);
    profile.Mark(Origin::DispatchSsbo, 0x1000, 64, &a);
    FootprintProfiler::DownloadSample old{.profiler = &profile};
    old.Add(0x1000, 64);
    profile.Clear(0x1000, 64);
    profile.TakeSnapshot(); // A report must not invalidate a captured owner ID.
    profile.Mark(Origin::DispatchSsbo, 0x1000, 64, &b);
    old.Commit(DownloadPath::HostRead, true);
    FootprintProfiler::DownloadSample current{.profiler = &profile};
    current.Add(0x1000, 64);
    current.Commit(DownloadPath::HostRead, true);
    auto snapshot = profile.TakeSnapshot();
    CHECK(Bytes(snapshot, 1) == 64);
    CHECK(Bytes(snapshot, 2) == 64);
    CHECK(snapshot.downloads[Host].operations == 2);
    snapshot = profile.TakeSnapshot();
    CHECK(snapshot.downloads[Host].operations == 0);
    CHECK(snapshot.live_segments == 1);
}

TEST_CASE("Footprint profile sparse marks preserve older writers in holes",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    FootprintProfiler profile;
    const auto old = Owner(1);
    Attribution sparse{.shader = 2, .known = true, .begin = 128, .end = 248,
        .intervals = {{128, 140}, {244, 248}}, .applied = true,
        .decision = FootprintDecision::Applied};
    profile.Mark(Origin::DispatchSsbo, 0x1000, 512, &old);
    profile.Mark(Origin::DispatchSsbo, 0x1000, 512, &sparse);
    FootprintProfiler::DownloadSample download{.profiler = &profile};
    download.Add(0x1000, 512);
    download.Commit(DownloadPath::HostRead, true);
    const auto snapshot = profile.TakeSnapshot();
    CHECK(Bytes(snapshot, 1) == 496);
    CHECK(Bytes(snapshot, 2) == 16);
}

TEST_CASE("Footprint profile splits cleared bytes and replaces unattributed GPU writes",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    FootprintProfiler profile;
    const auto owner = Owner(1);
    profile.Mark(Origin::DispatchSsbo, 0x1000, 256, &owner);
    profile.Clear(0x1040, 64);
    profile.Mark(Origin::DmaCopy, 0x1080, 64, nullptr);
    FootprintProfiler::DownloadSample download{.profiler = &profile};
    download.Add(0x1000, 256);
    download.Commit(DownloadPath::HostRead, true);
    const auto snapshot = profile.TakeSnapshot();
    CHECK(Bytes(snapshot, 1) == 128);
    CHECK(Bytes(snapshot, 0) == 128); // Cleared history plus a new unattributed DMA mark.
}

TEST_CASE("Footprint profile repeated marks coalesce metadata but retain event costs",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    FootprintProfiler profile;
    const auto owner = Owner(1);
    profile.Mark(Origin::DispatchSsbo, 0x1000, 4096, &owner, 0, 0);
    for (unsigned i = 0; i < 100; ++i) {
        profile.Mark(Origin::DispatchSsbo, 0x1000, 4096, &owner, 4096, 16);
    }
    profile.Mark(Origin::DispatchSsbo, 0x2000, 4096, &owner, 0, 0);
    const auto snapshot = profile.TakeSnapshot();
    CHECK(snapshot.live_segments == 1);
    REQUIRE(snapshot.rows.size() == 2);
    const auto& row = snapshot.rows[1];
    CHECK(row.marks == 102);
    CHECK(row.declared_bytes == 102 * 4096);
    CHECK(row.repeated_bytes == 100 * 4096);
    CHECK(row.fully_dirty_marks == 100);
    CHECK(row.canceled_readback_bytes == 1600);
}

TEST_CASE("Footprint profile limits degrade to unknown without restoring an old writer",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    SECTION("segment limit") {
        FootprintProfiler profile{2, 16};
        const auto owner = Owner(1);
        for (u64 i = 0; i < 3; ++i) profile.Mark(Origin::DispatchSsbo, 0x1000 + i * 8, 4, &owner);
        FootprintProfiler::DownloadSample download{.profiler = &profile};
        for (u64 i = 0; i < 3; ++i) download.Add(0x1000 + i * 8, 4);
        download.Commit(DownloadPath::HostRead, true);
        const auto snapshot = profile.TakeSnapshot();
        CHECK(snapshot.range_overflow == 1);
        CHECK(Bytes(snapshot, 1) == 0);
        CHECK(Bytes(snapshot, 0) == 12);
    }
    SECTION("owner limit") {
        FootprintProfiler profile{16, 2};
        const auto a = Owner(1), b = Owner(2);
        profile.Mark(Origin::DispatchSsbo, 0x1000, 64, &a);
        profile.Mark(Origin::DispatchSsbo, 0x1000, 64, &b);
        FootprintProfiler::DownloadSample download{.profiler = &profile};
        download.Add(0x1000, 64);
        download.Commit(DownloadPath::HostRead, true);
        const auto snapshot = profile.TakeSnapshot();
        CHECK(snapshot.owner_overflow == 1);
        CHECK(Bytes(snapshot, 1) == 0);
        CHECK(Bytes(snapshot, 0) == 64);
    }
}

TEST_CASE("Footprint profile counts one page conflict per owner and selected site",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    FootprintProfiler profile;
    const auto owner = Owner(1);
    profile.Mark(Origin::DispatchSsbo, 0x1080, 4, &owner);
    profile.Mark(Origin::DispatchSsbo, 0x10c0, 4, &owner);
    profile.Cpu(Site::CpuTracker, 0x1000, 4, true);
    profile.Cpu(Site::Preserve, 0x1000, 4, true);
    const auto snapshot = profile.TakeSnapshot();
    REQUIRE(snapshot.rows.size() == 2);
    CHECK(snapshot.rows[1].conflicts == 2);
    CHECK(snapshot.rows[1].conflict_sites[static_cast<size_t>(Site::CpuTracker)] == 1);
    CHECK(snapshot.rows[1].conflict_sites[static_cast<size_t>(Site::Preserve)] == 1);
}

TEST_CASE("Footprint profile async copies and surviving retirement bytes are separate",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    FootprintProfiler profile;
    const auto a = Owner(1), b = Owner(2);
    profile.Mark(Origin::DispatchSsbo, 0x1000, 128, &a);
    FootprintProfiler::DownloadSample copy{.profiler = &profile};
    copy.Add(0x1000, 128);
    copy.Commit(DownloadPath::AsyncCopy);
    profile.Mark(Origin::DispatchSsbo, 0x1040, 64, &b);
    FootprintProfiler::DownloadSample retire{.profiler = &profile};
    retire.Add(0x1000, 64); // Production passes only surviving pending_download_ranges.
    profile.Clear(0x1000, 64);
    retire.Commit(DownloadPath::AsyncRetire);
    const auto snapshot = profile.TakeSnapshot();
    CHECK(Bytes(snapshot, 1, DownloadPath::AsyncCopy) == 128);
    CHECK(Bytes(snapshot, 1, DownloadPath::AsyncRetire) == 64);
    CHECK(Bytes(snapshot, 2, DownloadPath::AsyncRetire) == 0);
    CHECK(snapshot.downloads[static_cast<size_t>(DownloadPath::AsyncCopy)].finish_operations == 0);
}

TEST_CASE("Footprint profile interval updates match a bytewise ownership model",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    FootprintProfiler profile;
    std::array<u64, 64> expected{};
    u32 seed = 0x789a;
    const auto next = [&] { seed = seed * 1664525U + 1013904223U; return seed; };
    for (unsigned step = 0; step < 256; ++step) {
        const size_t start = next() % expected.size();
        const size_t length = 1 + next() % (expected.size() - start);
        const u64 shader = next() % 7;
        if (shader) {
            const auto owner = Owner(shader);
            profile.Mark(Origin::DispatchSsbo, 0x1000 + start, length, &owner);
        } else {
            profile.Clear(0x1000 + start, length);
        }
        for (size_t i = start; i < start + length; ++i) expected[i] = shader;
        if (step % 8 != 0) continue;
        FootprintProfiler::DownloadSample download{.profiler = &profile};
        download.Add(0x1000, expected.size());
        download.Commit(DownloadPath::HostRead, true);
        const auto snapshot = profile.TakeSnapshot();
        for (u64 id = 0; id < 7; ++id) {
            u64 count{};
            for (u64 byte : expected) count += byte == id;
            CHECK(Bytes(snapshot, id) == count);
        }
    }
}

TEST_CASE("Footprint download phases accumulate once globally and reset with the window",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    using Phase = DownloadPhase;
    using Sample = FootprintProfiler::DownloadSample;
    using Ns = std::chrono::nanoseconds;
    FootprintProfiler profile;
    const auto a = Owner(1), b = Owner(2);
    profile.Mark(Origin::DispatchSsbo, 0x1000, 16, &a);
    profile.Mark(Origin::DispatchSsbo, 0x1010, 16, &b);
    Sample sample{.profiler = &profile};
    const Sample::Clock::time_point start{};
    sample.BeginPhase(Phase::Planning, start);
    sample.Add(0x1000, 32);
    sample.BeginPhase(Phase::CopyRecord, start + Ns{100});
    sample.AddCopies(2, 32);
    sample.BeginPhase(Phase::FinishWait, start + Ns{300});
    sample.BeginPhase(Phase::Writeback, start + Ns{600});
    sample.EndPhase(start + Ns{1000});
    sample.Commit(DownloadPath::HostRead, true);

    // Reusing the sample starts a new operation, without carrying its old durations or copies.
    sample.BeginPhase(Phase::Planning, start);
    sample.Add(0x1000, 32);
    sample.BeginPhase(Phase::Planning, start + Ns{50});
    sample.EndPhase(start + Ns{100});
    sample.AddCopies(5, 32);
    sample.Commit(DownloadPath::HostRead, true);
    sample.Commit(DownloadPath::HostRead, true); // An empty commit is not another Finish.
    const auto snapshot = profile.TakeSnapshot();
    const auto& total = snapshot.downloads[Host];
    CHECK(total.operations == 2);
    CHECK(total.finish_operations == 2);
    CHECK(total.fragments == 2);
    CHECK(total.bytes == 64);
    CHECK(total.copy_regions == 7);
    CHECK(total.transfer_bytes == 64);
    CHECK(total.finish_copy_regions == 7);
    CHECK(total.max_copy_regions_per_finish == 5);
    CHECK(total.finish_fragments == 2);
    CHECK(total.max_fragments_per_finish == 1);
    CHECK(total.phase_ns[static_cast<size_t>(Phase::Planning)] == 200);
    CHECK(total.phase_ns[static_cast<size_t>(Phase::CopyRecord)] == 200);
    CHECK(total.phase_ns[static_cast<size_t>(Phase::FinishWait)] == 300);
    CHECK(total.phase_ns[static_cast<size_t>(Phase::Writeback)] == 400);
    CHECK(Bytes(snapshot, 1) == 32);
    CHECK(Bytes(snapshot, 2) == 32);
    u64 associations{};
    for (const auto& row : snapshot.rows) associations += row.downloads[Host].finish_operations;
    CHECK(associations == 4);

    const auto empty = profile.TakeSnapshot();
    CHECK(empty.downloads[Host].copy_regions == 0);
    CHECK(empty.downloads[Host].transfer_bytes == 0);
    CHECK(empty.downloads[Host].finish_copy_regions == 0);
    CHECK(empty.downloads[Host].max_copy_regions_per_finish == 0);
    CHECK(empty.downloads[Host].finish_fragments == 0);
    CHECK(empty.downloads[Host].max_fragments_per_finish == 0);
    for (const auto duration : empty.downloads[Host].phase_ns) CHECK(duration == 0);
}

TEST_CASE("Footprint download copy metrics distinguish asynchronous copies and mapped reads",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    using Sample = FootprintProfiler::DownloadSample;
    using Ns = std::chrono::nanoseconds;
    FootprintProfiler profile;
    Sample sample{.profiler = &profile};
    const Sample::Clock::time_point start{};
    sample.BeginPhase(DownloadPhase::CopyRecord, start);
    sample.Add(0x1000, 96);
    sample.AddCopies(3, 96);
    sample.EndPhase(start + Ns{100});
    sample.Commit(DownloadPath::AsyncCopy);
    sample.Add(0x1000, 32);
    sample.Add(0x1020, 64);
    sample.Commit(DownloadPath::HostRead, true); // A mapped read records no transfer command.
    const auto snapshot = profile.TakeSnapshot();
    const auto& async = snapshot.downloads[static_cast<size_t>(DownloadPath::AsyncCopy)];
    CHECK(async.copy_regions == 3);
    CHECK(async.transfer_bytes == 96);
    CHECK(async.finish_copy_regions == 0);
    CHECK(async.max_copy_regions_per_finish == 0);
    CHECK(async.finish_fragments == 0);
    CHECK(async.max_fragments_per_finish == 0);
    CHECK(async.phase_ns[static_cast<size_t>(DownloadPhase::CopyRecord)] == 100);
    CHECK(snapshot.downloads[Host].finish_operations == 1);
    CHECK(snapshot.downloads[Host].copy_regions == 0);
    CHECK(snapshot.downloads[Host].transfer_bytes == 0);
    CHECK(snapshot.downloads[Host].max_copy_regions_per_finish == 0);
    CHECK(snapshot.downloads[Host].finish_fragments == 2);
    CHECK(snapshot.downloads[Host].max_fragments_per_finish == 2);
}

TEST_CASE("Footprint download disabled and empty samples retain no timing or copy metrics",
          "[gpu_write_trace][gpu_write_footprint_profile]") {
    using Sample = FootprintProfiler::DownloadSample;
    Sample disabled;
    disabled.BeginPhase(DownloadPhase::Planning);
    disabled.Add(0x1000, 4);
    disabled.AddCopies(1, 4);
    disabled.EndPhase();
    disabled.Commit(DownloadPath::Staging, true);
    CHECK(disabled.active_phase == DownloadPhase::Count);
    CHECK(disabled.bytes == 0);
    CHECK(disabled.copy_regions == 0);
    for (const auto duration : disabled.phase_ns) CHECK(duration == 0);

    FootprintProfiler profile;
    Sample empty{.profiler = &profile};
    const Sample::Clock::time_point start{};
    empty.BeginPhase(DownloadPhase::Planning, start);
    empty.EndPhase(start + std::chrono::nanoseconds{100});
    empty.Commit(DownloadPath::Staging);
    empty.Add(0x1000, 4);
    empty.Commit(DownloadPath::Staging, true);
    const auto snapshot = profile.TakeSnapshot();
    const auto& staging = snapshot.downloads[static_cast<size_t>(DownloadPath::Staging)];
    CHECK(staging.operations == 1);
    for (const auto duration : staging.phase_ns) CHECK(duration == 0);
}
