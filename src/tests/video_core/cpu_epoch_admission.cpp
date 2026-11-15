// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include "video_core/cpu_epoch_admission.h"

using namespace VideoCommon::CpuEpochAdmission;

TEST_CASE("CPU epoch audit keeps observed cohorts visible despite missing independent proofs",
          "[video_core][cpu_epoch_admission]") {
    Inputs input;
    input.classified = true;
    input.no_spatial_overlap = true;
    const auto result = Evaluate(input);
    REQUIRE(result.observed_candidate);
    REQUIRE_FALSE(result.can_open);
    REQUIRE(result.reasons ==
            (Bit(Reason::IdentityUnproven) | Bit(Reason::PendingReadbacksOrUnknown) |
             Bit(Reason::GpuDataUnresolved) | Bit(Reason::WriterBarrierMissing)));
    REQUIRE_FALSE(result.requires_download);
}

TEST_CASE("CPU epoch audit reports all blockers without early return",
          "[video_core][cpu_epoch_admission]") {
    Inputs input;
    input.observed_alias = true;
    input.gpu_dirty = true;
    const auto result = Evaluate(input);
    REQUIRE_FALSE(result.observed_candidate);
    REQUIRE_FALSE(result.can_open);
    REQUIRE(result.requires_download);
    REQUIRE(result.reasons ==
            (Bit(Reason::Unclassified) | Bit(Reason::SpatialOverlapOrUnknown) | Bit(Reason::Alias) |
             Bit(Reason::IdentityUnproven) | Bit(Reason::PendingReadbacksOrUnknown) |
             Bit(Reason::GpuDataUnresolved) | Bit(Reason::WriterBarrierMissing) |
             Bit(Reason::DownloadRequired)));
}

TEST_CASE("CPU epoch audit demands actual GPU data resolution even without historical dirtiness",
          "[video_core][cpu_epoch_admission]") {
    Inputs input{true, true, false, true, true, false, true, false};
    REQUIRE_FALSE(Evaluate(input).can_open);
    REQUIRE(Evaluate(input).reasons == Bit(Reason::GpuDataUnresolved));
    REQUIRE_FALSE(Evaluate(input).requires_download);
    input.gpu_dirty = true;
    REQUIRE(Evaluate(input).requires_download);
    input.gpu_data_resolved = true;
    REQUIRE_FALSE(Evaluate(input).requires_download);
    REQUIRE(Evaluate(input).can_open);
    REQUIRE(Evaluate(input).reasons == 0);
    input.writer_barrier_ready = false;
    REQUIRE_FALSE(Evaluate(input).can_open);
    REQUIRE(Evaluate(input).reasons == Bit(Reason::WriterBarrierMissing));
}

TEST_CASE("CPU epoch audit permits opening only with every independent proof",
          "[video_core][cpu_epoch_admission]") {
    const Inputs complete{true, true, false, true, true, false, true, true};
    REQUIRE(Evaluate(complete).can_open);
    for (u32 missing = 0; missing < 7; ++missing) {
        auto input = complete;
        switch (missing) {
        case 0:
            input.classified = false;
            break;
        case 1:
            input.no_spatial_overlap = false;
            break;
        case 2:
            input.observed_alias = true;
            break;
        case 3:
            input.identity_proven = false;
            break;
        case 4:
            input.no_pending_readbacks = false;
            break;
        case 5:
            input.writer_barrier_ready = false;
            break;
        case 6:
            input.gpu_data_resolved = false;
            break;
        }
        REQUIRE_FALSE(Evaluate(input).can_open);
        REQUIRE(Evaluate(input).reasons != 0);
    }
}
