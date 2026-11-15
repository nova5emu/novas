// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "common/common_types.h"

namespace VideoCommon::CpuEpochAdmission {

enum class Reason : u32 {
    Unclassified = 1U << 0,
    SpatialOverlapOrUnknown = 1U << 1,
    Alias = 1U << 2,
    IdentityUnproven = 1U << 3,
    PendingReadbacksOrUnknown = 1U << 4,
    GpuDataUnresolved = 1U << 5,
    WriterBarrierMissing = 1U << 6,
    DownloadRequired = 1U << 7,
};
constexpr u32 Bit(Reason reason) {
    return static_cast<u32>(reason);
}

struct Inputs {
    bool classified{};
    bool no_spatial_overlap{};
    bool observed_alias{};
    bool identity_proven{};
    bool no_pending_readbacks{};
    bool gpu_dirty{};
    bool writer_barrier_ready{};
    bool gpu_data_resolved{};
};
struct Result {
    u32 reasons{};
    bool observed_candidate{};
    bool requires_download{};
    bool can_open{};
};

// Audit only. Proof inputs must describe the current identity and actual writer barrier;
// historical absence of GPU writes or a Ready journal token does not establish either.
constexpr Result Evaluate(const Inputs& input) {
    Result result;
    result.observed_candidate =
        input.classified && input.no_spatial_overlap && !input.observed_alias;
    result.requires_download = input.gpu_dirty && !input.gpu_data_resolved;
    if (!input.classified) {
        result.reasons |= Bit(Reason::Unclassified);
    }
    if (!input.no_spatial_overlap) {
        result.reasons |= Bit(Reason::SpatialOverlapOrUnknown);
    }
    if (input.observed_alias) {
        result.reasons |= Bit(Reason::Alias);
    }
    if (!input.identity_proven) {
        result.reasons |= Bit(Reason::IdentityUnproven);
    }
    if (!input.no_pending_readbacks) {
        result.reasons |= Bit(Reason::PendingReadbacksOrUnknown);
    }
    if (!input.gpu_data_resolved) {
        result.reasons |= Bit(Reason::GpuDataUnresolved);
    }
    if (!input.writer_barrier_ready) {
        result.reasons |= Bit(Reason::WriterBarrierMissing);
    }
    if (result.requires_download) {
        result.reasons |= Bit(Reason::DownloadRequired);
    }
    result.can_open = result.reasons == 0;
    return result;
}

} // namespace VideoCommon::CpuEpochAdmission
