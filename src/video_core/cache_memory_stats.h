// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <array>

#include "common/common_types.h"

namespace VideoCommon::CacheMemoryStats {

/// NovaS diagnostics for the Android memory line: what the caches hold, updated once a frame.
inline std::atomic<u64> texture_registered_base_bytes{0};
inline std::atomic<u64> texture_registered_scaled_bytes{0};
inline std::atomic<u64> texture_images{0};   ///< Images registered in it
inline std::atomic<u64> texture_created{0};  ///< Images created since the start
inline std::atomic<u64> texture_deleted{0};  ///< Images deleted since the start
inline std::atomic<u64> buffer_bytes{0};     ///< The buffer cache's buffers
inline std::atomic<u64> staging_bytes{0};    ///< Staging buffers the runtime keeps
inline std::atomic<u64> staging_free_bytes{0};
inline std::atomic<u64> staging_pending_bytes{0};
inline std::atomic<u64> staging_stream_bytes{0};
inline std::atomic<u64> staging_sample_ns{0};
// Age buckets in ticks: 0, 1-9, 10-24, 25-49, 50+.
inline std::array<std::atomic<u64>, 5> texture_age_bytes{};
// Current frame, decoding, GPU pending, age protected, eligible, policy protected.
inline std::array<std::atomic<u64>, 6> texture_blocked_images{};
inline std::array<std::atomic<u64>, 6> texture_blocked_bytes{};
inline std::atomic<u64> texture_deferred_bytes{0};
inline std::atomic<u64> texture_snapshot_ns{0};

} // namespace VideoCommon::CacheMemoryStats
