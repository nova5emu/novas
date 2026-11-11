// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

namespace VideoCommon::CacheMemoryStats {

/// NovaS diagnostics for the Android memory line: what the caches hold, updated once a frame.
inline std::atomic<u64> texture_bytes{0};    ///< The texture cache's device memory
inline std::atomic<u64> texture_images{0};   ///< Images registered in it
inline std::atomic<u64> texture_created{0};  ///< Images created since the start
inline std::atomic<u64> texture_deleted{0};  ///< Images deleted since the start
inline std::atomic<u64> buffer_bytes{0};     ///< The buffer cache's buffers
inline std::atomic<u64> staging_bytes{0};    ///< Staging buffers the runtime keeps

} // namespace VideoCommon::CacheMemoryStats
