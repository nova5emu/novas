// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string>
#include "common/common_types.h"

namespace Shader::IR {
struct Program;
}

namespace VideoCommon::GpuWriteTrace {
// Called only with GPU_WRITE_TRACE enabled, outside signals. No files until top selection.
void ResetComputeDumps(bool active);
std::string PrepareComputeDump(const Shader::IR::Program& program);
void RegisterComputeDump(u64 shader, u64 pipeline, std::string ir, std::span<const u32> spirv);
void DumpTopComputeShaders(std::span<const u64> hashes);
} // namespace VideoCommon::GpuWriteTrace
