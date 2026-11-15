// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace VideoCommon::ComputeReleaseTrace {

bool Enabled();
u64 Begin(u64 asid, s32 channel, u64 qmd, u64 program);
void Shader(u64 dispatch, u64 hash, u64 tick);
void Range(u64 dispatch, u64 gpu, u64 device, u64 size, const char* kind, u32 binding,
           bool written, bool readable = true);
u64 Release(u64 dispatch, u64 gpu, u64 device, u32 payload, u32 control, const char* source);
void Published(u64 release, u32 value, const char* site = "callback");
void Discarded(u64 release, const char* reason);
// Observations describe accesses seen by the caller's hook, not all guest memory accesses.
void ObserveCpu(u64 device, u64 size, bool write, u64 cpu_va, u32 core, const char* site);
void ObserveAcquire(u64 device, u64 gpu, u64 asid, s32 channel, u32 value, const char* site,
                    bool value_known = true);

inline thread_local u64 current_release{};

} // namespace VideoCommon::ComputeReleaseTrace
