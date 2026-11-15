// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

namespace Common::CallerStacks {

/// NovaS diagnostics (Android only; nothing elsewhere): samples the call stack of `site` once
/// every `every` calls and logs the most frequent stacks every 10 s as offsets in this library,
/// for llvm-symbolizer against the unstripped library of the same build.
void Sample(const char* site, std::size_t every = 8);

} // namespace Common::CallerStacks
