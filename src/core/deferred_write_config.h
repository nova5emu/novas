// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

namespace Core::DeferredWritesConfig {

constexpr bool IsEnabled(const char* value) {
    return value && value[0] == '1' && value[1] == '\0';
}

constexpr bool ProducerRequested(const char* journal, const char* producer) {
    return IsEnabled(journal) && IsEnabled(producer);
}

constexpr bool CpuMappingsRequested(const char* journal, const char* producer,
                                    const char* cpu_epochs) {
    return ProducerRequested(journal, producer) || IsEnabled(cpu_epochs);
}

// Read at CPU/JIT construction, not from a signal handler or on every guest store.
inline bool ProducerRequested() {
    return ProducerRequested(std::getenv("EDEN_NX2_DEFERRED_WRITES"),
                             std::getenv("EDEN_NX2_DEFERRED_PRODUCER"));
}

inline bool CpuMappingsRequested() {
    return CpuMappingsRequested(std::getenv("EDEN_NX2_DEFERRED_WRITES"),
                                std::getenv("EDEN_NX2_DEFERRED_PRODUCER"),
                                std::getenv("EDEN_NX2_CPU_EPOCHS"));
}

inline bool CpuEpochsRequested() {
    return IsEnabled(std::getenv("EDEN_NX2_CPU_EPOCHS"));
}

} // namespace Core::DeferredWritesConfig
