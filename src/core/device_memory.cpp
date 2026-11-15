// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/deferred_write_config.h"
#include "core/device_memory.h"
#include "hle/kernel/board/nintendo/nx/k_system_control.h"

namespace Core {

#ifdef HAS_NCE
constexpr size_t VirtualReserveSize = 1ULL << 38;
#else
constexpr size_t VirtualReserveSize = 1ULL << 39;
#endif

DeviceMemory::DeviceMemory()
    : buffer{Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize(),
             VirtualReserveSize},
      cpu_mappings{Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize(),
                   DeferredWritesConfig::CpuMappingsRequested()},
      cpu_epochs{Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize(),
                 DeferredWritesConfig::CpuEpochsRequested()} {}

DeviceMemory::~DeviceMemory() = default;

} // namespace Core
