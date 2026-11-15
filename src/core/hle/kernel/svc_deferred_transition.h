// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/hle/kernel/svc.h"

namespace Kernel::Svc {

// IPC guards sit at its actual mutations, not across a blocking SVC. Nonreturning exits
// release their local scope explicitly before switching away and guard their worker separately.
constexpr bool NeedsDeferredTransition(u32 imm) {
    // NX2 SetGpuMemoryAttribute changes only the GpuAccessible metadata bit.
    if (imm == 0x80) return false;
    switch (static_cast<SvcId>(imm)) {
    case SvcId::SetHeapSize:
    case SvcId::SetMemoryPermission:
    case SvcId::SetMemoryAttribute:
    case SvcId::MapMemory:
    case SvcId::UnmapMemory:
    case SvcId::CreateThread:
    case SvcId::MapSharedMemory:
    case SvcId::UnmapSharedMemory:
    case SvcId::CreateTransferMemory:
    case SvcId::CloseHandle:
    case SvcId::MapPhysicalMemory:
    case SvcId::UnmapPhysicalMemory:
    case SvcId::CreateIoRegion:
    case SvcId::MapIoRegion:
    case SvcId::UnmapIoRegion:
    case SvcId::MapPhysicalMemoryUnsafe:
    case SvcId::UnmapPhysicalMemoryUnsafe:
    case SvcId::SetUnsafeLimit:
    case SvcId::CreateCodeMemory:
    case SvcId::ControlCodeMemory:
    case SvcId::CreateSharedMemory:
    case SvcId::MapTransferMemory:
    case SvcId::UnmapTransferMemory:
    case SvcId::CreateDeviceAddressSpace:
    case SvcId::AttachDeviceAddressSpace:
    case SvcId::DetachDeviceAddressSpace:
    case SvcId::MapDeviceAddressSpaceByForce:
    case SvcId::MapDeviceAddressSpaceAligned:
    case SvcId::UnmapDeviceAddressSpace:
    case SvcId::TerminateDebugProcess:
    case SvcId::SetProcessMemoryPermission:
    case SvcId::MapProcessMemory:
    case SvcId::UnmapProcessMemory:
    case SvcId::MapProcessCodeMemory:
    case SvcId::UnmapProcessCodeMemory:
    case SvcId::CreateProcess:
    case SvcId::StartProcess:
    case SvcId::TerminateProcess:
    case SvcId::MapInsecureMemory:
    case SvcId::UnmapInsecureMemory:
        return true;
    default:
        return false;
    }
}

} // namespace Kernel::Svc
