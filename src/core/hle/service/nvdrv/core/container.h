// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-FileCopyrightText: 2022 Skyline Team and Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include "common/container/unordered_map.h"

#include "core/device_memory_manager.h"
#include "core/hle/service/nvdrv/nvdata.h"

namespace Kernel {
class KProcess;
}

namespace Tegra::Host1x {
class Host1x;
} // namespace Tegra::Host1x

namespace Service::Nvidia::NvCore {

class HeapMapper;
class NvMap;
class SyncpointManager;

struct ContainerImpl;

// Kept by queued mappings and channels after the session id can be reused.
struct SessionLifetime {
    SessionLifetime(Tegra::Host1x::Host1x& host1x_, Kernel::KProcess* process_, Core::Asid asid_);
    ~SessionLifetime();

    SessionLifetime(const SessionLifetime&) = delete;
    SessionLifetime& operator=(const SessionLifetime&) = delete;

    void TrackDeviceRange(DAddr address, size_t size, size_t allocation_size);
    void FreeDeviceRange(DAddr address);

    Tegra::Host1x::Host1x& host1x;
    Kernel::KProcess* process;
    Core::Asid asid;
    std::shared_ptr<HeapMapper> mapper;

private:
    struct DeviceRange {
        size_t size;
        size_t allocation_size;
    };
    std::mutex ranges_mutex;
    std::map<DAddr, DeviceRange> device_ranges;
};

struct SessionId {
    size_t id;
};

struct Session {
    Session(SessionId id_, std::shared_ptr<SessionLifetime> lifetime_);
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) = default;
    Session& operator=(Session&&) = default;

    SessionId id;
    Kernel::KProcess* process;
    Core::Asid asid;
    bool has_preallocated_area{};
    std::shared_ptr<HeapMapper> mapper{};
    bool is_active{};
    s32 ref_count{};
    std::shared_ptr<SessionLifetime> lifetime;
};

class Container {
public:
    explicit Container(Tegra::Host1x::Host1x& host1x);
    ~Container();

    SessionId OpenSession(Kernel::KProcess* process);
    void CloseSession(SessionId id);

    Session* GetSession(SessionId id);
    std::shared_ptr<SessionLifetime> GetSessionLifetime(SessionId id);

    NvMap& GetNvMapFile();

    const NvMap& GetNvMapFile() const;

    SyncpointManager& GetSyncpointManager();

    const SyncpointManager& GetSyncpointManager() const;

    struct Host1xDeviceFileData {
        std::deque<u32> syncpts_accumulated{};
    };

    Host1xDeviceFileData& Host1xDeviceFile();

    const Host1xDeviceFileData& Host1xDeviceFile() const;

private:
    std::unique_ptr<ContainerImpl> impl;
};

} // namespace Service::Nvidia::NvCore
