// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <future>
#include <limits>
#include <map>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/deferred_kernel_writes.h"

namespace {
using namespace VideoCommon::DeferredWrites;

struct Fixture {
    std::shared_ptr<Journal> journal = std::make_shared<Journal>(2, 8);
    Core::DeferredKernelWrites registry;
    std::map<VAddr, PAddr> mapping;
    size_t reservations{};

    std::optional<PAddr> Resolve(VAddr address) const {
        const auto it = mapping.find(address);
        return it == mapping.end() ? std::nullopt : std::make_optional(it->second);
    }

    auto Reserve(PAddr physical) {
        ++reservations;
        const std::array ranges{TransitionRange{TransitionSpace::Physical, physical, 4096}};
        return BeginSharedWriterPermission(journal, ranges);
    }

    bool Apply(const Core::DeferredKernelWrites::Permission& pin) {
        return journal->DrainThrough(pin->Watermark(), [](const Write&) { return true; }) ==
                   DrainResult::Complete &&
               pin->MarkReady();
    }

    bool Prepare(VAddr address, u64 size) {
        return registry.Prepare(address, size,
            [&](VAddr page) { return Resolve(page); },
            [&](PAddr physical) { return Reserve(physical); },
            [&](const auto& pin) { return Apply(pin); });
    }

    bool Map(VAddr address, u64 size, std::optional<PAddr> physical) {
        return registry.WithMapping(address, size, physical,
            [&](PAddr page) { return Reserve(page); }, [&] {
                for (u64 offset = 0; offset < size; offset += 4096) {
                    if (physical) mapping[address + offset] = *physical + offset;
                    else mapping.erase(address + offset);
                }
            });
    }

    Publication Publish(PAddr physical, DAddr device, u8 value = 7) {
        const auto lease = journal->GrantLease(physical, device, 1);
        if (!lease) return {PublishStatus::LeaseUnavailable};
        Write write;
        write.physical_address = physical;
        write.device_address = device;
        write.mapping_generation = 1;
        write.lease_generation = lease->generation;
        write.size = 1;
        write.bytes[0] = value;
        return journal->TryPublishLeased(0, write, []() noexcept { return true; });
    }
};
} // namespace

TEST_CASE("Physical observer distinguishes new and existing exclusions only when enabled",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Map(0x10000, 8192, 0x4000));
    std::vector<std::pair<PAddr, bool>> observations;
    const auto prepare = [&](VAddr address, bool observe_pages) {
        return fixture.registry.Prepare(address, 4,
            [&](VAddr page) { return fixture.Resolve(page); },
            [&](PAddr physical) { return fixture.Reserve(physical); },
            [&](const auto& pin) { return fixture.Apply(pin); },
            [&](PAddr physical, bool newly_excluded) {
                observations.emplace_back(physical, newly_excluded);
            }, observe_pages);
    };
    REQUIRE(prepare(0x10000, true));
    REQUIRE(prepare(0x10000, true));
    REQUIRE(prepare(0x11000, false));
    REQUIRE(observations.size() == 2);
    CHECK(observations[0].first == 0x4000);
    CHECK(observations[0].second);
    CHECK(observations[1].first == 0x4000);
    CHECK_FALSE(observations[1].second);
    CHECK(fixture.reservations == 2);
    REQUIRE(prepare(0x11000, true));
    REQUIRE(observations.size() == 3);
    CHECK(observations[2].first == 0x5000);
    CHECK_FALSE(observations[2].second);
}

TEST_CASE("Kernel role prepares its pending prefix before entering scheduler locks",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Publish(0x4000, 0x8000).sequence == 1);
    REQUIRE(fixture.Prepare(0x10004, sizeof(u32)));
    CHECK(fixture.journal->AppliedThrough() == 1);
    CHECK(fixture.registry.IsPrepared(0x10004, sizeof(u32)));
    CHECK(fixture.registry.VirtualPageCount() == 1);
    CHECK(fixture.registry.PhysicalPageCount() == 1);
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x8000, 2));
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x18000, 3));
    CHECK(fixture.journal->GrantLease(0x5000, 0x9000, 1));
    REQUIRE(fixture.Prepare(0x10008, sizeof(u32)));
    CHECK(fixture.reservations == 1);
}

TEST_CASE("Kernel roles survive unmap remap and physical ABA",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Prepare(0x10000, sizeof(u32)));
    REQUIRE(fixture.Map(0x10000, 4096, std::nullopt));
    CHECK(fixture.registry.IsPrepared(0x10000, sizeof(u32)));
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x8000, 2));
    REQUIRE(fixture.Map(0x10000, 4096, 0x5000));
    CHECK_FALSE(fixture.journal->GrantLease(0x5000, 0x9000, 3));
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    CHECK(fixture.registry.PhysicalPageCount() == 2);
    CHECK(fixture.registry.VirtualPageCount() == 1);
    CHECK(fixture.reservations == 2);
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0xa000, 4));
    CHECK_FALSE(fixture.journal->GrantLease(0x5000, 0xb000, 5));
}

TEST_CASE("Captured legacy writer drains accepted data and blocks publications until its delayed commit",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    auto readback = fixture.journal->BeginReadback();
    const auto lease = fixture.journal->GrantLease(0x4000, 0x8000, 1);
    REQUIRE(lease);
    Write write;
    write.physical_address = 0x4000;
    write.device_address = 0x8000;
    write.mapping_generation = 1;
    write.lease_generation = lease->generation;
    write.size = 1;
    write.bytes[0] = 7;
    u8 backing{}, gpu{};
    REQUIRE(fixture.journal->TryPublishLeased(0, write, [&]() noexcept {
        backing = 7;
        return true;
    }).status == PublishStatus::Published);

    std::promise<void> reserved, continue_prepare;
    auto reserved_future = reserved.get_future();
    auto proceed = continue_prepare.get_future();
    bool prepared{};
    std::jthread writer{[&] {
        prepared = fixture.registry.Prepare(0x10000, 1,
            [&](VAddr page) { return fixture.Resolve(page); },
            [&](PAddr physical) { return fixture.Reserve(physical); }, [&](const auto& pin) {
                reserved.set_value();
                proceed.wait();
                return fixture.journal->DrainThrough(pin->Watermark(), [&](const Write& applied) {
                    gpu = applied.bytes[0];
                    return true;
                }) == DrainResult::Complete && pin->MarkReady();
            });
    }};
    reserved_future.wait();
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x18000, 2));
    CHECK(fixture.journal->TryPublishLeased(1, write, []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
    continue_prepare.set_value();
    writer.join();
    REQUIRE(prepared);
    CHECK(gpu == 7);

    // The caller may capture its pointer now, then commit after a remap and an alias lookup.
    REQUIRE(fixture.Map(0x10000, 4096, 0x5000));
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x18000, 3));
    CHECK_FALSE(fixture.journal->GrantLease(0x5000, 0x19000, 3));
    backing = 9;
    fixture.journal->WriteReadback(*readback, 0x4000, 1,
                                  [&](u64, u64) { backing = 0; });
    CHECK(backing == 9);
    CHECK_FALSE(fixture.journal->RetireThrough(1));
    readback.reset();
    CHECK(fixture.journal->RetireThrough(1));
}

TEST_CASE("Ready kernel roles avoid repeated preparation and remain excluded during suspension",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Prepare(0x10000, 4));
    auto suspension = fixture.journal->BeginSuspension();
    CHECK(fixture.journal->GetSuspensionState() == SuspensionState::Draining);
    unsigned preparations{};
    for (unsigned i = 0; i < 32; ++i) {
        REQUIRE(fixture.registry.Prepare(0x10000, 4,
            [&](VAddr page) { return fixture.Resolve(page); },
            [&](PAddr page) { return fixture.Reserve(page); }, [&](const auto&) {
                ++preparations;
                return false;
            }));
    }
    CHECK(preparations == 0);
    CHECK(fixture.reservations == 1);
    bool prepared{};
    REQUIRE(suspension.TryPrepareIdle(prepared) == SuspensionResult::Ready);
    suspension.Reset();
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x8000, 2));
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x18000, 3));
}

TEST_CASE("Kernel mapping installs every destination exclusion before exposing entries",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Prepare(0x10ffc, 8));
    CHECK(fixture.registry.VirtualPageCount() == 2);
    bool mutated{};
    REQUIRE(fixture.registry.WithMapping(0x10000, 8192, 0x4000,
        [&](PAddr page) { return fixture.Reserve(page); }, [&] {
            CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x8000, 1));
            CHECK_FALSE(fixture.journal->GrantLease(0x5000, 0x9000, 1));
            fixture.mapping[0x10000] = 0x4000;
            fixture.mapping[0x11000] = 0x5000;
            mutated = true;
        }));
    CHECK(mutated);
    CHECK(fixture.registry.IsPrepared(0x10ffc, 8));
    CHECK_FALSE(fixture.registry.IsPrepared(0x12000, 4));
}

TEST_CASE("Mapping with a pending destination never waits or mutates under its locks",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Prepare(0x10000, 4));
    const auto publication = fixture.Publish(0x5000, 0x9000);
    REQUIRE(publication.status == PublishStatus::Published);
    CHECK_FALSE(fixture.Map(0x10000, 4096, 0x5000));
    CHECK(fixture.Resolve(0x10000) == 0x4000);
    CHECK(fixture.journal->AppliedThrough() == 0);
    CHECK_FALSE(fixture.journal->GrantLease(0x5000, 0xa000, 1));
    REQUIRE(fixture.journal->DrainThrough(publication.sequence, [](const Write&) { return true; }) ==
            DrainResult::Complete);
    REQUIRE(fixture.Map(0x10000, 4096, 0x5000));
    CHECK(fixture.Resolve(0x10000) == 0x5000);
    CHECK(fixture.registry.IsPrepared(0x10000, 4));
}

TEST_CASE("Remap can complete while initial kernel role preparation waits outside registry lock",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Publish(0x4000, 0x8000).status == PublishStatus::Published);
    std::promise<void> entered;
    std::promise<void> release;
    auto released = release.get_future();
    bool prepared{};
    std::jthread worker{[&] {
        prepared = fixture.registry.Prepare(0x10000, 4,
            [&](VAddr page) { return fixture.Resolve(page); },
            [&](PAddr page) { return fixture.Reserve(page); }, [&](const auto& pin) {
                entered.set_value();
                released.wait();
                return fixture.Apply(pin);
            });
    }};
    entered.get_future().wait();
    const bool remapped = fixture.Map(0x10000, 4096, 0x5000);
    release.set_value();
    worker.join();
    REQUIRE(remapped);
    CHECK(prepared);
    CHECK(fixture.registry.IsPrepared(0x10000, 4));
    CHECK(fixture.Resolve(0x10000) == 0x5000);
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0xa000, 2));
    CHECK_FALSE(fixture.journal->GrantLease(0x5000, 0xb000, 2));
}

TEST_CASE("A second kernel caller cannot mistake a Preparing role for a ready exclusion",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Publish(0x4000, 0x8000).status == PublishStatus::Published);
    std::promise<void> entered;
    std::promise<void> release;
    auto released = release.get_future();
    bool first_prepared{};
    std::jthread first{[&] {
        first_prepared = fixture.registry.Prepare(0x10000, 4,
            [&](VAddr page) { return fixture.Resolve(page); },
            [&](PAddr page) { return fixture.Reserve(page); }, [&](const auto& pin) {
                entered.set_value();
                released.wait();
                return fixture.Apply(pin);
            });
    }};
    entered.get_future().wait();
    bool second_called{};
    const bool second_prepared = fixture.registry.Prepare(0x10000, 4,
        [&](VAddr page) { return fixture.Resolve(page); },
        [&](PAddr page) { return fixture.Reserve(page); }, [&](const auto& pin) {
            second_called = true;
            return pin->MarkReady();
        });
    const bool ready_before_drain = fixture.registry.IsPrepared(0x10000, 4);
    release.set_value();
    first.join();
    CHECK(second_called);
    CHECK_FALSE(second_prepared);
    CHECK_FALSE(ready_before_drain);
    CHECK(first_prepared);
    CHECK(fixture.registry.IsPrepared(0x10000, 4));
    CHECK(fixture.reservations == 1);
}

TEST_CASE("Failed kernel preparation keeps exclusions until retry or process teardown",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Publish(0x4000, 0x8000).status == PublishStatus::Published);
    CHECK_FALSE(fixture.registry.Prepare(0x10000, 4,
        [&](VAddr page) { return fixture.Resolve(page); },
        [&](PAddr page) { return fixture.Reserve(page); }, [](const auto&) { return false; }));
    CHECK_FALSE(fixture.registry.IsPrepared(0x10000, 4));
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x8000, 2));
    REQUIRE(fixture.Prepare(0x10000, 4));
    fixture.registry.Clear();
    CHECK(fixture.registry.PhysicalPageCount() == 0);
    CHECK(fixture.registry.VirtualPageCount() == 0);
    CHECK_FALSE(fixture.registry.IsPrepared(0x10000, 4));
    CHECK(fixture.journal->GrantLease(0x4000, 0x8000, 2));
}

TEST_CASE("Kernel registry rejects overflow and invalid mappings before mutation",
          "[deferred_writes][deferred_kernel_writes]") {
    Fixture fixture;
    constexpr auto max = (std::numeric_limits<u64>::max)();
    CHECK_FALSE(fixture.Prepare(max - 1, 4));
    CHECK_FALSE(fixture.registry.IsPrepared(max - 1, 4));
    CHECK_FALSE(fixture.Map(0x10001, 4096, 0x4000));
    CHECK_FALSE(fixture.Map(0x10000, 4095, 0x4000));
    CHECK_FALSE(fixture.Map(0x10000, 4096, 0x4001));
    CHECK_FALSE(fixture.Map(0x10000, 8192, max & ~u64{4095}));
    CHECK(fixture.mapping.empty());
    CHECK(fixture.registry.VirtualPageCount() == 0);
    CHECK(fixture.registry.PhysicalPageCount() == 0);
    CHECK(fixture.Prepare(max, 0));
    CHECK(fixture.registry.IsPrepared(max, 0));
}

TEST_CASE("Exclusive fast tags appear only after pending writes have drained",
          "[deferred_writes][deferred_exclusive_fast]") {
    Fixture fixture;
    const auto* table = fixture.registry.EnableFastTable();
    REQUIRE(table == fixture.registry.EnableFastTable());
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Publish(0x4000, 0x8000).sequence == 1);
    CHECK_FALSE(fixture.registry.IsPreparedFast(0x10004, 4));
    REQUIRE(fixture.registry.Prepare(0x10004, 4,
        [&](VAddr page) { return fixture.Resolve(page); },
        [&](PAddr page) { return fixture.Reserve(page); }, [&](const auto& pin) {
            CHECK_FALSE(fixture.registry.IsPreparedFast(0x10004, 4));
            return fixture.Apply(pin);
        }));
    CHECK(fixture.journal->AppliedThrough() == 1);
    CHECK(fixture.registry.IsPreparedFast(0x10004, 4));
    CHECK(table[0x10].load(std::memory_order_acquire) == 0x11);
    CHECK_FALSE(fixture.registry.IsPreparedFast(0x10ffc, 8));
    REQUIRE(fixture.Prepare(0x11000, 4)); // Unmapped role also protects future installations.
    CHECK(fixture.registry.IsPreparedFast(0x10ffc, 8));
}

TEST_CASE("Exclusive fast tags survive remap because replacement identities are pinned first",
          "[deferred_writes][deferred_exclusive_fast]") {
    Fixture fixture;
    fixture.registry.EnableFastTable();
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Prepare(0x10004, 4));
    REQUIRE(fixture.registry.WithMapping(0x10000, 4096, 0x5000,
        [&](PAddr page) { return fixture.Reserve(page); }, [&] {
            CHECK(fixture.registry.IsPreparedFast(0x10004, 4));
            CHECK_FALSE(fixture.journal->GrantLease(0x5000, 0x9000, 1));
            fixture.mapping[0x10000] = 0x5000;
        }));
    REQUIRE(fixture.Map(0x10000, 4096, std::nullopt));
    CHECK(fixture.registry.IsPreparedFast(0x10004, 4));
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    CHECK(fixture.registry.IsPreparedFast(0x10004, 4));
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x8000, 2));
    CHECK_FALSE(fixture.journal->GrantLease(0x5000, 0x9000, 2));
    fixture.registry.Clear();
    CHECK_FALSE(fixture.registry.IsPreparedFast(0x10004, 4));
    CHECK(fixture.journal->GrantLease(0x4000, 0x8000, 2));
}

TEST_CASE("Exclusive tagged cache collisions are misses and never authorize another page",
          "[deferred_writes][deferred_exclusive_fast]") {
    Fixture fixture;
    fixture.registry.EnableFastTable();
    constexpr VAddr first = 0x10000;
    constexpr VAddr collision = first + (Core::DeferredKernelWrites::FastTableMask + 1) * 4096;
    REQUIRE(fixture.Map(first, 4096, 0x4000));
    REQUIRE(fixture.Map(collision, 4096, 0x5000));
    REQUIRE(fixture.Prepare(first, 4));
    CHECK_FALSE(fixture.registry.IsPreparedFast(collision, 4));
    REQUIRE(fixture.Prepare(collision, 4));
    CHECK_FALSE(fixture.registry.IsPreparedFast(first, 4));
    CHECK(fixture.registry.IsPreparedFast(collision, 4));
    REQUIRE(fixture.Prepare(first, 4));
    CHECK(fixture.reservations == 2);
    CHECK(fixture.registry.IsPreparedFast(first, 4));
    CHECK_FALSE(fixture.registry.IsPreparedFast((std::numeric_limits<u64>::max)(), 2));
}

TEST_CASE("Concurrent preparation cannot expose a fast tag before Ready",
          "[deferred_writes][deferred_exclusive_fast]") {
    Fixture fixture;
    fixture.registry.EnableFastTable();
    REQUIRE(fixture.Map(0x10000, 4096, 0x4000));
    REQUIRE(fixture.Publish(0x4000, 0x8000).sequence == 1);
    std::promise<void> reserved, proceed;
    auto resume = proceed.get_future();
    auto ready = reserved.get_future();
    bool result{};
    std::jthread worker([&] {
        result = fixture.registry.Prepare(0x10004, 4,
            [&](VAddr page) { return fixture.Resolve(page); },
            [&](PAddr page) { return fixture.Reserve(page); }, [&](const auto& pin) {
                reserved.set_value();
                resume.wait();
                return fixture.Apply(pin);
            });
    });
    ready.wait();
    CHECK_FALSE(fixture.registry.IsPreparedFast(0x10004, 4));
    CHECK_FALSE(fixture.journal->GrantLease(0x4000, 0x8000, 2));
    proceed.set_value();
    worker.join();
    CHECK(result);
    CHECK(fixture.registry.IsPreparedFast(0x10004, 4));
}
