// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <future>
#include <map>
#include <optional>
#include <thread>
#include <catch2/catch_test_macros.hpp>
#include "core/deferred_cpu_mappings.h"

namespace {
using Registry = Core::DeferredCpuMappings;
using Status = Registry::Status;
struct Fixture {
    Registry registry{0x100000, true};
    int owner_a{}, owner_b{};
    std::map<std::pair<const void*, VAddr>, PAddr> mappings;
    Fixture() {
        registry.AllowHeapRange(&owner_a, 0x10000, 0x20000);
        registry.AllowHeapRange(&owner_b, 0x10000, 0x20000);
    }
    void Map(const void* owner, VAddr va, std::optional<PAddr> physical) {
        registry.WithMapping(va, 1, physical,
            [&](VAddr address) -> std::optional<PAddr> {
                const auto it = mappings.find({owner, address});
                return it == mappings.end() ? std::nullopt : std::make_optional(it->second);
            }, [&] {
                if (physical) mappings[{owner, va}] = *physical;
                else mappings.erase({owner, va});
            });
    }
    Status Check(const void* owner, VAddr va, PAddr physical, u64* generation = nullptr) {
        return registry.WithPage(owner, va, physical, [&](const Registry::Snapshot& proof) {
            CHECK(mappings.at({owner, va & ~VAddr{4095}}) == physical);
            if (generation) *generation = proof.generation;
        });
    }
};
} // namespace

TEST_CASE("CPU admission requires private heap and a unique mapping across processes",
          "[deferred_writes][deferred_cpu_mappings]") {
    Fixture f;
    f.Map(&f.owner_a, 0x10000, 0x4000);
    CHECK(f.Check(&f.owner_a, 0x10004, 0x4000) == Status::Eligible);
    f.Map(&f.owner_b, 0x10000, 0x4000);
    CHECK(f.Check(&f.owner_a, 0x10000, 0x4000) == Status::Alias);
    CHECK(f.Check(&f.owner_b, 0x10000, 0x4000) == Status::Alias);
    f.Map(&f.owner_b, 0x10000, std::nullopt);
    CHECK(f.Check(&f.owner_a, 0x10000, 0x4000) == Status::Eligible);
    f.Map(&f.owner_a, 0x30000, 0x5000);
    CHECK(f.Check(&f.owner_a, 0x30000, 0x5000) == Status::NotPrivateHeap);
}

TEST_CASE("CPU physical generations distinguish ABA without invalidating unrelated pages",
          "[deferred_writes][deferred_cpu_mappings]") {
    Fixture f;
    f.Map(&f.owner_a, 0x10000, 0x4000);
    u64 original{}, unrelated{}, replacement{};
    REQUIRE(f.Check(&f.owner_a, 0x10000, 0x4000, &original) == Status::Eligible);
    f.Map(&f.owner_a, 0x11000, 0x6000);
    REQUIRE(f.Check(&f.owner_a, 0x10000, 0x4000, &unrelated) == Status::Eligible);
    CHECK(original == unrelated);
    f.Map(&f.owner_a, 0x10000, 0x5000);
    CHECK(f.Check(&f.owner_a, 0x10000, 0x4000) == Status::MappingChanged);
    f.Map(&f.owner_a, 0x10000, 0x4000);
    REQUIRE(f.Check(&f.owner_a, 0x10000, 0x4000, &replacement) == Status::Eligible);
    CHECK(replacement > original);
}

TEST_CASE("Captured shared transfer and code identities never become eligible on reuse",
          "[deferred_writes][deferred_cpu_mappings]") {
    for (const auto reason : {Status::Shared, Status::Transfer, Status::Code, Status::CapturedWriter}) {
        Fixture f;
        f.Map(&f.owner_a, 0x10000, 0x4000);
        f.registry.ExcludePhysical(0x4004, 4, reason);
        CHECK(f.Check(&f.owner_a, 0x10000, 0x4000) == reason);
        f.Map(&f.owner_a, 0x10000, std::nullopt);
        f.Map(&f.owner_b, 0x10000, 0x4000);
        CHECK(f.Check(&f.owner_b, 0x10000, 0x4000) == reason);
    }
}

TEST_CASE("CPU proof holds mappings stable through a publication callback",
          "[deferred_writes][deferred_cpu_mappings]") {
    Fixture f;
    f.Map(&f.owner_a, 0x10000, 0x4000);
    std::promise<void> entered, release, remapping;
    auto held = entered.get_future();
    auto resume = release.get_future();
    auto started = remapping.get_future();
    std::atomic<bool> changed{};
    Status status = Status::Disabled;
    std::jthread publisher([&] {
        status = f.registry.WithPage(&f.owner_a, 0x10000, 0x4000, [&](const auto&) {
            entered.set_value();
            resume.wait();
            CHECK(f.mappings.at({&f.owner_a, 0x10000}) == 0x4000);
        });
    });
    held.wait();
    std::jthread mapper([&] {
        remapping.set_value();
        f.Map(&f.owner_a, 0x10000, 0x5000);
        changed.store(true);
    });
    started.wait();
    CHECK_FALSE(changed.load());
    release.set_value();
    publisher.join();
    mapper.join();
    CHECK(status == Status::Eligible);
    CHECK(changed.load());
    CHECK(f.Check(&f.owner_a, 0x10000, 0x5000) == Status::Eligible);
}

TEST_CASE("Finalization removes aliases and heap authority before physical reuse",
          "[deferred_writes][deferred_cpu_mappings]") {
    Fixture f;
    f.Map(&f.owner_a, 0x10000, 0x4000);
    f.registry.RemoveOwner(&f.owner_a);
    CHECK(f.Check(&f.owner_a, 0x10000, 0x4000) == Status::NotPrivateHeap);
    f.registry.RemoveFinalizedRange(0x4000, 4096);
    f.mappings.erase({&f.owner_a, 0x10000});
    f.Map(&f.owner_b, 0x10000, 0x4000);
    CHECK(f.Check(&f.owner_b, 0x10000, 0x4000) == Status::Eligible);
    f.registry.AllowHeapRange(&f.owner_b, 0x10000, 0x10000);
    CHECK(f.Check(&f.owner_b, 0x10000, 0x4000) == Status::NotPrivateHeap);
}

TEST_CASE("CPU admission stays closed when disabled or bookkeeping is inconsistent",
          "[deferred_writes][deferred_cpu_mappings]") {
    Registry disabled;
    bool mutated{}, published{};
    disabled.WithMapping(0x10000, 1, 0x4000, [](VAddr) { return std::optional<PAddr>{}; },
                         [&] { mutated = true; });
    CHECK(mutated);
    CHECK(disabled.WithPage(nullptr, 0x10000, 0x4000, [&](const auto&) { published = true; }) == Status::Disabled);
    CHECK_FALSE(published);
    Fixture f;
    f.Map(&f.owner_a, 0x10000, 0x4000);
    f.registry.RemoveFinalizedRange(0x7000, 4096); // No corresponding CPU mapping.
    CHECK(f.Check(&f.owner_a, 0x10000, 0x4000) == Status::MappingChanged);
}

TEST_CASE("Captured writes before journal creation remain excluded after heap admission",
          "[deferred_writes][deferred_cpu_mappings]") {
    Registry registry{0x100000, true};
    int owner{};
    registry.WithMapping(0x10000, 1, 0x4000,
        [](VAddr) { return std::optional<PAddr>{}; }, [] {});
    unsigned resolved{};
    registry.ExcludeVirtualRange(0x10ffc, 8, [&](VAddr page) -> std::optional<PAddr> {
        ++resolved;
        return page == 0x10000 ? std::make_optional<PAddr>(0x4000) : std::nullopt;
    }, Status::CapturedWriter);
    CHECK(resolved == 2);
    registry.AllowHeapRange(&owner, 0x10000, 0x20000);
    bool invoked{};
    CHECK(registry.WithPage(&owner, 0x10000, 0x4000,
        [&](const auto&) { invoked = true; }) == Status::CapturedWriter);
    CHECK_FALSE(invoked);
}
