// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <thread>
#include <cstring>
#include <catch2/catch_test_macros.hpp>
#include "core/arm/nce/nce_store_backing.h"
#include "core/arm/nce/nce_store_decoder.h"
#include "core/guest_memory.h"

namespace {
using Core::ExactStoreBacking::Commit;
using Core::ExactStoreBacking::Shape;
using Core::ExactStoreBacking::Validate;
using Core::ExactStoreBacking::Validation;
constexpr u64 Base = 0x4000;

struct ReservedSpanMemory {
    static constexpr bool HAS_FLUSH_INVALIDATION = false;
    std::array<u8, 32> bytes{};
    u32 pins{};
    u32 commits{};
    std::shared_ptr<void> BeginDeferredBackingWrite(u64, size_t) {
        ++pins;
        return {this, [this](void*) { --pins; }};
    }
    u8* GetSpan(u64 address, size_t) { return bytes.data() + address; }
    void ReadBlockUnsafe(u64 address, void* target, size_t size) {
        std::memcpy(target, bytes.data() + address, size);
    }
    void WriteBlockUnsafe(u64 address, const void* source, size_t size) {
        CHECK(pins != 0);
        ++commits;
        std::memcpy(bytes.data() + address, source, size);
    }
};

struct DecoderFixture {
    std::array<u64, 31> regs{};
    std::array<u128, 32> vectors{};
    u64 sp{0x5000};

    DecoderFixture() {
        regs.fill(0x4100);
        for (size_t reg = 0; reg < vectors.size(); ++reg) {
            auto* bytes = reinterpret_cast<u8*>(vectors[reg].data());
            for (size_t byte = 0; byte < 16; ++byte) bytes[byte] = static_cast<u8>(reg * 16 + byte);
        }
    }

    Core::NceExactStore::Result Decode(u32 instruction, u64 fault_address,
                                      Core::NceExactStore::Store& store) const {
        Core::NceExactStore::StoreVisitor visitor{regs, sp, vectors};
        return visitor.Decode(instruction, fault_address, store);
    }
};
}

TEST_CASE("NCE decodes the rejected Kokiri ST1 lane and ST2 interleaved opcodes",
          "[deferred_writes][nce_store_decoder]") {
    using namespace Core::NceExactStore;
    DecoderFixture fixture;
    Store store;
    REQUIRE(fixture.Decode(0x4d008100, 0x4100, store) == Result::Success);
    CHECK(store.address == 0x4100);
    CHECK(store.size == 4);
    CHECK(store.simd);
    CHECK_FALSE(store.pair);
    CHECK_FALSE(store.writeback);
    const std::array<u8, 4> lane{8, 9, 10, 11};
    CHECK(std::equal(lane.begin(), lane.end(), store.bytes.begin()));
    REQUIRE(fixture.Decode(0x0c0089a4, 0x410f, store) == Result::Success);
    CHECK(store.size == 16);
    CHECK_FALSE(store.pair);
    CHECK_FALSE(store.writeback);
    const std::array<u8, 16> interleaved{
        0x40, 0x41, 0x42, 0x43, 0x50, 0x51, 0x52, 0x53,
        0x44, 0x45, 0x46, 0x47, 0x54, 0x55, 0x56, 0x57};
    CHECK(std::equal(interleaved.begin(), interleaved.end(), store.bytes.begin()));
}

TEST_CASE("NCE ST1 lanes select byte halfword word and doubleword in both vector halves",
          "[deferred_writes][nce_store_decoder]") {
    using namespace Core::NceExactStore;
    DecoderFixture fixture;
    const auto saved_regs = fixture.regs;
    const auto saved_vectors = fixture.vectors;
    struct Lane { u32 opcode, size, s, element_size; };
    constexpr std::array lanes{Lane{0, 3, 1, 1}, Lane{1, 2, 1, 2},
                               Lane{2, 0, 1, 4}, Lane{2, 1, 0, 8}};
    for (u32 q : {0U, 1U}) {
        for (const auto& lane : lanes) {
            const u32 instruction = 0x0d000000 | (q << 30) | (lane.opcode << 14) |
                (lane.s << 12) | (lane.size << 10) | (8 << 5) | 31;
            Store store;
            REQUIRE(fixture.Decode(instruction, 0x4100, store) == Result::Success);
            CHECK(store.size == lane.element_size);
            const auto* source = reinterpret_cast<const u8*>(fixture.vectors[31].data());
            const u32 source_offset = (q + 1) * 8 - lane.element_size;
            CHECK(std::equal(store.bytes.begin(), store.bytes.begin() + store.size,
                             source + source_offset));
        }
    }
    CHECK(fixture.regs == saved_regs);
    CHECK(fixture.vectors == saved_vectors);
}

TEST_CASE("NCE SIMD structure stores postindex with immediate register and SP bases",
          "[deferred_writes][nce_store_decoder]") {
    using namespace Core::NceExactStore;
    DecoderFixture fixture;
    fixture.regs[3] = 0x70;
    for (u32 rn : {8U, 31U}) {
        const u64 address = rn == 31 ? fixture.sp : fixture.regs[rn];
        for (u32 rm : {3U, 31U}) {
            Store lane;
            const u32 lane_instruction = 0x4d808000 | (rm << 16) | (rn << 5) | 8;
            REQUIRE(fixture.Decode(lane_instruction, address, lane) == Result::Success);
            CHECK(lane.address == address);
            CHECK(lane.size == 4);
            CHECK(lane.writeback);
            CHECK(lane.base_register == rn);
            CHECK(lane.writeback_value == address + (rm == 31 ? 4 : 0x70));
            Store structure;
            const u32 structure_instruction = 0x4c808800 | (rm << 16) | (rn << 5) | 31;
            REQUIRE(fixture.Decode(structure_instruction, address, structure) == Result::Success);
            CHECK(structure.address == address);
            CHECK(structure.size == 32);
            CHECK(structure.pair);
            CHECK(structure.writeback);
            CHECK(structure.base_register == rn);
            CHECK(structure.writeback_value == address + (rm == 31 ? 32 : 0x70));
        }
    }
}

TEST_CASE("NCE ST2 interleaves element sizes and wraps vector register 31 to zero",
          "[deferred_writes][nce_store_decoder]") {
    using namespace Core::NceExactStore;
    DecoderFixture fixture;
    for (u32 q : {0U, 1U}) {
        for (u32 scale = 0; scale < 4; ++scale) {
            Store store;
            const u32 instruction = 0x0c008000 | (q << 30) | (scale << 10) | (8 << 5) | 31;
            const auto result = fixture.Decode(instruction, 0x4100, store);
            if (!q && scale == 3) {
                CHECK(result == Result::InvalidOperands);
                continue;
            }
            REQUIRE(result == Result::Success);
            CHECK(store.size == (q ? 32 : 16));
            CHECK(store.pair == (q != 0));
            const u32 element_size = 1U << scale;
            for (u32 byte = 0; byte < store.size; ++byte) {
                const u32 element = byte / element_size;
                const u32 source_reg = element % 2 == 0 ? 31 : 0;
                const u32 source_byte = (element / 2) * element_size + byte % element_size;
                const auto* source = reinterpret_cast<const u8*>(fixture.vectors[source_reg].data());
                CHECK(store.bytes[byte] == source[source_byte]);
            }
        }
    }
}

TEST_CASE("NCE SIMD decode rejection preserves output and scalar XZR remains zero",
          "[deferred_writes][nce_store_decoder]") {
    using namespace Core::NceExactStore;
    DecoderFixture fixture;
    Store store;
    store.address = 0xdead;
    store.size = 17;
    CHECK(fixture.Decode(0x0d004500, 0x4100, store) == Result::InvalidOperands);
    CHECK(fixture.Decode(0x0d009500, 0x4100, store) == Result::InvalidOperands);
    CHECK(fixture.Decode(0x4c000100, 0x4100, store) == Result::UnsupportedInstruction);
    fixture.regs[8] = 0x4ff8;
    CHECK(fixture.Decode(0x0c00891f, 0x4ff8, store) == Result::PageCrossing);
    CHECK(store.address == 0xdead);
    CHECK(store.size == 17);
    fixture.regs[8] = 0x4100;
    REQUIRE(fixture.Decode(0xf900011f, 0x4100, store) == Result::Success);
    CHECK(store.size == 8);
    for (u32 byte = 0; byte < store.size; ++byte) CHECK(store.bytes[byte] == 0);
    fixture.sp = 0x5008;
    CHECK(fixture.Decode(0x4d0083e0, fixture.sp, store) == Result::InvalidOperands);
}

TEST_CASE("Decoded ST1 and ST2 stores commit through the producer backing shape",
          "[deferred_writes][nce_store_decoder]") {
    using namespace Core::NceExactStore;
    DecoderFixture fixture;
    for (u32 instruction : {0x4d008100U, 0x0c0089a4U, 0x4c0089a4U}) {
        Store store;
        REQUIRE(fixture.Decode(instruction, 0x4100, store) == Result::Success);
        alignas(16) std::array<u8, 64> backing;
        backing.fill(0xa5);
        const Shape shape{store.pair ? store.size / 2 : store.size, store.pair, store.zero_block};
        const std::span<const u8> bytes{store.bytes.data(), store.size};
        REQUIRE(Validate(store.address, backing.data() + 16, bytes, shape) == Validation::Valid);
        REQUIRE(Commit(store.address, backing.data() + 16, bytes, shape));
        for (size_t byte = 0; byte < backing.size(); ++byte) {
            if (byte >= 16 && byte < 16 + store.size) CHECK(backing[byte] == store.bytes[byte - 16]);
            else CHECK(backing[byte] == 0xa5);
        }
    }
}

TEST_CASE("Writable guest spans retain reservations through changed-address writeback",
          "[deferred_writes][deferred_admission]") {
    using namespace Core::Memory;
    ReservedSpanMemory memory;
    {
        GuestMemoryScoped<ReservedSpanMemory, u8, GuestMemoryFlags::UnsafeReadWrite> span{
            memory, 0, 4};
        REQUIRE(memory.pins == 1);
        span[0] = 0x91;
        span.SetAddressAndSize(16, 4);
        REQUIRE(memory.pins == 2);
    }
    CHECK(memory.commits == 1);
    CHECK(memory.bytes[16] == 0x91);
    CHECK(memory.pins == 0);
}

TEST_CASE("Read-only guest spans never acquire a writer reservation",
          "[deferred_writes][deferred_admission]") {
    using namespace Core::Memory;
    ReservedSpanMemory memory;
    GuestMemoryScoped<ReservedSpanMemory, u8, GuestMemoryFlags::UnsafeRead> span{memory, 0, 4};
    CHECK(memory.pins == 0);
}

TEST_CASE("Exact backing stores preserve every byte outside their range",
          "[deferred_writes][nce_store_backing]") {
    for (u32 size : {1U, 2U, 4U, 8U, 16U}) {
        for (u32 offset = 0; offset < 32; ++offset) {
            CAPTURE(size, offset);
            alignas(16) std::array<u8, 64> backing;
            backing.fill(0x71);
            std::array<u8, 16> payload;
            payload.fill(0x93);
            REQUIRE(Commit(Base + offset, backing.data() + offset,
                           std::span{payload}.first(size), {size, false, false}));
            for (size_t i = 0; i < backing.size(); ++i)
                CHECK(backing[i] == (i >= offset && i < offset + size ? 0x93 : 0x71));
        }
    }
}

TEST_CASE("Exact paired stores preserve neighbors across aligned elements",
          "[deferred_writes][nce_store_backing]") {
    for (u32 element : {4U, 8U, 16U}) {
        for (u32 offset : {0U, element}) {
            CAPTURE(element, offset);
            alignas(16) std::array<u8, 64> backing;
            backing.fill(0x15);
            std::array<u8, 32> payload{};
            for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<u8>(i + 1);
            const auto bytes = std::span{payload}.first(element * 2);
            REQUIRE(Commit(Base + offset, backing.data() + offset, bytes, {element, true, false}));
            for (size_t i = 0; i < backing.size(); ++i)
                CHECK(backing[i] == (i >= offset && i < offset + bytes.size()
                                        ? payload[i - offset] : 0x15));
        }
    }
}

TEST_CASE("Rejected exact backing stores do not change backing",
          "[deferred_writes][nce_store_backing]") {
    alignas(16) std::array<u8, 64> backing;
    backing.fill(0x55);
    const auto original = backing;
    const std::array<u8, 32> payload{};
    CHECK(Validate(Base, backing.data() + 1, std::span{payload}.first(8), {8, false, false}) ==
          Validation::Unaligned);
    CHECK_FALSE(Commit(Base, backing.data() + 1, std::span{payload}.first(8), {8, false, false}));
    CHECK(Validate(0x4ffc, backing.data() + 12, std::span{payload}.first(8), {8, false, false}) ==
          Validation::PageCrossing);
    CHECK_FALSE(Commit(0x4ffc, backing.data() + 12, std::span{payload}.first(8), {8, false, false}));
    CHECK_FALSE(Commit(Base, backing.data(), payload, {32, false, false}));
    CHECK_FALSE(Commit(Base, backing.data(), std::span{payload}.first(3), {3, false, false}));
    CHECK_FALSE(Commit(Base, backing.data(), {}, {1, false, false}));
    CHECK(backing == original);
}

TEST_CASE("Exact DC ZVA backing commits only the aligned zero block",
          "[deferred_writes][nce_store_backing]") {
    alignas(64) std::array<u8, 192> backing;
    backing.fill(0x4e);
    std::array<u8, 64> zeros{};
    REQUIRE(Commit(Base + 64, backing.data() + 64, zeros, {16, false, true}));
    for (size_t i = 0; i < backing.size(); ++i)
        CHECK(backing[i] == (i >= 64 && i < 128 ? 0 : 0x4e));
    const auto original = backing;
    zeros[37] = 1;
    CHECK_FALSE(Commit(Base + 64, backing.data() + 64, zeros, {16, false, true}));
    CHECK(backing == original);
}

TEST_CASE("Concurrent masked exact stores retain both disjoint parts of one block",
          "[deferred_writes][nce_store_backing]") {
    alignas(16) std::array<u8, 16> backing{};
    const std::array<u8, 4> first{1, 2, 3, 4}, second{5, 6, 7, 8};
    bool first_ok = true, second_ok = true;
    std::jthread one([&] {
        for (u32 i = 0; i < 1000; ++i)
            first_ok &= Commit(Base + 3, backing.data() + 3, first, {4, false, false});
    });
    std::jthread two([&] {
        for (u32 i = 0; i < 1000; ++i)
            second_ok &= Commit(Base + 9, backing.data() + 9, second, {4, false, false});
    });
    one.join();
    two.join();
    CHECK(first_ok);
    CHECK(second_ok);
    for (size_t i = 0; i < backing.size(); ++i)
        CHECK(backing[i] == (i >= 3 && i < 7 ? first[i - 3] : i >= 9 && i < 13 ? second[i - 9] : 0));
}
