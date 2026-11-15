// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"

TEST_CASE("Page table block ids do not wrap", "[common]") {
    using Entry = Common::PageTable::PageEntryData;

    REQUIRE(Entry::BlockIdForMapping(0) == 0);
    REQUIRE(Entry::BlockIdForMapping(65534) == 65534);
    REQUIRE(Entry::BlockIdForMapping(65535) == Entry::INVALID_BLOCK_ID);
    REQUIRE(Entry::BlockIdForMapping(65536) == Entry::INVALID_BLOCK_ID);
    REQUIRE(Entry::BlockIdForMapping((std::numeric_limits<u64>::max)()) ==
            Entry::INVALID_BLOCK_ID);
}

TEST_CASE("Page table fast spans reject exhausted block ids", "[common]") {
    using Entry = Common::PageTable::PageEntryData;

    REQUIRE(Entry::CanFastSpan(7, 7));
    REQUIRE_FALSE(Entry::CanFastSpan(7, 8));
    REQUIRE_FALSE(Entry::CanFastSpan(Entry::INVALID_BLOCK_ID, Entry::INVALID_BLOCK_ID));
}

TEST_CASE("Marked cached page pointer resolves the same physical offset as its raw representation",
          "[common][page_table]") {
    using Entry = Common::PageTable::PageEntryData;
    constexpr uintptr_t guest_page = 0x4000000000;
    constexpr uintptr_t physical_offset = 0x12345000;
    for (const uintptr_t host_base : {uintptr_t{0x700000000000}, uintptr_t{0x10000000}}) {
        Entry entry;
        const uintptr_t bias = host_base + physical_offset - guest_page;
        entry.Store(true, Common::PageType::RasterizerCachedMemory, 7, bias);
        CHECK(entry.Pointer() == 0);
        CHECK(entry.Pointer(true) == Entry::ExtractPointer(entry.Raw(), true));
        for (const uintptr_t offset : {uintptr_t{0}, uintptr_t{4095}}) {
            const uintptr_t host = entry.Pointer(true) + guest_page + offset;
            CHECK(host == host_base + physical_offset + offset);
            CHECK(((host - host_base) & ~uintptr_t{4095}) == physical_offset);
        }
    }
}
