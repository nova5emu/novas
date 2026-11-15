// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "core/deferred_write_config.h"

TEST_CASE("Deferred producer requires two explicit switches",
          "[deferred_writes][deferred_exclusives]") {
    using Core::DeferredWritesConfig::ProducerRequested;
    CHECK_FALSE(ProducerRequested(nullptr, nullptr));
    CHECK_FALSE(ProducerRequested("1", nullptr));
    CHECK_FALSE(ProducerRequested(nullptr, "1"));
    CHECK_FALSE(ProducerRequested("1", "0"));
    CHECK_FALSE(ProducerRequested("0", "1"));
    CHECK_FALSE(ProducerRequested("", "1"));
    CHECK_FALSE(ProducerRequested("1", "true"));
    CHECK_FALSE(ProducerRequested("10", "1"));
    CHECK_FALSE(ProducerRequested("1", "10"));
    CHECK(ProducerRequested("1", "1"));
}

TEST_CASE("CPU identity mappings support epochs independently from deferred producers",
          "[deferred_writes][deferred_exclusives]") {
    using Core::DeferredWritesConfig::CpuMappingsRequested;
    CHECK_FALSE(CpuMappingsRequested(nullptr, nullptr, nullptr));
    CHECK(CpuMappingsRequested(nullptr, nullptr, "1"));
    CHECK(CpuMappingsRequested("0", "0", "1"));
    CHECK(CpuMappingsRequested("1", "0", "1"));
    CHECK(CpuMappingsRequested("0", "1", "1"));
    CHECK_FALSE(CpuMappingsRequested("0", "0", "0"));
    CHECK_FALSE(CpuMappingsRequested("1", "0", "0"));
    CHECK_FALSE(CpuMappingsRequested("0", "1", "0"));
    CHECK(CpuMappingsRequested("1", "1", "0"));
    CHECK(CpuMappingsRequested("1", "1", nullptr));
    CHECK_FALSE(CpuMappingsRequested(nullptr, nullptr, ""));
    CHECK_FALSE(CpuMappingsRequested(nullptr, nullptr, "true"));
    CHECK_FALSE(CpuMappingsRequested(nullptr, nullptr, "yes"));
    CHECK_FALSE(CpuMappingsRequested(nullptr, nullptr, "10"));
    CHECK_FALSE(CpuMappingsRequested(nullptr, nullptr, "01"));
    CHECK_FALSE(CpuMappingsRequested("10", "1", "0"));
    CHECK_FALSE(CpuMappingsRequested("1", "10", "0"));
}
