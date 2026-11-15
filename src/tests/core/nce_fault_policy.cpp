// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "core/arm/nce/nce_fault_policy.h"

namespace {
using Core::NceFaultPolicy::Classify;
using Core::NceFaultPolicy::Decision;
constexpr u64 DataAbort = u64{0x24} << 26;
constexpr u64 Write = u64{1} << 6;
}

TEST_CASE("NCE fault policy accepts translation stores only with deferred admission",
          "[deferred_writes][nce_fault_policy]") {
    for (u64 status = 0; status < 64; ++status) {
        CAPTURE(status);
        const bool permission = (status & 0x3c) == 0x0c;
        const bool translation = (status & 0x3c) == 0x04;
        CHECK(Classify(true, DataAbort | Write | status, true, false, false) ==
              (permission ? Decision::Write : Decision::Unsupported));
        CHECK(Classify(true, DataAbort | Write | status, true, false, true) ==
              (permission || translation ? Decision::Write : Decision::Unsupported));
        for (const bool deferred : {false, true}) {
            CHECK(Classify(true, DataAbort | status, true, false, deferred) ==
                  (permission ? Decision::Read : Decision::Unsupported));
        }
    }
}

TEST_CASE("NCE fault policy requires cached EL0 data access errors",
          "[deferred_writes][nce_fault_policy]") {
    for (const u64 status : {u64{7}, u64{15}}) {
        const u64 esr = DataAbort | Write | status;
        CHECK(Classify(false, esr, true, false, true) == Decision::Unsupported);
        CHECK(Classify(true, esr, false, false, true) == Decision::Unsupported);
        CHECK(Classify(true, esr, true, true, true) == Decision::Unsupported);
        for (u64 ec = 0; ec < 64; ++ec) {
            CAPTURE(ec);
            CHECK(Classify(true, (ec << 26) | Write | status, true, false, true) ==
                  (ec == 0x24 ? Decision::Write : Decision::Unsupported));
        }
    }
}
