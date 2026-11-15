// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "video_core/deferred_write_journal.h"
#include "video_core/query_cache.h"
#include "video_core/query_cache/query_stream.h"

namespace {
using namespace VideoCommon::DeferredWrites;

std::shared_ptr<Journal::WriterPermission> QueryPin(const std::shared_ptr<Journal>& journal) {
    const std::array ranges{
        TransitionRange{TransitionSpace::Physical, 0x4000, 16},
        TransitionRange{TransitionSpace::Device, 0x8000, 16},
    };
    auto pin = BeginSharedWriterPermission(journal, ranges);
    REQUIRE(pin);
    REQUIRE(pin->MarkReady());
    return pin;
}

class TestStreamer final : public VideoCommon::SimpleStreamer<VideoCommon::GuestQuery> {
public:
    TestStreamer() : SimpleStreamer<VideoCommon::GuestQuery>{0} {}
    using SimpleStreamer::Free;
    using SimpleStreamer::GetQuery;

    size_t WriteCounter(VAddr address, bool timestamp, u32 value,
                       std::optional<u32> = std::nullopt) override {
        return BuildQuery(timestamp, address, value);
    }
};

struct TestCounter {
    std::function<u64()> read;
    u64 Query(bool = false) { return read(); }
    bool WaitPending() const { return false; }
};
using LegacyQuery = VideoCommon::CachedQueryBase<TestCounter>;
} // namespace

TEST_CASE("Query slot invalidation and recycling preserve the callback writer pin",
          "[deferred_writes][deferred_queries]") {
    auto journal = std::make_shared<Journal>(1, 4);
    TestStreamer streamer;
    const auto id = streamer.WriteCounter(0x8000, false, 7);
    auto* query = streamer.GetQuery(id);
    query->deferred_writer = QueryPin(journal);
    auto callback_pin = query->deferred_writer;
    query->flags |= VideoCommon::QueryFlagBits::IsInvalidated;
    CHECK_FALSE(journal->GrantLease(0x4000, 0xa000, 1));

    streamer.Free(id);
    CHECK_FALSE(query->deferred_writer);
    CHECK_FALSE(journal->GrantLease(0x4000, 0x8000, 1));
    const auto reused = streamer.WriteCounter(0x8000, true, 9);
    REQUIRE(reused == id);
    CHECK_FALSE(streamer.GetQuery(reused)->deferred_writer);
    CHECK(streamer.GetQuery(reused)->value == 9);
    callback_pin.reset();
    CHECK(journal->GrantLease(0x4000, 0x8000, 1));
    streamer.Free(reused);
}

TEST_CASE("Legacy query keeps its writer pin while flushing and after moving",
          "[deferred_writes][deferred_queries]") {
    auto journal = std::make_shared<Journal>(1, 4);
    std::array<u64, 2> backing{};
    {
        LegacyQuery query{0x8000, reinterpret_cast<u8*>(backing.data())};
        query.SetDeferredWriter(QueryPin(journal));
        auto counter = std::make_shared<TestCounter>();
        counter->read = [&] {
            CHECK_FALSE(journal->GrantLease(0x4000, 0x8000, 1));
            return u64{23};
        };
        CHECK_FALSE(query.BindCounter(counter, 41));
        LegacyQuery moved{std::move(query)};
        CHECK_FALSE(query.GetDeferredWriter());
        CHECK(moved.Flush() == 23);
        CHECK(backing[0] == 23);
        CHECK(backing[1] == 41);
        CHECK_FALSE(journal->GrantLease(0x4000, 0x8000, 1));
    }
    CHECK(journal->GrantLease(0x4000, 0x8000, 1));
}

TEST_CASE("Legacy async query callback retains exclusion after its cached query is removed",
          "[deferred_writes][deferred_queries]") {
    auto journal = std::make_shared<Journal>(1, 4);
    std::array<u64, 2> backing{};
    std::function<void()> completion;
    {
        LegacyQuery query{0x8000, reinterpret_cast<u8*>(backing.data())};
        query.SetDeferredWriter(QueryPin(journal));
        completion = [pin = query.GetDeferredWriter(), journal, &backing] {
            REQUIRE(pin->Ready());
            CHECK_FALSE(journal->GrantLease(0x4000, 0x8000, 1));
            backing[0] = 57;
        };
    }
    CHECK_FALSE(journal->GrantLease(0x4000, 0x8000, 1));
    SECTION("Completion writes while admission remains closed") {
        completion();
        CHECK(backing[0] == 57);
    }
    SECTION("Cancellation releases without writing") {
        CHECK(backing[0] == 0);
    }
    completion = {};
    CHECK(journal->GrantLease(0x4000, 0x8000, 1));
}

TEST_CASE("Overlapping query objects retain independent exclusions until both are freed",
          "[deferred_writes][deferred_queries]") {
    auto journal = std::make_shared<Journal>(1, 4);
    TestStreamer streamer;
    const auto first = streamer.WriteCounter(0x8000, false, 1);
    const auto second = streamer.WriteCounter(0x8008, false, 2);
    streamer.GetQuery(first)->deferred_writer = QueryPin(journal);
    streamer.GetQuery(second)->deferred_writer = QueryPin(journal);
    streamer.Free(first);
    CHECK_FALSE(journal->GrantLease(0x4000, 0x8000, 1));
    streamer.Free(second);
    CHECK(journal->GrantLease(0x4000, 0x8000, 1));
}
