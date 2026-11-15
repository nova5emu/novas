// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "core/deferred_kernel_writes.h"
#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
#include "dynarmic/interface/A32/a32.h"
#include "dynarmic/interface/A64/a64.h"

namespace {
constexpr u64 DataAddress = 0x4000;
constexpr u32 SvcA64 = 0xd4000001;
constexpr u32 SvcA32 = 0xef000000;

struct Environment {
    Dynarmic::ExclusiveMonitor monitor{1};
    u64 data_address{DataAddress};
    std::vector<u32> code;
    alignas(16) std::array<u8, 16> data{};
    std::vector<char> events;
    std::vector<size_t> prepared_sizes;
    std::function<void()> before_exclusive;
    std::function<void()> at_memory;
    bool preparation_unlocked{true};
    bool memory_locked{true};
    bool invalid_access{};
    bool exception{};
    u64 ticks{100};

    std::optional<u32> ReadCode(u64 address) const {
        if ((address & 3) || address / 4 >= code.size()) return std::nullopt;
        return code[address / 4];
    }

    void Prepare(u64 address, size_t size) {
        preparation_unlocked &= *Dynarmic::GetExclusiveMonitorLockPointer(&monitor) == 0;
        invalid_access |= address != data_address || size > data.size();
        events.push_back('P');
        prepared_sizes.push_back(size);
        if (before_exclusive) before_exclusive();
    }

    template <typename T>
    T Raw() const {
        T value{};
        std::memcpy(&value, data.data(), sizeof(value));
        return value;
    }

    template <typename T>
    void Set(T value) {
        std::memcpy(data.data(), &value, sizeof(value));
    }

    u64 Read(u64 address, size_t size) {
        events.push_back('R');
        memory_locked &= *Dynarmic::GetExclusiveMonitorLockPointer(&monitor) != 0;
        if (at_memory) at_memory();
        if (address != data_address || size > sizeof(u64)) {
            invalid_access = true;
            return 0;
        }
        u64 value{};
        std::memcpy(&value, data.data(), size);
        return value;
    }

    Dynarmic::A64::Vector Read128(u64 address) {
        events.push_back('R');
        memory_locked &= *Dynarmic::GetExclusiveMonitorLockPointer(&monitor) != 0;
        if (at_memory) at_memory();
        invalid_access |= address != data_address;
        return Raw<Dynarmic::A64::Vector>();
    }

    template <typename T>
    bool WriteExclusive(u64 address, T value, T expected) {
        events.push_back('W');
        memory_locked &= *Dynarmic::GetExclusiveMonitorLockPointer(&monitor) != 0;
        if (at_memory) at_memory();
        if (address != data_address) {
            invalid_access = true;
            return false;
        }
        if (Raw<T>() != expected) return false;
        Set(value);
        return true;
    }

    void AddTicks(u64 count) { ticks = count < ticks ? ticks - count : 0; }
};

class Callbacks64 final : public Dynarmic::A64::UserCallbacks {
public:
    explicit Callbacks64(Environment& env_) : env{env_} {}
    Dynarmic::A64::Jit* jit{};
    std::optional<u32> MemoryReadCode(u64 address) override { return env.ReadCode(address); }
    u64 MemoryRead(u64 address, size_t size) override { return env.Read(address, size); }
    Dynarmic::A64::Vector MemoryRead128(u64 address) override { return env.Read128(address); }
    void MemoryWrite(u64, u64, size_t) override { env.invalid_access = true; }
    void MemoryWrite128(u64, Dynarmic::A64::Vector) override { env.invalid_access = true; }
    void MemoryPrepareExclusive(u64 address, size_t size) override { env.Prepare(address, size); }
    bool MemoryWriteExclusive8(u64 a, u8 v, u8 e) override { return env.WriteExclusive(a, v, e); }
    bool MemoryWriteExclusive16(u64 a, u16 v, u16 e) override { return env.WriteExclusive(a, v, e); }
    bool MemoryWriteExclusive32(u64 a, u32 v, u32 e) override { return env.WriteExclusive(a, v, e); }
    bool MemoryWriteExclusive64(u64 a, u64 v, u64 e) override { return env.WriteExclusive(a, v, e); }
    bool MemoryWriteExclusive128(u64 a, Dynarmic::A64::Vector v, Dynarmic::A64::Vector e) override {
        return env.WriteExclusive(a, v, e);
    }
    void CallSVC(u32) override { jit->HaltExecution(); }
    void ExceptionRaised(u64, Dynarmic::A64::Exception) override {
        env.exception = true;
        jit->HaltExecution();
    }
    void AddTicks(u64 count) override { env.AddTicks(count); }
    u64 GetTicksRemaining() override { return env.ticks; }
    u64 GetCNTPCT() override { return 0; }
private:
    Environment& env;
};

class Callbacks32 final : public Dynarmic::A32::UserCallbacks {
public:
    explicit Callbacks32(Environment& env_) : env{env_} {}
    Dynarmic::A32::Jit* jit{};
    std::optional<u32> MemoryReadCode(u32 address) override { return env.ReadCode(address); }
    u64 MemoryRead(u32 address, size_t size) override { return env.Read(address, size); }
    void MemoryWrite(u32, u64, size_t) override { env.invalid_access = true; }
    void MemoryPrepareExclusive(u32 address, size_t size) override { env.Prepare(address, size); }
    bool MemoryWriteExclusive8(u32 a, u8 v, u8 e) override { return env.WriteExclusive(a, v, e); }
    bool MemoryWriteExclusive16(u32 a, u16 v, u16 e) override { return env.WriteExclusive(a, v, e); }
    bool MemoryWriteExclusive32(u32 a, u32 v, u32 e) override { return env.WriteExclusive(a, v, e); }
    bool MemoryWriteExclusive64(u32 a, u64 v, u64 e) override { return env.WriteExclusive(a, v, e); }
    void CallSVC(u32) override { jit->HaltExecution(); }
    void ExceptionRaised(u32, Dynarmic::A32::Exception) override {
        env.exception = true;
        jit->HaltExecution();
    }
    void AddTicks(u64 count) override { env.AddTicks(count); }
    u64 GetTicksRemaining() override { return env.ticks; }
private:
    Environment& env;
};

std::array<u64, 31> Run64(Environment& env, bool enabled = true,
                         std::array<u64, 31> registers = {},
                         const std::atomic<u64>* fast_table = nullptr, u32 table_mask = 0,
                         std::atomic<u64>* fast_hits = nullptr) {
    Callbacks64 callbacks{env};
    Dynarmic::A64::UserConfig config{};
    config.callbacks = &callbacks;
    config.global_monitor = &env.monitor;
    config.fastmem_exclusive_access = false;
    config.prepare_exclusive_memory = enabled;
    if (fast_table) {
        config.fastmem_pointer = reinterpret_cast<uintptr_t>(env.data.data()) - env.data_address;
        config.fastmem_exclusive_access = true;
        config.exclusive_prepare_table = fast_table;
        config.exclusive_prepare_table_mask = table_mask;
        config.exclusive_prepare_fast_hits = fast_hits;
    }
    Dynarmic::A64::Jit jit{config};
    callbacks.jit = &jit;
    registers[0] = env.data_address;
    jit.SetRegisters(registers);
    jit.Run();
    CHECK_FALSE(env.exception);
    CHECK_FALSE(env.invalid_access);
    CHECK(env.memory_locked);
    CHECK(env.preparation_unlocked);
    return jit.GetRegisters();
}
} // namespace

TEST_CASE("A64 guarded fastmem prepares once then keeps scalar and pair exclusives inline",
          "[deferred_writes][deferred_exclusive_jit]") {
    for (u32 width = 0; width < 5; ++width) {
        CAPTURE(width);
        Environment env;
        env.Set(Dynarmic::A64::Vector{7, 11});
        std::array<std::atomic<u64>, 16> table{};
        std::atomic<u64> fast_hits{};
        const u64 page = DataAddress >> 12;
        // A colliding tag must not authorize this address.
        table[page & 15].store(page + 16 + 1);
        env.before_exclusive = [&] { table[page & 15].store(page + 1, std::memory_order_release); };
        if (width < 4) {
            env.code = {0x085f7c01 | (width << 30), 0x08027c03 | (width << 30), SvcA64};
        } else {
            env.code = {0xc87f0801, 0xc8251003, SvcA64};
        }
        std::array<u64, 31> registers{};
        registers[3] = 23;
        registers[4] = 29;
        const auto result = Run64(env, true, registers, table.data(), 15, &fast_hits);
        CHECK(result[1] == 7);
        CHECK(result[width == 4 ? 5 : 2] == 0);
        CHECK(env.Raw<u64>() == 23);
        if (width == 4) CHECK(env.Raw<Dynarmic::A64::Vector>() == Dynarmic::A64::Vector{23, 29});
        CHECK(env.events == std::vector<char>{'P'});
        CHECK(fast_hits.load() == 1);
        CHECK(env.prepared_sizes == std::vector<size_t>{size_t{1} << width});
    }
}

TEST_CASE("A64 guarded fastmem preserves CAS CASP and LDADD with an already excluded page",
          "[deferred_writes][deferred_exclusive_jit]") {
    for (u32 opcode : {0xc8a17c03U, 0x48227c04U, 0xf8230001U}) {
        CAPTURE(opcode);
        Environment env;
        env.Set(Dynarmic::A64::Vector{7, 11});
        std::array<std::atomic<u64>, 16> table{};
        std::atomic<u64> fast_hits{};
        table[(DataAddress >> 12) & 15].store((DataAddress >> 12) + 1);
        env.code = {opcode, SvcA64};
        std::array<u64, 31> registers{};
        registers[1] = 7;
        registers[2] = 7;
        registers[3] = opcode == 0x48227c04 ? 11 : 23;
        registers[4] = 23;
        registers[5] = 29;
        const auto result = Run64(env, true, registers, table.data(), 15, &fast_hits);
        CHECK(env.events.empty());
        CHECK(fast_hits.load() == 2);
        CHECK(env.prepared_sizes.empty());
        CHECK(env.Raw<u64>() == (opcode == 0xf8230001 ? 30 : 23));
        if (opcode == 0x48227c04) {
            CHECK(result[2] == 7);
            CHECK(result[3] == 11);
            CHECK(env.Raw<Dynarmic::A64::Vector>() == Dynarmic::A64::Vector{23, 29});
        } else {
            CHECK(result[1] == 7);
        }
    }
}

TEST_CASE("A64 guarded fastmem slow store preserves the scalar and vector input registers",
          "[deferred_writes][deferred_exclusive_jit]") {
    for (bool pair : {false, true}) {
        Environment env;
        env.Set(Dynarmic::A64::Vector{7, 11});
        std::array<std::atomic<u64>, 16> table{};
        // Both preparations deliberately miss: exercise the out-of-line call with live values.
        env.code = pair ? std::vector<u32>{0xc87f0801, 0xc8251003, SvcA64}
                        : std::vector<u32>{0xc85f7c01, 0xc8027c03, SvcA64};
        std::array<u64, 31> registers{};
        registers[3] = 23;
        registers[4] = 29;
        const auto result = Run64(env, true, registers, table.data(), 15);
        CHECK(result[1] == 7);
        CHECK(result[pair ? 5 : 2] == 0);
        CHECK(env.Raw<u64>() == 23);
        if (pair) CHECK(env.Raw<Dynarmic::A64::Vector>() == Dynarmic::A64::Vector{23, 29});
        CHECK(env.events == std::vector<char>{'P', 'P'});
        CHECK(env.prepared_sizes == std::vector<size_t>{pair ? 16U : 8U, pair ? 16U : 8U});
    }
}

TEST_CASE("A32 guarded fastmem skips the host after the first page preparation",
          "[deferred_writes][deferred_exclusive_jit]") {
    Environment env;
    env.Set(u64{7});
    env.code = {0xe1901f9f, 0xe1802f93, SvcA32};
    std::array<std::atomic<u64>, 16> table{};
    env.before_exclusive = [&] {
        table[(DataAddress >> 12) & 15].store((DataAddress >> 12) + 1, std::memory_order_release);
    };
    Callbacks32 callbacks{env};
    Dynarmic::A32::UserConfig config{};
    config.callbacks = &callbacks;
    config.global_monitor = &env.monitor;
    config.fastmem_pointer = reinterpret_cast<uintptr_t>(env.data.data()) - DataAddress;
    config.fastmem_exclusive_access = true;
    config.prepare_exclusive_memory = true;
    config.exclusive_prepare_table = table.data();
    config.exclusive_prepare_table_mask = 15;
    Dynarmic::A32::Jit jit{config};
    callbacks.jit = &jit;
    jit.SetCpsr(0x1d0);
    jit.Regs()[0] = static_cast<u32>(DataAddress);
    jit.Regs()[3] = 23;
    jit.Run();
    CHECK_FALSE(env.exception);
    CHECK_FALSE(env.invalid_access);
    CHECK(env.preparation_unlocked);
    CHECK(jit.Regs()[1] == 7);
    CHECK(jit.Regs()[2] == 0);
    CHECK(env.Raw<u64>() == 23);
    CHECK(env.events == std::vector<char>{'P'});
}

TEST_CASE("A64 guarded fastmem prepares the full crossing range despite a first-page cache hit",
          "[deferred_writes][deferred_exclusive_jit]") {
    Environment env;
    env.data_address = DataAddress + 4093;
    env.Set(u64{7});
    env.code = {0xc85f7c01, 0xc8027c03, SvcA64};
    std::array<std::atomic<u64>, 16> table{};
    table[(env.data_address >> 12) & 15].store((env.data_address >> 12) + 1);
    std::atomic<u64> fast_hits{};
    std::array<u64, 31> registers{};
    registers[3] = 23;
    const auto result = Run64(env, true, registers, table.data(), 15, &fast_hits);
    CHECK(result[1] == 7);
    CHECK(result[2] == 0);
    CHECK(env.Raw<u64>() == 23);
    CHECK(env.events == std::vector<char>{'P', 'P'});
    CHECK(env.prepared_sizes == std::vector<size_t>{8, 8});
    CHECK(fast_hits.load() == 0);
}

TEST_CASE("A64 exclusive preparation precedes the monitor for every scalar width",
          "[deferred_writes][deferred_exclusive_jit]") {
    for (u32 width = 0; width < 4; ++width) {
        CAPTURE(width);
        Environment env;
        env.Set(u64{7});
        // LDXR{B,H,W,X} r1,[x0]; STXR{B,H,W,X} w2,r3,[x0]; SVC.
        env.code = {0x085f7c01 | (width << 30), 0x08027c03 | (width << 30), SvcA64};
        std::array<u64, 31> regs{};
        regs[3] = 23;
        const auto result = Run64(env, true, regs);
        CHECK(result[1] == 7);
        CHECK(result[2] == 0);
        CHECK(env.Raw<u64>() == 23);
        CHECK(env.events == std::vector<char>{'P', 'R', 'P', 'W'});
        CHECK(env.prepared_sizes == std::vector<size_t>{size_t{1} << width, size_t{1} << width});
    }
}

TEST_CASE("A64 pair exclusive preparation covers the full 128-bit access",
          "[deferred_writes][deferred_exclusive_jit]") {
    Environment env;
    env.Set(Dynarmic::A64::Vector{7, 11});
    // LDXP x1,x2,[x0]; STXP w5,x3,x4,[x0]; SVC.
    env.code = {0xc87f0801, 0xc8251003, SvcA64};
    std::array<u64, 31> regs{};
    regs[3] = 23;
    regs[4] = 29;
    const auto result = Run64(env, true, regs);
    CHECK(result[1] == 7);
    CHECK(result[2] == 11);
    CHECK(result[5] == 0);
    CHECK(env.Raw<Dynarmic::A64::Vector>() == Dynarmic::A64::Vector{23, 29});
    CHECK(env.events == std::vector<char>{'P', 'R', 'P', 'W'});
    CHECK(env.prepared_sizes == std::vector<size_t>{16, 16});
}

TEST_CASE("A64 exclusive preparation preserves failed store and disabled behavior",
          "[deferred_writes][deferred_exclusive_jit]") {
    Environment env;
    env.Set(u64{7});
    env.code = {0xc85f7c01, 0xc8027c03, SvcA64};
    std::array<u64, 31> regs{};
    regs[3] = 23;
    SECTION("Disabled hook leaves the existing monitor path") {
        const auto result = Run64(env, false, regs);
        CHECK(result[2] == 0);
        CHECK(env.Raw<u64>() == 23);
        CHECK(env.prepared_sizes.empty());
        CHECK(env.events == std::vector<char>{'R', 'W'});
    }
    SECTION("A value changed during preparation makes STXR fail") {
        env.before_exclusive = [&] {
            if (env.prepared_sizes.size() == 2) env.Set(u64{31});
        };
        const auto result = Run64(env, true, regs);
        CHECK(result[1] == 7);
        CHECK(result[2] == 1);
        CHECK(env.Raw<u64>() == 31);
        CHECK(env.events == std::vector<char>{'P', 'R', 'P', 'W'});
    }
    SECTION("STXR with no local reservation performs no memory access") {
        env.code = {0xc8027c03, SvcA64};
        const auto result = Run64(env, true, regs);
        CHECK(result[2] == 1);
        CHECK(env.Raw<u64>() == 7);
        CHECK(env.events.empty());
    }
}

TEST_CASE("A64 LSE CAS CASP and LDADD use preparation before both monitor operations",
          "[deferred_writes][deferred_exclusive_jit]") {
    SECTION("CAS equal and unequal") {
        for (u64 expected : {u64{7}, u64{8}}) {
            Environment env;
            env.Set(u64{7});
            env.code = {0xc8a17c03, SvcA64}; // CAS x1,x3,[x0].
            std::array<u64, 31> regs{};
            regs[1] = expected;
            regs[3] = 23;
            const auto result = Run64(env, true, regs);
            CHECK(result[1] == 7);
            CHECK(env.Raw<u64>() == (expected == 7 ? 23 : 7));
            CHECK(env.events == std::vector<char>{'P', 'R', 'P', 'W'});
            CHECK(env.prepared_sizes == std::vector<size_t>{8, 8});
        }
    }
    SECTION("CASP 128-bit pair") {
        Environment env;
        env.Set(Dynarmic::A64::Vector{7, 11});
        env.code = {0x48227c04, SvcA64}; // CASP x2,x3,x4,x5,[x0].
        std::array<u64, 31> regs{};
        regs[2] = 7;
        regs[3] = 11;
        regs[4] = 23;
        regs[5] = 29;
        const auto result = Run64(env, true, regs);
        CHECK(result[2] == 7);
        CHECK(result[3] == 11);
        CHECK(env.Raw<Dynarmic::A64::Vector>() == Dynarmic::A64::Vector{23, 29});
        CHECK(env.events == std::vector<char>{'P', 'R', 'P', 'W'});
        CHECK(env.prepared_sizes == std::vector<size_t>{16, 16});
    }
    SECTION("LDADD") {
        Environment env;
        env.Set(u64{7});
        env.code = {0xf8230001, SvcA64}; // LDADD x3,x1,[x0].
        std::array<u64, 31> regs{};
        regs[3] = 23;
        const auto result = Run64(env, true, regs);
        CHECK(result[1] == 7);
        CHECK(env.Raw<u64>() == 30);
        CHECK(env.events == std::vector<char>{'P', 'R', 'P', 'W'});
        CHECK(env.prepared_sizes == std::vector<size_t>{8, 8});
    }
}

TEST_CASE("A64 exclusive preparation drains a real journal and retains the physical exclusion",
          "[deferred_writes][deferred_exclusive_jit]") {
    using namespace VideoCommon::DeferredWrites;
    bool fastmem = false;
    SECTION("HostCall exclusive path") {}
    SECTION("Guarded inline exclusive path") { fastmem = true; }
    auto journal = std::make_shared<Journal>(1, 4);
    Core::DeferredKernelWrites registry;
    const auto* table = fastmem ? registry.EnableFastTable() : nullptr;
    Environment env;
    env.Set(u64{1});
    const auto lease = journal->GrantLease(DataAddress, 0x8000, 1);
    REQUIRE(lease);
    Write write;
    write.physical_address = DataAddress;
    write.device_address = 0x8000;
    write.mapping_generation = 1;
    write.lease_generation = lease->generation;
    write.size = sizeof(u64);
    const u64 committed = 7;
    std::memcpy(write.bytes.data(), &committed, sizeof(committed));
    REQUIRE(journal->TryPublishLeased(0, write, [&]() noexcept {
        env.Set(committed);
        return true;
    }).sequence == 1);

    u64 gpu_value = 1;
    size_t applications{};
    bool prepared = true;
    bool applied_before_memory = true;
    bool grants_excluded = true;
    auto apply = [&](const Write& pending) {
        ++applications;
        std::memcpy(&gpu_value, pending.bytes.data(), sizeof(gpu_value));
        return true;
    };
    env.before_exclusive = [&] {
        prepared &= registry.Prepare(DataAddress, env.prepared_sizes.back(),
            [](VAddr page) { return std::optional<PAddr>{page}; },
            [&](PAddr physical) {
                const std::array ranges{TransitionRange{TransitionSpace::Physical, physical, 4096}};
                return BeginSharedWriterPermission(journal, ranges);
            },
            [&](const auto& permission) {
                return journal->DrainThrough(permission->Watermark(), apply) == DrainResult::Complete &&
                       permission->MarkReady();
            });
        grants_excluded &= !journal->GrantLease(DataAddress, 0x18000, 2);
    };
    env.at_memory = [&] { applied_before_memory &= journal->AppliedThrough() == 1; };
    env.code = {0xc85f7c01, 0xc8027c03, SvcA64}; // LDXR x1; STXR w2,x3.
    std::array<u64, 31> registers{};
    registers[3] = 23;
    const auto result = Run64(env, true, registers, table,
                              static_cast<u32>(Core::DeferredKernelWrites::FastTableMask));
    CHECK(prepared);
    CHECK(applied_before_memory);
    CHECK(grants_excluded);
    CHECK(result[1] == committed);
    CHECK(result[2] == 0);
    CHECK(env.Raw<u64>() == 23);
    CHECK(gpu_value == committed);
    CHECK(applications == 1);
    CHECK(env.events == (fastmem ? std::vector<char>{'P'} : std::vector<char>{'P', 'R', 'P', 'W'}));
    CHECK(registry.PhysicalPageCount() == 1);
    CHECK_FALSE(journal->GrantLease(DataAddress, 0x8000, 2));
    CHECK(journal->TryPublishLeased(0, write, []() noexcept { return true; }).status ==
          PublishStatus::LeaseUnavailable);
    CHECK(journal->DrainThrough(1, apply) == DrainResult::Complete);
    CHECK(applications == 1);
    CHECK(env.Raw<u64>() == 23);
    registry.Clear();
    CHECK(journal->GrantLease(DataAddress, 0x8000, 2));
}

TEST_CASE("A32 exclusive preparation runs before the shared monitor for every width",
          "[deferred_writes][deferred_exclusive_jit]") {
    struct Instructions { u32 load; u32 store; size_t size; };
    const std::array instructions{
        Instructions{0xe1d01f9f, 0xe1c02f93, 1}, // LDREXB r1,[r0]; STREXB r2,r3,[r0].
        Instructions{0xe1f01f9f, 0xe1e02f93, 2}, // LDREXH r1,[r0]; STREXH r2,r3,[r0].
        Instructions{0xe1901f9f, 0xe1802f93, 4}, // LDREX r1,[r0]; STREX r2,r3,[r0].
        Instructions{0xe1b02f9f, 0xe1a01f94, 8}, // LDREXD r2,r3,[r0]; STREXD r1,r4,r5,[r0].
    };
    for (const auto& instruction : instructions) {
        CAPTURE(instruction.size);
        Environment env;
        env.Set(instruction.size == 8 ? (u64{11} << 32) | 7 : u64{7});
        env.code = {instruction.load, instruction.store, SvcA32};
        Callbacks32 callbacks{env};
        Dynarmic::A32::UserConfig config{};
        config.callbacks = &callbacks;
        config.global_monitor = &env.monitor;
        config.fastmem_exclusive_access = false;
        config.prepare_exclusive_memory = true;
        Dynarmic::A32::Jit jit{config};
        callbacks.jit = &jit;
        jit.SetCpsr(0x1d0);
        jit.Regs()[0] = static_cast<u32>(DataAddress);
        jit.Regs()[3] = 23;
        jit.Regs()[4] = 23;
        jit.Regs()[5] = 29;
        jit.Run();
        CHECK_FALSE(env.exception);
        CHECK_FALSE(env.invalid_access);
        CHECK(env.preparation_unlocked);
        CHECK(env.memory_locked);
        if (instruction.size == 8) {
            CHECK(jit.Regs()[2] == 7);
            CHECK(jit.Regs()[3] == 11);
            CHECK(jit.Regs()[1] == 0);
            CHECK(env.Raw<u64>() == ((u64{29} << 32) | 23));
        } else {
            CHECK(jit.Regs()[1] == 7);
            CHECK(jit.Regs()[2] == 0);
            CHECK(env.Raw<u64>() == 23);
        }
        CHECK(env.events == std::vector<char>{'P', 'R', 'P', 'W'});
        CHECK(env.prepared_sizes == std::vector<size_t>{instruction.size, instruction.size});
    }
}
