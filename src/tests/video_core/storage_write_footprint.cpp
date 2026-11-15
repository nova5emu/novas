// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/ir_opt/storage_write_footprint.h"
#include "video_core/buffer_cache/storage_base_snapshot.h"
#include "video_core/buffer_cache/download_copy_plan.h"
#include "video_core/gpu_write_trace_dump.h"

namespace {
namespace IR = Shader::IR;
using Reason = Shader::StorageWriteReason;
struct Fixture {
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;

    Fixture() {
        program.stage = Shader::Stage::Compute;
        program.blocks.push_back(&block);
        program.post_order_blocks.push_back(&block);
        program.info.storage_buffers_descriptors.push_back({0, 0, 1, true});
        program.info.storage_buffers_descriptors.push_back({0, 16, 1, true});
    }
    IR::Value Emit(IR::Opcode opcode, std::initializer_list<IR::Value> args) {
        block.AppendNewInst(opcode, args);
        return IR::Value{&block.Instructions().back()};
    }
    void Store(u32 binding, IR::Value offset) {
        Emit(IR::Opcode::WriteStorage32, {IR::Value{binding}, offset, IR::Value{u32{1}}});
    }
    Shader::StorageWriteFootprint Analyze(size_t index = 0) {
        Shader::Optimization::CollectStorageWriteFootprints(program);
        return index < program.info.storage_write_footprints.size()
            ? program.info.storage_write_footprints[index] : Shader::StorageWriteFootprint{};
    }
};
IR::Value U(u32 value) { return IR::Value{value}; }
}

TEST_CASE("GPU write trace dump contains translated stores and binding rejection reasons",
          "[storage_footprint]") {
    Fixture f;
    f.program.workgroup_size = {64, 1, 1};
    f.Store(0, U(64));
    f.Store(1, f.Emit(IR::Opcode::GetCbufU32, {U(0), U(32)}));
    f.Analyze();
    const auto dump = VideoCommon::GpuWriteTrace::PrepareComputeDump(f.program);
    CHECK(dump.find("ir=post_translate_pre_spirv block=64x1x1") != std::string::npos);
    CHECK(dump.find("binding=0 cb=0+0x0") != std::string::npos);
    CHECK(dump.find("reason=none envelope=[0x40,0x44)") != std::string::npos);
    CHECK(dump.find("reason=dynamic_or_unsupported_offset") != std::string::npos);
    CHECK(dump.find("WriteStorage32") != std::string::npos);
    CHECK(f.program.info.storage_write_footprints[0].known);
    CHECK_FALSE(f.program.info.storage_write_footprints[1].known);
}

TEST_CASE("Storage footprint unions constant stores without changing IR", "[storage_footprint]") {
    Fixture f;
    f.Store(0, U(64));
    f.Store(0, U(132));
    const auto count = f.block.Instructions().size();
    const auto range = f.Analyze();
    REQUIRE(range.known);
    CHECK(range.reason == Reason::None);
    CHECK(range.begin == 64);
    CHECK(range.end == 136);
    CHECK(f.block.Instructions().size() == count);
    CHECK_FALSE(f.Analyze(1).known); // No write proves no footprint, not an empty optimization.
    CHECK(f.Analyze(1).reason == Reason::NoWrites);
}

TEST_CASE("Storage footprint handles constant modular arithmetic and vector width",
          "[storage_footprint]") {
    Fixture f;
    const auto offset = f.Emit(IR::Opcode::IAdd32, {U(0xfffffff0), U(32)});
    const auto vector = f.Emit(IR::Opcode::CompositeConstructU32x4, {U(1), U(2), U(3), U(4)});
    f.Emit(IR::Opcode::WriteStorage128, {U(0), offset, vector});
    const auto range = f.Analyze();
    REQUIRE(range.known);
    CHECK(range.begin == 16);
    CHECK(range.end == 32);
}

TEST_CASE("Storage footprint includes compare exchange even when its result is unused",
          "[storage_footprint]") {
    Fixture f;
    f.Emit(IR::Opcode::StorageAtomicCompareExchange32, {U(0), U(128), U(0), U(1)});
    const auto range = f.Analyze();
    REQUIRE(range.known);
    CHECK(range.begin == 128);
    CHECK(range.end == 132);
}

TEST_CASE("Storage footprint covers 64-bit and paired atomics", "[storage_footprint]") {
    Fixture f;
    SECTION("64-bit integer") {
        f.Emit(IR::Opcode::StorageAtomicIAdd64, {U(0), U(128), IR::Value{u64{1}}});
    }
    SECTION("paired 32-bit") {
        const auto pair = f.Emit(IR::Opcode::CompositeConstructU32x2, {U(1), U(2)});
        f.Emit(IR::Opcode::StorageAtomicIAdd32x2, {U(0), U(128), pair});
    }
    const auto range = f.Analyze();
    REQUIRE(range.known);
    CHECK(range.begin == 128);
    CHECK(range.end == 136);
}

TEST_CASE("Storage footprint bounds alignment residual without reading a constant buffer",
          "[storage_footprint]") {
    Fixture f;
    const auto base = f.Emit(IR::Opcode::GetCbufU32, {U(0), U(16)});
    const auto source = f.Emit(IR::Opcode::IAdd32, {base, U(512)});
    const auto aligned = f.Emit(IR::Opcode::BitwiseAnd32, {base, U(~u32{255})});
    f.Store(0, f.Emit(IR::Opcode::ISub32, {source, aligned}));
    const auto range = f.Analyze();
    REQUIRE(range.known);
    CHECK(range.begin == 512);
    CHECK(range.end == 771); // base residual can be 255, plus the four-byte store.
}

TEST_CASE("Storage footprint rejects dynamic writes without discarding independent bindings",
          "[storage_footprint]") {
    Fixture f;
    f.Store(0, U(64));
    f.Store(0, f.Emit(IR::Opcode::GetCbufU32, {U(0), U(32)}));
    f.Store(1, U(96));
    CHECK_FALSE(f.Analyze().known);
    CHECK(f.Analyze().reason == Reason::DynamicOrUnsupportedOffset);
    CHECK(f.Analyze(1).known);
}

TEST_CASE("Storage footprint rejects ambiguous destinations and runtime globals",
          "[storage_footprint]") {
    Fixture f;
    f.Store(0, U(64));
    Reason expected{};
    SECTION("dynamic SSBO index") {
        expected = Reason::DynamicBinding;
        const auto index = f.Emit(IR::Opcode::GetCbufU32, {U(0), U(32)});
        f.Emit(IR::Opcode::WriteStorage32, {index, U(128), U(1)});
    }
    SECTION("unresolved global store") {
        expected = Reason::UnresolvedGlobal;
        f.Emit(IR::Opcode::WriteGlobal32, {IR::Value{u64{4096}}, U(1)});
    }
    SECTION("native bounds") {
        expected = Reason::NativeGlobalBounds;
        f.program.info.native_global_bounds = true;
    }
    SECTION("indirect descriptor") {
        expected = Reason::IndirectDescriptor;
        f.program.info.storage_buffers_descriptors[0].indirect_offset = 16;
    }
    SECTION("graphics stage") {
        expected = Reason::NotCompute;
        f.program.stage = Shader::Stage::Fragment;
    }
    CHECK_FALSE(f.Analyze().known);
    CHECK(f.Analyze().reason == expected);
    REQUIRE(f.program.info.storage_write_footprints.size() == 2);
    if (expected != Reason::IndirectDescriptor) {
        CHECK(f.Analyze(1).reason == expected);
    }
}

TEST_CASE("Storage footprint rejects wrapped envelopes and mismatched pointer bases",
          "[storage_footprint]") {
    Fixture f;
    const auto base = f.Emit(IR::Opcode::GetCbufU32, {U(0), U(16)});
    Reason expected = Reason::DynamicOrUnsupportedOffset;
    SECTION("end crosses 32-bit address space") {
        expected = Reason::Overflow;
        f.Store(0, U(0xffffffff));
    }
    SECTION("alignment residual may wrap") {
        expected = Reason::Overflow;
        const auto source = f.Emit(IR::Opcode::IAdd32, {base, U(0xfffffff0)});
        const auto aligned = f.Emit(IR::Opcode::BitwiseAnd32, {base, U(~u32{255})});
        f.Store(0, f.Emit(IR::Opcode::ISub32, {source, aligned}));
    }
    SECTION("different cbuf word") {
        const auto other = f.Emit(IR::Opcode::GetCbufU32, {U(0), U(20)});
        const auto aligned = f.Emit(IR::Opcode::BitwiseAnd32, {other, U(~u32{255})});
        f.Store(0, f.Emit(IR::Opcode::ISub32, {base, aligned}));
    }
    SECTION("separate load of the same cbuf word") {
        const auto other = f.Emit(IR::Opcode::GetCbufU32, {U(0), U(16)});
        const auto aligned = f.Emit(IR::Opcode::BitwiseAnd32, {other, U(~u32{255})});
        f.Store(0, f.Emit(IR::Opcode::ISub32, {base, aligned}));
    }
    CHECK_FALSE(f.Analyze().known);
    CHECK(f.Analyze().reason == expected);
}

TEST_CASE("Storage footprint reports traversal budget without erasing another binding",
          "[storage_footprint]") {
    Fixture f;
    auto offset = U(16);
    for (unsigned i = 0; i < 256; ++i) {
        offset = f.Emit(IR::Opcode::IAdd32, {offset, U(1)});
    }
    f.Store(0, offset);
    f.Store(1, U(64));
    CHECK(f.Analyze().reason == Reason::Budget);
    CHECK_FALSE(f.Analyze().known);
    CHECK(f.Analyze(1).known);
}

TEST_CASE("Storage footprint reasons distinguish absent analysis from rejection",
          "[storage_footprint]") {
    Fixture f;
    CHECK(f.program.info.storage_write_footprints.empty());
    CHECK(Shader::StorageWriteFootprint{}.reason == Reason::NotCollected);
    CHECK(Shader::StorageWriteReasonName(Reason::NotCollected) == "not_collected");
    CHECK(Shader::StorageWriteReasonName(Reason::UnsupportedStore) == "unsupported_store");
    CHECK(f.Analyze().reason == Reason::NoWrites);
    f.Store(0, U(64));
    CHECK(f.Analyze().reason == Reason::None);
}

namespace {
IR::Value DescriptorOffset(Fixture& f, IR::Value extra, u32 binding = 0, u32 cbuf_offset = 32) {
    f.program.info.storage_buffers_descriptors[binding].cbuf_offset = cbuf_offset;
    const auto vector = f.Emit(IR::Opcode::GetCbufU32x2, {U(0), U(cbuf_offset)});
    const auto low = f.Emit(IR::Opcode::CompositeExtractU32x2, {vector, U(0)});
    const auto source = f.Emit(IR::Opcode::IAdd32, {low, extra});
    const auto separate_load = f.Emit(IR::Opcode::GetCbufU32, {U(0), U(cbuf_offset)});
    const auto aligned = f.Emit(IR::Opcode::BitwiseAnd32, {separate_load, U(~u32{15})});
    return f.Emit(IR::Opcode::ISub32, {source, aligned});
}
auto Evaluate(const Shader::StorageWriteFootprint& footprint, std::array<u32, 3> grid = {1, 1, 1},
              std::array<u32, 3> block = {1, 1, 1}, std::optional<u32> base = std::nullopt,
              u64 size = 65536) {
    return Shader::Optimization::EvaluateStorageWriteFootprint(footprint, grid, block, false, base, size);
}
}

TEST_CASE("Storage footprint defers descriptor word equivalence until a snapshot is supplied",
          "[storage_footprint]") {
    Fixture f;
    f.Store(0, DescriptorOffset(f, U(32)));
    const auto footprint = f.Analyze();
    CHECK_FALSE(footprint.known);
    REQUIRE(footprint.symbolic);
    CHECK(footprint.reason == Reason::NeedsDispatch);
    REQUIRE(footprint.terms.size() == 1);
    CHECK(footprint.terms[0].needs_base);
    CHECK(Evaluate(footprint).reason == Reason::BaseRequired);
    const auto result = Evaluate(footprint, {1, 1, 1}, {1, 1, 1}, 0x1003);
    REQUIRE(result.known);
    REQUIRE(result.intervals.size() == 1);
    CHECK(result.intervals[0].begin == 32);
    CHECK(result.intervals[0].end == 39);
}

TEST_CASE("Storage footprint recomputes affine invocation ranges for each launch",
          "[storage_footprint]") {
    Fixture f;
    const auto groups = f.Emit(IR::Opcode::WorkgroupId, {});
    const auto group = f.Emit(IR::Opcode::CompositeExtractU32x3, {groups, U(0)});
    const auto locals = f.Emit(IR::Opcode::LocalInvocationId, {});
    const auto local = f.Emit(IR::Opcode::CompositeExtractU32x3, {locals, U(0)});
    const auto scaled_group = f.Emit(IR::Opcode::IMul32, {group, U(32)});
    const auto id = f.Emit(IR::Opcode::IAdd32, {scaled_group, local});
    const auto bytes = f.Emit(IR::Opcode::ShiftLeftLogical32, {id, U(2)});
    f.Store(0, DescriptorOffset(f, bytes));
    const auto footprint = f.Analyze();
    REQUIRE(footprint.symbolic);
    const auto first = Evaluate(footprint, {2, 1, 1}, {32, 1, 1}, 0x2000);
    REQUIRE(first.known);
    REQUIRE(first.intervals.size() == 1);
    CHECK(first.intervals[0].begin == 0);
    CHECK(first.intervals[0].end == 256);
    const auto second = Evaluate(footprint, {3, 1, 1}, {32, 1, 1}, 0x2000);
    REQUIRE(second.known);
    CHECK(second.intervals[0].end == 384);
}

TEST_CASE("Storage footprint retains holes between stores and rejects partial terms",
          "[storage_footprint]") {
    Fixture f;
    f.Store(0, DescriptorOffset(f, U(0)));
    f.Store(0, DescriptorOffset(f, U(5440)));
    const auto footprint = f.Analyze();
    REQUIRE(footprint.terms.size() == 2);
    const auto result = Evaluate(footprint, {1, 1, 1}, {1, 1, 1}, 0x1000);
    REQUIRE(result.known);
    REQUIRE(result.intervals.size() == 2);
    CHECK(result.intervals[0].end == 4);
    CHECK(result.intervals[1].begin == 5440);
    f.Store(0, f.Emit(IR::Opcode::GetCbufU32, {U(3), U(0)}));
    const auto rejected = f.Analyze();
    CHECK_FALSE(rejected.symbolic);
    CHECK_FALSE(rejected.known);
    CHECK(rejected.terms.empty());
}

TEST_CASE("Storage footprint canceled descriptor loads still require a snapshot",
          "[storage_footprint]") {
    Fixture f;
    f.program.info.storage_buffers_descriptors[0].cbuf_offset = 32;
    const auto a = f.Emit(IR::Opcode::GetCbufU32, {U(0), U(32)});
    const auto b = f.Emit(IR::Opcode::GetCbufU32, {U(0), U(32)});
    f.Store(0, f.Emit(IR::Opcode::ISub32, {a, b}));
    const auto footprint = f.Analyze();
    REQUIRE(footprint.symbolic);
    CHECK(Evaluate(footprint).reason == Reason::BaseRequired);
}

TEST_CASE("Storage footprint evaluation checks caps and binding bounds without clipping",
          "[storage_footprint]") {
    Shader::StorageWriteFootprint footprint;
    footprint.symbolic = true;
    footprint.reason = Reason::NeedsDispatch;
    footprint.terms.push_back({.constant = 64, .width = 4});
    CHECK(Evaluate(footprint, {0, 1, 1}).reason == Reason::InvalidDimensions);
    CHECK(Evaluate(footprint, {4097, 1, 1}).reason == Reason::InvocationLimit);
    CHECK(Evaluate(footprint, {~0U, ~0U, ~0U}).reason == Reason::InvocationLimit);
    CHECK(Evaluate(footprint, {1, 1, 1}, {1, 1, 1}, std::nullopt, 67).reason == Reason::OutsideBinding);
    CHECK(Shader::Optimization::EvaluateStorageWriteFootprint(
        footprint, {1, 1, 1}, {1, 1, 1}, true, std::nullopt, 1024).reason == Reason::IndirectDispatch);
    footprint.terms.resize(9, footprint.terms[0]);
    CHECK(Evaluate(footprint, {4096, 1, 1}).reason == Reason::IntervalLimit);
}

TEST_CASE("Storage footprint evaluates U32 arithmetic modulo and rejects crossing stores",
          "[storage_footprint]") {
    Shader::StorageWriteFootprint footprint;
    footprint.symbolic = true;
    Shader::StorageWriteTerm term{.constant = 16, .width = 4};
    term.coefficients[0] = ~u32{15}; // -16 modulo 2^32.
    footprint.terms.push_back(term);
    auto result = Evaluate(footprint, {2, 1, 1});
    REQUIRE(result.known);
    REQUIRE(result.intervals.size() == 2);
    CHECK(result.intervals[0].begin == 0);
    CHECK(result.intervals[1].begin == 16);
    footprint.terms[0].constant = 0xffffffff;
    CHECK(Evaluate(footprint).reason == Reason::Overflow);
}

TEST_CASE("Storage footprint covers the constant destination layout in the 59b8 dump",
          "[storage_footprint]") {
    Fixture f;
    const auto vector = f.Emit(IR::Opcode::CompositeConstructU32x4, {U(0), U(0), U(0), U(0)});
    for (u32 offset = 0; offset <= 560; offset += 16) {
        if (offset != 16) {
            f.Emit(IR::Opcode::WriteStorage128, {U(0), DescriptorOffset(f, U(offset)), vector});
        }
    }
    f.Emit(IR::Opcode::WriteStorage128, {U(0), DescriptorOffset(f, U(5456)), vector});
    const auto pair = f.Emit(IR::Opcode::CompositeConstructU32x2, {U(0), U(0)});
    f.Emit(IR::Opcode::WriteStorage64, {U(0), DescriptorOffset(f, U(16)), pair});
    for (u32 offset : {24U, 28U, 5440U}) f.Store(0, DescriptorOffset(f, U(offset)));
    const auto footprint = f.Analyze();
    REQUIRE(footprint.symbolic);
    REQUIRE(footprint.terms.size() == 40);
    const auto result = Evaluate(footprint, {1, 1, 1}, {1, 1, 1}, 0x93000, 0x3980);
    REQUIRE(result.known);
    REQUIRE(result.intervals.size() == 3);
    CHECK(result.intervals[0].begin == 0);
    CHECK(result.intervals[0].end == 576);
    CHECK(result.intervals[1].begin == 5440);
    CHECK(result.intervals[1].end == 5444);
    CHECK(result.intervals[2].begin == 5456);
    CHECK(result.intervals[2].end == 5472);
    CHECK(Shader::Optimization::CanApplyStorageWriteFootprint(
        footprint, result, 0x12eb94000, 0x3980, false));
}

TEST_CASE("Storage footprint leaves bitfield invocation expressions unproven",
          "[storage_footprint]") {
    Fixture f;
    const auto vector = f.Emit(IR::Opcode::WorkgroupId, {});
    const auto group = f.Emit(IR::Opcode::CompositeExtractU32x3, {vector, U(0)});
    const auto sliced = f.Emit(IR::Opcode::BitFieldUExtract, {group, U(0), U(16)});
    f.Store(0, DescriptorOffset(f, sliced));
    const auto footprint = f.Analyze();
    CHECK_FALSE(footprint.known);
    CHECK_FALSE(footprint.symbolic);
    CHECK(footprint.terms.empty());
    CHECK(footprint.reason == Reason::DynamicOrUnsupportedOffset);
}

TEST_CASE("Storage footprint covers all seven f837 stores without filling record holes",
          "[storage_footprint]") {
    Fixture f;
    const auto groups = f.Emit(IR::Opcode::WorkgroupId, {});
    const auto group = f.Emit(IR::Opcode::CompositeExtractU32x3, {groups, U(0)});
    const auto locals = f.Emit(IR::Opcode::LocalInvocationId, {});
    const auto local = f.Emit(IR::Opcode::CompositeExtractU32x3, {locals, U(0)});
    const auto group_base = f.Emit(IR::Opcode::ShiftLeftLogical32, {group, U(6)});
    const auto gid = f.Emit(IR::Opcode::IAdd32, {group_base, local});
    const auto record = f.Emit(IR::Opcode::ShiftLeftLogical32, {gid, U(8)});
    const auto pair = f.Emit(IR::Opcode::CompositeConstructU32x2, {U(0), U(0)});
    for (u32 k : {128U, 136U, 144U, 152U, 244U, 160U, 168U}) {
        const auto extra = f.Emit(IR::Opcode::IAdd32, {record, U(k)});
        const auto offset = DescriptorOffset(f, extra, 1, 336);
        if (k == 128 || k == 144 || k == 160) {
            f.Emit(IR::Opcode::WriteStorage64, {U(1), offset, pair});
        } else {
            f.Store(1, offset);
        }
    }
    const auto footprint = f.Analyze(1);
    REQUIRE(footprint.symbolic);
    REQUIRE(footprint.terms.size() == 7);
    const auto result = Evaluate(footprint, {1, 1, 1}, {64, 1, 1}, 0x2b64fb00, 0x4000);
    REQUIRE(result.known);
    REQUIRE(result.intervals.size() == 256);
    u64 bytes{};
    for (const auto& interval : result.intervals) bytes += interval.end - interval.begin;
    CHECK(bytes == 2560);
    CHECK(Shader::Optimization::CanApplyStorageWriteFootprint(
        footprint, result, 0x2b64fb00, 0x4000, false));
    CHECK(result.intervals[0].begin == 128);
    CHECK(result.intervals[0].end == 140);
    CHECK(result.intervals[2].begin == 160);
    CHECK(result.intervals[2].end == 172);
    CHECK(result.intervals.back().begin == 63 * 256 + 244);
    CHECK(result.intervals.back().end == 63 * 256 + 248);
    CHECK(Evaluate(footprint, {2, 1, 1}, {64, 1, 1}, 0x2b64fb00, 0x4000).reason ==
          Reason::OutsideBinding);
    const auto dump = VideoCommon::GpuWriteTrace::PrepareComputeDump(f.program);
    CHECK(dump.find("store_term binding=1 index=0 width=8 constant=0x80") != std::string::npos);
    CHECK(dump.find("wg_coeff=16384,0,0 local_coeff=256,0,0 needs_base=true") != std::string::npos);
}

TEST_CASE("Storage footprint application requires a complete aligned resolved binding",
          "[storage_footprint]") {
    Fixture f;
    f.Store(0, DescriptorOffset(f, U(128)));
    const auto recipe = f.Analyze();
    REQUIRE(recipe.coherence_eligible);
    auto result = Evaluate(recipe, {1, 1, 1}, {1, 1, 1}, 0x1000, 4096);
    const auto can_apply = [&](u64 address = 0x1000, u64 size = 4096, bool split = false) {
        return Shader::Optimization::CanApplyStorageWriteFootprint(recipe, result, address, size, split);
    };
    REQUIRE(can_apply());
    CHECK_FALSE(can_apply(0x1000, 4096, true));
    CHECK_FALSE(can_apply(0));
    CHECK_FALSE(can_apply(0x1001));
    CHECK_FALSE(can_apply(0x1000, 131));
    CHECK_FALSE(can_apply((u64{1} << 36) - 4));
    SECTION("no snapshot") { result = Evaluate(recipe); }
    SECTION("partial store exceeds the binding") { result.intervals.push_back({4092, 4100}); }
    SECTION("empty proof") { result.intervals.clear(); }
    SECTION("overlapping proof") { result.intervals.push_back({128, 132}); }
    SECTION("unaligned copy") { result.intervals[0].end = 131; }
    SECTION("unknown expression") { result.known = false; }
    CHECK_FALSE(can_apply());
}

TEST_CASE("Storage footprint application retains full bindings for subword stores and atomics",
          "[storage_footprint]") {
    Fixture f;
    f.Store(0, U(64));
    SECTION("unsigned byte") { f.Emit(IR::Opcode::WriteStorageU8, {U(0), U(129), U(1)}); }
    SECTION("signed byte") { f.Emit(IR::Opcode::WriteStorageS8, {U(0), U(129), U(1)}); }
    SECTION("unsigned halfword") { f.Emit(IR::Opcode::WriteStorageU16, {U(0), U(130), U(1)}); }
    SECTION("signed halfword") { f.Emit(IR::Opcode::WriteStorageS16, {U(0), U(130), U(1)}); }
    SECTION("atomic") { f.Emit(IR::Opcode::StorageAtomicIAdd32, {U(0), U(128), U(1)}); }
    const auto recipe = f.Analyze();
    const auto result = Evaluate(recipe);
    REQUIRE(result.known); // Diagnostic knowledge alone does not authorize narrowing.
    CHECK_FALSE(recipe.coherence_eligible);
    CHECK_FALSE(Shader::Optimization::CanApplyStorageWriteFootprint(
        recipe, result, 0x1000, 4096, false));
}

TEST_CASE("Storage footprint application rejects unresolved writes and native global bounds",
          "[storage_footprint]") {
    Fixture f;
    f.Store(0, U(64));
    SECTION("native bounds") { f.program.info.native_global_bounds = true; }
    SECTION("global store") { f.Emit(IR::Opcode::WriteGlobal32, {IR::Value{u64{4096}}, U(1)}); }
    SECTION("indirect descriptor") { f.program.info.storage_buffers_descriptors[0].indirect_offset = 16; }
    SECTION("unknown offset") { f.Store(0, f.Emit(IR::Opcode::GetCbufU32, {U(0), U(100)})); }
    const auto recipe = f.Analyze();
    CHECK_FALSE(Shader::Optimization::CanApplyStorageWriteFootprint(
        recipe, Evaluate(recipe), 0x1000, 4096, false));
}

TEST_CASE("Storage base proof uses the submitted UBO bytes including alignment residual",
          "[storage_footprint]") {
    Fixture f;
    f.Store(0, DescriptorOffset(f, U(128)));
    const auto recipe = f.Analyze();
    REQUIRE(Evaluate(recipe).reason == Reason::BaseRequired);
    std::array<u8, 48> backing{};
    const u32 descriptor = 0x1008;
    std::memcpy(backing.data() + 32, &descriptor, sizeof(descriptor));
    const auto submitted = backing;
    // Changing backing after submission cannot change the proof for this dispatch.
    backing.fill(0);
    const auto snapshot = VideoCommon::ReadStorageBaseSnapshot(submitted, 32);
    REQUIRE(snapshot == descriptor);
    const auto evaluated = Evaluate(recipe, {1, 1, 1}, {1, 1, 1}, snapshot, 4096);
    REQUIRE(evaluated.known);
    REQUIRE(evaluated.intervals.size() == 1);
    CHECK(evaluated.intervals[0].begin == 136);
    CHECK(evaluated.intervals[0].end == 140);
    const auto aligned_binding = Evaluate(recipe, {1, 1, 1}, {1, 1, 1}, 0x1000, 4096);
    REQUIRE(aligned_binding.known);
    CHECK(aligned_binding.intervals[0].begin != evaluated.intervals[0].begin);
}

TEST_CASE("Storage base snapshot requires the full backend UBO vector",
          "[storage_footprint]") {
    const std::array<u8, 48> submitted{};
    CHECK(VideoCommon::ReadStorageBaseSnapshot(submitted, 32) == 0);
    CHECK(VideoCommon::ReadStorageBaseSnapshot(submitted, 44) == 0);
    CHECK_FALSE(VideoCommon::ReadStorageBaseSnapshot(submitted, 48));
    CHECK_FALSE(VideoCommon::ReadStorageBaseSnapshot(submitted, 33));
    CHECK_FALSE(VideoCommon::ReadStorageBaseSnapshot(
        std::span<const u8>{submitted}.first(36), 32));
    CHECK_FALSE(VideoCommon::ReadStorageBaseSnapshot(submitted, ~u32{0}));
}

TEST_CASE("Storage base streaming rejects stale backing and writable aliases",
          "[storage_footprint]") {
    using VideoCommon::StorageBaseRangesOverlap;
    using VideoCommon::StorageBaseSnapshotEligibility;
    CHECK(StorageBaseSnapshotEligibility(32, 256, true, false, false) == Reason::None);
    // Dirtiness anywhere in the copied UBO disallows streaming, even outside the word at 32.
    const bool dirty_tail = StorageBaseRangesOverlap(0x1000, 256, 0x10f0, 4);
    REQUIRE(dirty_tail);
    CHECK(StorageBaseSnapshotEligibility(32, 256, true, dirty_tail, false) ==
          Reason::BaseGpuModified);
    const bool adjacent_same_page = StorageBaseRangesOverlap(0x1000, 256, 0x1100, 16);
    CHECK_FALSE(adjacent_same_page);
    CHECK(StorageBaseSnapshotEligibility(32, 256, true, false, adjacent_same_page) == Reason::None);
    const bool aliases_ubo = StorageBaseRangesOverlap(0x1000, 256, 0x1080, 16);
    REQUIRE(aliases_ubo);
    CHECK(StorageBaseSnapshotEligibility(32, 256, true, false, aliases_ubo) ==
          Reason::BaseWritableAlias);
    CHECK(StorageBaseSnapshotEligibility(32, 256, false, false, false) == Reason::BaseSnapshotBounds);
    CHECK(StorageBaseSnapshotEligibility(32, 36, true, false, false) == Reason::BaseSnapshotBounds);
    CHECK(StorageBaseSnapshotEligibility(32, 65536, true, false, false) == Reason::None);
    CHECK(StorageBaseSnapshotEligibility(32, 65552, true, false, false) == Reason::BaseSnapshotLimit);
    CHECK(Shader::StorageWriteReasonName(Reason::BaseGpuModified) ==
          "base_required_gpu_modified");
    CHECK(Shader::StorageWriteReasonName(Reason::BaseWritableAlias) ==
          "base_required_writable_alias");
}

TEST_CASE("Storage base alias checks handle contained and high address ranges",
          "[storage_footprint]") {
    using VideoCommon::StorageBaseRangesOverlap;
    constexpr u64 high = (std::numeric_limits<u64>::max)() - 4095;
    CHECK(StorageBaseRangesOverlap(0x1000, 256, 0x1080, 4));
    CHECK(StorageBaseRangesOverlap(0x1080, 4, 0x1000, 256));
    CHECK_FALSE(StorageBaseRangesOverlap(0x1000, 0, 0x1000, 256));
    CHECK_FALSE(StorageBaseRangesOverlap(0x1000, 256, 0x1000, 0));
    CHECK_FALSE(StorageBaseRangesOverlap(high, 4096, 0x1000, 256));
    CHECK(StorageBaseRangesOverlap(high, 4096, high + 4092, 4));
}

TEST_CASE("Storage footprint coalesces Kokiri records without bridging their large holes",
          "[storage_footprint]") {
    std::vector<Shader::Optimization::StorageWriteInterval> intervals;
    for (u64 i = 0; i < 64; ++i) {
        for (const auto& range : std::array<std::pair<u64, u64>, 4>{
                 {{128, 140}, {144, 156}, {160, 172}, {244, 248}}}) {
            intervals.push_back({256 * i + range.first, 256 * i + range.second});
        }
    }
    const auto exact = intervals;
    REQUIRE(Shader::Optimization::CoalesceStorageWriteIntervals(intervals));
    REQUIRE(intervals.size() == 64);
    CHECK(intervals.front().begin == 128);
    CHECK(intervals.front().end == 248);
    CHECK(intervals.back().begin == 63 * 256 + 128);
    CHECK(intervals.back().end == 63 * 256 + 248);
    for (const auto& range : exact) {
        CHECK(std::ranges::any_of(intervals, [&](const auto& envelope) {
            return envelope.begin <= range.begin && envelope.end >= range.end;
        }));
    }
    CHECK(Shader::Optimization::ValidStorageWriteIntervals(intervals, 16384));
}

TEST_CASE("Storage footprint fragmentation limit rejects sparse large sets",
          "[storage_footprint]") {
    std::vector<Shader::Optimization::StorageWriteInterval> intervals;
    for (u64 i = 0; i < 65; ++i) intervals.push_back({i * 256, i * 256 + 4});
    CHECK_FALSE(Shader::Optimization::CoalesceStorageWriteIntervals(intervals));
    REQUIRE(intervals.size() == 65);
    intervals.pop_back();
    CHECK(Shader::Optimization::CoalesceStorageWriteIntervals(intervals));
    intervals = {{0, 4}, {132, 136}};
    CHECK(Shader::Optimization::CoalesceStorageWriteIntervals(intervals));
    CHECK(intervals.size() == 1); // Gap exactly 128.
    intervals = {{0, 4}, {136, 140}};
    CHECK(Shader::Optimization::CoalesceStorageWriteIntervals(intervals));
    CHECK(intervals.size() == 2);
    intervals.clear();
    CHECK_FALSE(Shader::Optimization::CoalesceStorageWriteIntervals(intervals));
}

TEST_CASE("Download envelopes preserve CPU holes including stores during the wait",
          "[storage_footprint]") {
    using namespace VideoCommon;
    std::vector<BufferDownloadFragment> fragments{
        {{1}, {144, 0, 12}}, {{1}, {128, 0, 12}}, {{1}, {244, 0, 4}}};
    const auto plan = PlanDownloadCopies(fragments);
    REQUIRE(plan.batches.size() == 1);
    REQUIRE(plan.batches[0].copies.size() == 1);
    const auto& transfer = plan.batches[0].copies[0];
    CHECK(transfer.src_offset == 128);
    CHECK(transfer.size == 120);
    CHECK(plan.transfer_bytes == 120);
    CHECK(plan.staging_size == 128);
    CHECK(plan.copy_regions == 1);
    std::array<u8, 256> gpu{}, backing{};
    for (size_t i = 0; i < gpu.size(); ++i) gpu[i] = static_cast<u8>(i);
    backing.fill(0xee);
    // A nonzero staging allocation offset must not leak into host span addressing.
    constexpr u64 staging_offset = 96;
    std::vector<u8> staging(staging_offset + plan.staging_size);
    std::memcpy(staging.data() + staging_offset + transfer.dst_offset,
                gpu.data() + transfer.src_offset, transfer.size);
    backing[160] = 0xab; // CPU store to a hole while Finish is waiting.
    for (const auto& fragment : fragments) {
        const auto& copy = fragment.copy;
        std::memcpy(backing.data() + copy.src_offset,
                    staging.data() + staging_offset + copy.dst_offset, copy.size);
    }
    for (size_t i = 0; i < backing.size(); ++i) {
        const bool written = (i >= 128 && i < 140) || (i >= 144 && i < 156) ||
                             (i >= 244 && i < 248);
        CHECK(backing[i] == (written ? gpu[i] : i == 160 ? 0xab : 0xee));
    }
}

TEST_CASE("Download envelopes isolate buffers and large gaps with aligned staging offsets",
          "[storage_footprint]") {
    using namespace VideoCommon;
    std::vector<BufferDownloadFragment> fragments{
        {{2}, {0, 0, 4}}, {{1}, {8204, 0, 4}}, {{1}, {4100, 0, 4}}, {{1}, {0, 0, 4}}};
    const auto plan = PlanDownloadCopies(fragments);
    REQUIRE(plan.batches.size() == 2);
    REQUIRE(plan.batches[0].copies.size() == 2);
    REQUIRE(plan.batches[1].copies.size() == 1);
    CHECK(plan.batches[0].buffer_id == Common::SlotId{1});
    CHECK(plan.batches[1].buffer_id == Common::SlotId{2});
    CHECK(plan.batches[0].copies[0].size == 4104); // Gap exactly 4096 merged.
    CHECK(plan.batches[0].copies[1].src_offset == 8204); // Gap 4100 retained.
    CHECK(plan.batches[0].copies[1].dst_offset == 4160);
    CHECK(plan.batches[1].copies[0].dst_offset == 4224);
    CHECK(plan.staging_size == 4288);
    CHECK(plan.transfer_bytes == 4112);
    CHECK(plan.copy_regions == 3);
    CHECK(fragments[1].copy.dst_offset == 4100);
    CHECK(fragments[2].copy.dst_offset == 4160);
    CHECK(fragments[3].copy.dst_offset == 4224);
    std::vector<BufferDownloadFragment> empty;
    const auto no_copies = PlanDownloadCopies(empty);
    CHECK(no_copies.batches.empty());
    CHECK(no_copies.staging_size == 0);
}
