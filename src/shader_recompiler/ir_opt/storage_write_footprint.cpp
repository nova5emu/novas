// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/ir_opt/storage_write_footprint.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>

#include "shader_recompiler/frontend/ir/program.h"

namespace Shader {
std::string_view StorageWriteReasonName(StorageWriteReason reason) {
    switch (reason) {
    case StorageWriteReason::NotCollected: return "not_collected";
    case StorageWriteReason::None: return "none";
    case StorageWriteReason::NoWrites: return "no_writes";
    case StorageWriteReason::NotCompute: return "not_compute";
    case StorageWriteReason::NativeGlobalBounds: return "native_global_bounds";
    case StorageWriteReason::IndirectDescriptor: return "indirect_descriptor";
    case StorageWriteReason::UnresolvedGlobal: return "unresolved_global";
    case StorageWriteReason::DynamicBinding: return "dynamic_binding";
    case StorageWriteReason::UnsupportedStore: return "unsupported_store";
    case StorageWriteReason::DynamicOrUnsupportedOffset: return "dynamic_or_unsupported_offset";
    case StorageWriteReason::Overflow: return "overflow";
    case StorageWriteReason::Budget: return "budget";
    case StorageWriteReason::NeedsDispatch: return "needs_dispatch";
    case StorageWriteReason::BaseRequired: return "base_required";
    case StorageWriteReason::IndirectDispatch: return "indirect_dispatch";
    case StorageWriteReason::InvalidDimensions: return "invalid_dimensions";
    case StorageWriteReason::InvocationLimit: return "invocation_limit";
    case StorageWriteReason::IntervalLimit: return "interval_limit";
    case StorageWriteReason::OutsideBinding: return "outside_binding";
    case StorageWriteReason::BaseGpuModified: return "base_required_gpu_modified";
    case StorageWriteReason::BaseWritableAlias: return "base_required_writable_alias";
    case StorageWriteReason::BaseSnapshotBounds: return "base_required_snapshot_bounds";
    case StorageWriteReason::BaseSnapshotLimit: return "base_required_snapshot_limit";
    case StorageWriteReason::FragmentationLimit: return "fragmentation_limit";
    }
    return "invalid_reason";
}
} // namespace Shader

namespace Shader::Optimization {
namespace {
struct Bounds {
    u32 low;
    u32 high;
};

IR::Value Unwrap(IR::Value value) {
    for (unsigned i = 0; i < 32 && value.IsIdentity(); ++i) {
        value = value.Inst()->Arg(0);
    }
    return value;
}

bool SameBase(IR::Value lhs, IR::Value rhs) {
    lhs = Unwrap(lhs);
    rhs = Unwrap(rhs);
    // Only the same SSA value: identical addresses do not prove two memory loads equal.
    return lhs == rhs;
}

std::optional<Bounds> Evaluate(IR::Value value, unsigned& budget, StorageWriteReason& reason) {
    if (budget == 0) {
        reason = StorageWriteReason::Budget;
        return std::nullopt;
    }
    --budget;
    value = Unwrap(value);
    if (value.Type() != IR::Type::U32) return std::nullopt;
    if (value.IsImmediate()) return Bounds{value.U32(), value.U32()};
    const auto* inst = value.Inst();
    const auto op = inst->GetOpcode();
    if (op != IR::Opcode::IAdd32 && op != IR::Opcode::ISub32 &&
        op != IR::Opcode::IMul32 && op != IR::Opcode::BitwiseAnd32 &&
        op != IR::Opcode::BitwiseOr32 && op != IR::Opcode::BitwiseXor32)
        return std::nullopt;
    const auto lhs = Unwrap(inst->Arg(0));
    const auto rhs = Unwrap(inst->Arg(1));
    if (op == IR::Opcode::ISub32 && !rhs.IsImmediate() && !rhs.IsEmpty()) {
        const auto* aligned = rhs.Inst();
        if (aligned->GetOpcode() == IR::Opcode::BitwiseAnd32) {
            auto base = Unwrap(aligned->Arg(0));
            auto mask = Unwrap(aligned->Arg(1));
            if (base.IsImmediate()) std::swap(base, mask);
            if (mask.IsImmediate() && mask.Type() == IR::Type::U32) {
                const u32 low_mask = ~mask.U32();
                auto source = lhs;
                u32 constant = 0;
                bool matches = SameBase(source, base);
                if (!matches && !source.IsImmediate() && !source.IsEmpty() &&
                    source.Inst()->GetOpcode() == IR::Opcode::IAdd32) {
                    auto a = Unwrap(source.Inst()->Arg(0));
                    auto b = Unwrap(source.Inst()->Arg(1));
                    if (a.IsImmediate()) std::swap(a, b);
                    matches = b.IsImmediate() && b.Type() == IR::Type::U32 && SameBase(a, base);
                    if (matches) constant = b.U32();
                }
                // Modular cancellation: (base+c)-(base&mask) == c+(base&~mask).
                if (matches && (low_mask & (low_mask + 1U)) == 0) {
                    if (u64{constant} + low_mask > std::numeric_limits<u32>::max()) {
                        reason = StorageWriteReason::Overflow;
                        return std::nullopt;
                    }
                    return Bounds{constant, constant + low_mask};
                }
            }
        }
    }
    const auto a = Evaluate(lhs, budget, reason);
    if (!a) return std::nullopt;
    const auto b = Evaluate(rhs, budget, reason);
    if (!b) return std::nullopt;
    if (a->low == a->high && b->low == b->high) {
        u32 result{};
        switch (op) {
        case IR::Opcode::IAdd32: result = a->low + b->low; break;
        case IR::Opcode::ISub32: result = a->low - b->low; break;
        case IR::Opcode::IMul32: result = a->low * b->low; break;
        case IR::Opcode::BitwiseAnd32: result = a->low & b->low; break;
        case IR::Opcode::BitwiseOr32: result = a->low | b->low; break;
        case IR::Opcode::BitwiseXor32: result = a->low ^ b->low; break;
        default: return std::nullopt;
        }
        return Bounds{result, result};
    }
    if (op == IR::Opcode::IAdd32) {
        if (u64{a->high} + b->high <= std::numeric_limits<u32>::max())
            return Bounds{a->low + b->low, a->high + b->high};
        reason = StorageWriteReason::Overflow;
    } else if (op == IR::Opcode::ISub32) {
        if (a->low >= b->high)
            return Bounds{a->low - b->high, a->high - b->low};
        reason = StorageWriteReason::Overflow;
    }
    return std::nullopt;
}

bool DescriptorBase(IR::Value value, const StorageBufferDescriptor& descriptor) {
    value = Unwrap(value);
    if (value.IsImmediate() || value.IsEmpty()) return false;
    const auto* inst = value.Inst();
    bool vector_low{};
    if (inst->GetOpcode() == IR::Opcode::CompositeExtractU32x2) {
        const auto lane = Unwrap(inst->Arg(1));
        const auto vector = Unwrap(inst->Arg(0));
        if (!lane.IsImmediate() || lane.Type() != IR::Type::U32 || lane.U32() != 0 ||
            vector.IsImmediate() || vector.IsEmpty()) return false;
        inst = vector.Inst();
        vector_low = true;
    }
    if (inst->GetOpcode() != (vector_low ? IR::Opcode::GetCbufU32x2 : IR::Opcode::GetCbufU32))
        return false;
    const auto bank = Unwrap(inst->Arg(0));
    const auto offset = Unwrap(inst->Arg(1));
    return bank.IsImmediate() && bank.Type() == IR::Type::U32 &&
           offset.IsImmediate() && offset.Type() == IR::Type::U32 &&
           bank.U32() == descriptor.cbuf_index && offset.U32() == descriptor.cbuf_offset &&
           (offset.U32() % (vector_low ? 8 : 4)) == 0;
}

struct Affine {
    std::array<u32, 6> coefficients{};
    u32 constant{}, base{};
    bool reads_base{};
    bool IsConstant() const {
        return base == 0 && std::ranges::all_of(coefficients, [](u32 value) { return value == 0; });
    }
};

std::optional<Affine> ParseAffine(IR::Value value, const StorageBufferDescriptor& descriptor,
                                unsigned& budget) {
    if (budget == 0) return std::nullopt;
    --budget;
    value = Unwrap(value);
    if (value.Type() != IR::Type::U32) return std::nullopt;
    if (value.IsImmediate()) return Affine{.constant = value.U32()};
    if (DescriptorBase(value, descriptor)) return Affine{.base = 1, .reads_base = true};
    const auto* inst = value.Inst();
    const auto op = inst->GetOpcode();
    if (op == IR::Opcode::CompositeExtractU32x3) {
        const auto vector = Unwrap(inst->Arg(0));
        const auto lane = Unwrap(inst->Arg(1));
        if (vector.IsImmediate() || vector.IsEmpty() || !lane.IsImmediate() ||
            lane.Type() != IR::Type::U32 || lane.U32() >= 3) return std::nullopt;
        const auto source = vector.Inst()->GetOpcode();
        if (source != IR::Opcode::WorkgroupId && source != IR::Opcode::LocalInvocationId)
            return std::nullopt;
        Affine result;
        result.coefficients[(source == IR::Opcode::WorkgroupId ? 0 : 3) + lane.U32()] = 1;
        return result;
    }
    if (op != IR::Opcode::IAdd32 && op != IR::Opcode::ISub32 && op != IR::Opcode::IMul32 &&
        op != IR::Opcode::ShiftLeftLogical32) return std::nullopt;
    auto lhs = ParseAffine(inst->Arg(0), descriptor, budget);
    if (!lhs) return std::nullopt;
    auto rhs = ParseAffine(inst->Arg(1), descriptor, budget);
    if (!rhs) return std::nullopt;
    const bool reads_base = lhs->reads_base || rhs->reads_base;
    if (op == IR::Opcode::IAdd32 || op == IR::Opcode::ISub32) {
        const u32 sign = op == IR::Opcode::IAdd32 ? 1U : ~0U;
        lhs->constant += sign * rhs->constant;
        lhs->base += sign * rhs->base;
        lhs->reads_base = reads_base;
        for (size_t i = 0; i < 6; ++i) lhs->coefficients[i] += sign * rhs->coefficients[i];
        return lhs;
    }
    if (op == IR::Opcode::IMul32 && lhs->IsConstant()) std::swap(lhs, rhs);
    if (!rhs->IsConstant()) return std::nullopt;
    if (op == IR::Opcode::ShiftLeftLogical32 && rhs->constant >= 32) return std::nullopt;
    const u32 multiplier = op == IR::Opcode::ShiftLeftLogical32 ? (1U << rhs->constant) : rhs->constant;
    lhs->constant *= multiplier;
    lhs->base *= multiplier;
    lhs->reads_base = reads_base;
    for (auto& coefficient : lhs->coefficients) coefficient *= multiplier;
    return lhs;
}

std::optional<StorageWriteTerm> ParseTerm(IR::Value offset, u32 width,
                                        const StorageBufferDescriptor& descriptor) {
    offset = Unwrap(offset);
    unsigned budget = 256;
    if (!offset.IsImmediate() && !offset.IsEmpty() &&
        offset.Inst()->GetOpcode() == IR::Opcode::ISub32) {
        const auto rhs = Unwrap(offset.Inst()->Arg(1));
        if (!rhs.IsImmediate() && !rhs.IsEmpty() && rhs.Inst()->GetOpcode() == IR::Opcode::BitwiseAnd32) {
            auto base = Unwrap(rhs.Inst()->Arg(0));
            auto mask = Unwrap(rhs.Inst()->Arg(1));
            if (base.IsImmediate()) std::swap(base, mask);
            if (mask.IsImmediate() && mask.Type() == IR::Type::U32 && DescriptorBase(base, descriptor)) {
                const u32 low_mask = ~mask.U32();
                const auto affine = ParseAffine(offset.Inst()->Arg(0), descriptor, budget);
                if (affine && affine->base == 1 && (low_mask & (low_mask + 1U)) == 0) {
                    return StorageWriteTerm{affine->coefficients, affine->constant, width, mask.U32(), true};
                }
            }
        }
    }
    budget = 256;
    const auto affine = ParseAffine(offset, descriptor, budget);
    if (!affine || affine->base != 0) return std::nullopt;
    return StorageWriteTerm{affine->coefficients, affine->constant, width, ~0U, affine->reads_base};
}

u32 WriteWidth(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::WriteStorageU8: return 1;
    case IR::Opcode::WriteStorageS8: return 1;
    case IR::Opcode::WriteStorageU16: return 2;
    case IR::Opcode::WriteStorageS16: return 2;
    case IR::Opcode::WriteStorage32: return 4;
    case IR::Opcode::WriteStorage64: return 8;
    case IR::Opcode::WriteStorage128: return 16;
    case IR::Opcode::StorageAtomicIAdd32: return 4;
    case IR::Opcode::StorageAtomicSMin32: return 4;
    case IR::Opcode::StorageAtomicUMin32: return 4;
    case IR::Opcode::StorageAtomicSMax32: return 4;
    case IR::Opcode::StorageAtomicUMax32: return 4;
    case IR::Opcode::StorageAtomicInc32: return 4;
    case IR::Opcode::StorageAtomicDec32: return 4;
    case IR::Opcode::StorageAtomicAnd32: return 4;
    case IR::Opcode::StorageAtomicOr32: return 4;
    case IR::Opcode::StorageAtomicXor32: return 4;
    case IR::Opcode::StorageAtomicExchange32: return 4;
    case IR::Opcode::StorageAtomicCompareExchange32: return 4;
    case IR::Opcode::StorageAtomicIAdd64: return 8;
    case IR::Opcode::StorageAtomicSMin64: return 8;
    case IR::Opcode::StorageAtomicUMin64: return 8;
    case IR::Opcode::StorageAtomicSMax64: return 8;
    case IR::Opcode::StorageAtomicUMax64: return 8;
    case IR::Opcode::StorageAtomicAnd64: return 8;
    case IR::Opcode::StorageAtomicOr64: return 8;
    case IR::Opcode::StorageAtomicXor64: return 8;
    case IR::Opcode::StorageAtomicExchange64: return 8;
    case IR::Opcode::StorageAtomicIAdd32x2: return 8;
    case IR::Opcode::StorageAtomicSMin32x2: return 8;
    case IR::Opcode::StorageAtomicUMin32x2: return 8;
    case IR::Opcode::StorageAtomicSMax32x2: return 8;
    case IR::Opcode::StorageAtomicUMax32x2: return 8;
    case IR::Opcode::StorageAtomicAnd32x2: return 8;
    case IR::Opcode::StorageAtomicOr32x2: return 8;
    case IR::Opcode::StorageAtomicXor32x2: return 8;
    case IR::Opcode::StorageAtomicExchange32x2: return 8;
    case IR::Opcode::StorageAtomicAddF32: return 4;
    case IR::Opcode::StorageAtomicAddF16x2: return 4;
    case IR::Opcode::StorageAtomicAddF32x2: return 8;
    case IR::Opcode::StorageAtomicMinF16x2: return 4;
    case IR::Opcode::StorageAtomicMinF32x2: return 8;
    case IR::Opcode::StorageAtomicMaxF16x2: return 4;
    case IR::Opcode::StorageAtomicMaxF32x2: return 8;
    default: return 0;
    }
}
} // namespace

void CollectStorageWriteFootprints(IR::Program& program) {
    auto& footprints = program.info.storage_write_footprints;
    footprints.clear();
    std::array<bool, Info::MAX_SSBOS> unknown{};
    std::array<bool, Info::MAX_SSBOS> incomplete_terms{};
    const size_t count = program.info.storage_buffers_descriptors.size();
    footprints.resize(count);
    const auto reject_all = [&](StorageWriteReason reason) {
        for (auto& footprint : footprints) {
            footprint = {};
            footprint.reason = reason;
        }
    };
    if (program.stage != Stage::Compute) {
        reject_all(StorageWriteReason::NotCompute);
        return;
    }
    if (program.info.native_global_bounds) {
        reject_all(StorageWriteReason::NativeGlobalBounds);
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        unknown[i] = program.info.storage_buffers_descriptors[i].indirect_offset != ~0U;
        incomplete_terms[i] = unknown[i];
        footprints[i].reason = unknown[i] ? StorageWriteReason::IndirectDescriptor
                                          : StorageWriteReason::NoWrites;
        footprints[i].coherence_eligible = !unknown[i];
    }
    for (const auto* block : program.post_order_blocks) {
        for (const auto& inst : block->Instructions()) {
            const auto name = IR::NameOf(inst.GetOpcode());
            if (name.starts_with("WriteGlobal") || name.starts_with("GlobalAtomic")) {
                reject_all(StorageWriteReason::UnresolvedGlobal);
                return;
            }
            if (!name.starts_with("WriteStorage") && !name.starts_with("StorageAtomic")) continue;
            const u32 width = WriteWidth(inst.GetOpcode());
            if (width == 0) {
                reject_all(StorageWriteReason::UnsupportedStore);
                return;
            }
            const auto binding = Unwrap(inst.Arg(0));
            if (!binding.IsImmediate() || binding.Type() != IR::Type::U32 || binding.U32() >= count) {
                reject_all(StorageWriteReason::DynamicBinding);
                return;
            }
            const size_t index = binding.U32();
            if (program.info.storage_buffers_descriptors[index].indirect_offset != ~0U) continue;
            auto& footprint = footprints[index];
            footprint.coherence_eligible &= inst.GetOpcode() == IR::Opcode::WriteStorage32 ||
                inst.GetOpcode() == IR::Opcode::WriteStorage64 ||
                inst.GetOpcode() == IR::Opcode::WriteStorage128;
            unsigned budget = 256;
            auto reason = StorageWriteReason::DynamicOrUnsupportedOffset;
            const auto offset = Evaluate(inst.Arg(1), budget, reason);
            if (!incomplete_terms[index]) {
                auto term = ParseTerm(inst.Arg(1), width, program.info.storage_buffers_descriptors[index]);
                if (!term && offset && offset->low == offset->high) {
                    term = StorageWriteTerm{.constant = offset->low, .width = width};
                }
                const bool constant_overflows = offset && offset->low == offset->high &&
                    u64{offset->high} + width > (u64{1} << 32);
                if (!term || constant_overflows || footprint.terms.size() >= 32768) {
                    incomplete_terms[index] = true;
                    footprint.terms.clear();
                } else {
                    footprint.terms.push_back(*term);
                }
            }
            if (unknown[index]) continue;
            if (!offset || u64{offset->high} + width > (u64{1} << 32)) {
                unknown[index] = true;
                footprint.known = false;
                footprint.begin = footprint.end = 0;
                footprint.reason = offset ? StorageWriteReason::Overflow : reason;
                continue;
            }
            // Backends address typed elements by dividing the byte offset; round outward.
            const u64 begin = u64{offset->low} & ~u64{width - 1};
            const u64 end = u64{offset->high} + width;
            footprint.begin = footprint.known ? std::min(footprint.begin, begin) : begin;
            footprint.end = footprint.known ? std::max(footprint.end, end) : end;
            footprint.known = true;
            footprint.reason = StorageWriteReason::None;
        }
    }
    for (auto& footprint : footprints) {
        if (!footprint.known && !footprint.terms.empty()) {
            footprint.symbolic = true;
            footprint.reason = StorageWriteReason::NeedsDispatch;
        }
    }
}

EvaluatedStorageWriteFootprint EvaluateStorageWriteFootprint(
    const StorageWriteFootprint& footprint, const std::array<u32, 3>& grid,
    const std::array<u32, 3>& block, bool indirect, std::optional<u32> base_low, u64 binding_size) {
    const auto fail = [](StorageWriteReason reason) {
        return EvaluatedStorageWriteFootprint{.reason = reason};
    };
    if (!footprint.known && !footprint.symbolic) return fail(footprint.reason);
    if (indirect) return fail(StorageWriteReason::IndirectDispatch);
    const std::array<u32, 6> dimensions{grid[0], grid[1], grid[2], block[0], block[1], block[2]};
    u64 invocations = 1;
    for (u32 dimension : dimensions) {
        if (dimension == 0) return fail(StorageWriteReason::InvalidDimensions);
        if (dimension > 4096 / invocations) return fail(StorageWriteReason::InvocationLimit);
        invocations *= dimension;
    }
    const bool needs_base = std::ranges::any_of(footprint.terms, [](const auto& term) {
        return term.needs_base;
    });
    if (footprint.known && (footprint.terms.empty() || (needs_base && !base_low))) {
        if (footprint.begin >= footprint.end || footprint.end > binding_size)
            return fail(StorageWriteReason::OutsideBinding);
        return {.known = true, .reason = StorageWriteReason::None,
                .intervals = {{footprint.begin, footprint.end}}};
    }
    if (needs_base && !base_low) return fail(StorageWriteReason::BaseRequired);
    if (footprint.terms.empty()) return fail(StorageWriteReason::DynamicOrUnsupportedOffset);
    if (footprint.terms.size() > 32768 / invocations) return fail(StorageWriteReason::IntervalLimit);
    EvaluatedStorageWriteFootprint result{.known = true, .reason = StorageWriteReason::None};
    result.intervals.reserve(static_cast<size_t>(invocations) * footprint.terms.size());
    for (u64 invocation = 0; invocation < invocations; ++invocation) {
        u64 remainder = invocation;
        std::array<u32, 6> ids{};
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = static_cast<u32>(remainder % dimensions[i]);
            remainder /= dimensions[i];
        }
        for (const auto& term : footprint.terms) {
            if (term.width != 1 && term.width != 2 && term.width != 4 &&
                term.width != 8 && term.width != 16)
                return fail(StorageWriteReason::UnsupportedStore);
            u32 offset = term.constant;
            for (size_t i = 0; i < ids.size(); ++i) offset += term.coefficients[i] * ids[i];
            if (term.needs_base) offset += *base_low & ~term.base_mask;
            const u64 begin = u64{offset} & ~u64{term.width - 1};
            const u64 end = u64{offset} + term.width;
            if (end > (u64{1} << 32)) return fail(StorageWriteReason::Overflow);
            if (end > binding_size) return fail(StorageWriteReason::OutsideBinding);
            result.intervals.push_back({begin, end});
        }
    }
    std::ranges::sort(result.intervals, [](const auto& lhs, const auto& rhs) {
        return lhs.begin != rhs.begin ? lhs.begin < rhs.begin : lhs.end < rhs.end;
    });
    size_t output{};
    for (const auto interval : result.intervals) {
        if (output != 0 && interval.begin <= result.intervals[output - 1].end) {
            result.intervals[output - 1].end = std::max(result.intervals[output - 1].end, interval.end);
        } else {
            result.intervals[output++] = interval;
        }
    }
    result.intervals.resize(output);
    return result;
}

bool StorageWriteFootprintsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_WRITE_FOOTPRINT");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

bool ValidStorageWriteIntervals(std::span<const StorageWriteInterval> intervals, u64 binding_size) {
    if (intervals.empty()) return false;
    u64 previous_end{};
    for (const auto& interval : intervals) {
        if (interval.begin < previous_end || interval.begin >= interval.end ||
            interval.end > binding_size) return false;
        previous_end = interval.end;
    }
    return true;
}

bool StorageWriteFootprintProfilingEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_WRITE_FOOTPRINT_PROFILE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled || StorageWriteFootprintsEnabled();
}

bool CoalesceStorageWriteIntervals(std::vector<StorageWriteInterval>& intervals,
                                   u64 max_gap, size_t max_intervals) {
    size_t output{};
    for (const auto interval : intervals) {
        if (output && (interval.begin <= intervals[output - 1].end ||
                       interval.begin - intervals[output - 1].end <= max_gap)) {
            intervals[output - 1].end = (std::max)(intervals[output - 1].end, interval.end);
        } else {
            intervals[output++] = interval;
        }
    }
    intervals.resize(output);
    return !intervals.empty() && intervals.size() <= max_intervals;
}

bool CanApplyStorageWriteFootprint(const StorageWriteFootprint& recipe,
                                  const EvaluatedStorageWriteFootprint& evaluated,
                                  u64 binding_address, u64 binding_size, bool split) {
    if (!recipe.coherence_eligible || !evaluated.known || split || binding_address == 0 ||
        (binding_address & 3) != 0 || binding_address >= (u64{1} << 36) ||
        binding_size > (u64{1} << 36) - binding_address ||
        !ValidStorageWriteIntervals(evaluated.intervals, binding_size)) return false;
    // Readback copies must remain dword aligned; never extend them across unproven holes.
    return std::ranges::all_of(evaluated.intervals, [](const auto& interval) {
        return ((interval.begin | interval.end) & 3) == 0;
    });
}

void CollectStorageWriteFootprintsIfEnabled(IR::Program& program) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_WRITE_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    if (enabled || StorageWriteFootprintProfilingEnabled()) CollectStorageWriteFootprints(program);
}
} // namespace Shader::Optimization
