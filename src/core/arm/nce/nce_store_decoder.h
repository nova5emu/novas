// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstring>
#include <optional>
#include <span>

#include "core/arm/nce/nce_exact_store.h"
#include "core/arm/nce/visitor_base.h"

namespace Core::NceExactStore {

// Preparation only: no memory accesses or register updates. All other visitor entries reject.
class StoreVisitor final : public VisitorBase {
public:
    StoreVisitor(std::span<const u64, 31> regs_, u64 sp_, std::span<const u128, 32> vectors_,
                 u64 dczid_el0_ = 0x10)
        : regs{regs_}, sp{sp_}, vectors{vectors_}, dczid_el0{dczid_el0_} {}

    Result Decode(u32 instruction, u64 fault_address, Store& out) {
        const auto decoded = Dynarmic::A64::Decode<VisitorBase, bool>(*this, instruction);
        if (!decoded || !*decoded) {
            return failure;
        }
        constexpr u64 page_mask = 0xfff;
        if ((store.address & page_mask) + store.size > page_mask + 1) {
            return Result::PageCrossing;
        }
        // Tagged addresses need separate architectural handling.
        if (store.address >> 48 || fault_address < store.address ||
            fault_address - store.address >= store.size) {
            return Result::AddressMismatch;
        }
        // The buffer cache checks alignment and preserves the required store atomicity.
        out = store;
        return Result::Success;
    }

    bool DC_ZVA(Reg Rt) override {
        if ((dczid_el0 & 0x10) != 0) {
            failure = Result::DcZvaProhibited;
            return false;
        }
        const u32 size = 4U << (dczid_el0 & 0xf);
        if (size > DcZvaZeros.size()) {
            failure = Result::DcZvaBlockSize;
            return false;
        }
        store.address = GetReg(Rt) & ~(static_cast<u64>(size) - 1);
        store.size = size;
        store.zero_block = true;
        return true;
    }

    bool STRx_LDRx_imm_1(Imm<2> size, Imm<2> opc, Imm<9> imm9, bool not_postindex, Reg Rn,
                        Reg Rt) override {
        if (opc != 0) {
            return false;
        }
        return Single(size.ZeroExtend<u32>(), imm9.SignExtend<u64>(), !not_postindex, true, Rn,
                      Rt);
    }

    bool STRx_LDRx_imm_2(Imm<2> size, Imm<2> opc, Imm<12> imm12, Reg Rn, Reg Rt) override {
        if (opc != 0) {
            return false;
        }
        const u32 scale = size.ZeroExtend<u32>();
        return Single(scale, imm12.ZeroExtend<u64>() << scale, false, false, Rn, Rt);
    }

    bool STURx_LDURx(Imm<2> size, Imm<2> opc, Imm<9> imm9, Reg Rn, Reg Rt) override {
        if (opc != 0) {
            return false;
        }
        return Single(size.ZeroExtend<u32>(), imm9.SignExtend<u64>(), false, false, Rn, Rt);
    }

    bool STRx_reg(Imm<2> size, Imm<1> opc_1, Reg Rm, Imm<3> option, bool S, Reg Rn,
                  Reg Rt) override {
        if (opc_1 != 0) {
            return false;
        }
        const u32 scale = size.ZeroExtend<u32>();
        u64 offset{};
        return RegisterOffset(Rm, option, S ? scale : 0, offset) &&
               Single(scale, offset, false, false, Rn, Rt);
    }

    bool STP_LDP_gen(Imm<2> opc, bool not_postindex, bool wback, Imm<1> L, Imm<7> imm7,
                     Reg Rt2, Reg Rn, Reg Rt) override {
        if (L != 0) {
            return false;
        }
        if (opc != 0 && opc != 2) {
            return Invalid();
        }
        return Pair(2 + opc.Bit<1>(), imm7, !not_postindex, wback, Rt2, Rn, Rt);
    }

    bool STNP_LDNP_gen(Imm<1> upper_opc, Imm<1> L, Imm<7> imm7, Reg Rt2, Reg Rn,
                       Reg Rt) override {
        if (L != 0) {
            return false;
        }
        return Pair(2 + upper_opc.ZeroExtend<u32>(), imm7, false, false, Rt2, Rn, Rt);
    }

    bool STR_imm_fpsimd_1(Imm<2> size, Imm<1> opc, Imm<9> imm9, bool not_postindex,
                         Reg Rn, Vec Vt) override {
        return SimdSingle((opc.ZeroExtend<u32>() << 2) | size.ZeroExtend<u32>(),
                          imm9.SignExtend<u64>(), !not_postindex, true, Rn, Vt);
    }

    bool STR_imm_fpsimd_2(Imm<2> size, Imm<1> opc, Imm<12> imm12, Reg Rn, Vec Vt) override {
        const u32 scale = (opc.ZeroExtend<u32>() << 2) | size.ZeroExtend<u32>();
        return SimdSingle(scale, imm12.ZeroExtend<u64>() << scale, false, false, Rn, Vt);
    }

    bool STUR_fpsimd(Imm<2> size, Imm<1> opc, Imm<9> imm9, Reg Rn, Vec Vt) override {
        return SimdSingle((opc.ZeroExtend<u32>() << 2) | size.ZeroExtend<u32>(),
                          imm9.SignExtend<u64>(), false, false, Rn, Vt);
    }

    bool STR_reg_fpsimd(Imm<2> size, Imm<1> opc, Reg Rm, Imm<3> option, bool S,
                       Reg Rn, Vec Vt) override {
        const u32 scale = (opc.ZeroExtend<u32>() << 2) | size.ZeroExtend<u32>();
        u64 offset{};
        return RegisterOffset(Rm, option, S ? scale : 0, offset) &&
               SimdSingle(scale, offset, false, false, Rn, Vt);
    }

    bool STP_LDP_fpsimd(Imm<2> opc, bool not_postindex, bool wback, Imm<1> L,
                       Imm<7> imm7, Vec Vt2, Reg Rn, Vec Vt) override {
        return L == 0 && SimdPair(opc, imm7, !not_postindex, wback, Vt2, Rn, Vt);
    }

    bool STNP_LDNP_fpsimd(Imm<2> opc, Imm<1> L, Imm<7> imm7, Vec Vt2, Reg Rn,
                        Vec Vt) override {
        return L == 0 && SimdPair(opc, imm7, false, false, Vt2, Rn, Vt);
    }

    bool ST1_sngl_1(bool Q, Imm<2> opcode, bool S, Imm<2> size, Reg Rn, Vec Vt) override {
        return SimdLane(Q, opcode, S, size, Rn, Vt, std::nullopt);
    }

    bool ST1_sngl_2(bool Q, Reg Rm, Imm<2> opcode, bool S, Imm<2> size,
                    Reg Rn, Vec Vt) override {
        return SimdLane(Q, opcode, S, size, Rn, Vt, Rm);
    }

    bool STx_mult_1(bool Q, Imm<4> opcode, Imm<2> size, Reg Rn, Vec Vt) override {
        return SimdStructurePair(Q, opcode, size, Rn, Vt, std::nullopt);
    }

    bool STx_mult_2(bool Q, Reg Rm, Imm<4> opcode, Imm<2> size, Reg Rn, Vec Vt) override {
        return SimdStructurePair(Q, opcode, size, Rn, Vt, Rm);
    }

private:
    bool SimdLane(bool Q, Imm<2> opcode, bool S, Imm<2> size, Reg Rn, Vec Vt,
                  std::optional<Reg> postindex) {
        u32 scale = opcode.ZeroExtend<u32>();
        u32 index{};
        switch (scale) {
        case 0:
            index = (u32{Q} << 3) | (u32{S} << 2) | size.ZeroExtend<u32>();
            break;
        case 1:
            if (size.Bit<0>()) return Invalid();
            index = (u32{Q} << 2) | (u32{S} << 1) | u32{size.Bit<1>()};
            break;
        case 2:
            if (size.Bit<1>()) return Invalid();
            if (size.Bit<0>()) {
                if (S) return Invalid();
                scale = 3;
                index = Q;
            } else {
                index = (u32{Q} << 1) | u32{S};
            }
            break;
        default:
            return Invalid();
        }
        const u32 element_size = 1U << scale;
        u64 increment{};
        if (postindex) increment = *postindex == Reg::R31 ? element_size : GetReg(*postindex);
        if (!Address(increment, postindex.has_value(), postindex.has_value(),
                     Rn, Reg::R31, Reg::R31, false)) return false;
        store.simd = true;
        store.size = element_size;
        const auto* source = reinterpret_cast<const u8*>(vectors[static_cast<u32>(Vt)].data());
        std::memcpy(store.bytes.data(), source + index * element_size, element_size);
        return true;
    }

    bool SimdStructurePair(bool Q, Imm<4> opcode, Imm<2> size, Reg Rn, Vec Vt,
                           std::optional<Reg> postindex) {
        // ST2 interleaves the elements of two consecutive vector registers.
        if (opcode != 0b1000) return false;
        if (!Q && size == 3) return Invalid();
        const u32 vector_size = Q ? 16 : 8;
        const u32 element_size = 1U << size.ZeroExtend<u32>();
        const u32 total_size = vector_size * 2;
        u64 increment{};
        if (postindex) increment = *postindex == Reg::R31 ? total_size : GetReg(*postindex);
        // A 32-byte payload uses the existing two-16-byte commit shape.
        if (!Address(increment, postindex.has_value(), postindex.has_value(),
                     Rn, Reg::R31, Reg::R31, Q)) return false;
        store.simd = true;
        store.size = total_size;
        const u32 first_vector = static_cast<u32>(Vt);
        const auto* first = reinterpret_cast<const u8*>(vectors[first_vector].data());
        const auto* second = reinterpret_cast<const u8*>(vectors[(first_vector + 1) % 32].data());
        for (u32 offset = 0; offset < vector_size; offset += element_size) {
            std::memcpy(store.bytes.data() + offset * 2, first + offset, element_size);
            std::memcpy(store.bytes.data() + offset * 2 + element_size, second + offset, element_size);
        }
        return true;
    }

    bool Invalid() {
        failure = Result::InvalidOperands;
        return false;
    }

    u64 GetReg(Reg reg) const {
        return reg == Reg::R31 ? 0 : regs[static_cast<u32>(reg)];
    }

    bool RegisterOffset(Reg Rm, Imm<3> option, u32 shift, u64& offset) {
        offset = GetReg(Rm);
        switch (option.ZeroExtend<u32>()) {
        case 0b010: // UXTW
            offset = static_cast<u32>(offset);
            break;
        case 0b011: // LSL
        case 0b111: // SXTX
            break;
        case 0b110: // SXTW
            offset = static_cast<u64>(static_cast<s64>(static_cast<s32>(offset)));
            break;
        default:
            return Invalid();
        }
        offset <<= shift;
        return true;
    }

    bool Address(u64 offset, bool postindex, bool wback, Reg Rn, Reg Rt, Reg Rt2,
                 bool pair) {
        if (wback && Rn != Reg::SP && (Rn == Rt || (pair && Rn == Rt2))) {
            return Invalid();
        }
        if (Rn == Reg::SP && (sp & 15) != 0) {
            return Invalid();
        }
        const u64 base = Rn == Reg::SP ? sp : GetReg(Rn);
        store.address = base + (postindex ? 0 : offset);
        store.writeback_value = base + offset;
        store.base_register = static_cast<u32>(Rn);
        store.writeback = wback;
        store.pair = pair;
        return true;
    }

    bool Single(u32 scale, u64 offset, bool postindex, bool wback, Reg Rn, Reg Rt) {
        if (!Address(offset, postindex, wback, Rn, Rt, Reg::R31, false)) {
            return false;
        }
        store.size = 1U << scale;
        const u64 value = GetReg(Rt);
        std::memcpy(store.bytes.data(), &value, store.size);
        return true;
    }

    bool Pair(u32 scale, Imm<7> imm7, bool postindex, bool wback, Reg Rt2, Reg Rn, Reg Rt) {
        if (!Address(imm7.SignExtend<u64>() << scale, postindex, wback, Rn, Rt, Rt2, true)) {
            return false;
        }
        const u32 element_size = 1U << scale;
        store.size = element_size * 2;
        const u64 first = GetReg(Rt);
        const u64 second = GetReg(Rt2);
        std::memcpy(store.bytes.data(), &first, element_size);
        std::memcpy(store.bytes.data() + element_size, &second, element_size);
        return true;
    }

    bool SimdSingle(u32 scale, u64 offset, bool postindex, bool wback, Reg Rn, Vec Vt) {
        if (scale > 4) {
            return Invalid();
        }
        // SIMD sources do not alias the general-purpose writeback register.
        if (!Address(offset, postindex, wback, Rn, Reg::R31, Reg::R31, false)) {
            return false;
        }
        store.simd = true;
        store.size = 1U << scale;
        std::memcpy(store.bytes.data(), vectors[static_cast<u32>(Vt)].data(), store.size);
        return true;
    }

    bool SimdPair(Imm<2> opc, Imm<7> imm7, bool postindex, bool wback,
                  Vec Vt2, Reg Rn, Vec Vt) {
        if (opc == 3) {
            return Invalid();
        }
        const u32 scale = 2 + opc.ZeroExtend<u32>();
        if (!Address(imm7.SignExtend<u64>() << scale, postindex, wback,
                     Rn, Reg::R31, Reg::R31, true)) {
            return false;
        }
        store.simd = true;
        const u32 element_size = 1U << scale;
        store.size = element_size * 2;
        std::memcpy(store.bytes.data(), vectors[static_cast<u32>(Vt)].data(), element_size);
        std::memcpy(store.bytes.data() + element_size,
                    vectors[static_cast<u32>(Vt2)].data(), element_size);
        return true;
    }

    std::span<const u64, 31> regs;
    u64 sp;
    std::span<const u128, 32> vectors;
    u64 dczid_el0;
    Store store{};
    Result failure{Result::UnsupportedInstruction};
};

} // namespace Core::NceExactStore
