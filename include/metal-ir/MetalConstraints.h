#pragma once

/// Metal GPU constraints by address space and context.
///
/// Centralizes the rules currently sprinkled across 15+ passes:
/// - Which types are valid per address space
/// - When barriers/volatile/bitcasts are needed
/// - MMA-specific overrides
///
/// Passes query this instead of reimplementing the same checks.

#include "llvm/IR/Type.h"

namespace metalir {

// ── Address space constants ───────────────────────────────────────────────

namespace AS {
  constexpr unsigned Device = 1;
  constexpr unsigned Constant = 2;
  constexpr unsigned Threadgroup = 3;
} // namespace AS

// ── Metal type constraints ────────────────────────────────────────────────

/// Metal has no double, no i1 in memory, no float8.
/// When MMA intrinsics are present, device pointers must all be float*.
struct MetalConstraints {
  bool hasMMA = false;

  /// Can this type be stored/loaded in the given address space?
  bool isValidMemoryType(llvm::Type *T) const {
    if (T->isDoubleTy()) return false;       // no f64
    if (T->isIntegerTy(1)) return false;      // i1 → must use i8
    return true;
  }

  /// What pointee type should a pointer in this AS have?
  /// Returns nullptr if no override needed.
  llvm::Type *requiredPointeeType(unsigned addrSpace,
                                   llvm::LLVMContext &Ctx) const {
    if (hasMMA && addrSpace == AS::Device)
      return llvm::Type::getFloatTy(Ctx);     // GPU JIT crashes on half*/i32*
    return nullptr;
  }

  /// Should i1 GEPs be rewritten to i8?
  static constexpr bool remapI1ToI8 = true;

  /// Should bfloat GEPs be lowered to half? (same bit width)
  static constexpr bool remapBFloatToHalf = true;

  /// Maximum threadgroup memory in bytes.
  static constexpr unsigned maxTGBytes = 32 * 1024;

  /// Maximum pointer PHIs per basic block before converting to i64.
  static constexpr unsigned maxPtrPhisPerBlock = 32;

  /// Does this address space require explicit barriers between
  /// store and subsequent load across threads?
  static bool needsBarriers(unsigned addrSpace) {
    return addrSpace == AS::Threadgroup;
  }

  /// Should device loads be marked volatile to prevent GPU JIT hoisting
  /// when a store to the same pointer exists in the loop?
  bool needsVolatileDeviceLoads() const {
    return true; // always — GPU JIT hoists non-volatile loads
  }

  /// When MMA present, should all device loads be widened to float?
  bool widenDeviceLoadsToFloat() const {
    return hasMMA;
  }
};

} // namespace metalir
