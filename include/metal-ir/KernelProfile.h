#pragma once

/// KernelProfile — pre-computed facts about each kernel function.
///
/// Instead of every pass scanning instructions for the same patterns
/// (has device writes? has per-thread index? has TG stores? etc.),
/// this analysis computes all facts once. Passes query the profile
/// to decide what transforms to apply.
///
/// This is the declarative counterpart to MetalConstraints:
///   MetalConstraints = "what Metal requires" (rules)
///   KernelProfile    = "what this kernel does" (facts)
///
/// A pass combines both: if Constraints say "TG needs barriers" and
/// Profile says "function has TG stores", then insert barriers.

#include "metal-ir/MetalConstraints.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace metalir {

/// Facts about a single kernel function, computed once.
struct KernelFacts {
  // ── Memory access patterns ──────────────────────────────────────────
  bool hasDeviceStore = false;    // any store to AS::Device
  bool hasDeviceLoad = false;     // any load from AS::Device
  bool hasTGStore = false;        // any store to AS::Threadgroup
  bool hasTGLoad = false;         // any load from AS::Threadgroup
  bool hasAtomics = false;        // any air.atomic.* call

  // ── Index usage ─────────────────────────────────────────────────────
  bool hasPerThreadIndex = false; // uses thread_position / simdlane
  bool hasProgramIndex = false;   // uses threadgroup_position (program_id)

  // ── Type usage ──────────────────────────────────────────────────────
  bool hasNonFloatDeviceLoad = false;  // device load of half/i32/etc
  bool hasNonFloatDeviceStore = false; // device store of half/i32/etc
  bool hasStructPhi = false;           // struct-typed PHI nodes
  bool hasMMA = false;                 // calls simdgroup MMA intrinsics

  // ── Derived properties ──────────────────────────────────────────────

  /// Scalar kernel: writes device memory but has no per-thread indexing.
  /// Needs tid==0 guard around stores.
  bool isScalarKernel() const {
    return (hasDeviceStore || hasAtomics) && !hasPerThreadIndex;
  }

  /// Needs TG barriers: has both TG stores and TG loads.
  bool needsTGBarriers() const {
    return hasTGStore;
  }

  /// Needs device load widening: has non-float device loads (MMA context).
  bool needsDeviceLoadWidening() const {
    return hasMMA && (hasNonFloatDeviceLoad || hasNonFloatDeviceStore);
  }

  /// Needs volatile device loads: has device stores in loops.
  /// (This is a coarse approximation — the actual pass checks loops.)
  bool hasDeviceStoreLoadPattern() const {
    return hasDeviceStore && hasDeviceLoad;
  }
};

/// Module-level analysis: computes KernelFacts for each non-declaration function.
class KernelProfileAnalysis
    : public llvm::AnalysisInfoMixin<KernelProfileAnalysis> {
  friend llvm::AnalysisInfoMixin<KernelProfileAnalysis>;
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::DenseMap<llvm::Function *, KernelFacts>;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

} // namespace metalir
