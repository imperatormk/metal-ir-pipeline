#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace metalir {

// ── Pass declarations ────────────────────────────────────────────────────
//
// Each pass has:
//   run()      — do the transform
//   needsRun() — static check: does the module still contain constructs
//                this pass should eliminate? Used as postcondition in
//                debug builds to catch pass misordering.
//
// PreservedAnalyses contract:
//   - Passes that only rename/replace instructions preserve CFGAnalyses
//   - Passes that add/remove BBs or change branches invalidate everything
//   - Stubs preserve all (no-op)

// Pass 0: Metal can't call functions — inline everything.
struct InlineNonKernelFunctionsPass
    : llvm::PassInfoMixin<InlineNonKernelFunctionsPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 1: GPU JIT crashes on struct phi nodes — split to scalar phis.
struct DecomposeStructPhisPass
    : llvm::PassInfoMixin<DecomposeStructPhisPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 2: ~63 ptr phi limit — convert ptr phis to i64.
struct PtrPhiToI64Pass : llvm::PassInfoMixin<PtrPhiToI64Pass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 3: Rename air.threadgroup.barrier → air.wg.barrier.
struct BarrierRenamePass : llvm::PassInfoMixin<BarrierRenamePass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 4: Insert barriers around TG accesses (WAR hazards).
struct TGBarrierInsertPass : llvm::PassInfoMixin<TGBarrierInsertPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 5: llvm.minimum/maximum → air.fmin/fmax + NaN propagation.
struct NaNMinMaxPass : llvm::PassInfoMixin<NaNMinMaxPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 6: fneg → fsub -0.0.
struct LowerFNegPass : llvm::PassInfoMixin<LowerFNegPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 7: bitcast zeroinitializer → zero of dest type.
struct BitcastZeroInitPass : llvm::PassInfoMixin<BitcastZeroInitPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 8: LLVM intrinsics → AIR intrinsics (rename declarations).
struct LLVMToAIRIntrinsicsPass
    : llvm::PassInfoMixin<LLVMToAIRIntrinsicsPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 9: llvm.smin/smax/umin/umax → icmp + select.
struct LowerIntMinMaxPass : llvm::PassInfoMixin<LowerIntMinMaxPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 10: i64 simd_shuffle → 2×i32 shuffles.
struct SplitI64ShufflePass : llvm::PassInfoMixin<SplitI64ShufflePass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 11: atomicrmw → air.atomic.* intrinsic calls.
struct LowerAtomicRMWPass : llvm::PassInfoMixin<LowerAtomicRMWPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 12: Remove unreferenced addrspace(3) globals.
struct TGGlobalDeadElimPass : llvm::PassInfoMixin<TGGlobalDeadElimPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 13: Merge __tg_cvt_* into __tg_dot_* (non-overlapping lifetimes).
struct TGGlobalCoalescePass : llvm::PassInfoMixin<TGGlobalCoalescePass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 14: Byte→float TG globals, flatten GEPs, split regions.
struct TGGlobalGEPRewritePass : llvm::PassInfoMixin<TGGlobalGEPRewritePass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 15: Opaque → typed pointers (Metal GPU JIT requires typed).
struct InferTypedPointersPass : llvm::PassInfoMixin<InferTypedPointersPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 17: Decompose bf16 casts via float intermediate.
struct BFloat16CastDecomposePass
    : llvm::PassInfoMixin<BFloat16CastDecomposePass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 5b: Scalar params → ptr addrspace(2) constant buffer + load.
struct ScalarBufferPackingPass
    : llvm::PassInfoMixin<ScalarBufferPackingPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 18: Guard scalar stores with tid==0.
struct ScalarStoreGuardPass : llvm::PassInfoMixin<ScalarStoreGuardPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 19: System value calls → kernel params + !air.kernel metadata.
struct AIRSystemValuesPass : llvm::PassInfoMixin<AIRSystemValuesPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 23: i1 GEPs → i8 GEPs (Metal has no i1 memory type).
struct NormalizeI1PointersPass
    : llvm::PassInfoMixin<NormalizeI1PointersPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 22: Alloca i64 sizes → i32 (Metal v1 bitcode requirement).
struct NormalizeAllocasPass : llvm::PassInfoMixin<NormalizeAllocasPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 20: Mark device loads volatile to prevent GPU JIT hoisting.
struct DeviceLoadsVolatilePass
    : llvm::PassInfoMixin<DeviceLoadsVolatilePass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pass 21: Widen all device loads to float when MMA present.
struct WidenDeviceLoadsPass : llvm::PassInfoMixin<WidenDeviceLoadsPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// ── Analysis: MMA Presence ────────────────────────────────────────────────
// Cached answer to "does this module have simdgroup_matrix_8x8 intrinsics?"
// Queried by InferTypedPointers, WidenDeviceLoads,
// AIRSystemValues (scalar packing), and BitcodeWriter.

struct MMAPresence {
  bool hasMMA = false;
};

struct MMAPresenceAnalysis : llvm::AnalysisInfoMixin<MMAPresenceAnalysis> {
  using Result = MMAPresence;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static llvm::AnalysisKey Key;
};

// ── Analysis: TG Memory Budget ───────────────────────────────────────────

struct TGMemoryBudget {
  static constexpr unsigned kMaxBytes = 32 * 1024;
  unsigned usedBytes = 0;
  void addGlobal(llvm::StringRef name, unsigned bytes);
  bool fits(unsigned additionalBytes) const;
};

struct TGMemoryAnalysis : llvm::AnalysisInfoMixin<TGMemoryAnalysis> {
  using Result = TGMemoryBudget;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static llvm::AnalysisKey Key;
};

// ── Pipeline builder ─────────────────────────────────────────────────────

void buildMetalIRPipeline(llvm::ModulePassManager &MPM);

} // namespace metalir
