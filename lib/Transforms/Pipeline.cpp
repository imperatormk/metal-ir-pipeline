// Pipeline builder + TG memory analysis + stub passes.
// Implemented passes live in their own files.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"

using namespace llvm;

namespace metalir {

// ── MMA Presence Analysis ────────────────────────────────────────────────

AnalysisKey MMAPresenceAnalysis::Key;

MMAPresence MMAPresenceAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  MMAPresence result;
  for (auto &F : M) {
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_")) {
      result.hasMMA = true;
      break;
    }
  }
  return result;
}

// ── TG Memory Analysis ──────────────────────────────────────────────────

AnalysisKey TGMemoryAnalysis::Key;

void TGMemoryBudget::addGlobal(StringRef name, unsigned bytes) {
  usedBytes += bytes;
}

bool TGMemoryBudget::fits(unsigned additionalBytes) const {
  return (usedBytes + additionalBytes) <= kMaxBytes;
}

TGMemoryBudget TGMemoryAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  TGMemoryBudget budget;
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == 3)
      budget.addGlobal(GV.getName(),
                        M.getDataLayout().getTypeAllocSize(GV.getValueType()));
  return budget;
}

// ── Pipeline Builder ─────────────────────────────────────────────────────

void buildMetalIRPipeline(ModulePassManager &MPM) {
  int stopAfter = -1;
  if (auto *env = getenv("METALIR_STOP_AFTER"))
    stopAfter = atoi(env);
  int passNum = 0;
  auto maybeAdd = [&](auto pass) {
    if (stopAfter < 0 || passNum < stopAfter)
      MPM.addPass(std::move(pass));
    passNum++;
  };

  // Phase 1: Structural transforms
  maybeAdd(InlineNonKernelFunctionsPass());
  maybeAdd(DecomposeStructPhisPass());
  maybeAdd(PtrPhiToI64Pass());

  // Phase 2: Barrier handling
  maybeAdd(BarrierRenamePass());
  maybeAdd(TGBarrierInsertPass());

  // Phase 3: Instruction lowering (independent, any order)
  maybeAdd(NaNMinMaxPass());
  maybeAdd(LowerFNegPass());
  maybeAdd(BitcastZeroInitPass());
  maybeAdd(LLVMToAIRIntrinsicsPass());
  maybeAdd(LowerIntMinMaxPass());
  maybeAdd(SplitI64ShufflePass());
  maybeAdd(LowerAtomicRMWPass());

  // Phase 4: TG memory management (strict order)
  maybeAdd(TGGlobalDeadElimPass());
  maybeAdd(TGGlobalCoalescePass());
  maybeAdd(TGGlobalGEPRewritePass());

  // Phase 5: Type system
  maybeAdd(InferTypedPointersPass());
  maybeAdd(MMATypedPointersPass());

  // Phase 6: Kernel ABI
  maybeAdd(ScalarBufferPackingPass());
  maybeAdd(ScalarStoreGuardPass());
  maybeAdd(AIRSystemValuesPass());

  // Phase 7: Pre-serialization normalization (part 1)
  maybeAdd(NormalizeI1PointersPass());

  // Phase 8: Device memory fixups
  maybeAdd(DeviceLoadsVolatilePass());
  maybeAdd(WidenDeviceLoadsPass());

  // Phase 9: Cast decomposition (after WidenDeviceLoads so trunc→sext folds)
  maybeAdd(BFloat16CastDecomposePass());

  // Phase 10: Pre-serialization normalization (part 2, after widening)
  maybeAdd(NormalizeAllocasPass());
}

// ── Stubs — passes not yet ported ────────────────────────────────────────

#define STUB_PASS(Name)                                                        \
  PreservedAnalyses Name::run(Module &M, ModuleAnalysisManager &AM) {          \
    return PreservedAnalyses::all();                                            \
  }                                                                            \
  bool Name::needsRun(Module &M) { return false; }

// InlineNonKernelFunctionsPass implemented in InlineNonKernel.cpp
// DecomposeStructPhisPass implemented in DecomposeStructPhis.cpp
// PtrPhiToI64Pass implemented in PtrPhiToI64.cpp
// TGBarrierInsertPass implemented in TGBarrierInsert.cpp
// SplitI64ShufflePass implemented in SplitI64Shuffle.cpp
// TGGlobalCoalescePass implemented in TGGlobalCoalesce.cpp
// TGGlobalGEPRewritePass implemented in TGGlobalGEPRewrite.cpp
// MMATypedPointersPass implemented in MMATypedPointers.cpp
// ScalarStoreGuardPass implemented in ScalarStoreGuard.cpp
// DeviceLoadsVolatilePass implemented in DeviceLoadsVolatile.cpp
// WidenDeviceLoadsPass implemented in WidenDeviceLoads.cpp

#undef STUB_PASS

} // namespace metalir
