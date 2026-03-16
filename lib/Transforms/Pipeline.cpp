// Pipeline builder + TG memory analysis + stub passes.
// Implemented passes live in their own files.

#include "metal-ir/Pipeline.h"
#include "metal-ir/MetalConstraints.h"
#include "metal-ir/PassUtil.h"

#include "llvm/Support/raw_ostream.h"

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
    if (GV.getAddressSpace() == AS::Threadgroup)
      budget.addGlobal(GV.getName(),
                        M.getDataLayout().getTypeAllocSize(GV.getValueType()));
  return budget;
}

// ── Pipeline Builder ─────────────────────────────────────────────────────

// ── Postcondition wrapper ──────────────────────────────────────────────
// Wraps a pass to verify needsRun() returns false after execution.
// Enabled by METALIR_VERIFY=1 env var. Catches pass misordering bugs.

template <typename PassT>
struct VerifyingPass : PassInfoMixin<VerifyingPass<PassT>> {
  PassT inner;
  const char *passLabel;
  VerifyingPass(PassT p, const char *n) : inner(std::move(p)), passLabel(n) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    auto PA = inner.run(M, AM);
    if (PassT::needsRun(M))
      errs() << "METALIR_VERIFY: " << passLabel
             << " did not eliminate all targets\n";
    return PA;
  }
};

void buildMetalIRPipeline(ModulePassManager &MPM) {
  int stopAfter = -1;
  if (auto *env = getenv("METALIR_STOP_AFTER"))
    stopAfter = atoi(env);
  bool verify = getenv("METALIR_VERIFY") != nullptr;
  int passNum = 0;
  auto maybeAdd = [&](auto pass, const char *name = "") {
    if (stopAfter >= 0 && passNum >= stopAfter) { passNum++; return; }
    if (verify)
      MPM.addPass(VerifyingPass(std::move(pass), name));
    else
      MPM.addPass(std::move(pass));
    passNum++;
  };

  // Phase 1: Structural transforms
  maybeAdd(InlineNonKernelFunctionsPass(), "InlineNonKernel");
  maybeAdd(DecomposeStructPhisPass(), "DecomposeStructPhis");
  maybeAdd(PtrPhiToI64Pass(), "PtrPhiToI64");

  // Phase 2: Barrier handling
  maybeAdd(BarrierRenamePass(), "BarrierRename");
  maybeAdd(TGBarrierInsertPass(), "TGBarrierInsert");

  // Phase 3: Instruction lowering (independent, any order)
  maybeAdd(NaNMinMaxPass(), "NaNMinMax");
  maybeAdd(LowerFNegPass(), "LowerFNeg");
  maybeAdd(BitcastZeroInitPass(), "BitcastZeroInit");
  maybeAdd(LLVMToAIRIntrinsicsPass(), "LLVMToAIRIntrinsics");
  maybeAdd(LowerIntMinMaxPass(), "LowerIntMinMax");
  maybeAdd(SplitI64ShufflePass(), "SplitI64Shuffle");
  maybeAdd(LowerAtomicRMWPass(), "LowerAtomicRMW");

  // Phase 4: TG memory management (strict order)
  maybeAdd(TGGlobalDeadElimPass(), "TGGlobalDeadElim");
  maybeAdd(TGGlobalCoalescePass(), "TGGlobalCoalesce");
  maybeAdd(TGGlobalGEPRewritePass(), "TGGlobalGEPRewrite");

  // Phase 5: Type system
  maybeAdd(InferTypedPointersPass(), "InferTypedPointers");
  maybeAdd(MMATypedPointersPass(), "MMATypedPointers");

  // Phase 6: Kernel ABI
  maybeAdd(ScalarBufferPackingPass(), "ScalarBufferPacking");
  maybeAdd(ScalarStoreGuardPass(), "ScalarStoreGuard");
  maybeAdd(AIRSystemValuesPass(), "AIRSystemValues");

  // Phase 7: Pre-serialization normalization (part 1)
  maybeAdd(NormalizeI1PointersPass(), "NormalizeI1Pointers");

  // Phase 8: Device memory fixups
  maybeAdd(DeviceLoadsVolatilePass(), "DeviceLoadsVolatile");
  maybeAdd(WidenDeviceLoadsPass(), "WidenDeviceLoads");

  // Phase 9: Cast decomposition (after WidenDeviceLoads so trunc→sext folds)
  maybeAdd(BFloat16CastDecomposePass(), "BFloat16CastDecompose");

  // Phase 10: Pre-serialization normalization (part 2, after widening)
  maybeAdd(NormalizeAllocasPass(), "NormalizeAllocas");
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
