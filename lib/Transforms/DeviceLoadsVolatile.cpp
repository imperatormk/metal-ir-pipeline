// Pass 20: Mark device loads in loops as volatile.
//
// Metal's GPU JIT hoists non-volatile addrspace(1) loads out of loops,
// even when a store to the same pointer exists in the loop body. This
// causes loops with load+store patterns to read stale values.
//
// Fix: find back-edges (loops), collect stored device pointers,
// mark loads from those pointers as volatile.
//
// Uses LLVM's DominatorTree to detect back-edges properly,
// instead of MetalASM's index-based heuristic.

#include "metal-ir/Pipeline.h"
#include "metal-ir/IRUtil.h"
#include "metal-ir/KernelProfile.h"
#include "metal-ir/PassUtil.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool DeviceLoadsVolatilePass::needsRun(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    unsigned bbCount = 0;
    bool hasDeviceLoad = false;
    for (auto &BB : F) {
      bbCount++;
      for (auto &I : BB)
        if (isDeviceLoad(&I) && !cast<LoadInst>(&I)->isVolatile())
          hasDeviceLoad = true;
    }
    if (bbCount > 1 && hasDeviceLoad)
      return true;
  }
  return false;
}

PreservedAnalyses DeviceLoadsVolatilePass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  bool changed = false;
  auto &profiles = MAM.getResult<KernelProfileAnalysis>(M);

  for (auto &F : M) {
    if (F.isDeclaration()) continue;

    // Early exit: no device store+load pattern means no volatile marking needed
    auto it = profiles.find(&F);
    if (it != profiles.end() && !it->second.hasDeviceStoreLoadPattern())
      continue;

    DominatorTree DT(F);
    LoopInfo LI(DT);

    for (auto *L : LI.getLoopsInPreorder()) {
      SmallPtrSet<Value *, 8> storedPtrs;
      for (auto *BB : L->blocks())
        for (auto &I : *BB)
          if (isDeviceStore(&I))
            storedPtrs.insert(cast<StoreInst>(&I)->getPointerOperand());
      if (storedPtrs.empty()) continue;

      for (auto *BB : L->blocks()) {
        for (auto &I : *BB) {
          auto *LdI = dyn_cast<LoadInst>(&I);
          if (LdI && isDeviceLoad(&I) && !LdI->isVolatile() &&
              storedPtrs.count(LdI->getPointerOperand())) {
            LdI->setVolatile(true);
            changed = true;
          }
        }
      }
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
