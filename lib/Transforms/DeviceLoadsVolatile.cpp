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
#include "metal-ir/PassUtil.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool DeviceLoadsVolatilePass::needsRun(Module &M) {
  // Needs to run if there are device loads inside loops
  // (cheap approximation: any device load in a function with >1 BB)
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    unsigned bbCount = 0;
    bool hasDeviceLoad = false;
    for (auto &BB : F) {
      bbCount++;
      for (auto &I : BB)
        if (auto *LI = dyn_cast<LoadInst>(&I))
          if (LI->getPointerAddressSpace() == 1 && !LI->isVolatile())
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

  for (auto &F : M) {
    if (F.isDeclaration()) continue;

    // Build dominator tree and loop info
    DominatorTree DT(F);
    LoopInfo LI(DT);

    for (auto *L : LI.getLoopsInPreorder()) {
      // Collect device pointers stored in this loop
      SmallPtrSet<Value *, 8> storedPtrs;
      for (auto *BB : L->blocks()) {
        for (auto &I : *BB) {
          if (auto *SI = dyn_cast<StoreInst>(&I)) {
            if (SI->getPointerAddressSpace() == 1)
              storedPtrs.insert(SI->getPointerOperand());
          }
        }
      }
      if (storedPtrs.empty()) continue;

      // Mark device loads from stored pointers as volatile
      for (auto *BB : L->blocks()) {
        for (auto &I : *BB) {
          if (auto *LI = dyn_cast<LoadInst>(&I)) {
            if (LI->getPointerAddressSpace() == 1 && !LI->isVolatile()) {
              if (storedPtrs.count(LI->getPointerOperand())) {
                LI->setVolatile(true);
                changed = true;
              }
            }
          }
        }
      }
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
