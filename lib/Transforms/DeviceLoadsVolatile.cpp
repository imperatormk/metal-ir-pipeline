// Pass 20: Mark device loads in loops as volatile.
//
// Metal's GPU JIT hoists non-volatile addrspace(1) loads out of loops,
// even when a store to the same pointer exists in the loop body. This
// causes loops with load+store patterns to read stale values.
//
// Fix: find back-edges (loops), collect stored device pointers,
// mark loads from those pointers as volatile.
//
// Also handles cross-threadgroup synchronization via atomic CAS spin-locks:
// when a function uses CAS atomics, device loads/stores in critical sections
// (after lock acquire) must be volatile to prevent the GPU from reordering
// them across the atomic boundary. Without this, threadgroup B may read
// stale data written by threadgroup A under a spin-lock.
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

// Check if function contains any CAS atomic calls
static bool hasCASAtomic(Function &F) {
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (auto *Callee = CI->getCalledFunction())
          if (Callee->getName().contains("cmpxchg"))
            return true;
  return false;
}

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

    // If the function uses CAS atomics (spin-lock pattern), mark ALL
    // device loads and stores as volatile to prevent GPU reordering
    // across the atomic boundary.
    if (hasCASAtomic(F)) {
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *LdI = dyn_cast<LoadInst>(&I)) {
            if (isDeviceLoad(&I) && !LdI->isVolatile()) {
              LdI->setVolatile(true);
              changed = true;
            }
          } else if (auto *StI = dyn_cast<StoreInst>(&I)) {
            if (isDeviceStore(&I) && !StI->isVolatile()) {
              StI->setVolatile(true);
              changed = true;
            }
          }
        }
      }
      continue;
    }

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
