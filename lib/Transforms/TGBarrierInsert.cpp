// Pass 4: TGBarrierInsert — insert barriers around TG memory accesses.
//
// Metal requires explicit barriers for threadgroup memory coherence.
// Without barriers between TG stores and subsequent TG loads (across
// threads), data races cause corruption.
//
// Strategy:
// 1. Insert barrier before TG stores in straight-line blocks
// 2. For conditional branches to TG-store blocks, insert barrier at join
// 3. WAR hazard: barrier between TG load and branch to TG-store block
//
// Skip blocks that are conditional branch targets (barrier divergence
// causes GPU hang).

#include "metal-ir/Pipeline.h"
#include "metal-ir/AIRIntrinsics.h"
#include "metal-ir/IRUtil.h"
#include "metal-ir/KernelProfile.h"
#include "metal-ir/MetalConstraints.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

static bool isBarrierCall(Instruction *I) {
  if (auto *CI = dyn_cast<CallInst>(I))
    if (auto *F = CI->getCalledFunction())
      return F->getName() == air::kBarrier ||
             F->getName() == air::kBarrierOld;
  return false;
}

static CallInst *createBarrier(IRBuilder<> &B, Module &M) {
  auto &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), {I32, I32}, false);
  FunctionCallee FC = M.getOrInsertFunction(air::kBarrier, FTy);
  return B.CreateCall(FC, {ConstantInt::get(I32, 2), ConstantInt::get(I32, 1)});
}

bool TGBarrierInsertPass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (isTGStore(&I))
          return true;
  return false;
}

PreservedAnalyses TGBarrierInsertPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  bool changed = false;
  auto &profiles = AM.getResult<KernelProfileAnalysis>(M);

  for (auto &F : M) {
    if (F.isDeclaration()) continue;

    // Early exit: KernelProfile says no TG stores in this function
    auto it = profiles.find(&F);
    if (it != profiles.end() && !it->second.needsTGBarriers())
      continue;

    // Collect blocks with TG stores and TG loads
    SmallPtrSet<BasicBlock *, 8> tgStoreBlocks, tgLoadBlocks;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (isTGStore(&I)) tgStoreBlocks.insert(&BB);
        if (isTGLoad(&I)) tgLoadBlocks.insert(&BB);
      }
    }
    if (tgStoreBlocks.empty()) continue;

    // Collect conditional branch targets (barrier divergence risk)
    SmallPtrSet<BasicBlock *, 8> condTargets;
    for (auto &BB : F) {
      auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
      if (BI && BI->isConditional()) {
        condTargets.insert(BI->getSuccessor(0));
        condTargets.insert(BI->getSuccessor(1));
      }
    }

    // Strategy 1: barrier before TG stores in non-conditional-target blocks
    for (auto &BB : F) {
      if (!tgStoreBlocks.count(&BB) || condTargets.count(&BB))
        continue;
      for (auto it = BB.begin(); it != BB.end(); ++it) {
        if (!isTGStore(&*it)) continue;
        // Check if already preceded by barrier
        if (it != BB.begin()) {
          auto prev = std::prev(it);
          if (isBarrierCall(&*prev))
            continue;
        }
        IRBuilder<> B(&*it);
        createBarrier(B, M);
        changed = true;
      }
    }

    // Strategy 2: for conditional branches to TG-store blocks,
    // insert barrier at the join block (false-branch target)
    for (auto &BB : F) {
      auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
      if (!BI || !BI->isConditional()) continue;

      BasicBlock *trueBB = BI->getSuccessor(0);
      BasicBlock *falseBB = BI->getSuccessor(1);

      if (tgStoreBlocks.count(trueBB)) {
        // Insert barrier at start of join block (falseBB)
        if (falseBB->empty() || !isBarrierCall(&falseBB->front())) {
          IRBuilder<> B(&*falseBB->getFirstInsertionPt());
          createBarrier(B, M);
          changed = true;
        }
      }
    }

    // Strategy 3: WAR hazard — barrier between TG load and TG store
    // When a block has a TG load and later reaches a TG store (in same
    // block or successor) without an intervening barrier, insert one.
    // This prevents a fast warp from overwriting TG data that a slow
    // warp hasn't read yet (e.g., multi-reduction kernels reusing TG).
    for (auto &BB : F) {
      if (!tgLoadBlocks.count(&BB)) continue;

      // Find the last TG load in this block
      Instruction *lastTGLoad = nullptr;
      for (auto &I : BB) {
        if (isTGLoad(&I)) lastTGLoad = &I;
      }
      if (!lastTGLoad) continue;

      // Check if there's a barrier after the last TG load in this block
      bool hasBarrierAfterLoad = false;
      for (auto it = lastTGLoad->getIterator(); it != BB.end(); ++it) {
        if (isBarrierCall(&*it)) {
          hasBarrierAfterLoad = true;
          break;
        }
      }
      if (hasBarrierAfterLoad) continue;

      // Check if any successor has a TG store (directly or via
      // conditional branch to a TG-store block)
      auto *term = BB.getTerminator();
      bool succHasTGStore = false;
      for (unsigned i = 0; i < term->getNumSuccessors(); i++) {
        BasicBlock *succ = term->getSuccessor(i);
        if (tgStoreBlocks.count(succ)) {
          succHasTGStore = true;
          break;
        }
        // Also check one level deeper: succ branches to TG store
        if (auto *succBI = dyn_cast<BranchInst>(succ->getTerminator())) {
          for (unsigned j = 0; j < succBI->getNumSuccessors(); j++) {
            if (tgStoreBlocks.count(succBI->getSuccessor(j))) {
              // Only if no barrier in succ before the branch
              bool succHasBarrier = false;
              for (auto &SI : *succ)
                if (isBarrierCall(&SI)) { succHasBarrier = true; break; }
              if (!succHasBarrier) succHasTGStore = true;
            }
          }
        }
      }

      if (succHasTGStore) {
        IRBuilder<> B(term);
        createBarrier(B, M);
        changed = true;
      }
    }
  }

  if (!changed) return PreservedAnalyses::all();
  // Doesn't add/remove BBs, just inserts instructions
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

} // namespace metalir
