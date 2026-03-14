// Pass 22: NormalizeAllocas — pre-serialization IR cleanup.
//
// 1. Convert alloca i64 sizes to i32 (Metal v1 bitcode requirement)
// 2. Remove no-op pointer bitcasts (bitcast ptr to ptr is identity in
//    opaque pointer IR but encodes as a real CAST in v1 typed bitcode)

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

bool NormalizeAllocasPass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (auto *AI = dyn_cast<AllocaInst>(&I))
          if (AI->getArraySize()->getType()->isIntegerTy(64))
            return true;
        if (auto *BC = dyn_cast<BitCastInst>(&I))
          if (BC->getSrcTy() == BC->getDestTy())
            return true;
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getPointerAddressSpace() == 1 &&
              (GEP->getSourceElementType()->isHalfTy() ||
               GEP->getSourceElementType()->isBFloatTy()))
            return true;
      }
  return false;
}

PreservedAnalyses NormalizeAllocasPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  bool changed = false;
  Type *I32 = Type::getInt32Ty(M.getContext());

  // Strip 'disjoint' flag from 'or' instructions.
  // Metal v1 bitcode doesn't support this LLVM 19+ flag.
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *BO = dyn_cast<PossiblyDisjointInst>(&I))
          if (BO->isDisjoint()) {
            BO->setIsDisjoint(false);
            changed = true;
          }

  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        Instruction &I = *it++;

        // Normalize alloca i64 → i32
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          auto *Size = dyn_cast<ConstantInt>(AI->getArraySize());
          if (Size && Size->getType()->isIntegerTy(64)) {
            AI->setOperand(0, ConstantInt::get(I32, Size->getZExtValue()));
            changed = true;
          }
          continue;
        }

        // Keep all no-op bitcasts (ptr → ptr same type).
        // In Metal v1 bitcode they change the typed pointer
        // (e.g., i8* → <2 x float>*, or float* → bfloat*).
        if (auto *BC = dyn_cast<BitCastInst>(&I)) {
          continue;
        }

        // Fix GEP source type mismatch: gep half, ptr, idx where
        // downstream load/store is float → gep float, ptr, idx/2
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (GEP->getPointerAddressSpace() == 1 &&
              GEP->getNumIndices() == 1 &&
              (GEP->getSourceElementType()->isHalfTy() ||
               GEP->getSourceElementType()->isBFloatTy())) {
            // Recursive check: does any transitive user load/store float?
            std::function<bool(Value *)> hasFloatUse = [&](Value *V) -> bool {
              for (auto *U : V->users()) {
                if (auto *LI = dyn_cast<LoadInst>(U))
                  if (LI->getType()->isFloatTy()) return true;
                if (auto *SI = dyn_cast<StoreInst>(U))
                  if (SI->getValueOperand()->getType()->isFloatTy()) return true;
                if (isa<GetElementPtrInst>(U))
                  if (hasFloatUse(U)) return true;
              }
              return false;
            };
            if (hasFloatUse(GEP)) {
              IRBuilder<> B(GEP);
              Type *F32 = Type::getFloatTy(M.getContext());
              Value *idx = GEP->getOperand(1);
              Value *newIdx = B.CreateLShr(idx,
                  ConstantInt::get(idx->getType(), 1), "idx_f");
              auto *newGEP = B.CreateInBoundsGEP(F32, GEP->getPointerOperand(),
                                                   newIdx, GEP->getName());
              GEP->replaceAllUsesWith(newGEP);
              GEP->eraseFromParent();
              changed = true;
            }
          }
        }
      }
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
