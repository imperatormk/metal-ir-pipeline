// Pass 22: NormalizeAllocas — pre-serialization IR cleanup.
//
// 1. Convert alloca i64 sizes to i32 (Metal v1 bitcode requirement)
// 2. Remove no-op pointer bitcasts (bitcast ptr to ptr is identity in
//    opaque pointer IR but encodes as a real CAST in v1 typed bitcode)

#include "metal-ir/Pipeline.h"
#include "metal-ir/MetalConstraints.h"
#include "metal-ir/PassUtil.h"
#include "metal-ir/IRUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

bool NormalizeAllocasPass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          if (AI->getArraySize()->getType()->isIntegerTy(64))
            return true;
          // Alloca in non-entry block: needs hoisting
          if (&BB != &F.getEntryBlock())
            return true;
        }
        if (auto *BC = dyn_cast<BitCastInst>(&I))
          if (BC->getSrcTy() == BC->getDestTy())
            return true;
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          unsigned addrSpace = GEP->getPointerAddressSpace();
          if ((addrSpace == AS::Device || addrSpace == AS::Threadgroup) &&
              (GEP->getSourceElementType()->isHalfTy() ||
               GEP->getSourceElementType()->isBFloatTy()))
            return true;
          // float GEP through bfloat/half typed TG pointer
          if (addrSpace == AS::Threadgroup && GEP->getSourceElementType()->isFloatTy() &&
              GEP->getNumIndices() == 1 && !isa<BitCastInst>(GEP->getPointerOperand())) {
            // Check if pointer traces back to a bfloat/half TG global
            Value *ptr = GEP->getPointerOperand();
            if (auto *pGEP = dyn_cast<GetElementPtrInst>(ptr)) {
              if (auto *GV = dyn_cast<GlobalVariable>(pGEP->getPointerOperand())) {
                if (auto *AT = dyn_cast<ArrayType>(GV->getValueType()))
                  if (AT->getElementType()->isBFloatTy() || AT->getElementType()->isHalfTy())
                    return true;
              }
            }
          }
        }
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

  // Hoist allocas from non-entry blocks to the entry block.
  // Metal GPU JIT crashes on allocas in loop bodies.
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    BasicBlock &entry = F.getEntryBlock();
    Instruction *insertPt = &*entry.getFirstInsertionPt();
    for (auto &BB : F) {
      if (&BB == &entry) continue;
      for (auto it = BB.begin(); it != BB.end();) {
        auto *AI = dyn_cast<AllocaInst>(&*it++);
        if (!AI) continue;
        AI->moveBefore(insertPt->getIterator());
        changed = true;
      }
    }
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

        // Insert bitcast ptr→ptr before store when the pointer's typed
        // pointee differs from the stored value type. Covers:
        // - Device (AS1): non-float store through float* param
        // - Threadgroup (AS3): bfloat store through half* TG global
        // Metal v1 typed pointers require explicit bitcast to change type.
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *ptr = SI->getPointerOperand();
          unsigned addrSpace = ptr->getType()->getPointerAddressSpace();
          if ((addrSpace == AS::Device || addrSpace == AS::Threadgroup) && !isa<BitCastInst>(ptr)) {
            // Check if GEP source type differs from store value type
            Type *valTy = SI->getValueOperand()->getType();
            bool needsBitcast = false;
            if (addrSpace == AS::Device && !valTy->isFloatTy())
              needsBitcast = true;
            if (addrSpace == AS::Threadgroup) {
              if (auto *GEP = dyn_cast<GetElementPtrInst>(ptr))
                if (GEP->getSourceElementType() != valTy)
                  needsBitcast = true;
            }
            if (needsBitcast) {
              auto *BC = CastInst::Create(Instruction::BitCast, ptr,
                                          ptr->getType(), "", SI);
              SI->setOperand(1, BC);
              changed = true;
            }
          }
          continue;
        }

        // Insert bitcast ptr→ptr before load from TG memory when the
        // GEP source type differs from the loaded value type.
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          Value *ptr = LI->getPointerOperand();
          if (ptr->getType()->getPointerAddressSpace() == AS::Threadgroup &&
              !isa<BitCastInst>(ptr)) {
            if (auto *GEP = dyn_cast<GetElementPtrInst>(ptr))
              if (GEP->getSourceElementType() != LI->getType()) {
                auto *BC = CastInst::Create(Instruction::BitCast, ptr,
                                            ptr->getType(), "", LI);
                LI->setOperand(0, BC);
                changed = true;
              }
          }
          continue;
        }

        // Fix GEP source type mismatch: gep half, ptr, idx where
        // downstream load/store is float → gep float, ptr, idx/2
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          unsigned addrSpace = GEP->getPointerAddressSpace();

          // Float GEP through bfloat/half typed TG pointer: insert bitcast
          // so Metal v1 typed pointer changes from bfloat* to float*.
          if (addrSpace == AS::Threadgroup && GEP->getSourceElementType()->isFloatTy() &&
              GEP->getNumIndices() == 1 &&
              !isa<BitCastInst>(GEP->getPointerOperand())) {
            Value *ptr = GEP->getPointerOperand();
            bool needsBitcast = false;
            if (auto *pGEP = dyn_cast<GetElementPtrInst>(ptr)) {
              if (auto *GV = dyn_cast<GlobalVariable>(pGEP->getPointerOperand())) {
                if (auto *AT = dyn_cast<ArrayType>(GV->getValueType()))
                  if (AT->getElementType()->isBFloatTy() || AT->getElementType()->isHalfTy())
                    needsBitcast = true;
              }
            }
            if (needsBitcast) {
              auto *BC = CastInst::Create(Instruction::BitCast,
                  ptr, ptr->getType(), "", GEP);
              GEP->setOperand(0, BC);
              changed = true;
            }
          }

          if ((addrSpace == AS::Device || addrSpace == AS::Threadgroup) &&
              GEP->getNumIndices() == 1 &&
              (GEP->getSourceElementType()->isHalfTy() ||
               GEP->getSourceElementType()->isBFloatTy())) {
            if (metalir::hasFloatUse(GEP)) {
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
            } else if (GEP->getSourceElementType()->isBFloatTy() &&
                       !isa<BitCastInst>(GEP->getPointerOperand())) {
              // bfloat GEP through a non-bfloat typed pointer (e.g., float*
              // from a merged TG global or device param). Insert bitcast
              // ptr→ptr so Metal v1 bitcode sees the typed pointer change.
              auto *BC = CastInst::Create(Instruction::BitCast,
                  GEP->getPointerOperand(),
                  GEP->getPointerOperand()->getType(), "", GEP);
              GEP->setOperand(0, BC);
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
