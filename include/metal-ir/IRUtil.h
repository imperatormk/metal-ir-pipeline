#pragma once

/// Shared IR utility functions for Metal IR pipeline passes.
///
/// These eliminate duplicated patterns across TGGlobalGEPRewrite,
/// TGGlobalCoalesce, TGBarrierInsert, NormalizeAllocas, etc.

#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

namespace metalir {

// ── Address space checks ──────────────────────────────────────────────────
// V->getType()->getPointerAddressSpace() == N appears in 15+ passes.

inline bool isDevicePtr(llvm::Value *V) {
  return V->getType()->isPointerTy() &&
         V->getType()->getPointerAddressSpace() == 1;
}

inline bool isTGPtr(llvm::Value *V) {
  return V->getType()->isPointerTy() &&
         V->getType()->getPointerAddressSpace() == 3;
}

inline bool isConstPtr(llvm::Value *V) {
  return V->getType()->isPointerTy() &&
         V->getType()->getPointerAddressSpace() == 2;
}

// ── Recursive element type inference ──────────────────────────────────────
// Walks users transitively to find the actual store/load element type
// for a pointer value. Used by TGGlobalGEPRewrite and NormalizeAllocas.

inline llvm::Type *inferElementType(llvm::Value *V) {
  for (auto *U : V->users()) {
    if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(U))
      if (SI->getPointerOperand() == V)
        return SI->getValueOperand()->getType();
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U))
      return LI->getType();
    if (llvm::isa<llvm::GetElementPtrInst>(U) ||
        llvm::isa<llvm::GEPOperator>(U) ||
        llvm::isa<llvm::BitCastInst>(U))
      if (llvm::Type *T = inferElementType(U))
        return T;
  }
  return nullptr;
}

// ── ConstantExpr expansion ────────────────────────────────────────────────
// Expand all ConstantExpr users of a global into instructions.
// Used by TGGlobalGEPRewrite and BitcodeEmitter.

inline void expandConstantExprUsers(llvm::GlobalVariable *GV) {
  llvm::SmallVector<std::pair<llvm::ConstantExpr *, llvm::Instruction *>, 4>
      toExpand;
  for (auto *U : GV->users()) {
    auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(U);
    if (!CE) continue;
    for (auto *CEU : CE->users())
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(CEU))
        toExpand.push_back({CE, I});
  }
  for (auto &[CE, I] : toExpand) {
    auto *Inst = CE->getAsInstruction();
    Inst->insertBefore(I->getIterator());
    I->replaceUsesOfWith(CE, Inst);
  }
  llvm::SmallVector<llvm::ConstantExpr *, 4> dead;
  for (auto *U : GV->users())
    if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(U))
      if (CE->use_empty())
        dead.push_back(CE);
  for (auto *CE : dead)
    CE->destroyConstant();
}

// ── Collect i8-source GEPs ────────────────────────────────────────────────
// Recursively collect all i8-source GEPs in the user chain of V.
// Used by TGGlobalGEPRewrite.

inline void collectI8Geps(llvm::Value *V,
                           llvm::SmallVectorImpl<llvm::GetElementPtrInst *> &out) {
  for (auto *U : V->users()) {
    if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(U)) {
      if (GEP->getSourceElementType()->isIntegerTy(8))
        out.push_back(GEP);
      else
        collectI8Geps(GEP, out);
    }
  }
}

// ── GEP byte-to-element index conversion ──────────────────────────────────
// Divides byte index by element size. Appears 8+ times in TGGlobalGEPRewrite.

inline llvm::Value *createElementIndex(llvm::IRBuilder<> &B,
                                        llvm::Value *byteIdx,
                                        unsigned elemSizeBytes) {
  if (elemSizeBytes == 1) return byteIdx;
  return B.CreateUDiv(byteIdx,
                       llvm::ConstantInt::get(byteIdx->getType(), elemSizeBytes),
                       "elem_idx");
}

} // namespace metalir
