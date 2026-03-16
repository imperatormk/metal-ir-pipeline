#pragma once

/// Shared IR utility functions for Metal IR pipeline passes.
///
/// These eliminate duplicated patterns across TGGlobalGEPRewrite,
/// TGGlobalCoalesce, TGBarrierInsert, NormalizeAllocas, etc.

#include "metal-ir/MetalConstraints.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

namespace metalir {

// ── Pointer address space checks ──────────────────────────────────────────

inline bool isDevicePtr(llvm::Value *V) {
  return V->getType()->isPointerTy() &&
         V->getType()->getPointerAddressSpace() == AS::Device;
}

inline bool isTGPtr(llvm::Value *V) {
  return V->getType()->isPointerTy() &&
         V->getType()->getPointerAddressSpace() == AS::Threadgroup;
}

inline bool isConstPtr(llvm::Value *V) {
  return V->getType()->isPointerTy() &&
         V->getType()->getPointerAddressSpace() == AS::Constant;
}

// ── Instruction-level address space checks ────────────────────────────────
// Eliminate the `dyn_cast<StoreInst> + getPointerAddressSpace()` pattern
// that appears in TGBarrierInsert, DeviceLoadsVolatile, WidenDeviceLoads, etc.

inline bool isTGStore(llvm::Instruction *I) {
  if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(I))
    return SI->getPointerAddressSpace() == AS::Threadgroup;
  return false;
}

inline bool isTGLoad(llvm::Instruction *I) {
  if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(I))
    return LI->getPointerAddressSpace() == AS::Threadgroup;
  return false;
}

inline bool isDeviceStore(llvm::Instruction *I) {
  if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(I))
    return SI->getPointerAddressSpace() == AS::Device;
  return false;
}

inline bool isDeviceLoad(llvm::Instruction *I) {
  if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(I))
    return LI->getPointerAddressSpace() == AS::Device;
  return false;
}

// ── Module-level TG global collection ─────────────────────────────────────
// Every TG pass (DeadElim, Coalesce, GEPRewrite, Preamble) scans globals
// the same way. Centralize it.

inline void collectTGGlobals(
    llvm::Module &M,
    llvm::SmallVectorImpl<llvm::GlobalVariable *> &out) {
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == AS::Threadgroup)
      out.push_back(&GV);
}

/// Collect only [N x i8] byte globals in threadgroup memory.
inline void collectTGByteGlobals(
    llvm::Module &M,
    llvm::SmallVectorImpl<llvm::GlobalVariable *> &out) {
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != AS::Threadgroup) continue;
    auto *AT = llvm::dyn_cast<llvm::ArrayType>(GV.getValueType());
    if (AT && AT->getElementType()->isIntegerTy(8))
      out.push_back(&GV);
  }
}

/// Collect typed (non-i8) array globals in threadgroup memory.
inline void collectTGTypedGlobals(
    llvm::Module &M,
    llvm::SmallVectorImpl<llvm::GlobalVariable *> &out) {
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != AS::Threadgroup) continue;
    auto *AT = llvm::dyn_cast<llvm::ArrayType>(GV.getValueType());
    if (AT && !AT->getElementType()->isIntegerTy(8))
      out.push_back(&GV);
  }
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

// ── Recursive float-use check ─────────────────────────────────────────────
// Checks if any transitive user of V loads/stores float.
// Used by NormalizeAllocas to decide half→float GEP rewriting.

inline bool hasFloatUse(llvm::Value *V) {
  for (auto *U : V->users()) {
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U))
      if (LI->getType()->isFloatTy()) return true;
    if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(U))
      if (SI->getValueOperand()->getType()->isFloatTy()) return true;
    if (llvm::isa<llvm::GetElementPtrInst>(U))
      if (hasFloatUse(U)) return true;
  }
  return false;
}

// ── Vec1 scalarization ────────────────────────────────────────────────────
// Replace <1 x T> store/load through a pointer with scalar store/load.
// Duplicated in TGGlobalGEPRewrite and TGGlobalCoalesce.
// Returns true if any changes were made.

inline bool scalarizeVec1Users(llvm::Value *V, llvm::Type *I32Ty) {
  bool changed = false;
  llvm::SmallVector<llvm::Instruction *, 8> vec1Users;
  std::function<void(llvm::Value *)> findVec1 = [&](llvm::Value *V) {
    for (auto *U : V->users()) {
      if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(U)) {
        if (SI->getPointerOperand() == V) {
          auto *VT = llvm::dyn_cast<llvm::FixedVectorType>(
              SI->getValueOperand()->getType());
          if (VT && VT->getNumElements() == 1) vec1Users.push_back(SI);
        }
      } else if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
        auto *VT = llvm::dyn_cast<llvm::FixedVectorType>(LI->getType());
        if (VT && VT->getNumElements() == 1) vec1Users.push_back(LI);
      } else if (llvm::isa<llvm::GetElementPtrInst>(U)) {
        findVec1(U);
      }
    }
  };
  findVec1(V);
  for (auto *I : vec1Users) {
    if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(I)) {
      llvm::IRBuilder<> B(SI);
      llvm::Value *scalar = B.CreateExtractElement(
          SI->getValueOperand(), llvm::ConstantInt::get(I32Ty, 0));
      B.CreateAlignedStore(scalar, SI->getPointerOperand(), SI->getAlign(),
                           SI->isVolatile());
      SI->eraseFromParent();
      changed = true;
    } else if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(I)) {
      llvm::IRBuilder<> B(LI);
      auto *VT = llvm::cast<llvm::FixedVectorType>(LI->getType());
      auto *scalar = B.CreateAlignedLoad(VT->getElementType(),
                                          LI->getPointerOperand(),
                                          LI->getAlign(), LI->isVolatile());
      llvm::Value *vec = B.CreateInsertElement(
          llvm::UndefValue::get(VT), scalar, llvm::ConstantInt::get(I32Ty, 0));
      LI->replaceAllUsesWith(vec);
      LI->eraseFromParent();
      changed = true;
    }
  }
  return changed;
}

// ── Fold extract(insert(undef, x, 0), 0) → x ─────────────────────────────
// Cleans up after vec1 scalarization. Returns true if any changes were made.

inline bool foldExtractInsert(llvm::Function &F) {
  bool changed = false;
  for (auto &BB : F) {
    for (auto it = BB.begin(); it != BB.end();) {
      llvm::Instruction &I = *it++;
      if (auto *EE = llvm::dyn_cast<llvm::ExtractElementInst>(&I)) {
        if (auto *IE =
                llvm::dyn_cast<llvm::InsertElementInst>(EE->getVectorOperand())) {
          auto *VT = llvm::dyn_cast<llvm::FixedVectorType>(IE->getType());
          if (VT && VT->getNumElements() == 1) {
            EE->replaceAllUsesWith(IE->getOperand(1));
            EE->eraseFromParent();
            if (IE->use_empty()) IE->eraseFromParent();
            changed = true;
          }
        }
      }
    }
  }
  return changed;
}

// Module-wide version
inline bool foldExtractInsert(llvm::Module &M) {
  bool changed = false;
  for (auto &F : M)
    changed |= foldExtractInsert(F);
  return changed;
}

} // namespace metalir
