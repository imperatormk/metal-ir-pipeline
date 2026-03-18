// Pass 14: TGGlobalGEPRewrite — retype [N x i8] TG globals for Metal.
//
// Triton emits threadgroup memory as [N x i8] with byte-offset GEPs.
// Metal's typed pointer system needs GEP source types to match store/load types.
//
// Four sub-passes run in sequence:
//   14a. SplitMixed: split mixed-type byte globals at constant offsets
//   14b. MergeMMA: merge byte globals into MMA globals (same TG memory)
//   14c. Retype: [N x i8] → [M x T] based on store/load types
//   14d. Preamble: insert gep [M x T], @g, 0, 0 at function entry

#include "metal-ir/Pipeline.h"
#include "metal-ir/MetalConstraints.h"
#include "metal-ir/PassUtil.h"
#include "metal-ir/IRUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool TGGlobalGEPRewritePass::needsRun(Module &M) {
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == AS::Threadgroup && isa<ArrayType>(GV.getValueType()))
      return true;
  return false;
}

// ── Helper: rewrite byte GEPs to typed element GEPs ─────────────────────
// Shared by MergeMMA, Retype, and Strategy C. Rewrites GEPs on oldGV
// to use newGV with element type elemTy.
static bool rewriteByteGEPs(GlobalVariable *oldGV, GlobalVariable *newGV,
                              ArrayType *oldAT, ArrayType *newAT,
                              Type *elemTy, unsigned elemSize,
                              LLVMContext &Ctx) {
  bool changed = false;
  SmallVector<GetElementPtrInst *, 16> users;
  for (auto *U : oldGV->users())
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
      users.push_back(GEP);

  for (auto *GEP : users) {
    if (GEP->getPointerOperand() != oldGV) continue;
    IRBuilder<> B(GEP);
    Type *srcTy = GEP->getSourceElementType();

    if (srcTy == oldAT) {
      // gep [N x i8], @old, 0, byteIdx → gep [M x T], @new, 0, elemIdx
      Value *byteIdx = GEP->getNumIndices() >= 2
                           ? GEP->getOperand(2)
                           : ConstantInt::get(Type::getInt64Ty(Ctx), 0);
      Value *elemIdx;
      if (auto *CI = dyn_cast<ConstantInt>(byteIdx))
        elemIdx = ConstantInt::get(CI->getType(), CI->getZExtValue() / elemSize);
      else
        elemIdx = B.CreateUDiv(byteIdx, ConstantInt::get(byteIdx->getType(), elemSize));
      auto *newGEP = GetElementPtrInst::CreateInBounds(
          newAT, newGV,
          {ConstantInt::get(Type::getInt64Ty(Ctx), 0), elemIdx},
          GEP->getName());
      newGEP->insertBefore(B.GetInsertPoint());
      GEP->replaceAllUsesWith(newGEP);
      GEP->eraseFromParent();
    } else if (srcTy->isIntegerTy(8)) {
      // gep i8, @old, byteIdx → gep T, @new, elemIdx
      Value *byteIdx = GEP->getOperand(1);
      Value *elemIdx;
      if (auto *CI = dyn_cast<ConstantInt>(byteIdx))
        elemIdx = ConstantInt::get(CI->getType(), CI->getZExtValue() / elemSize);
      else
        elemIdx = B.CreateUDiv(byteIdx, ConstantInt::get(byteIdx->getType(), elemSize));
      auto *newGEP = GetElementPtrInst::CreateInBounds(
          elemTy, newGV, elemIdx, GEP->getName());
      newGEP->insertBefore(B.GetInsertPoint());
      GEP->replaceAllUsesWith(newGEP);
      GEP->eraseFromParent();
    } else {
      GEP->setOperand(0, newGV);
    }
    changed = true;
  }

  // Redirect remaining direct (non-GEP) instruction users
  SmallVector<Instruction *, 4> directUsers;
  for (auto *U : oldGV->users()) {
    auto *I = dyn_cast<Instruction>(U);
    if (!I || isa<GetElementPtrInst>(I)) continue;
    directUsers.push_back(I);
  }
  for (auto *I : directUsers) {
    for (unsigned op = 0; op < I->getNumOperands(); op++)
      if (I->getOperand(op) == oldGV)
        I->setOperand(op, newGV);
    changed = true;
  }
  return changed;
}

// ── 14a: Split mixed-type byte globals at constant offsets ──────────────
static bool splitMixedByteGlobals(Module &M,
                                    SmallVectorImpl<GlobalVariable *> &byteGlobals) {
  bool changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();

  for (size_t gi = 0; gi < byteGlobals.size(); gi++) {
    auto *GV = byteGlobals[gi];
    expandConstantExprUsers(GV);
    auto *oldAT = cast<ArrayType>(GV->getValueType());
    uint64_t totalBytes = oldAT->getNumElements();

    SmallPtrSet<Type *, 4> allScalarTypes;
    SmallVector<int64_t, 4> constOffsets;
    std::function<void(Value *, int64_t)> collectTypes = [&](Value *V, int64_t baseOff) {
      for (auto *U : V->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() == V) {
            Type *T = SI->getValueOperand()->getType();
            if (T->isIntegerTy() || T->isFloatingPointTy())
              allScalarTypes.insert(T);
          }
        } else if (auto *LI = dyn_cast<LoadInst>(U)) {
          Type *T = LI->getType();
          if (T->isIntegerTy() || T->isFloatingPointTy())
            allScalarTypes.insert(T);
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          APInt off(64, 0);
          if (GEP->accumulateConstantOffset(DL, off)) {
            int64_t byteOff = off.getSExtValue();
            if (byteOff != 0) constOffsets.push_back(byteOff);
            collectTypes(GEP, baseOff + byteOff);
          } else {
            collectTypes(GEP, baseOff);
          }
        } else if (isa<BitCastInst>(U)) {
          collectTypes(U, baseOff);
        }
      }
    };
    collectTypes(GV, 0);

    if (allScalarTypes.size() <= 1 || constOffsets.empty()) continue;

    llvm::sort(constOffsets);
    constOffsets.erase(std::unique(constOffsets.begin(), constOffsets.end()),
                       constOffsets.end());

    DenseMap<int64_t, GlobalVariable *> splitMap;
    for (int64_t off : constOffsets) {
      uint64_t regionSize = totalBytes - off;
      if (regionSize == 0) continue;
      auto *splitAT = ArrayType::get(Type::getInt8Ty(Ctx), regionSize);
      auto *splitGV = new GlobalVariable(
          M, splitAT, false, GV->getLinkage(),
          UndefValue::get(splitAT),
          GV->getName() + "__off" + Twine(off),
          GV, GlobalVariable::NotThreadLocal, AS::Threadgroup);
      splitGV->setAlignment(GV->getAlign());
      splitMap[off] = splitGV;
    }

    auto *newAT = ArrayType::get(Type::getInt8Ty(Ctx), constOffsets[0]);
    auto *newGV = new GlobalVariable(
        M, newAT, false, GV->getLinkage(),
        UndefValue::get(newAT), GV->getName().str(),
        GV, GlobalVariable::NotThreadLocal, AS::Threadgroup);
    newGV->setAlignment(GV->getAlign());

    SmallVector<GetElementPtrInst *, 8> users;
    for (auto *U : GV->users())
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
        users.push_back(GEP);

    for (auto *GEP : users) {
      if (GEP->getPointerOperand() != GV) continue;
      APInt off(64, 0);
      if (GEP->accumulateConstantOffset(DL, off)) {
        int64_t byteOff = off.getSExtValue();
        if (byteOff == 0) {
          GEP->setOperand(0, newGV);
          if (GEP->getSourceElementType() == oldAT)
            GEP->setSourceElementType(newAT);
        } else {
          auto sit = splitMap.find(byteOff);
          if (sit == splitMap.end()) continue;
          GEP->replaceAllUsesWith(sit->second);
          GEP->eraseFromParent();
        }
      } else {
        GEP->setOperand(0, newGV);
        if (GEP->getSourceElementType() == oldAT)
          GEP->setSourceElementType(newAT);
      }
      changed = true;
    }

    if (GV->use_empty()) GV->eraseFromParent();
    byteGlobals[gi] = newGV;
    for (auto &kv : splitMap)
      byteGlobals.push_back(kv.second);
    changed = true;
  }
  return changed;
}

// ── 14b: Merge byte globals into MMA globals ───────────────────────────
static bool mergeByteMMA(Module &M,
                           SmallVectorImpl<GlobalVariable *> &byteGlobals,
                           SmallVectorImpl<GlobalVariable *> &mmaGlobals) {
  if (byteGlobals.empty() || mmaGlobals.size() != 1)
    return false;

  bool changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  Type *I32 = Type::getInt32Ty(Ctx);

  auto *byteGV = byteGlobals[0];
  expandConstantExprUsers(byteGV);

  // Check for wide vector stores
  bool hasWideVec = false;
  {
    std::function<void(Value *)> check = [&](Value *V) {
      for (auto *U : V->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U))
          if (SI->getPointerOperand() == V)
            if (auto *VT = dyn_cast<FixedVectorType>(SI->getValueOperand()->getType()))
              if (VT->getNumElements() > 1) hasWideVec = true;
        if (isa<GetElementPtrInst>(U)) check(U);
      }
    };
    check(byteGV);
  }

  // Scalarize <1 x T>
  changed |= scalarizeVec1Users(byteGV, I32);
  changed |= foldExtractInsert(M);

  if (hasWideVec) return changed;

  // Find best byte global to merge
  auto *mmaGV = mmaGlobals[0];
  auto *mmaAT = cast<ArrayType>(mmaGV->getValueType());
  Type *mmaElemTy = mmaAT->getElementType();

  int bestIdx = -1;
  uint64_t bestBytes = 0;
  for (int i = 0; i < (int)byteGlobals.size(); i++) {
    auto *bAT = cast<ArrayType>(byteGlobals[i]->getValueType());
    uint64_t bBytes = bAT->getNumElements();
    Type *inferred = inferElementType(byteGlobals[i]);
    bool typeMatch = inferred && (inferred == mmaElemTy ||
        (inferred->isIntegerTy(32) && mmaElemTy->isFloatTy()) ||
        (inferred->isFloatTy() && mmaElemTy->isIntegerTy(32)));
    if (typeMatch && bBytes > bestBytes) {
      bestIdx = i;
      bestBytes = bBytes;
    }
  }
  if (bestIdx < 0) {
    for (int i = 0; i < (int)byteGlobals.size(); i++) {
      auto *bAT = cast<ArrayType>(byteGlobals[i]->getValueType());
      if (bAT->getNumElements() > bestBytes) {
        bestBytes = bAT->getNumElements();
        bestIdx = i;
      }
    }
  }

  byteGV = byteGlobals[bestIdx >= 0 ? bestIdx : 0];
  auto *byteAT = cast<ArrayType>(byteGV->getValueType());
  uint64_t byteBytes = byteAT->getNumElements();
  unsigned mmaElemSize = DL.getTypeAllocSize(mmaAT->getElementType());
  uint64_t mmaBytes = mmaAT->getNumElements() * mmaElemSize;

  Type *mergeElemTy = mmaElemTy;
  unsigned mergeElemSize = mmaElemSize;
  if (mergeElemSize == 0) {
    mergeElemTy = Type::getFloatTy(Ctx);
    mergeElemSize = 4;
  }
  uint64_t mergedElemCount = (std::max(byteBytes, mmaBytes) + mergeElemSize - 1) / mergeElemSize;

  auto *mergedAT = ArrayType::get(mergeElemTy, mergedElemCount);
  expandConstantExprUsers(byteGV);

  auto *mergedGV = new GlobalVariable(
      M, mergedAT, false, byteGV->getLinkage(),
      UndefValue::get(mergedAT), byteGV->getName().str(),
      byteGV, GlobalVariable::NotThreadLocal, AS::Threadgroup);
  mergedGV->setAlignment(byteGV->getAlign());

  changed |= rewriteByteGEPs(byteGV, mergedGV, byteAT, mergedAT,
                               mergeElemTy, mergeElemSize, Ctx);

  if (byteGV->use_empty()) byteGV->eraseFromParent();
  mmaGV->replaceAllUsesWith(mergedGV);
  mmaGV->eraseFromParent();

  if (bestIdx >= 0)
    byteGlobals.erase(byteGlobals.begin() + bestIdx);
  else
    byteGlobals.clear();

  return true;
}

// ── 14c: Retype [N x i8] → [M x T] ────────────────────────────────────
static bool retypeByteGlobals(Module &M) {
  bool changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  Type *I32 = Type::getInt32Ty(Ctx);

  SmallVector<GlobalVariable *, 4> byteGlobals;
  collectTGByteGlobals(M, byteGlobals);

  for (auto *GV : byteGlobals) {
    expandConstantExprUsers(GV);
    Type *storeTy = inferElementType(GV);
    if (!storeTy) continue;

    // Check for mixed types — insert bitcasts instead of retyping
    {
      SmallPtrSet<Type *, 4> storeTypes;
      std::function<void(Value *)> collectTypes = [&](Value *V) {
        for (auto *U : V->users()) {
          if (auto *SI = dyn_cast<StoreInst>(U))
            if (SI->getPointerOperand() == V)
              storeTypes.insert(SI->getValueOperand()->getType());
          if (auto *LI = dyn_cast<LoadInst>(U))
            storeTypes.insert(LI->getType());
          if (isa<GetElementPtrInst>(U))
            collectTypes(U);
        }
      };
      collectTypes(GV);
      if (storeTypes.size() > 1) {
        std::function<void(Value *)> insertBitcasts = [&](Value *V) {
          for (auto *U : make_early_inc_range(V->users())) {
            if (auto *SI = dyn_cast<StoreInst>(U)) {
              if (SI->getPointerOperand() == V &&
                  !SI->getValueOperand()->getType()->isIntegerTy(8)) {
                auto *BC = new BitCastInst(V, V->getType(), "");
                BC->insertBefore(SI->getIterator());
                SI->setOperand(1, BC);
                changed = true;
              }
            } else if (auto *LI = dyn_cast<LoadInst>(U)) {
              if (!LI->getType()->isIntegerTy(8)) {
                auto *BC = new BitCastInst(V, V->getType(), "");
                BC->insertBefore(LI->getIterator());
                LI->setOperand(0, BC);
                changed = true;
              }
            } else if (isa<GetElementPtrInst>(U)) {
              insertBitcasts(U);
            }
          }
        };
        insertBitcasts(GV);
        continue;
      }
    }

    // Scalarize <1 x T>
    changed |= scalarizeVec1Users(GV, I32);
    changed |= foldExtractInsert(M);

    storeTy = inferElementType(GV);
    if (!storeTy) continue;

    auto *oldAT = cast<ArrayType>(GV->getValueType());
    uint64_t totalBytes = oldAT->getNumElements();

    Type *elemTy = storeTy;
    if (elemTy->isBFloatTy()) elemTy = Type::getHalfTy(Ctx);

    unsigned elemSize = DL.getTypeAllocSize(elemTy);
    if (elemSize == 0) continue;
    uint64_t numElems = totalBytes / elemSize;
    if (numElems == 0) continue;

    auto *newAT = ArrayType::get(elemTy, numElems);
    auto *newGV = new GlobalVariable(
        M, newAT, GV->isConstant(), GV->getLinkage(),
        UndefValue::get(newAT), GV->getName() + ".typed",
        GV, GlobalVariable::NotThreadLocal, AS::Threadgroup);
    newGV->setAlignment(GV->getAlign());

    changed |= rewriteByteGEPs(GV, newGV, oldAT, newAT, elemTy, elemSize, Ctx);

    if (GV->use_empty()) GV->eraseFromParent();

    // Clean up residual i8 GEPs
    SmallVector<GetElementPtrInst *, 8> residualI8;
    collectI8Geps(newGV, residualI8);
    for (auto *GEP : residualI8) {
      IRBuilder<> B(GEP);
      Value *byteIdx = GEP->getOperand(1);
      Value *newIdx;
      if (auto *CI = dyn_cast<ConstantInt>(byteIdx))
        newIdx = ConstantInt::get(CI->getType(), CI->getZExtValue() / elemSize);
      else
        newIdx = B.CreateUDiv(byteIdx, ConstantInt::get(byteIdx->getType(), elemSize));
      auto *newGEP = B.CreateInBoundsGEP(elemTy, GEP->getPointerOperand(),
                                           newIdx, GEP->getName());
      GEP->replaceAllUsesWith(newGEP);
      GEP->eraseFromParent();
      changed = true;
    }
  }

  // Strategy C: split remaining byte globals at constant offsets + retype
  SmallVector<GlobalVariable *, 4> remaining;
  collectTGByteGlobals(M, remaining);

  for (auto *GV : remaining) {
    expandConstantExprUsers(GV);
    auto *oldAT = cast<ArrayType>(GV->getValueType());
    uint64_t totalBytes = oldAT->getNumElements();

    SmallVector<int64_t, 4> offsets;
    bool hasDynamic = false;
    for (auto *U : GV->users()) {
      auto *GEP = dyn_cast<GetElementPtrInst>(U);
      if (!GEP) continue;
      APInt off(64, 0);
      if (GEP->accumulateConstantOffset(DL, off)) {
        int64_t byteOff = off.getSExtValue();
        if (byteOff != 0) offsets.push_back(byteOff);
      } else {
        hasDynamic = true;
      }
    }

    if (hasDynamic || offsets.empty()) continue;

    llvm::sort(offsets);
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());

    DenseMap<int64_t, GlobalVariable *> splitMap;
    for (int64_t off : offsets) {
      uint64_t regionSize = totalBytes - off;
      if (regionSize == 0) continue;
      auto *splitAT = ArrayType::get(Type::getInt8Ty(Ctx), regionSize);
      auto *splitGV = new GlobalVariable(
          M, splitAT, false, GV->getLinkage(),
          UndefValue::get(splitAT),
          GV->getName() + "__off" + Twine(off),
          GV, GlobalVariable::NotThreadLocal, AS::Threadgroup);
      splitGV->setAlignment(GV->getAlign());
      splitMap[off] = splitGV;
    }

    auto *newAT = ArrayType::get(Type::getInt8Ty(Ctx), offsets[0]);
    auto *newGV = new GlobalVariable(
        M, newAT, false, GV->getLinkage(),
        UndefValue::get(newAT), GV->getName().str(),
        GV, GlobalVariable::NotThreadLocal, AS::Threadgroup);
    newGV->setAlignment(GV->getAlign());

    SmallVector<GetElementPtrInst *, 8> users;
    for (auto *U : GV->users())
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
        users.push_back(GEP);

    for (auto *GEP : users) {
      if (GEP->getPointerOperand() != GV) continue;
      APInt off(64, 0);
      if (!GEP->accumulateConstantOffset(DL, off)) continue;
      int64_t byteOff = off.getSExtValue();
      if (byteOff == 0) {
        GEP->setOperand(0, newGV);
        if (GEP->getSourceElementType() == oldAT)
          GEP->setSourceElementType(newAT);
        changed = true;
      } else {
        auto sit = splitMap.find(byteOff);
        if (sit == splitMap.end()) continue;
        GEP->replaceAllUsesWith(sit->second);
        GEP->eraseFromParent();
        changed = true;
      }
    }

    if (GV->use_empty()) GV->eraseFromParent();

    // Retype each split region
    SmallVector<GlobalVariable *, 4> toRetype;
    toRetype.push_back(newGV);
    for (auto &kv : splitMap) toRetype.push_back(kv.second);
    for (auto *splitGV : toRetype) {
      Type *elemTy = inferElementType(splitGV);
      if (!elemTy) continue;
      if (elemTy->isBFloatTy()) elemTy = Type::getHalfTy(Ctx);
      unsigned eSize = DL.getTypeAllocSize(elemTy);
      if (eSize == 0) continue;
      auto *splitOldAT = cast<ArrayType>(splitGV->getValueType());
      uint64_t nElems = splitOldAT->getNumElements() / eSize;
      if (nElems == 0) continue;
      auto *typedAT = ArrayType::get(elemTy, nElems);
      auto *typedGV = new GlobalVariable(
          M, typedAT, false, splitGV->getLinkage(),
          UndefValue::get(typedAT), splitGV->getName().str() + ".typed",
          splitGV, GlobalVariable::NotThreadLocal, AS::Threadgroup);
      typedGV->setAlignment(splitGV->getAlign());

      changed |= rewriteByteGEPs(splitGV, typedGV, splitOldAT, typedAT,
                                   elemTy, eSize, Ctx);
      if (splitGV->use_empty()) splitGV->eraseFromParent();
    }
  }
  return changed;
}

// ── 14d: Insert preamble GEPs for array TG globals ─────────────────────
static bool insertPreambleGEPs(Module &M) {
  bool changed = false;
  auto &Ctx = M.getContext();

  SmallVector<GlobalVariable *, 8> allTGGlobals;
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == AS::Threadgroup && isa<ArrayType>(GV.getValueType()))
      allTGGlobals.push_back(&GV);

  for (auto &F : M) {
    if (F.isDeclaration()) continue;

    SmallPtrSet<GlobalVariable *, 4> usedGlobals;
    for (auto &BB : F)
      for (auto &I : BB)
        for (auto &Op : I.operands())
          if (auto *GV = dyn_cast<GlobalVariable>(Op))
            if (GV->getAddressSpace() == AS::Threadgroup)
              usedGlobals.insert(GV);

    if (usedGlobals.empty()) continue;

    DenseMap<GlobalVariable *, Value *> preambleMap;
    for (auto *GV : allTGGlobals) {
      if (!usedGlobals.count(GV)) continue;
      if (!isa<ArrayType>(GV->getValueType())) continue;

      bool needsPreamble = false;
      for (auto *U : GV->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I || I->getFunction() != &F) continue;
        if (auto *GEPUser = dyn_cast<GetElementPtrInst>(U)) {
          if (GEPUser->getSourceElementType() != GV->getValueType())
            needsPreamble = true;
        } else {
          needsPreamble = true;
        }
      }
      if (!needsPreamble) continue;

      auto *AT = cast<ArrayType>(GV->getValueType());
      auto *GEP = GetElementPtrInst::CreateInBounds(
          AT, GV,
          {ConstantInt::get(Type::getInt64Ty(Ctx), 0),
           ConstantInt::get(Type::getInt64Ty(Ctx), 0)},
          "__base_" + GV->getName());
      GEP->insertBefore(F.getEntryBlock().getFirstInsertionPt());
      preambleMap[GV] = GEP;
    }

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getName().starts_with("__base_"))
            continue;
        for (unsigned i = 0; i < I.getNumOperands(); i++) {
          auto *GV = dyn_cast<GlobalVariable>(I.getOperand(i));
          if (!GV) continue;
          auto pit = preambleMap.find(GV);
          if (pit == preambleMap.end()) continue;
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
            if (i == 0 && GEP->getSourceElementType() == GV->getValueType())
              continue;
          I.setOperand(i, pit->second);
          changed = true;
        }
      }
    }
  }
  return changed;
}

// ── 14e: Fix residual i8 GEPs on typed TG pointers ─────────────────────
// After 14a-14d, some i8-source GEPs may remain on pointers derived from
// typed (non-i8) TG GEPs.  Example:
//   %p = gep float, @tg_global, i64 256    ; float-typed base pointer
//   %q = gep i8,    %p,         i32 %off   ; byte-offset on float* -- BAD
// Metal GPU JIT cannot handle this type mismatch. Convert the i8 GEP to
// use the same element type as the producing GEP.
static bool fixResidualI8GEPs(Module &M) {
  bool changed = false;
  auto &DL = M.getDataLayout();

  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<GetElementPtrInst *, 16> i8Geps;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getSourceElementType()->isIntegerTy(8) &&
              GEP->getPointerAddressSpace() == AS::Threadgroup) {
            // Only fix when the pointer comes from a non-i8 typed GEP
            auto *srcGEP = dyn_cast<GetElementPtrInst>(GEP->getPointerOperand());
            if (srcGEP && !srcGEP->getSourceElementType()->isIntegerTy(8))
              i8Geps.push_back(GEP);
          }

    for (auto *GEP : i8Geps) {
      // Use the element type from the source GEP
      auto *srcGEP = cast<GetElementPtrInst>(GEP->getPointerOperand());
      Type *elemTy = srcGEP->getSourceElementType();
      // For array types, get the element type
      if (auto *AT = dyn_cast<ArrayType>(elemTy))
        elemTy = AT->getElementType();
      unsigned elemSize = DL.getTypeAllocSize(elemTy);
      if (elemSize == 0 || elemSize == 1) continue;

      // Convert byte index to element index
      IRBuilder<> B(GEP);
      Value *byteIdx = GEP->getOperand(1);
      Value *elemIdx;
      if (auto *CI = dyn_cast<ConstantInt>(byteIdx))
        elemIdx = ConstantInt::get(CI->getType(), CI->getZExtValue() / elemSize);
      else
        elemIdx = B.CreateUDiv(byteIdx,
                    ConstantInt::get(byteIdx->getType(), elemSize));
      auto *newGEP = B.CreateInBoundsGEP(elemTy, GEP->getPointerOperand(),
                                           elemIdx, GEP->getName());
      GEP->replaceAllUsesWith(newGEP);
      GEP->eraseFromParent();
      changed = true;
    }
  }
  return changed;
}

// ── Main pass: orchestrate sub-passes ──────────────────────────────────
PreservedAnalyses TGGlobalGEPRewritePass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  bool changed = false;

  // Collect byte and MMA globals
  SmallVector<GlobalVariable *, 4> byteGlobals;
  SmallVector<GlobalVariable *, 4> mmaGlobals;
  collectTGByteGlobals(M, byteGlobals);
  collectTGTypedGlobals(M, mmaGlobals);

  // 14a: Split mixed-type byte globals
  changed |= splitMixedByteGlobals(M, byteGlobals);

  // 14b: Merge byte globals into MMA globals
  changed |= mergeByteMMA(M, byteGlobals, mmaGlobals);

  // 14c: Retype remaining [N x i8] → [M x T]
  changed |= retypeByteGlobals(M);

  // 14d: Insert preamble GEPs
  changed |= insertPreambleGEPs(M);

  // 14e: Fix residual i8 GEPs on typed TG pointers
  changed |= fixResidualI8GEPs(M);

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace metalir
