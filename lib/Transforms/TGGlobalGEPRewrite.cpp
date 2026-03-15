// Pass 14: TGGlobalGEPRewrite — retype [N x i8] TG globals for Metal.
//
// Triton emits threadgroup memory as [N x i8] with byte-offset GEPs.
// Metal's typed pointer system needs GEP source types to match store/load types.
//
// Three strategies based on what's stored/loaded:
//   A. Scalar stores (float, i32, half): retype global [N x i8] → [M x T],
//      rewrite ALL GEPs to use new typed array
//   B. Vector stores (<4 x i32>, <1 x float>): keep global as [N x i8],
//      rewrite i8 GEPs to use vector source type for correct typed pointer
//   C. Split at offsets: for remaining byte globals with constant-offset GEPs

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool TGGlobalGEPRewritePass::needsRun(Module &M) {
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == 3 && isa<ArrayType>(GV.getValueType()))
      return true;
  return false;
}

/// Walk users transitively to find actual store/load element type.
static Type *inferTGElementType(Value *V) {
  for (auto *U : V->users()) {
    if (auto *SI = dyn_cast<StoreInst>(U))
      if (SI->getPointerOperand() == V)
        return SI->getValueOperand()->getType();
    if (auto *LI = dyn_cast<LoadInst>(U))
      return LI->getType();
    if (isa<GetElementPtrInst>(U) || isa<GEPOperator>(U) || isa<BitCastInst>(U))
      if (Type *T = inferTGElementType(U))
        return T;
  }
  return nullptr;
}

/// Expand all ConstantExpr users of a global into instructions.
static void expandConstantExprUsers(GlobalVariable *GV) {
  SmallVector<std::pair<ConstantExpr *, Instruction *>, 4> toExpand;
  for (auto *U : GV->users()) {
    auto *CE = dyn_cast<ConstantExpr>(U);
    if (!CE) continue;
    for (auto *CEU : CE->users())
      if (auto *I = dyn_cast<Instruction>(CEU))
        toExpand.push_back({CE, I});
  }
  for (auto &[CE, I] : toExpand) {
    auto *Inst = CE->getAsInstruction();
    Inst->insertBefore(I);
    I->replaceUsesOfWith(CE, Inst);
  }
  SmallVector<ConstantExpr *, 4> dead;
  for (auto *U : GV->users())
    if (auto *CE = dyn_cast<ConstantExpr>(U))
      if (CE->use_empty())
        dead.push_back(CE);
  for (auto *CE : dead)
    CE->destroyConstant();
}

/// Recursively collect all i8-source GEPs in the user chain of V.
static void collectI8Geps(Value *V, SmallVectorImpl<GetElementPtrInst *> &out) {
  for (auto *U : V->users()) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (GEP->getSourceElementType()->isIntegerTy(8))
        out.push_back(GEP);
      else
        collectI8Geps(GEP, out);
    }
  }
}

PreservedAnalyses TGGlobalGEPRewritePass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  bool changed = false;
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  Type *I32 = Type::getInt32Ty(Ctx);

  SmallVector<GlobalVariable *, 4> byteGlobals;
  SmallVector<GlobalVariable *, 4> mmaGlobals;
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != 3) continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (!AT) continue;
    if (AT->getElementType()->isIntegerTy(8))
      byteGlobals.push_back(&GV);
    else
      mmaGlobals.push_back(&GV);
  }

  // ── Early split for mixed-type byte globals ────────────────────
  // cummax-style kernels use a single [N x i8] global for two data types
  // (e.g., float values at offset K, i64 indices at offset 0). Split
  // these into separate typed globals BEFORE the merge step, so each
  // region can be merged/retyped independently.
  for (size_t gi = 0; gi < byteGlobals.size(); gi++) {
    auto *GV = byteGlobals[gi];
    expandConstantExprUsers(GV);
    auto *oldAT = cast<ArrayType>(GV->getValueType());
    uint64_t totalBytes = oldAT->getNumElements();

    // Collect store/load types at different constant base offsets
    SmallPtrSet<Type *, 4> allScalarTypes;
    SmallVector<int64_t, 4> constOffsets;
    bool hasNonConstBase = false;
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
            // Dynamic index GEP — walk its users
            collectTypes(GEP, baseOff);
          }
        } else if (isa<BitCastInst>(U)) {
          collectTypes(U, baseOff);
        }
      }
    };
    collectTypes(GV, 0);

    if (allScalarTypes.size() <= 1 || constOffsets.empty()) continue;

    // Mixed types with constant offsets — split
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
          GV, GlobalVariable::NotThreadLocal, 3);
      splitGV->setAlignment(GV->getAlign());
      splitMap[off] = splitGV;
    }

    // Shrink original to first offset
    auto *newAT = ArrayType::get(Type::getInt8Ty(Ctx), constOffsets[0]);
    auto *newGV = new GlobalVariable(
        M, newAT, false, GV->getLinkage(),
        UndefValue::get(newAT), GV->getName().str(),
        GV, GlobalVariable::NotThreadLocal, 3);
    newGV->setAlignment(GV->getAlign());

    // Rewrite GEPs
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
          changed = true;
        } else {
          auto sit = splitMap.find(byteOff);
          if (sit == splitMap.end()) continue;
          GEP->replaceAllUsesWith(sit->second);
          GEP->eraseFromParent();
          changed = true;
        }
      } else {
        // Dynamic GEP at base offset 0
        GEP->setOperand(0, newGV);
        if (GEP->getSourceElementType() == oldAT)
          GEP->setSourceElementType(newAT);
        changed = true;
      }
    }

    if (GV->use_empty()) GV->eraseFromParent();

    // Update byteGlobals: replace old with new + splits
    byteGlobals[gi] = newGV;
    for (auto &kv : splitMap)
      byteGlobals.push_back(kv.second);
    changed = true;
  }

  // ── Byte→MMA merge ─────────────────────────────────────────────
  // When both [N x i8] scratch and [M x float] MMA globals exist,
  // merge into one global. They alias the same memory (barrier-separated).
  // Only when exactly 1 MMA global remains after coalesce.
  // Before merging, scalarize <1 x T> on the byte global so
  // inferTGElementType returns scalar float (enabling the merge).
  // Check for wide vector stores on byte global
  bool hasWideVec = false;
  if (!byteGlobals.empty() && mmaGlobals.size() == 1) {
    auto *byteGV = byteGlobals[0];
    expandConstantExprUsers(byteGV);
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

    // Scalarize <1 x T> for merge candidate only
    SmallVector<Instruction *, 8> vec1Users;
    std::function<void(Value *)> findVec1 = [&](Value *V) {
      for (auto *U : V->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() == V) {
            auto *VT = dyn_cast<FixedVectorType>(SI->getValueOperand()->getType());
            if (VT && VT->getNumElements() == 1) vec1Users.push_back(SI);
          }
        } else if (auto *LI = dyn_cast<LoadInst>(U)) {
          auto *VT = dyn_cast<FixedVectorType>(LI->getType());
          if (VT && VT->getNumElements() == 1) vec1Users.push_back(LI);
        } else if (isa<GetElementPtrInst>(U)) {
          findVec1(U);
        }
      }
    };
    findVec1(byteGV);
    for (auto *I : vec1Users) {
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        IRBuilder<> B(SI);
        Value *scalar = B.CreateExtractElement(SI->getValueOperand(),
            ConstantInt::get(I32, 0));
        B.CreateAlignedStore(scalar, SI->getPointerOperand(),
            SI->getAlign(), SI->isVolatile());
        SI->eraseFromParent();
        changed = true;
      } else if (auto *LI = dyn_cast<LoadInst>(I)) {
        IRBuilder<> B(LI);
        auto *VT = cast<FixedVectorType>(LI->getType());
        auto *scalar = B.CreateAlignedLoad(VT->getElementType(),
            LI->getPointerOperand(), LI->getAlign(), LI->isVolatile());
        Value *vec = B.CreateInsertElement(
            UndefValue::get(VT), scalar, ConstantInt::get(I32, 0));
        LI->replaceAllUsesWith(vec);
        LI->eraseFromParent();
        changed = true;
      }
    }
    // Fold extract(insert(undef, x, 0), 0) → x
    for (auto &F : M) {
      for (auto &BB : F) {
        for (auto it = BB.begin(); it != BB.end();) {
          Instruction &I = *it++;
          if (auto *EE = dyn_cast<ExtractElementInst>(&I)) {
            if (auto *IE = dyn_cast<InsertElementInst>(EE->getVectorOperand())) {
              auto *VT = dyn_cast<FixedVectorType>(IE->getType());
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
    }
  }
  // After early split, try to merge each remaining byte global with
  // the MMA global if types are compatible and it saves TG memory.
  if (!byteGlobals.empty() && mmaGlobals.size() == 1 && !hasWideVec) {
    // Find the best byte global to merge: the one whose inferred type
    // matches the MMA element type, or the largest one.
    auto *mmaGVCandidate = mmaGlobals[0];
    auto *mmaATCandidate = cast<ArrayType>(mmaGVCandidate->getValueType());
    Type *mmaElemTy = mmaATCandidate->getElementType();

    int bestIdx = -1;
    uint64_t bestBytes = 0;
    for (int i = 0; i < (int)byteGlobals.size(); i++) {
      auto *bAT = cast<ArrayType>(byteGlobals[i]->getValueType());
      uint64_t bBytes = bAT->getNumElements();
      // Prefer type-compatible merge (inferred type matches MMA element)
      Type *inferred = inferTGElementType(byteGlobals[i]);
      bool typeMatch = inferred && (inferred == mmaElemTy ||
          (inferred->isIntegerTy(32) && mmaElemTy->isFloatTy()) ||
          (inferred->isFloatTy() && mmaElemTy->isIntegerTy(32)));
      if (typeMatch && bBytes > bestBytes) {
        bestIdx = i;
        bestBytes = bBytes;
      }
    }
    // If no type match found, fall back to largest byte global (original behavior)
    if (bestIdx < 0) {
      for (int i = 0; i < (int)byteGlobals.size(); i++) {
        auto *bAT = cast<ArrayType>(byteGlobals[i]->getValueType());
        if (bAT->getNumElements() > bestBytes) {
          bestBytes = bAT->getNumElements();
          bestIdx = i;
        }
      }
    }
    auto *byteGV = byteGlobals[bestIdx >= 0 ? bestIdx : 0];
    auto *mmaGV = mmaGlobals[0];
    auto *byteAT = cast<ArrayType>(byteGV->getValueType());
    auto *mmaAT = cast<ArrayType>(mmaGV->getValueType());

    uint64_t byteBytes = byteAT->getNumElements();
    unsigned mmaElemSize = DL.getTypeAllocSize(mmaAT->getElementType());
    uint64_t mmaBytes = mmaAT->getNumElements() * mmaElemSize;
    // Use MMA element type for the merged global (may be float, i64, etc.)
    Type *mergeElemTy = mmaElemTy;
    unsigned mergeElemSize = mmaElemSize;
    // Fall back to float if MMA type is unusual
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
        byteGV, GlobalVariable::NotThreadLocal, 3);
    mergedGV->setAlignment(byteGV->getAlign());

    SmallVector<GetElementPtrInst *, 16> byteGEPs;
    for (auto *U : byteGV->users())
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
        byteGEPs.push_back(GEP);

    for (auto *GEP : byteGEPs) {
      if (GEP->getPointerOperand() != byteGV) continue;
      IRBuilder<> B(GEP);
      Type *srcTy = GEP->getSourceElementType();

      if (srcTy == byteAT) {
        Value *byteIdx = GEP->getNumIndices() >= 2
                             ? GEP->getOperand(2)
                             : ConstantInt::get(Type::getInt64Ty(Ctx), 0);
        Value *elemIdx;
        if (auto *CI = dyn_cast<ConstantInt>(byteIdx))
          elemIdx = ConstantInt::get(CI->getType(), CI->getZExtValue() / mergeElemSize);
        else
          elemIdx = B.CreateUDiv(byteIdx, ConstantInt::get(byteIdx->getType(), mergeElemSize));
        auto *newGEP = GetElementPtrInst::CreateInBounds(
            mergedAT, mergedGV,
            {ConstantInt::get(Type::getInt64Ty(Ctx), 0), elemIdx},
            GEP->getName());
        newGEP->insertBefore(B.GetInsertPoint());
        GEP->replaceAllUsesWith(newGEP);
        GEP->eraseFromParent();
      } else if (srcTy->isIntegerTy(8)) {
        Value *byteIdx = GEP->getOperand(1);
        Value *elemIdx;
        if (auto *CI = dyn_cast<ConstantInt>(byteIdx))
          elemIdx = ConstantInt::get(CI->getType(), CI->getZExtValue() / mergeElemSize);
        else
          elemIdx = B.CreateUDiv(byteIdx, ConstantInt::get(byteIdx->getType(), mergeElemSize));
        auto *newGEP = GetElementPtrInst::CreateInBounds(
            mergeElemTy, mergedGV, elemIdx, GEP->getName());
        newGEP->insertBefore(B.GetInsertPoint());
        GEP->replaceAllUsesWith(newGEP);
        GEP->eraseFromParent();
      } else {
        GEP->setOperand(0, mergedGV);
      }
    }

    if (byteGV->use_empty()) byteGV->eraseFromParent();
    mmaGV->replaceAllUsesWith(mergedGV);
    mmaGV->eraseFromParent();
    // Remove merged byte global from list
    if (bestIdx >= 0)
      byteGlobals.erase(byteGlobals.begin() + bestIdx);
    else
      byteGlobals.clear();
    changed = true;
  }

  // Re-collect byte globals after potential merge
  byteGlobals.clear();
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != 3) continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (AT && AT->getElementType()->isIntegerTy(8))
      byteGlobals.push_back(&GV);
  }

  // Track which globals were fully handled (no preamble needed)
  SmallPtrSet<GlobalVariable *, 4> handledGlobals;

  for (auto *GV : byteGlobals) {
    expandConstantExprUsers(GV);

    Type *storeTy = inferTGElementType(GV);
    if (!storeTy) continue;

    // Check ALL store/load types through the global. If there's a mix
    // of scalar and vector (>1 element) types, skip retyping — the typed
    // pointer mismatch causes materializeAll.
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
      // Mixed types: either scalar+vector or multiple different scalar types
      // (e.g., float + i32 in argmin/argmax). All need bitcast insertion.
      bool isMixed = storeTypes.size() > 1;
      if (isMixed) {
        // Mixed types: insert bitcast ptr→ptr before each store/load
        // whose type differs from the GEP source type (i8).
        // In Metal v1 bitcode these bitcasts change the typed pointer.
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

    // Scalarize <1 x T> store/load for THIS global before retyping.
    // After retype, pointer is typed as T* — <1 x T> through T* crashes.
    {
      SmallVector<Instruction *, 8> vec1Users;
      std::function<void(Value *)> findVec1 = [&](Value *V) {
        for (auto *U : V->users()) {
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            if (SI->getPointerOperand() == V) {
              auto *VT = dyn_cast<FixedVectorType>(SI->getValueOperand()->getType());
              if (VT && VT->getNumElements() == 1) vec1Users.push_back(SI);
            }
          } else if (auto *LI = dyn_cast<LoadInst>(U)) {
            auto *VT = dyn_cast<FixedVectorType>(LI->getType());
            if (VT && VT->getNumElements() == 1) vec1Users.push_back(LI);
          } else if (isa<GetElementPtrInst>(U)) {
            findVec1(U);
          }
        }
      };
      findVec1(GV);
      for (auto *I : vec1Users) {
        if (auto *SI = dyn_cast<StoreInst>(I)) {
          IRBuilder<> B(SI);
          Value *scalar = B.CreateExtractElement(SI->getValueOperand(),
              ConstantInt::get(I32, 0));
          B.CreateAlignedStore(scalar, SI->getPointerOperand(),
              SI->getAlign(), SI->isVolatile());
          SI->eraseFromParent();
          changed = true;
        } else if (auto *LI = dyn_cast<LoadInst>(I)) {
          IRBuilder<> B(LI);
          auto *VT = cast<FixedVectorType>(LI->getType());
          auto *scalar = B.CreateAlignedLoad(VT->getElementType(),
              LI->getPointerOperand(), LI->getAlign(), LI->isVolatile());
          Value *vec = B.CreateInsertElement(
              UndefValue::get(VT), scalar, ConstantInt::get(I32, 0));
          LI->replaceAllUsesWith(vec);
          LI->eraseFromParent();
          changed = true;
        }
      }
    }
    // Fold extract(insert(undef, x, 0), 0) → x
    for (auto &F : M) {
      for (auto &BB : F) {
        for (auto it = BB.begin(); it != BB.end();) {
          Instruction &I = *it++;
          if (auto *EE = dyn_cast<ExtractElementInst>(&I)) {
            if (auto *IE = dyn_cast<InsertElementInst>(EE->getVectorOperand())) {
              auto *VT = dyn_cast<FixedVectorType>(IE->getType());
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
    }

    // Re-infer storeTy after scalarization (<1 x float> → float)
    storeTy = inferTGElementType(GV);
    if (!storeTy) continue;

    auto *oldAT = cast<ArrayType>(GV->getValueType());
    uint64_t totalBytes = oldAT->getNumElements();

    // ── Retype global: [N x i8] → [M x T] where T = storeTy ─────
    {
      Type *elemTy = storeTy;
      if (elemTy->isBFloatTy())
        elemTy = Type::getHalfTy(Ctx);

      unsigned elemSize = DL.getTypeAllocSize(elemTy);
      if (elemSize == 0) continue;
      uint64_t numElems = totalBytes / elemSize;
      if (numElems == 0) continue;

      auto *newAT = ArrayType::get(elemTy, numElems);
      auto *newGV = new GlobalVariable(
          M, newAT, GV->isConstant(), GV->getLinkage(),
          UndefValue::get(newAT), GV->getName() + ".typed",
          GV, GlobalVariable::NotThreadLocal, 3);
      newGV->setAlignment(GV->getAlign());

      // Rewrite all GEP users
      SmallVector<GetElementPtrInst *, 16> users;
      for (auto *U : GV->users())
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
          users.push_back(GEP);

      for (auto *GEP : users) {
        if (GEP->getPointerOperand() != GV) continue;
        Type *srcTy = GEP->getSourceElementType();
        IRBuilder<> B(GEP);

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
          // Already-typed GEP: just redirect pointer to new global
          GEP->setOperand(0, newGV);
        }
        changed = true;
      }

      // Redirect remaining direct (non-GEP) instruction uses of the old
      // global to the new typed global. These are loads/stores at offset 0
      // that don't go through a GEP (e.g., `load i32, @global_smem`).
      {
        SmallVector<Instruction *, 4> directUsers;
        for (auto *U : GV->users()) {
          auto *I = dyn_cast<Instruction>(U);
          if (!I) continue;
          if (isa<GetElementPtrInst>(I)) continue;  // already handled
          directUsers.push_back(I);
        }
        for (auto *I : directUsers) {
          for (unsigned op = 0; op < I->getNumOperands(); op++) {
            if (I->getOperand(op) == GV)
              I->setOperand(op, newGV);
          }
          changed = true;
        }
      }

      if (GV->use_empty()) GV->eraseFromParent();

      // Clean up any remaining i8 GEPs in the chain of the new global.
      // These can appear from constant expr expansion leaving chained i8 GEPs.
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
      // newGV still needs preamble (gep [M x T], @new, 0, 0 → T*)
    }
  }

  // ── Strategy C: Split remaining byte globals at offsets ──────────
  SmallVector<GlobalVariable *, 4> remainingByteGlobals;
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != 3) continue;
    if (handledGlobals.count(&GV)) continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (AT && AT->getElementType()->isIntegerTy(8))
      remainingByteGlobals.push_back(&GV);
  }

  for (auto *GV : remainingByteGlobals) {
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

    // Create split globals
    DenseMap<int64_t, GlobalVariable *> splitMap;
    for (int64_t off : offsets) {
      uint64_t regionSize = totalBytes - off;
      if (regionSize == 0) continue;
      auto *splitAT = ArrayType::get(Type::getInt8Ty(Ctx), regionSize);
      auto *splitGV = new GlobalVariable(
          M, splitAT, false, GV->getLinkage(),
          UndefValue::get(splitAT),
          GV->getName() + "__off" + Twine(off),
          GV, GlobalVariable::NotThreadLocal, 3);
      splitGV->setAlignment(GV->getAlign());
      splitMap[off] = splitGV;
    }

    // Resize original
    auto *newAT = ArrayType::get(Type::getInt8Ty(Ctx), offsets[0]);
    auto *newGV = new GlobalVariable(
        M, newAT, false, GV->getLinkage(),
        UndefValue::get(newAT), GV->getName().str(),
        GV, GlobalVariable::NotThreadLocal, 3);
    newGV->setAlignment(GV->getAlign());

    // Rewrite GEPs
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

    // Retype split globals: [N x i8] → [M x T] based on store/load types.
    // This enables proper typed pointers in Metal bitcode.
    SmallVector<GlobalVariable *, 4> toRetype;
    toRetype.push_back(newGV);
    for (auto &kv : splitMap)
      toRetype.push_back(kv.second);
    for (auto *splitGV : toRetype) {
      Type *elemTy = inferTGElementType(splitGV);
      if (!elemTy) continue;
      if (elemTy->isBFloatTy()) elemTy = Type::getHalfTy(Ctx);
      unsigned eSize = DL.getTypeAllocSize(elemTy);
      if (eSize == 0) continue;
      auto *splitOldAT = cast<ArrayType>(splitGV->getValueType());
      uint64_t nBytes = splitOldAT->getNumElements();
      uint64_t nElems = nBytes / eSize;
      if (nElems == 0) continue;
      auto *typedAT = ArrayType::get(elemTy, nElems);
      auto *typedGV = new GlobalVariable(
          M, typedAT, false, splitGV->getLinkage(),
          UndefValue::get(typedAT), splitGV->getName().str() + ".typed",
          splitGV, GlobalVariable::NotThreadLocal, 3);
      typedGV->setAlignment(splitGV->getAlign());

      SmallVector<GetElementPtrInst *, 8> splitUsers;
      for (auto *U : splitGV->users())
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
          splitUsers.push_back(GEP);
      for (auto *GEP : splitUsers) {
        if (GEP->getPointerOperand() != splitGV) continue;
        IRBuilder<> B(GEP);
        Type *srcTy = GEP->getSourceElementType();
        if (srcTy == splitOldAT) {
          Value *byteIdx = GEP->getNumIndices() >= 2
                               ? GEP->getOperand(2)
                               : ConstantInt::get(Type::getInt64Ty(Ctx), 0);
          Value *eIdx;
          if (auto *CI = dyn_cast<ConstantInt>(byteIdx))
            eIdx = ConstantInt::get(CI->getType(), CI->getZExtValue() / eSize);
          else
            eIdx = B.CreateUDiv(byteIdx, ConstantInt::get(byteIdx->getType(), eSize));
          auto *nGEP = GetElementPtrInst::CreateInBounds(
              typedAT, typedGV,
              {ConstantInt::get(Type::getInt64Ty(Ctx), 0), eIdx},
              GEP->getName());
          nGEP->insertBefore(B.GetInsertPoint());
          GEP->replaceAllUsesWith(nGEP);
          GEP->eraseFromParent();
        } else if (srcTy->isIntegerTy(8)) {
          Value *byteIdx = GEP->getOperand(1);
          Value *eIdx;
          if (auto *CI = dyn_cast<ConstantInt>(byteIdx))
            eIdx = ConstantInt::get(CI->getType(), CI->getZExtValue() / eSize);
          else
            eIdx = B.CreateUDiv(byteIdx, ConstantInt::get(byteIdx->getType(), eSize));
          auto *nGEP = GetElementPtrInst::CreateInBounds(
              elemTy, typedGV, eIdx, GEP->getName());
          nGEP->insertBefore(B.GetInsertPoint());
          GEP->replaceAllUsesWith(nGEP);
          GEP->eraseFromParent();
        } else {
          // Non-i8, non-array GEP: redirect pointer
          GEP->setOperand(0, typedGV);
        }
        changed = true;
      }
      // Redirect any remaining direct users (non-GEP)
      SmallVector<Instruction *, 4> directUsers;
      for (auto *U : splitGV->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I || isa<GetElementPtrInst>(I)) continue;
        directUsers.push_back(I);
      }
      for (auto *I : directUsers) {
        for (unsigned op = 0; op < I->getNumOperands(); op++)
          if (I->getOperand(op) == splitGV)
            I->setOperand(op, typedGV);
        changed = true;
      }
      if (splitGV->use_empty()) splitGV->eraseFromParent();
    }
  }

  // ── Phase 2: Preamble GEPs for array TG globals ─────────────────
  SmallVector<GlobalVariable *, 8> allTGGlobals;
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == 3 && isa<ArrayType>(GV.getValueType()))
      if (!handledGlobals.count(&GV))
        allTGGlobals.push_back(&GV);

  for (auto &F : M) {
    if (F.isDeclaration()) continue;

    SmallPtrSet<GlobalVariable *, 4> usedGlobals;
    for (auto &BB : F)
      for (auto &I : BB)
        for (auto &Op : I.operands())
          if (auto *GV = dyn_cast<GlobalVariable>(Op))
            if (GV->getAddressSpace() == 3 && !handledGlobals.count(GV))
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

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace metalir
