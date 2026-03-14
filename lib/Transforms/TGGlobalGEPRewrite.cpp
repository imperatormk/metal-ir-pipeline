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
    if (isa<GetElementPtrInst>(U) || isa<GEPOperator>(U))
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

  SmallVector<GlobalVariable *, 4> byteGlobals;
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != 3) continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (AT && AT->getElementType()->isIntegerTy(8))
      byteGlobals.push_back(&GV);
  }

  // Track which globals were fully handled (no preamble needed)
  SmallPtrSet<GlobalVariable *, 4> handledGlobals;

  for (auto *GV : byteGlobals) {
    Type *storeTy = inferTGElementType(GV);
    if (!storeTy) continue;

    expandConstantExprUsers(GV);

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
