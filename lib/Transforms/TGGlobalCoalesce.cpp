// Pass 13: TGGlobalCoalesce — merge __tg_cvt_* into __tg_dot_ab_* globals.
//
// ConvertLayoutOp creates __tg_cvt_* threadgroup buffers. These don't
// overlap in lifetime with __tg_dot_ab_* (cvt finishes before dot starts,
// or runs after dot completes). Merging saves TG memory.
//
// When element types differ (e.g. i32 vs float) but have the same byte
// size, the cvt stores/loads are rewritten with bitcasts to match the
// dot_ab element type. This is critical because Metal GPU JIT crashes
// when a kernel has multiple TG globals with MMA intrinsics.
//
// Only when MMA present (without MMA, cvt and dot overlap).

#include "metal-ir/Pipeline.h"
#include "metal-ir/MetalConstraints.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool TGGlobalCoalescePass::needsRun(Module &M) {
  // Need MMA + both cvt and dot globals
  bool hasMMA = false, hasCvt = false, hasDot = false;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      hasMMA = true;
  if (!hasMMA) return false;
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != AS::Threadgroup) continue;
    if (GV.getName().starts_with("__tg_cvt_")) hasCvt = true;
    if (GV.getName().starts_with("__tg_dot_")) hasDot = true;
  }
  return hasCvt && hasDot;
}

PreservedAnalyses TGGlobalCoalescePass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  auto &MMA = MAM.getResult<MMAPresenceAnalysis>(M);
  if (!MMA.hasMMA) return PreservedAnalyses::all();

  bool changed = false;

  // Collect cvt and dot globals
  SmallVector<GlobalVariable *, 4> cvtGlobals;
  SmallVector<GlobalVariable *, 4> dotAbGlobals;

  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != AS::Threadgroup) continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (!AT || AT->getNumElements() <= 64) continue;

    if (GV.getName().starts_with("__tg_cvt_"))
      cvtGlobals.push_back(&GV);
    else if (GV.getName().starts_with("__tg_dot_") &&
             GV.getName().contains("_ab_"))
      dotAbGlobals.push_back(&GV);
  }

  if (cvtGlobals.empty() || dotAbGlobals.empty())
    return PreservedAnalyses::all();

  auto &DL = M.getDataLayout();

  for (auto *cvt : cvtGlobals) {
    auto *cvtAT = dyn_cast<ArrayType>(cvt->getValueType());
    if (!cvtAT) continue;

    // Find a dot_ab global to merge into.
    // Prefer matching element type; fall back to same-size element type.
    GlobalVariable *target = nullptr;
    size_t targetIdx = 0;
    for (size_t i = 0; i < dotAbGlobals.size(); i++) {
      auto *dot = dotAbGlobals[i];
      auto *dotAT = dyn_cast<ArrayType>(dot->getValueType());
      if (!dotAT) continue;
      if (dotAT->getElementType() == cvtAT->getElementType()) {
        target = dot;
        targetIdx = i;
        break;
      }
    }

    // If no exact match, try same-size element types (e.g., i32 and float)
    bool needsBitcast = false;
    if (!target) {
      unsigned cvtElemSize = DL.getTypeSizeInBits(cvtAT->getElementType());
      for (size_t i = 0; i < dotAbGlobals.size(); i++) {
        auto *dot = dotAbGlobals[i];
        auto *dotAT = dyn_cast<ArrayType>(dot->getValueType());
        if (!dotAT) continue;
        unsigned dotElemSize = DL.getTypeSizeInBits(dotAT->getElementType());
        if (dotElemSize == cvtElemSize) {
          target = dot;
          targetIdx = i;
          needsBitcast = true;
          break;
        }
      }
    }

    if (!target) continue;

    auto *targetAT = cast<ArrayType>(target->getValueType());
    Type *targetElemTy = targetAT->getElementType();

    // Resize target if cvt is larger (in elements — same element size)
    if (cvtAT->getNumElements() > targetAT->getNumElements()) {
      auto *newAT = ArrayType::get(targetElemTy,
                                    cvtAT->getNumElements());
      auto *newGV = new GlobalVariable(
          M, newAT, false, target->getLinkage(),
          UndefValue::get(newAT), target->getName(),
          target, GlobalVariable::NotThreadLocal,
          target->getAddressSpace());
      target->replaceAllUsesWith(newGV);
      target->eraseFromParent();
      target = newGV;
      dotAbGlobals[targetIdx] = newGV;
    }

    if (needsBitcast) {
      // Rewrite cvt users: change GEP source type and add bitcasts for
      // stores/loads through the cvt global.
      // In opaque-pointer LLVM, GEPs use an explicit source element type
      // independent of the pointer type. But when we merge cvt into target,
      // the element stride must match. Since elem sizes are equal, we can
      // just replace the global and insert bitcasts at store/load sites.

      // Collect all GEPs of cvt to rewrite
      SmallVector<GetElementPtrInst *, 16> gepsToRewrite;
      for (auto *U : cvt->users())
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
          gepsToRewrite.push_back(GEP);

      for (auto *GEP : gepsToRewrite) {
        // Create new GEP with target's element type but same indices
        IRBuilder<> B(GEP);
        SmallVector<Value *, 4> indices;
        for (auto &Idx : GEP->indices())
          indices.push_back(Idx);

        Value *newGEP;
        if (GEP->getSourceElementType() == cvtAT) {
          // GEP into [N x i32] — replace with [N x float]
          newGEP = B.CreateGEP(target->getValueType(), target, indices,
                                GEP->getName(), GEP->isInBounds());
        } else {
          // GEP into i32 — replace with float
          // First GEP to get base of target, then element GEP
          newGEP = B.CreateGEP(targetElemTy, target, indices,
                                GEP->getName(), GEP->isInBounds());
        }

        // Rewrite users of this GEP's result: insert bitcasts at stores/loads
        SmallVector<Instruction *, 8> users;
        for (auto *U : GEP->users())
          users.push_back(cast<Instruction>(U));

        for (auto *U : users) {
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            if (SI->getPointerOperand() == GEP) {
              // store i32 %v, ptr %gep → store float (bitcast %v), ptr %newGEP
              IRBuilder<> SB(SI);
              Value *val = SI->getValueOperand();
              Value *cast = SB.CreateBitCast(val, targetElemTy, val->getName() + "_bc");
              SB.CreateAlignedStore(cast, newGEP, SI->getAlign(), SI->isVolatile());
              SI->eraseFromParent();
            }
          } else if (auto *LI = dyn_cast<LoadInst>(U)) {
            // load i32, ptr %gep → bitcast (load float, ptr %newGEP) to i32
            IRBuilder<> LB(LI);
            auto *newLoad = LB.CreateAlignedLoad(targetElemTy, newGEP,
                                                   LI->getAlign(),
                                                   LI->getName() + "_fl");
            if (LI->isVolatile()) newLoad->setVolatile(true);
            Value *cast = LB.CreateBitCast(newLoad, LI->getType(), LI->getName());
            LI->replaceAllUsesWith(cast);
            LI->eraseFromParent();
          } else {
            // Other users (unlikely): just use the new GEP
            U->replaceUsesOfWith(GEP, newGEP);
          }
        }

        if (GEP->use_empty())
          GEP->eraseFromParent();
      }
    } else {
      // Same element type — simple replacement
      cvt->replaceAllUsesWith(target);
    }

    cvt->eraseFromParent();
    changed = true;
  }

  return preserveIf(changed);
}

} // namespace metalir
