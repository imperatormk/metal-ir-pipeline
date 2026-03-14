// Pass 13: TGGlobalCoalesce — merge __tg_cvt_* into __tg_dot_ab_* globals.
//
// ConvertLayoutOp creates __tg_cvt_* threadgroup buffers. These don't
// overlap in lifetime with __tg_dot_ab_* (cvt finishes before dot starts,
// or runs after dot completes). Merging saves TG memory.
//
// Only when MMA present (without MMA, cvt and dot overlap).

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"

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
    if (GV.getAddressSpace() != 3) continue;
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
    if (GV.getAddressSpace() != 3) continue;
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

  for (auto *cvt : cvtGlobals) {
    auto *cvtAT = dyn_cast<ArrayType>(cvt->getValueType());
    if (!cvtAT) continue;

    // Find a dot_ab global with matching element type
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
    if (!target) continue;

    // Resize target if cvt is larger
    auto *targetAT = cast<ArrayType>(target->getValueType());
    if (cvtAT->getNumElements() > targetAT->getNumElements()) {
      auto *newAT = ArrayType::get(cvtAT->getElementType(),
                                    cvtAT->getNumElements());
      auto *newGV = new GlobalVariable(
          M, newAT, false, target->getLinkage(),
          UndefValue::get(newAT), target->getName(),
          target, GlobalVariable::NotThreadLocal,
          target->getAddressSpace());
      target->replaceAllUsesWith(newGV);
      target->eraseFromParent();
      target = newGV;
      dotAbGlobals[targetIdx] = newGV;  // update stale pointer
    }

    // Replace all uses of cvt with target
    cvt->replaceAllUsesWith(target);
    cvt->eraseFromParent();
    changed = true;
  }

  return preserveIf(changed);
}

} // namespace metalir
