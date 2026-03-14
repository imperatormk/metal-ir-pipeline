// Pass 7: bitcast zeroinitializer → zero of dest type.
// Metal GPU JIT crashes on bitcast <2 x i64> zeroinitializer → <64 x float>.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool BitcastZeroInitPass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *BC = dyn_cast<BitCastInst>(&I))
          if (auto *Op = dyn_cast<Constant>(BC->getOperand(0)))
            if (isa<ConstantAggregateZero>(Op) || Op->isNullValue())
              return true;
  return false;
}

PreservedAnalyses BitcastZeroInitPass::run(Module &M, ModuleAnalysisManager &AM) {
  bool changed = false;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        auto *I = &*it++;
        auto *BC = dyn_cast<BitCastInst>(I);
        if (!BC) continue;
        auto *Op = dyn_cast<Constant>(BC->getOperand(0));
        if (!Op || (!isa<ConstantAggregateZero>(Op) && !Op->isNullValue()))
          continue;
        BC->replaceAllUsesWith(Constant::getNullValue(BC->getType()));
        BC->eraseFromParent();
        changed = true;
      }
    }
  }
  return preserveIf(changed);
}

} // namespace metalir
