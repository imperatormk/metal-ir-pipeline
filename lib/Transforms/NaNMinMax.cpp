// Pass 5: llvm.minimum/maximum → air.fmin/fmax + NaN propagation select.
// llvm.minimum propagates NaN; air.fmin ignores NaN (minnum semantics).

#include "metal-ir/Pipeline.h"
#include "metal-ir/AIRIntrinsics.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;
namespace metalir {

bool NaNMinMaxPass::needsRun(Module &M) {
  for (auto &mapping : air::kNaNMinMax)
    if (M.getFunction(mapping.llvmName))
      return true;
  return false;
}

PreservedAnalyses NaNMinMaxPass::run(Module &M, ModuleAnalysisManager &AM) {
  bool changed = false;

  for (auto &mapping : air::kNaNMinMax) {
    if (auto *F = M.getFunction(mapping.llvmName))
      F->setName(mapping.airName);
  }

  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        auto *CI = dyn_cast<CallInst>(&*it++);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef name = CI->getCalledFunction()->getName();
        bool isMinMax = false;
        for (auto &mapping : air::kNaNMinMax)
          if (name == mapping.airName) { isMinMax = true; break; }
        if (!isMinMax) continue;

        IRBuilder<> B(CI->getNextNode());
        Value *A = CI->getArgOperand(0);
        Value *Bv = CI->getArgOperand(1);
        Value *IsNaN = B.CreateFCmpUNO(A, Bv, "nan_check");
        Value *NaN = ConstantFP::getNaN(CI->getType());
        Value *Sel = B.CreateSelect(IsNaN, NaN, CI, CI->getName() + ".nan");
        CI->replaceAllUsesWith(Sel);
        cast<SelectInst>(Sel)->setOperand(2, CI);
        changed = true;
      }
    }
  }
  return preserveIf(changed);
}

} // namespace metalir
