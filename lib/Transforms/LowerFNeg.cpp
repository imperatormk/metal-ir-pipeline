// Pass 6: fneg → fsub -0.0. Metal GPU JIT doesn't support fneg.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool LowerFNegPass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *U = dyn_cast<UnaryOperator>(&I))
          if (U->getOpcode() == Instruction::FNeg)
            return true;
  return false;
}

PreservedAnalyses LowerFNegPass::run(Module &M, ModuleAnalysisManager &AM) {
  bool changed = false;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        auto *I = &*it++;
        if (auto *Neg = dyn_cast<UnaryOperator>(I)) {
          if (Neg->getOpcode() != Instruction::FNeg)
            continue;
          IRBuilder<> B(Neg);
          Value *Sub = B.CreateFSub(
              ConstantFP::getNegativeZero(Neg->getType()),
              Neg->getOperand(0), Neg->getName());
          Neg->replaceAllUsesWith(Sub);
          Neg->eraseFromParent();
          changed = true;
        }
      }
    }
  }
  return preserveIf(changed);
}

} // namespace metalir
