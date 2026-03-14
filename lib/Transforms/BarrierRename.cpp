// Pass 3: Rename air.threadgroup.barrier → air.wg.barrier, fix args (1,4)→(2,1).

#include "metal-ir/Pipeline.h"
#include "metal-ir/AIRIntrinsics.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool BarrierRenamePass::needsRun(Module &M) {
  return M.getFunction(air::kBarrierOld) != nullptr;
}

PreservedAnalyses BarrierRenamePass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  auto *OldBarrier = M.getFunction(air::kBarrierOld);
  if (!OldBarrier) return PreservedAnalyses::all();

  auto *NewBarrier = M.getFunction(air::kBarrier);
  if (!NewBarrier)
    NewBarrier = Function::Create(OldBarrier->getFunctionType(),
                                  OldBarrier->getLinkage(),
                                  air::kBarrier, &M);

  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || CI->getCalledFunction() != OldBarrier)
          continue;
        CI->setCalledFunction(NewBarrier);
        if (CI->arg_size() >= 2) {
          if (auto *C0 = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
            if (C0->getZExtValue() == 1)
              CI->setArgOperand(0, ConstantInt::get(C0->getType(), 2));
          if (auto *C1 = dyn_cast<ConstantInt>(CI->getArgOperand(1)))
            if (C1->getZExtValue() == 4)
              CI->setArgOperand(1, ConstantInt::get(C1->getType(), 1));
        }
      }
    }
  }

  if (OldBarrier->use_empty())
    OldBarrier->eraseFromParent();

  return changedNonCFG();
}

} // namespace metalir
