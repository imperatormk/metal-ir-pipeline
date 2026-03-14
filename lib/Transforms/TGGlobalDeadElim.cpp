// Pass 12: Remove unreferenced addrspace(3) globals.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;
namespace metalir {

bool TGGlobalDeadElimPass::needsRun(Module &M) {
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == 3 && GV.use_empty())
      return true;
  return false;
}

PreservedAnalyses TGGlobalDeadElimPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  bool changed = false;
  SmallVector<GlobalVariable *, 4> dead;
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == 3 && GV.use_empty())
      dead.push_back(&GV);
  for (auto *GV : dead) {
    GV->eraseFromParent();
    changed = true;
  }
  return preserveIf(changed);
}

} // namespace metalir
