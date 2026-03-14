// Pass 8: Rename LLVM intrinsic declarations to AIR equivalents.

#include "metal-ir/Pipeline.h"
#include "metal-ir/AIRIntrinsics.h"
#include "metal-ir/PassUtil.h"

using namespace llvm;
namespace metalir {

bool LLVMToAIRIntrinsicsPass::needsRun(Module &M) {
  for (auto &mapping : air::kIntrinsicRenames)
    if (M.getFunction(mapping.llvmName))
      return true;
  return false;
}

PreservedAnalyses LLVMToAIRIntrinsicsPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  bool changed = false;
  for (auto &mapping : air::kIntrinsicRenames) {
    if (auto *F = M.getFunction(mapping.llvmName)) {
      F->setName(mapping.airName);
      changed = true;
    }
  }
  return preserveIf(changed);
}

} // namespace metalir
