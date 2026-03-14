// Pass 17: Decompose bf16 casts via float intermediate.
// Metal GPU JIT treats sitofp iN→bfloat as sitofp iN→half (wrong).

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool BFloat16CastDecomposePass::needsRun(Module &M) {
  Type *BF16 = Type::getBFloatTy(M.getContext());
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if ((isa<SIToFPInst>(&I) || isa<UIToFPInst>(&I)) && I.getType() == BF16)
          return true;
  return false;
}

PreservedAnalyses BFloat16CastDecomposePass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  bool changed = false;
  Type *BF16 = Type::getBFloatTy(M.getContext());
  Type *F32 = Type::getFloatTy(M.getContext());

  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        auto *I = &*it++;
        if ((isa<SIToFPInst>(I) || isa<UIToFPInst>(I)) && I->getType() == BF16) {
          IRBuilder<> B(I);
          Value *ToFloat = isa<SIToFPInst>(I)
              ? B.CreateSIToFP(I->getOperand(0), F32, "to_f32")
              : B.CreateUIToFP(I->getOperand(0), F32, "to_f32");
          Value *Trunc = B.CreateFPTrunc(ToFloat, BF16, I->getName());
          I->replaceAllUsesWith(Trunc);
          I->eraseFromParent();
          changed = true;
        }
      }
    }
  }
  return preserveIf(changed);
}

} // namespace metalir
