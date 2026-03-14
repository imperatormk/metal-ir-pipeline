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

  // Phase 1: Decompose sitofp/uitofp iN→bfloat via float intermediate
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

  // Phase 2: Lower sitofp/uitofp i8/i16→float to air.convert intrinsics.
  // Metal's bitcode doesn't support LLVM sitofp for sub-32-bit integer types.
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        auto *I = &*it++;
        bool isSigned = isa<SIToFPInst>(I);
        if (!isSigned && !isa<UIToFPInst>(I)) continue;
        if (I->getType() != F32) continue;

        Type *srcTy = I->getOperand(0)->getType();
        unsigned bits = srcTy->getIntegerBitWidth();
        if (bits >= 32) continue; // i32→float is fine as sitofp

        // Build intrinsic name: air.convert.f.f32.{s|u}.i{8|16}
        std::string name = "air.convert.f.f32.";
        name += isSigned ? "s" : "u";
        name += ".i" + std::to_string(bits);

        auto *ConvFn = M.getOrInsertFunction(
            name, FunctionType::get(F32, {srcTy}, false)).getCallee();

        IRBuilder<> B(I);
        auto *Call = B.CreateCall(
            cast<Function>(ConvFn)->getFunctionType(),
            ConvFn, {I->getOperand(0)}, I->getName());
        I->replaceAllUsesWith(Call);
        I->eraseFromParent();
        changed = true;
      }
    }
  }
  return preserveIf(changed);
}

} // namespace metalir
