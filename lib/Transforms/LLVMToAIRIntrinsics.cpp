// Pass 8: Rename LLVM intrinsic declarations to AIR equivalents,
// and inline software implementations for unsupported intrinsics.

#include "metal-ir/Pipeline.h"
#include "metal-ir/AIRIntrinsics.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;
namespace metalir {

bool LLVMToAIRIntrinsicsPass::needsRun(Module &M) {
  for (auto &mapping : air::kIntrinsicRenames)
    if (M.getFunction(mapping.llvmName))
      return true;
  if (M.getFunction("__mulhi")) return true;
  return false;
}

// Lower __mulhi(a, b) → (i32)((zext64(a) * zext64(b)) >> 32)
static bool lowerMulHi(Module &M) {
  auto *F = M.getFunction("__mulhi");
  if (!F) return false;

  auto *I32 = Type::getInt32Ty(M.getContext());
  auto *I64 = Type::getInt64Ty(M.getContext());

  SmallVector<CallInst *, 8> calls;
  for (auto *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      calls.push_back(CI);

  for (auto *CI : calls) {
    IRBuilder<> B(CI);
    Value *a = B.CreateZExt(CI->getArgOperand(0), I64);
    Value *b = B.CreateZExt(CI->getArgOperand(1), I64);
    Value *mul = B.CreateMul(a, b);
    Value *hi = B.CreateLShr(mul, 32);
    Value *result = B.CreateTrunc(hi, I32);
    CI->replaceAllUsesWith(result);
    CI->eraseFromParent();
  }

  if (F->use_empty())
    F->eraseFromParent();
  return true;
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
  changed |= lowerMulHi(M);
  return preserveIf(changed);
}

} // namespace metalir
