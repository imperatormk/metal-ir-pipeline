// Pass 10: SplitI64Shuffle — i64 simd_shuffle → 2× i32 shuffles.
// Metal GPU JIT crashes on i64 simd_shuffle. Split into lo/hi i32 halves.
//
// call i64 @air.simd_shuffle.i64(i64 %val, i16 %offset)
// →
// %lo = trunc i64 %val to i32
// %hi = trunc i64 (lshr %val, 32) to i32
// %shuf_lo = call i32 @air.simd_shuffle.i32(i32 %lo, i16 %offset)
// %shuf_hi = call i32 @air.simd_shuffle.i32(i32 %hi, i16 %offset)
// %result = or i64 (zext %shuf_lo), (shl (zext %shuf_hi), 32)

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool SplitI64ShufflePass::needsRun(Module &M) {
  for (auto &F : M)
    if (F.isDeclaration() && F.getName().starts_with("air.simd_shuffle") &&
        F.getName().ends_with(".i64"))
      return true;
  return false;
}

PreservedAnalyses SplitI64ShufflePass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  bool changed = false;
  auto &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I16 = Type::getInt16Ty(Ctx);

  // Collect i64 shuffle calls
  SmallVector<CallInst *, 16> calls;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (auto *Callee = CI->getCalledFunction())
            if (Callee->getName().starts_with("air.simd_shuffle") &&
                Callee->getName().ends_with(".i64"))
              calls.push_back(CI);

  for (auto *CI : calls) {
    // Get or create the i32 variant
    std::string i32Name = CI->getCalledFunction()->getName().str();
    // Replace ".i64" suffix with ".i32"
    i32Name.replace(i32Name.size() - 4, 4, ".i32");

    FunctionType *i32FTy = FunctionType::get(I32, {I32, I16}, false);
    FunctionCallee i32FC = M.getOrInsertFunction(i32Name, i32FTy);

    IRBuilder<> B(CI);
    Value *val = CI->getArgOperand(0);   // i64 value
    Value *offset = CI->getArgOperand(1); // i16 offset

    // Split i64 → lo/hi i32
    Value *lo = B.CreateTrunc(val, I32, CI->getName() + "_lo");
    Value *hiPre = B.CreateLShr(val, ConstantInt::get(I64, 32),
                                 CI->getName() + "_hi_pre");
    Value *hi = B.CreateTrunc(hiPre, I32, CI->getName() + "_hi");

    // Shuffle both halves
    Value *shufLo = B.CreateCall(i32FC, {lo, offset}, CI->getName() + "_slo");
    Value *shufHi = B.CreateCall(i32FC, {hi, offset}, CI->getName() + "_shi");

    // Recombine: result = zext(shuf_lo) | (zext(shuf_hi) << 32)
    Value *loExt = B.CreateZExt(shufLo, I64, CI->getName() + "_loext");
    Value *hiExt = B.CreateZExt(shufHi, I64, CI->getName() + "_hiext");
    Value *hiShl = B.CreateShl(hiExt, ConstantInt::get(I64, 32),
                                CI->getName() + "_hishl");
    Value *result = B.CreateOr(loExt, hiShl, CI->getName());

    CI->replaceAllUsesWith(result);
    CI->eraseFromParent();
    changed = true;
  }

  // Remove i64 shuffle declarations if unused
  SmallVector<Function *, 4> deadDecls;
  for (auto &F : M)
    if (F.isDeclaration() && F.getName().starts_with("air.simd_shuffle") &&
        F.getName().ends_with(".i64") && F.use_empty())
      deadDecls.push_back(&F);
  for (auto *F : deadDecls)
    F->eraseFromParent();

  return preserveIf(changed);
}

} // namespace metalir
