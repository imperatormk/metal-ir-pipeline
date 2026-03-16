// Pass 18: ScalarStoreGuard — guard scalar device stores with tid.x == 0.
//
// Scalar kernels (device writes, no per-thread indexing) need a tid==0
// guard so only one thread writes. Uses KernelProfile to determine
// which functions are scalar kernels instead of scanning inline.

#include "metal-ir/Pipeline.h"
#include "metal-ir/AIRIntrinsics.h"
#include "metal-ir/IRUtil.h"
#include "metal-ir/KernelProfile.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool ScalarStoreGuardPass::needsRun(Module &M) {
  // Quick check without full analysis — any function with device writes
  // and no per-thread index calls
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    bool hasDeviceWrite = false, hasPerThreadIdx = false;
    for (auto &BB : F)
      for (auto &I : BB) {
        if (isDeviceStore(&I)) hasDeviceWrite = true;
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (auto *Callee = CI->getCalledFunction()) {
            StringRef name = Callee->getName();
            if (name.starts_with(air::kCallTid) ||
                name.starts_with(air::kCallTidTG) ||
                name.starts_with(air::kCallSimdlane))
              hasPerThreadIdx = true;
            if (name.starts_with("air.atomic.global"))
              hasDeviceWrite = true;
          }
      }
    if (hasDeviceWrite && !hasPerThreadIdx) return true;
  }
  return false;
}

PreservedAnalyses ScalarStoreGuardPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  // Skip if IR already has !air.kernel metadata (pre-lowered)
  if (M.getNamedMetadata(air::kNMDKernel))
    return PreservedAnalyses::all();

  auto &profiles = AM.getResult<KernelProfileAnalysis>(M);

  bool changed = false;
  auto &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);

  for (auto &F : M) {
    if (F.isDeclaration()) continue;

    auto it = profiles.find(&F);
    if (it == profiles.end()) continue;
    auto &facts = it->second;

    if (!facts.isScalarKernel()) continue;

    // Add tid.x == 0 guard
    BasicBlock &entry = F.getEntryBlock();
    auto *tidTGTy = ArrayType::get(I32, 3);
    FunctionType *tidFTy = FunctionType::get(tidTGTy, {}, false);
    FunctionCallee tidFC = M.getOrInsertFunction(air::kCallTidTG, tidFTy);

    BasicBlock *bodyBB = entry.splitBasicBlock(entry.getFirstNonPHIIt(), "body");
    BasicBlock *exitBB = BasicBlock::Create(Ctx, "exit", &F);
    IRBuilder<> exitB(exitBB);
    exitB.CreateRetVoid();

    entry.getTerminator()->eraseFromParent();
    IRBuilder<> B(&entry);
    Value *tidResult = B.CreateCall(tidFC, {}, "guard_tid");
    Value *tidX = B.CreateExtractValue(tidResult, {0}, "guard_tid_x");
    Value *isT0 = B.CreateICmpEQ(tidX, ConstantInt::get(I32, 0), "guard_is_t0");
    B.CreateCondBr(isT0, bodyBB, exitBB);

    changed = true;
  }

  if (!changed) return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

} // namespace metalir
