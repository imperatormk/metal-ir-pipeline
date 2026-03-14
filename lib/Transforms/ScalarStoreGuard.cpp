// Pass 18: ScalarStoreGuard — guard scalar device stores with tid.x == 0.
//
// Kernels that use only threadgroup_position_in_grid (program_id) —
// not thread_position_in_grid or thread_position_in_threadgroup —
// are "scalar" kernels where all threads compute the same result.
// Device stores must be guarded so only thread 0 writes, otherwise
// all threads stomp each other.
//
// Transform:
//   entry:
//     <scalar kernel body with device stores>
// →
//   entry:
//     %tid = call @air.thread_position_in_threadgroup()
//     %tid_x = extractvalue %tid, 0
//     %is_t0 = icmp eq i32 %tid_x, 0
//     br i1 %is_t0, label %body, label %exit
//   body:
//     <original kernel body>
//   exit:
//     ret void
//
// NOTE: This pass changes the CFG (adds basic blocks).

#include "metal-ir/Pipeline.h"
#include "metal-ir/AIRIntrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool ScalarStoreGuardPass::needsRun(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    bool hasDeviceWrite = false;
    bool hasPerThreadIdx = false;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I))
          if (SI->getPointerAddressSpace() == 1)
            hasDeviceWrite = true;
        if (auto *CI = dyn_cast<CallInst>(&I)) {
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
      }
    }
    if (hasDeviceWrite && !hasPerThreadIdx)
      return true;
  }
  return false;
}

PreservedAnalyses ScalarStoreGuardPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  bool changed = false;
  auto &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);

  // Skip if IR already has !air.kernel metadata (pre-lowered)
  if (M.getNamedMetadata(air::kNMDKernel))
    return PreservedAnalyses::all();

  for (auto &F : M) {
    if (F.isDeclaration()) continue;

    // Scan for device writes (stores + atomics) and per-thread index usage
    bool hasDeviceWrite = false;
    bool hasPerThreadIdx = false;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I))
          if (SI->getPointerAddressSpace() == 1)
            hasDeviceWrite = true;
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (auto *Callee = CI->getCalledFunction()) {
            StringRef name = Callee->getName();
            if (name.starts_with(air::kCallTid) ||
                name.starts_with(air::kCallTidTG) ||
                name.starts_with(air::kCallSimdlane))
              hasPerThreadIdx = true;
            // Atomic intrinsics write to device memory
            if (name.starts_with("air.atomic.global"))
              hasDeviceWrite = true;
          }
        }
      }
    }

    if (!hasDeviceWrite || hasPerThreadIdx)
      continue;

    // This is a scalar kernel — add tid.x == 0 guard
    BasicBlock &entry = F.getEntryBlock();

    // Ensure air.thread_position_in_threadgroup is declared
    auto *tidTGTy = ArrayType::get(I32, 3);
    FunctionType *tidFTy = FunctionType::get(tidTGTy, {}, false);
    FunctionCallee tidFC = M.getOrInsertFunction(air::kCallTidTG, tidFTy);

    // Split entry block: everything after the guard goes to "body"
    BasicBlock *bodyBB = entry.splitBasicBlock(entry.getFirstNonPHIIt(), "body");

    // Create exit block with just ret void
    BasicBlock *exitBB = BasicBlock::Create(Ctx, "exit", &F);
    IRBuilder<> exitB(exitBB);
    exitB.CreateRetVoid();

    // Replace entry's unconditional branch (from splitBasicBlock) with guard
    entry.getTerminator()->eraseFromParent();
    IRBuilder<> B(&entry);
    Value *tidResult = B.CreateCall(tidFC, {}, "guard_tid");
    Value *tidX = B.CreateExtractValue(tidResult, {0}, "guard_tid_x");
    Value *isT0 = B.CreateICmpEQ(tidX, ConstantInt::get(I32, 0), "guard_is_t0");
    B.CreateCondBr(isT0, bodyBB, exitBB);

    changed = true;
  }

  if (!changed) return PreservedAnalyses::all();
  // This pass DOES change the CFG
  return PreservedAnalyses::none();
}

} // namespace metalir
