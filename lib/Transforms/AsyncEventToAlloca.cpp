// ═══════════════════════════════════════════════════════════════════════
// AsyncEventToAlloca — convert __tg_async_events TG global to stack alloca.
//
// The Triton MLIR lowering stores async copy event pointers in a TG global:
//   @__tg_async_events = internal addrspace(3) global [2 x ptr addrspace(3)]
//
// Metal's reference pattern uses a stack alloca instead:
//   %ev = alloca %event_t addrspace(3)*, align 8
//
// The GPU JIT compiler handles the alloca pattern correctly but may reject
// the TG global pattern. This pass converts the TG global to an alloca
// at the start of the kernel function.
//
// Transform:
//   @__tg_async_events = ... global [2 x ptr addrspace(3)] undef
//   %gep = getelementptr [2 x ptr as3], ptr as3 @__tg_async_events, 0, 0
//   store ptr as3 %event, ptr as3 %gep
//   call void @air.wait_simdgroup_events(i32 1, ptr as3 %gep)
//
// Becomes:
//   %ev_alloca = alloca ptr addrspace(3), align 8   ; stack slot for 1 event
//   store ptr as3 %event, ptr %ev_alloca
//   call void @air.wait_simdgroup_events(i32 1, ptr %ev_alloca)
// ═══════════════════════════════════════════════════════════════════════

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

bool AsyncEventToAllocaPass::needsRun(Module &M) {
  // Run if there's an old-style TG global OR if there are async copy calls
  // (need to insert bitcasts for typed-pointer correctness)
  if (M.getNamedGlobal("__tg_async_events"))
    return true;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_async_copy_2d.") ||
        F.getName() == "air.wait_simdgroup_events")
      return true;
  return false;
}

PreservedAnalyses AsyncEventToAllocaPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  bool changed = false;
  auto &Ctx = M.getContext();

  // Part 1: Convert __tg_async_events TG global to stack alloca
  if (auto *GV = M.getNamedGlobal("__tg_async_events")) {
    Function *kernel = nullptr;
    for (auto &F : M)
      if (!F.isDeclaration()) { kernel = &F; break; }
    if (kernel) {
      Type *ptrAs3 = PointerType::get(Ctx, 3);
      IRBuilder<> AllocaBuilder(&kernel->getEntryBlock().front());
      auto *alloca = AllocaBuilder.CreateAlloca(ptrAs3, nullptr, "ev_alloca");
      alloca->setAlignment(Align(8));

      // Replace all uses — the alloca is ptr addrspace(0), not addrspace(3).
      // In opaque-pointer LLVM these are just different ptr types.
      // We do NOT addrspacecast — the alloca stays in addrspace(0) which is
      // correct for Metal event storage (thread-local, not threadgroup).
      SmallVector<User *, 16> users(GV->users().begin(), GV->users().end());
      for (auto *U : users) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          // Replace GEP of global with direct alloca reference
          GEP->replaceAllUsesWith(alloca);
          GEP->eraseFromParent();
        } else if (auto *I = dyn_cast<Instruction>(U)) {
          I->replaceUsesOfWith(GV, alloca);
        }
      }
      GV->eraseFromParent();
      changed = true;
    }
  }

  // Part 2: Insert no-op bitcasts for async copy and wait pointer arguments.
  // The TG global coalescing pass changes [N x i8] → [N/4 x float], which
  // makes GEP results `float addrspace(3)*` in typed-pointer terms. But
  // async copy requires `i8 addrspace(3)*`. A no-op bitcast (ptr→ptr in
  // opaque-pointer IR) creates a new value that InferTypedPointers can
  // assign the correct `i8*` pointee type to.
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end(); ++it) {
        auto *CI = dyn_cast<CallInst>(&*it);
        if (!CI || !CI->getCalledFunction()) continue;
        StringRef name = CI->getCalledFunction()->getName();
        bool isAsyncCopy = name.starts_with("air.simdgroup_async_copy_2d.");
        bool isWaitEvents = (name == "air.wait_simdgroup_events");
        if (!isAsyncCopy && !isWaitEvents) continue;

        for (unsigned i = 0; i < CI->arg_size(); i++) {
          Value *arg = CI->getArgOperand(i);
          if (!arg->getType()->isPointerTy()) continue;
          if (isa<BitCastInst>(arg)) continue;
          // Insert bitcast ptr→ptr before the call
          auto *BC = CastInst::Create(Instruction::BitCast, arg,
                                      arg->getType(), "", CI);
          CI->setArgOperand(i, BC);
          changed = true;
        }
      }
    }
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace metalir
