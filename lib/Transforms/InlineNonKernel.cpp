// Pass 0: InlineNonKernelFunctions — Metal can't call functions, inline all.
//
// Metal kernel bodies must be self-contained — no function calls except
// to AIR intrinsics (which are declarations, not definitions).
// Uses LLVM's InlineFunction() — no manual inlining needed.

#include "metal-ir/Pipeline.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
namespace metalir {

bool InlineNonKernelFunctionsPass::needsRun(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (auto *Callee = CI->getCalledFunction())
            if (!Callee->isDeclaration())
              return true;
  }
  return false;
}

PreservedAnalyses InlineNonKernelFunctionsPass::run(Module &M,
                                                     ModuleAnalysisManager &AM) {
  bool changed = false;
  SmallPtrSet<Function *, 4> wasCallee;

  // Metal has no call stack — ALL non-kernel functions must be inlined.
  // Strip noinline attributes so InlineFunction succeeds even on functions
  // marked noinline by the frontend (e.g. Triton's noinline=True).
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    if (F.hasFnAttribute(Attribute::NoInline))
      F.removeFnAttr(Attribute::NoInline);
  }

  // Repeat until no more inlining (handles nested calls: A→B→C)
  for (unsigned round = 0; round < 20; round++) {
    SmallVector<CallInst *, 16> toInline;
    for (auto &F : M) {
      if (F.isDeclaration()) continue;
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (auto *Callee = CI->getCalledFunction())
              if (!Callee->isDeclaration())
                toInline.push_back(CI);
    }

    if (toInline.empty()) break;

    for (auto *CI : toInline) {
      if (auto *Callee = CI->getCalledFunction())
        wasCallee.insert(Callee);
      InlineFunctionInfo IFI;
      auto result = InlineFunction(*CI, IFI);
      if (result.isSuccess())
        changed = true;
    }
  }

  // Remove functions that were inlined (were callees, now unused).
  // Keep entry points (functions that were never called by others).

  SmallVector<Function *, 4> deadFns;
  for (auto &F : M)
    if (!F.isDeclaration() && F.use_empty() && wasCallee.count(&F))
      deadFns.push_back(&F);
  for (auto *F : deadFns)
    F->eraseFromParent();

  if (!changed) return PreservedAnalyses::all();
  return PreservedAnalyses::none(); // CFG changes from inlining
}

} // namespace metalir
