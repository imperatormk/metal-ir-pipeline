// NormalizeI1Pointers — rewrite i1 GEPs to i8 GEPs.
//
// Metal GPU JIT has no i1 memory type. GEPs with i1 source type
// (e.g., `getelementptr i1, ptr %p, i32 %idx`) crash the JIT.
// Rewrite to `getelementptr i8, ptr %p, i32 %idx` which has
// the same byte offset since i1 is 1-byte aligned in memory.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace metalir {

bool NormalizeI1PointersPass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (GEP->getSourceElementType()->isIntegerTy(1))
            return true;
  return false;
}

PreservedAnalyses NormalizeI1PointersPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  bool changed = false;
  Type *I8 = Type::getInt8Ty(M.getContext());

  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *GEP = dyn_cast<GetElementPtrInst>(&I);
        if (!GEP || !GEP->getSourceElementType()->isIntegerTy(1))
          continue;

        // Replace: gep i1, ptr %p, idx → gep i8, ptr %p, idx
        // Same byte offset since data layout has i1:8:8
        GEP->setSourceElementType(I8);
        GEP->setResultElementType(I8);
        changed = true;
      }
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
