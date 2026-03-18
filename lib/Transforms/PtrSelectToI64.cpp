// Pass 2b: PtrSelectToI64 — convert pointer selects to i64 selects.
// Metal GPU JIT crashes on `select` between typed pointers.
// Lower to: ptrtoint → select i64 → inttoptr.
//
// select i1 %cond, ptr addrspace(1) %a, ptr addrspace(1) %b
// →
// %a_i64 = ptrtoint ptr addrspace(1) %a to i64
// %b_i64 = ptrtoint ptr addrspace(1) %b to i64
// %sel_i64 = select i1 %cond, i64 %a_i64, i64 %b_i64
// %result = inttoptr i64 %sel_i64 to ptr addrspace(1)

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool PtrSelectToI64Pass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *Sel = dyn_cast<SelectInst>(&I))
          if (Sel->getType()->isPointerTy())
            return true;
  return false;
}

PreservedAnalyses PtrSelectToI64Pass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  bool changed = false;
  Type *I64 = Type::getInt64Ty(M.getContext());

  for (auto &F : M) {
    SmallVector<SelectInst *, 8> ptrSelects;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *Sel = dyn_cast<SelectInst>(&I))
          if (Sel->getType()->isPointerTy())
            ptrSelects.push_back(Sel);

    for (auto *Sel : ptrSelects) {
      IRBuilder<> B(Sel);
      Value *trueI64 = B.CreatePtrToInt(Sel->getTrueValue(), I64,
                                          Sel->getName() + "_t_i64");
      Value *falseI64 = B.CreatePtrToInt(Sel->getFalseValue(), I64,
                                           Sel->getName() + "_f_i64");
      Value *selI64 = B.CreateSelect(Sel->getCondition(), trueI64, falseI64,
                                       Sel->getName() + "_i64");
      Value *result = B.CreateIntToPtr(selI64, Sel->getType(),
                                         Sel->getName() + "_ptr");
      Sel->replaceAllUsesWith(result);
      Sel->eraseFromParent();
      changed = true;
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
