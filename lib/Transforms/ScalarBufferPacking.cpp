// Pass 5b: ScalarBufferPacking — rewrite scalar params to constant buffer loads.
//
// Metal kernels can only receive buffers and system values as parameters.
// Triton emits raw scalar params (float %scale, i32 %n). This pass rewrites
// each scalar param to a ptr addrspace(2) (constant buffer) param + load:
//
//   define void @kern(float %x) → define void @kern(ptr addrspace(2) %x_ptr)
//   %x = load float, ptr addrspace(2) %x_ptr
//
// Must run BEFORE AIRSystemValues (which generates !air.kernel metadata).

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

bool ScalarBufferPackingPass::needsRun(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    for (auto &Arg : F.args())
      if (!Arg.getType()->isPointerTy() && !Arg.getType()->isVectorTy())
        return true;
  }
  return false;
}

PreservedAnalyses ScalarBufferPackingPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  bool changed = false;

  // Only rewrite kernel entry points, not helper functions.
  // A kernel is a non-declaration function that is NOT called by others.
  SmallPtrSet<Function *, 4> calledFns;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (auto *Callee = CI->getCalledFunction())
            calledFns.insert(Callee);

  SmallVector<Function *, 4> funcs;
  for (auto &F : M)
    if (!F.isDeclaration() && !calledFns.count(&F))
      funcs.push_back(&F);

  for (auto *FPtr : funcs) {
    Function &F = *FPtr;

    // Collect param indices described as system values in pre-baked metadata
    SmallDenseSet<unsigned, 4> sysValParams;
    if (auto *KMD = M.getNamedMetadata("air.kernel")) {
      for (unsigned k = 0; k < KMD->getNumOperands(); k++) {
        auto *Node = KMD->getOperand(k);
        if (Node->getNumOperands() < 1) continue;
        auto *FnMD = dyn_cast_if_present<ValueAsMetadata>(Node->getOperand(0));
        if (!FnMD || FnMD->getValue() != &F) continue;
        // Scan param nodes for system value entries
        for (unsigned n = 1; n < Node->getNumOperands(); n++) {
          auto *Sub = dyn_cast_if_present<MDNode>(Node->getOperand(n));
          if (!Sub) continue;
          for (unsigned s = 0; s < Sub->getNumOperands(); s++) {
            auto *ParamNode = dyn_cast_if_present<MDNode>(Sub->getOperand(s));
            if (!ParamNode || ParamNode->getNumOperands() < 2) continue;
            // Check if second operand is a system value string
            if (auto *Str = dyn_cast<MDString>(ParamNode->getOperand(1))) {
              if (Str->getString().starts_with("air.thread") ||
                  Str->getString().starts_with("air.threadgroup"))
                if (auto *Idx = dyn_cast<ConstantAsMetadata>(
                        ParamNode->getOperand(0)))
                  sysValParams.insert(
                      cast<ConstantInt>(Idx->getValue())->getZExtValue());
            }
          }
        }
      }
    }

    // Find scalar params (non-pointer, non-vector, not system values)
    SmallVector<unsigned, 4> scalarParamIndices;
    for (unsigned i = 0; i < F.arg_size(); i++) {
      Type *T = F.getArg(i)->getType();
      if (!T->isPointerTy() && !T->isVectorTy() && !sysValParams.count(i))
        scalarParamIndices.push_back(i);
    }

    if (scalarParamIndices.empty())
      continue;

    // Build new function type: replace scalar params with ptr addrspace(2)
    auto *OldFTy = F.getFunctionType();
    SmallVector<Type *, 8> newParamTypes;
    for (unsigned i = 0; i < OldFTy->getNumParams(); i++) {
      if (std::find(scalarParamIndices.begin(), scalarParamIndices.end(), i) !=
          scalarParamIndices.end()) {
        newParamTypes.push_back(PointerType::get(M.getContext(), 2));
      } else {
        newParamTypes.push_back(OldFTy->getParamType(i));
      }
    }

    auto *NewFTy = FunctionType::get(OldFTy->getReturnType(), newParamTypes,
                                      OldFTy->isVarArg());

    // Create new function, move body
    auto *NewF = Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(),
                                   "", &M);
    NewF->copyAttributesFrom(&F);
    NewF->splice(NewF->begin(), &F);

    // Map old args → new args, insert loads for scalar params
    auto newArgIt = NewF->arg_begin();
    for (unsigned i = 0; i < F.arg_size(); i++) {
      Argument *oldArg = F.getArg(i);
      Argument *newArg = &*newArgIt++;

      if (std::find(scalarParamIndices.begin(), scalarParamIndices.end(), i) !=
          scalarParamIndices.end()) {
        // Scalar → ptr addrspace(2): insert load at entry
        newArg->setName(oldArg->getName().str() + "_ptr");
        IRBuilder<> B(&*NewF->getEntryBlock().getFirstInsertionPt());
        auto *Load = B.CreateLoad(oldArg->getType(), newArg, oldArg->getName());
        oldArg->replaceAllUsesWith(Load);
      } else {
        newArg->setName(oldArg->getName());
        oldArg->replaceAllUsesWith(newArg);
      }
    }

    // Update metadata references (e.g., !air.kernel) from old → new function
    for (auto &NMD : M.named_metadata()) {
      for (unsigned i = 0; i < NMD.getNumOperands(); i++) {
        auto *Node = NMD.getOperand(i);
        for (unsigned j = 0; j < Node->getNumOperands(); j++) {
          if (auto *VMD = dyn_cast_if_present<ValueAsMetadata>(
                  Node->getOperand(j))) {
            if (VMD->getValue() == &F) {
              Node->replaceOperandWith(
                  j, ValueAsMetadata::get(NewF));
            }
          }
        }
      }
    }

    // Replace old function
    std::string fname = F.getName().str();
    F.eraseFromParent();
    NewF->setName(fname);

    changed = true;
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace metalir
