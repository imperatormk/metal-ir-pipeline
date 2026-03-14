// Pass 9: llvm.smin/smax/umin/umax → icmp + select.
// Metal GPU JIT doesn't support LLVM integer min/max intrinsics.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"

using namespace llvm;
namespace metalir {

bool LowerIntMinMaxPass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *II = dyn_cast<IntrinsicInst>(&I))
          switch (II->getIntrinsicID()) {
          case Intrinsic::smin: case Intrinsic::smax:
          case Intrinsic::umin: case Intrinsic::umax:
            return true;
          default: break;
          }
  return false;
}

PreservedAnalyses LowerIntMinMaxPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  bool changed = false;
  struct Entry { Intrinsic::ID id; CmpInst::Predicate pred; };
  static const Entry entries[] = {
      {Intrinsic::smin, CmpInst::ICMP_SLT},
      {Intrinsic::smax, CmpInst::ICMP_SGT},
      {Intrinsic::umin, CmpInst::ICMP_ULT},
      {Intrinsic::umax, CmpInst::ICMP_UGT},
  };

  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        auto *II = dyn_cast<IntrinsicInst>(&*it++);
        if (!II) continue;
        CmpInst::Predicate pred = CmpInst::BAD_ICMP_PREDICATE;
        for (auto &e : entries)
          if (II->getIntrinsicID() == e.id) { pred = e.pred; break; }
        if (pred == CmpInst::BAD_ICMP_PREDICATE) continue;

        IRBuilder<> B(II);
        Value *A = II->getArgOperand(0), *Bv = II->getArgOperand(1);
        Value *Cmp = B.CreateICmp(pred, A, Bv, II->getName() + ".cmp");
        Value *Sel = B.CreateSelect(Cmp, A, Bv, II->getName());
        II->replaceAllUsesWith(Sel);
        II->eraseFromParent();
        changed = true;
      }
    }
  }
  return preserveIf(changed);
}

} // namespace metalir
