// Pass 2: PtrPhiToI64 — convert ptr phis to i64 phis.
// Metal GPU JIT has a ~63 ptr-typed phi node limit per block.
// When exceeded, convert ptr phis to i64: ptrtoint before, inttoptr after.
//
// phi ptr addrspace(1) [ %p1, %bb1 ], [ %p2, %bb2 ]
// →
// (in predecessors: %p1_i64 = ptrtoint ptr %p1 to i64)
// %phi_i64 = phi i64 [ %p1_i64, %bb1 ], [ %p2_i64, %bb2 ]
// %phi_ptr = inttoptr i64 %phi_i64 to ptr addrspace(1)

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

static constexpr unsigned kPtrPhiLimit = 32;

bool PtrPhiToI64Pass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F) {
      unsigned ptrPhiCount = 0;
      for (auto &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (PN->getType()->isPointerTy())
            ptrPhiCount++;
      if (ptrPhiCount > kPtrPhiLimit)
        return true;
    }
  return false;
}

PreservedAnalyses PtrPhiToI64Pass::run(Module &M,
                                        ModuleAnalysisManager &AM) {
  bool changed = false;
  Type *I64 = Type::getInt64Ty(M.getContext());

  for (auto &F : M) {
    for (auto &BB : F) {
      // Count ptr phis
      SmallVector<PHINode *, 16> ptrPhis;
      for (auto &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (PN->getType()->isPointerTy())
            ptrPhis.push_back(PN);

      if (ptrPhis.size() <= kPtrPhiLimit)
        continue;

      for (auto *PN : ptrPhis) {
        Type *ptrTy = PN->getType();

        // Insert ptrtoint in each predecessor (before terminator)
        PHINode *newPhi = PHINode::Create(I64, PN->getNumIncomingValues(),
                                           PN->getName() + "_i64");
        newPhi->insertBefore(PN->getIterator());

        for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
          Value *inVal = PN->getIncomingValue(i);
          BasicBlock *inBB = PN->getIncomingBlock(i);

          // Insert ptrtoint before the predecessor's terminator
          IRBuilder<> PredB(inBB->getTerminator());
          Value *asInt = PredB.CreatePtrToInt(inVal, I64,
                                               inVal->getName() + "_p2i");
          newPhi->addIncoming(asInt, inBB);
        }

        // Insert inttoptr after all phis in this block
        IRBuilder<> B(&*BB.getFirstNonPHIIt());
        Value *backToPtr = B.CreateIntToPtr(newPhi, ptrTy,
                                             PN->getName() + "_ptr");

        PN->replaceAllUsesWith(backToPtr);
        PN->eraseFromParent();
        changed = true;
      }
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
