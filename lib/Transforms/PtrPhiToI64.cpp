// Pass 2: PtrPhiToI64 — convert ptr phis to i64 phis.
// Two triggers:
//   (a) Metal GPU JIT has a ~63 ptr-typed phi node limit per block.
//       When exceeded, convert ALL ptr phis in the block to i64.
//   (b) Any ptr phi with an undef/poison incoming value crashes the Metal
//       GPU JIT ("materializeAll"). When this occurs, convert ALL ptr phis
//       in the function to i64 — not just the one with undef — because
//       related phis in other blocks may feed into each other and create
//       typed pointer mismatches (e.g. float* vs i32* phi incoming values).
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
#include "llvm/IR/Constants.h"

using namespace llvm;
namespace metalir {

static constexpr unsigned kPtrPhiLimit = 32;

/// Does this ptr phi have any undef/poison incoming values?
static bool hasUndefIncoming(PHINode *PN) {
  for (unsigned i = 0; i < PN->getNumIncomingValues(); i++)
    if (isa<UndefValue>(PN->getIncomingValue(i)))
      return true;
  return false;
}

bool PtrPhiToI64Pass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F) {
      unsigned ptrPhiCount = 0;
      for (auto &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (PN->getType()->isPointerTy()) {
            ptrPhiCount++;
            if (hasUndefIncoming(PN))
              return true;
          }
      if (ptrPhiCount > kPtrPhiLimit)
        return true;
    }
  return false;
}

/// Convert a single ptr phi to i64.
static void convertPtrPhiToI64(PHINode *PN, Type *I64) {
  Type *ptrTy = PN->getType();

  PHINode *newPhi = PHINode::Create(I64, PN->getNumIncomingValues(),
                                     PN->getName() + "_i64");
  newPhi->insertBefore(PN->getIterator());

  for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
    Value *inVal = PN->getIncomingValue(i);
    BasicBlock *inBB = PN->getIncomingBlock(i);

    Value *asInt;
    if (isa<UndefValue>(inVal)) {
      // undef/poison ptr → zero i64 (Metal can't handle undef ptrs)
      asInt = ConstantInt::get(I64, 0);
    } else if (isa<ConstantPointerNull>(inVal)) {
      asInt = ConstantInt::get(I64, 0);
    } else {
      // Insert ptrtoint before the predecessor's terminator
      IRBuilder<> PredB(inBB->getTerminator());
      asInt = PredB.CreatePtrToInt(inVal, I64,
                                   inVal->getName() + "_p2i");
    }
    newPhi->addIncoming(asInt, inBB);
  }

  // Insert inttoptr after all phis in this block
  BasicBlock *BB = PN->getParent();
  IRBuilder<> B(&*BB->getFirstNonPHIIt());
  Value *backToPtr = B.CreateIntToPtr(newPhi, ptrTy,
                                       PN->getName() + "_ptr");

  PN->replaceAllUsesWith(backToPtr);
  PN->eraseFromParent();
}

PreservedAnalyses PtrPhiToI64Pass::run(Module &M,
                                        ModuleAnalysisManager &AM) {
  bool changed = false;
  Type *I64 = Type::getInt64Ty(M.getContext());

  for (auto &F : M) {
    // Check if any ptr phi in this function has undef/poison.
    // If so, convert ALL ptr phis in the function — not just the one with
    // undef — because interconnected phis across blocks create typed pointer
    // mismatches that crash the Metal GPU JIT.
    bool functionHasUndefPtrPhi = false;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (PN->getType()->isPointerTy() && hasUndefIncoming(PN)) {
            functionHasUndefPtrPhi = true;
            break;
          }

    for (auto &BB : F) {
      SmallVector<PHINode *, 16> ptrPhis;
      for (auto &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (PN->getType()->isPointerTy())
            ptrPhis.push_back(PN);

      // Convert all ptr phis in this block if:
      // (a) block exceeds the ptr phi limit, or
      // (b) function has any ptr phi with undef/poison
      if (ptrPhis.size() <= kPtrPhiLimit && !functionHasUndefPtrPhi)
        continue;

      for (auto *PN : ptrPhis) {
        convertPtrPhiToI64(PN, I64);
        changed = true;
      }
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
