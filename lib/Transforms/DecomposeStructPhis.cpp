// Pass 1: DecomposeStructPhis — split struct phi nodes into scalar phis.
// Metal GPU JIT crashes on struct-typed phi nodes.
//
// Transform:
//   %s = phi { float, float } [ %iv1, %bb1 ], [ %iv2, %bb2 ]
//   %a = extractvalue { float, float } %s, 0
//   %b = extractvalue { float, float } %s, 1
// →
//   %s_0 = phi float [ (elem 0 of %iv1), %bb1 ], [ (elem 0 of %iv2), %bb2 ]
//   %s_1 = phi float [ (elem 1 of %iv1), %bb1 ], [ (elem 1 of %iv2), %bb2 ]
//   ; %a replaced by %s_0, %b replaced by %s_1
//
// The incoming values are typically insertvalue chains:
//   %iv = insertvalue { float, float } undef, float %x, 0
//   %iv2 = insertvalue { float, float } %iv, float %y, 1
// We trace these chains to find the scalar value for each element.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

/// Trace an insertvalue chain to find the scalar value for element `idx`.
/// Returns nullptr if we can't resolve it.
static Value *traceInsertValueElement(Value *V, unsigned idx) {
  // Walk the insertvalue chain backwards
  while (auto *IV = dyn_cast<InsertValueInst>(V)) {
    if (IV->getNumIndices() == 1 && IV->getIndices()[0] == idx)
      return IV->getInsertedValueOperand();
    // This insertvalue writes a different element — look at its aggregate
    V = IV->getAggregateOperand();
  }
  // If we hit undef, the element is undef
  if (isa<UndefValue>(V))
    return UndefValue::get(
        cast<StructType>(V->getType())->getElementType(idx));
  // If we hit a zeroinitializer, element is zero
  if (isa<ConstantAggregateZero>(V))
    return Constant::getNullValue(
        cast<StructType>(V->getType())->getElementType(idx));
  // If we hit another struct phi that was already decomposed,
  // or a constant aggregate, extract the element
  if (auto *C = dyn_cast<Constant>(V))
    if (auto *ST = dyn_cast<StructType>(V->getType()))
      if (idx < ST->getNumElements())
        return C->getAggregateElement(idx);
  return nullptr;
}

bool DecomposeStructPhisPass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (isa<StructType>(PN->getType()))
            return true;
  return false;
}

PreservedAnalyses DecomposeStructPhisPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  bool changed = false;

  for (auto &F : M) {
    // Collect struct phis (can't modify while iterating)
    SmallVector<PHINode *, 8> structPhis;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          if (auto *ST = dyn_cast<StructType>(PN->getType()))
            structPhis.push_back(PN);

    if (structPhis.empty()) continue;

    for (auto *PN : structPhis) {
      auto *ST = cast<StructType>(PN->getType());
      unsigned numElems = ST->getNumElements();
      IRBuilder<> B(PN);

      // Create scalar phis
      SmallVector<PHINode *, 4> scalarPhis;
      for (unsigned i = 0; i < numElems; i++) {
        auto *SP = B.CreatePHI(ST->getElementType(i),
                                PN->getNumIncomingValues(),
                                PN->getName() + "_" + Twine(i));
        scalarPhis.push_back(SP);
      }

      // Populate scalar phi incoming values
      for (unsigned inc = 0; inc < PN->getNumIncomingValues(); inc++) {
        Value *inVal = PN->getIncomingValue(inc);
        BasicBlock *inBB = PN->getIncomingBlock(inc);

        for (unsigned i = 0; i < numElems; i++) {
          Value *elem = traceInsertValueElement(inVal, i);
          if (!elem) {
            // Fallback: insert an extractvalue in the predecessor
            // (must insert before terminator)
            IRBuilder<> PredB(inBB->getTerminator());
            elem = PredB.CreateExtractValue(inVal, i,
                inVal->getName() + "_ext" + Twine(i));
          }
          scalarPhis[i]->addIncoming(elem, inBB);
        }
      }

      // Replace extractvalue users with scalar phis
      SmallVector<Instruction *, 8> toRemove;
      for (auto *U : PN->users()) {
        if (auto *EV = dyn_cast<ExtractValueInst>(U)) {
          if (EV->getNumIndices() == 1) {
            unsigned idx = EV->getIndices()[0];
            if (idx < numElems) {
              EV->replaceAllUsesWith(scalarPhis[idx]);
              toRemove.push_back(EV);
            }
          }
        }
      }

      // Remove extractvalue instructions
      for (auto *I : toRemove)
        I->eraseFromParent();

      // If the struct phi has remaining uses (shouldn't normally),
      // reconstruct the struct from scalar phis
      if (!PN->use_empty()) {
        IRBuilder<> AfterB(&*PN->getParent()->getFirstNonPHIIt());
        Value *agg = UndefValue::get(ST);
        for (unsigned i = 0; i < numElems; i++)
          agg = AfterB.CreateInsertValue(agg, scalarPhis[i],
                                          i, PN->getName() + "_rebuild");
        PN->replaceAllUsesWith(agg);
      }

      PN->eraseFromParent();
      changed = true;
    }

    // Clean up dead insertvalue chains
    // (insertvalues whose only use was as phi incoming value)
    bool cleanedUp = true;
    while (cleanedUp) {
      cleanedUp = false;
      for (auto &BB : F) {
        for (auto it = BB.begin(); it != BB.end();) {
          auto *IV = dyn_cast<InsertValueInst>(&*it++);
          if (IV && IV->use_empty()) {
            IV->eraseFromParent();
            cleanedUp = true;
          }
        }
      }
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
