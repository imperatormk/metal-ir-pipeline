#include "metal-ir/ValueEnumerator.h"
#include "metal-ir/BitcodeEncoding.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

namespace metalir {

// Forward declaration
void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                         ArrayRef<const Constant *> constants,
                         unsigned codeSize);

void emitFunctionBlock(BitstreamWriter &W, const Function &F,
                        ValueEnumerator &E) {
  W.EnterSubblock(bitc::FUNCTION_BLOCK_ID, 5);

  // Build local value ID map
  DenseMap<const Value *, unsigned> localMap;
  unsigned nextID = E.globalValues.size() + E.moduleConstants.size();

  for (auto &Arg : F.args())
    localMap[&Arg] = nextID++;

  // Collect function-level constants
  SmallVector<const Constant *, 32> funcConsts;
  for (auto &BB : F)
    for (auto &I : BB)
      for (auto &Op : I.operands())
        if (auto *C = dyn_cast<Constant>(Op))
          if (!isa<GlobalValue>(C) && !localMap.count(C) &&
              !E.globalValueMap.count(C) && !E.hasModuleConst(C)) {
            localMap[C] = nextID++;
            funcConsts.push_back(C);
          }

  // Instruction results
  for (auto &BB : F)
    for (auto &I : BB)
      if (!I.getType()->isVoidTy())
        localMap[&I] = nextID++;

  auto getAbsID = [&](const Value *V) -> unsigned {
    auto it = localMap.find(V);
    if (it != localMap.end()) return it->second;
    if (E.globalValueMap.count(V)) return E.globalIdx(V);
    if (auto *C = dyn_cast<Constant>(V))
      if (E.hasModuleConst(C)) return E.moduleConstIdx(C);
    return 0;
  };

  // Relative value IDs (current instruction ID minus referenced value ID)
  unsigned curInstID = E.globalValues.size() + E.moduleConstants.size()
                       + F.arg_size() + funcConsts.size();
  auto getID = [&](const Value *V) -> unsigned {
    return curInstID - getAbsID(V);
  };

  // DECLAREBLOCKS
  unsigned numBBs = 0;
  for (auto &BB : F) (void)BB, numBBs++;
  SmallVector<uint64_t, 1> DV = {numBBs};
  W.EmitRecord(bitc::FUNC_CODE_DECLAREBLOCKS, DV);

  // Function constants
  emitConstantsBlock(W, E, funcConsts, 5);

  // BB index helper
  SmallVector<const BasicBlock *, 8> bbList;
  for (auto &BB : F) bbList.push_back(&BB);
  auto bbIdx = [&](const BasicBlock *BB) -> unsigned {
    for (unsigned i = 0; i < bbList.size(); i++)
      if (bbList[i] == BB) return i;
    return 0;
  };

  // Emit instructions
  for (auto &BB : F) {
    for (auto &I : BB) {
      SmallVector<uint64_t, 16> V;

      if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
        V.push_back(getID(BO->getOperand(0)));
        V.push_back(getID(BO->getOperand(1)));
        V.push_back(encodeBinop(BO->getOpcode()));
        if (BO->getType()->isFPOrFPVectorTy()) V.push_back(0);
        W.EmitRecord(bitc::FUNC_CODE_INST_BINOP, V);
      } else if (auto *CI = dyn_cast<CastInst>(&I)) {
        V.push_back(getID(CI->getOperand(0)));
        // For casts producing pointers, use PTM-inferred pointee
        // (Metal v1 needs correct typed pointer per value usage)
        if (CI->getType()->isPointerTy()) {
          V.push_back(E.ptrTypeIdxForValue(CI));
        } else {
          V.push_back(E.typeIdx(CI->getDestTy()));
        }
        V.push_back(encodeCast(CI->getOpcode()));
        W.EmitRecord(bitc::FUNC_CODE_INST_CAST, V);
      } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
        V.push_back(getID(LI->getPointerOperand()));
        V.push_back(E.typeIdx(LI->getType()));
        V.push_back(Log2_32(LI->getAlign().value()) + 1);
        V.push_back(LI->isVolatile() ? 1 : 0);
        W.EmitRecord(bitc::FUNC_CODE_INST_LOAD, V);
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        V.push_back(getID(SI->getPointerOperand()));
        V.push_back(getID(SI->getValueOperand()));
        V.push_back(Log2_32(SI->getAlign().value()) + 1);
        V.push_back(SI->isVolatile() ? 1 : 0);
        W.EmitRecord(bitc::FUNC_CODE_INST_STORE, V);
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        V.push_back(GEP->isInBounds() ? 1 : 0);
        V.push_back(E.typeIdx(GEP->getSourceElementType()));
        for (auto &Op : GEP->operands()) V.push_back(getID(Op));
        W.EmitRecord(bitc::FUNC_CODE_INST_GEP, V);
      } else if (auto *EE = dyn_cast<ExtractElementInst>(&I)) {
        V.push_back(getID(EE->getVectorOperand()));
        V.push_back(getID(EE->getIndexOperand()));
        W.EmitRecord(bitc::FUNC_CODE_INST_EXTRACTELT, V);
      } else if (auto *IE = dyn_cast<InsertElementInst>(&I)) {
        V.push_back(getID(IE->getOperand(0)));
        V.push_back(getID(IE->getOperand(1)));
        V.push_back(getID(IE->getOperand(2)));
        W.EmitRecord(bitc::FUNC_CODE_INST_INSERTELT, V);
      } else if (auto *SV = dyn_cast<ShuffleVectorInst>(&I)) {
        V.push_back(getID(SV->getOperand(0)));
        V.push_back(getID(SV->getOperand(1)));
        V.push_back(getID(SV->getShuffleMaskForBitcode()));
        W.EmitRecord(bitc::FUNC_CODE_INST_SHUFFLEVEC, V);
      } else if (auto *IV = dyn_cast<InsertValueInst>(&I)) {
        V.push_back(getID(IV->getAggregateOperand()));
        V.push_back(getID(IV->getInsertedValueOperand()));
        for (unsigned idx : IV->getIndices()) V.push_back(idx);
        W.EmitRecord(bitc::FUNC_CODE_INST_INSERTVAL, V);
      } else if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
        V.push_back(getID(EV->getAggregateOperand()));
        for (unsigned idx : EV->getIndices()) V.push_back(idx);
        W.EmitRecord(bitc::FUNC_CODE_INST_EXTRACTVAL, V);
      } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
        V.push_back(getID(Sel->getTrueValue()));
        V.push_back(getID(Sel->getFalseValue()));
        V.push_back(getID(Sel->getCondition()));
        W.EmitRecord(Sel->getType()->isVectorTy()
                         ? bitc::FUNC_CODE_INST_VSELECT
                         : bitc::FUNC_CODE_INST_SELECT, V);
      } else if (auto *Cmp = dyn_cast<CmpInst>(&I)) {
        V.push_back(getID(Cmp->getOperand(0)));
        V.push_back(getID(Cmp->getOperand(1)));
        V.push_back(Cmp->getPredicate());
        W.EmitRecord(bitc::FUNC_CODE_INST_CMP2, V);
      } else if (auto *PN = dyn_cast<PHINode>(&I)) {
        V.push_back(E.typeIdx(PN->getType()));
        for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
          // PHI uses signed relative IDs (back-edge values have higher absID
          // than current, producing negative relative ID = forward reference)
          int64_t relID = (int64_t)curInstID - (int64_t)getAbsID(PN->getIncomingValue(i));
          // Signed VBR: positive n → 2n, negative n → (-2n)+1
          uint64_t encoded = (relID >= 0) ? ((uint64_t)relID << 1)
                                           : ((uint64_t)(-relID) << 1) | 1;
          V.push_back(encoded);
          V.push_back(bbIdx(PN->getIncomingBlock(i)));
        }
        W.EmitRecord(bitc::FUNC_CODE_INST_PHI, V);
      } else if (auto *BI = dyn_cast<BranchInst>(&I)) {
        if (BI->isUnconditional()) {
          V.push_back(bbIdx(BI->getSuccessor(0)));
        } else {
          V.push_back(bbIdx(BI->getSuccessor(0)));
          V.push_back(bbIdx(BI->getSuccessor(1)));
          V.push_back(getID(BI->getCondition()));
        }
        W.EmitRecord(bitc::FUNC_CODE_INST_BR, V);
      } else if (auto *RI = dyn_cast<ReturnInst>(&I)) {
        if (RI->getReturnValue()) V.push_back(getID(RI->getReturnValue()));
        W.EmitRecord(bitc::FUNC_CODE_INST_RET, V);
      } else if (isa<UnreachableInst>(&I)) {
        W.EmitRecord(bitc::FUNC_CODE_INST_UNREACHABLE, V);
      } else if (auto *CI = dyn_cast<CallInst>(&I)) {
        // CALL: [paramattr, cc_flags, fnty, fnid, ...args, [sentinel]]
        // MMA load calls need paramattr=1 (nocapture+readonly on ptr param)
        bool isMMALoadCall = false;
        if (auto *Callee = CI->getCalledFunction())
          isMMALoadCall = Callee->getName().starts_with("air.simdgroup_matrix_8x8_load");
        V.push_back(isMMALoadCall ? 1 : 0);

        // Detect MMA intrinsics that need operand bundle encoding
        bool isMMAWithBundles = false;
        if (auto *Callee = CI->getCalledFunction()) {
          StringRef name = Callee->getName();
          isMMAWithBundles =
              name.starts_with("air.simdgroup_matrix_8x8_load") ||
              name.starts_with("air.simdgroup_matrix_8x8_multiply_accumulate");
        }

        uint64_t flags = 0;
        if (CI->isTailCall()) flags |= 1;
        if (CI->getCallingConv() != CallingConv::C)
          flags |= (uint64_t)CI->getCallingConv() << 1;
        flags |= (1 << 15); // explicit function type
        if (isMMAWithBundles) flags |= (1 << 17); // operand bundles
        V.push_back(flags);

        if (isMMAWithBundles) {
          // MMA bundle encoding: sentinel 254, then function type, then callee
          V.push_back(254);
          V.push_back(E.typeIdx(CI->getFunctionType()));
        } else {
          V.push_back(E.typeIdx(CI->getFunctionType()));
        }

        V.push_back(getID(CI->getCalledOperand()));
        for (unsigned i = 0; i < CI->arg_size(); i++)
          V.push_back(getID(CI->getArgOperand(i)));
        W.EmitRecord(bitc::FUNC_CODE_INST_CALL, V);
      } else if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        V.push_back(E.typeIdx(AI->getAllocatedType()));
        V.push_back(E.typeIdx(AI->getArraySize()->getType()));
        V.push_back(getID(AI->getArraySize()));
        V.push_back(1 << 6); // explicit type flag, no alignment (Metal convention)
        W.EmitRecord(bitc::FUNC_CODE_INST_ALLOCA, V);
      }

      if (!I.getType()->isVoidTy())
        curInstID++;
    }
  }

  W.ExitBlock();
}

} // namespace metalir
