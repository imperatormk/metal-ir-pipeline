#include "metal-ir/ValueEnumerator.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

namespace metalir {

void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                         ArrayRef<const Constant *> constants,
                         unsigned codeSize) {
  if (constants.empty()) return;
  W.EnterSubblock(bitc::CONSTANTS_BLOCK_ID, codeSize);

  Type *lastType = nullptr;
  for (auto *C : constants) {
    SmallVector<uint64_t, 8> V;
    if (C->getType() != lastType) {
      lastType = C->getType();
      V.push_back(E.typeIdx(lastType));
      W.EmitRecord(bitc::CST_CODE_SETTYPE, V);
      V.clear();
    }
    if (isa<UndefValue>(C)) {
      W.EmitRecord(bitc::CST_CODE_UNDEF, V);
    } else if (auto *CI = dyn_cast<ConstantInt>(C)) {
      // Always use INTEGER for ints (even zero) — Metal doesn't
      // accept NULL for integer types in some contexts.
      // Use ZExtValue for i1 (Metal convention), SExtValue for wider types.
      int64_t v = CI->getType()->isIntegerTy(1)
                      ? (int64_t)CI->getZExtValue()
                      : CI->getSExtValue();
      V.push_back(v >= 0 ? uint64_t(v) << 1 : (uint64_t(-v) << 1) | 1);
      W.EmitRecord(bitc::CST_CODE_INTEGER, V);
    } else if (auto *CF = dyn_cast<ConstantFP>(C)) {
      V.push_back(CF->getValueAPF().bitcastToAPInt().getZExtValue());
      W.EmitRecord(bitc::CST_CODE_FLOAT, V);
    } else if (auto *CDV = dyn_cast<ConstantDataVector>(C)) {
      for (unsigned i = 0; i < CDV->getNumElements(); i++) {
        if (CDV->getElementType()->isIntegerTy())
          V.push_back(CDV->getElementAsInteger(i));
        else
          V.push_back(CDV->getElementAsAPFloat(i).bitcastToAPInt().getZExtValue());
      }
      W.EmitRecord(bitc::CST_CODE_DATA, V);
    } else if (auto *CDA = dyn_cast<ConstantDataArray>(C)) {
      // Metal v1: emit as AGGREGATE referencing individual element constants
      for (unsigned i = 0; i < CDA->getNumElements(); i++)
        V.push_back(E.moduleConstIdx(CDA->getElementAsConstant(i)));
      W.EmitRecord(bitc::CST_CODE_AGGREGATE, V);
    } else if (isa<ConstantArray>(C) || isa<ConstantStruct>(C) ||
               isa<ConstantVector>(C)) {
      // Aggregate: list element value IDs (elements already in table)
      for (unsigned i = 0; i < C->getNumOperands(); i++)
        V.push_back(E.moduleConstIdx(cast<Constant>(C->getOperand(i))));
      W.EmitRecord(bitc::CST_CODE_AGGREGATE, V);
    } else if (C->isNullValue()) {
      // NULL for aggregate zero, pointer null, vector zero
      W.EmitRecord(bitc::CST_CODE_NULL, V);
    } else {
      W.EmitRecord(bitc::CST_CODE_NULL, V);
    }
  }
  W.ExitBlock();
}

} // namespace metalir
