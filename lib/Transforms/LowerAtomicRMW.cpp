// Pass 11: atomicrmw → air.atomic.* intrinsic calls.

#include "metal-ir/Pipeline.h"
#include "metal-ir/AIRIntrinsics.h"
#include "metal-ir/MetalConstraints.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool LowerAtomicRMWPass::needsRun(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (isa<AtomicRMWInst>(&I))
          return true;
  return false;
}

PreservedAnalyses LowerAtomicRMWPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  auto &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);
  bool changed = false;

  SmallVector<AtomicRMWInst *, 16> atomics;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
          atomics.push_back(RMW);

  for (auto *RMW : atomics) {
    unsigned AS = RMW->getPointerOperand()->getType()->getPointerAddressSpace();
    auto locality = (AS == metalir::AS::Threadgroup) ? air::AtomicLocality::Local
                              : air::AtomicLocality::Global;
    air::AtomicOp airOp;
    switch (RMW->getOperation()) {
    case AtomicRMWInst::Xchg: airOp = air::AtomicOp::Xchg; break;
    case AtomicRMWInst::Add:  airOp = air::AtomicOp::Add; break;
    case AtomicRMWInst::FAdd: airOp = air::AtomicOp::Add; break;
    case AtomicRMWInst::Sub:  airOp = air::AtomicOp::Sub; break;
    case AtomicRMWInst::And:  airOp = air::AtomicOp::And; break;
    case AtomicRMWInst::Or:   airOp = air::AtomicOp::Or; break;
    case AtomicRMWInst::Xor:  airOp = air::AtomicOp::Xor; break;
    case AtomicRMWInst::Max:  airOp = air::AtomicOp::Max; break;
    case AtomicRMWInst::Min:  airOp = air::AtomicOp::Min; break;
    case AtomicRMWInst::UMax: airOp = air::AtomicOp::UMax; break;
    case AtomicRMWInst::UMin: airOp = air::AtomicOp::UMin; break;
    default: continue;
    }

    Type *valTy = RMW->getValOperand()->getType();
    auto airTy = valTy->isFloatTy() ? air::AtomicType::F32
                                     : air::AtomicType::I32;
    std::string name = air::atomicName(locality, airOp, airTy);

    auto *ptrTy = PointerType::get(Ctx, AS);
    FunctionType *FTy = FunctionType::get(valTy, {ptrTy, valTy, I32, I32, I1}, false);
    FunctionCallee FC = M.getOrInsertFunction(name, FTy);

    IRBuilder<> B(RMW);
    Value *result = B.CreateCall(FC, {
        RMW->getPointerOperand(), RMW->getValOperand(),
        ConstantInt::get(I32, 0), ConstantInt::get(I32, 1),
        ConstantInt::get(I1, 1),
    }, RMW->getName());
    RMW->replaceAllUsesWith(result);
    RMW->eraseFromParent();
    changed = true;
  }
  return preserveIf(changed);
}

} // namespace metalir
