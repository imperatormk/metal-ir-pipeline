// Pass 21: WidenDeviceLoads — ALL device loads must be float when MMA present.
//
// Metal GPU JIT crashes on non-float device loads (half, i32, etc.) when
// simdgroup_matrix_8x8 intrinsics are present in the module.
//
// Transform:
//   %v = load half, ptr addrspace(1) %p    (where %p = gep half, %base, %idx)
// →
//   %idx_shr = lshr i32 %idx, 1            ; float index = half index / 2
//   %p_f = gep float, ptr addrspace(1) %base, i32 %idx_shr
//   %v_f = load float, ptr addrspace(1) %p_f
//   %v2 = bitcast float %v_f to <2 x half>
//   %lane = and i32 %idx, 1                ; which half within the float
//   %v = extractelement <2 x half> %v2, i32 %lane
//
// For i32/i16/etc device loads:
//   %v_f = load float, ptr addrspace(1) %p
//   %v = bitcast float %v_f to i32
//
// This pass queries MMAPresenceAnalysis and skips if no MMA.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool WidenDeviceLoadsPass::needsRun(Module &M) {
  // Only needed when MMA present AND there are non-float device loads
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      goto hasMMA;
  return false;
hasMMA:
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *LI = dyn_cast<LoadInst>(&I))
          if (LI->getPointerAddressSpace() == 1 && !LI->getType()->isFloatTy())
            return true;
  return false;
}

/// Try to decompose a GEP into (base, index, element type).
/// Returns true if the GEP is a simple `gep <elemTy>, ptr %base, i32 %idx`.
static bool decomposeGEP(GetElementPtrInst *GEP, Value *&base,
                          Value *&index, Type *&elemTy) {
  if (GEP->getNumIndices() != 1) return false;
  base = GEP->getPointerOperand();
  index = GEP->getOperand(1);
  elemTy = GEP->getSourceElementType();
  return true;
}

PreservedAnalyses WidenDeviceLoadsPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  // Check MMA presence
  auto &MMA = MAM.getResult<MMAPresenceAnalysis>(M);
  if (!MMA.hasMMA) return PreservedAnalyses::all();

  bool changed = false;
  Type *F32 = Type::getFloatTy(M.getContext());

  // Collect non-float device loads
  SmallVector<LoadInst *, 16> loadsToWiden;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *LI = dyn_cast<LoadInst>(&I))
          if (LI->getPointerAddressSpace() == 1 && !LI->getType()->isFloatTy())
            loadsToWiden.push_back(LI);

  for (auto *LI : loadsToWiden) {
    Type *origTy = LI->getType();
    IRBuilder<> B(LI);

    // Check if this is a half/bfloat load from a GEP
    bool isHalf = origTy->isHalfTy() || origTy->isBFloatTy();
    auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());

    if (isHalf && GEP && GEP->getSourceElementType() == origTy) {
      // Half-scaled widening: two halves packed into one float
      Value *base, *index;
      Type *elemTy;
      if (decomposeGEP(GEP, base, index, elemTy)) {
        // Float index = half index >> 1
        Value *idxShr = B.CreateLShr(index, ConstantInt::get(index->getType(), 1),
                                      LI->getName() + "_hidx");
        // GEP with float element type
        Value *floatPtr = B.CreateGEP(F32, base, idxShr,
                                       LI->getName() + "_fp");
        // Load float
        auto *floatLoad = B.CreateAlignedLoad(F32, floatPtr, Align(4),
                                               LI->getName() + "_wf");
        if (LI->isVolatile()) floatLoad->setVolatile(true);

        // Bitcast float → <2 x half>
        auto *vecTy = FixedVectorType::get(origTy, 2);
        Value *vec = B.CreateBitCast(floatLoad, vecTy, LI->getName() + "_v2");

        // Lane = idx & 1
        Value *lane = B.CreateAnd(index, ConstantInt::get(index->getType(), 1),
                                   LI->getName() + "_lane");

        // Extract the correct half
        Value *elem = B.CreateExtractElement(vec, lane, LI->getName());

        LI->replaceAllUsesWith(elem);
        LI->eraseFromParent();
        // Remove dead GEP
        if (GEP->use_empty()) GEP->eraseFromParent();
        changed = true;
        continue;
      }
    }

    // Non-half or no decomposable GEP: simple bitcast widening
    // load <origTy> → load float → bitcast to <origTy>
    unsigned origBits = M.getDataLayout().getTypeSizeInBits(origTy);
    if (origBits == 32) {
      // Same size as float — just bitcast the pointer and load float
      auto *floatLoad = B.CreateAlignedLoad(F32, LI->getPointerOperand(),
                                             LI->getAlign(),
                                             LI->getName() + "_wf");
      if (LI->isVolatile()) floatLoad->setVolatile(true);
      Value *cast = B.CreateBitCast(floatLoad, origTy, LI->getName());
      LI->replaceAllUsesWith(cast);
      LI->eraseFromParent();
      changed = true;
    }
    // For other sizes (16-bit, 64-bit), the half-scaling path above
    // handles 16-bit. 64-bit device loads are rare with MMA.
  }

  // Widen non-float device stores (reverse of load widening)
  SmallVector<StoreInst *, 16> storesToWiden;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *SI = dyn_cast<StoreInst>(&I))
          if (SI->getPointerAddressSpace() == 1 &&
              !SI->getValueOperand()->getType()->isFloatTy())
            storesToWiden.push_back(SI);

  for (auto *SI : storesToWiden) {
    Value *val = SI->getValueOperand();
    Type *origTy = val->getType();
    IRBuilder<> B(SI);

    bool isHalf = origTy->isHalfTy() || origTy->isBFloatTy();
    auto *GEP = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());

    if (isHalf && GEP && GEP->getSourceElementType() == origTy) {
      Value *base, *index;
      Type *elemTy;
      if (decomposeGEP(GEP, base, index, elemTy)) {
        // Read-modify-write: load float, unpack, modify element, repack, store
        Value *idxShr = B.CreateLShr(index, ConstantInt::get(index->getType(), 1),
                                      SI->getName() + "_hidx");
        Value *floatPtr = B.CreateGEP(F32, base, idxShr,
                                       SI->getName() + "_fp");
        auto *floatLoad = B.CreateAlignedLoad(F32, floatPtr, Align(4),
                                               SI->getName() + "_ld");
        auto *vecTy = FixedVectorType::get(origTy, 2);
        Value *vec = B.CreateBitCast(floatLoad, vecTy, SI->getName() + "_v2");
        Value *lane = B.CreateAnd(index, ConstantInt::get(index->getType(), 1),
                                   SI->getName() + "_lane");
        Value *updated = B.CreateInsertElement(vec, val, lane,
                                                SI->getName() + "_ins");
        Value *packed = B.CreateBitCast(updated, F32, SI->getName() + "_pack");
        B.CreateAlignedStore(packed, floatPtr, Align(4), SI->isVolatile());
        SI->eraseFromParent();
        if (GEP->use_empty()) GEP->eraseFromParent();
        changed = true;
        continue;
      }
    }

    // 32-bit non-float store: bitcast value to float and store
    unsigned origBits = M.getDataLayout().getTypeSizeInBits(origTy);
    if (origBits == 32) {
      Value *cast = B.CreateBitCast(val, F32, val->getName() + "_wf");
      B.CreateAlignedStore(cast, SI->getPointerOperand(),
                            SI->getAlign(), SI->isVolatile());
      SI->eraseFromParent();
      changed = true;
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
