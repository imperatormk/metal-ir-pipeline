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
/// Flattens GEP chains: if %base is itself a GEP with the same element type,
/// combines the indices so that the word/lane split is computed from the total
/// element offset from the true base pointer.
static bool decomposeGEP(GetElementPtrInst *GEP, Value *&base,
                          Value *&index, Type *&elemTy) {
  if (GEP->getNumIndices() != 1) return false;
  base = GEP->getPointerOperand();
  index = GEP->getOperand(1);
  elemTy = GEP->getSourceElementType();

  // Flatten GEP chains: GEP(GEP(base, idx1), idx2) → (base, idx1+idx2)
  // This is critical for transposed stores where the outer GEP provides
  // the column offset and the inner GEP provides the row*stride offset,
  // or vice versa. Without flattening, two adjacent half stores may both
  // compute the same word index and lane, causing the second to overwrite
  // the first.
  while (auto *baseGEP = dyn_cast<GetElementPtrInst>(base)) {
    if (baseGEP->getNumIndices() != 1) break;
    if (baseGEP->getSourceElementType() != elemTy) break;
    // Combine: total index = baseGEP.index + index
    IRBuilder<> B(GEP);
    Value *baseIdx = baseGEP->getOperand(1);
    // Widen to matching types if needed (e.g. i32 + i64)
    if (baseIdx->getType() != index->getType()) {
      unsigned bw1 = baseIdx->getType()->getIntegerBitWidth();
      unsigned bw2 = index->getType()->getIntegerBitWidth();
      if (bw1 < bw2)
        baseIdx = B.CreateSExt(baseIdx, index->getType());
      else
        index = B.CreateSExt(index, baseIdx->getType());
    }
    index = B.CreateAdd(baseIdx, index, "idx_flat");
    base = baseGEP->getPointerOperand();
  }

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
    } else if (origBits == 8 && GEP && GEP->getSourceElementType() == origTy) {
      // i8 load: 4 bytes per float. Load float at index/4, extract byte at index%4.
      // load i8, gep i8 base idx → load float gep float base (idx>>2),
      //   bitcast to i32, shift right by (lane*8), trunc to i8
      Value *base, *index;
      Type *elemTy;
      if (decomposeGEP(GEP, base, index, elemTy)) {
        auto *I32 = Type::getInt32Ty(M.getContext());
        Value *idxShr = B.CreateLShr(index, ConstantInt::get(index->getType(), 2),
                                      LI->getName() + "_bidx");
        Value *floatPtr = B.CreateGEP(F32, base, idxShr,
                                       LI->getName() + "_fp");
        auto *floatLoad = B.CreateAlignedLoad(F32, floatPtr, Align(4),
                                               LI->getName() + "_wf");
        if (LI->isVolatile()) floatLoad->setVolatile(true);

        // bitcast float → i32, shift right by (lane * 8), trunc to i8
        Value *asI32 = B.CreateBitCast(floatLoad, I32, LI->getName() + "_i32");
        Value *lane = B.CreateAnd(index, ConstantInt::get(index->getType(), 3),
                                   LI->getName() + "_lane");
        Value *shiftAmt = B.CreateShl(lane, ConstantInt::get(lane->getType(), 3),
                                       LI->getName() + "_sh");
        Value *shifted = B.CreateLShr(asI32, shiftAmt, LI->getName() + "_sr");
        Value *elem = B.CreateTrunc(shifted, origTy, LI->getName());

        LI->replaceAllUsesWith(elem);
        LI->eraseFromParent();
        if (GEP->use_empty()) GEP->eraseFromParent();
        changed = true;
      }
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

  // Clean up dead device-memory GEPs with non-float element types.
  // GEP flattening can leave intermediate GEPs dead. These must be removed
  // before serialization because Metal AIR typed pointers may not support
  // all element types (e.g. half*, bfloat*, i8*).
  if (changed) {
    bool progress = true;
    while (progress) {
      progress = false;
      for (auto &F : M)
        for (auto &BB : F)
          for (auto II = BB.begin(); II != BB.end(); ) {
            auto *GEP = dyn_cast<GetElementPtrInst>(&*II++);
            if (!GEP) continue;
            if (!GEP->use_empty()) continue;
            if (GEP->getPointerAddressSpace() != 1) continue;
            auto *srcTy = GEP->getSourceElementType();
            if (!srcTy->isFloatTy()) {
              GEP->eraseFromParent();
              progress = true;
            }
          }
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
