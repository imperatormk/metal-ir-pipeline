// Pass 21: WidenDeviceLoads — widen non-float device loads/stores/GEPs.
//
// Two phases:
//
// Phase A (MMA-gated): When simdgroup_matrix_8x8 intrinsics are present,
// ALL device loads must be float. Uses GEP decomposition for precise
// half-scaled widening with lane extraction.
//
// Phase B (unconditional): Non-float device loads/stores/GEPs connected
// to phi nodes crash Metal GPU JIT regardless of MMA. The serializer
// infers pointer types from usage, so a phi pointer used by `load bfloat`
// gets `bfloat*` type in Metal AIR, which crashes the GPU JIT.

#include "metal-ir/Pipeline.h"
#include "metal-ir/IRUtil.h"
#include "metal-ir/KernelProfile.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool WidenDeviceLoadsPass::needsRun(Module &M) {
  // Phase A: MMA present AND non-float device loads
  bool hasMMA = false;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      { hasMMA = true; break; }

  if (hasMMA) {
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB)
          if (isDeviceLoad(&I) && !cast<LoadInst>(&I)->getType()->isFloatTy())
            return true;
  }

  // Phase B: ptr addrspace(1) phi with non-float uses
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN || !PN->getType()->isPointerTy()) continue;
        if (PN->getType()->getPointerAddressSpace() != AS::Device) continue;
        for (auto *U : PN->users()) {
          if (auto *LI = dyn_cast<LoadInst>(U))
            if (!LI->getType()->isFloatTy()) return true;
          if (auto *SI = dyn_cast<StoreInst>(U))
            if (SI->getPointerOperand() == PN &&
                !SI->getValueOperand()->getType()->isFloatTy())
              return true;
          if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
            if (!GEP->getSourceElementType()->isFloatTy()) return true;
        }
      }

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

// ── Phase B helpers ─────────────────────────────────────────────────────

/// Check if a ptr addrspace(1) value traces back to a phi node.
static bool isPtrFromPhi(Value *ptr) {
  if (isa<PHINode>(ptr)) return true;
  if (auto *EV = dyn_cast<ExtractValueInst>(ptr))
    return isa<PHINode>(EV->getAggregateOperand());
  if (auto *ITP = dyn_cast<IntToPtrInst>(ptr))
    return isa<PHINode>(ITP->getOperand(0));
  return false;
}

/// Check if a value transitively feeds into a phi (loop increment pattern).
static bool feedsIntoPhi(Value *start) {
  SmallVector<Value *, 8> worklist;
  SmallPtrSet<Value *, 16> visited;
  worklist.push_back(start);
  while (!worklist.empty()) {
    Value *V = worklist.pop_back_val();
    if (!visited.insert(V).second) continue;
    for (auto *U : V->users()) {
      if (isa<PHINode>(U)) return true;
      if (auto *IV = dyn_cast<InsertValueInst>(U))
        worklist.push_back(IV);
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
        if (GEP->getPointerOperand() == V)
          worklist.push_back(GEP);
    }
  }
  return false;
}

/// Phase B: widen non-float device ops connected to phi nodes.
static bool widenPhiConnectedOps(Module &M, Type *F32) {
  bool changed = false;

  // B1: Insert bitcast ptr→ptr before non-float device loads from phi pointers.
  // This breaks the pointer type chain: the phi keeps float* (from PTM/GEP
  // rewrite), the bitcast result gets bfloat* (from load type inference),
  // and Metal accepts bfloat* on non-phi values.
  // Use CastInst::Create directly — IRBuilder::CreateBitCast folds ptr→ptr away.
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *LI = dyn_cast<LoadInst>(&I);
        if (!LI) continue;
        if (LI->getPointerAddressSpace() != AS::Device) continue;
        if (LI->getType()->isFloatTy()) continue;
        if (!isPtrFromPhi(LI->getPointerOperand())) continue;
        auto *bc = CastInst::Create(Instruction::BitCast,
                                     LI->getPointerOperand(),
                                     LI->getPointerOperand()->getType(),
                                     "", LI);
        LI->setOperand(0, bc);
        changed = true;
      }

  // B2: Insert bitcast ptr→ptr before non-float device stores to phi pointers.
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI) continue;
        if (SI->getPointerAddressSpace() != AS::Device) continue;
        if (SI->getValueOperand()->getType()->isFloatTy()) continue;
        Value *storePtr = SI->getPointerOperand();
        bool fromPhi = isPtrFromPhi(storePtr);
        if (!fromPhi) {
          if (auto *ITP = dyn_cast<IntToPtrInst>(storePtr))
            if (auto *sel = dyn_cast<SelectInst>(ITP->getOperand(0)))
              for (unsigned i = 1; i <= 2; i++)
                if (auto *PTI = dyn_cast<PtrToIntInst>(sel->getOperand(i)))
                  if (isPtrFromPhi(PTI->getOperand(0)))
                    { fromPhi = true; break; }
        }
        if (!fromPhi) continue;
        auto *bc = CastInst::Create(Instruction::BitCast,
                                     storePtr, storePtr->getType(),
                                     "", SI);
        SI->setOperand(1, bc);
        changed = true;
      }

  // B3: Convert non-float device GEPs connected to phis to raw byte arithmetic.
  // gep bfloat, ptr, N → ptrtoint ptr, add N*sizeof(bfloat), inttoptr
  // This avoids typed pointer issues (no bfloat* or half* GEP results)
  // while preserving exact byte offsets (no halving precision loss).
  SmallVector<GetElementPtrInst *, 16> gepsToWiden;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *GEP = dyn_cast<GetElementPtrInst>(&I);
        if (!GEP) continue;
        if (GEP->getPointerAddressSpace() != AS::Device) continue;
        if (GEP->getNumIndices() != 1) continue;
        auto *srcTy = GEP->getSourceElementType();
        if (srcTy->isFloatTy()) continue;
        bool isHalf = srcTy->isHalfTy() || srcTy->isBFloatTy();
        if (!isHalf && !srcTy->isIntegerTy(32)) continue;
        // Only rewrite GEPs that feed BACK into a phi (loop increments).
        // Do NOT rewrite initial setup GEPs from phi bases — they may have
        // odd indices that can't be halved (e.g. thread ID offsets).
        bool connected = feedsIntoPhi(GEP);
        if (!connected) continue;
        gepsToWiden.push_back(GEP);
      }

  auto *I64 = Type::getInt64Ty(M.getContext());
  for (auto *GEP : gepsToWiden) {
    auto *srcTy = GEP->getSourceElementType();
    unsigned elemBytes = M.getDataLayout().getTypeAllocSize(srcTy);
    IRBuilder<> B(GEP);
    Value *base = GEP->getPointerOperand();
    Value *idx = GEP->getOperand(1);

    // Convert to byte arithmetic: ptrtoint + add(idx * elemBytes) + inttoptr
    Value *baseInt = B.CreatePtrToInt(base, I64, GEP->getName() + "_p2i");
    Value *idxExt = B.CreateSExt(idx, I64, GEP->getName() + "_ext");
    Value *byteOff = B.CreateMul(idxExt,
        ConstantInt::get(I64, elemBytes), GEP->getName() + "_off");
    Value *resultInt = B.CreateAdd(baseInt, byteOff, GEP->getName() + "_add");
    Value *resultPtr = B.CreateIntToPtr(resultInt, GEP->getType(),
                                         GEP->getName() + "_ptr");
    GEP->replaceAllUsesWith(resultPtr);
    GEP->eraseFromParent();
    changed = true;
  }

  return changed;
}

// ── Main pass entry ─────────────────────────────────────────────────────

PreservedAnalyses WidenDeviceLoadsPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  bool changed = false;
  Type *F32 = Type::getFloatTy(M.getContext());

  // ── Phase A: MMA-gated full widening ──────────────────────────────────
  MetalConstraints constraints;
  constraints.hasMMA = MAM.getResult<MMAPresenceAnalysis>(M).hasMMA;

  if (constraints.widenDeviceLoadsToFloat()) {
    auto &profiles = MAM.getResult<KernelProfileAnalysis>(M);

    // Collect non-float device loads
    SmallVector<LoadInst *, 16> loadsToWiden;
    for (auto &F : M) {
      auto it = profiles.find(&F);
      if (it != profiles.end() && !it->second.needsDeviceLoadWidening()
          && !it->second.hasNonFloatDeviceStore)
        continue;
      for (auto &BB : F)
        for (auto &I : BB)
          if (isDeviceLoad(&I) && !cast<LoadInst>(&I)->getType()->isFloatTy())
            loadsToWiden.push_back(cast<LoadInst>(&I));
    }

    for (auto *LI : loadsToWiden) {
      Type *origTy = LI->getType();
      IRBuilder<> B(LI);

      bool isHalf = origTy->isHalfTy() || origTy->isBFloatTy();
      auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());

      if (isHalf && GEP && GEP->getSourceElementType() == origTy) {
        Value *base, *index;
        Type *elemTy;
        if (decomposeGEP(GEP, base, index, elemTy)) {
          Value *idxShr = B.CreateLShr(index, ConstantInt::get(index->getType(), 1),
                                        LI->getName() + "_hidx");
          Value *floatPtr = B.CreateGEP(F32, base, idxShr,
                                         LI->getName() + "_fp");
          auto *floatLoad = B.CreateAlignedLoad(F32, floatPtr, Align(4),
                                                 LI->getName() + "_wf");
          if (LI->isVolatile()) floatLoad->setVolatile(true);

          auto *vecTy = FixedVectorType::get(origTy, 2);
          Value *vec = B.CreateBitCast(floatLoad, vecTy, LI->getName() + "_v2");
          Value *lane = B.CreateAnd(index, ConstantInt::get(index->getType(), 1),
                                     LI->getName() + "_lane");
          Value *elem = B.CreateExtractElement(vec, lane, LI->getName());

          LI->replaceAllUsesWith(elem);
          LI->eraseFromParent();
          if (GEP->use_empty()) GEP->eraseFromParent();
          changed = true;
          continue;
        }
      }

      unsigned origBits = M.getDataLayout().getTypeSizeInBits(origTy);
      if (origBits == 32) {
        auto *floatLoad = B.CreateAlignedLoad(F32, LI->getPointerOperand(),
                                               LI->getAlign(),
                                               LI->getName() + "_wf");
        if (LI->isVolatile()) floatLoad->setVolatile(true);
        Value *cast = B.CreateBitCast(floatLoad, origTy, LI->getName());
        LI->replaceAllUsesWith(cast);
        LI->eraseFromParent();
        changed = true;
      } else if (origBits == 8 && GEP && GEP->getSourceElementType() == origTy) {
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
    }

    // Widen non-float device stores (MMA path)
    SmallVector<StoreInst *, 16> storesToWiden;
    for (auto &F : M) {
      auto it = profiles.find(&F);
      if (it != profiles.end() && !it->second.hasNonFloatDeviceStore)
        continue;
      for (auto &BB : F)
        for (auto &I : BB)
          if (isDeviceStore(&I) &&
              !cast<StoreInst>(&I)->getValueOperand()->getType()->isFloatTy())
            storesToWiden.push_back(cast<StoreInst>(&I));
    }

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

      unsigned origBits = M.getDataLayout().getTypeSizeInBits(origTy);
      if (origBits == 32) {
        Value *cast = B.CreateBitCast(val, F32, val->getName() + "_wf");
        B.CreateAlignedStore(cast, SI->getPointerOperand(),
                              SI->getAlign(), SI->isVolatile());
        SI->eraseFromParent();
        changed = true;
      }
    }
  }

  // ── Phase B: unconditional phi-connected widening ─────────────────────
  changed |= widenPhiConnectedOps(M, F32);

  // ── Cleanup: dead non-float device GEPs ───────────────────────────────
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
            if (GEP->getPointerAddressSpace() != AS::Device) continue;
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
