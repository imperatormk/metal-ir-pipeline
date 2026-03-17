#include "metal-ir/PointeeTypeMap.h"
#include "metal-ir/IRUtil.h"
#include "metal-ir/MetalConstraints.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

AnalysisKey PointeeTypeAnalysis::Key;

// ── Infer pointee type from usage ────────────────────────────────────────
//
// Delegates to inferElementType (IRUtil.h) for load/store/GEP recursion,
// then falls back to GEP source type and atomic intrinsic name inference.

Type *PointeeTypeMap::inferFromUsage(Value *ptr) {
  // Prioritize load/store types over GEP source types.
  // Recurse through GEP chains to find the ultimate store/load type.
  Type *gepType = nullptr;
  Type *atomicType = nullptr;
  for (auto *U : ptr->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U))
      return LI->getType();
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getPointerOperand() == ptr)
        return SI->getValueOperand()->getType();
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (Type *T = inferFromUsage(GEP))
        return T;
      if (!gepType)
        gepType = GEP->getSourceElementType();
    }
    if (auto *CI = dyn_cast<CallInst>(U)) {
      if (auto *Callee = CI->getCalledFunction()) {
        StringRef name = Callee->getName();
        if (name.starts_with("air.atomic.") && !atomicType) {
          if (name.ends_with(".i32"))
            atomicType = Type::getInt32Ty(ptr->getContext());
          else if (name.ends_with(".f32"))
            atomicType = Type::getFloatTy(ptr->getContext());
        }
      }
    }
  }
  if (gepType) return gepType;
  return atomicType;
}

// ── Collapse device pointers to float* ───────────────────────────────────
//
// When MMA intrinsics (simdgroup_multiply_accumulate) are present, the Metal
// GPU JIT crashes on ANY non-float device pointer. This collapses all
// addrspace(1) entries to float*.

void PointeeTypeMap::collapseDevicePointersToFloat(Module &M) {
  Type *F32 = Type::getFloatTy(M.getContext());
  for (auto &[ptr, ty] : map) {
    // Check if this is a device pointer (addrspace 1)
    auto *ptrTy = ptr->getType();
    if (auto *PT = dyn_cast<PointerType>(ptrTy)) {
      if (PT->getAddressSpace() == AS::Device)
        ty = F32;
    }
  }
}

// ── Remap i1 → i8 ───────────────────────────────────────────────────────
//
// Metal has no i1 memory type. Pointers to i1 crash the GPU JIT.
// Remap to i8 (booleans are i8 in Metal memory).

void PointeeTypeMap::remapI1ToI8(Module &M) {
  Type *I8 = Type::getInt8Ty(M.getContext());
  for (auto &[ptr, ty] : map) {
    if (ty && ty->isIntegerTy(1))
      ty = I8;
  }
}

// ── Initial analysis: scan all pointers and infer types ──────────────────

PointeeTypeMap PointeeTypeAnalysis::run(Module &M,
                                         ModuleAnalysisManager &AM) {
  PointeeTypeMap ptm;

  // Phase 1: Function parameters
  for (auto &F : M) {
    for (auto &Arg : F.args()) {
      if (!Arg.getType()->isPointerTy())
        continue;
      if (auto *ty = PointeeTypeMap::inferFromUsage(&Arg))
        ptm.set(&Arg, ty);
    }
  }

  // Phase 2: Instructions that produce pointers
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (!I.getType()->isPointerTy())
          continue;
        if (auto *ty = PointeeTypeMap::inferFromUsage(&I))
          ptm.set(&I, ty);
      }
    }
  }

  // Phase 2b: Force float* for device pointer phi nodes.
  // Non-float typed device pointers in phi nodes crash Metal GPU JIT.
  {
    Type *F32 = Type::getFloatTy(M.getContext());
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB) {
          auto *PN = dyn_cast<PHINode>(&I);
          if (!PN || !PN->getType()->isPointerTy()) continue;
          if (PN->getType()->getPointerAddressSpace() != AS::Device) continue;
          ptm.set(PN, F32);
        }
  }

  // Phase 3: Global variables (TG and device)
  for (auto &GV : M.globals()) {
    if (GV.getType()->isPointerTy()) {
      // For globals, the pointee type is the value type
      ptm.set(&GV, GV.getValueType());
    }
  }

  return ptm;
}

} // namespace metalir
