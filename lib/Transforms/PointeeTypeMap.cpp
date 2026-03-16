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
  // Primary: load/store types via recursive user walk (shared with TG passes)
  if (Type *T = inferElementType(ptr))
    return T;

  // Fallback 1: GEP source element type
  for (auto *U : ptr->users())
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
      return GEP->getSourceElementType();

  // Fallback 2: atomic intrinsic name → type
  for (auto *U : ptr->users()) {
    if (auto *CI = dyn_cast<CallInst>(U)) {
      if (auto *Callee = CI->getCalledFunction()) {
        StringRef name = Callee->getName();
        if (name.starts_with("air.atomic.")) {
          if (name.ends_with(".i32"))
            return Type::getInt32Ty(ptr->getContext());
          if (name.ends_with(".f32"))
            return Type::getFloatTy(ptr->getContext());
        }
      }
    }
  }
  return nullptr;
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
