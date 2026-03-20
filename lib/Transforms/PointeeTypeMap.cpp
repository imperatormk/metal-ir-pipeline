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
  // NOTE: Do NOT follow atomic intrinsic calls through GEP chains.
  // When a float buffer pointer goes through a float GEP to a CAS call
  // (which operates on i32), the GEP source type (float) must win.
  // The atomic type mismatch is resolved separately by
  // InferTypedPointersPass Phase 1b (ptrtoint+inttoptr insertion).
  Type *gepType = nullptr;
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
        // Only use atomic type if the pointer is NOT a GEP result.
        // GEP results must keep their source element type for consistency;
        // the atomic type mismatch is handled by inserting ptrtoint+inttoptr.
        if (!isa<GetElementPtrInst>(ptr) &&
            name.starts_with("air.atomic.")) {
          if (name.ends_with(".i32"))
            return Type::getInt32Ty(ptr->getContext());
          else if (name.ends_with(".f32"))
            return Type::getFloatTy(ptr->getContext());
        }
      }
    }
  }
  return gepType;
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
//
// This analysis MUST be self-contained — it may be re-run after pipeline
// passes invalidate it. All Metal-specific overrides (MMA, async copy)
// must be here, not only in InferTypedPointersPass.

static constexpr const char *kMMALoad =
    "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
static constexpr const char *kMMAStore =
    "air.simdgroup_matrix_8x8_store.v64f32.p3f32";
static constexpr const char *kMMALoadDev =
    "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
static constexpr const char *kMMAStoreDev =
    "air.simdgroup_matrix_8x8_store.v64f32.p1f32";

PointeeTypeMap PointeeTypeAnalysis::run(Module &M,
                                         ModuleAnalysisManager &AM) {
  PointeeTypeMap ptm;
  Type *F32 = Type::getFloatTy(M.getContext());
  Type *I8 = Type::getInt8Ty(M.getContext());

  // Detect MMA and async copy presence
  bool hasMMA = false;
  bool hasAsyncCopy = false;
  for (auto &F : M) {
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      hasMMA = true;
    if (F.getName().starts_with("air.simdgroup_async_copy_2d."))
      hasAsyncCopy = true;
  }

  // Phase 1: Function parameters — infer from usage
  for (auto &F : M)
    for (auto &Arg : F.args())
      if (Arg.getType()->isPointerTy())
        if (auto *ty = PointeeTypeMap::inferFromUsage(&Arg))
          ptm.set(&Arg, ty);

  // Phase 2: Instructions that produce pointers
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (!I.getType()->isPointerTy()) continue;
        if (auto *ty = PointeeTypeMap::inferFromUsage(&I))
          ptm.set(&I, ty);
      }

  // Phase 2b: Force float* for device pointer phi nodes
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN || !PN->getType()->isPointerTy()) continue;
        if (PN->getType()->getPointerAddressSpace() != AS::Device) continue;
        ptm.set(PN, F32);
      }

  // Phase 3: Global variables
  for (auto &GV : M.globals())
    if (GV.getType()->isPointerTy())
      ptm.set(&GV, GV.getValueType());

  // Phase 4: Fill gaps — phi incoming, inttoptr, GEP
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (!I.getType()->isPointerTy() || ptm.has(&I)) continue;
        if (auto *PHI = dyn_cast<PHINode>(&I))
          for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i)
            if (auto *ty = ptm.get(PHI->getIncomingValue(i))) {
              ptm.set(&I, ty);
              break;
            }
        if (isa<IntToPtrInst>(&I))
          if (auto *ty = PointeeTypeMap::inferFromUsage(&I))
            ptm.set(&I, ty);
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          ptm.set(&I, GEP->getResultElementType());
      }

  // Phase 5: i1* → i8*
  ptm.remapI1ToI8(M);

  // Phase 6: MMA — collapse device pointers to float*
  if (hasMMA) {
    ptm.collapseDevicePointersToFloat(M);

    // Default unresolved device pointers to float*
    for (auto &F : M)
      for (auto &Arg : F.args())
        if (Arg.getType()->isPointerTy() &&
            Arg.getType()->getPointerAddressSpace() == AS::Device &&
            !ptm.has(&Arg))
          ptm.set(&Arg, F32);

    // MMA declaration params → typed pointer (float*/half*/bfloat*)
    {
      Type *F16 = Type::getHalfTy(M.getContext());
      Type *BF16 = Type::getBFloatTy(M.getContext());
      for (auto &F : M) {
        if (!F.isDeclaration()) continue;
        StringRef name = F.getName();
        Type *ptrPointee = nullptr;
        if (name == kMMALoad || name == kMMAStore ||
            name == kMMALoadDev || name == kMMAStoreDev)
          ptrPointee = F32;
        else if (name.contains("p1f16") &&
                 name.starts_with("air.simdgroup_matrix_8x8_"))
          ptrPointee = F16;
        else if ((name.contains("p1bf16") || name.contains("p3bf16")) &&
                 name.starts_with("air.simdgroup_matrix_8x8_"))
          ptrPointee = BF16;
        else if ((name.contains("p1f16") || name.contains("p3f16")) &&
                 name.starts_with("air.simdgroup_matrix_8x8_"))
          ptrPointee = F16;
        if (ptrPointee)
          for (auto &Arg : F.args())
            if (Arg.getType()->isPointerTy())
              ptm.set(&Arg, ptrPointee);
      }
    }

    // Kernel device pointer args → float*
    for (auto &F : M)
      if (!F.isDeclaration())
        for (auto &Arg : F.args())
          if (Arg.getType()->isPointerTy() &&
              Arg.getType()->getPointerAddressSpace() == AS::Device)
            ptm.set(&Arg, F32);

    // MMA call site pointer operands → typed pointer
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction()) continue;
          StringRef name = CI->getCalledFunction()->getName();
          if (!name.starts_with("air.simdgroup_matrix_8x8_")) continue;
          Type *ptrPointee = nullptr;
          if (name == kMMALoad || name == kMMAStore ||
              name == kMMALoadDev || name == kMMAStoreDev)
            ptrPointee = F32;
          else if (name.contains("p1f16") || name.contains("p3f16"))
            ptrPointee = Type::getHalfTy(M.getContext());
          else if (name.contains("p1bf16") || name.contains("p3bf16"))
            ptrPointee = Type::getBFloatTy(M.getContext());
          else if (name.contains("p3f32"))
            ptrPointee = F32;
          if (ptrPointee)
            for (unsigned i = 0; i < CI->arg_size(); i++)
              if (CI->getArgOperand(i)->getType()->isPointerTy())
                ptm.set(CI->getArgOperand(i), ptrPointee);
        }
  }

  // Phase 7: Async copy overrides (AFTER MMA collapse, re-applies i8*)
  if (hasAsyncCopy) {
    StructType *eventTy = StructType::getTypeByName(M.getContext(), "event_t");
    if (!eventTy)
      eventTy = StructType::create(M.getContext(), "event_t");

    for (auto &F : M) {
      if (!F.isDeclaration()) continue;
      StringRef name = F.getName();

      if (name.starts_with("air.simdgroup_async_copy_2d.")) {
        // Return type → event_t (set on call results)
        for (auto &FN : M)
          for (auto &BB : FN)
            for (auto &I : BB)
              if (auto *CI = dyn_cast<CallInst>(&I))
                if (CI->getCalledFunction() == &F)
                  ptm.set(CI, eventTy);
        // Pointer params → i8* (byte copy)
        for (unsigned i = 0; i < F.arg_size(); i++) {
          auto &Arg = *F.getArg(i);
          if (!Arg.getType()->isPointerTy()) continue;
          ptm.set(&Arg, I8);
          for (auto *U : F.users())
            if (auto *CI = dyn_cast<CallInst>(U))
              if (i < CI->arg_size())
                ptm.set(CI->getArgOperand(i), I8);
        }
      }

      if (name == "air.wait_simdgroup_events") {
        // Param 1: pointer to event_t pointer storage
        Type *ptrAs3 = PointerType::get(M.getContext(), 3);
        if (F.arg_size() >= 2) {
          auto &Arg = *F.getArg(1);
          if (Arg.getType()->isPointerTy()) {
            ptm.set(&Arg, ptrAs3);
            for (auto *U : F.users())
              if (auto *CI = dyn_cast<CallInst>(U))
                if (CI->arg_size() >= 2)
                  ptm.set(CI->getArgOperand(1), ptrAs3);
          }
        }
      }
    }

    // Event alloca: stores event_t pointers
    Type *ptrAs3 = PointerType::get(M.getContext(), 3);
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *AI = dyn_cast<AllocaInst>(&I))
            if (AI->getAllocatedType()->isPointerTy() &&
                AI->getAllocatedType()->getPointerAddressSpace() == 3)
              ptm.set(AI, ptrAs3);

    // Propagate event_t through identity bitcasts
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB) {
          auto *BC = dyn_cast<BitCastInst>(&I);
          if (!BC || BC->getSrcTy() != BC->getDestTy()) continue;
          if (!BC->getType()->isPointerTy()) continue;
          if (auto *srcTy = ptm.get(BC->getOperand(0)))
            if (srcTy == eventTy || isa<PointerType>(srcTy))
              ptm.set(BC, srcTy);
        }
  }

  // Phase 8: Fix up ptr-to-ptr bitcasts for typed pointer transitions.
  // WidenDeviceLoads inserts identity bitcasts (ptr→ptr) before non-float
  // device loads from phi pointers. MMA collapse clobbers their PTM to float*.
  // Re-infer from load/store usage.
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC || !BC->getType()->isPointerTy()) continue;
        if (BC->getSrcTy() != BC->getDestTy()) continue;
        if (BC->getType()->getPointerAddressSpace() != AS::Device) continue;
        for (auto *U : BC->users()) {
          if (auto *LI = dyn_cast<LoadInst>(U)) {
            if (!LI->getType()->isFloatTy()) {
              ptm.set(BC, LI->getType());
              break;
            }
          }
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            if (SI->getPointerOperand() == BC &&
                !SI->getValueOperand()->getType()->isFloatTy()) {
              ptm.set(BC, SI->getValueOperand()->getType());
              break;
            }
          }
        }
      }

  // Phase 9: Function pointer
  for (auto &F : M)
    if (!F.isDeclaration()) {
      ptm.set(&F, F.getFunctionType());
      break;
    }

  return ptm;
}

} // namespace metalir
