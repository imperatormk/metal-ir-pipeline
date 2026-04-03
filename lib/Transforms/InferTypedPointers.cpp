// ═══════════════════════════════════════════════════════════════════════
// Pass 15: InferTypedPointers
//
// Metal GPU JIT requires typed pointers in bitcode. LLVM 19+ only has
// opaque pointers. This pass populates the PointeeTypeMap analysis with
// inferred types for all pointer values.
//
// Includes MMA-specific overrides (formerly Pass 16: MMATypedPointers):
// when MMA intrinsics are present, all device ptrs → float*, MMA
// load/store params → float*, call site args → float*.
// ═══════════════════════════════════════════════════════════════════════

#include "metal-ir/Pipeline.h"
#include "metal-ir/KernelProfile.h"
#include "metal-ir/MetalConstraints.h"
#include "metal-ir/PointeeTypeMap.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

static constexpr const char *kMMALoad = "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
static constexpr const char *kMMAStore = "air.simdgroup_matrix_8x8_store.v64f32.p3f32";
static constexpr const char *kMMALoadDev = "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
static constexpr const char *kMMAStoreDev = "air.simdgroup_matrix_8x8_store.v64f32.p1f32";

bool InferTypedPointersPass::needsRun(Module &M) {
  // Always useful to run — populates PointeeTypeMap for bitcode emission
  return true;
}

PreservedAnalyses InferTypedPointersPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  // Get or create the pointee type map.
  // PointeeTypeAnalysis does the initial inference pass.
  // We refine it here with Metal-specific rules.
  auto &PTM = AM.getResult<PointeeTypeAnalysis>(M);
  auto &profiles = AM.getResult<KernelProfileAnalysis>(M);

  MetalConstraints constraints;
  constraints.hasMMA = AM.getResult<MMAPresenceAnalysis>(M).hasMMA;
  Type *F32 = Type::getFloatTy(M.getContext());

  // Phase 0: Map function pointers.
  for (auto &F : M) {
    if (!F.isDeclaration()) {
      PTM.set(&F, F.getFunctionType());
      break;
    }
  }

  // Phase 1: Fill gaps — pointers with no inferred type.
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (!I.getType()->isPointerTy() || PTM.has(&I))
          continue;

        // Follow phi incoming values
        if (auto *PHI = dyn_cast<PHINode>(&I)) {
          for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
            if (auto *ty = PTM.get(PHI->getIncomingValue(i))) {
              PTM.set(&I, ty);
              break;
            }
          }
        }

        // inttoptr: look at what the result is used for
        if (isa<IntToPtrInst>(&I)) {
          if (auto *ty = PointeeTypeMap::inferFromUsage(&I))
            PTM.set(&I, ty);
        }

        // GEP result: inherits source element type
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          PTM.set(&I, GEP->getResultElementType());
        }
      }
    }
  }

  // Phase 1b: Atomic intrinsic call site pointer fixup.
  //
  // Metal GPU JIT requires typed pointer types in bitcode to match the
  // intrinsic's expected type. For air.atomic.global.cmpxchg.weak.i32,
  // the device pointer param must be i32* — but the MLIR lowering often
  // passes a float* GEP result (the original buffer stores floats, the
  // CAS operates on i32 bitcasts). The reference Metal compiler inserts
  // an explicit `bitcast float* to i32*` before the call.
  //
  // In opaque-pointer LLVM IR, we can't insert ptr-to-ptr bitcasts
  // (LLVM removes them as no-ops), so we insert ptrtoint+inttoptr to
  // create a new SSA value, then set its PTM entry to the expected type.
  {
    Type *I32 = Type::getInt32Ty(M.getContext());
    Type *I64 = Type::getInt64Ty(M.getContext());
    SmallVector<std::pair<CallInst *, unsigned>, 8> fixups;

    for (auto &F : M) {
      for (auto &BB : F) {
        for (auto &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction()) continue;
          StringRef name = CI->getCalledFunction()->getName();
          if (!name.starts_with("air.atomic.global.")) continue;

          // Determine expected pointee type from intrinsic name suffix
          Type *expectedPointee = nullptr;
          if (name.ends_with(".i32"))
            expectedPointee = I32;
          else if (name.ends_with(".f32"))
            expectedPointee = F32;
          else
            continue;

          // Check device pointer arg (param 0 for all air.atomic.global.*)
          Value *ptrArg = CI->getArgOperand(0);
          if (!ptrArg->getType()->isPointerTy()) continue;
          unsigned addrSpace = ptrArg->getType()->getPointerAddressSpace();
          if (addrSpace != AS::Device && addrSpace != AS::Threadgroup) continue;

          Type *currentPointee = PTM.get(ptrArg);
          if (currentPointee == expectedPointee) continue;

          fixups.push_back({CI, 0});
        }
      }
    }

    for (auto &[CI, argIdx] : fixups) {
      StringRef name = CI->getCalledFunction()->getName();
      Type *expectedPointee = name.ends_with(".i32") ? I32 : F32;
      Value *ptrArg = CI->getArgOperand(argIdx);
      unsigned addrSpace = ptrArg->getType()->getPointerAddressSpace();

      IRBuilder<> B(CI);
      Value *asInt = B.CreatePtrToInt(ptrArg, I64);
      Value *newPtr = B.CreateIntToPtr(asInt, PointerType::get(M.getContext(), addrSpace));
      CI->setArgOperand(argIdx, newPtr);
      PTM.set(newPtr, expectedPointee);
    }
  }

  // Phase 2: Apply Metal constraints — i1* → i8*
  PTM.remapI1ToI8(M);

  // Phase 3: If MMA is used in a function, collapse only that function's
  // device pointers to float*. Non-MMA functions can keep narrower pointee
  // types and avoid the widening path.
  if (constraints.hasMMA) {
    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      auto it = profiles.find(&F);
      if (it == profiles.end() || !it->second.hasMMA)
        continue;
      for (auto &Arg : F.args())
        if (Arg.getType()->isPointerTy() &&
            Arg.getType()->getPointerAddressSpace() == AS::Device)
          PTM.set(&Arg, F32);
      for (auto &BB : F)
        for (auto &I : BB)
          if (I.getType()->isPointerTy() &&
              I.getType()->getPointerAddressSpace() == AS::Device)
            PTM.set(&I, F32);
    }
  }

  // Phase 4: Default unresolved device pointers to float* in MMA functions.
  for (auto &F : M) {
    auto it = profiles.find(&F);
    if (it == profiles.end() || !it->second.hasMMA)
      continue;
    for (auto &Arg : F.args()) {
      if (!Arg.getType()->isPointerTy() || PTM.has(&Arg))
        continue;
      if (auto *PT = dyn_cast<PointerType>(Arg.getType())) {
        if (PT->getAddressSpace() == AS::Device)
          PTM.set(&Arg, F32);
      }
    }
  }

  // Phase 5: MMA-specific overrides (formerly MMATypedPointersPass)
  if (constraints.hasMMA) {
    // MMA load/store declaration params → typed pointer.
    // p3f32/p1f32 variants get float*, p1f16 variants get half*.
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
        else if (name.contains("p1bf16") &&
                 name.starts_with("air.simdgroup_matrix_8x8_"))
          ptrPointee = BF16;
        else if (name.contains("p3f16") &&
                 name.starts_with("air.simdgroup_matrix_8x8_"))
          ptrPointee = F16;
        else if (name.contains("p3bf16") &&
                 name.starts_with("air.simdgroup_matrix_8x8_"))
          ptrPointee = BF16;
        if (ptrPointee) {
          for (auto &Arg : F.args())
            if (Arg.getType()->isPointerTy())
              PTM.set(&Arg, ptrPointee);
        }
      }
    }

    // MMA kernel device pointer args → float*
    for (auto &F : M) {
      if (F.isDeclaration()) continue;
      auto it = profiles.find(&F);
      if (it == profiles.end() || !it->second.hasMMA)
        continue;
      for (auto &Arg : F.args())
        if (Arg.getType()->isPointerTy() &&
            Arg.getType()->getPointerAddressSpace() == AS::Device)
          PTM.set(&Arg, F32);
    }

    // MMA call site pointer operands → appropriate type
    {
      Type *F16 = Type::getHalfTy(M.getContext());
      Type *BF16 = Type::getBFloatTy(M.getContext());
      for (auto &F : M) {
        for (auto &BB : F) {
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
              ptrPointee = F16;
            else if (name.contains("p1bf16") || name.contains("p3bf16"))
              ptrPointee = BF16;
            else if (name.contains("p3f32"))
              ptrPointee = F32;
            if (ptrPointee)
              for (unsigned i = 0; i < CI->arg_size(); i++)
                if (CI->getArgOperand(i)->getType()->isPointerTy())
                  PTM.set(CI->getArgOperand(i), ptrPointee);
          }
        }
      }
    }
  }

  // Phase 6: Async copy intrinsic pointer types.
  //
  // MUST run AFTER Phase 3 (MMA collapse) and Phase 5 (MMA overrides)
  // because those phases clobber all device pointer PTM entries to float*.
  // Async copy intrinsics require i8* for both TG and device pointer params,
  // and event_t* for return types and event storage.
  //
  // air.simdgroup_async_copy_2d returns %event_t addrspace(3)* — an opaque
  // event handle stored in threadgroup memory. The __tg_async_events global
  // stores these event pointers. We must:
  //   1. Create the %event_t opaque struct type if not present
  //   2. Set return type of async_copy intrinsics to event_t
  //   3. Set TG pointer params to i8* (raw byte copies)
  //   4. Set device pointer params to i8* (raw byte copies)
  //   5. Set __tg_async_events GEP/bitcast types to event_t
  //   6. Set wait_simdgroup_events pointer param to event_t
  {
    bool hasAsyncCopy = false;
    for (auto &F : M)
      if (F.getName().starts_with("air.simdgroup_async_copy_2d."))
        hasAsyncCopy = true;

    if (hasAsyncCopy) {
      Type *I8 = Type::getInt8Ty(M.getContext());

      // Get or create %event_t = type opaque
      StructType *eventTy = StructType::getTypeByName(M.getContext(), "event_t");
      if (!eventTy)
        eventTy = StructType::create(M.getContext(), "event_t");

      for (auto &F : M) {
        if (!F.isDeclaration()) continue;
        StringRef name = F.getName();

        if (name.starts_with("air.simdgroup_async_copy_2d.")) {
          // Return type: %event_t* (pointer to opaque event in TG)
          for (auto &FN : M)
            for (auto &BB : FN)
              for (auto &I : BB)
                if (auto *CI = dyn_cast<CallInst>(&I))
                  if (CI->getCalledFunction() == &F)
                    PTM.set(CI, eventTy);

          // Pointer params: TG ptr (param 2) = i8*, device ptr (param 6) = i8*
          // (they copy raw bytes — sizeof/alignof are explicit params 0,1)
          for (unsigned i = 0; i < F.arg_size(); i++) {
            auto &Arg = *F.getArg(i);
            if (!Arg.getType()->isPointerTy()) continue;
            PTM.set(&Arg, I8);
            // Also set call site arg values — overrides MMA's float*
            for (auto *U : F.users())
              if (auto *CI = dyn_cast<CallInst>(U))
                if (i < CI->arg_size())
                  PTM.set(CI->getArgOperand(i), I8);
          }
        }

        if (name == "air.wait_simdgroup_events") {
          // Param 1: pointer to event_t pointer storage.
          // The data at the pointed-to location is event_t addrspace(3)*
          // (an event pointer), so the pointee type is ptr addrspace(3).
          // The inner ptr resolves to event_t via inferredPointee in the
          // ValueEnumerator, giving: POINTER(POINTER(event_t, 3), 3).
          Type *ptrAs3 = PointerType::get(M.getContext(), 3);
          if (F.arg_size() >= 2) {
            auto &Arg = *F.getArg(1);
            if (Arg.getType()->isPointerTy()) {
              PTM.set(&Arg, ptrAs3);
              for (auto *U : F.users())
                if (auto *CI = dyn_cast<CallInst>(U))
                  if (CI->arg_size() >= 2)
                    PTM.set(CI->getArgOperand(1), ptrAs3);
            }
          }
        }
      }

      // Event storage: either __tg_async_events TG global (before
      // AsyncEventToAlloca) or ev_alloca stack alloca (after).
      Type *ptrAs3_ev = PointerType::get(M.getContext(), 3);
      if (auto *GV = M.getNamedGlobal("__tg_async_events")) {
        PTM.set(GV, GV->getValueType());
        for (auto *U : GV->users()) {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
            PTM.set(GEP, ptrAs3_ev);
            for (auto *GU : GEP->users())
              if (auto *BC = dyn_cast<BitCastInst>(GU))
                if (BC->getSrcTy() == BC->getDestTy())
                  PTM.set(BC, ptrAs3_ev);
          }
        }
      }
      // Event alloca: any alloca of ptr addrspace(3) stores event pointers.
      // Handles both named (ev_alloca from AsyncEventToAlloca) and unnamed
      // (from MLIR lowering) allocas.
      for (auto &FN : M) {
        for (auto &BB : FN) {
          for (auto &I : BB) {
            auto *AI = dyn_cast<AllocaInst>(&I);
            if (!AI) continue;
            // Check allocated type: ptr addrspace(3) or [N x ptr addrspace(3)]
            Type *allocTy = AI->getAllocatedType();
            bool isEventStorage = false;
            if (allocTy->isPointerTy() &&
                allocTy->getPointerAddressSpace() == 3)
              isEventStorage = true;
            if (auto *AT = dyn_cast<ArrayType>(allocTy))
              if (AT->getElementType()->isPointerTy() &&
                  AT->getElementType()->getPointerAddressSpace() == 3)
                isEventStorage = true;
            if (isEventStorage)
              PTM.set(AI, ptrAs3_ev);
          }
        }
      }
    }
  }

  // Phase 7: Fix up ptr-to-ptr bitcasts that serve as typed pointer transitions.
  //
  // WidenDeviceLoads Phase B inserts identity bitcasts (ptr→ptr in opaque IR)
  // before non-float device loads from phi pointers. These bitcasts exist so
  // the serializer can assign a different typed pointer (e.g., half*) to the
  // bitcast result vs. the phi source (float*).
  //
  // The MMA collapse (Phase 3) and MMA overrides (Phase 5) may have clobbered
  // the PTM entries for these bitcasts to float*. Re-infer from usage: if a
  // ptr-to-ptr bitcast feeds a non-float load or store, set its PTM to the
  // load/store element type.
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC || !BC->getType()->isPointerTy()) continue;
        if (BC->getSrcTy() != BC->getDestTy()) continue;
        unsigned addrSpace = BC->getType()->getPointerAddressSpace();
        if (addrSpace != AS::Device) continue;

        // Check what this bitcast feeds — look for loads/stores
        for (auto *U : BC->users()) {
          if (auto *LI = dyn_cast<LoadInst>(U)) {
            if (!LI->getType()->isFloatTy()) {
              PTM.set(BC, LI->getType());
              break;
            }
          }
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            if (SI->getPointerOperand() == BC &&
                !SI->getValueOperand()->getType()->isFloatTy()) {
              PTM.set(BC, SI->getValueOperand()->getType());
              break;
            }
          }
        }
      }
    }
  }

  return PreservedAnalyses::all();
}

} // namespace metalir
