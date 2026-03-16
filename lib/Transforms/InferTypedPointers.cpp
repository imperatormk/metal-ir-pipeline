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
#include "metal-ir/MetalConstraints.h"
#include "metal-ir/PointeeTypeMap.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

static constexpr const char *kMMALoad = "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
static constexpr const char *kMMAStore = "air.simdgroup_matrix_8x8_store.v64f32.p3f32";

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

  bool hasMMA = AM.getResult<MMAPresenceAnalysis>(M).hasMMA;
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

  // Phase 2: Apply Metal constraints — i1* → i8*
  PTM.remapI1ToI8(M);

  // Phase 3: If MMA present, all device pointers → float*
  if (hasMMA)
    PTM.collapseDevicePointersToFloat(M);

  // Phase 4: Default unresolved device pointers to float*
  for (auto &F : M) {
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
  if (hasMMA) {
    // MMA load/store declaration params → float*
    for (auto &F : M) {
      if (!F.isDeclaration()) continue;
      StringRef name = F.getName();
      if (name == kMMALoad || name == kMMAStore) {
        for (auto &Arg : F.args())
          if (Arg.getType()->isPointerTy())
            PTM.set(&Arg, F32);
      }
    }

    // ALL kernel device pointer args → float* (GPU JIT crashes on half*)
    for (auto &F : M) {
      if (F.isDeclaration()) continue;
      for (auto &Arg : F.args())
        if (Arg.getType()->isPointerTy() &&
            Arg.getType()->getPointerAddressSpace() == AS::Device)
          PTM.set(&Arg, F32);
    }

    // MMA call site pointer operands → float*
    for (auto &F : M) {
      for (auto &BB : F) {
        for (auto &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction()) continue;
          StringRef name = CI->getCalledFunction()->getName();
          if (name == kMMALoad || name == kMMAStore)
            for (unsigned i = 0; i < CI->arg_size(); i++)
              if (CI->getArgOperand(i)->getType()->isPointerTy())
                PTM.set(CI->getArgOperand(i), F32);
        }
      }
    }
  }

  return PreservedAnalyses::all();
}

} // namespace metalir
