// Pass 16: MMATypedPointers — ensure MMA intrinsic pointer params have
// correct pointee types in the PointeeTypeMap.
//
// When MMA intrinsics are present:
// - simdgroup_matrix_8x8_load: ptr addrspace(3) param → float* (TG)
// - simdgroup_matrix_8x8_store: ptr addrspace(3) param → float* (TG)
// - All device (addrspace 1) pointers → float* (GPU JIT crashes on half*)
// - All atomic intrinsic ptr params → typed
//
// In LLVM's in-memory IR, pointers are always opaque. This pass records
// the intended pointee types in PointeeTypeMap, which the bitcode emitter
// uses to emit typed POINTER records.
//
// Queries MMAPresenceAnalysis — no-op if no MMA intrinsics.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "metal-ir/PointeeTypeMap.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

static constexpr const char *kMMALoad = "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
static constexpr const char *kMMAStore = "air.simdgroup_matrix_8x8_store.v64f32.p3f32";
static constexpr const char *kMMAMul = "air.simdgroup_matrix_8x8_multiply_accumulate";

bool MMATypedPointersPass::needsRun(Module &M) {
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      return true;
  return false;
}

PreservedAnalyses MMATypedPointersPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  auto &MMA = MAM.getResult<MMAPresenceAnalysis>(M);
  if (!MMA.hasMMA) return PreservedAnalyses::all();

  auto &PTM = MAM.getResult<PointeeTypeAnalysis>(M);
  Type *F32 = Type::getFloatTy(M.getContext());
  bool changed = false;

  // MMA load/store pointer params → float*
  for (auto &F : M) {
    if (!F.isDeclaration()) continue;
    StringRef name = F.getName();
    if (name == kMMALoad || name == kMMAStore) {
      for (auto &Arg : F.args()) {
        if (Arg.getType()->isPointerTy()) {
          PTM.set(&Arg, F32);
          changed = true;
        }
      }
    }
  }

  // When MMA present, ALL device pointer params on kernel functions → float*
  // (GPU JIT crashes on half*/i32* device ptrs with MMA)
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    for (auto &Arg : F.args()) {
      if (Arg.getType()->isPointerTy() &&
          Arg.getType()->getPointerAddressSpace() == 1) {
        PTM.set(&Arg, F32);
        changed = true;
      }
    }
  }

  // Fix call sites: record pointee types for pointer operands
  // passed to MMA intrinsics
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction()) continue;
        StringRef name = CI->getCalledFunction()->getName();
        if (name == kMMALoad || name == kMMAStore) {
          for (unsigned i = 0; i < CI->arg_size(); i++) {
            Value *arg = CI->getArgOperand(i);
            if (arg->getType()->isPointerTy()) {
              PTM.set(arg, F32);
              changed = true;
            }
          }
        }
      }
    }
  }

  return preserveIf(changed);
}

} // namespace metalir
