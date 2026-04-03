// KernelProfile analysis — compute per-function facts once.
//
// Scans each function for memory access patterns, index usage, and
// type usage. Passes query the result instead of re-scanning.

#include "metal-ir/KernelProfile.h"
#include "metal-ir/AIRIntrinsics.h"
#include "metal-ir/IRUtil.h"

using namespace llvm;

namespace metalir {

AnalysisKey KernelProfileAnalysis::Key;

KernelProfileAnalysis::Result
KernelProfileAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  Result profiles;

  for (auto &F : M) {
    if (F.isDeclaration()) continue;

    KernelFacts facts;
    for (auto &BB : F) {
      for (auto &I : BB) {
        // Store checks
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          unsigned addrSpace = SI->getPointerAddressSpace();
          if (addrSpace == AS::Device) {
            facts.hasDeviceStore = true;
            if (!SI->getValueOperand()->getType()->isFloatTy())
              facts.hasNonFloatDeviceStore = true;
          }
          if (addrSpace == AS::Threadgroup)
            facts.hasTGStore = true;
        }

        // Load checks
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          unsigned addrSpace = LI->getPointerAddressSpace();
          if (addrSpace == AS::Device) {
            facts.hasDeviceLoad = true;
            if (!LI->getType()->isFloatTy())
              facts.hasNonFloatDeviceLoad = true;
          }
          if (addrSpace == AS::Threadgroup)
            facts.hasTGLoad = true;
        }

        // PHI checks
        if (auto *PHI = dyn_cast<PHINode>(&I)) {
          if (PHI->getType()->isStructTy())
            facts.hasStructPhi = true;
        }

        // Call checks — index usage and atomics
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (auto *Callee = CI->getCalledFunction()) {
            StringRef name = Callee->getName();

            if (name.starts_with("air.simdgroup_matrix_8x8_"))
              facts.hasMMA = true;

            // Per-thread index
            if (name.starts_with(air::kCallTid) ||
                name.starts_with(air::kCallTidTG) ||
                name.starts_with(air::kCallSimdlane))
              facts.hasPerThreadIndex = true;

            // Program index
            if (name.starts_with(air::kCallPid))
              facts.hasProgramIndex = true;

            // Atomics
            if (name.starts_with("air.atomic."))
              facts.hasAtomics = true;

            // Atomic globals count as device writes
            if (name.starts_with("air.atomic.global"))
              facts.hasDeviceStore = true;
          }
        }
      }
    }

    profiles[&F] = facts;
  }

  return profiles;
}

} // namespace metalir
