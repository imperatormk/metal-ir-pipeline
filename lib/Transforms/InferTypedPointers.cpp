// ═══════════════════════════════════════════════════════════════════════
// Pass 15: InferTypedPointers
//
// Metal GPU JIT requires typed pointers in bitcode. LLVM 19+ only has
// opaque pointers. This pass populates the PointeeTypeMap analysis with
// inferred types for all pointer values.
//
// Unlike MetalASM which mutates pointer types in its own IR, we keep
// the LLVM Module unchanged (opaque ptrs) and store type info in the
// side table. The custom bitcode writer reads the side table.
//
// This pass implements the equivalent of MetalASM's
// transformInferOpaquePointerTypes (lines 1454-1830).
// ═══════════════════════════════════════════════════════════════════════

#include "metal-ir/Pipeline.h"
#include "metal-ir/PointeeTypeMap.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

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

  bool hasMMA = false;
  for (auto &F : M) {
    if (F.getName().contains("multiply_accumulate")) {
      hasMMA = true;
      break;
    }
  }

  // Phase 0: Map function pointers.
  // In LLVM opaque ptr world, all functions share `ptr` type.
  // Record the kernel function's type as the pointee for ptr-to-fn.
  for (auto &F : M) {
    if (!F.isDeclaration()) {
      PTM.set(&F, F.getFunctionType());
      break; // only need one (v1 FUNCTION records use fn type directly)
    }
  }

  // Phase 1: Fill gaps — pointers with no inferred type.
  // Try harder: follow phi chains, inttoptr sources, etc.
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

  // Phase 2: Apply Metal constraints
  // i1* → i8* (Metal has no i1 memory type)
  PTM.remapI1ToI8(M);

  // Phase 3: If MMA present, all device pointers → float*
  if (hasMMA)
    PTM.collapseDevicePointersToFloat(M);

  // Phase 4: Default unresolved device pointers to float*
  Type *F32 = Type::getFloatTy(M.getContext());
  for (auto &F : M) {
    for (auto &Arg : F.args()) {
      if (!Arg.getType()->isPointerTy() || PTM.has(&Arg))
        continue;
      if (auto *PT = dyn_cast<PointerType>(Arg.getType())) {
        if (PT->getAddressSpace() == 1) // device
          PTM.set(&Arg, F32);
      }
    }
  }

  return PreservedAnalyses::all(); // We only wrote to the side table
}

} // namespace metalir
