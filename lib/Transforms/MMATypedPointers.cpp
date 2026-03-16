// Pass 16: MMATypedPointers — MERGED into InferTypedPointers (Pass 15).
//
// This pass is now a no-op. All MMA-specific typed pointer logic has been
// folded into InferTypedPointersPass::run() as Phase 5. Kept as a stub
// for pipeline compatibility until removed from Pipeline.h.

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "metal-ir/PointeeTypeMap.h"

using namespace llvm;
namespace metalir {

bool MMATypedPointersPass::needsRun(Module &M) { return false; }

PreservedAnalyses MMATypedPointersPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  return PreservedAnalyses::all();
}

} // namespace metalir
