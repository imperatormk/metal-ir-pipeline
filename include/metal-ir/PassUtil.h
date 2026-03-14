#pragma once

#include "llvm/IR/PassManager.h"

namespace metalir {

/// Return PreservedAnalyses for a pass that changed instructions
/// but did not modify the CFG (no BBs added/removed, no branch changes).
inline llvm::PreservedAnalyses changedNonCFG() {
  llvm::PreservedAnalyses PA;
  PA.preserveSet<llvm::CFGAnalyses>();
  return PA;
}

/// Standard return for passes that track a `changed` bool.
/// Preserves CFG when changed, preserves all when not.
inline llvm::PreservedAnalyses preserveIf(bool changed) {
  if (!changed) return llvm::PreservedAnalyses::all();
  return changedNonCFG();
}

} // namespace metalir
