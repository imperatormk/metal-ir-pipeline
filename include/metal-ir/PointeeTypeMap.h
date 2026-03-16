#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Type.h"

namespace metalir {

// ── Pointee Type Map ─────────────────────────────────────────────────────
//
// Metal GPU JIT requires typed pointers in bitcode (POINTER records with
// pointee type), but LLVM 19+ only has opaque pointers (ptr addrspace(N)).
//
// This side table tracks what each pointer "actually points to" as the IR
// passes transform it. Passes that need typed pointer info read/write this
// map. The custom bitcode writer consumes it to emit typed POINTER records.
//
// Example flow:
//
//   LLVM Module (opaque ptrs):
//     %p = getelementptr float, ptr addrspace(1) %buf, i32 %idx
//     %v = load float, ptr addrspace(1) %p
//
//   PointeeTypeMap after InferTypedPointersPass:
//     %buf → float    (inferred from GEP source type)
//     %p   → float    (inferred from load type)
//
//   Custom bitcode writer:
//     %buf emitted as float addrspace(1)* (POINTER record with pointee=float)
//     %p   emitted as float addrspace(1)*
//
// This replaces MetalASM's approach of having typed pointers in its own IR
// representation. The LLVM Module stays valid (opaque ptrs), and the type
// info lives in this side table.
//
// ── Key rules from Metal GPU JIT ─────────────────────────────────────────
//
// 1. ALL device pointers (addrspace 1) must be typed in bitcode
// 2. When MMA intrinsics are present, ALL device ptrs must be float*
//    (even half/i32 — loads get widened to float by WidenDeviceLoadsPass)
// 3. TG pointers (addrspace 3) must be typed (usually float*)
// 4. i1* crashes GPU JIT — remap to i8*
// 5. Constant buffer ptrs (addrspace 2) follow scalar packing rules

class PointeeTypeMap {
public:
  // Set the pointee type for a pointer value.
  void set(llvm::Value *ptr, llvm::Type *pointeeTy) {
    map[ptr] = pointeeTy;
  }

  // Get the pointee type, or nullptr if unknown.
  llvm::Type *get(llvm::Value *ptr) const {
    auto it = map.find(ptr);
    return it != map.end() ? it->second : nullptr;
  }

  // Check if a pointer has a known pointee type.
  bool has(llvm::Value *ptr) const {
    return map.count(ptr);
  }

  // Remove a pointer's entry (used when erasing instructions).
  void remove(llvm::Value *ptr) {
    map.erase(ptr);
  }

  // Iterate all entries.
  auto begin() const { return map.begin(); }
  auto end() const { return map.end(); }
  size_t size() const { return map.size(); }

  // ── Inference helpers ────────────────────────────────────────────────

  // Infer pointee type from how a pointer is used (loads, stores, GEPs).
  // Returns nullptr if no usage gives a clear type.
  static llvm::Type *inferFromUsage(llvm::Value *ptr);

  // Apply the "MMA present → all device ptrs are float*" rule.
  void collapseDevicePointersToFloat(llvm::Module &M);

  // Apply the "i1* → i8*" rule.
  void remapI1ToI8(llvm::Module &M);

private:
  llvm::DenseMap<llvm::Value *, llvm::Type *> map;
};

// ── LLVM Analysis wrapper ────────────────────────────────────────────────
// Shared across passes via the AnalysisManager.

struct PointeeTypeAnalysis : llvm::AnalysisInfoMixin<PointeeTypeAnalysis> {
  using Result = PointeeTypeMap;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static llvm::AnalysisKey Key;
};

} // namespace metalir
