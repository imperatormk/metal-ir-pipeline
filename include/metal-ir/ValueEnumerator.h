#pragma once

#include "metal-ir/PointeeTypeMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include <vector>

namespace metalir {

/// Type table entry: LLVM Type* + pointee (for pointer types).
/// Non-pointer types have pointee = nullptr.
/// This allows multiple typed pointer entries for the same LLVM ptr type
/// (e.g., float addrspace(3)* AND [128 x float] addrspace(3)*).
struct TypeEntry {
  llvm::Type *type;
  llvm::Type *pointee; // nullptr for non-pointers

  bool operator==(const TypeEntry &O) const {
    return type == O.type && pointee == O.pointee;
  }
};

} // namespace metalir

// DenseMap support for TypeEntry
namespace llvm {
template <> struct DenseMapInfo<metalir::TypeEntry> {
  static metalir::TypeEntry getEmptyKey() {
    return {DenseMapInfo<Type *>::getEmptyKey(), nullptr};
  }
  static metalir::TypeEntry getTombstoneKey() {
    return {DenseMapInfo<Type *>::getTombstoneKey(), nullptr};
  }
  static unsigned getHashValue(const metalir::TypeEntry &E) {
    return hash_combine(DenseMapInfo<Type *>::getHashValue(E.type),
                        DenseMapInfo<Type *>::getHashValue(E.pointee));
  }
  static bool isEqual(const metalir::TypeEntry &A,
                       const metalir::TypeEntry &B) {
    return A == B;
  }
};
} // namespace llvm

namespace metalir {

class ValueEnumerator {
public:
  // Type table: each entry is (Type*, pointee) for correct typed pointer emission
  std::vector<TypeEntry> types;
  llvm::DenseMap<TypeEntry, unsigned> typeMap;

  std::vector<const llvm::Value *> globalValues;
  llvm::DenseMap<const llvm::Value *, unsigned> globalValueMap;

  std::vector<const llvm::Constant *> moduleConstants;
  llvm::DenseMap<const llvm::Constant *, unsigned> moduleConstMap;

  const PointeeTypeMap &PTM;
  // Inferred pointee types for pointer Types (from first usage seen)
  mutable llvm::DenseMap<llvm::Type *, llvm::Type *> inferredPointee;

  /// Per-function-type param type indices. When a FunctionType has pointer
  /// params with different pointee types (e.g., float* and i32*), the generic
  /// typeIdx(ptrType) would return the same index for all. This map stores
  /// the correct per-param type table index for each FunctionType.
  llvm::DenseMap<llvm::FunctionType *, llvm::SmallVector<unsigned, 8>>
      funcTypeParamIndices;

  ValueEnumerator(llvm::Module &M, const PointeeTypeMap &PTM);

  /// Get type index for a non-pointer type.
  unsigned typeIdx(llvm::Type *T);

  /// Get type index for a pointer type with specific pointee.
  unsigned ptrTypeIdx(llvm::Type *PtrTy, llvm::Type *pointee);

  /// Get type index for a global variable's pointer type (uses value type as pointee).
  unsigned globalPtrTypeIdx(const llvm::GlobalVariable *GV);

  unsigned globalIdx(const llvm::Value *V) const;
  unsigned moduleConstIdx(const llvm::Constant *C) const;
  bool hasModuleConst(const llvm::Constant *C) const;

  /// Default pointee for a pointer type (from inference / fallback).
  llvm::Type *pointeeType(llvm::Type *PtrTy) const;

  /// Pointee for a specific value (PTM → inference → default).
  llvm::Type *pointeeTypeForValue(const llvm::Value *V) const;

private:
  unsigned addType(llvm::Type *T);
  unsigned addFunctionType(llvm::FunctionType *FT, const llvm::Function *F);
  unsigned addEntry(TypeEntry E);
  void addModuleConstant(const llvm::Constant *C);
  void collectMetadataConstants(const llvm::MDNode *N);
};

} // namespace metalir
