#include "metal-ir/ValueEnumerator.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

ValueEnumerator::ValueEnumerator(Module &M, const PointeeTypeMap &PTM)
    : PTM(PTM) {
  auto &Ctx = M.getContext();

  // ── Phase 1: Infer pointee types for pointer Types ─────────────────

  // Function pointer: ptr as0 → kernel's function type
  Type *ptrAs0 = PointerType::get(Ctx, 0);
  for (auto &F : M)
    if (!F.isDeclaration()) {
      inferredPointee[ptrAs0] = F.getFunctionType();
      break;
    }

  // Device/TG pointers: infer from first arg usage
  for (auto &F : M)
    for (auto &Arg : F.args())
      if (Arg.getType()->isPointerTy() && !inferredPointee.count(Arg.getType()))
        if (auto *ty = PointeeTypeMap::inferFromUsage(const_cast<Argument*>(&Arg)))
          inferredPointee[Arg.getType()] = ty;

  // PTM overrides — but skip global variables (they get separate TypeEntry
  // via globalPtrTypeIdx, not the shared inferredPointee)
  for (auto &[V, T] : PTM)
    if (V->getType()->isPointerTy() && !isa<GlobalVariable>(V) &&
        !inferredPointee.count(V->getType()))
      inferredPointee[V->getType()] = T;

  // ── Phase 2: Enumerate types ───────────────────────────────────────

  addType(Type::getVoidTy(Ctx));
  addType(Type::getFloatTy(Ctx));

  // Enumerate function types — definitions first, then declarations.
  // Must process definitions first so their per-param pointee inference
  // populates funcTypeParamIndices before any recursive addType call
  // from declaration processing caches the function type with wrong params.
  for (auto &F : M)
    if (!F.isDeclaration())
      addFunctionType(F.getFunctionType(), &F);
  for (auto &F : M)
    if (F.isDeclaration())
      addFunctionType(F.getFunctionType(), &F);
  // Create function pointer types for definitions (kernels) only.
  // Declarations (intrinsics) don't need function pointers in Metal v1.
  for (auto &F : M)
    if (!F.isDeclaration())
      ptrTypeIdx(ptrAs0, F.getFunctionType());

  addType(Type::getMetadataTy(Ctx));
  addType(Type::getLabelTy(Ctx));

  // Global variable types — create per-global typed pointer entries
  for (auto &GV : M.globals()) {
    addType(GV.getValueType());
    globalPtrTypeIdx(&GV); // creates ptr(valueType, addrspace) entry
  }

  // Instruction result + operand types — enumerate ALL types used by
  // instructions so the type table is complete before emission.
  // For GEPs into arrays, the result pointer needs a separate typed entry.
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        addType(I.getType());
        // Operand types (including constants like i64 0 in GEPs)
        for (auto &Op : I.operands())
          if (!isa<BasicBlock>(Op))
            addType(Op->getType());
        // GEP result: create ptr(elementType, addrspace) entry
        // Use PTM override for device (AS 1) pointers (e.g., store float
        // through i8* GEP should produce float*, not i8*).
        // For TG (AS 3) byte globals, keep GEP's own result element type —
        // the byte global stays as [N x i8] and GEP results must be i8*.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (GEP->getType()->isPointerTy()) {
            Type *resultPointee = GEP->getResultElementType();
            unsigned AS = GEP->getType()->getPointerAddressSpace();
            if (AS != 3 || !resultPointee->isIntegerTy(8)) {
              if (auto *ptmTy = PTM.get(GEP))
                resultPointee = ptmTy;
            }
            ptrTypeIdx(PointerType::get(M.getContext(), AS), resultPointee);
          }
        }
        // Bitcast ptr→ptr: in Metal v1 these change typed pointer.
        // Create a separate typed pointer entry from PTM.
        if (auto *BC = dyn_cast<BitCastInst>(&I)) {
          if (BC->getType()->isPointerTy() && BC->getSrcTy() == BC->getDestTy()) {
            if (auto *ptmTy = PTM.get(BC)) {
              unsigned AS = BC->getType()->getPointerAddressSpace();
              ptrTypeIdx(PointerType::get(M.getContext(), AS), ptmTy);
            }
          }
        }
      }
    }
  }

  // ── Phase 3: Value IDs (globals first, then functions) ─────────────

  for (auto &GV : M.globals()) {
    globalValueMap[&GV] = globalValues.size();
    globalValues.push_back(&GV);
  }
  // Definitions first, then declarations (matches MetalASM ordering)
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    globalValueMap[&F] = globalValues.size();
    globalValues.push_back(&F);
  }
  for (auto &F : M) {
    if (!F.isDeclaration()) continue;
    globalValueMap[&F] = globalValues.size();
    globalValues.push_back(&F);
  }

  // ── Phase 4: Module constants ──────────────────────────────────────

  for (auto &NMD : M.named_metadata())
    for (unsigned i = 0; i < NMD.getNumOperands(); i++)
      collectMetadataConstants(NMD.getOperand(i));

  for (auto &GV : M.globals())
    if (GV.hasInitializer())
      addModuleConstant(GV.getInitializer());
}

// ═══════════════════════════════════════════════════════════════════════
// Type queries
// ═══════════════════════════════════════════════════════════════════════

unsigned ValueEnumerator::typeIdx(Type *T) {
  if (isa<PointerType>(T))
    return ptrTypeIdx(T, pointeeType(T));
  TypeEntry E{T, nullptr};
  auto it = typeMap.find(E);
  if (it != typeMap.end()) return it->second;
  return addType(T);
}

unsigned ValueEnumerator::ptrTypeIdxForValue(const Value *V) {
  Type *pointee = nullptr;
  if (auto *ty = PTM.get(const_cast<Value*>(V)))
    pointee = ty;
  if (!pointee)
    pointee = pointeeType(V->getType());
  return ptrTypeIdx(V->getType(), pointee);
}

unsigned ValueEnumerator::ptrTypeIdx(Type *PtrTy, Type *pointee) {
  TypeEntry E{PtrTy, pointee};
  auto it = typeMap.find(E);
  if (it != typeMap.end()) return it->second;
  // Ensure pointee is in table first
  addType(pointee);
  return addEntry(E);
}

unsigned ValueEnumerator::globalPtrTypeIdx(const GlobalVariable *GV) {
  return ptrTypeIdx(GV->getType(), GV->getValueType());
}

unsigned ValueEnumerator::globalIdx(const Value *V) const {
  auto it = globalValueMap.find(V);
  assert(it != globalValueMap.end());
  return it->second;
}

unsigned ValueEnumerator::moduleConstIdx(const Constant *C) const {
  auto it = moduleConstMap.find(C);
  assert(it != moduleConstMap.end());
  return globalValues.size() + it->second;
}

bool ValueEnumerator::hasModuleConst(const Constant *C) const {
  return moduleConstMap.count(C);
}

Type *ValueEnumerator::pointeeType(Type *PtrTy) const {
  auto it = inferredPointee.find(PtrTy);
  if (it != inferredPointee.end()) return it->second;
  if (auto *PT = dyn_cast<PointerType>(PtrTy)) {
    unsigned AS = PT->getAddressSpace();
    if (AS == 1 || AS == 3) return Type::getFloatTy(PtrTy->getContext());
  }
  return Type::getFloatTy(PtrTy->getContext());
}

Type *ValueEnumerator::pointeeTypeForValue(const Value *V) const {
  if (auto *ty = PTM.get(const_cast<Value*>(V)))
    return ty;
  if (auto *ty = PointeeTypeMap::inferFromUsage(const_cast<Value*>(V)))
    return ty;
  return pointeeType(V->getType());
}

// ═══════════════════════════════════════════════════════════════════════
// Internal
// ═══════════════════════════════════════════════════════════════════════

unsigned ValueEnumerator::addEntry(TypeEntry E) {
  auto it = typeMap.find(E);
  if (it != typeMap.end()) return it->second;
  unsigned idx = types.size();
  typeMap[E] = idx;
  types.push_back(E);
  return idx;
}

unsigned ValueEnumerator::addType(Type *T) {
  if (isa<PointerType>(T))
    return ptrTypeIdx(T, pointeeType(T));

  // FunctionTypes are handled by addFunctionType for proper per-param
  // pointee tracking. If we get here via a generic path, use the
  // stored indices or fall through to simple entry creation.
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    TypeEntry E{T, nullptr};
    auto it = typeMap.find(E);
    if (it != typeMap.end()) return it->second;
    // Not yet enumerated — add with default pointees (no Function context)
    return addFunctionType(FT, nullptr);
  }

  TypeEntry E{T, nullptr};
  auto it = typeMap.find(E);
  if (it != typeMap.end()) return it->second;

  // Add components first (no forward refs)
  if (auto *VT = dyn_cast<VectorType>(T)) {
    addType(VT->getElementType());
  } else if (auto *AT = dyn_cast<ArrayType>(T)) {
    addType(AT->getElementType());
  } else if (auto *ST = dyn_cast<StructType>(T)) {
    if (!ST->isOpaque())
      for (auto *ET : ST->elements()) addType(ET);
  }

  // Re-check after recursive adds
  it = typeMap.find(E);
  if (it != typeMap.end()) return it->second;

  return addEntry(E);
}

unsigned ValueEnumerator::addFunctionType(FunctionType *FT, const Function *F) {
  TypeEntry E{FT, nullptr};
  auto it = typeMap.find(E);
  if (it != typeMap.end()) return it->second;

  // Build per-param type indices with correct pointee types
  SmallVector<unsigned, 8> paramIndices;

  // Add return type first
  addType(FT->getReturnType());

  // Add each param type — for pointers, use per-param pointee inference
  for (unsigned i = 0; i < FT->getNumParams(); i++) {
    Type *PT = FT->getParamType(i);
    if (!PT->isPointerTy()) {
      paramIndices.push_back(addType(PT));
      continue;
    }
    // Infer pointee for this specific param
    Type *pointee = nullptr;

    // For atomic intrinsics, the device pointer param must match the
    // atomic type (i32 or f32) — NOT the kernel buffer's default pointee.
    // E.g., air.atomic.global.cmpxchg.weak.i32 needs i32*, not float*.
    if (F && F->isDeclaration()) {
      StringRef name = F->getName();
      if (name.starts_with("air.atomic.")) {
        unsigned AS = cast<PointerType>(PT)->getAddressSpace();
        if (AS == 1 || AS == 3) {
          // Determine pointee from intrinsic name suffix
          if (name.ends_with(".i32"))
            pointee = Type::getInt32Ty(F->getContext());
          else if (name.ends_with(".f32"))
            pointee = Type::getFloatTy(F->getContext());
        }
      }
    }

    if (!pointee && F && !F->isDeclaration() && i < F->arg_size())
      pointee = pointeeTypeForValue(F->getArg(i));
    // For declarations, infer from call site arguments
    if (!pointee && F && F->isDeclaration()) {
      for (auto *U : F->users()) {
        if (auto *CI = dyn_cast<CallInst>(U)) {
          if (i < CI->arg_size()) {
            pointee = pointeeTypeForValue(CI->getArgOperand(i));
            if (pointee) break;
          }
        }
      }
    }
    if (!pointee)
      pointee = pointeeType(PT);
    paramIndices.push_back(ptrTypeIdx(PT, pointee));
  }

  // Store per-param indices for TypeTableWriter
  funcTypeParamIndices[FT] = paramIndices;

  // Re-check (recursive adds may have added this type)
  it = typeMap.find(E);
  if (it != typeMap.end()) return it->second;

  return addEntry(E);
}

void ValueEnumerator::addModuleConstant(const Constant *C) {
  if (moduleConstMap.count(C) || globalValueMap.count(C))
    return;
  // ConstantDataArray/Vector have packed data with no sub-constant operands.
  // Extract elements as individual constants so they can be referenced by
  // AGGREGATE records (Metal v1 doesn't support DATA for array globals).
  if (auto *CDA = dyn_cast<ConstantDataSequential>(C)) {
    for (unsigned i = 0; i < CDA->getNumElements(); i++)
      addModuleConstant(CDA->getElementAsConstant(i));
  }
  for (unsigned i = 0; i < C->getNumOperands(); i++)
    if (auto *OC = dyn_cast<Constant>(C->getOperand(i)))
      addModuleConstant(OC);
  addType(C->getType());
  moduleConstMap[C] = moduleConstants.size();
  moduleConstants.push_back(C);
}

void ValueEnumerator::collectMetadataConstants(const MDNode *N) {
  for (unsigned i = 0; i < N->getNumOperands(); i++) {
    if (auto *VAM = dyn_cast_or_null<ValueAsMetadata>(N->getOperand(i)))
      if (auto *C = dyn_cast<Constant>(VAM->getValue()))
        addModuleConstant(C);
    if (auto *Sub = dyn_cast_or_null<MDNode>(N->getOperand(i)))
      collectMetadataConstants(Sub);
  }
}

} // namespace metalir
