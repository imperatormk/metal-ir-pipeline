// BitcodeEmitter — Top-level orchestrator for Metal v1 bitcode emission.
// Delegates to: ValueEnumerator, TypeTableWriter, ConstantsWriter,
//               MetadataWriter, FunctionWriter.

#include "metal-ir/BitcodeEmitter.h"
#include "metal-ir/BitcodeEncoding.h"
#include "metal-ir/MetalConstraints.h"
#include "metal-ir/ValueEnumerator.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace metalir {

// Lower all ConstantExpr operands in instructions to real instructions.
// Metal's GPU JIT doesn't handle constant expression records in bitcode;
// MetalASM similarly lowers them before serialization.
//
// For byte-stride GEPs on threadgroup float globals (e.g.,
//   gep i8, @tg_global, i64 byte_offset  where @tg_global is [N x float]),
// converts to float-element GEPs (gep float, @base, i64 float_index)
// because Metal v1 typed-pointer bitcode requires GEP source type to match
// the pointer's pointee type.
static void lowerConstantExprs(Module &M) {
  auto &Ctx = M.getContext();
  Type *floatTy = Type::getFloatTy(Ctx);
  Type *i64Ty = Type::getInt64Ty(Ctx);
  unsigned floatSize = M.getDataLayout().getTypeAllocSize(floatTy);

  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<std::pair<Instruction *, unsigned>, 32> worklist;
    bool changed = true;
    while (changed) {
      changed = false;
      worklist.clear();
      for (auto &BB : F)
        for (auto &I : BB)
          for (unsigned i = 0; i < I.getNumOperands(); i++)
            if (isa<ConstantExpr>(I.getOperand(i)))
              worklist.push_back({&I, i});
      for (auto &[I, OpIdx] : worklist) {
        auto *CE = cast<ConstantExpr>(I->getOperand(OpIdx));
        Instruction *NewI = CE->getAsInstruction();

        // Convert byte-stride GEPs on TG float globals to float-element GEPs.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(NewI)) {
          if (GEP->getSourceElementType()->isIntegerTy(8) &&
              GEP->getPointerAddressSpace() == metalir::AS::Threadgroup &&
              GEP->getNumIndices() == 1) {
            // Check if base is a TG global with float array element type
            Value *base = GEP->getPointerOperand();
            GlobalVariable *GV = dyn_cast<GlobalVariable>(base);
            if (GV) {
              Type *elemTy = nullptr;
              if (auto *AT = dyn_cast<ArrayType>(GV->getValueType()))
                elemTy = AT->getElementType();
              if (elemTy && elemTy->isFloatTy()) {
                Value *byteIdx = GEP->idx_begin()->get();
                if (auto *CI = dyn_cast<ConstantInt>(byteIdx)) {
                  uint64_t byteOff = CI->getZExtValue();
                  if (byteOff % floatSize == 0) {
                    // Find or create a base float* from the GV.
                    // Look for an existing gep [N x float], @GV, 0, 0 in the
                    // function's entry block.
                    Value *floatBase = nullptr;
                    for (auto *U : GV->users()) {
                      auto *BaseGEP = dyn_cast<GetElementPtrInst>(U);
                      if (!BaseGEP || !BaseGEP->getParent())
                        continue;
                      if (BaseGEP->getFunction() == &F &&
                          BaseGEP->getSourceElementType() == GV->getValueType() &&
                          BaseGEP->getNumIndices() == 2) {
                        floatBase = BaseGEP;
                        break;
                      }
                    }
                    if (!floatBase) {
                      // Create base GEP: gep [N x float], @GV, 0, 0
                      auto *baseGEP = GetElementPtrInst::CreateInBounds(
                          GV->getValueType(), GV,
                          {ConstantInt::get(i64Ty, 0),
                           ConstantInt::get(i64Ty, 0)});
                      baseGEP->insertBefore(
                          &*F.getEntryBlock().getFirstInsertionPt());
                      floatBase = baseGEP;
                    }
                    // Create: gep float, %base, i64 (byteOff/4)
                    auto *floatGEP = GetElementPtrInst::CreateInBounds(
                        elemTy, floatBase,
                        {ConstantInt::get(i64Ty, byteOff / floatSize)});
                    floatGEP->insertBefore(I);
                    I->setOperand(OpIdx, floatGEP);
                    NewI->deleteValue(); // discard the byte GEP
                    changed = true;
                    continue;
                  }
                }
              }
            }
          }
        }

        NewI->insertBefore(I);
        I->setOperand(OpIdx, NewI);
        changed = true;
      }
    }

    // Identity ptr-to-ptr bitcasts (same opaque type, different typed pointer
    // semantics) are kept — the FunctionWriter handles them by emitting a
    // bitcast to the correct typed pointer type inferred from PTM/usage.
  }
}

// Fix GEP source type / pointer pointee mismatches for Metal typed bitcode.
//
// Metal v1 typed-pointer bitcode requires the GEP source element type to match
// the pointer's pointee type. When a GEP uses a different element type than the
// pointer (e.g., `gep half, float*3 %ptr`), the GPU JIT rejects it.
//
// For TG (AS3) pointers where the buffer is float-typed (from MMA merge) but
// accessed with half/i8 GEPs (from pipelined loads), we insert identity
// bitcasts before the GEP. The bitcast creates a new pointer value that the
// PTM can type as half* instead of float*, making the GEP consistent:
//   %bc = bitcast float*3 %ptr to float*3  (identity in opaque-ptr IR)
//   %p = gep half, float*3 %bc, i32 %idx
// Then PTM sets %bc → half, so typed bitcode sees: gep half, half*3 %bc, idx
//
// For device (AS1) pointers with i8 GEPs (from async copy byte offsets),
// convert to float-stride GEPs since all device pointers are float*.
static void fixGEPTypeMismatches(Module &M, PointeeTypeMap &PTM) {
  bool hasMMA = false;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      hasMMA = true;
  if (!hasMMA) return;

  auto &Ctx = M.getContext();
  Type *floatTy = Type::getFloatTy(Ctx);

  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<GetElementPtrInst *, 8> toFix;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          Type *srcTy = GEP->getSourceElementType();
          if (srcTy == floatTy) continue;
          if (GEP->getNumIndices() != 1) continue;
          if (!srcTy->isIntegerTy() && !srcTy->isHalfTy() &&
              !srcTy->isBFloatTy())
            continue;
          unsigned AS = GEP->getPointerAddressSpace();
          if (AS != metalir::AS::Device && AS != metalir::AS::Threadgroup)
            continue;
          toFix.push_back(GEP);
        }

    for (auto *GEP : toFix) {
      Type *srcTy = GEP->getSourceElementType();
      unsigned srcSize = srcTy->getPrimitiveSizeInBits();
      Value *ptr = GEP->getPointerOperand();

      // Same-size types (e.g., i32 vs float, both 4 bytes):
      // Just change the GEP source element type to float. The stride is
      // identical so the pointer arithmetic doesn't change.
      if (srcSize == 32) {
        GEP->setSourceElementType(floatTy);
        GEP->setResultElementType(floatTy);
        continue;
      }

      // Different-size types (half=16bit, i8=8bit vs float=32bit):
      // Insert identity bitcast before GEP to create a new pointer value
      // with the correct PTM entry. The bitcast is a no-op in opaque-ptr IR
      // but gives the serializer a different typed pointer for the GEP source.
      auto *BC = CastInst::Create(Instruction::BitCast, ptr,
                                   ptr->getType(), "", GEP);
      GEP->setOperand(0, BC);
      PTM.set(BC, srcTy);
    }
  }
}

// Fix air.arg_type_name / air.arg_type_size in kernel metadata to match
// actual parameter pointee types from PTM. The transform pipeline may set
// all buffer type names to "float" even when the actual type is bfloat/char.
// Metal GPU JIT validates these metadata entries against the bitcode types.
static void fixKernelArgMetadata(Module &M, const PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto *airKernel = M.getNamedMetadata("air.kernel");
  if (!airKernel) return;

  for (unsigned k = 0; k < airKernel->getNumOperands(); k++) {
    auto *kernelMD = airKernel->getOperand(k);
    if (kernelMD->getNumOperands() < 3) continue;
    // kernelMD: {fn, attrs, argDescs}
    auto *argDescs = dyn_cast_or_null<MDNode>(kernelMD->getOperand(2));
    if (!argDescs) continue;

    auto *fnVAM = dyn_cast_or_null<ValueAsMetadata>(kernelMD->getOperand(0));
    if (!fnVAM) continue;
    auto *fn = dyn_cast<Function>(fnVAM->getValue());
    if (!fn) continue;

    for (unsigned a = 0; a < argDescs->getNumOperands(); a++) {
      auto *argMD = dyn_cast_or_null<MDNode>(argDescs->getOperand(a));
      if (!argMD || argMD->getNumOperands() < 2) continue;

      // Check if this is a buffer arg (has "air.buffer" string)
      bool isBuffer = false;
      for (unsigned i = 1; i < argMD->getNumOperands(); i++)
        if (auto *S = dyn_cast_or_null<MDString>(argMD->getOperand(i)))
          if (S->getString() == "air.buffer") { isBuffer = true; break; }
      if (!isBuffer) continue;

      // Get the arg index from the first operand
      auto *idxVAM = dyn_cast_or_null<ValueAsMetadata>(argMD->getOperand(0));
      if (!idxVAM) continue;
      auto *idxCI = dyn_cast<ConstantInt>(idxVAM->getValue());
      if (!idxCI) continue;
      unsigned argIdx = idxCI->getZExtValue();
      if (argIdx >= fn->arg_size()) continue;

      // Infer pointee type from PTM, following through bitcasts
      Argument *arg = fn->getArg(argIdx);
      if (!arg->getType()->isPointerTy()) continue;
      Type *pointee = nullptr;
      if (auto *ty = PTM.get(arg)) pointee = ty;
      if (!pointee) pointee = PointeeTypeMap::inferFromUsage(arg);
      // Follow through bitcasts if inference failed on the arg directly
      if (!pointee || pointee->isFloatTy()) {
        for (auto *U : arg->users()) {
          if (auto *BC = dyn_cast<BitCastInst>(U)) {
            if (auto *ty = PTM.get(BC)) {
              if (!ty->isFloatTy()) { pointee = ty; break; }
            }
            Type *bcTy = PointeeTypeMap::inferFromUsage(BC);
            if (bcTy && !bcTy->isFloatTy()) { pointee = bcTy; break; }
          }
        }
      }
      if (!pointee) continue;

      // Determine correct type name, size, and alignment
      StringRef typeName;
      unsigned typeSize = 0, typeAlign = 0;
      if (pointee->isBFloatTy()) {
        typeName = "bfloat"; typeSize = 2; typeAlign = 2;
      } else if (pointee->isFloatTy()) {
        typeName = "float"; typeSize = 4; typeAlign = 4;
      } else if (pointee->isHalfTy()) {
        typeName = "half"; typeSize = 2; typeAlign = 2;
      } else if (pointee->isIntegerTy(8)) {
        typeName = "char"; typeSize = 1; typeAlign = 1;
      } else if (pointee->isIntegerTy(16)) {
        typeName = "short"; typeSize = 2; typeAlign = 2;
      } else if (pointee->isIntegerTy(32)) {
        typeName = "int"; typeSize = 4; typeAlign = 4;
      } else {
        continue; // Unknown type, don't change
      }

      // Rebuild the metadata node with corrected values
      SmallVector<Metadata *, 16> newOps;
      for (unsigned i = 0; i < argMD->getNumOperands(); i++) {
        Metadata *op = argMD->getOperand(i);
        if (i + 1 < argMD->getNumOperands()) {
          if (auto *prevS = dyn_cast_or_null<MDString>(argMD->getOperand(i))) {
            if (prevS->getString() == "air.arg_type_name" &&
                i + 1 < argMD->getNumOperands()) {
              newOps.push_back(op);
              newOps.push_back(MDString::get(Ctx, typeName));
              i++; // skip original type name
              continue;
            }
            if (prevS->getString() == "air.arg_type_size" &&
                i + 1 < argMD->getNumOperands()) {
              newOps.push_back(op);
              newOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), typeSize)));
              i++; // skip original size
              continue;
            }
            if (prevS->getString() == "air.arg_type_align_size" &&
                i + 1 < argMD->getNumOperands()) {
              newOps.push_back(op);
              newOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), typeAlign)));
              i++; // skip original align
              continue;
            }
          }
        }
        newOps.push_back(op);
      }
      auto *newArgMD = MDNode::get(Ctx, newOps);
      argDescs->replaceOperandWith(a, newArgMD);
    }
  }
}

// Forward declarations (defined in separate .cpp files)
void emitTypeBlock(BitstreamWriter &W, ValueEnumerator &E);
void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                         ArrayRef<const Constant *> constants, unsigned codeSize);
void emitMetadataKindBlock(BitstreamWriter &W);
void emitMetadataBlock(BitstreamWriter &W, Module &M, ValueEnumerator &E);
void emitOperandBundleTagsBlock(BitstreamWriter &W);
void emitSinglethreadBlock(BitstreamWriter &W);
void emitFunctionBlock(BitstreamWriter &W, const Function &F, ValueEnumerator &E);

// Remove truly redundant ptr-to-ptr bitcasts where the PTM has the SAME
// pointee type on both sides. Bitcasts that serve as typed pointer
// transitions (where PTM records different types) must be kept.
// Only removes bitcasts where BOTH sides have the same PTM entry.
static void removeRedundantBitcasts(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<BitCastInst *, 16> toRemove;
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC || BC->getSrcTy() != BC->getDestTy())
          continue;
        // Only remove if BOTH sides have the same PTM-recorded type.
        // If either side has no PTM entry, keep the bitcast (it may serve
        // as a type transition that the serializer needs).
        Type *srcPT = PTM.get(BC->getOperand(0));
        Type *dstPT = PTM.get(BC);
        if (!srcPT || !dstPT) continue; // Unknown — keep
        if (srcPT != dstPT) continue;   // Different — keep
        toRemove.push_back(BC);
      }
    }
    for (auto *BC : toRemove) {
      PTM.remove(BC);
      BC->replaceAllUsesWith(BC->getOperand(0));
      BC->eraseFromParent();
    }
  }
}

std::vector<uint8_t> emitMetalBitcode(Module &M, const PointeeTypeMap &PTM) {
  SmallVector<char, 0> Buf;
  BitstreamWriter W(Buf);

  // BC magic
  W.Emit('B', 8); W.Emit('C', 8); W.Emit(0xC0, 8); W.Emit(0xDE, 8);

  // IDENTIFICATION
  W.EnterSubblock(bitc::IDENTIFICATION_BLOCK_ID, 5);
  emitString(W, bitc::IDENTIFICATION_CODE_STRING, "MetalIR");
  { SmallVector<uint64_t, 1> V = {0}; W.EmitRecord(bitc::IDENTIFICATION_CODE_EPOCH, V); }
  W.ExitBlock();

  // Pre-serialization IR fixups.
  {
    auto &mutablePTM = const_cast<PointeeTypeMap &>(PTM);
    removeRedundantBitcasts(M, mutablePTM);
  }

  // Convert byte-stride device GEPs to float-stride before lowering.
  {
    auto &mutablePTM2 = const_cast<PointeeTypeMap &>(PTM);
    fixGEPTypeMismatches(M, mutablePTM2);
  }

  // Lower ConstantExpr operands to real instructions before enumeration.
  lowerConstantExprs(M);

  // Fix kernel argument metadata to match actual pointee types.
  fixKernelArgMetadata(M, PTM);

  // Debug: dump post-lowering IR
  if (getenv("METALIR_DUMP_LOWERED")) {
    std::error_code EC;
    raw_fd_ostream dump("/tmp/metalir_after_lowering.ll", EC);
    if (!EC) M.print(dump, nullptr);
  }

  // Enumerate
  ValueEnumerator E(M, PTM);

  // MODULE_BLOCK (CodeSize=4)
  W.EnterSubblock(bitc::MODULE_BLOCK_ID, 4);

  { SmallVector<uint64_t, 1> V = {1}; W.EmitRecord(bitc::MODULE_CODE_VERSION, V); }

  // Emit PARAMATTR blocks BEFORE TYPE_BLOCK (Metal requires this order).
  // MMA load needs nocapture+readonly on its pointer param.
  bool hasMMALoad = false;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_load"))
      hasMMALoad = true;

  if (hasMMALoad) {
    // PARAMATTR_GROUP_BLOCK: group 1 = {param 1: nocapture, readonly}
    W.EnterSubblock(bitc::PARAMATTR_GROUP_BLOCK_ID, 4);
    {
      // Group entry: [grp_id, param_idx, attr_kind, ...]
      // nocapture = enum 21, readonly = enum 36 (LLVM attr enum IDs)
      SmallVector<uint64_t, 8> Grp;
      Grp.push_back(1);  // group ID
      Grp.push_back(1);  // param index (1 = first ptr param of MMA load)
      Grp.push_back(0); Grp.push_back(11); // enum: noCapture (ATTR_KIND=11)
      Grp.push_back(0); Grp.push_back(21); // enum: readOnly (ATTR_KIND=21)
      W.EmitRecord(3, Grp); // PARAMATTR_GRP_CODE_ENTRY = 3
    }
    W.ExitBlock();

    // PARAMATTR_BLOCK: list 1 = [group 1]
    W.EnterSubblock(bitc::PARAMATTR_BLOCK_ID, 4);
    {
      SmallVector<uint64_t, 2> List;
      List.push_back(1); // group ID
      W.EmitRecord(2, List); // PARAMATTR_CODE_ENTRY = 2
    }
    W.ExitBlock();
  }

  emitTypeBlock(W, E);

  // Emit target triple — Metal GPU JIT expects it for proper codegen.
  // Use module value if set, otherwise default Metal AIR triple.
  {
    std::string T = M.getTargetTriple().str();
    if (T.empty()) T = "air64_v28-apple-macosx26.0.0";
    emitString(W, bitc::MODULE_CODE_TRIPLE, T);
  }
  // Emit data layout — Metal GPU JIT uses this for type size/alignment.
  {
    auto DLStr = M.getDataLayoutStr();
    if (!DLStr.empty()) {
      emitString(W, bitc::MODULE_CODE_DATALAYOUT, DLStr);
    } else {
      emitString(W, bitc::MODULE_CODE_DATALAYOUT,
          "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
          "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32"
          "-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
          "-v192:256:256-v256:256:256-v512:512:512"
          "-v1024:1024:1024-n8:16:32");
    }
  }

  if (!M.getSourceFileName().empty())
    emitString(W, bitc::MODULE_CODE_SOURCE_FILENAME, M.getSourceFileName());

  // GLOBALVAR and FUNCTION records — emit in globalValues order
  // (globals first, then functions, matching value ID assignment)
  for (auto *V : E.globalValues) {
    if (auto *G = dyn_cast<GlobalVariable>(V)) {
      SmallVector<uint64_t, 14> Ops;
      Ops.push_back(E.globalPtrTypeIdx(G)); // ptr-to-valueType
      Ops.push_back(G->isConstant() ? 1 : 0);
      Ops.push_back(G->hasInitializer() ? E.moduleConstIdx(G->getInitializer()) + 1 : 0);
      Ops.push_back(encodeLinkage(G->getLinkage()));
      Ops.push_back(G->getAlign() ? Log2_32(G->getAlign()->value()) + 1 : 0);
      for (int i = 0; i < 3; i++) Ops.push_back(0);
      Ops.push_back(G->hasGlobalUnnamedAddr() ? 1 : 0);
      Ops.push_back(G->isExternallyInitialized() ? 1 : 0);
      Ops.push_back(0); Ops.push_back(0);
      Ops.push_back(G->getAddressSpace());
      Ops.push_back(0);
      W.EmitRecord(bitc::MODULE_CODE_GLOBALVAR, Ops);
    } else if (auto *Fn = dyn_cast<Function>(V)) {
      SmallVector<uint64_t, 17> Ops;
      Ops.push_back(E.typeIdx(Fn->getFunctionType()));
      Ops.push_back(Fn->getCallingConv());
      Ops.push_back(Fn->isDeclaration() ? 1 : 0);
      Ops.push_back(encodeLinkage(Fn->getLinkage()));
      // paramattr: 1 for MMA load (nocapture+readonly), 0 otherwise
      bool isMMALoadFn = Fn->getName().starts_with("air.simdgroup_matrix_8x8_load");
      Ops.push_back(isMMALoadFn && hasMMALoad ? 1 : 0);
      Ops.push_back(0); // align
      for (int i = 0; i < 10; i++) Ops.push_back(0);
      Ops.push_back(Fn->getAddressSpace());
      W.EmitRecord(bitc::MODULE_CODE_FUNCTION, Ops);
    }
  }

  emitConstantsBlock(W, E, E.moduleConstants, 5);
  emitMetadataKindBlock(W);
  emitMetadataBlock(W, M, E);
  emitOperandBundleTagsBlock(W);
  emitSinglethreadBlock(W);

  for (auto *V : E.globalValues)
    if (auto *F = dyn_cast<Function>(V))
      if (!F->isDeclaration())
        emitFunctionBlock(W, *F, E);

  // VALUE_SYMTAB
  W.EnterSubblock(bitc::VALUE_SYMTAB_BLOCK_ID, 4);
  for (unsigned i = 0; i < E.globalValues.size(); i++) {
    if (!E.globalValues[i]->hasName()) continue;
    SmallVector<uint64_t, 32> NV;
    NV.push_back(i);
    for (char C : E.globalValues[i]->getName())
      NV.push_back((uint64_t)(unsigned char)C);
    W.EmitRecord(bitc::VST_CODE_ENTRY, NV);
  }
  W.ExitBlock();

  W.ExitBlock(); // MODULE_BLOCK

  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

} // namespace metalir
