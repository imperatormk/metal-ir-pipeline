// Pass 5b: ScalarBufferPacking — pack all scalar params into ONE device buffer.
//
// Triton's Python driver packs all scalars into ONE device buffer
// (ptr addrspace(1)). This pass must match: replace all scalar/AS(2) params
// with ONE AS(1) param + GEP/load preamble.
//
// Scalar params come in two forms:
//   a) raw scalars (float %x, i32 %n)
//   b) ptr addrspace(2) from Triton MLIR
//
// Must run BEFORE AIRSystemValues (which generates !air.kernel metadata).

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace metalir {

static std::pair<unsigned, unsigned> scalarSizeAlign(Type *T) {
  if (T->isFloatTy()) return {4, 4};
  if (T->isDoubleTy()) return {8, 8};
  if (T->isHalfTy() || T->isBFloatTy()) return {2, 2};
  if (auto *IT = dyn_cast<IntegerType>(T)) {
    unsigned bits = IT->getBitWidth();
    if (bits <= 8) return {1, 1};
    if (bits <= 16) return {2, 2};
    if (bits <= 32) return {4, 4};
    return {8, 8};
  }
  return {4, 4};
}

static Type *inferConstPtrLoadType(Function &F, unsigned paramIdx) {
  Argument *Arg = F.getArg(paramIdx);
  for (auto *U : Arg->users())
    if (auto *LI = dyn_cast<LoadInst>(U))
      return LI->getType();
  return nullptr;
}

bool ScalarBufferPackingPass::needsRun(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    for (auto &Arg : F.args()) {
      Type *T = Arg.getType();
      if (!T->isPointerTy() && !T->isVectorTy())
        return true;
      if (T->isPointerTy() && T->getPointerAddressSpace() == 2)
        return true;
    }
  }
  return false;
}

PreservedAnalyses ScalarBufferPackingPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  bool changed = false;

  SmallPtrSet<Function *, 4> calledFns;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (auto *Callee = CI->getCalledFunction())
            calledFns.insert(Callee);

  SmallVector<Function *, 4> funcs;
  for (auto &F : M)
    if (!F.isDeclaration() && !calledFns.count(&F))
      funcs.push_back(&F);

  bool hasMMA = AM.getResult<MMAPresenceAnalysis>(M).hasMMA;

  for (auto *FPtr : funcs) {
    Function &F = *FPtr;

    // Collect system value param indices from pre-baked metadata
    SmallDenseSet<unsigned, 4> sysValParams;
    if (auto *KMD = M.getNamedMetadata("air.kernel")) {
      for (unsigned k = 0; k < KMD->getNumOperands(); k++) {
        auto *Node = KMD->getOperand(k);
        if (Node->getNumOperands() < 1) continue;
        auto *FnMD = dyn_cast_if_present<ValueAsMetadata>(Node->getOperand(0));
        if (!FnMD || FnMD->getValue() != &F) continue;
        for (unsigned n = 1; n < Node->getNumOperands(); n++) {
          auto *Sub = dyn_cast_if_present<MDNode>(Node->getOperand(n));
          if (!Sub) continue;
          for (unsigned s = 0; s < Sub->getNumOperands(); s++) {
            auto *ParamNode = dyn_cast_if_present<MDNode>(Sub->getOperand(s));
            if (!ParamNode || ParamNode->getNumOperands() < 2) continue;
            if (auto *Str = dyn_cast<MDString>(ParamNode->getOperand(1)))
              if (Str->getString().starts_with("air.thread") ||
                  Str->getString().starts_with("air.threadgroup"))
                if (auto *Idx = dyn_cast<ConstantAsMetadata>(
                        ParamNode->getOperand(0)))
                  sysValParams.insert(
                      cast<ConstantInt>(Idx->getValue())->getZExtValue());
          }
        }
      }
    }

    // Collect all scalar-like params
    struct ScalarParam {
      unsigned origIdx;
      Type *scalarType;
      bool isConstPtr;
      bool isDead;
    };
    SmallVector<ScalarParam, 8> scalarParams;

    // Identify descriptor groups to infer types for dead params.
    // Matches the Python driver's _expand_descriptor layout by pattern-
    // matching param groups. Consider passing descriptor types explicitly
    // from Python (via IR metadata) to avoid coupling with driver layout.
    // Between consecutive device pointers (addrspace(1)), scalar params
    // (addrspace(2)) form a descriptor group. For no-metadata descriptors,
    // _expand_descriptor produces: [i64×(2N), i1, i1, i32×N, i64×N]
    // where N = ndim and group_size = 4*N + 2.
    // We build a lookup from param index → descriptor-pattern type.
    SmallDenseMap<unsigned, Type *, 16> descriptorTypeForParam;
    {
      // Scan for groups of AS(2) params after each AS(1) param
      unsigned i = 0;
      while (i < F.arg_size()) {
        Type *T = F.getArg(i)->getType();
        bool isDevicePtr = T->isPointerTy() && T->getPointerAddressSpace() == 1;
        if (isDevicePtr) {
          // Count consecutive AS(2) params after this device ptr
          unsigned groupStart = i + 1;
          unsigned j = groupStart;
          while (j < F.arg_size()) {
            Type *Tj = F.getArg(j)->getType();
            if (!Tj->isPointerTy() || Tj->getPointerAddressSpace() != 2)
              break;
            j++;
          }
          unsigned count = j - groupStart;
          // Check if count matches descriptor pattern: count = 4*N + 2
          if (count >= 2 && (count - 2) % 4 == 0) {
            unsigned ndim = (count - 2) / 4;
            // Validate: live params' types must match pattern
            // Pattern: [i64×(2N), i1, i1, i32×N, i64×N]
            auto getPatternType = [&](unsigned offset) -> Type * {
              if (offset < 2 * ndim)
                return Type::getInt64Ty(M.getContext());
              if (offset < 2 * ndim + 2)
                return Type::getInt1Ty(M.getContext());
              if (offset < 3 * ndim + 2)
                return Type::getInt32Ty(M.getContext());
              return Type::getInt64Ty(M.getContext());
            };
            bool valid = true;
            for (unsigned off = 0; off < count; off++) {
              Type *liveTy = inferConstPtrLoadType(F, groupStart + off);
              if (liveTy) {
                // Live param — check that its size matches the pattern
                auto [liveSize, _a] = scalarSizeAlign(liveTy);
                auto [patSize, _b] = scalarSizeAlign(getPatternType(off));
                if (liveSize != patSize) { valid = false; break; }
              }
            }
            if (valid) {
              for (unsigned off = 0; off < count; off++)
                descriptorTypeForParam[groupStart + off] = getPatternType(off);
            }
          }
        }
        i++;
      }
    }

    for (unsigned i = 0; i < F.arg_size(); i++) {
      if (sysValParams.count(i)) continue;
      Type *T = F.getArg(i)->getType();
      if (T->isPointerTy()) {
        if (T->getPointerAddressSpace() == 2) {
          Type *loadTy = inferConstPtrLoadType(F, i);
          // If inferred type is a pointer, use i64 (pointers are 64-bit)
          if (loadTy && loadTy->isPointerTy())
            loadTy = Type::getInt64Ty(M.getContext());
          if (loadTy) {
            scalarParams.push_back({i, loadTy, true, false});
          } else {
            // Dead param — use descriptor pattern type for correct layout,
            // fallback to i32 if not part of a recognized descriptor group.
            Type *deadTy = Type::getInt32Ty(M.getContext());
            auto it = descriptorTypeForParam.find(i);
            if (it != descriptorTypeForParam.end())
              deadTy = it->second;
            scalarParams.push_back({i, deadTy, true, true});
          }
        }
        continue;
      }
      if (T->isVectorTy()) continue;
      scalarParams.push_back({i, T, false, false});
    }

    if (scalarParams.empty()) continue;

    // Compute byte offsets (matching Python _compute_scalar_layout)
    SmallVector<unsigned, 8> fieldOffsets;
    unsigned currentOffset = 0;
    for (auto &sp : scalarParams) {
      auto [size, align] = scalarSizeAlign(sp.scalarType);
      unsigned padding = (align - (currentOffset % align)) % align;
      currentOffset += padding;
      fieldOffsets.push_back(currentOffset);
      currentOffset += size;
    }

    // Always use float as GEP element type to get float* typed pointer.
    // Metal v1 bitcode rejects load type != pointee type (e.g. load i32 from i8*).
    Type *bufElemTy = Type::getFloatTy(M.getContext());
    unsigned bufElemSize = 4;

    // Build new param list: remove all scalar params, add ONE ptr AS(1) at END.
    // The Python driver passes (ptr0, ptr1, ..., scalar_buf), so the packed
    // scalar buffer must be the LAST device buffer param (before system values).
    SmallDenseSet<unsigned, 8> scalarIdxSet;
    for (auto &sp : scalarParams) scalarIdxSet.insert(sp.origIdx);

    SmallVector<Type *, 8> newParamTypes;
    SmallVector<unsigned, 8> oldToNew(F.arg_size(), UINT_MAX);
    unsigned bufNewIdx = UINT_MAX;
    unsigned ni = 0;
    // First pass: add all non-scalar params (device ptrs, system values, etc.)
    for (unsigned i = 0; i < F.arg_size(); i++) {
      if (scalarIdxSet.count(i)) continue;
      oldToNew[i] = ni;
      newParamTypes.push_back(F.getArg(i)->getType());
      ni++;
    }
    // Then append the packed scalar buffer
    bufNewIdx = ni;
    newParamTypes.push_back(PointerType::get(M.getContext(), 1));
    ni++;

    // Save names before splice
    SmallVector<std::string, 8> oldArgNames;
    for (unsigned i = 0; i < F.arg_size(); i++)
      oldArgNames.push_back(F.getArg(i)->getName().str());

    auto *NewFTy = FunctionType::get(F.getFunctionType()->getReturnType(),
                                      newParamTypes, false);
    auto *NewF = Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(),
                                   "", &M);
    NewF->copyAttributesFrom(&F);
    NewF->splice(NewF->begin(), &F);

    // Wire non-scalar args
    for (unsigned i = 0; i < F.arg_size(); i++) {
      if (scalarIdxSet.count(i)) continue;
      Argument *oldArg = F.getArg(i);
      Argument *newArg = NewF->getArg(oldToNew[i]);
      newArg->setName(oldArgNames[i]);
      oldArg->replaceAllUsesWith(newArg);
    }

    Argument *bufArg = NewF->getArg(bufNewIdx);
    bufArg->setName("_scalar_buf");

    // Build preamble: GEP + load for each live scalar
    auto &EntryBB = NewF->getEntryBlock();
    SmallVector<Instruction *, 16> preamble;

    for (unsigned j = 0; j < scalarParams.size(); j++) {
      auto &sp = scalarParams[j];
      Argument *oldArg = F.getArg(sp.origIdx);
      std::string name = oldArgNames[sp.origIdx];
      if (sp.isConstPtr && name.size() > 4 &&
          name.substr(name.size() - 4) == "_ptr")
        name = name.substr(0, name.size() - 4);

      if (sp.isDead) {
        if (!oldArg->use_empty())
          oldArg->replaceAllUsesWith(UndefValue::get(oldArg->getType()));
        continue;
      }

      unsigned gepIdx = fieldOffsets[j] / bufElemSize;

      // GEP into scalar buffer (manual creation, no IRBuilder)
      auto *gep = GetElementPtrInst::CreateInBounds(
          bufElemTy, bufArg,
          ConstantInt::get(Type::getInt64Ty(M.getContext()), gepIdx),
          name + "_gep");
      preamble.push_back(gep);

      // Load as float (matches GEP element type for typed pointer consistency),
      // then convert to actual scalar type if needed.
      Value *loaded;
      if (sp.scalarType == bufElemTy) {
        auto *ld = new LoadInst(bufElemTy, gep, name, false, Align(4));
        preamble.push_back(ld);
        loaded = ld;
      } else {
        unsigned scalarBits = sp.scalarType->getScalarSizeInBits();
        if (sp.scalarType->isPointerTy()) {
          // Pointer: load as i64 (ptrtoint), then inttoptr
          auto *rawLoad = new LoadInst(bufElemTy, gep, name + "_raw", false, Align(4));
          preamble.push_back(rawLoad);
          auto *asI32 = CastInst::Create(Instruction::BitCast, rawLoad,
                                          Type::getInt32Ty(M.getContext()),
                                          name + "_i32");
          preamble.push_back(asI32);
          auto *asI64 = CastInst::Create(Instruction::ZExt, asI32,
                                          Type::getInt64Ty(M.getContext()),
                                          name + "_i64");
          preamble.push_back(asI64);
          auto *ptr = CastInst::Create(Instruction::IntToPtr, asI64,
                                        sp.scalarType, name);
          preamble.push_back(ptr);
          loaded = ptr;
        } else if (scalarBits == 32) {
          // Same size as float: bitcast (float↔i32)
          auto *rawLoad = new LoadInst(bufElemTy, gep, name + "_raw", false, Align(4));
          preamble.push_back(rawLoad);
          auto *cast = CastInst::Create(Instruction::BitCast, rawLoad,
                                         sp.scalarType, name);
          preamble.push_back(cast);
          loaded = cast;
        } else {
          // Different size (i1, i8, i16, i64, f16, f64): load as i32, trunc/ext
          auto *rawLoad = new LoadInst(bufElemTy, gep, name + "_raw", false, Align(4));
          preamble.push_back(rawLoad);
          // float → i32 first
          auto *asI32 = CastInst::Create(Instruction::BitCast, rawLoad,
                                          Type::getInt32Ty(M.getContext()),
                                          name + "_i32");
          preamble.push_back(asI32);
          // i32 → target type
          if (sp.scalarType->isIntegerTy()) {
            unsigned targetBits = sp.scalarType->getIntegerBitWidth();
            Instruction::CastOps op;
            if (targetBits < 32)
              op = Instruction::Trunc;
            else if (targetBits > 32)
              op = Instruction::ZExt;
            else
              op = Instruction::BitCast;
            auto *conv = CastInst::Create(op, asI32,
                                            sp.scalarType, name);
            preamble.push_back(conv);
            loaded = conv;
          } else if (sp.scalarType->isHalfTy() || sp.scalarType->isBFloatTy()) {
            auto *asI16 = CastInst::Create(Instruction::Trunc, asI32,
                                            Type::getInt16Ty(M.getContext()),
                                            name + "_i16");
            preamble.push_back(asI16);
            auto *cast = CastInst::Create(Instruction::BitCast, asI16,
                                           sp.scalarType, name);
            preamble.push_back(cast);
            loaded = cast;
          } else if (sp.scalarType->isDoubleTy()) {
            // i64 scalars: would need two float loads — for now just load directly
            auto *ld = new LoadInst(sp.scalarType, gep, name, false, Align(4));
            // Remove the rawLoad we already added
            preamble.pop_back(); // asI32
            preamble.pop_back(); // rawLoad
            preamble.push_back(ld);
            loaded = ld;
          } else {
            // Fallback: load actual type directly
            auto *ld = new LoadInst(sp.scalarType, gep, name, false, Align(4));
            preamble.pop_back(); // asI32
            preamble.pop_back(); // rawLoad
            preamble.push_back(ld);
            loaded = ld;
          }
        }
      }
      auto *load = loaded;

      if (sp.isConstPtr) {
        // Replace loads through old AS(2) ptr
        SmallVector<LoadInst *, 4> oldLoads;
        for (auto *U : oldArg->users())
          if (auto *LI = dyn_cast<LoadInst>(U))
            oldLoads.push_back(LI);
        for (auto *LI : oldLoads) {
          LI->replaceAllUsesWith(load);
          LI->eraseFromParent();
        }
        if (!oldArg->use_empty())
          oldArg->replaceAllUsesWith(UndefValue::get(oldArg->getType()));
      } else {
        oldArg->replaceAllUsesWith(load);
      }
    }

    // Insert preamble at entry (reverse so first GEP comes first)
    for (auto *I : llvm::reverse(preamble))
      I->insertBefore(EntryBB, EntryBB.begin());

    // Update metadata references
    for (auto &NMD : M.named_metadata())
      for (unsigned i = 0; i < NMD.getNumOperands(); i++) {
        auto *Node = NMD.getOperand(i);
        for (unsigned j = 0; j < Node->getNumOperands(); j++)
          if (auto *VMD = dyn_cast_if_present<ValueAsMetadata>(Node->getOperand(j)))
            if (VMD->getValue() == &F)
              Node->replaceOperandWith(j, ValueAsMetadata::get(NewF));
      }

    std::string fname = F.getName().str();
    F.eraseFromParent();
    NewF->setName(fname);
    changed = true;
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace metalir
