// ═══════════════════════════════════════════════════════════════════════
// Pass 19: AIRSystemValues
//
// Metal kernels receive system values (thread ID, group ID, etc.) as
// function parameters, not via intrinsic calls. This pass:
//
// 1. Finds calls to air.thread_position_in_grid() etc.
// 2. Replaces them with function parameters (<3 x i32> vectors)
// 3. Adds extractelement instructions to get individual components
// 4. Emits !air.kernel metadata (required by Metal runtime)
//
// Equivalent to MetalASM's transformAirSystemValues (lines 3209-3849),
// minus Pass 5b (scalar buffer packing) which is separate.
// ═══════════════════════════════════════════════════════════════════════

#include "metal-ir/Pipeline.h"
#include "metal-ir/AIRIntrinsics.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace metalir {

// System value intrinsic names — from central AIRIntrinsics.h
static constexpr const char *kPidName = air::kCallPid;
static constexpr const char *kTidName = air::kCallTid;
static constexpr const char *kTidTGName = air::kCallTidTG;
static constexpr const char *kSimdlaneName = air::kCallSimdlane;
static constexpr const char *kNumProgName = air::kCallNumProg;

// Info about a system value parameter to add
struct SysValParam {
  const char *intrinsicName;
  const char *paramName;
  bool isVector; // true = <3 x i32>, false = i32
  const char *dims[3]; // component names: "pid_x", "pid_y", "pid_z"
};

static const SysValParam kSysVals[] = {
    {kPidName, "pid", true, {"pid_x", "pid_y", "pid_z"}},
    {kTidName, "tid", true, {"tid_x", "tid_y", "tid_z"}},
    {kTidTGName, "tidtg", true, {"tidtg_x", "tidtg_y", "tidtg_z"}},
    {kSimdlaneName, "simdlane", false, {"simdlane", "", ""}},
    {kNumProgName, "numprog", true, {"numprog_x", "numprog_y", "numprog_z"}},
};

/// Find a declaration whose name starts with the given prefix.
static Function *findDeclByPrefix(Module &M, StringRef prefix) {
  for (auto &F : M)
    if (F.isDeclaration() && F.getName().starts_with(prefix))
      return &F;
  return nullptr;
}

bool AIRSystemValuesPass::needsRun(Module &M) {
  for (auto &sv : kSysVals)
    if (findDeclByPrefix(M, sv.intrinsicName))
      return true;
  return false;
}

PreservedAnalyses AIRSystemValuesPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  auto &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *Vec3 = FixedVectorType::get(I32, 3);

  bool changed = false;

  // (Declarations found by prefix matching during per-function scan below)

  // Collect non-declaration functions (we'll modify the module)
  SmallVector<Function *, 4> funcs;
  for (auto &F : M)
    if (!F.isDeclaration())
      funcs.push_back(&F);

  for (auto *FPtr : funcs) {
    Function &F = *FPtr;

    // Scan: which system values does this function use?
    struct UsedSysVal {
      const SysValParam *info;
      SmallVector<CallInst *, 4> calls;
    };
    SmallVector<UsedSysVal, 5> usedSysVals;

    for (auto &sv : kSysVals) {
      auto *Decl = findDeclByPrefix(M, sv.intrinsicName);
      if (!Decl)
        continue;

      SmallVector<CallInst *, 4> calls;
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            if (CI->getCalledFunction() == Decl)
              calls.push_back(CI);
          }
        }
      }
      if (!calls.empty())
        usedSysVals.push_back({&sv, std::move(calls)});
    }

    if (usedSysVals.empty())
      continue;

    // Add new parameters to function
    // We need to create a new function with extra params and replace the old one
    auto *OldFTy = F.getFunctionType();
    SmallVector<Type *, 8> newParamTypes(OldFTy->params());
    for (auto &usv : usedSysVals)
      newParamTypes.push_back(usv.info->isVector ? Vec3 : I32);

    auto *NewFTy = FunctionType::get(OldFTy->getReturnType(), newParamTypes,
                                      OldFTy->isVarArg());

    // Create new function and move body from old
    auto *NewF = Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(),
                                   "", &M);
    NewF->copyAttributesFrom(&F);
    NewF->splice(NewF->begin(), &F);

    // Map old args → new args
    auto newArgIt = NewF->arg_begin();
    for (auto &oldArg : F.args()) {
      newArgIt->setName(oldArg.getName());
      oldArg.replaceAllUsesWith(&*newArgIt);
      ++newArgIt;
    }

    // Take the old function's name and remove the old function
    std::string fname = F.getName().str();
    F.eraseFromParent();
    NewF->setName(fname);

    // Now process system values — insert extractelement at entry
    auto &EntryBB = NewF->getEntryBlock();

    for (auto &usv : usedSysVals) {
      Argument *sysArg = &*newArgIt++;
      sysArg->setName(usv.info->paramName);

      if (usv.info->isVector) {
        // Extract x, y, z components — reset insertion point each time
        Value *components[3];
        for (int i = 0; i < 3; i++) {
          IRBuilder<> B(&*EntryBB.begin());
          components[i] = B.CreateExtractElement(
              sysArg, ConstantInt::get(I32, i), usv.info->dims[i]);
        }

        // Replace call + extractvalue chains
        for (auto *CI : usv.calls) {
          // Find extractvalue users of this call
          SmallVector<ExtractValueInst *, 4> extracts;
          for (auto *U : CI->users())
            if (auto *EV = dyn_cast<ExtractValueInst>(U))
              extracts.push_back(EV);

          for (auto *EV : extracts) {
            unsigned idx = EV->getIndices()[0];
            EV->replaceAllUsesWith(components[std::min(idx, 2u)]);
            EV->eraseFromParent();
          }
          // If call has remaining uses (shouldn't happen), replace with undef
          if (!CI->use_empty())
            CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
          CI->eraseFromParent();
        }
      } else {
        // Scalar (simdlane) — direct replacement
        for (auto *CI : usv.calls) {
          CI->replaceAllUsesWith(sysArg);
          CI->eraseFromParent();
        }
      }
    }

    changed = true;
  }

  // Remove intrinsic declarations (prefix match)
  for (auto &sv : kSysVals) {
    if (auto *F = findDeclByPrefix(M, sv.intrinsicName)) {
      if (F->use_empty())
        F->eraseFromParent();
    }
  }

  // Emit !air.kernel metadata
  // Format: !air.kernel = !{!N}
  //         !N = !{!"kernel_name", !params...}
  // Each param: !{i32 index, !"air.buffer", !"air.location_index", i32 loc,
  //              i32 1, !"air.read_write"} for device buffers
  //             !{i32 index, !"air.position_in_grid"} for system values
  // Always generate !air.kernel metadata (even if no system value calls
  // were converted — kernels still need metadata for PSO creation).
  // Skip if metadata already exists (pre-baked in test kernels).
  if (!M.getNamedMetadata(air::kNMDKernel)) {
    auto *KernelMD = M.getOrInsertNamedMetadata(air::kNMDKernel);

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;

      // air.kernel format: !{ptr @fn, !{}, !{param_nodes...}}
      // First: function ref. Second: empty node. Third: param list node.
      SmallVector<Metadata *, 16> paramNodes;

      unsigned argIdx = 0;
      auto *FTy = F.getFunctionType();

      // Buffer params (device AS=1, constant AS=2)
      for (unsigned i = 0; i < FTy->getNumParams(); i++) {
        auto *ParamTy = FTy->getParamType(i);
        if (!ParamTy->isPointerTy())
          continue;

        unsigned AS = cast<PointerType>(ParamTy)->getAddressSpace();
        if (AS != 1 && AS != 2)
          continue;

        StringRef argName = F.getArg(i)->getName();
        if (argName.empty()) {
          static char nameBuf[16];
          snprintf(nameBuf, sizeof(nameBuf), "%u", i);
          argName = nameBuf;
        }
        const char *accessMode = (AS == 2) ? air::kMDRead : air::kMDReadWrite;
        paramNodes.push_back(MDNode::get(
            Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32, i)),
                  MDString::get(Ctx, air::kMDBuffer),
                  MDString::get(Ctx, air::kMDLocationIndex),
                  ConstantAsMetadata::get(ConstantInt::get(I32, argIdx)),
                  ConstantAsMetadata::get(ConstantInt::get(I32, 1)),
                  MDString::get(Ctx, accessMode),
                  MDString::get(Ctx, air::kMDAddressSpace),
                  ConstantAsMetadata::get(ConstantInt::get(I32, AS)),
                  MDString::get(Ctx, air::kMDArgTypeSize),
                  ConstantAsMetadata::get(ConstantInt::get(I32, 4)),
                  MDString::get(Ctx, air::kMDArgTypeAlignSize),
                  ConstantAsMetadata::get(ConstantInt::get(I32, 4)),
                  MDString::get(Ctx, air::kMDArgTypeName),
                  MDString::get(Ctx, "float"),
                  MDString::get(Ctx, air::kMDArgName),
                  MDString::get(Ctx, argName)}));
        argIdx++;
      }

      // System value params (at the end) — only vector (<3 x i32>) or
      // scalar i32 types, never pointers (which may have colliding names
      // from ScalarBufferPacking, e.g., "tid_ptr")
      for (auto &Arg : F.args()) {
        Type *ArgTy = Arg.getType();
        if (ArgTy->isPointerTy())
          continue;
        StringRef name = Arg.getName();
        StringRef airAttr;
        // Match by prefix — LLVM may append numeric suffixes to avoid
        // name collisions (e.g., "tid" → "tid1")
        if (name.starts_with("pid"))
          airAttr = air::kMDThreadgroupPositionInGrid;
        else if (name.starts_with("tidtg"))
          airAttr = air::kMDThreadPositionInTG;
        else if (name.starts_with("tid"))
          airAttr = air::kMDThreadPositionInGrid;
        else if (name.starts_with("simdlane"))
          airAttr = air::kMDThreadIndexInSimdgroup;
        else if (name.starts_with("numprog"))
          airAttr = air::kMDThreadgroupsPerGrid;
        else
          continue;

        // System value — no arg_type_size/align (matching MetalASM)
        StringRef typeName = (name == "simdlane") ? "uint" : "uint3";
        paramNodes.push_back(MDNode::get(
            Ctx, {ConstantAsMetadata::get(
                      ConstantInt::get(I32, Arg.getArgNo())),
                  MDString::get(Ctx, airAttr),
                  MDString::get(Ctx, air::kMDArgTypeName),
                  MDString::get(Ctx, typeName),
                  MDString::get(Ctx, air::kMDArgName),
                  MDString::get(Ctx, name)}));
      }

      // Assemble: !{ptr @fn, !{}, !{params...}}
      auto *emptyNode = MDNode::get(Ctx, {});
      auto *paramsNode = MDNode::get(Ctx, paramNodes);
      KernelMD->addOperand(MDNode::get(Ctx, {
          ValueAsMetadata::get(&F), emptyNode, paramsNode}));
    }

    changed = true;
  }

  // Version metadata (always emit if missing)
  auto *VerMD = M.getOrInsertNamedMetadata(air::kNMDVersion);
  if (VerMD->getNumOperands() == 0) {
    VerMD->addOperand(MDNode::get(
        Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32, 2)),
              ConstantAsMetadata::get(ConstantInt::get(I32, 8)),
              ConstantAsMetadata::get(ConstantInt::get(I32, 0))}));
    changed = true;
  }

  // Language version metadata (required for PSO creation)
  if (!M.getNamedMetadata(air::kNMDLanguageVersion)) {
    auto *LangMD = M.getOrInsertNamedMetadata(air::kNMDLanguageVersion);
    LangMD->addOperand(MDNode::get(
        Ctx, {MDString::get(Ctx, "Metal"),
              ConstantAsMetadata::get(ConstantInt::get(I32, 3)),
              ConstantAsMetadata::get(ConstantInt::get(I32, 2)),
              ConstantAsMetadata::get(ConstantInt::get(I32, 0))}));
    changed = true;
  }

  // Module flags (air.max_* limits) — required by Metal compiler
  if (!M.getModuleFlag("air.max_device_buffers")) {
    M.addModuleFlag(Module::Max, "air.max_device_buffers",
                    ConstantInt::get(I32, 31));
    M.addModuleFlag(Module::Max, "air.max_constant_buffers",
                    ConstantInt::get(I32, 31));
    M.addModuleFlag(Module::Max, "air.max_threadgroup_buffers",
                    ConstantInt::get(I32, 31));
    M.addModuleFlag(Module::Max, "air.max_textures",
                    ConstantInt::get(I32, 128));
    M.addModuleFlag(Module::Max, "air.max_read_write_textures",
                    ConstantInt::get(I32, 8));
    M.addModuleFlag(Module::Max, "air.max_samplers",
                    ConstantInt::get(I32, 16));
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace metalir
