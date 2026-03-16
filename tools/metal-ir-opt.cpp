// metal-ir-opt: standalone tool for Metal IR pipeline
//
// Usage:
//   metal-ir-opt input.ll -o output.metallib            (full pipeline → metallib)
//   metal-ir-opt input.ll --emit-llvm -o output.ll      (full pipeline → IR dump)
//   metal-ir-opt input.ll --pass=decompose-struct-phis --emit-llvm  (single pass)
//   metal-ir-opt input.ll --pass=barrier-rename,tg-barrier-insert --emit-llvm
//
// Pass names correspond to Pipeline.h pass structs in kebab-case:
//   inline, decompose-struct-phis, ptr-phi-to-i64, barrier-rename,
//   tg-barrier-insert, nan-min-max, lower-fneg, bitcast-zero-init,
//   llvm-to-air-intrinsics, lower-int-min-max, split-i64-shuffle,
//   lower-atomic-rmw, tg-global-dead-elim, tg-global-coalesce,
//   tg-global-gep-rewrite, infer-typed-pointers, mma-typed-pointers,
//   scalar-buffer-packing, scalar-store-guard, air-system-values,
//   normalize-i1-pointers, device-loads-volatile, widen-device-loads,
//   bfloat16-cast-decompose, normalize-allocas

#include "metal-ir/Pipeline.h"
#include "metal-ir/KernelProfile.h"
#include "metal-ir/PointeeTypeMap.h"
#include "metal-ir/MetallibWriter.h"

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> InputFilename(cl::Positional,
                                           cl::desc("<input .ll file>"),
                                           cl::Required);

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                            cl::value_desc("filename"),
                                            cl::init("-"));

static cl::opt<bool> EmitLLVM("emit-llvm",
                               cl::desc("Emit LLVM IR instead of metallib"),
                               cl::init(false));

static cl::opt<std::string> PassNames("pass",
                                       cl::desc("Run specific pass(es), comma-separated"),
                                       cl::value_desc("name[,name,...]"),
                                       cl::init(""));

/// Add a single named pass to the pipeline. Returns false if unknown.
static bool addNamedPass(ModulePassManager &MPM, StringRef name) {
  if (name == "inline")
    MPM.addPass(metalir::InlineNonKernelFunctionsPass());
  else if (name == "decompose-struct-phis")
    MPM.addPass(metalir::DecomposeStructPhisPass());
  else if (name == "ptr-phi-to-i64")
    MPM.addPass(metalir::PtrPhiToI64Pass());
  else if (name == "barrier-rename")
    MPM.addPass(metalir::BarrierRenamePass());
  else if (name == "tg-barrier-insert")
    MPM.addPass(metalir::TGBarrierInsertPass());
  else if (name == "nan-min-max")
    MPM.addPass(metalir::NaNMinMaxPass());
  else if (name == "lower-fneg")
    MPM.addPass(metalir::LowerFNegPass());
  else if (name == "bitcast-zero-init")
    MPM.addPass(metalir::BitcastZeroInitPass());
  else if (name == "llvm-to-air-intrinsics")
    MPM.addPass(metalir::LLVMToAIRIntrinsicsPass());
  else if (name == "lower-int-min-max")
    MPM.addPass(metalir::LowerIntMinMaxPass());
  else if (name == "split-i64-shuffle")
    MPM.addPass(metalir::SplitI64ShufflePass());
  else if (name == "lower-atomic-rmw")
    MPM.addPass(metalir::LowerAtomicRMWPass());
  else if (name == "tg-global-dead-elim")
    MPM.addPass(metalir::TGGlobalDeadElimPass());
  else if (name == "tg-global-coalesce")
    MPM.addPass(metalir::TGGlobalCoalescePass());
  else if (name == "tg-global-gep-rewrite")
    MPM.addPass(metalir::TGGlobalGEPRewritePass());
  else if (name == "infer-typed-pointers")
    MPM.addPass(metalir::InferTypedPointersPass());
  else if (name == "scalar-buffer-packing")
    MPM.addPass(metalir::ScalarBufferPackingPass());
  else if (name == "scalar-store-guard")
    MPM.addPass(metalir::ScalarStoreGuardPass());
  else if (name == "air-system-values")
    MPM.addPass(metalir::AIRSystemValuesPass());
  else if (name == "normalize-i1-pointers")
    MPM.addPass(metalir::NormalizeI1PointersPass());
  else if (name == "device-loads-volatile")
    MPM.addPass(metalir::DeviceLoadsVolatilePass());
  else if (name == "widen-device-loads")
    MPM.addPass(metalir::WidenDeviceLoadsPass());
  else if (name == "bfloat16-cast-decompose")
    MPM.addPass(metalir::BFloat16CastDecomposePass());
  else if (name == "normalize-allocas")
    MPM.addPass(metalir::NormalizeAllocasPass());
  else
    return false;
  return true;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Metal IR Pipeline Tool\n");

  LLVMContext Context;
  SMDiagnostic Err;

  auto M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  // Set up analysis managers
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  MAM.registerPass([&] { return metalir::MMAPresenceAnalysis(); });
  MAM.registerPass([&] { return metalir::TGMemoryAnalysis(); });
  MAM.registerPass([&] { return metalir::PointeeTypeAnalysis(); });
  MAM.registerPass([&] { return metalir::KernelProfileAnalysis(); });

  ModulePassManager MPM;

  if (PassNames.empty()) {
    // Full pipeline
    metalir::buildMetalIRPipeline(MPM);
  } else {
    // Parse comma-separated pass names
    SmallVector<StringRef, 8> names;
    StringRef(PassNames).split(names, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (auto name : names) {
      name = name.trim();
      if (!addNamedPass(MPM, name)) {
        errs() << "Unknown pass: " << name << "\n";
        return 1;
      }
    }
  }

  MPM.run(*M, MAM);

  // Output
  std::error_code EC;
  auto Out = std::make_unique<ToolOutputFile>(OutputFilename, EC,
                                               sys::fs::OF_None);
  if (EC) {
    errs() << "Error opening output: " << EC.message() << "\n";
    return 1;
  }

  if (EmitLLVM || !PassNames.empty()) {
    // Single-pass mode always emits IR (metallib needs full pipeline)
    M->print(Out->os(), nullptr);
  } else {
    auto &PTM = MAM.getResult<metalir::PointeeTypeAnalysis>(*M);
    metalir::writeMetallib(*M, PTM, Out->os());
  }

  Out->keep();
  return 0;
}
