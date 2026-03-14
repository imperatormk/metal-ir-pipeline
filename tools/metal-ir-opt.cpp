// metal-ir-opt: standalone tool for Metal IR pipeline
//
// Usage:
//   metal-ir-opt input.ll -o output.metallib
//   metal-ir-opt input.ll --emit-llvm -o output.ll   (dump after transforms)
//
// This is the C++ equivalent of MetalASM's applyAirTransforms + serialize.

#include "metal-ir/Pipeline.h"
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
                               cl::desc("Emit LLVM IR after transforms instead of metallib"),
                               cl::init(false));

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Metal IR Pipeline Tool\n");

  LLVMContext Context;
  SMDiagnostic Err;

  // Parse input .ll
  auto M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  // Run the Metal IR pipeline
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

  ModulePassManager MPM;
  metalir::buildMetalIRPipeline(MPM);
  MPM.run(*M, MAM);

  // Output
  std::error_code EC;
  auto Out = std::make_unique<ToolOutputFile>(OutputFilename, EC,
                                               sys::fs::OF_None);
  if (EC) {
    errs() << "Error opening output: " << EC.message() << "\n";
    return 1;
  }

  if (EmitLLVM) {
    M->print(Out->os(), nullptr);
  } else {
    auto &PTM = MAM.getResult<metalir::PointeeTypeAnalysis>(*M);
    metalir::writeMetallib(*M, PTM, Out->os());
  }

  Out->keep();
  return 0;
}
