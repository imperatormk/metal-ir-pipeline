// C bridge for metal-ir-pipeline — drop-in replacement for MetalASM bridge.
//
// metalir_compile(ir, outLen, errbuf, errlen) → malloc'd metallib bytes
// Same ABI as metalasm_compile() so Triton can load either dylib.

#include "metal-ir/Pipeline.h"
#include "metal-ir/MetallibWriter.h"

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <cstdlib>
#include <cstring>
#include <sys/sysctl.h>

using namespace llvm;

static void setError(char *errbuf, int errlen, const char *msg) {
  if (errbuf && errlen > 0) {
    strncpy(errbuf, msg, errlen - 1);
    errbuf[errlen - 1] = 0;
  }
}

extern "C" {

__attribute__((visibility("default")))
void *metalir_compile(const char *llText, uint64_t *outLen,
                      char *errbuf, int errlen) {
  LLVMContext Ctx;
  SMDiagnostic Err;

  auto Buf = MemoryBuffer::getMemBuffer(llText, "triton_kernel");
  auto M = parseIR(*Buf, Err, Ctx);
  if (!M) {
    std::string msg;
    raw_string_ostream OS(msg);
    Err.print("metalir", OS);
    setError(errbuf, errlen, msg.c_str());
    *outLen = 0;
    return nullptr;
  }

  // Debug: dump original IR before transforms
  if (getenv("METALIR_DUMP_IR")) {
    std::error_code EC;
    raw_fd_ostream dump("/tmp/metalir_original.ll", EC);
    if (!EC) M->print(dump, nullptr);
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

  // Don't set target triple or datalayout on the module — the metallib header
  // has its own platform bytes, and these in the bitcode can cause
  // materializeAll failures for complex kernels on some GPU JIT versions.
  // Skip datalayout too:
  // M->setDataLayout(...);

  // Debug: dump transformed IR if METALIR_DUMP_IR is set
  if (getenv("METALIR_DUMP_IR")) {
    std::error_code EC;
    raw_fd_ostream dump("/tmp/metalir_transformed.ll", EC);
    if (!EC) M->print(dump, nullptr);
  }

  // Serialize
  auto &PTM = MAM.getResult<metalir::PointeeTypeAnalysis>(*M);
  auto bytes = metalir::serializeMetallib(*M, PTM);

  if (bytes.empty()) {
    setError(errbuf, errlen, "serializeMetallib returned empty");
    *outLen = 0;
    return nullptr;
  }

  void *result = malloc(bytes.size());
  memcpy(result, bytes.data(), bytes.size());
  *outLen = bytes.size();
  return result;
}

__attribute__((visibility("default")))
void metalir_free(void *ptr) { free(ptr); }

} // extern "C"
