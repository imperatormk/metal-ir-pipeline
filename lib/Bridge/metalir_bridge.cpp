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

  // Ensure target triple + datalayout (Triton MLIR→LLVM often omits them).
  // Query actual macOS version for the triple.
  {
    unsigned maj = 26, min = 0, pat = 0;
#if __has_include(<sys/sysctl.h>)
    char buf[64] = {};
    size_t len = sizeof(buf);
    if (sysctlbyname("kern.osproductversion", buf, &len, nullptr, 0) == 0)
      sscanf(buf, "%u.%u.%u", &maj, &min, &pat);
#endif
    char triple[128];
    snprintf(triple, sizeof(triple), "air64_v28-apple-macosx%u.%u.%u", maj, min, pat);
    M->setTargetTriple(Triple(triple));
  }
  // Always set the full AIR datalayout (Triton MLIR may emit a truncated one)
  M->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
                     "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32"
                     "-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
                     "-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
                     "-n8:16:32");

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
