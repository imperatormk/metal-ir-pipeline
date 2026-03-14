# metal-ir-pipeline

C++ reimplementation of MetalASM's IR transform pipeline as proper LLVM passes.

## Why

MetalASM (Swift) transforms LLVM IR text with 23 string-munging passes that are
disconnected — no shared analysis, no declared dependencies, touching one breaks
another. This project re-expresses them as composable LLVM `PassInfoMixin` passes
operating on `llvm::Module`, with:

- **Declared dependencies** via the LLVM pass manager
- **Shared analysis** (TGMemoryAnalysis tracks 32KB threadgroup budget)
- **Testable in isolation** — each pass can run standalone on a `.ll` file
- **Pure C++** — no Swift dependency, can be a triton-ext shared library

## Status

Scaffolding. All 22 passes are declared with stubs. Next: port one pass at a time,
starting with the simplest (LowerFNeg, BitcastZeroInit) and working up to the
complex ones (AIRSystemValues, TGGlobalGEPRewrite).

## Build

```bash
cmake -B build -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build
```

## Test

```bash
# Run pipeline on a .ll file, emit transformed IR
./build/tools/metal-ir-opt test/simple_kernel.ll --emit-llvm -o out.ll

# Smoke test
ctest --test-dir build
```

## Architecture

```
include/metal-ir/
  Pipeline.h        — All pass declarations + TGMemoryAnalysis + buildMetalIRPipeline()
  MetallibWriter.h  — Module → metallib serialization

lib/Transforms/
  Pipeline.cpp      — Pass implementations + pipeline builder

lib/Serialization/
  MetallibWriter.cpp — MTLB format writer (bitcode + header + metadata)

tools/
  metal-ir-opt.cpp  — CLI: .ll → pipeline → .metallib or .ll

test/
  simple_kernel.ll  — Smoke test kernel
```

## Pass Inventory

| # | Pass | Complexity | Can move to MLIR? |
|---|------|-----------|-------------------|
| 0 | InlineNonKernelFunctions | High | No (Metal limitation) |
| 1 | DecomposeStructPhis | Medium | Yes — don't emit struct phis |
| 2 | PtrPhiToI64 | Medium | Partially |
| 3 | BarrierRename | Trivial | Yes — emit correct name |
| 4 | TGBarrierInsert | High | Partially |
| 5 | NaNMinMax | Low | Yes |
| 6 | LowerFNeg | Trivial | Yes — emit fsub |
| 7 | BitcastZeroInit | Trivial | Yes |
| 8 | LLVMToAIRIntrinsics | Low | Yes — emit AIR names directly |
| 9 | LowerIntMinMax | Low | Yes |
| 10 | SplitI64Shuffle | Medium | Partially |
| 11 | LowerAtomicRMW | Medium | Yes |
| 12 | TGGlobalDeadElim | Low | No (post-LLVM cleanup) |
| 13 | TGGlobalCoalesce | Medium | No (needs whole-module view) |
| 14 | TGGlobalGEPRewrite | High | No (Metal-specific) |
| 15 | InferTypedPointers | High | No (Metal requires typed ptrs) |
| 16 | MMATypedPointers | Medium | No (Metal-specific) |
| 17 | BFloat16CastDecompose | Low | Yes |
| 18 | ScalarStoreGuard | Low | No (Metal-specific) |
| 19 | AIRSystemValues | High | Partially |
| 20 | DeviceLoadsVolatile | Low | No (Metal GPU JIT bug) |
| 21 | WidenDeviceLoads | High | No (Metal GPU JIT bug) |
