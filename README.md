# metal-ir-pipeline

Compiles LLVM IR into Metal GPU binaries (`.metallib`). Takes standard LLVM IR as input, applies a series of transforms to conform to Metal AIR constraints, serializes to Metal v1 bitcode, and produces a ready-to-load `.metallib`.

Built as a shared library (`libMetalIRBridge.dylib`) for integration with compilers that target Apple GPUs, and as a standalone CLI tool (`metal-ir-opt`) for testing.

## What it does

Apple's Metal GPU runtime expects bitcode in a specific dialect of LLVM IR called AIR (Apple Intermediate Representation), packaged in the `.metallib` container format. AIR differs from standard LLVM IR in several ways:

- **Typed pointers** — Metal uses LLVM's legacy typed pointer representation (`float*`, `i32*`), not opaque `ptr`
- **AIR intrinsics** — Standard LLVM operations must be lowered to `air.*` intrinsic calls (barriers, atomics, shuffles, system values)
- **Threadgroup memory constraints** — Global variables in address space 3, max 32KB, specific layout rules
- **No bfloat16 type** — Metal v1 bitcode predates LLVM's bfloat support; must be lowered to bit manipulation
- **No sub-32-bit integer operations** — `sitofp i8→float` must go through `i32` intermediate
- **Metallib container** — MTLB header + entry table + function metadata + wrapped bitcode

This pipeline handles all of these transformations automatically.

## Build

Requires an LLVM 19+ installation (headers + libraries).

```bash
cmake -B build -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build
```

## Usage

### As a library

```c
// Link against libMetalIRBridge.dylib
extern "C" int metalir_compile(
    const char *llvm_ir,        // LLVM IR text (.ll format)
    unsigned ir_len,
    const char **out_metallib,  // Output metallib bytes
    unsigned *out_len,
    char **out_error
);
```

### As a CLI tool

```bash
# LLVM IR → metallib
./build/tools/metal-ir-opt input.ll -o output.metallib

# LLVM IR → transformed LLVM IR (for debugging)
./build/tools/metal-ir-opt input.ll --emit-llvm -o transformed.ll
```

## Transform passes

The pipeline applies 26 passes in a fixed order. Each pass is a standard LLVM `PassInfoMixin` operating on `llvm::Module`.

| Pass | Purpose |
|------|---------|
| InlineNonKernel | Inline all non-kernel functions (Metal has no call stack) |
| DecomposeStructPhis | Split struct-typed phi nodes into scalar phis (GPU JIT crash) |
| PtrPhiToI64 | Convert pointer phi nodes to integer (ptrtoint/inttoptr) |
| BarrierRename | Rename barrier intrinsics to AIR calling convention |
| TGBarrierInsert | Insert threadgroup memory barriers for WAR hazards |
| NaNMinMax | Replace min/max with NaN-safe AIR intrinsics |
| LowerFNeg | Lower `fneg` to `fsub -0.0, x` |
| BitcastZeroInit | Zero-initialize bitcast-created values |
| LLVMToAIRIntrinsics | Lower LLVM intrinsics to AIR equivalents (`air.popcount`, `air.clz`, etc.) |
| LowerIntMinMax | Lower integer min/max to AIR intrinsics |
| SplitI64Shuffle | Split 64-bit simd shuffles into two 32-bit shuffles |
| LowerAtomicRMW | Lower `atomicrmw` to AIR atomic intrinsic calls |
| NormalizeAllocas | Insert bitcasts for type-mismatched threadgroup stores |
| NormalizeI1Pointers | Rewrite `i1*` pointers to `i8*` (Metal has no i1 memory type) |
| ScalarBufferPacking | Pack scalar kernel parameters into a single device buffer |
| TGGlobalDeadElim | Remove unused threadgroup globals |
| TGGlobalCoalesce | Merge threadgroup globals to reduce count (GPU JIT constraint) |
| TGGlobalGEPRewrite | Rewrite GEPs for merged/retyped threadgroup globals |
| InferTypedPointers | Infer typed pointer representations from usage patterns |
| MMATypedPointers | Set correct typed pointers for MMA (matrix multiply-accumulate) intrinsics |
| BFloat16CastDecompose | Lower bfloat16 casts to bit manipulation + f32 intermediates |
| ScalarStoreGuard | Guard scalar stores against out-of-bounds threadgroup access |
| DeviceLoadsVolatile | Mark device loads volatile to prevent GPU JIT hoisting bugs |
| WidenDeviceLoads | Widen sub-32-bit device loads to 32-bit with byte extraction |
| AIRSystemValues | Attach AIR metadata for kernel parameters and system values |

### Serialization

After transforms, the module is serialized to Metal v1 bitcode and packaged:

| Component | Purpose |
|-----------|---------|
| TypeTableWriter | Emit type table with typed pointer records |
| ValueEnumerator | Assign value IDs with per-parameter pointee type inference |
| ConstantsWriter | Emit module and function-level constants |
| FunctionWriter | Emit function bodies (instructions, basic blocks) |
| MetadataWriter | Emit AIR kernel metadata (parameters, system values) |
| MetallibWriter | Package bitcode into MTLB container format |

## Architecture

```
include/metal-ir/
  Pipeline.h         — Pass declarations, analysis, pipeline builder
  MetallibWriter.h   — Module → metallib serialization API
  BitcodeEncoding.h  — Bitcode encoding helpers
  ValueEnumerator.h  — Value/type enumeration for bitcode emission
  PassUtil.h         — Shared pass utilities
  PointeeTypeMap.h   — Pointer type inference from usage

lib/Transforms/      — 26 transform passes (one .cpp each)
lib/Serialization/   — Bitcode + metallib serialization
lib/Bridge/          — C API bridge (metalir_compile)

tools/
  metal-ir-opt.cpp   — CLI tool

test/
  simple_kernel.ll   — Smoke test kernel
```

## Integration

Currently used as the compilation backend for the [Triton Apple GPU backend](https://github.com/imperatormk/triton/tree/apple-mps-backend), replacing the previous Swift-based MetalASM pipeline. The shared library is loaded at runtime via `ctypes` and called through the C bridge API.

## License

MIT
