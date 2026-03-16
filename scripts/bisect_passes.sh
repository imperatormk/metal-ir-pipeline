#!/bin/bash
# Bisect which IR transform pass breaks the fused_recurrent kernel.
# Uses MetalASM as the oracle (known to produce correct bitcode).

LLIR="/tmp/metalir_original.ll"
LLVM_DIR="$HOME/projects/oss/llvm"
PIPELINE_DIR="$HOME/projects/oss/metal-ir-pipeline"

PASSES=(
  InlineNonKernelFunctions
  DecomposeStructPhis
  PtrPhiToI64
  BarrierRename
  TGBarrierInsert
  NaNMinMax
  LowerFNeg
  BitcastZeroInit
  LLVMToAIRIntrinsics
  LowerIntMinMax
  SplitI64Shuffle
  LowerAtomicRMW
  TGGlobalDeadElim
  TGGlobalCoalesce
  TGGlobalGEPRewrite
  InferTypedPointers
  MMATypedPointers
  ScalarBufferPacking
  ScalarStoreGuard
  AIRSystemValues
  NormalizeI1Pointers
  DeviceLoadsVolatile
  WidenDeviceLoads
  BFloat16CastDecompose
  NormalizeAllocas
)

echo "Testing original IR..."
cd "$LLVM_DIR"
TEST_LLIR="$LLIR" swift test --filter testExternalLLIR 2>&1 | grep -q "passed"
if [ $? -ne 0 ]; then
  echo "  FAIL on original — cannot bisect"
  exit 1
fi
echo "  PASS"

# For each pass count, run pipeline with only that many passes
for i in $(seq 1 ${#PASSES[@]}); do
  PASS_NAME="${PASSES[$((i-1))]}"

  # Use metal-ir-opt with --stop-after (not supported) — instead,
  # modify pipeline temporarily... actually let's just use the
  # METALIR_STOP_AFTER env var if we add one.
  # For now, just test each dumped intermediate.
  echo -n "  Pass $i/${#PASSES[@]} ($PASS_NAME)... "

  # Run pipeline stopping at pass $i
  METALIR_STOP_AFTER=$i "$PIPELINE_DIR/build/tools/metal-ir-opt" \
    "$LLIR" --emit-llvm -o "/tmp/pass_${i}.ll" 2>/dev/null

  if [ ! -f "/tmp/pass_${i}.ll" ]; then
    echo "SKIP (no output)"
    continue
  fi

  TEST_LLIR="/tmp/pass_${i}.ll" swift test --filter testExternalLLIR 2>&1 | grep -q "passed"
  if [ $? -eq 0 ]; then
    echo "PASS"
  else
    echo "FAIL ← this pass breaks it!"
    echo ""
    echo "Last good: pass $((i-1)) (${PASSES[$((i-2))]})"
    echo "First bad: pass $i ($PASS_NAME)"
    exit 0
  fi
done

echo "All passes pass — issue is in serialization, not transforms"
