#pragma once

#include <string>

namespace metalir {
namespace air {

// ── Barriers ─────────────────────────────────────────────────────────

constexpr const char *kBarrier = "air.wg.barrier";
constexpr const char *kBarrierOld = "air.threadgroup.barrier";

// ── System value intrinsic calls (pre-lowering) ─────────────────────

constexpr const char *kCallPid = "air.threadgroup_position_in_grid";
constexpr const char *kCallTid = "air.thread_position_in_grid";
constexpr const char *kCallTidTG = "air.thread_position_in_threadgroup";
constexpr const char *kCallSimdlane = "air.thread_index_in_simdgroup";
constexpr const char *kCallNumProg = "air.threadgroups_per_grid";

// ── System value metadata attributes ─────────────────────────────────

constexpr const char *kMDThreadPositionInGrid = "air.thread_position_in_grid";
constexpr const char *kMDThreadgroupPositionInGrid = "air.threadgroup_position_in_grid";
constexpr const char *kMDThreadPositionInTG = "air.thread_position_in_threadgroup";
constexpr const char *kMDThreadIndexInSimdgroup = "air.thread_index_in_simdgroup";
constexpr const char *kMDThreadgroupsPerGrid = "air.threadgroups_per_grid";

// ── Math intrinsics (LLVM name → AIR name) ───────────────────────────

struct IntrinsicMapping {
  const char *llvmName;
  const char *airName;
};

// Direct renames (same semantics)
constexpr IntrinsicMapping kIntrinsicRenames[] = {
    {"llvm.maxnum.f32", "air.fmax.f32"},
    {"llvm.minnum.f32", "air.fmin.f32"},
    {"llvm.maxnum.f16", "air.fmax.f16"},
    {"llvm.minnum.f16", "air.fmin.f16"},

    {"llvm.sin.f32", "air.fast_sin.f32"},
    {"llvm.cos.f32", "air.fast_cos.f32"},
    {"llvm.sin.f16", "air.fast_sin.f16"},
    {"llvm.cos.f16", "air.fast_cos.f16"},

    {"llvm.exp.f32", "air.fast_exp.f32"},
    {"llvm.log.f32", "air.fast_log.f32"},
    {"llvm.exp2.f32", "air.fast_exp2.f32"},
    {"llvm.log2.f32", "air.fast_log2.f32"},
    {"llvm.exp.f16", "air.fast_exp.f16"},
    {"llvm.log.f16", "air.fast_log.f16"},
    {"llvm.exp2.f16", "air.fast_exp2.f16"},
    {"llvm.log2.f16", "air.fast_log2.f16"},

    {"llvm.sqrt.f32", "air.fast_sqrt.f32"},
    {"llvm.fabs.f32", "air.fabs.f32"},
    {"llvm.floor.f32", "air.fast_floor.f32"},
    {"llvm.ceil.f32", "air.fast_ceil.f32"},
    {"llvm.sqrt.f16", "air.fast_sqrt.f16"},
    {"llvm.fabs.f16", "air.fabs.f16"},
    {"llvm.floor.f16", "air.fast_floor.f16"},
    {"llvm.ceil.f16", "air.fast_ceil.f16"},

    {"llvm.fma.f32", "air.fma.f32"},
    {"llvm.fma.f16", "air.fma.f16"},

    {"llvm.rint.f32", "air.fast_rint.f32"},
    {"llvm.rint.f16", "air.fast_rint.f16"},
};

// NaN-propagating min/max → fmin/fmax (with NaN check expansion)
constexpr IntrinsicMapping kNaNMinMax[] = {
    {"llvm.minimum.f32", "air.fmin.f32"},
    {"llvm.maximum.f32", "air.fmax.f32"},
    {"llvm.minimum.f16", "air.fmin.f16"},
    {"llvm.maximum.f16", "air.fmax.f16"},
};

// ── Atomic intrinsic name builder ────────────────────────────────────

enum class AtomicLocality { Global, Local };
enum class AtomicOp { Xchg, Add, Sub, Max, Min, UMax, UMin, And, Or, Xor };
enum class AtomicType { I32, F32 };

/// Build an AIR atomic intrinsic name with correct sign prefix rules.
/// xchg has NO sign prefix. Signed ops get ".s", unsigned get ".u".
inline std::string atomicName(AtomicLocality loc, AtomicOp op, AtomicType ty) {
  std::string name = "air.atomic.";
  name += (loc == AtomicLocality::Global) ? "global" : "local";
  name += '.';

  switch (op) {
  case AtomicOp::Xchg: name += "xchg"; break;
  case AtomicOp::Add:  name += "add.s"; break;
  case AtomicOp::Sub:  name += "sub.s"; break;
  case AtomicOp::Max:  name += "max.s"; break;
  case AtomicOp::Min:  name += "min.s"; break;
  case AtomicOp::UMax: name += "umax.u"; break;
  case AtomicOp::UMin: name += "umin.u"; break;
  case AtomicOp::And:  name += "and.s"; break;
  case AtomicOp::Or:   name += "or.s"; break;
  case AtomicOp::Xor:  name += "xor.s"; break;
  }
  name += '.';

  switch (ty) {
  case AtomicType::I32: name += "i32"; break;
  case AtomicType::F32: name += "f32"; break;
  }

  return name;
}

// ── MMA intrinsic names ──────────────────────────────────────────────

// f32 MMA — threadgroup memory (addrspace 3)
constexpr const char *kMMALoadTG = "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
constexpr const char *kMMAStoreTG = "air.simdgroup_matrix_8x8_store.v64f32.p3f32";

// f32 MMA — device memory (addrspace 1)
constexpr const char *kMMALoadDevice = "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
constexpr const char *kMMAStoreDevice = "air.simdgroup_matrix_8x8_store.v64f32.p1f32";

// f16 MMA — device memory (addrspace 1)
constexpr const char *kMMALoadDeviceF16 = "air.simdgroup_matrix_8x8_load.v64f16.p1f16";
constexpr const char *kMMADeviceF16 = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32";

// bf16 MMA — threadgroup memory (addrspace 3)
constexpr const char *kMMALoadTGBF16 = "air.simdgroup_matrix_8x8_load.v64bf16.p3bf16";

// bf16 MMA — device memory (addrspace 1)
constexpr const char *kMMALoadDeviceBF16 = "air.simdgroup_matrix_8x8_load.v64bf16.p1bf16";
constexpr const char *kMMADeviceBF16 = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64bf16.v64bf16.v64f32";

// ── Metadata string constants ────────────────────────────────────────

constexpr const char *kMDBuffer = "air.buffer";
constexpr const char *kMDLocationIndex = "air.location_index";
constexpr const char *kMDRead = "air.read";
constexpr const char *kMDReadWrite = "air.read_write";
constexpr const char *kMDAddressSpace = "air.address_space";
constexpr const char *kMDArgTypeSize = "air.arg_type_size";
constexpr const char *kMDArgTypeAlignSize = "air.arg_type_align_size";
constexpr const char *kMDArgTypeName = "air.arg_type_name";
constexpr const char *kMDArgName = "air.arg_name";

// Named metadata
constexpr const char *kNMDKernel = "air.kernel";
constexpr const char *kNMDVersion = "air.version";
constexpr const char *kNMDLanguageVersion = "air.language_version";

} // namespace air
} // namespace metalir
