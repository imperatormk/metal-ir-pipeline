#pragma once

#include "metal-ir/PointeeTypeMap.h"
#include "llvm/IR/Module.h"
#include <vector>

namespace metalir {

/// Emit LLVM Module as bitcode with Metal-compatible typed pointers.
///
/// LLVM 17+ only supports opaque pointers in-memory, but Metal's GPU JIT
/// requires typed POINTER records (code 8) in bitcode. This emitter walks
/// the Module and uses the PointeeTypeMap to emit typed pointer type records
/// instead of opaque pointer records (code 25).
///
/// This replaces LLVM's WriteBitcodeToFile for Metal targets.
std::vector<uint8_t> emitMetalBitcode(llvm::Module &M,
                                       const PointeeTypeMap &PTM);

} // namespace metalir
