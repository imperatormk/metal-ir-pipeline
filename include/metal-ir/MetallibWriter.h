#pragma once

#include "metal-ir/PointeeTypeMap.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace metalir {

struct MetallibOptions {
  uint8_t platform = 0x81;  // macOS = 0x81, iOS = 0x82
  uint16_t osMajor = 26, osMinor = 0, osPatch = 0; // Always 26.0.0 in metallib
  uint16_t metalMajor = 3, metalMinor = 2;          // Metal language version
};

// Write the module as a metallib to the output stream.
//
// The PointeeTypeMap is critical: LLVM's Module has opaque pointers,
// but Metal's bitcode format requires typed POINTER records. The writer
// uses the map to emit:
//   ptr addrspace(1) %buf  →  float addrspace(1)* %buf  (in bitcode)
//
// Without the map, all pointers would be opaque and Metal would reject
// the metallib.
//
// Pipeline:
//   Module + PointeeTypeMap
//     → custom bitcode (typed POINTER records, not LLVM's BitcodeWriter)
//     → bitcode wrapper (0x0B17C0DE magic)
//     → metallib container (MTLB header + 4 sections)
bool writeMetallib(llvm::Module &M, const PointeeTypeMap &PTM,
                   llvm::raw_ostream &OS,
                   const MetallibOptions &opts = {});

// Convenience: write to a byte vector.
std::vector<uint8_t> serializeMetallib(llvm::Module &M,
                                       const PointeeTypeMap &PTM,
                                       const MetallibOptions &opts = {});

} // namespace metalir
