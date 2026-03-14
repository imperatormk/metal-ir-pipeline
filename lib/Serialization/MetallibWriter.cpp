// ═══════════════════════════════════════════════════════════════════════
// MetallibWriter — LLVM Module → metallib container
//
// Format (confirmed by diffing xcrun metallib output):
//   [0:88]    MTLB header (magic + platform + filesize + 4 section descs)
//   [88:...]  Section 0: entry headers (u32 count + per-entry blocks)
//   [gap]     2× ENDT (8 bytes, NOT counted in section 0 size)
//   [...]     Section 1: function list (u32(4) + ENDT)
//   [...]     Section 2: public metadata (u32(4) + ENDT)
//   [...]     Section 3: wrapped bitcode (0x0B17C0DE wrapper + raw bitcode)
//
// Entry header tags are 4-byte ASCII + 2-byte LE length + payload.
// Exception: ENDT is just 4 bytes (no length field).
// ═══════════════════════════════════════════════════════════════════════

#include "metal-ir/MetallibWriter.h"
#include "metal-ir/BitcodeEmitter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

using namespace llvm;

namespace metalir {

// ── Helpers ──────────────────────────────────────────────────────────────

static void writeU16(raw_ostream &OS, uint16_t v) {
  OS.write(reinterpret_cast<const char *>(&v), 2);
}
static void writeU32(raw_ostream &OS, uint32_t v) {
  OS.write(reinterpret_cast<const char *>(&v), 4);
}
static void writeU64(raw_ostream &OS, uint64_t v) {
  OS.write(reinterpret_cast<const char *>(&v), 8);
}

static void writeTag(raw_ostream &OS, const char *name, uint16_t len) {
  OS.write(name, 4);
  writeU16(OS, len);
}

static void writeENDT(raw_ostream &OS) { OS.write("ENDT", 4); }

// ── Bitcode ──────────────────────────────────────────────────────────────

// generateBitcode is now in BitcodeEmitter.cpp (emitMetalBitcode)
// It emits typed pointers using the PointeeTypeMap.

static std::vector<uint8_t> wrapBitcode(const std::vector<uint8_t> &bc) {
  std::string buf;
  raw_string_ostream rso(buf);
  writeU32(rso, 0x0B17C0DE);     // wrapper magic
  writeU32(rso, 0);              // version
  writeU32(rso, 20);             // offset to bitcode
  writeU32(rso, bc.size());      // bitcode size
  writeU32(rso, 0xFFFFFFFF);    // CPU type
  rso.write(reinterpret_cast<const char *>(bc.data()), bc.size());
  rso.flush();
  return std::vector<uint8_t>(buf.begin(), buf.end());
}

// ── Entry header (tags only, no size prefix, no ENDT) ────────────────────

static std::string buildEntryTags(StringRef name, ArrayRef<uint8_t> hash,
                                   uint64_t bitcodeSize,
                                   const MetallibOptions &opts) {
  std::string buf;
  raw_string_ostream OS(buf);

  // NAME
  writeTag(OS, "NAME", name.size() + 1);
  OS << name;
  OS.write('\0');

  // TYPE (2 = kernel)
  writeTag(OS, "TYPE", 1);
  OS.write(char(2));

  // HASH (SHA256 of wrapped bitcode)
  writeTag(OS, "HASH", 32);
  OS.write(reinterpret_cast<const char *>(hash.data()), 32);

  // MDSZ (u64 = bitcode section size)
  writeTag(OS, "MDSZ", 8);
  writeU64(OS, bitcodeSize);

  // OFFT (3 × u64, all zero for single-entry)
  writeTag(OS, "OFFT", 24);
  writeU64(OS, 0);
  writeU64(OS, 0);
  writeU64(OS, 0);

  // VERS (air_major=2, air_minor=8, metal_major, metal_minor)
  writeTag(OS, "VERS", 8);
  writeU16(OS, 2);
  writeU16(OS, 8);
  writeU16(OS, opts.metalMajor);
  writeU16(OS, opts.metalMinor);

  OS.flush();
  return buf;
}

// ── Main writer ──────────────────────────────────────────────────────────

bool writeMetallib(Module &M, const PointeeTypeMap &PTM,
                   raw_ostream &OS, const MetallibOptions &opts) {
  // Emit bitcode with typed pointers (the whole point of this project)
  auto bitcode = emitMetalBitcode(M, PTM);
  auto wrappedBC = wrapBitcode(bitcode);
  auto hash = SHA256::hash(ArrayRef<uint8_t>(wrappedBC));

  // Collect kernel names from !air.kernel metadata
  SmallVector<std::string, 4> kernelNames;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    if (auto *KMD = M.getNamedMetadata("air.kernel")) {
      for (unsigned i = 0; i < KMD->getNumOperands(); i++) {
        auto *Node = KMD->getOperand(i);
        if (Node->getNumOperands() > 0) {
          if (auto *FnMD = dyn_cast_if_present<ValueAsMetadata>(Node->getOperand(0))) {
            if (FnMD->getValue() == &F)
              kernelNames.push_back(F.getName().str());
          }
        }
      }
    }
  }

  if (kernelNames.empty()) {
    OS.write(reinterpret_cast<const char *>(bitcode.data()), bitcode.size());
    return true;
  }

  // ── Build section 0: entry headers ─────────────────────────────────────
  // Format: u32(entry_count) + per entry: u32(entry_size) + tags
  // entry_size = tags.size() + 4(ENDT) + 4(ENDT)
  // The 2 ENDTs are written AFTER section 0 (in a gap, not counted in sec0 size)

  std::string sec0;
  {
    raw_string_ostream S0(sec0);
    writeU32(S0, kernelNames.size());
    S0.flush();
  }

  // Build per-entry tag data
  SmallVector<std::string, 4> entryTags;
  for (auto &name : kernelNames) {
    entryTags.push_back(buildEntryTags(name, hash, wrappedBC.size(), opts));
  }

  // Append entry size + tags to sec0
  for (auto &tags : entryTags) {
    std::string entry;
    raw_string_ostream ES(entry);
    uint32_t entrySize = tags.size() + 4 + 4; // tags + ENDT + ENDT
    writeU32(ES, entrySize);
    ES << tags;
    ES.flush();
    sec0 += entry;
  }

  // The 2 ENDTs per entry go in a gap after section 0
  std::string endtGap;
  for (size_t i = 0; i < kernelNames.size(); i++) {
    endtGap += "ENDT";
    endtGap += "ENDT";
  }

  // Section 1 & 2: u32(4) + ENDT
  std::string sec12;
  {
    raw_string_ostream S12(sec12);
    writeU32(S12, 4);
    writeENDT(S12);
    S12.flush();
  }

  // ── Compute layout ─────────────────────────────────────────────────────

  uint64_t headerSize = 88;
  uint64_t sec0_offset = headerSize;
  uint64_t sec0_size = sec0.size();
  uint64_t gap_size = endtGap.size();
  uint64_t sec1_offset = sec0_offset + sec0_size + gap_size;
  uint64_t sec1_size = sec12.size();
  uint64_t sec2_offset = sec1_offset + sec1_size;
  uint64_t sec2_size = sec12.size();
  uint64_t sec3_offset = sec2_offset + sec2_size;
  uint64_t sec3_size = wrappedBC.size();
  uint64_t totalSize = sec3_offset + sec3_size;

  // ── Write MTLB header ─────────────────────────────────────────────────

  OS.write("MTLB", 4);

  uint8_t platform[12] = {0x01, 0x80, 0x02, 0x00, 0x09, 0x00,
                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  platform[7] = opts.platform;
  platform[8] = opts.osMajor & 0xFF;
  OS.write(reinterpret_cast<const char *>(platform), 12);

  writeU64(OS, totalSize);

  writeU64(OS, sec0_offset); writeU64(OS, sec0_size);
  writeU64(OS, sec1_offset); writeU64(OS, sec1_size);
  writeU64(OS, sec2_offset); writeU64(OS, sec2_size);
  writeU64(OS, sec3_offset); writeU64(OS, sec3_size);

  // ── Write body ─────────────────────────────────────────────────────────

  OS << sec0;
  OS << endtGap;
  OS << sec12;  // section 1
  OS << sec12;  // section 2
  OS.write(reinterpret_cast<const char *>(wrappedBC.data()), wrappedBC.size());

  return true;
}

std::vector<uint8_t> serializeMetallib(Module &M, const PointeeTypeMap &PTM,
                                       const MetallibOptions &opts) {
  std::string buf;
  raw_string_ostream rso(buf);
  writeMetallib(M, PTM, rso, opts);
  rso.flush();
  return std::vector<uint8_t>(buf.begin(), buf.end());
}

} // namespace metalir
