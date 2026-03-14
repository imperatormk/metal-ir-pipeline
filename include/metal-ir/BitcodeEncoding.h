#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"

namespace metalir {

inline unsigned encodeBinop(unsigned Op) {
  using namespace llvm;
  switch (Op) {
  case Instruction::Add: case Instruction::FAdd: return 0;
  case Instruction::Sub: case Instruction::FSub: return 1;
  case Instruction::Mul: case Instruction::FMul: return 2;
  case Instruction::UDiv: return 3;
  case Instruction::SDiv: case Instruction::FDiv: return 4;
  case Instruction::URem: return 5;
  case Instruction::SRem: case Instruction::FRem: return 6;
  case Instruction::Shl: return 7;
  case Instruction::LShr: return 8;
  case Instruction::AShr: return 9;
  case Instruction::And: return 10;
  case Instruction::Or: return 11;
  case Instruction::Xor: return 12;
  default: return 0;
  }
}

inline unsigned encodeCast(unsigned Op) {
  using namespace llvm;
  switch (Op) {
  case Instruction::Trunc: return 0;
  case Instruction::ZExt: return 1;
  case Instruction::SExt: return 2;
  case Instruction::FPToUI: return 3;
  case Instruction::FPToSI: return 4;
  case Instruction::UIToFP: return 5;
  case Instruction::SIToFP: return 6;
  case Instruction::FPTrunc: return 7;
  case Instruction::FPExt: return 8;
  case Instruction::PtrToInt: return 9;
  case Instruction::IntToPtr: return 10;
  case Instruction::BitCast: return 11;
  case Instruction::AddrSpaceCast: return 12;
  default: return 0;
  }
}

inline unsigned encodeLinkage(llvm::GlobalValue::LinkageTypes L) {
  using namespace llvm;
  switch (L) {
  case GlobalValue::ExternalLinkage: return 0;
  case GlobalValue::WeakAnyLinkage: return 1;
  case GlobalValue::InternalLinkage: return 3;
  case GlobalValue::PrivateLinkage: return 9;
  default: return 0;
  }
}

inline void emitString(llvm::BitstreamWriter &W, unsigned Code,
                        llvm::StringRef S) {
  llvm::SmallVector<uint64_t, 64> V;
  for (char C : S) V.push_back((uint64_t)(unsigned char)C);
  W.EmitRecord(Code, V);
}

} // namespace metalir
