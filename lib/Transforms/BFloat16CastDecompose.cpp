// Pass 17: Decompose bf16 casts via float intermediate.
// Metal GPU JIT treats sitofp iN→bfloat as sitofp iN→half (wrong).

#include "metal-ir/Pipeline.h"
#include "metal-ir/PassUtil.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
namespace metalir {

bool BFloat16CastDecomposePass::needsRun(Module &M) {
  Type *BF16 = Type::getBFloatTy(M.getContext());
  Type *F32 = Type::getFloatTy(M.getContext());
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (isa<SIToFPInst>(&I) || isa<UIToFPInst>(&I)) {
          // bf16 casts (Phase 1)
          if (I.getType() == BF16) return true;
          // sub-32-bit int to float casts (Phase 2)
          if (I.getType() == F32) {
            unsigned bits = I.getOperand(0)->getType()->getIntegerBitWidth();
            if (bits < 32) return true;
          }
        }
  return false;
}

PreservedAnalyses BFloat16CastDecomposePass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  bool changed = false;
  Type *BF16 = Type::getBFloatTy(M.getContext());
  Type *F32 = Type::getFloatTy(M.getContext());

  // Phase 1: Decompose sitofp/uitofp iN→bfloat via float intermediate
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        auto *I = &*it++;
        if ((isa<SIToFPInst>(I) || isa<UIToFPInst>(I)) && I->getType() == BF16) {
          IRBuilder<> B(I);
          Value *ToFloat = isa<SIToFPInst>(I)
              ? B.CreateSIToFP(I->getOperand(0), F32, "to_f32")
              : B.CreateUIToFP(I->getOperand(0), F32, "to_f32");
          // Decompose f32→bf16 via bit manipulation instead of fptrunc,
          // because Metal v1 bitcode doesn't support bfloat in CAST ops.
          // bf16 is just the upper 16 bits of f32.
          Type *I32T = Type::getInt32Ty(M.getContext());
          Type *I16T = Type::getInt16Ty(M.getContext());
          Value *AsInt = B.CreateBitCast(ToFloat, I32T, "f32_bits");
          Value *Shifted = B.CreateLShr(AsInt, 16, "bf16_bits");
          Value *Narrow = B.CreateTrunc(Shifted, I16T, "bf16_i16");
          Value *Trunc = B.CreateBitCast(Narrow, BF16, I->getName());
          I->replaceAllUsesWith(Trunc);
          I->eraseFromParent();
          changed = true;
        }
      }
    }
  }

  // Phase 2: Widen sitofp/uitofp i8/i16→float to i32 intermediate.
  // Metal GPU JIT with v1 bitcode doesn't support sitofp for sub-32-bit
  // integer types. Widen to i32 first: sext/zext iN→i32 + sitofp/uitofp i32→float.
  // This also eliminates i8/i16 values from the IR, avoiding typed pointer
  // issues with the i8 type in Metal bitcode.
  Type *I32 = Type::getInt32Ty(M.getContext());
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        auto *I = &*it++;
        bool isSigned = isa<SIToFPInst>(I);
        if (!isSigned && !isa<UIToFPInst>(I)) continue;
        if (I->getType() != F32) continue;

        Type *srcTy = I->getOperand(0)->getType();
        unsigned bits = srcTy->getIntegerBitWidth();
        if (bits >= 32) continue; // i32→float is fine as sitofp

        IRBuilder<> B(I);
        Value *wide = isSigned
            ? B.CreateSExt(I->getOperand(0), I32, "sext_i32")
            : B.CreateZExt(I->getOperand(0), I32, "zext_i32");
        Value *fp = isSigned
            ? B.CreateSIToFP(wide, F32, I->getName())
            : B.CreateUIToFP(wide, F32, I->getName());
        I->replaceAllUsesWith(fp);
        I->eraseFromParent();
        changed = true;
      }
    }
  }
  // Phase 3: Fold sext/zext(trunc i32 to iN) to i32 into bit operations.
  // Eliminates sub-32-bit integer intermediates:
  //   sext i8 (trunc i32 %x to i8) to i32 → (shl i32 %x, 24) ashr 24
  //   zext i8 (trunc i32 %x to i8) to i32 → and i32 %x, 255
  //   sext i16 (trunc i32 %x to i16) to i32 → (shl i32 %x, 16) ashr 16
  //   zext i16 (trunc i32 %x to i16) to i32 → and i32 %x, 65535
  SmallVector<Instruction *, 16> deadInsts;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *Ext = dyn_cast<CastInst>(&I);
        if (!Ext) continue;
        bool isSExt = isa<SExtInst>(Ext);
        if (!isSExt && !isa<ZExtInst>(Ext)) continue;
        if (Ext->getType() != I32) continue;

        auto *Trn = dyn_cast<TruncInst>(Ext->getOperand(0));
        if (!Trn || Trn->getOperand(0)->getType() != I32) continue;

        unsigned narrowBits = Trn->getType()->getIntegerBitWidth();
        unsigned shiftAmt = 32 - narrowBits;
        IRBuilder<> B(Ext);
        Value *src = Trn->getOperand(0);
        Value *result;
        if (isSExt) {
          Value *shl = B.CreateShl(src, shiftAmt, "sext_shl");
          result = B.CreateAShr(shl, shiftAmt, Ext->getName());
        } else {
          uint32_t mask = (1u << narrowBits) - 1;
          result = B.CreateAnd(src, mask, Ext->getName());
        }
        Ext->replaceAllUsesWith(result);
        deadInsts.push_back(Ext);
        changed = true;
      }
    }
  }
  for (auto *I : deadInsts) I->eraseFromParent();

  // Phase 4: Remove dead trunc-to-sub32 instructions.
  bool progress = true;
  while (progress) {
    progress = false;
    for (auto &F : M) {
      for (auto &BB : F) {
        for (auto it = BB.begin(); it != BB.end();) {
          auto *Trn = dyn_cast<TruncInst>(&*it++);
          if (!Trn) continue;
          unsigned bits = Trn->getType()->getIntegerBitWidth();
          if (bits >= 32) continue;
          if (Trn->use_empty()) {
            Trn->eraseFromParent();
            progress = true;
          }
        }
      }
    }
  }

  // Phase 5: Remove dead air.convert declarations.
  for (auto it = M.begin(); it != M.end();) {
    Function &Fn = *it++;
    if (Fn.isDeclaration() && Fn.use_empty() &&
        Fn.getName().starts_with("air.convert"))
      Fn.eraseFromParent();
  }

  return preserveIf(changed);
}

} // namespace metalir
