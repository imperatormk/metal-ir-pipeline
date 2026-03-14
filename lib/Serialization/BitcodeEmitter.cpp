// BitcodeEmitter — Top-level orchestrator for Metal v1 bitcode emission.
// Delegates to: ValueEnumerator, TypeTableWriter, ConstantsWriter,
//               MetadataWriter, FunctionWriter.

#include "metal-ir/BitcodeEmitter.h"
#include "metal-ir/BitcodeEncoding.h"
#include "metal-ir/ValueEnumerator.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace metalir {

// Forward declarations (defined in separate .cpp files)
void emitTypeBlock(BitstreamWriter &W, ValueEnumerator &E);
void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                         ArrayRef<const Constant *> constants, unsigned codeSize);
void emitMetadataKindBlock(BitstreamWriter &W);
void emitMetadataBlock(BitstreamWriter &W, Module &M, ValueEnumerator &E);
void emitOperandBundleTagsBlock(BitstreamWriter &W);
void emitSinglethreadBlock(BitstreamWriter &W);
void emitFunctionBlock(BitstreamWriter &W, const Function &F, ValueEnumerator &E);

std::vector<uint8_t> emitMetalBitcode(Module &M, const PointeeTypeMap &PTM) {
  SmallVector<char, 0> Buf;
  BitstreamWriter W(Buf);

  // BC magic
  W.Emit('B', 8); W.Emit('C', 8); W.Emit(0xC0, 8); W.Emit(0xDE, 8);

  // IDENTIFICATION
  W.EnterSubblock(bitc::IDENTIFICATION_BLOCK_ID, 5);
  emitString(W, bitc::IDENTIFICATION_CODE_STRING, "MetalIR");
  { SmallVector<uint64_t, 1> V = {0}; W.EmitRecord(bitc::IDENTIFICATION_CODE_EPOCH, V); }
  W.ExitBlock();

  // Enumerate
  ValueEnumerator E(M, PTM);

  // MODULE_BLOCK (CodeSize=4)
  W.EnterSubblock(bitc::MODULE_BLOCK_ID, 4);

  { SmallVector<uint64_t, 1> V = {1}; W.EmitRecord(bitc::MODULE_CODE_VERSION, V); }

  // Emit PARAMATTR blocks BEFORE TYPE_BLOCK (Metal requires this order).
  // MMA load needs nocapture+readonly on its pointer param.
  bool hasMMALoad = false;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_load"))
      hasMMALoad = true;

  if (hasMMALoad) {
    // PARAMATTR_GROUP_BLOCK: group 1 = {param 1: nocapture, readonly}
    W.EnterSubblock(bitc::PARAMATTR_GROUP_BLOCK_ID, 4);
    {
      // Group entry: [grp_id, param_idx, attr_kind, ...]
      // nocapture = enum 21, readonly = enum 36 (LLVM attr enum IDs)
      SmallVector<uint64_t, 8> Grp;
      Grp.push_back(1);  // group ID
      Grp.push_back(1);  // param index (1 = first ptr param of MMA load)
      Grp.push_back(0); Grp.push_back(11); // enum: noCapture (ATTR_KIND=11)
      Grp.push_back(0); Grp.push_back(21); // enum: readOnly (ATTR_KIND=21)
      W.EmitRecord(3, Grp); // PARAMATTR_GRP_CODE_ENTRY = 3
    }
    W.ExitBlock();

    // PARAMATTR_BLOCK: list 1 = [group 1]
    W.EnterSubblock(bitc::PARAMATTR_BLOCK_ID, 4);
    {
      SmallVector<uint64_t, 2> List;
      List.push_back(1); // group ID
      W.EmitRecord(2, List); // PARAMATTR_CODE_ENTRY = 2
    }
    W.ExitBlock();
  }

  emitTypeBlock(W, E);

  { std::string T = M.getTargetTriple().str();
    if (!T.empty()) emitString(W, bitc::MODULE_CODE_TRIPLE, T); }

  if (!M.getDataLayoutStr().empty())
    emitString(W, bitc::MODULE_CODE_DATALAYOUT, M.getDataLayoutStr());

  if (!M.getSourceFileName().empty())
    emitString(W, bitc::MODULE_CODE_SOURCE_FILENAME, M.getSourceFileName());

  // GLOBALVAR and FUNCTION records — emit in globalValues order
  // (globals first, then functions, matching value ID assignment)
  for (auto *V : E.globalValues) {
    if (auto *G = dyn_cast<GlobalVariable>(V)) {
      SmallVector<uint64_t, 14> Ops;
      Ops.push_back(E.globalPtrTypeIdx(G)); // ptr-to-valueType
      Ops.push_back(G->isConstant() ? 1 : 0);
      Ops.push_back(G->hasInitializer() ? E.moduleConstIdx(G->getInitializer()) + 1 : 0);
      Ops.push_back(encodeLinkage(G->getLinkage()));
      Ops.push_back(G->getAlign() ? Log2_32(G->getAlign()->value()) + 1 : 0);
      for (int i = 0; i < 3; i++) Ops.push_back(0);
      Ops.push_back(G->hasGlobalUnnamedAddr() ? 1 : 0);
      Ops.push_back(G->isExternallyInitialized() ? 1 : 0);
      Ops.push_back(0); Ops.push_back(0);
      Ops.push_back(G->getAddressSpace());
      Ops.push_back(0);
      W.EmitRecord(bitc::MODULE_CODE_GLOBALVAR, Ops);
    } else if (auto *Fn = dyn_cast<Function>(V)) {
      SmallVector<uint64_t, 17> Ops;
      Ops.push_back(E.typeIdx(Fn->getFunctionType()));
      Ops.push_back(Fn->getCallingConv());
      Ops.push_back(Fn->isDeclaration() ? 1 : 0);
      Ops.push_back(encodeLinkage(Fn->getLinkage()));
      // paramattr: 1 for MMA load (nocapture+readonly), 0 otherwise
      bool isMMALoadFn = Fn->getName().starts_with("air.simdgroup_matrix_8x8_load");
      Ops.push_back(isMMALoadFn && hasMMALoad ? 1 : 0);
      Ops.push_back(0); // align
      for (int i = 0; i < 10; i++) Ops.push_back(0);
      Ops.push_back(Fn->getAddressSpace());
      W.EmitRecord(bitc::MODULE_CODE_FUNCTION, Ops);
    }
  }

  emitConstantsBlock(W, E, E.moduleConstants, 5);
  emitMetadataKindBlock(W);
  emitMetadataBlock(W, M, E);
  emitOperandBundleTagsBlock(W);
  emitSinglethreadBlock(W);

  for (auto *V : E.globalValues)
    if (auto *F = dyn_cast<Function>(V))
      if (!F->isDeclaration())
        emitFunctionBlock(W, *F, E);

  // VALUE_SYMTAB
  W.EnterSubblock(bitc::VALUE_SYMTAB_BLOCK_ID, 4);
  for (unsigned i = 0; i < E.globalValues.size(); i++) {
    if (!E.globalValues[i]->hasName()) continue;
    SmallVector<uint64_t, 32> NV;
    NV.push_back(i);
    for (char C : E.globalValues[i]->getName())
      NV.push_back((uint64_t)(unsigned char)C);
    W.EmitRecord(bitc::VST_CODE_ENTRY, NV);
  }
  W.ExitBlock();

  W.ExitBlock(); // MODULE_BLOCK

  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

} // namespace metalir
