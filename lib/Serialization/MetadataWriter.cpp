#include "metal-ir/ValueEnumerator.h"
#include "metal-ir/BitcodeEncoding.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace metalir {

// Metadata ID assignment: strings first, then values, then nodes.
struct MetadataEnumerator {
  std::vector<StringRef> strings;
  DenseMap<const MDString *, unsigned> stringMap;

  struct ValEntry { unsigned typeIdx; unsigned valueID; };
  std::vector<ValEntry> values;
  DenseMap<const ValueAsMetadata *, unsigned> valueMap;

  std::vector<const MDNode *> nodes;
  DenseMap<const MDNode *, unsigned> nodeMap;

  void collect(Module &M, ValueEnumerator &E) {
    for (auto &NMD : M.named_metadata())
      for (unsigned i = 0; i < NMD.getNumOperands(); i++)
        visitNode(NMD.getOperand(i), E);
  }

  void visitNode(const MDNode *N, ValueEnumerator &E) {
    if (nodeMap.count(N)) return;
    for (unsigned i = 0; i < N->getNumOperands(); i++) {
      auto *Op = N->getOperand(i).get();
      if (!Op) continue;
      if (auto *S = dyn_cast<MDString>(Op)) {
        if (!stringMap.count(S)) {
          stringMap[S] = strings.size();
          strings.push_back(S->getString());
        }
      } else if (auto *VAM = dyn_cast<ValueAsMetadata>(Op)) {
        if (!valueMap.count(VAM)) {
          ValEntry ve;
          ve.typeIdx = E.typeIdx(VAM->getValue()->getType());
          if (E.globalValueMap.count(VAM->getValue()))
            ve.valueID = E.globalIdx(VAM->getValue());
          else if (auto *C = dyn_cast<Constant>(VAM->getValue()))
            ve.valueID = E.moduleConstIdx(C);
          else
            ve.valueID = 0;
          valueMap[VAM] = values.size();
          values.push_back(ve);
        }
      } else if (auto *Sub = dyn_cast<MDNode>(Op)) {
        visitNode(Sub, E);
      }
    }
    nodeMap[N] = nodes.size();
    nodes.push_back(N);
  }

  uint64_t operandID(const Metadata *Op) const {
    if (!Op) return 0;
    if (auto *S = dyn_cast<MDString>(Op)) return stringMap.lookup(S) + 1;
    if (auto *VAM = dyn_cast<ValueAsMetadata>(Op))
      return strings.size() + valueMap.lookup(VAM) + 1;
    if (auto *N = dyn_cast<MDNode>(Op))
      return strings.size() + values.size() + nodeMap.lookup(N) + 1;
    return 0;
  }
};

void emitMetadataKindBlock(BitstreamWriter &W) {
  W.EnterSubblock(bitc::METADATA_KIND_BLOCK_ID, 3);
  static const char *kinds[] = {
    "dbg","tbaa","prof","fpmath","range","tbaa.struct","invariant.load",
    "alias.scope","noalias","nontemporal","llvm.mem.parallel_loop_access",
    "nonnull","dereferenceable","dereferenceable_or_null","make.implicit",
    "unpredictable","invariant.group","align","llvm.loop","type",
    "section_prefix","absolute_symbol","associated","callees","irr_loop",
    "llvm.access.group","callback","llvm.preserve.access.index",
    "vcall_visibility","noundef","annotation",
    "heapallocsite","air.function_groups",
  };
  for (unsigned i = 0; i < sizeof(kinds)/sizeof(kinds[0]); i++) {
    SmallVector<uint64_t, 32> V;
    V.push_back(i);
    for (const char *p = kinds[i]; *p; p++) V.push_back((unsigned char)*p);
    W.EmitRecord(bitc::METADATA_KIND, V);
  }
  W.ExitBlock();
}

void emitMetadataBlock(BitstreamWriter &W, Module &M, ValueEnumerator &E) {
  MetadataEnumerator MD;
  MD.collect(M, E);
  if (MD.strings.empty() && MD.values.empty() && MD.nodes.empty()) return;

  W.EnterSubblock(bitc::METADATA_BLOCK_ID, 4);

  for (auto &S : MD.strings)
    emitString(W, bitc::METADATA_STRING_OLD, S);

  for (auto &V : MD.values) {
    SmallVector<uint64_t, 2> Ops = {V.typeIdx, V.valueID};
    W.EmitRecord(bitc::METADATA_VALUE, Ops);
  }

  for (auto *N : MD.nodes) {
    SmallVector<uint64_t, 8> Ops;
    for (unsigned i = 0; i < N->getNumOperands(); i++)
      Ops.push_back(MD.operandID(N->getOperand(i).get()));
    W.EmitRecord(N->isDistinct() ? bitc::METADATA_DISTINCT_NODE
                                  : bitc::METADATA_NODE, Ops);
  }

  for (auto &NMD : M.named_metadata()) {
    emitString(W, bitc::METADATA_NAME, NMD.getName());
    SmallVector<uint64_t, 4> Ops;
    for (unsigned i = 0; i < NMD.getNumOperands(); i++)
      Ops.push_back(MD.strings.size() + MD.values.size() +
                     MD.nodeMap.lookup(NMD.getOperand(i)));
    W.EmitRecord(bitc::METADATA_NAMED_NODE, Ops);
  }

  W.ExitBlock();
}

void emitOperandBundleTagsBlock(BitstreamWriter &W) {
  W.EnterSubblock(bitc::OPERAND_BUNDLE_TAGS_BLOCK_ID, 3);
  static const char *tags[] = {
    "deopt","funclet","gc-transition","cfguardtarget",
    "preallocated","gc-live","clang.arc.attachedcall","ptrauth",
  };
  for (auto *t : tags)
    emitString(W, bitc::OPERAND_BUNDLE_TAG, t);
  W.ExitBlock();
}

void emitSinglethreadBlock(BitstreamWriter &W) {
  W.EnterSubblock(26, 2);
  emitString(W, 1, "singlethread");
  W.ExitBlock();
}

} // namespace metalir
