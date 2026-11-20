#include "packed_weights.h"
#include "input_dimensions.h"
#include "plan_checks.h"
#include "relation_layout.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace prism {

constexpr int kRelationRowsExpected = kRelationSegments * kRelationEntries;
constexpr int kMaxExact = 16;             // 価値合成の共有メモリに置く、ゲートに直接入る特徴 x^G の数の上限
constexpr int kTileRelationClasses = 30;  // セグメント 1・2 の 4 bit に入る、牌の関係のクラスの上限

// FFN の重みのスロットの項目の幅。
int ffnWidth(const Architecture& a, int entry) {
  return entry == kCandidateFfnEntry ? a.candidateFfn : a.layer[entry / kTokenGroups].ffn[entry % kTokenGroups];
}

// 重みのスロットの項目がこの形にあるか。組ごとの項目は、その層の FFN の組の幅が 0 でないときだけある。
bool entryHeld(const Architecture& a, Weight weight, int entry) {
  switch (weightIndex(weight)) {
    case WeightIndex::kSingle:
      return entry == 0 && (!rereadWeight(weight) || a.reread());
    case WeightIndex::kLayer:
      return entry >= 0 && entry < a.depth;
    case WeightIndex::kFfn:
      return entry == kCandidateFfnEntry || (entry >= 0 && entry / kTokenGroups < a.depth &&
                                             a.layer[entry / kTokenGroups].ffn[entry % kTokenGroups] > 0);
    case WeightIndex::kGroup:
      return entry >= 0 && entry < kTokenGroups && a.layer[a.depth - 1].ffn[entry] > 0;
  }
  return false;
}

// 重みのスロットの項目の要素の数。
size_t weightElements(Weight weight, int entry, const PackHeader& header) {
  const Architecture& a = header.architecture;
  const size_t h = a.hidden;
  const size_t att = a.attention();
  const size_t g = a.gate;
  const size_t kinds = static_cast<size_t>(a.gateKinds);
  const size_t actionExact = static_cast<size_t>(header.width(Section::kActionExact));
  switch (weight) {
    case Weight::kTileRelation:
      return static_cast<size_t>(a.tiles) * a.tiles;
    case Weight::kAttentionNorm:
    case Weight::kOutputBias:
    case Weight::kFfnNorm:
    case Weight::kDownBias:
    case Weight::kFinalNorm:
    case Weight::kQueryNorm:
    case Weight::kCandidateOutputBias:
    case Weight::kRereadQueryNorm:
    case Weight::kRereadEventNorm:
    case Weight::kRereadOutputBias:
    case Weight::kDiscardNorm:
    case Weight::kGateNorm:
      return h;
    case Weight::kQkv:
      return 3 * att * h;
    case Weight::kQkvBias:
      return 3 * att;
    case Weight::kRelationBias:
      return static_cast<size_t>(header.value(Schema::kRelationRows)) * a.heads();
    case Weight::kOutput:
    case Weight::kCandidateOutput:
    case Weight::kRereadOutput:
      return h * att;
    case Weight::kUp:
    case Weight::kDown:
      return static_cast<size_t>(ffnWidth(a, entry)) * h;
    case Weight::kUpBias:
      return static_cast<size_t>(ffnWidth(a, entry));
    case Weight::kTileEntry:
      return h * h;
    case Weight::kQuery:
    case Weight::kRereadQuery:
      return att * h;
    case Weight::kQueryBias:
    case Weight::kRereadQueryBias:
      return att;
    case Weight::kKeyValue:
    case Weight::kRereadKeyValue:
      return 2 * att * h;
    case Weight::kKeyValueBias:
    case Weight::kRereadKeyValueBias:
      return 2 * att;
    case Weight::kCodeKey:
    case Weight::kCodeValue:
      return static_cast<size_t>(a.keyCodes) * att;
    case Weight::kRereadCodeKey:
    case Weight::kRereadCodeValue:
      return static_cast<size_t>(a.rereadCodes()) * att;
    case Weight::kDiscardHidden:
      return g * h;
    case Weight::kDiscardHiddenBias:
    case Weight::kDiscardOutput:
    case Weight::kGateHiddenBias:
      return g;
    case Weight::kDiscardExact:
      return static_cast<size_t>(header.width(Section::kDiscardExact));
    case Weight::kGateHidden:
      return g * 3 * h;
    case Weight::kGateKindOffset:
    case Weight::kGateOutput:
      return kinds * g;
    case Weight::kGateExactProjection:
      return kinds * g * actionExact;
    case Weight::kGateOutputBias:
      return kinds;
    case Weight::kGateExactOutput:
      return kinds * actionExact;
    case Weight::kCount:
      break;
  }
  throw std::invalid_argument("unknown weight slot");
}

std::string weightName(Weight weight, int entry) {
  return "weight slot " + std::to_string(ordinal(weight)) + " entry " + std::to_string(entry);
}

// BF16・FP32 のビットが有限の値でないか（指数が全て 1 の NaN と ±∞）。有限の境界値と符号つきの 0 は有限。
bool isNonFinite(uint16_t bits) { return (bits & 0x7f80u) == 0x7f80u; }

bool isNonFinite(uint32_t bits) { return (bits & 0x7f800000u) == 0x7f800000u; }

// pack の offset バイトから始まる count 個の要素（Bits は BF16 なら uint16_t、FP32 なら uint32_t）に有限でない値があるか。
template <class Bits>
bool containsNonFinite(const uint8_t* pack, size_t offset, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    Bits bits;
    std::memcpy(&bits, pack + offset + i * sizeof(Bits), sizeof(Bits));
    if (isNonFinite(bits)) {
      return true;
    }
  }
  return false;
}

// WeightPack の BF16・FP32 の部分（重みのスロットの全ての項目と、埋め込みの表・W_d・b）に NaN と ±∞ がないことを確かめる（ヘッダーを
// 照合した後に呼ぶ）。有限でない重みは、それを使う判断のスコアを黙って NaN にするので、公開のときに名前を付けて拒む。これでネットワークの中の
// NaN は、演算が作るものだけになる（活性の丸め bf16RoundActivation は NaN を特別に扱わない。届く NaN についての契約は inference_device.h）。
void requireFinite(const PackHeader& header, const PlanShape& shape, const uint8_t* pack) {
  for (int w = 0; w < kWeightSlots; ++w) {
    const Weight weight = static_cast<Weight>(w);
    const int bytes = weightElementBytes(weight);
    for (int e = 0; e < indexEntries(weightIndex(weight)); ++e) {
      const int32_t offset = header.offset(weight, e);
      if (offset < 0 || bytes == 1) {  // 1 バイトの要素は牌の関係のクラス（整数）
        continue;
      }
      const size_t count = weightElements(weight, e, header);
      const bool nonFinite = bytes == 2 ? containsNonFinite<uint16_t>(pack, static_cast<size_t>(offset), count)
                                        : containsNonFinite<uint32_t>(pack, static_cast<size_t>(offset), count);
      require(!nonFinite, weightName(weight, e) + " contains a non-finite value");
    }
  }
  for (int e = 0; e < kEmbeddingSlots; ++e) {
    const EmbeddingSpec& spec = header.embedding[e];
    if (spec.tables == 0) {
      continue;
    }
    const std::string name = "embedding slot " + std::to_string(e);
    for (int t = 0; t < spec.tables; ++t) {
      const EmbeddingTable& table = spec.table[t];
      require(!containsNonFinite<uint32_t>(pack, static_cast<size_t>(table.offset),
                                           static_cast<size_t>(table.rows) * shape.hidden()),
              name + " table " + std::to_string(t) + " contains a non-finite value");
    }
    require(!containsNonFinite<uint16_t>(pack, static_cast<size_t>(spec.denseOffset),
                                         static_cast<size_t>(spec.dense) * shape.hidden()),
            name + " dense weights contain a non-finite value");
    require(!containsNonFinite<uint32_t>(pack, static_cast<size_t>(spec.biasOffset), shape.hidden()),
            name + " bias contains a non-finite value");
    if (spec.foldOffset < 0) {
      continue;
    }
    const size_t width = static_cast<size_t>(3) * shape.attention();
    for (int t = 0; t < spec.tables; ++t) {
      const EmbeddingTable& table = spec.table[t];
      require(!containsNonFinite<uint32_t>(pack, static_cast<size_t>(table.foldOffset),
                                           static_cast<size_t>(table.rows) * width),
              name + " folded table " + std::to_string(t) + " contains a non-finite value");
    }
    require(!containsNonFinite<uint32_t>(pack, static_cast<size_t>(spec.foldOffset),
                                         static_cast<size_t>(spec.dense + 2) * width),
            name + " folded dense weights or biases contain a non-finite value");
  }
}

PackedWeights::PackedWeights(const PlanShape& shape) {
  const Architecture& a = shape.architecture;
  const int h = a.hidden;
  const int att = a.attention();
  index.fill(-1);
  size_t offset = 0;
  const auto add = [&](Weight weight, int entry, int firstRow, int rows, int columns) {
    index[weightFirstEntry(weight) + entry] = static_cast<int>(fragments.size());
    fragments.push_back(FragmentCopy{weight, entry, firstRow, rows, columns, offset});
    offset += static_cast<size_t>(rows) * columns;
  };
  for (int l = 0; l < a.depth; ++l) {
    add(Weight::kOutput, l, 0, h, att);
  }
  std::vector<int> ffnEntries;
  for (int l = 0; l < a.depth; ++l) {
    for (int g = 0; g < kTokenGroups; ++g) {
      if (a.layer[l].ffn[g] > 0) {
        ffnEntries.push_back(ffnEntry(l, static_cast<TokenGroup>(g)));
      }
    }
  }
  ffnEntries.push_back(kCandidateFfnEntry);
  for (int entry : ffnEntries) {
    add(Weight::kUp, entry, 0, ffnWidth(a, entry), h);
  }
  for (int entry : ffnEntries) {
    add(Weight::kDown, entry, 0, h, ffnWidth(a, entry));
  }
  add(Weight::kKeyValue, 0, 0, 2 * att, h);
  add(Weight::kCandidateOutput, 0, 0, h, att);
  // 2 つ目からの局・席・牌の層が使うイベントのキーとバリュー（作るのは最後の全トークンの層の前段 FFN）: W_qkv の k・v の行 [A, 3A)。
  int firstKv = a.allTokenLayers() + (shape.options.has(PlanOptions::kCausal) ? 0 : 1);
  for (int l = firstKv; l < a.depth; ++l) {
    add(Weight::kQkv, l, att, 2 * att, h);
  }
  if (a.reread()) {
    add(Weight::kRereadKeyValue, 0, 0, 2 * att, h);
    add(Weight::kRereadOutput, 0, 0, h, att);
  }
  fragmentElements = offset;
}

static void validateHeader(const PlanShape& shape, const PackHeader& header, bool folded, size_t bytes) {
  const Architecture& a = shape.architecture;
  require(std::memcmp(&header.architecture, &a, sizeof(Architecture)) == 0,
          "the weight pack was built for another network architecture than the plan");
  if (shape.declaration) validateDeclaredWeights(*shape.declaration, header);
  for (int s = 0; s < kSections; ++s) {
    require(header.sectionWidth[s] > 0, "input section " + std::to_string(s) + " has no width");
  }
  const int rowWidth = header.width(Section::kRowColumn);
  for (Schema field : {Schema::kRowFirstEvent, Schema::kRowEvents, Schema::kRowFirstAction, Schema::kRowActions,
                       Schema::kRowFirstTransition, Schema::kRowTransitions}) {
    require(header.value(field) >= 0 && header.value(field) < rowWidth, "row column out of range");
  }
  const int actionWidth = header.width(Section::kActionColumn);
  for (Schema field : {Schema::kActionRow, Schema::kActionFirstTransition, Schema::kActionTransitions,
                       Schema::kActionRole, Schema::kActionBase, Schema::kActionRepresentative}) {
    require(header.value(field) >= 0 && header.value(field) < actionWidth, "action column out of range");
  }
  const int relationWidth = header.width(Section::kRelation);
  require(relationWidth <= kMaxAttributes, "relation attributes exceed " + std::to_string(kMaxAttributes) + " columns");
  for (Schema field : {Schema::kRelationTile, Schema::kRelationOwner, Schema::kRelationEvent, Schema::kRelationRiichi,
                       Schema::kRelationLastDiscard, Schema::kRelationLink}) {
    require(header.value(field) >= 0 && header.value(field) < relationWidth, "relation column out of range");
  }
  require(header.value(Schema::kRelationSafety) >= 0 &&
              header.value(Schema::kRelationSafety) <= relationWidth - a.opponents(),
          "relation safety columns of the " + std::to_string(a.opponents()) + " opponents are out of range");
  const int transitionWidth = header.width(Section::kTransitionCategory);
  require(header.value(Schema::kTransitionCallDiscard) >= 0 &&
              header.value(Schema::kTransitionCallDiscard) < transitionWidth &&
              header.value(Schema::kTransitionMainTile) >= 0 &&
              header.value(Schema::kTransitionMainTile) < transitionWidth,
          "transition columns out of range");
  const int codeWidth = shape.options.has(PlanOptions::kSegments) ? shape.options.segments.codeWidth : a.fixed();
  require(header.width(Section::kTransitionCode) == codeWidth, "transition code width does not match the plan");
  require(header.value(Schema::kRelationRows) == kRelationRowsExpected,
          "the relation bias table must have " + std::to_string(kRelationRowsExpected) + " rows");
  require(header.width(Section::kActionExact) <= kMaxExact, "gate path features exceed " + std::to_string(kMaxExact));
  for (int r = 0; r < kMaxRoles; ++r) {
    require(header.composition[r] >= 0 && header.composition[r] < ordinal(Composition::kCount),
            "role composition out of range");
    require(header.gate[r] >= -1 && header.gate[r] < a.gateKinds, "role gate kind out of range");
    require(r < a.roles || (header.composition[r] == ordinal(Composition::kZero) && header.gate[r] == -1),
            "role " + std::to_string(r) + " beyond the " + std::to_string(a.roles) + " roles must be empty");
  }
  for (int w = 0; w < kWeightSlots; ++w) {
    const Weight weight = static_cast<Weight>(w);
    for (int e = 0; e < indexEntries(weightIndex(weight)); ++e) {
      const int32_t offset = header.offset(weight, e);
      if (!entryHeld(a, weight, e)) {
        require(offset == -1, weightName(weight, e) + " is not part of the architecture");
        continue;
      }
      if (offset < 0) {
        require(!shape.declaration, weightName(weight, e) + " is missing from the collection pack");
        continue;
      }
      const size_t size = weightElements(weight, e, header) * weightElementBytes(weight);
      require(static_cast<size_t>(offset) % kPackAlignment == 0 && static_cast<size_t>(offset) >= sizeof(PackHeader) &&
                  static_cast<size_t>(offset) + size <= bytes,
              weightName(weight, e) + " [" + std::to_string(offset) + ", +" + std::to_string(size) +
                  ") is outside the weight pack of " + std::to_string(bytes) + " bytes");
    }
  }
  for (int e = 0; e < kEmbeddingSlots; ++e) {
    const EmbeddingSpec& spec = header.embedding[e];
    const std::string name = "embedding slot " + std::to_string(e);
    if (spec.tables == 0) {
      require(!shape.declaration, name + " is missing from the collection pack");
      require(spec.dense == 0, name + " has dense features but no tables");
      continue;
    }
    const Axis axis = slotAxis(static_cast<EmbeddingSlot>(e));
    require(spec.category >= 0 && spec.category < kSections &&
                sectionElementBytes(static_cast<Section>(spec.category)) == 1 &&
                sectionAxis(static_cast<Section>(spec.category)) == axis,
            name + " category section is invalid");
    require(spec.scalar == -1 || (spec.scalar >= 0 && spec.scalar < kSections &&
                                  sectionElementBytes(static_cast<Section>(spec.scalar)) == 2 &&
                                  sectionAxis(static_cast<Section>(spec.scalar)) == axis),
            name + " scalar section is invalid");
    require(spec.exact == -1 || (spec.exact >= 0 && spec.exact < kSections &&
                                 static_cast<Section>(spec.exact) == Section::kDiscardExact &&
                                 sectionAxis(static_cast<Section>(spec.exact)) == axis),
            name + " exact section is invalid");
    require(spec.tables > 0 && spec.tables <= kMaxTables, name + " table count out of range");
    const int categoryWidth = header.sectionWidth[spec.category];
    const auto* vocabulary = shape.declaration ? &shape.declaration->inputs[spec.category].vocabularies : nullptr;
    if (vocabulary)
      require(vocabulary->size() == static_cast<size_t>(categoryWidth), name + " is missing its column vocabularies");

    // 局・席・牌・イベントは、層 0 の QKV を折り込んだ表を持つ。遷移は持たない。
    const bool folding = folded && static_cast<EmbeddingSlot>(e) != EmbeddingSlot::kTransition;
    const size_t foldRow = static_cast<size_t>(3) * shape.attention() * sizeof(float);
    const auto inside = [&](int32_t offset, size_t size) {
      return offset >= 0 && static_cast<size_t>(offset) % kVectorAlignment == 0 &&
             static_cast<size_t>(offset) + size <= bytes;
    };
    for (int t = 0; t < spec.tables; ++t) {
      const EmbeddingTable& table = spec.table[t];
      require(table.members > 0 && table.members <= kMaxMembers && table.rows > 0, name + " table shape is invalid");
      require(
          table.offset >= 0 && static_cast<size_t>(table.offset) % kVectorAlignment == 0 &&
              static_cast<size_t>(table.offset) + static_cast<size_t>(table.rows) * shape.hidden() * sizeof(float) <=
                  bytes,
          name + " table " + std::to_string(t) + " is outside the weight pack");
      require(
          folding ? inside(table.foldOffset, static_cast<size_t>(table.rows) * foldRow) : table.foldOffset == -1,
          name + " folded table " + std::to_string(t) + (folding ? " is outside the weight pack" : " must be absent"));
      int64_t rows = 1;
      for (int m = 0; m < table.members; ++m) {
        require(table.column[m] >= 0 && table.column[m] < categoryWidth && table.stride[m] >= 0,
                name + " table member out of range");
        if (vocabulary) {
          require(table.stride[m] == rows, name + " has an incompatible embedding stride");
          rows *= (*vocabulary)[table.column[m]];
          require(rows <= table.rows, name + " has an incompatible embedding vocabulary");
        }
      }
      if (vocabulary) require(rows == table.rows, name + " has an incompatible embedding table size");
    }
    require(spec.dense >= 0 && spec.dense <= kMaxDense, name + " dense count out of range");
    if (spec.dense > 0) {
      require(spec.denseOffset >= 0 && static_cast<size_t>(spec.denseOffset) % kVectorAlignment == 0 &&
                  static_cast<size_t>(spec.denseOffset) + static_cast<size_t>(spec.dense) * shape.hidden() * 2 <= bytes,
              name + " dense weights are outside the weight pack");
    }
    require(spec.biasOffset >= 0 && static_cast<size_t>(spec.biasOffset) % kVectorAlignment == 0 &&
                static_cast<size_t>(spec.biasOffset) + shape.hidden() * sizeof(float) <= bytes,
            name + " bias is outside the weight pack");
    require(folding ? inside(spec.foldOffset, static_cast<size_t>(spec.dense + 2) * foldRow) : spec.foldOffset == -1,
            name + (folding ? " folded dense weights and biases are outside the weight pack"
                            : " folded dense weights must be absent"));
    for (int k = 0; k < spec.dense; ++k) {
      const int source = spec.source[k] >> 16;
      const int index = spec.source[k] & 0xffff;
      if (source == ordinal(DenseSource::kFlag)) {
        require(index < categoryWidth, name + " flag column out of range");
      } else if (source == ordinal(DenseSource::kScalar)) {
        require(spec.scalar >= 0 && index < header.sectionWidth[spec.scalar], name + " scalar column out of range");
      } else if (source == ordinal(DenseSource::kExact)) {
        require(spec.exact >= 0 && index < header.sectionWidth[spec.exact], name + " exact column out of range");
      } else {
        require(false, name + " has an unknown dense source");
      }
    }
  }
}

PackHeader PackedWeights::inspect(const PlanShape& shape, const CandidatePackLayout& candidate, bool folded,
                                  const void* host, size_t bytes, int64_t version) const {
  require(host != nullptr, "missing weight pack");
  require(bytes <= shape.weightBytes, "weight pack of " + std::to_string(bytes) + " bytes exceeds the plan capacity " +
                                          std::to_string(shape.weightBytes));
  require(bytes >= sizeof(PackHeader), "weight pack is smaller than its header");
  require(version >= 0, "weights version must not be negative");
  PackHeader header;
  std::memcpy(&header, host, sizeof(PackHeader));
  const auto& options = shape.options;
  size_t baseBytes = bytes;
  if (options.has(PlanOptions::kCount)) {
    const size_t offset = options.countOffset;
    require(offset <= bytes && candidate.countBytes() <= bytes - offset, "count weights are missing");
    baseBytes = std::min(baseBytes, offset);
    const char* data = static_cast<const char*>(host) + offset;
    for (size_t i = 0; i < candidate.gate; ++i) {
      float value;
      std::memcpy(&value, data + i * sizeof(value), sizeof(value));
      require(std::isfinite(value), "non-finite count gate");
    }
    data += candidate.gate * sizeof(float);
    for (size_t i = 0; i < 2 * candidate.projection; ++i) {
      uint16_t value;
      std::memcpy(&value, data + i * sizeof(value), sizeof(value));
      require((value & 0x7f80u) != 0x7f80u, "non-finite count projection");
    }
  }
  if (options.has(PlanOptions::kSegments)) {
    const size_t offset = options.segmentsOffset;
    require(offset <= bytes && candidate.segmentsBytes() <= bytes - offset, "segment weights are missing");
    baseBytes = std::min(baseBytes, offset);
    const char* data = static_cast<const char*>(host) + offset;
    for (size_t i = 0; i < candidate.segments; ++i) {
      float value;
      std::memcpy(&value, data + i * sizeof(value), sizeof(value));
      require(std::isfinite(value), "non-finite segment weight");
    }
  }
  validateHeader(shape, header, folded, baseBytes);
  const int32_t tileRelation = header.offset(Weight::kTileRelation, 0);
  if (tileRelation >= 0) {
    const auto* classes = static_cast<const uint8_t*>(host) + tileRelation;
    const size_t count = weightElements(Weight::kTileRelation, 0, header);
    for (size_t i = 0; i < count; ++i) {
      require(classes[i] < kTileRelationClasses, "tile relation class out of range");
    }
  }
  requireFinite(header, shape, static_cast<const uint8_t*>(host));
  return header;
}

}  // namespace prism
