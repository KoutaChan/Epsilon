#pragma once

#include "../../main/native/model_declaration.h"
#include "execution_plan.h"
#include "input_dimensions.h"
#include "packed_weights.h"
#include "relation_layout.h"

#include <cassert>
#include <cstring>
#include <vector>

namespace prism::test {

// 行の列: 最初のイベント、イベントの数、最初の遷移、遷移の数。
inline constexpr int kRowWidth = 6;
inline constexpr int kFirstTransitionColumn = 2;
inline constexpr int kTransitionsColumn = 3;

inline size_t aligned(size_t n) { return (n + kPackAlignment - 1) / kPackAlignment * kPackAlignment; }

inline schema::Model collectionModel(WidthCapability width, RuleCapability rule, bool causal, bool reread,
                                     bool segmentAttention = false, int depth = 2) {
  schema::Model m{};
  m.seats = rule.seats;
  m.tiles = rule.tiles;
  m.maxEvents = kMaxRowTokens - rule.fixed();
  m.hidden = width.width;
  m.heads = m.hidden / 32;
  m.headWidth = 16;
  m.epsilon = 0.00001f;
  m.attentionScale = 0.25f;
  m.temperature = 1;
  m.gateWidth = kGateStep;
  m.candidateWidth = kFfnChunk;
  m.keyCodes = 3;
  m.tileKeyFirstCode = 1;
  m.tileRelations = 1;
  m.activation = schema::Element::BF16;
  m.accumulation = m.score = schema::Element::FP32;
  m.representativeOrder = schema::Order::LEGAL_ACTION;
  m.codeWidth = rule.fixed();
  using S = schema::Stage;
  if (!width.standard || segmentAttention) {
    m.countWidth = kCountWidth;
    m.segments.push_back({"slot", m.codeWidth, 0, 2});
    m.codeWidth += rule.tiles;
    m.tableRows = 2;
    m.stages = {S::COUNT_TILES, S::SEGMENT_BIAS};
  }
  m.stages.insert(m.stages.end(), {S::CANDIDATE_ATTENTION, S::CANDIDATE_FFN});
  if (reread) m.stages.push_back(S::EVENT_REREAD);
  m.stages.insert(m.stages.end(), {S::DISCARD_VALUE, S::COMPOSE_VALUES});
  for (const auto& s : schema::sections) m.inputs.push_back({s.name, s.axis, s.element, 1, {}});
  for (auto section : {Section::kRoundCategory, Section::kSeatCategory, Section::kTileCategory, Section::kEventCategory,
                       Section::kTransitionCategory})
    m.inputs[int(section)].vocabularies = {1};
  m.inputs[int(Section::kRelation)].width = kMaxAttributes;
  m.inputs[int(Section::kTransitionCode)].width = m.codeWidth;
  m.inputs[int(Section::kRowColumn)].width = 6;
  m.inputs[int(Section::kActionColumn)].width = 7;
  m.layers = {{schema::Queries::ALL_TOKENS, schema::Queries::ALL_TOKENS,
               causal ? schema::Mask::PUBLIC_PREFIX : schema::Mask::ALL_KEYS, schema::Source::EMBEDDING,
               schema::Normalization::PRE_ATTENTION_AND_PRE_FFN, schema::FeedForward::SHARED, kFfnChunk, kFfnChunk},
              {schema::Queries::FIXED_TOKENS, schema::Queries::ALL_TOKENS, schema::Mask::ALL_KEYS,
               schema::Source::PREVIOUS_LAYER, schema::Normalization::PRE_ATTENTION_AND_PRE_FFN,
               schema::FeedForward::BY_TOKEN, 2 * kFfnChunk, kFfnChunk}};
  if (depth == 3) m.layers.push_back(m.layers.back());
  m.roles = {{"discard", schema::Composition::DISCARD_VALUE}, {"gate", schema::Composition::TURN_GATE}};
  using A = schema::Arithmetic;
  m.arithmetic = {A::EMBED_COLUMN_FMA, A::BF16_TILE_K_ASCENDING, A::PAIR_BIAS_01_23_45_6, A::RESIDUAL_ROUND_BF16,
                  width.folded ? A::FOLDED_QKV_UNROUNDED_RMS : A::RMS_XOR_1_2};
  if (causal) m.arithmetic.insert(m.arithmetic.end(), {A::PREFIX_FULL_SPLIT_BF16, A::PREFIX_EDGE_FP32_FMA});
  if (m.countWidth) m.arithmetic.insert(m.arithmetic.end(), {A::COUNT_TILE_MUL_ADD, A::SEGMENT_DECLARATION_ORDER});
  m.arithmetic.insert(m.arithmetic.end(), {A::CANDIDATE_KEY_ORDER, A::VALUE_ACTION_ORDER});
  m.weights = {{"parameter", schema::Element::FP32, {1}}};
  return m;
}

inline std::vector<uint8_t> weights(const PlanShape& shape) {
  const auto& model = *shape.declaration;
  PackHeader h{};
  h.architecture = shape.architecture;
  for (int i = 0; i < kSections; ++i) h.sectionWidth[i] = model.inputs[i].width;
  h.schema[int(Schema::kRowEvents)] = 1;
  h.schema[int(Schema::kRowFirstTransition)] = kFirstTransitionColumn;
  h.schema[int(Schema::kRowTransitions)] = kTransitionsColumn;
  h.schema[int(Schema::kRelationRows)] = kRelationSegments * kRelationEntries;
  std::fill(std::begin(h.composition), std::end(h.composition), int(Composition::kZero));
  std::fill(std::begin(h.gate), std::end(h.gate), -1);
  h.composition[0] = int(Composition::kDiscardValue);
  h.composition[1] = int(Composition::kTurnGate);
  h.gate[1] = 0;
  size_t cursor = aligned(sizeof(h));
  auto reserve = [&](size_t n) {
    int offset = static_cast<int>(cursor);
    cursor += aligned(n);
    return offset;
  };
  for (int w = 0; w < kWeightSlots; ++w) {
    auto slot = static_cast<Weight>(w);
    for (int i = 0; i < indexEntries(weightIndex(slot)); ++i)
      h.weight[weightFirstEntry(slot) + i] =
          entryHeld(h.architecture, slot, i) ? reserve(weightElements(slot, i, h) * weightElementBytes(slot)) : -1;
  }
  Section categories[] = {Section::kRoundCategory, Section::kSeatCategory, Section::kTileCategory,
                          Section::kEventCategory, Section::kTransitionCategory};
  bool folded = widthCapability(shape.hidden())->folded;
  for (int i = 0; i < kEmbeddingSlots; ++i) {
    auto& e = h.embedding[i];
    e.category = int(categories[i]);
    e.scalar = e.exact = -1;
    e.tables = 1;
    auto& table = e.table[0];
    table.rows = table.members = 1;
    table.stride[0] = 1;
    table.offset = reserve(model.hidden * 4);
    e.biasOffset = reserve(model.hidden * 4);
    bool fold = folded && i != int(EmbeddingSlot::kTransition);
    table.foldOffset = fold ? reserve(3 * shape.attention() * 4) : -1;
    e.foldOffset = fold ? reserve(2 * 3 * shape.attention() * 4) : -1;
  }
  assert(cursor <= (shape.options.countOffset ? size_t(shape.options.countOffset) : shape.weightBytes));
  std::vector<uint8_t> data(shape.options.countOffset ? shape.weightBytes : cursor);
  std::memcpy(data.data(), &h, sizeof(h));
  return data;
}

// rows 行のバッチ。行 i はイベントを events[i] 個（足りない分は 0 個）、遷移を 1 個持つ。入力は 0 で埋める。
struct RowBatch {
  BatchShape batch{};
  std::vector<uint8_t> input;

  int32_t* row(int i) {
    return reinterpret_cast<int32_t*>(input.data() + batch.section[int(Section::kRowColumn)]) + i * kRowWidth;
  }
};

inline RowBatch rowBatch(const PlanShape& shape, int rows, const std::vector<int>& events) {
  RowBatch b;
  b.batch.rows = b.batch.actions = b.batch.transitions = rows;
  for (int i = 0; i < rows && i < int(events.size()); ++i) b.batch.events += events[i];
  for (int i = 0; i < kSections; ++i) {
    b.batch.section[i] = b.batch.bytes;
    auto section = static_cast<Section>(i);
    b.batch.bytes +=
        aligned(axisCount(sectionAxis(section), b.batch, shape.architecture.seats, shape.architecture.tiles) *
                shape.declaration->inputs[i].width * sectionElementBytes(section));
  }
  b.input.resize(b.batch.bytes);
  for (int i = 0, first = 0; i < rows; ++i) {
    const int count = i < int(events.size()) ? events[i] : 0;
    b.row(i)[0] = first;
    b.row(i)[1] = count;
    b.row(i)[kFirstTransitionColumn] = i;
    b.row(i)[kTransitionsColumn] = 1;
    first += count;
  }
  return b;
}

inline PlanShape compileModel(const schema::Model& m) {
  auto bytes = schema::encode(m);
  PlanShape capacity{};
  capacity.rows = capacity.transitions = capacity.actions = 1024;
  capacity.events = 512;
  capacity.inputBytes = 4 << 20;
  capacity.weightBytes = 20 << 20;
  return declaredPlan(bytes.data(), bytes.size(), schema::version, capacity, m.countWidth ? 18 << 20 : 0,
                      m.countWidth ? 19 << 20 : 0);
}
}
