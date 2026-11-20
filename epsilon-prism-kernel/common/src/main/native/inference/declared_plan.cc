#include "execution_plan.h"
#include "../model_declaration.h"
#include "plan_checks.h"

#include <algorithm>

namespace prism {
namespace {
bool gated(schema::Composition role) {
  return role != schema::Composition::DISCARD_VALUE && role != schema::Composition::ZERO;
}

Composition packComposition(schema::Composition role) {
  switch (role) {
    case schema::Composition::DISCARD_VALUE:
      return Composition::kDiscardValue;
    case schema::Composition::RIICHI_GATE:
      return Composition::kRiichiGate;
    case schema::Composition::RESPONSE_GATE:
      return Composition::kResponseGate;
    case schema::Composition::TURN_GATE:
      return Composition::kTurnGate;
    case schema::Composition::ZERO:
      return Composition::kZero;
  }
  throw std::invalid_argument("Unknown action composition");
}

void checkArithmetic(const schema::Model& model, bool folded, bool causal, bool segments) {
  using A = schema::Arithmetic;
  std::vector<A> order{A::EMBED_COLUMN_FMA, A::BF16_TILE_K_ASCENDING, A::PAIR_BIAS_01_23_45_6, A::RESIDUAL_ROUND_BF16,
                       folded ? A::FOLDED_QKV_UNROUNDED_RMS : A::RMS_XOR_1_2};
  if (causal) order.insert(order.end(), {A::PREFIX_FULL_SPLIT_BF16, A::PREFIX_EDGE_FP32_FMA});
  if (segments) order.insert(order.end(), {A::COUNT_TILE_MUL_ADD, A::SEGMENT_DECLARATION_ORDER});
  order.insert(order.end(), {A::CANDIDATE_KEY_ORDER, A::VALUE_ACTION_ORDER});
  require(model.arithmetic == order, "Unsupported accumulation or rounding order");
}
}

PlanShape declaredPlan(const uint8_t* data, size_t length, int abi, PlanShape shape, int countOffset,
                       int segmentsOffset) {
  auto model = std::make_shared<const schema::Model>(schema::decode(data, length, abi));
  const auto& m = *model;
  require(m.activation == schema::Element::BF16 && m.accumulation == schema::Element::FP32 &&
              m.score == schema::Element::FP32 && m.representativeOrder == schema::Order::LEGAL_ACTION,
          "Unsupported collection precision or action order");
  const auto* capability = widthCapability(m.hidden);
  require(capability && m.headWidth == 16 && int64_t(m.heads) * m.headWidth == m.hidden / 2,
          "No collection tile for the declared attention heads");
  require(m.layers.size() >= 2 && m.layers.size() <= kMaxDepth, "Unsupported backbone depth");
  require(m.roles.size() <= kMaxRoles && m.segments.size() <= kMaxSegmentSpans, "Too many roles or candidate segments");
  static_assert(std::size(schema::sections) == kSections);
  require(m.inputs.size() == kSections, "Missing collection input sections");
  for (int i = 0; i < kSections; ++i)
    require(m.inputs[i].name == schema::sections[i].name, "Unsupported collection input section order");
  using S = schema::Stage;
  bool reread = std::find(m.stages.begin(), m.stages.end(), S::EVENT_REREAD) != m.stages.end();
  bool segments = !m.segments.empty();
  bool causal = m.layers.front().mask == schema::Mask::PUBLIC_PREFIX;
  checkArithmetic(m, capability->folded, causal, segments);
  Architecture a{};
  a.hidden = m.hidden;
  a.depth = static_cast<int>(m.layers.size());
  a.epsilonBits = __builtin_bit_cast(int32_t, m.epsilon);
  a.scaleBits = __builtin_bit_cast(int32_t, m.attentionScale);
  a.tauBits = __builtin_bit_cast(int32_t, m.temperature);
  for (auto& layer : a.layer) layer.role = -1;
  require(m.layers.front().queries == schema::Queries::ALL_TOKENS &&
              m.layers.back().queries == schema::Queries::FIXED_TOKENS,
          "The backbone requires all-token layers followed by fixed-token layers");
  bool fixedSeen = false;
  for (int i = 0; i < a.depth; ++i) {
    const auto& layer = m.layers[i];
    require(layer.keys == schema::Queries::ALL_TOKENS && (!i || layer.mask == schema::Mask::ALL_KEYS) &&
                layer.normalization == schema::Normalization::PRE_ATTENTION_AND_PRE_FFN,
            "Unsupported backbone attention");
    bool fixed = layer.queries == schema::Queries::FIXED_TOKENS;
    bool typed = layer.ffn == schema::FeedForward::BY_TOKEN;
    require(fixed || (!fixedSeen && !typed), "Unsupported backbone query order or typed all-token FFN");
    require(layer.contextWidth % kFfnChunk == 0 && layer.tileWidth % kFfnChunk == 0,
            "Unsupported backbone FFN tile width");
    fixedSeen |= fixed;
    a.layer[i] = {ordinal(fixed ? LayerRole::kFixedTokens : LayerRole::kAllTokens),
                  typed,
                  {typed ? 0 : layer.contextWidth, typed ? layer.contextWidth : 0, typed ? layer.tileWidth : 0}};
  }
  a.candidateFfn = m.candidateWidth;
  a.stages = ordinal(reread ? CandidateStages::kReread : CandidateStages::kOne);
  a.gate = m.gateWidth;
  a.gateKinds = static_cast<int>(
      std::count_if(m.roles.begin(), m.roles.end(), [](const auto& role) { return gated(role.composition); }));
  a.roles = static_cast<int>(m.roles.size());
  a.seats = m.seats;
  a.tiles = m.tiles;
  a.keyCodes = m.keyCodes;
  a.tileRelations = m.tileRelations;
  a.maxEvents = m.maxEvents;
  shape.architecture = a;
  auto& options = shape.options;
  options = {};
  options.features = causal ? PlanOptions::kCausal : 0;
  options.countOffset = countOffset;
  options.segmentsOffset = segmentsOffset;
  if (segments) {
    options.features |= PlanOptions::kCount | PlanOptions::kSegments;
    options.countWidth = m.countWidth;
    options.countFirstCode = m.tileKeyFirstCode;
    options.segments.codeWidth = m.codeWidth;
    options.segments.tableRows = m.tableRows;
    for (const auto& span : m.segments)
      options.segments.spans[options.segments.count++] = {span.column, span.firstRow, span.rows};
  }
  shape.declaration = std::move(model);
  return checkedPlanShape(shape);
}

void validateDeclaredWeights(const schema::Model& model, const PackHeader& header) {
  for (int i = 0; i < kSections; ++i)
    require(header.sectionWidth[i] == model.inputs[i].width, "Packed input width differs from the declaration");
  int gate = 0;
  for (size_t i = 0; i < model.roles.size(); ++i) {
    const auto role = model.roles[i].composition;
    require(header.composition[i] == ordinal(packComposition(role)) && header.gate[i] == (gated(role) ? gate++ : -1),
            "Packed action composition differs from the declaration");
  }
}
}
