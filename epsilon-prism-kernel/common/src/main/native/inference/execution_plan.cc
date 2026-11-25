#include "execution_plan.h"
#include "../model_schema.h"
#include "plan_checks.h"
#include "relation_layout.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace prism {

constexpr int kMaxCodes = kU8Codes - 1;

static void validateDimensions(const Architecture& a) {
  const auto* capability = widthCapability(a.hidden);
  require(capability, "no compiled inference kernels for hidden width " + std::to_string(a.hidden));
  require(!a.reread() || capability->reread, "the inference width does not support reread");
  for (float value : {a.epsilon(), a.scale(), a.tau()}) {
    require(std::isfinite(value) && value > 0.0f, "epsilon, scale and tau must be positive and finite");
  }
  require(a.candidateFfn > 0 && a.candidateFfn % kFfnChunk == 0,
          "candidate ffn must be a positive multiple of " + std::to_string(kFfnChunk));
  require(a.stages == ordinal(CandidateStages::kOne) || a.stages == ordinal(CandidateStages::kReread),
          "unknown candidate stages " + std::to_string(a.stages));
  require(a.gate > 0 && a.gate % kGateStep == 0, "gate must be a positive multiple of " + std::to_string(kGateStep));
  require(a.gateKinds > 0 && a.gateKinds <= a.roles && a.roles <= kMaxRoles,
          "need 0 < gate kinds <= roles <= " + std::to_string(kMaxRoles));
  require(a.seats >= kMinSeats && a.seats <= kMaxSeats, "seats out of range: " + std::to_string(a.seats));
  require(a.tiles > 0 && a.tiles <= kMaxTiles, "tiles out of range: " + std::to_string(a.tiles));
  require(std::any_of(kRuleCapabilities.begin(), kRuleCapabilities.end(),
                      [&](auto rule) { return rule.seats == a.seats && rule.tiles == a.tiles; }),
          "no candidate attention tile for this seat and tile count");
  require(a.keyCodes > 0 && a.tileRelations > 0 && a.maxEvents >= 0, "counts must be positive");
  require(a.fixed() <= kMaxFixedTokens && a.maxEvents <= kMaxRowTokens - a.fixed(),
          "rows of " + std::to_string(a.fixed()) + " fixed and " + std::to_string(int64_t(a.fixed()) + a.maxEvents) +
              " tokens exceed " + std::to_string(kMaxFixedTokens) + " and " + std::to_string(kMaxRowTokens));
  require(int64_t(a.keyCodes) + 1 + a.tileRelations <= kMaxCodes,
          "code tables exceed " + std::to_string(kMaxCodes) + " rows");
  require(a.opponents() <= kRelationOpponents,
          "relation codes hold the safety of at most " + std::to_string(kRelationOpponents) + " opponents");
}

void validateArchitecture(const Architecture& a) {
  validateDimensions(a);
  require(a.depth >= 2 && a.depth <= kMaxDepth, "depth must be 2.." + std::to_string(kMaxDepth));
  const int allTokens = a.allTokenLayers();
  require(allTokens >= 1 && allTokens < a.depth, "the backbone needs all-token layers followed by fixed-token layers");
  for (int l = 0; l < kMaxDepth; ++l) {
    const LayerSpec& layer = a.layer[l];
    if (l >= a.depth) {
      require(layer.role == -1 && layer.typed == 0 && layer.ffn[0] == 0 && layer.ffn[1] == 0 && layer.ffn[2] == 0,
              "layer " + std::to_string(l) + " beyond the depth must be empty");
      continue;
    }
    const bool fixedLayer = l >= allTokens;
    require(layer.role == ordinal(fixedLayer ? LayerRole::kFixedTokens : LayerRole::kAllTokens),
            "layer " + std::to_string(l) + " has role " + std::to_string(layer.role) +
                "; all-token layers come first, then fixed-token layers");
    const bool typed = layer.typed != 0;
    require(layer.typed == 0 || layer.typed == 1, "layer " + std::to_string(l) + " typed flag must be 0 or 1");
    require(!typed || fixedLayer, "all-token layer " + std::to_string(l) + " must have a shared FFN");
    for (int g = 0; g < kTokenGroups; ++g) {
      const bool used = typed ? g != ordinal(TokenGroup::kAll) : g == ordinal(TokenGroup::kAll);
      require(used ? layer.ffn[g] > 0 && layer.ffn[g] % kFfnChunk == 0 : layer.ffn[g] == 0,
              "layer " + std::to_string(l) + " FFN group " + std::to_string(g) + " width " +
                  std::to_string(layer.ffn[g]) + " is invalid");
    }
  }
}

bool acceptsLayer(Kernel kernel, int layer, const Architecture& a) {
  const bool inside = layer >= 0 && layer < a.depth;
  switch (kernel) {
    case Kernel::kAttentionAll:
    case Kernel::kMlpAll:
      return inside && a.role(layer) == LayerRole::kAllTokens;
    case Kernel::kAttentionFixed:
      return inside && a.role(layer) == LayerRole::kFixedTokens;
    case Kernel::kMlpFixed:
      return inside && a.role(layer) == LayerRole::kFixedTokens && layer < a.depth - 1;
    case Kernel::kMlpLast:
      return layer == a.depth - 1;
    case Kernel::kRereadKeyValue:
    case Kernel::kReread:
      return layer == 0 && a.reread();
    case Kernel::kEmbed:
    case Kernel::kRelation:
    case Kernel::kCandidate:
    case Kernel::kValueGraph:
      return layer == 0;
    case Kernel::kCount:
      break;
  }
  return false;
}

PlanShape checkedPlanShape(const PlanShape& shape) {
  if (shape.declaration)
    validateDimensions(shape.architecture);
  else
    validateArchitecture(shape.architecture);
  require(shape.rows > 0 && shape.events >= 0 && shape.transitions >= shape.rows && shape.actions >= shape.rows,
          "invalid batch capacity");
  // Attention は行を起動のグリッドの y に置く。
  require(shape.rows <= kMaxGridRows, "a batch of " + std::to_string(shape.rows) + " rows exceeds the " +
                                          std::to_string(kMaxGridRows) + " rows of one attention launch");
  // 埋め込みは入力の領域を、32 bit のバイトオフセットで読む。
  require(shape.inputBytes > 0 && shape.inputBytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "input capacity must be positive and stay below the 2 GiB that embedding addresses with 32-bit offsets");
  require(shape.weightBytes >= sizeof(PackHeader) &&
              shape.weightBytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "weight pack capacity must hold the header and stay below 2 GiB");
  require(shape.tokens() <= static_cast<size_t>(std::numeric_limits<int>::max()) - 64 &&
              shape.transitions <= std::numeric_limits<int>::max() - 64,
          "batch capacity exceeds the signed kernel index range");
  return shape;
}

namespace {
// 単独演算を呼ぶ入口では、保存した形から各層のクエリ、マスク、射影元を復元する。
std::vector<schema::Layer> packedLayers(const PlanShape& shape) {
  const auto& a = shape.architecture;
  std::vector<schema::Layer> layers;
  for (int i = 0; i < a.depth; ++i) {
    const bool typed = a.typed(i);
    layers.push_back(
        {a.role(i) == LayerRole::kAllTokens ? schema::Queries::ALL_TOKENS : schema::Queries::FIXED_TOKENS,
         schema::Queries::ALL_TOKENS,
         i == 0 && shape.options.has(PlanOptions::kCausal) ? schema::Mask::PUBLIC_PREFIX : schema::Mask::ALL_KEYS,
         i == 0 ? schema::Source::EMBEDDING : schema::Source::PREVIOUS_LAYER,
         schema::Normalization::PRE_ATTENTION_AND_PRE_FFN,
         typed ? schema::FeedForward::BY_TOKEN : schema::FeedForward::SHARED,
         a.ffn(i, typed ? TokenGroup::kContext : TokenGroup::kAll),
         a.ffn(i, typed ? TokenGroup::kTile : TokenGroup::kAll)});
  }
  return layers;
}

CandidatePackLayout describeCandidatePack(const PlanShape& shape, const WidthCapability& capability) {
  const auto& a = shape.architecture;
  const auto& options = shape.options;
  const bool count = options.has(PlanOptions::kCount);
  const bool segments = options.has(PlanOptions::kSegments);
  constexpr int features = PlanOptions::kCausal | PlanOptions::kCount | PlanOptions::kSegments;
  require((options.features & ~features) == 0, "unknown plan feature");
  require(count == segments, "count and segments must be enabled together");
  require(count ? capability.core : capability.standard, "unsupported candidate mode for this inference width");
  require(!options.has(PlanOptions::kCausal) || !a.reread(),
          "causal attention requires the layer-zero all-token path without reread");

  CandidatePackLayout pack;
  const auto requireSpan = [&](int32_t offset, size_t bytes, const char* name) {
    const size_t start = static_cast<size_t>(offset);
    require(start >= sizeof(PackHeader) && start % kPackAlignment == 0 && start <= shape.weightBytes &&
                bytes <= shape.weightBytes - start,
            name);
  };
  if (count) {
    require(options.countFirstCode >= 0 && options.countFirstCode < a.keyCodes, "invalid counting codes");
    require(options.countWidth == kCountWidth, "count width does not match the kernel");
    pack.gate = static_cast<size_t>(a.keyCodes - options.countFirstCode) * kCountWidth;
    pack.projection = static_cast<size_t>(a.hidden) * kCountWidth;
    pack.countWorkspace =
        static_cast<size_t>(shape.rows) * a.tiles * kCountWidth + static_cast<size_t>(shape.transitions) * a.hidden;
    requireSpan(options.countOffset, pack.countBytes(), "count weights are outside the plan capacity");
  } else {
    require(options.countOffset == 0 && options.countFirstCode == 0 && options.countWidth == 0,
            "count data requires counting");
  }
  if (segments) {
    require(!a.reread() && options.segments.valid(a.fixed(), a.tiles), "invalid segment layout");
    pack.segments = static_cast<size_t>(options.segments.tableRows) * 2 * a.attention();
    requireSpan(options.segmentsOffset, pack.segmentsBytes(), "segment weights are outside the plan capacity");
  } else {
    require(options.segmentsOffset == 0 && options.segments.count == 0 && options.segments.codeWidth == 0 &&
                options.segments.tableRows == 0,
            "segment data requires segments");
  }
  if (count && segments) {
    require(
        static_cast<size_t>(options.countOffset) + pack.countBytes() <= static_cast<size_t>(options.segmentsOffset) ||
            static_cast<size_t>(options.segmentsOffset) + pack.segmentsBytes() <=
                static_cast<size_t>(options.countOffset),
        "candidate weight ranges overlap");
  }
  return pack;
}

}  // namespace

ExecutionPlan::ExecutionPlan(const PlanShape& shape, const WidthCapability& capability)
    : candidate(describeCandidatePack(shape, capability)) {
  using R = Region;
  const auto mask = [](std::initializer_list<R> regions) {
    uint32_t bits = 0;
    for (auto r : regions) bits |= regionBit(r);
    return bits;
  };
  const auto add = [&](Kernel kernel, int layer, bool compact, std::initializer_list<R> inputs,
                       std::initializer_list<R> outputs) {
    steps.push_back({kernel, layer, compact, mask(inputs), mask(outputs)});
  };
  const auto& a = shape.architecture;
  const auto standalone = shape.declaration ? std::vector<schema::Layer>{} : packedLayers(shape);
  const auto& layers = shape.declaration ? shape.declaration->layers : standalone;
  const int depth = static_cast<int>(layers.size());
  const int firstFixed =
      static_cast<int>(std::find_if(layers.begin(), layers.end(),
                                    [](const auto& layer) { return layer.queries == schema::Queries::FIXED_TOKENS; }) -
                       layers.begin());
  const int copiedLayers = depth - firstFixed - 1;
  const bool pages = layers.front().mask == schema::Mask::PUBLIC_PREFIX && firstFixed == 1;
  using Stage = schema::Stage;
  const bool reread = shape.declaration ? std::find(shape.declaration->stages.begin(), shape.declaration->stages.end(),
                                                    Stage::EVENT_REREAD) != shape.declaration->stages.end()
                                        : a.reread();
  add(Kernel::kEmbed, 0, capability.folded, {R::kInput}, {R::kResidual, R::kStatic, R::kQkv0});
  if (!capability.folded) steps.back().outputs |= regionBit(R::kNormalized);
  if (pages) steps.back().outputs |= regionBit(capability.folded ? R::kAttention : R::kState);
  add(Kernel::kRelation, 0, false, {R::kInput}, {R::kRelationCode});
  for (int l = 0; l < depth; ++l) {
    const auto& layer = layers[l];
    bool all = layer.queries == schema::Queries::ALL_TOKENS;
    bool embedding = layer.source == schema::Source::EMBEDDING;
    add(all ? Kernel::kAttentionAll : Kernel::kAttentionFixed, l, false, {R::kInput, R::kRelationCode},
        {R::kAttention});
    steps.back().causal = layer.mask == schema::Mask::PUBLIC_PREFIX;
    steps.back().copiedKeysIndex = all ? -1 : (l - firstFixed) - 1;
    steps.back().inputs |= regionBit(embedding && capability.folded ? R::kQkv0 : R::kNormalized);
    if (pages && embedding && !capability.folded) steps.back().outputs |= regionBit(R::kQkv0);
    if (!all && (l - firstFixed) > 0) steps.back().inputs |= regionBit(R::kEventKeyValue);
    if (pages && !all && (l - firstFixed) == 0) steps.back().inputs |= regionBit(R::kQkv0);
    if (l + 1 == depth) {
      add(Kernel::kMlpLast, l, false, {R::kAttention, R::kResidual}, {R::kState, R::kMemory});
    } else {
      add(all ? Kernel::kMlpAll : Kernel::kMlpFixed, l, all, {R::kAttention, R::kResidual},
          {R::kNormalized, R::kResidual});
      if (all && layers[l + 1].queries == schema::Queries::FIXED_TOKENS) {
        steps.back().lastAllTokenLayer = true;
        if (copiedLayers) steps.back().outputs |= regionBit(R::kEventKeyValue);
        if (reread) steps.back().outputs |= regionBit(R::kRereadMemory);
        if (pages) steps.back().outputs |= regionBit(R::kQkv0);
        if (pages && !capability.folded) steps.back().inputs |= regionBit(R::kState);
      }
    }
  }
  // 候補の Attention と FFN、打牌価値と合成は各一組で提出する。
  std::vector<Stage> packedStages;
  if (!shape.declaration) {
    if (candidate.gate) packedStages = {Stage::COUNT_TILES, Stage::SEGMENT_BIAS};
    packedStages.insert(packedStages.end(), {Stage::CANDIDATE_ATTENTION, Stage::CANDIDATE_FFN});
    if (reread) packedStages.push_back(Stage::EVENT_REREAD);
    packedStages.insert(packedStages.end(), {Stage::DISCARD_VALUE, Stage::COMPOSE_VALUES});
  }
  const auto& stages = shape.declaration ? shape.declaration->stages : packedStages;
  auto next = stages.begin();
  auto take = [&](Stage stage) {
    require(next != stages.end() && *next == stage, "Unsupported candidate stage order");
    ++next;
  };
  if (candidate.gate) {
    take(Stage::COUNT_TILES);
    take(Stage::SEGMENT_BIAS);
  }
  take(Stage::CANDIDATE_ATTENTION);
  take(Stage::CANDIDATE_FFN);
  if (reread) add(Kernel::kRereadKeyValue, 0, false, {R::kState}, {R::kRereadMemory});
  add(Kernel::kCandidate, 0, false, {R::kInput, R::kStatic, R::kState, R::kMemory}, {R::kTransition});
  if (reread) {
    take(Stage::EVENT_REREAD);
    add(Kernel::kReread, 0, false, {R::kInput, R::kTransition, R::kRereadMemory}, {R::kTransition, R::kValue});
  } else {
    steps.back().outputs |= regionBit(R::kValue);
  }
  take(Stage::DISCARD_VALUE);
  take(Stage::COMPOSE_VALUES);
  require(next == stages.end(), "Unsupported additional candidate stages");
  add(Kernel::kValueGraph, 0, false, {R::kInput, R::kState, R::kTransition, R::kValue}, {R::kScores});
}

}  // namespace prism
