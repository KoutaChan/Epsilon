#include "execution_plan.h"
#include "collection_fixture.h"
#include "packed_weights.h"
#include "workspace_layout.h"
#include "relation_terms.h"

#include <cassert>
#include <climits>
#include <cstring>
#include <algorithm>
#include <stdexcept>

using namespace prism;

namespace {

PlanShape shapeFor(const WidthCapability& cap, RuleCapability rule, bool reread, int all, int depth) {
  PlanShape shape{};
  auto& a = shape.architecture;
  a.hidden = cap.width;
  a.epsilonBits = a.scaleBits = a.tauBits = 0x3f800000;
  a.depth = depth;
  for (int l = 0; l < kMaxDepth; ++l) {
    a.layer[l].role = l < depth ? static_cast<int>(l < all ? LayerRole::kAllTokens : LayerRole::kFixedTokens) : -1;
    if (l < depth) a.layer[l].ffn[0] = 2 * kFfnChunk;
  }
  a.candidateFfn = kFfnChunk;
  a.gate = kGateStep;
  a.roles = a.gateKinds = 1;
  a.seats = rule.seats;
  a.tiles = rule.tiles;
  a.keyCodes = a.tileRelations = 1;
  a.stages = static_cast<int>(reread ? CandidateStages::kReread : CandidateStages::kOne);
  a.maxEvents = 33;
  shape.rows = 5;
  shape.events = 67;
  shape.transitions = 17;
  shape.actions = 19;
  shape.inputBytes = 65536;
  shape.weightBytes = 1 << 24;
  if (!cap.standard) {
    auto& o = shape.options;
    o.features = PlanOptions::kCount | PlanOptions::kSegments;
    o.countOffset = 1 << 16;
    o.countWidth = kCountWidth;
    o.segmentsOffset = 1 << 20;
    o.segments = {rule.fixed() + rule.tiles, 1, 1, {{rule.fixed(), 0, 1}}};
  }
  return shape;
}

void inspect(PlanShape shape, const WidthCapability& cap) {
  const auto& a = shape.architecture;
  checkedPlanShape(shape);
  ExecutionPlan plan(shape, cap);
  WorkspaceLayout layout(shape, plan, cap.folded, !cap.folded);
  PackedWeights pack(shape);
  int next = 0;
  auto check = [&](Kernel k, int l) {
    auto& step = plan.steps.at(next++);
    assert(step.kernel == k && step.layer == l && acceptsLayer(k, l, a));
  };
  check(Kernel::kEmbed, 0);
  check(Kernel::kRelation, 0);
  for (int l = 0; l < a.depth; ++l) {
    bool all = l < a.allTokenLayers();
    check(all ? Kernel::kAttentionAll : Kernel::kAttentionFixed, l);
    const auto& attention = plan.steps.at(next - 1);
    assert(attention.causal == (shape.options.has(PlanOptions::kCausal) && l == 0));
    assert(attention.reads(Region::kNormalized) == (!cap.folded || l != 0));
    assert(attention.copiedKeysIndex == (all ? -1 : a.fixedIndex(l) - 1));
    check(all ? Kernel::kMlpAll : (l + 1 == a.depth ? Kernel::kMlpLast : Kernel::kMlpFixed), l);
  }
  if (a.reread()) check(Kernel::kRereadKeyValue, 0);
  check(Kernel::kCandidate, 0);
  if (a.reread()) check(Kernel::kReread, 0);
  check(Kernel::kValueGraph, 0);
  assert(size_t(next) == plan.steps.size());
  assert(layout.capacity[int(Region::kNormalized)] == shape.tokens() * a.hidden * sizeof(uint16_t));
  assert(layout.capacity[int(Region::kRereadMemory)] == (a.reread() ? shape.tokens() * 2 * a.attention() * 2 : 0));
  size_t distinctBytes = 0;
  std::array<int, kRegions> first, last;
  first.fill(next);
  last.fill(-1);
  for (int r = 0; r < kRegions; ++r) {
    assert(layout.offset[r] % kPackAlignment == 0);
    assert(layout.offset[r] + layout.capacity[r] <= layout.bytes);
    distinctBytes += layout.capacity[r];
    for (int i = 0; i < next; ++i) {
      const auto& s = plan.steps[i];
      if ((s.inputs | s.outputs) & regionBit(static_cast<Region>(r))) {
        first[r] = std::min(first[r], i);
        last[r] = i;
      }
    }
  }
  assert(layout.bytes < distinctBytes);
  for (int r = 0; r < kRegions; ++r) {
    for (int q = 0; q < r; ++q) {
      if (last[q] < first[r] || last[r] < first[q]) continue;
      assert(layout.offset[r] + layout.capacity[r] <= layout.offset[q] ||
             layout.offset[q] + layout.capacity[q] <= layout.offset[r]);
    }
  }
  if (cap.folded && shape.options.has(PlanOptions::kCausal))
    assert(layout.offset[int(Region::kAttention)] == layout.capacity[int(Region::kInput)] + kPackAlignment);
  size_t count = 0;
  for (const auto& copy : pack.fragments) {
    assert(copy.offset == count);
    assert(&pack.fragments.at(pack.index[weightFirstEntry(copy.weight) + copy.entry]) == &copy);
    count += size_t(copy.rows) * copy.columns;
  }
  assert(count == pack.fragmentElements);
  assert((pack.index[weightFirstEntry(Weight::kRereadOutput)] >= 0) == a.reread());
  assert(pack.version == -1);
  EventPageRequirements page{24, 0, cap.folded ? 0 : 65, 32, cap.folded};
  for (int slots : {0, 1, 31, 32, 33, 63, 64, 65, 66, 67, 68}) {
    EventPageLayout pageLayout(shape, page, slots);
    assert(pageLayout.projections % kPackAlignment == 0 && pageLayout.projections <= pageLayout.bytes);
    const int rows = pageLayout.projectionRows;
    assert(rows == 0 || ((rows & (rows - 1)) == 0 && rows <= slots && rows <= page.projectionLimit));
    const int fitting = EventPageLayout::fittingSlots(shape, page, pageLayout.bytes);
    assert(fitting >= slots);
    assert(EventPageLayout(shape, page, fitting + 1).bytes > pageLayout.bytes);
    if (slots && pageLayout.bytes > EventPageLayout(shape, page, slots - 1).bytes)
      assert(EventPageLayout::fittingSlots(shape, page, pageLayout.bytes - 1) < slots);
  }
}

void checkRelationCodes() {
  const RelationColumns columns{9, 0, 1, 2, 3, 4, 5, 6, 3};
  const uint8_t query[] = {1, 4, 30, 20, 20, 17, 1, 2, 3};
  const uint8_t key[] = {2, 1, 17, 17, 0, 30, 3, 2, 1};
  const auto q = relationToken(3, query, 37, columns);
  const auto k = relationToken(3, key, 37, columns);
  assert(q.ageStep() == 1 && k.ageStep() == 3 && k.phase() == 2);
  int code[kRelationSegments];
  relationSegments<BlockKind::EVENT, BlockKind::EVENT>(q, k, 15, 4, code);
  assert(packCode(code) == 0x113210f);
  relationSegments<BlockKind::MIXED, BlockKind::MIXED>(q, k, 15, 4, code);
  assert(packCode(code) == 0x113210f);

  const uint8_t seat[] = {0, 2, 0, 0, 0, 0, 0, 0, 0};
  const uint8_t tile[] = {3, 0, 0, 0, 0, 0, 3, 1, 2};
  const auto owner = relationToken(1, seat, 0, columns);
  const auto discard = relationToken(2, tile, 0, columns);
  relationSegments<BlockKind::FIXED, BlockKind::FIXED>(owner, discard, 14, 3, code);
  assert(packCode(code) == 0x6008006);
  relationSegments<BlockKind::FIXED, BlockKind::FIXED>(discard, owner, 14, 3, code);
  assert(packCode(code) == 0xa005009);
}

template <class F>
void rejects(F operation) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  assert(false);
}

void checkDeclaredPlan() {
  for (auto width : kWidthCapabilities)
    for (auto rule : kRuleCapabilities) {
      for (bool causal : {false, true})
        for (bool reread : {false, true}) {
          if (reread && (causal || !width.reread)) continue;
          auto model = test::collectionModel(width, rule, causal, reread);
          auto shape = test::compileModel(model);
          ExecutionPlan plan(shape, width);
          assert(plan.steps[2].kernel == Kernel::kAttentionAll && plan.steps[2].causal == causal);
          assert(plan.steps[4].kernel == Kernel::kAttentionFixed && !plan.steps[4].causal);
          assert(shape.architecture.ffn(1, TokenGroup::kContext) == 2 * kFfnChunk &&
                 shape.architecture.ffn(1, TokenGroup::kTile) == kFfnChunk);
          assert(shape.architecture.reread() == reread);
          auto rejectsModel = [&](auto change) {
            auto bad = model;
            change(bad);
            rejects([&] {
              auto shape = test::compileModel(bad);
              ExecutionPlan plan(shape, width);
            });
          };
          rejectsModel([](auto& m) { std::swap(m.stages[0], m.stages[1]); });
          rejectsModel([](auto& m) { m.layers[0].keys = schema::Queries::FIXED_TOKENS; });
          rejectsModel([](auto& m) { m.layers.back().mask = schema::Mask::PUBLIC_PREFIX; });
          rejectsModel([](auto& m) { m.layers.back().source = schema::Source::EMBEDDING; });
          rejectsModel([](auto& m) { std::swap(m.arithmetic.front(), m.arithmetic.back()); });
          rejectsModel([](auto& m) { std::swap(m.inputs[0], m.inputs[1]); });
          rejectsModel([](auto& m) { m.headWidth *= 2; });
          auto bytes = schema::encode(model);
          rejects([&] { declaredPlan(bytes.data(), bytes.size(), schema::version + 1, shape, 0, 0); });
          rejects([&] { declaredPlan(bytes.data(), bytes.size() - 1, schema::version, shape, 0, 0); });
          PackHeader header{};
          for (int i = 0; i < kSections; ++i) header.sectionWidth[i] = model.inputs[i].width;
          header.composition[0] = int(Composition::kDiscardValue);
          header.composition[1] = int(Composition::kTurnGate);
          header.gate[0] = -1;
          validateDeclaredWeights(model, header);
          ++header.sectionWidth[0];
          rejects([&] { validateDeclaredWeights(model, header); });
          --header.sectionWidth[0];
          header.gate[1] = -1;
          rejects([&] { validateDeclaredWeights(model, header); });
        }
    }
}

// 三麻の再読モデルの各段が実際に使う範囲を、計画の入出力マスクから独立に照合する。
void checkRereadWorkspace() {
  using R = Region;
  auto model = test::collectionModel(kWidthFolded, {3, 27}, false, true, false, 3);
  model.layers[0].contextWidth = model.layers[0].tileWidth = 512;
  for (int i : {1, 2}) {
    model.layers[i].contextWidth = 1536;
    model.layers[i].tileWidth = 768;
  }
  model.gateWidth = 512;
  model.candidateWidth = 1024;
  auto shape = test::compileModel(model);
  struct Use {
    R region;
    int first, last;
  };
  constexpr Use uses[] = {{R::kInput, 0, 11},     {R::kResidual, 0, 7},      {R::kQkv0, 0, 2},
                          {R::kStatic, 0, 9},     {R::kRelationCode, 1, 6},  {R::kAttention, 2, 7},
                          {R::kNormalized, 3, 6}, {R::kEventKeyValue, 3, 6}, {R::kRereadMemory, 3, 10},
                          {R::kState, 7, 11},     {R::kMemory, 7, 9},        {R::kTransition, 9, 11},
                          {R::kValue, 10, 11},    {R::kScores, 11, 11}};
  const int fixedTokens = 1 + model.seats + model.tiles;
  const int tokensPerRow = fixedTokens + model.maxEvents;
  const size_t kvBytesPerToken = 2 * shape.attention() * sizeof(uint16_t);
  const size_t normalizedBytesPerToken = shape.hidden() * sizeof(uint16_t);
  // 全行のイベントが上限まであるとき、符号付き32ビットのバイト位置で読める最後の行数。
  const int maxRows = INT32_MAX / (tokensPerRow * std::max(kvBytesPerToken, normalizedBytesPerToken));
  for (int rows : {1, maxRows, maxRows + 1}) {
    shape.rows = rows;
    shape.events = rows == 1 ? 1 : rows * model.maxEvents;
    shape.transitions = rows * model.tiles;
    shape.actions = shape.transitions + rows;
    ExecutionPlan graph(shape, kWidthFolded);
    if (rows > maxRows) {
      rejects([&] { WorkspaceLayout layout(shape, graph, true, false); });
      continue;
    }
    WorkspaceLayout layout(shape, graph, true, false);
    assert(graph.steps.size() == 12);
    assert(graph.steps[3].kernel == Kernel::kMlpAll);
    assert(graph.steps[8].kernel == Kernel::kRereadKeyValue);
    assert(graph.steps[10].kernel == Kernel::kReread);
    const size_t fixedEnd = size_t(fixedTokens) * rows * kvBytesPerToken;
    assert(layout.capacity[int(R::kRereadMemory)] == fixedEnd + shape.events * kvBytesPerToken);
    assert(layout.capacity[int(R::kEventKeyValue)] == shape.events * kvBytesPerToken);
    for (auto a : uses) {
      assert(graph.steps.at(a.first).reads(a.region) || graph.steps.at(a.first).writes(a.region));
      assert(graph.steps.at(a.last).reads(a.region) || graph.steps.at(a.last).writes(a.region));
      for (auto b : uses) {
        if (a.region == b.region || a.last < b.first || b.last < a.first) continue;
        const int i = int(a.region), j = int(b.region);
        assert(layout.offset[i] + layout.capacity[i] <= layout.offset[j] ||
               layout.offset[j] + layout.capacity[j] <= layout.offset[i]);
      }
    }
  }
}

// ページつきの形（因果マスクの全トークンの層 1 つと固定クエリの層 2 つ）で、各段が起動のコードで読み書きする範囲を、計画の入出力
// マスクから独立に照合する。ページを使わない提出の範囲はこの中に入る。
void checkPagedWorkspace() {
  using R = Region;
  struct Use {
    R region;
    int first, last;
  };
  // 幅 256: 埋め込みはイベントの特徴を Attention の領域に詰めて読み、前段 FFN は最初の固定クエリの層の K/V を QKV0 に書く。
  // その層は QKV0 から K/V を読む。
  constexpr Use folded[] = {{R::kInput, 0, 9},      {R::kAttention, 0, 7},     {R::kResidual, 0, 7},
                            {R::kStatic, 0, 8},     {R::kQkv0, 0, 4},          {R::kRelationCode, 1, 6},
                            {R::kNormalized, 3, 6}, {R::kEventKeyValue, 3, 6}, {R::kState, 7, 9},
                            {R::kMemory, 7, 8},     {R::kTransition, 8, 9},    {R::kValue, 8, 9},
                            {R::kScores, 9, 9}};
  // 幅 384: 埋め込みは固定トークンの代表を STATE に置き、前段 FFN がそこから各行の残差を書く。最初の固定クエリの層は、前段 FFN が
  // QKV0 からページに写した K/V を読む。
  constexpr Use projected[] = {{R::kInput, 0, 9},        {R::kState, 0, 9},      {R::kResidual, 0, 7},
                               {R::kNormalized, 0, 6},   {R::kQkv0, 0, 3},       {R::kStatic, 0, 8},
                               {R::kRelationCode, 1, 6}, {R::kAttention, 2, 7},  {R::kEventKeyValue, 3, 6},
                               {R::kMemory, 7, 8},       {R::kTransition, 8, 9}, {R::kValue, 8, 9},
                               {R::kScores, 9, 9}};
  constexpr Kernel kernels[] = {Kernel::kEmbed,          Kernel::kRelation,  Kernel::kAttentionAll,   Kernel::kMlpAll,
                                Kernel::kAttentionFixed, Kernel::kMlpFixed,  Kernel::kAttentionFixed, Kernel::kMlpLast,
                                Kernel::kCandidate,      Kernel::kValueGraph};
  const auto check = [&](const WidthCapability& width, const auto& uses) {
    auto shape = test::compileModel(test::collectionModel(width, kRuleCapabilities.back(), true, false, false, 3));
    for (int rows : {1, 5}) {
      shape.rows = rows;
      shape.events = rows * shape.architecture.maxEvents;
      shape.transitions = rows * shape.architecture.tiles;
      shape.actions = shape.transitions + rows;
      const ExecutionPlan graph(shape, width);
      const WorkspaceLayout layout(shape, graph, width.folded, !width.folded);
      assert(graph.steps.size() == sizeof(kernels) / sizeof(kernels[0]));
      for (size_t i = 0; i < graph.steps.size(); ++i) assert(graph.steps[i].kernel == kernels[i]);
      size_t end = 0;
      for (const auto& a : uses) {
        const int i = int(a.region);
        end = std::max(end, layout.offset[i] + layout.capacity[i]);
        assert(graph.steps.at(a.first).reads(a.region) || graph.steps.at(a.first).writes(a.region));
        assert(graph.steps.at(a.last).reads(a.region) || graph.steps.at(a.last).writes(a.region));
        for (const auto& b : uses) {
          if (a.region == b.region || a.last < b.first || b.last < a.first) continue;
          const int j = int(b.region);
          assert(layout.offset[i] + layout.capacity[i] <= layout.offset[j] ||
                 layout.offset[j] + layout.capacity[j] <= layout.offset[i]);
        }
      }
      if (!width.folded) {
        // 射影を共有する表（トークンごとの代表・計算するトークン・検索表と、代表の件数）は、埋め込みから前段 FFN まで使うので、
        // 全ての領域の後ろに置く。
        assert(layout.projectionIndexOffset >= end);
        assert(layout.bytes - layout.projectionIndexOffset >= (3 * size_t(shape.tokens()) + 1) * sizeof(int));
      }
    }
  };
  check(kWidthFolded, folded);
  check(kWidthProjected, projected);
}

void checkWeightPack() {
  auto shape = shapeFor(kWidthFolded, kRuleCapabilities.back(), false, 1, 2);
  const ExecutionPlan graph(shape, kWidthFolded);
  PackedWeights packed(shape);
  PackHeader header{};
  header.architecture = shape.architecture;
  std::fill(std::begin(header.sectionWidth), std::end(header.sectionWidth), 1);
  header.sectionWidth[int(Section::kRelation)] = shape.architecture.opponents();
  header.sectionWidth[int(Section::kTransitionCode)] = shape.architecture.fixed();
  header.schema[int(Schema::kRelationRows)] = kRelationSegments * kRelationEntries;
  std::fill(std::begin(header.composition), std::end(header.composition), int(Composition::kZero));
  std::fill(std::begin(header.gate), std::end(header.gate), -1);
  std::fill(std::begin(header.weight), std::end(header.weight), -1);
  const size_t offset = (sizeof(header) + kPackAlignment - 1) / kPackAlignment * kPackAlignment;
  header.weight[weightFirstEntry(Weight::kQueryNorm)] = static_cast<int>(offset);
  std::vector<uint8_t> bytes(offset + shape.hidden() * sizeof(float));
  const auto inspect = [&](size_t length, int64_t version) {
    std::memcpy(bytes.data(), &header, sizeof(header));
    return packed.inspect(shape, graph.candidate, true, bytes.data(), length, version);
  };
  assert(inspect(bytes.size(), 0).offset(Weight::kQueryNorm, 0) == static_cast<int>(offset));
  rejects([&] { inspect(sizeof(header) - 1, 0); });
  rejects([&] { inspect(bytes.size() - 1, 0); });
  rejects([&] { inspect(bytes.size(), -1); });
  // NaN と ±∞ は拒み、有限の境界値（最大・最小の正規数、非正規数）と符号つきの 0 は通す。
  const auto last = bytes.data() + bytes.size() - sizeof(uint32_t);
  for (uint32_t bits : {0x7fc00001u, 0xffc00000u, 0x7f800000u, 0xff800000u}) {
    std::memcpy(last, &bits, sizeof(bits));
    rejects([&] { inspect(bytes.size(), 1); });
  }
  for (uint32_t bits : {0x7f7fffffu, 0xff7fffffu, 0x00800000u, 0x00000001u, 0x80000001u, 0x80000000u, 0u}) {
    std::memcpy(last, &bits, sizeof(bits));
    assert(inspect(bytes.size(), 1).offset(Weight::kQueryNorm, 0) == static_cast<int>(offset));
  }
  const uint32_t nan = 0x7fc00001u;
  std::memcpy(last, &nan, sizeof(nan));
  rejects([&] { inspect(bytes.size(), 1); });
  std::fill(bytes.begin() + offset, bytes.end(), 0);
  header.weight[weightFirstEntry(Weight::kQueryNorm)]++;
  rejects([&] { inspect(bytes.size(), 1); });
  assert(packed.version == -1 && !packed.bytes);
}

}  // namespace

int main() {
  checkRelationCodes();
  checkWeightPack();
  checkDeclaredPlan();
  checkRereadWorkspace();
  checkPagedWorkspace();
  for (const auto& cap : kWidthCapabilities) {
    for (auto rule : kRuleCapabilities) {
      for (int depth = 2; depth <= kMaxDepth; ++depth) {
        for (int all = 1; all < depth; ++all) {
          auto shape = shapeFor(cap, rule, false, all, depth);
          inspect(shape, cap);
          if (cap.reread) inspect(shapeFor(cap, rule, true, all, depth), cap);
          if (all == 1) {
            shape.options.features |= PlanOptions::kCausal;
            inspect(shape, cap);
          }
        }
      }
    }
  }
  auto s = shapeFor(kWidthFolded, kRuleCapabilities.front(), false, 1, 2);
  s.rows = s.transitions = s.actions = INT_MAX;
  rejects([&] { checkedPlanShape(s); });
  s = shapeFor(kWidthFolded, kRuleCapabilities.front(), false, 1, 2);
  s.architecture.maxEvents = INT_MAX;
  rejects([&] { checkedPlanShape(s); });
  s = shapeFor(kWidthProjected, kRuleCapabilities.front(), false, 1, 2);
  s.options.segmentsOffset = s.options.countOffset;
  rejects([&] { ExecutionPlan plan(s, kWidthProjected); });
}
