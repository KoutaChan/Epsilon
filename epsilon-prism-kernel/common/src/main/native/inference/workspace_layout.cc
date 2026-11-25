#include "cache/event_page_layout.h"
#include "workspace_layout.h"
#include "plan_checks.h"
#include "relation_layout.h"

#include <algorithm>
#include <limits>

namespace prism {

namespace {
constexpr size_t kBufferBytes = std::numeric_limits<int32_t>::max();
size_t alignUp(size_t value) { return (value + kPackAlignment - 1) / kPackAlignment * kPackAlignment; }

}

WorkspaceLayout::WorkspaceLayout(const PlanShape& shape, const ExecutionPlan& plan, bool folded,
                                 bool sharedProjection) {
  const auto& a = shape.architecture;
  const size_t pairWidth = static_cast<size_t>(2) * shape.attention();
  capacity[ordinal(Region::kInput)] = shape.inputBytes;
  capacity[ordinal(Region::kResidual)] = shape.tokens() * shape.hidden() * sizeof(uint16_t);
  capacity[ordinal(Region::kNormalized)] = shape.tokens() * shape.hidden() * sizeof(uint16_t);
  capacity[ordinal(Region::kQkv0)] = shape.tokens() * 3 * shape.attention() * sizeof(uint16_t);
  capacity[ordinal(Region::kAttention)] = shape.tokens() * shape.attention() * sizeof(uint16_t);
  capacity[ordinal(Region::kRelationCode)] =
      (size_t(shape.rows) * a.codeBlocks() * a.codeBlocks() * kCodeBlockWords) * sizeof(uint32_t);
  capacity[ordinal(Region::kState)] = shape.fixed() * shape.hidden() * sizeof(uint16_t);
  capacity[ordinal(Region::kMemory)] = shape.fixed() * pairWidth * sizeof(uint16_t);
  capacity[ordinal(Region::kStatic)] = static_cast<size_t>(shape.transitions) * shape.hidden() * sizeof(uint16_t);
  capacity[ordinal(Region::kTransition)] = static_cast<size_t>(shape.transitions) * shape.hidden() * sizeof(uint16_t);
  capacity[ordinal(Region::kValue)] = static_cast<size_t>(shape.transitions) * sizeof(float);
  capacity[ordinal(Region::kScores)] = static_cast<size_t>(shape.actions) * sizeof(float);
  capacity[ordinal(Region::kEventKeyValue)] =
      static_cast<size_t>(a.copiedEventLayers()) * shape.events * pairWidth * sizeof(uint16_t);
  capacity[ordinal(Region::kRereadMemory)] = a.reread() ? shape.tokens() * pairWidth * sizeof(uint16_t) : 0;
  // 全トークンの層と固定クエリの層は n を、候補層は Cross-Attention のキーとバリューを、device::Buffer の 32 bit のバイトオフセットで読む。
  require(capacity[ordinal(Region::kNormalized)] <= kBufferBytes,
          "the normalized region of " + std::to_string(capacity[ordinal(Region::kNormalized)]) +
              " bytes exceeds the 2 GiB that all-token attention/fixed-token attention address with buffer offsets");
  require(capacity[ordinal(Region::kMemory)] <= kBufferBytes,
          "the memory region of " + std::to_string(capacity[ordinal(Region::kMemory)]) +
              " bytes exceeds the 2 GiB that candidate attention addresses with buffer offsets");
  require(capacity[ordinal(Region::kRereadMemory)] <= kBufferBytes,
          "the reread memory region exceeds the 2 GiB that event reread addresses with buffer offsets");

  struct Lifetime {
    int first = kKernels * kMaxDepth;
    int last = -1;
  };
  std::array<Lifetime, kRegions> lifetime{};
  for (int i = 0; i < static_cast<int>(plan.steps.size()); ++i) {
    uint32_t used = plan.steps[i].inputs | plan.steps[i].outputs;
    for (int r = 0; r < kRegions; ++r) {
      if (used & regionBit(static_cast<Region>(r))) {
        lifetime[r].first = std::min(lifetime[r].first, i);
        lifetime[r].last = i;
      }
    }
  }
  std::array<int, kRegions> order{};
  for (int r = 0; r < kRegions; ++r) order[r] = r;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return lifetime[a].first < lifetime[b].first; });
  // 折り込み埋め込みのイベント列は、入力の直後の Attention 領域へ一時的に展開する。
  if (folded && shape.options.has(PlanOptions::kCausal)) {
    auto at = std::find(order.begin(), order.end(), ordinal(Region::kAttention));
    std::rotate(order.begin() + 1, at, at + 1);
  }
  struct Block {
    size_t offset;
    size_t capacity;
    int last;
  };
  std::vector<Block> blocks;
  for (int r : order) {
    const size_t n = alignUp(capacity[r]) + kPackAlignment;
    auto best = blocks.end();
    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
      if (it->last < lifetime[r].first && it->capacity >= n && (best == blocks.end() || it->capacity < best->capacity))
        best = it;
    }
    if (best == blocks.end()) {
      offset[r] = bytes;
      blocks.push_back({bytes, n, lifetime[r].last});
      bytes += n;
    } else {
      offset[r] = best->offset;
      best->last = std::max(best->last, lifetime[r].last);
    }
  }
  if (sharedProjection && shape.options.has(PlanOptions::kCausal)) {
    projectionIndexOffset = bytes;
    bytes += (3 * shape.tokens() + 1) * sizeof(int);
  }
}

static cache::PageBudget pageBudget(const PlanShape& shape, EventPageRequirements req) {
  require(req.projectionLimit >= 0 && req.projectionKeyBytes >= 0, "invalid projection dimensions");
  return {size_t(shape.rows),
          size_t(shape.events),
          req.candidateCodeBytes,
          req.eventInputBytes,
          size_t(shape.hidden()),
          size_t(shape.attention()),
          size_t(shape.architecture.copiedEventLayers()),
          size_t(req.projectionKeyBytes),
          kPackAlignment,
          req.projectionLimit,
          req.folded};
}
EventPageLayout::EventPageLayout(const PlanShape& shape, EventPageRequirements req, int slots) {
  auto layout = pageBudget(shape, req).layout(slots);
  map = layout.indices;
  features = layout.raw;
  embedding = layout.input;
  fixedKeys = layout.fixedKeys;
  extraKeys = layout.extraKeys;
  scratch = layout.scratch;
  projections = layout.projections;
  bytes = layout.bytes;
  projectionRows = layout.rows;
}
int EventPageLayout::fittingSlots(const PlanShape& shape, EventPageRequirements req, size_t budget) {
  return pageBudget(shape, req).fit(budget);
}
}  // namespace prism
