#pragma once

#include "weight_pack_format.h"
#include "candidate_segments.h"

#include <array>
#include <memory>
#include <vector>

namespace prism {
namespace schema {
struct Model;
}

// 作業領域を分けた各領域。
enum class Region : int {
  kInput,
  kResidual,
  kNormalized,
  kQkv0,
  kAttention,
  kRelationCode,
  kState,
  kMemory,
  kStatic,
  kTransition,
  kValue,
  kScores,
  kEventKeyValue,
  kRereadMemory,
  kCount
};

// カーネル。
enum class Kernel : int {
  kEmbed,
  kRelation,
  kAttentionAll,
  kMlpAll,
  kAttentionFixed,
  kMlpFixed,
  kMlpLast,
  kRereadKeyValue,
  kCandidate,
  kReread,
  kValueGraph,
  kCount
};

constexpr int kRegions = static_cast<int>(Region::kCount);
constexpr int kKernels = static_cast<int>(Kernel::kCount);

// 候補の数え上げ、区分表、因果マスクを指定する。
struct PlanOptions {
  enum Feature : int32_t { kCausal = 1, kCount = 2, kSegments = 4 };
  int32_t features = 0;
  int32_t countOffset = 0;
  int32_t countFirstCode = 0;
  int32_t countWidth = 0;
  int32_t segmentsOffset = 0;
  SegmentLayout segments{};

  bool has(Feature feature) const { return (features & feature) != 0; }
};

constexpr int kPlanOptionsInts = 5 + kSegmentLayoutInts;
static_assert(sizeof(PlanOptions) == kPlanOptionsInts * sizeof(int32_t));

struct CandidatePackLayout {
  size_t gate = 0;
  size_t projection = 0;
  size_t segments = 0;
  size_t countWorkspace = 0;

  size_t countBytes() const { return gate * sizeof(float) + 2 * projection * sizeof(uint16_t); }
  size_t segmentsBytes() const { return segments * sizeof(float); }
};

// 計画の形。
struct PlanShape {
  Architecture architecture;
  int rows;  // 容量
  int events;
  int transitions;
  int actions;
  size_t inputBytes;
  size_t weightBytes;
  PlanOptions options{};
  std::shared_ptr<const schema::Model> declaration{};

  int hidden() const { return architecture.hidden; }
  int attention() const { return architecture.attention(); }
  int heads() const { return architecture.heads(); }
  size_t fixed() const { return static_cast<size_t>(architecture.fixed()) * rows; }
  size_t tokens() const { return fixed() + events; }
};

// 1 つのバッチの大きさ（行・イベント・遷移・手の数とバイト数）と、領域の位置。
struct BatchShape {
  int rows;
  int events;
  int transitions;
  int actions;
  size_t bytes;
  size_t section[kSections];
};

// 形の記述が決まりを守っているかを確かめる。
void validateArchitecture(const Architecture& architecture);

// この形で、このカーネルがこの層を回すか。
bool acceptsLayer(Kernel kernel, int layer, const Architecture& architecture);

constexpr uint32_t regionBit(Region region) { return uint32_t{1} << static_cast<int>(region); }
static_assert(kRegions <= 32);

struct PlanStep {
  Kernel kernel;
  int layer;
  bool compactEvents;
  uint32_t inputs;
  uint32_t outputs;
  bool causal = false;
  bool lastAllTokenLayer = false;
  int copiedKeysIndex = -1;

  bool reads(Region r) const { return (inputs & regionBit(r)) != 0; }
  bool writes(Region r) const { return (outputs & regionBit(r)) != 0; }
};

// 同じステップで読み書きする領域の寿命は重なる。
struct ExecutionPlan {
  ExecutionPlan(const PlanShape& shape, const WidthCapability& capability);
  std::vector<PlanStep> steps;
  CandidatePackLayout candidate;
};
// 宣言の層・候補段・入力列を検査し、公開済みの重み形式に対応する計画を作る。
PlanShape declaredPlan(const uint8_t* data, size_t length, int abi, PlanShape capacity, int countOffset,
                       int segmentsOffset);
void validateDeclaredWeights(const schema::Model& model, const PackHeader& header);
PlanShape checkedPlanShape(const PlanShape& shape);

}  // namespace prism
