#include "prism_hip_plan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>

#include "prism_hip_kernels.h"
#include "prism_hip_relation.h"

namespace prism {
namespace {

constexpr int kSpecialisedHidden = 256;  // カーネルが特化している H（Java の HipShape.HIDDEN）
constexpr int kFfnChunk = 128;           // K3 の F の段（Java の HipShape.FFN_CHUNK）
constexpr int kGateStep = 64;            // G の倍数（Java の HipShape.GATE_STEP）
constexpr size_t kSlack = 256;           // 区画ごとの余り（検査つきの組み立てが外れた読み書きを区画の先頭へ向けても外に出ない）
constexpr int kRelationRowsExpected = kRelationSegments * kRelationEntries;
constexpr int kMaxExact = 16;            // K7 の LDS の門の道
constexpr int kTileRelationClasses = 30;  // 区画 1・2 の 4 bit に入る牌の関係の類の上限

size_t alignUp(size_t value) {
  return (value + kPackAlignment - 1) / kPackAlignment * kPackAlignment;
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

template <class E>
int ordinal(E value) {
  return static_cast<int>(value);
}

// 区画の行の軸（Java の HipSection.Axis）。
enum class Axis { kRow, kSeat, kTile, kEvent, kToken, kTransition, kAction };

Axis sectionAxis(Section section) {
  switch (section) {
    case Section::kRoundCategory:
    case Section::kRoundScalar:
    case Section::kRowColumn:
      return Axis::kRow;
    case Section::kSeatCategory:
    case Section::kSeatScalar:
      return Axis::kSeat;
    case Section::kTileCategory:
      return Axis::kTile;
    case Section::kEventCategory:
    case Section::kEventScalar:
      return Axis::kEvent;
    case Section::kRelation:
      return Axis::kToken;
    case Section::kTransitionCategory:
    case Section::kTransitionScalar:
    case Section::kTransitionCode:
    case Section::kDiscardExact:
      return Axis::kTransition;
    case Section::kActionColumn:
    case Section::kActionExact:
      return Axis::kAction;
    case Section::kCount:
      break;
  }
  throw std::invalid_argument("unknown input section");
}

Axis slotAxis(EmbeddingSlot slot) {
  switch (slot) {
    case EmbeddingSlot::kRound:
      return Axis::kRow;
    case EmbeddingSlot::kSeat:
      return Axis::kSeat;
    case EmbeddingSlot::kTile:
      return Axis::kTile;
    case EmbeddingSlot::kEvent:
      return Axis::kEvent;
    case EmbeddingSlot::kTransition:
      return Axis::kTransition;
    case EmbeddingSlot::kCount:
      break;
  }
  throw std::invalid_argument("unknown embedding slot");
}

size_t axisCount(Axis axis, const BatchShape& batch, int seats, int tiles) {
  switch (axis) {
    case Axis::kRow:
      return batch.rows;
    case Axis::kSeat:
      return static_cast<size_t>(seats) * batch.rows;
    case Axis::kTile:
      return static_cast<size_t>(tiles) * batch.rows;
    case Axis::kEvent:
      return batch.events;
    case Axis::kToken:
      return static_cast<size_t>(1 + seats + tiles) * batch.rows + batch.events;
    case Axis::kTransition:
      return batch.transitions;
    case Axis::kAction:
      return batch.actions;
  }
  return 0;
}

// 重みの置き場の要素の数（Java の HipWeight.elements と同じ式）。
size_t weightElements(Weight weight, const PlanShape& shape, const PackHeader& header) {
  const size_t h = shape.hidden;
  const size_t a = shape.attention();
  const size_t f = shape.ffn;
  const size_t g = shape.gate;
  const size_t kinds = static_cast<size_t>(header.value(Schema::kGateKinds));
  const size_t actionExact = static_cast<size_t>(header.width(Section::kActionExact));
  switch (weight) {
    case Weight::kTileRelation: {
      const size_t tiles = static_cast<size_t>(header.value(Schema::kTiles));
      return tiles * tiles;
    }
    case Weight::kAttentionNorm:
    case Weight::kOutputBias:
    case Weight::kFfnNorm:
    case Weight::kDownBias:
    case Weight::kFinalNorm:
    case Weight::kQueryNorm:
    case Weight::kCandidateOutputBias:
    case Weight::kCandidateFfnNorm:
    case Weight::kCandidateDownBias:
    case Weight::kDiscardNorm:
    case Weight::kGateNorm:
      return h;
    case Weight::kQkv:
      return 3 * a * h;
    case Weight::kQkvBias:
      return 3 * a;
    case Weight::kRelationBias:
      return static_cast<size_t>(header.value(Schema::kRelationRows)) * shape.heads();
    case Weight::kOutput:
    case Weight::kCandidateOutput:
      return h * a;
    case Weight::kUp:
    case Weight::kCandidateUp:
      return f * h;
    case Weight::kUpBias:
    case Weight::kCandidateUpBias:
      return f;
    case Weight::kDown:
    case Weight::kCandidateDown:
      return h * f;
    case Weight::kTileEntry:
      return h * h;
    case Weight::kQuery:
      return a * h;
    case Weight::kQueryBias:
      return a;
    case Weight::kKeyValue:
      return 2 * a * h;
    case Weight::kKeyValueBias:
      return 2 * a;
    case Weight::kCodeKey:
    case Weight::kCodeValue:
      return static_cast<size_t>(header.value(Schema::kCodes)) * a;
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

std::string weightName(Weight weight) {
  return "weight slot " + std::to_string(ordinal(weight));
}

// 1 層の QKV の断片の写しの BF16 の数（W_qkv を並べ替えただけなので同じ 3A·H）。
size_t qkvFragmentElements(const PlanShape& shape) {
  return static_cast<size_t>(3) * shape.attention() * shape.hidden;
}

// 断片の順の写し（prism_hip_mlp.h の FragmentWeights）を作る MLP の行列。写しの区画にはこの順に、層ごとの重みは深さの層ぶん、
// それ以外は 1 つずつ続ける。
constexpr Weight kMlpFragmentWeights[] = {Weight::kOutput, Weight::kUp, Weight::kDown, Weight::kKeyValue,
                                          Weight::kCandidateOutput, Weight::kCandidateUp, Weight::kCandidateDown};

// 行列 [rows, columns]。
struct MatrixShape {
  int rows;
  int columns;

  size_t elements() const { return static_cast<size_t>(rows) * columns; }
};

// 写しを作る行列の形（weightElements の rows·columns と同じ。launchMlpFragments が重みの要素の数と照らす）。MLP の行列だけを
// 受けるので、他の重みは default で断る。
MatrixShape mlpFragmentMatrix(Weight weight, const PlanShape& shape) {
  switch (weight) {
    case Weight::kOutput:
    case Weight::kCandidateOutput:
      return {shape.hidden, shape.attention()};
    case Weight::kUp:
    case Weight::kCandidateUp:
      return {shape.ffn, shape.hidden};
    case Weight::kDown:
    case Weight::kCandidateDown:
      return {shape.hidden, shape.ffn};
    case Weight::kKeyValue:
      return {2 * shape.attention(), shape.hidden};
    default:
      break;
  }
  throw std::invalid_argument(weightName(weight) + " has no MLP fragment copy");
}

int mlpFragmentLayers(Weight weight, const PlanShape& shape) {
  return weightLayered(weight) ? shape.depth : 1;
}

// 写しの区画の中の (weight, layer) の写しの先頭（BF16 の数）。
size_t mlpFragmentOffset(Weight weight, int layer, const PlanShape& shape) {
  size_t offset = 0;
  for (Weight w : kMlpFragmentWeights) {
    const size_t elements = mlpFragmentMatrix(w, shape).elements();
    if (w == weight) {
      return offset + layer * elements;
    }
    offset += mlpFragmentLayers(w, shape) * elements;
  }
  throw std::invalid_argument(weightName(weight) + " has no MLP fragment copy");
}

// 写しの区画の BF16 の数。
size_t mlpFragmentElements(const PlanShape& shape) {
  size_t elements = 0;
  for (Weight w : kMlpFragmentWeights) {
    elements += mlpFragmentLayers(w, shape) * mlpFragmentMatrix(w, shape).elements();
  }
  return elements;
}

// BF16・FP32 のビットが NaN か（指数が全て 1 で仮数が 0 でない）。
bool isNan(uint16_t bits) {
  return (bits & 0x7fffu) > 0x7f80u;
}

bool isNan(uint32_t bits) {
  return (bits & 0x7fffffffu) > 0x7f800000u;
}

// pack の offset バイトからの count 個の要素（Bits は BF16 なら uint16_t、FP32 なら uint32_t）に NaN があるか。
template <class Bits>
bool containsNan(const uint8_t* pack, size_t offset, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    Bits bits;
    std::memcpy(&bits, pack + offset + i * sizeof(Bits), sizeof(Bits));
    if (isNan(bits)) {
      return true;
    }
  }
  return false;
}

// WeightPack の BF16・FP32 の部分（重みの置き場と、埋め込みの表・W_d・b）に NaN がないこと（頭を照らした後に呼ぶ）。NaN の重みは使う
// 判断のスコアを黙って NaN にするので、公開のときに名前つきで拒む。網の中の NaN は演算が作るものだけになる（K2〜K6 の活性の
// 丸め bf16RoundActivation は NaN の選択を持たず、届く NaN についての約束は prism_hip_common.h）。
void requireNoNan(const PackHeader& header, const PlanShape& shape, const uint8_t* pack) {
  for (int w = 0; w < kWeightSlots; ++w) {
    const Weight weight = static_cast<Weight>(w);
    const int bytes = weightElementBytes(weight);
    for (int l = 0; l < kMaxDepth; ++l) {
      const int32_t offset = header.weight[w][l];
      if (offset < 0 || bytes == 1) {  // 1 バイトは牌の関係の類（整数）
        continue;
      }
      const size_t count = weightElements(weight, shape, header);
      const bool nan = bytes == 2 ? containsNan<uint16_t>(pack, static_cast<size_t>(offset), count)
                                  : containsNan<uint32_t>(pack, static_cast<size_t>(offset), count);
      require(!nan, weightName(weight) + " layer " + std::to_string(l) + " contains NaN");
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
      require(!containsNan<uint32_t>(pack, static_cast<size_t>(table.offset),
                                     static_cast<size_t>(table.rows) * shape.hidden),
              name + " table " + std::to_string(t) + " contains NaN");
    }
    require(!containsNan<uint16_t>(pack, static_cast<size_t>(spec.denseOffset),
                                   static_cast<size_t>(spec.dense) * shape.hidden),
            name + " dense weights contain NaN");
    require(!containsNan<uint32_t>(pack, static_cast<size_t>(spec.biasOffset), shape.hidden),
            name + " bias contains NaN");
  }
}

}  // namespace

DeviceScope::DeviceScope(int device) : device_(device) {
  PRISM_HIP(hipGetDevice(&previous_));
  if (previous_ != device_) {
    PRISM_HIP(hipSetDevice(device_));
  }
}

DeviceScope::~DeviceScope() {
  if (previous_ >= 0 && previous_ != device_) {
    (void)hipSetDevice(previous_);
  }
}

Plan::Plan(int device, hipStream_t stream, const PlanShape& shape) : device_(device), stream_(stream), shape_(shape) {
  require(shape.hidden == kSpecialisedHidden, "HIP kernels are specialised for hidden " +
                                                  std::to_string(kSpecialisedHidden) + " only, got " +
                                                  std::to_string(shape.hidden));
  require(shape.ffn > 0 && shape.ffn % kFfnChunk == 0,
          "ffn must be a positive multiple of " + std::to_string(kFfnChunk) + ", got " + std::to_string(shape.ffn));
  require(shape.gate > 0 && shape.gate % kGateStep == 0,
          "gate must be a positive multiple of " + std::to_string(kGateStep) + ", got " + std::to_string(shape.gate));
  require(shape.depth >= 2 && shape.depth <= kMaxDepth, "depth must be 2.." + std::to_string(kMaxDepth));
  require(std::isfinite(shape.epsilon) && shape.epsilon > 0.0f, "epsilon must be positive and finite");
  require(shape.fixedTokens > 0 && shape.fixedTokens <= kMaxFixedTokens, "fixed tokens per row out of range");
  require(shape.rows > 0 && shape.events >= 0 && shape.transitions >= shape.rows && shape.actions >= shape.rows,
          "invalid batch capacity");
  // K1 は入力の区画を 32 bit のバイトの位置で読む。
  require(shape.inputBytes > 0 && shape.inputBytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "input capacity must be positive and stay below the 2 GiB that K1 addresses with 32-bit offsets");
  require(shape.weightBytes >= sizeof(PackHeader) &&
              shape.weightBytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "weight pack capacity must hold the header and stay below 2 GiB");

  regionCapacity_[ordinal(Region::kInput)] = shape.inputBytes;
  regionCapacity_[ordinal(Region::kResidual)] = shape.tokens() * shape.hidden * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kNormalized)] = shape.tokens() * shape.hidden * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kAttention)] = shape.tokens() * shape.attention() * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kRelationCode)] = codeIndex(shape.rows, 0, 0) * sizeof(uint32_t);
  regionCapacity_[ordinal(Region::kState)] = shape.fixed() * shape.hidden * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kMemory)] = shape.fixed() * 2 * shape.attention() * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kStatic)] = static_cast<size_t>(shape.transitions) * shape.hidden * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kTransition)] =
      static_cast<size_t>(shape.transitions) * shape.hidden * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kValue)] = static_cast<size_t>(shape.transitions) * sizeof(float);
  regionCapacity_[ordinal(Region::kScores)] = static_cast<size_t>(shape.actions) * sizeof(float);
  // K2・K4 は n を、K6 は交差注意の鍵と値を、生の buffer の 32 bit のバイトの位置で読む。
  require(regionCapacity_[ordinal(Region::kNormalized)] <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "the normalized region of " + std::to_string(regionCapacity_[ordinal(Region::kNormalized)]) +
              " bytes exceeds the 2 GiB that K2/K4 address with buffer offsets");
  require(regionCapacity_[ordinal(Region::kMemory)] <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "the memory region of " + std::to_string(regionCapacity_[ordinal(Region::kMemory)]) +
              " bytes exceeds the 2 GiB that K6 addresses with buffer offsets");
  // K3・K5・K6 は MLP の重みの写しを生の buffer の 32 bit のバイトの位置で読む。
  const size_t mlpFragmentBytes = mlpFragmentElements(shape) * sizeof(uint16_t);
  require(mlpFragmentBytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "the MLP weight copies of " + std::to_string(mlpFragmentBytes) +
              " bytes exceed the 2 GiB that K3/K5/K6 address with buffer offsets");

  DeviceScope scope(device_);
  hipDeviceProp_t properties;
  PRISM_HIP(hipGetDeviceProperties(&properties, device_));
  const std::string arch = properties.gcnArchName;
  const std::string expected = PRISM_HIP_ARCH;
  require(arch == expected || arch.rfind(expected + ":", 0) == 0,
          "GPU " + std::to_string(device_) + " is " + arch + " but libprism_hip.so was built for " + expected);

  size_t bytes = 0;
  for (int r = 0; r < kRegions; ++r) {
    regionOffset_[r] = bytes;
    bytes += alignUp(regionCapacity_[r]) + kSlack;
  }
  try {
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&workspace_), bytes));
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&weights_), shape.weightBytes + kSlack));
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&relationBias_),
                        shape.depth * relationBiasFloats(shape.heads() / 2) * sizeof(float) + kSlack));
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&qkvFragments_),
                        shape.depth * qkvFragmentElements(shape) * sizeof(uint16_t) + kSlack));
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&mlpFragments_), mlpFragmentBytes + kSlack));
    for (int k = 0; k < kKernels; ++k) {
      for (int l = 0; l < kMaxLaunches; ++l) {
        PRISM_HIP(hipEventCreate(&start_[k][l]));
        PRISM_HIP(hipEventCreate(&stop_[k][l]));
      }
    }
  } catch (...) {
    release();
    throw;
  }
}

Plan::~Plan() {
  int previous = -1;
  const bool switched = hipGetDevice(&previous) == hipSuccess && previous != device_ && hipSetDevice(device_) == hipSuccess;
  release();
  if (switched) {
    (void)hipSetDevice(previous);
  }
}

void Plan::release() noexcept {
  // 解放の前に、この計画の仕事が終わるのを待つ（作業域を使うカーネルが残らないように）。
  if (workspace_ != nullptr || weights_ != nullptr || relationBias_ != nullptr || qkvFragments_ != nullptr ||
      mlpFragments_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  for (int k = 0; k < kKernels; ++k) {
    for (int l = 0; l < kMaxLaunches; ++l) {
      if (start_[k][l] != nullptr) {
        (void)hipEventDestroy(start_[k][l]);
        start_[k][l] = nullptr;
      }
      if (stop_[k][l] != nullptr) {
        (void)hipEventDestroy(stop_[k][l]);
        stop_[k][l] = nullptr;
      }
    }
  }
  if (mlpFragments_ != nullptr) {
    (void)hipFree(mlpFragments_);
    mlpFragments_ = nullptr;
  }
  if (qkvFragments_ != nullptr) {
    (void)hipFree(qkvFragments_);
    qkvFragments_ = nullptr;
  }
  if (relationBias_ != nullptr) {
    (void)hipFree(relationBias_);
    relationBias_ = nullptr;
  }
  if (weights_ != nullptr) {
    (void)hipFree(weights_);
    weights_ = nullptr;
  }
  if (workspace_ != nullptr) {
    (void)hipFree(workspace_);
    workspace_ = nullptr;
  }
}

size_t Plan::regionBytes(Region region) const {
  require(ordinal(region) >= 0 && ordinal(region) < kRegions, "unknown workspace region");
  return regionCapacity_[ordinal(region)];
}

void Plan::validateHeader(const PackHeader& header, size_t bytes) const {
  require(header.magic == kPackMagic, "the weight pack has no PRISM HIP header");
  require(header.ints == kHeaderInts, "the weight pack header has " + std::to_string(header.ints) +
                                          " ints but libprism_hip.so expects " + std::to_string(kHeaderInts));
  for (int s = 0; s < kSections; ++s) {
    require(header.sectionWidth[s] > 0, "input section " + std::to_string(s) + " has no width");
  }
  const int seats = header.value(Schema::kSeats);
  const int tiles = header.value(Schema::kTiles);
  require(seats > 0 && tiles > 0 && tiles <= kMaxTiles && 1 + seats + tiles == shape_.fixedTokens,
          "seats " + std::to_string(seats) + " and tiles " + std::to_string(tiles) + " do not match " +
              std::to_string(shape_.fixedTokens) + " fixed tokens per row");
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
  require(header.value(Schema::kRelationSafety) >= 0 && header.value(Schema::kRelationSafeties) >= 0 &&
              header.value(Schema::kRelationSafety) + header.value(Schema::kRelationSafeties) <= relationWidth,
          "relation safety columns out of range");
  const int transitionWidth = header.width(Section::kTransitionCategory);
  require(header.value(Schema::kTransitionCallDiscard) >= 0 &&
              header.value(Schema::kTransitionCallDiscard) < transitionWidth &&
              header.value(Schema::kTransitionMainTile) >= 0 &&
              header.value(Schema::kTransitionMainTile) < transitionWidth,
          "transition columns out of range");
  require(header.width(Section::kTransitionCode) == shape_.fixedTokens,
          "every transition needs one key code per fixed token");
  require(header.value(Schema::kCodes) > 0, "key codes must be positive");
  require(header.value(Schema::kRelationRows) == kRelationRowsExpected,
          "the relation bias table must have " + std::to_string(kRelationRowsExpected) + " rows");
  require(header.width(Section::kActionExact) <= kMaxExact, "gate path features exceed " + std::to_string(kMaxExact));
  const int kinds = header.value(Schema::kGateKinds);
  const int roles = header.value(Schema::kRoles);
  require(kinds > 0, "gate kinds must be positive");
  require(roles > 0 && roles <= kMaxRoles, "roles out of range");
  for (int r = 0; r < kMaxRoles; ++r) {
    require(header.composition[r] >= 0 && header.composition[r] < ordinal(Composition::kCount),
            "role composition out of range");
    require(header.gate[r] >= -1 && header.gate[r] < kinds, "role gate kind out of range");
  }
  for (int w = 0; w < kWeightSlots; ++w) {
    const Weight weight = static_cast<Weight>(w);
    for (int l = 0; l < kMaxDepth; ++l) {
      const int32_t offset = header.weight[w][l];
      const bool allowed = weightLayered(weight) ? l < shape_.depth : l == 0;
      if (!allowed) {
        require(offset == -1, weightName(weight) + " has an unexpected layer " + std::to_string(l));
        continue;
      }
      if (offset < 0) {
        continue;
      }
      const size_t size = weightElements(weight, shape_, header) * weightElementBytes(weight);
      require(static_cast<size_t>(offset) % kPackAlignment == 0 && static_cast<size_t>(offset) >= sizeof(PackHeader) &&
                  static_cast<size_t>(offset) + size <= bytes,
              weightName(weight) + " layer " + std::to_string(l) + " [" + std::to_string(offset) + ", +" +
                  std::to_string(size) + ") is outside the weight pack of " + std::to_string(bytes) + " bytes");
    }
  }
  for (int e = 0; e < kEmbeddingSlots; ++e) {
    const EmbeddingSpec& spec = header.embedding[e];
    const std::string name = "embedding slot " + std::to_string(e);
    if (spec.tables == 0) {
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
    for (int t = 0; t < spec.tables; ++t) {
      const EmbeddingTable& table = spec.table[t];
      require(table.members > 0 && table.members <= kMaxMembers && table.rows > 0, name + " table shape is invalid");
      require(table.offset >= 0 && static_cast<size_t>(table.offset) % 16 == 0 &&
                  static_cast<size_t>(table.offset) + static_cast<size_t>(table.rows) * shape_.hidden * sizeof(float) <=
                      bytes,
              name + " table " + std::to_string(t) + " is outside the weight pack");
      for (int m = 0; m < table.members; ++m) {
        require(table.column[m] >= 0 && table.column[m] < categoryWidth && table.stride[m] >= 0,
                name + " table member out of range");
      }
    }
    require(spec.dense >= 0 && spec.dense <= kMaxDense, name + " dense count out of range");
    if (spec.dense > 0) {
      require(spec.denseOffset >= 0 && static_cast<size_t>(spec.denseOffset) % 16 == 0 &&
                  static_cast<size_t>(spec.denseOffset) + static_cast<size_t>(spec.dense) * shape_.hidden * 2 <= bytes,
              name + " dense weights are outside the weight pack");
    }
    require(spec.biasOffset >= 0 && static_cast<size_t>(spec.biasOffset) % 16 == 0 &&
                static_cast<size_t>(spec.biasOffset) + shape_.hidden * sizeof(float) <= bytes,
            name + " bias is outside the weight pack");
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

void Plan::uploadWeights(const void* host, size_t bytes, int64_t version) {
  require(bytes <= shape_.weightBytes, "weight pack of " + std::to_string(bytes) + " bytes exceeds the plan capacity " +
                                           std::to_string(shape_.weightBytes));
  require(bytes >= sizeof(PackHeader), "weight pack is smaller than its header");
  require(version >= 0, "weights version must not be negative");
  PackHeader header;
  std::memcpy(&header, host, sizeof(PackHeader));
  validateHeader(header, bytes);
  const int32_t tileRelation = header.weight[ordinal(Weight::kTileRelation)][0];
  if (tileRelation >= 0) {
    const auto* classes = static_cast<const uint8_t*>(host) + tileRelation;
    const size_t count = weightElements(Weight::kTileRelation, shape_, header);
    for (size_t i = 0; i < count; ++i) {
      require(classes[i] < kTileRelationClasses, "tile relation class out of range");
    }
  }
  requireNoNan(header, shape_, static_cast<const uint8_t*>(host));
  DeviceScope scope(device_);
  // 写しと重みから作るものが済むまでは重みのない計画として扱う（途中で失敗しても、新しい重みと古い表・写しの組で forward が回らない）。
  version_ = -1;
  header_ = header;
  weightsUsed_ = bytes;
  PRISM_HIP(hipMemcpyAsync(weights_, host, bytes, hipMemcpyHostToDevice, stream_));
  // 偏りの表と QKV・MLP の重みの断片の写しは重みだけで決まり、重みを替える道はここだけなので、公開のたびに作り直す（古いものは
  // 残らない）。pack にない層・重みの分は作らず、カーネルが引くときに止める（relationBias・qkvFragments・mlpFragments）。
  for (int l = 0; l < shape_.depth; ++l) {
    if (header_.weight[ordinal(Weight::kRelationBias)][l] >= 0) {
      const RelationBiasArgs args{weight<float>(Weight::kRelationBias, l), relationBias<float>(l), shape_.heads()};
      launchRelationBias(args, shape_.heads() / 2, stream_);
    }
    if (header_.weight[ordinal(Weight::kQkv)][l] >= 0) {
      const QkvFragmentArgs args{weight<uint16_t>(Weight::kQkv, l), qkvFragments<uint16_t>(l)};
      launchQkvFragments(shape_.hidden, args, shape_.heads() / 2, stream_);
    }
  }
  for (Weight w : kMlpFragmentWeights) {
    const MatrixShape matrix = mlpFragmentMatrix(w, shape_);
    for (int l = 0; l < mlpFragmentLayers(w, shape_); ++l) {
      if (header_.weight[ordinal(w)][l] >= 0) {
        const MlpFragmentArgs args{weight<uint16_t>(w, l), mlpFragments<uint16_t>(w, l), matrix.rows, matrix.columns};
        launchMlpFragments(args, stream_);
      }
    }
  }
  PRISM_HIP(hipStreamSynchronize(stream_));
  version_ = version;
}

void Plan::copyIn(Region region, const void* host, size_t bytes) {
  require(bytes <= regionBytes(region), "copy of " + std::to_string(bytes) + " bytes exceeds the region");
  DeviceScope scope(device_);
  PRISM_HIP(hipMemcpyAsync(workspace_ + regionOffset_[ordinal(region)], host, bytes, hipMemcpyHostToDevice, stream_));
}

void Plan::copyOut(Region region, void* host, size_t bytes) {
  require(bytes <= regionBytes(region), "copy of " + std::to_string(bytes) + " bytes exceeds the region");
  DeviceScope scope(device_);
  PRISM_HIP(hipMemcpyAsync(host, workspace_ + regionOffset_[ordinal(region)], bytes, hipMemcpyDeviceToHost, stream_));
}

void Plan::validate(const BatchShape& batch) const {
  require(version_ >= 0, "weights have not been uploaded to the plan");
  require(batch.rows > 0 && batch.rows <= shape_.rows && batch.events >= 0 && batch.events <= shape_.events &&
              batch.transitions >= batch.rows && batch.transitions <= shape_.transitions &&
              batch.actions >= batch.rows && batch.actions <= shape_.actions,
          "batch counts exceed the plan capacity");
  require(batch.bytes > 0 && batch.bytes <= shape_.inputBytes, "batch blob exceeds the input capacity");
  const int seats = header_.value(Schema::kSeats);
  const int tiles = header_.value(Schema::kTiles);
  for (int s = 0; s < kSections; ++s) {
    const Section section = static_cast<Section>(s);
    const size_t extent =
        axisCount(sectionAxis(section), batch, seats, tiles) * header_.sectionWidth[s] * sectionElementBytes(section);
    require(batch.section[s] % 16 == 0 && batch.section[s] <= batch.bytes && extent <= batch.bytes - batch.section[s],
            "input section " + std::to_string(s) + " at " + std::to_string(batch.section[s]) + " (+" +
                std::to_string(extent) + ") is outside the blob of " + std::to_string(batch.bytes) + " bytes");
  }
}

template <class T>
Span<T> Plan::regionSpan(Region region, size_t elementOffset) const {
  const size_t capacity = regionCapacity_[ordinal(region)] / sizeof(T);
  require(elementOffset <= capacity, "region offset out of range");
  return Span<T>{reinterpret_cast<T*>(workspace_ + regionOffset_[ordinal(region)]) + elementOffset,
                 capacity - elementOffset};
}

size_t Plan::weightBytes(Weight weight) const {
  return weightElements(weight, shape_, header_) * weightElementBytes(weight);
}

template <class T>
Span<const T> Plan::weight(Weight weight, int layer) const {
  const int32_t offset = header_.weight[ordinal(weight)][weightLayered(weight) ? layer : 0];
  require(offset >= 0, weightName(weight) + " layer " + std::to_string(layer) + " is absent from the weight pack");
  return Span<const T>{reinterpret_cast<const T*>(weights_ + offset), weightBytes(weight) / sizeof(T)};
}

template <class T>
Span<const T> Plan::section(const BatchShape& batch, Section section) const {
  const int s = ordinal(section);
  const size_t elements = axisCount(sectionAxis(section), batch, header_.value(Schema::kSeats),
                                    header_.value(Schema::kTiles)) *
                          header_.sectionWidth[s];
  return Span<const T>{reinterpret_cast<const T*>(workspace_ + regionOffset_[ordinal(Region::kInput)] + batch.section[s]),
                       elements};
}

template <class T>
Span<T> Plan::relationBias(int layer) const {
  require(header_.weight[ordinal(Weight::kRelationBias)][layer] >= 0,
          weightName(Weight::kRelationBias) + " layer " + std::to_string(layer) + " is absent from the weight pack");
  const size_t floats = relationBiasFloats(shape_.heads() / 2);
  return Span<T>{relationBias_ + layer * floats, floats};
}

template <class T>
Span<T> Plan::qkvFragments(int layer) const {
  require(header_.weight[ordinal(Weight::kQkv)][layer] >= 0,
          weightName(Weight::kQkv) + " layer " + std::to_string(layer) + " is absent from the weight pack");
  const size_t elements = qkvFragmentElements(shape_);
  return Span<T>{qkvFragments_ + layer * elements, elements};
}

template <class T>
Span<T> Plan::mlpFragments(Weight weight, int layer) const {
  require(header_.weight[ordinal(weight)][weightLayered(weight) ? layer : 0] >= 0,
          weightName(weight) + " layer " + std::to_string(layer) + " is absent from the weight pack");
  return Span<T>{mlpFragments_ + mlpFragmentOffset(weight, layer, shape_), mlpFragmentMatrix(weight, shape_).elements()};
}

MlpWeights Plan::mlpWeights(int layer) const {
  return MlpWeights{mlpFragments<const uint16_t>(Weight::kOutput, layer),
                    weight<float>(Weight::kOutputBias, layer),
                    weight<float>(Weight::kFfnNorm, layer),
                    mlpFragments<const uint16_t>(Weight::kUp, layer),
                    weight<float>(Weight::kUpBias, layer),
                    mlpFragments<const uint16_t>(Weight::kDown, layer),
                    weight<float>(Weight::kDownBias, layer)};
}

MlpWeights Plan::candidateWeights() const {
  return MlpWeights{mlpFragments<const uint16_t>(Weight::kCandidateOutput, 0),
                    weight<float>(Weight::kCandidateOutputBias, 0),
                    weight<float>(Weight::kCandidateFfnNorm, 0),
                    mlpFragments<const uint16_t>(Weight::kCandidateUp, 0),
                    weight<float>(Weight::kCandidateUpBias, 0),
                    mlpFragments<const uint16_t>(Weight::kCandidateDown, 0),
                    weight<float>(Weight::kCandidateDownBias, 0)};
}

TokenLayout Plan::layout(const BatchShape& batch) const {
  return TokenLayout{batch.rows, header_.value(Schema::kSeats), header_.value(Schema::kTiles)};
}

void Plan::launch(Kernel kernel, int layer, const BatchShape& batch, Pass pass) {
  const int hidden = shape_.hidden;
  const TokenLayout tokens = layout(batch);
  const size_t fixed = static_cast<size_t>(tokens.fixed()) * batch.rows;
  const float scale = __builtin_bit_cast(float, header_.value(Schema::kScaleBits));
  const float tau = __builtin_bit_cast(float, header_.value(Schema::kTauBits));
  switch (kernel) {
    case Kernel::kEmbed: {
      EmbedArgs args{};
      args.pack = Span<const uint8_t>{reinterpret_cast<const uint8_t*>(weights_), weightsUsed_};
      args.spec = reinterpret_cast<const EmbeddingSpec*>(weights_ + offsetof(PackHeader, embedding));
      args.input = Span<const uint8_t>{reinterpret_cast<const uint8_t*>(workspace_ + regionOffset_[ordinal(Region::kInput)]),
                                       regionCapacity_[ordinal(Region::kInput)]};
      for (int s = 0; s < kSections; ++s) {
        args.section[s] = batch.section[s];
        args.sectionWidth[s] = header_.sectionWidth[s];
      }
      const size_t starts[kEmbeddingSlots] = {0, static_cast<size_t>(batch.rows),
                                              static_cast<size_t>(1 + tokens.seats) * batch.rows, fixed, 0};
      const int counts[kEmbeddingSlots] = {batch.rows, tokens.seats * batch.rows, tokens.tiles * batch.rows,
                                           batch.events, batch.transitions};
      int blocks = 0;
      for (int e = 0; e < kEmbeddingSlots; ++e) {
        require(header_.embedding[e].tables > 0,
                "embedding slot " + std::to_string(e) + " is absent from the weight pack");
        args.count[e] = counts[e];
        args.blockStart[e] = blocks;
        blocks += (counts[e] + 63) / 64;
        const bool transition = static_cast<EmbeddingSlot>(e) == EmbeddingSlot::kTransition;
        args.out[e] = regionSpan<uint16_t>(transition ? Region::kStatic : Region::kResidual, starts[e] * hidden);
        args.normalized[e] =
            transition ? Span<uint16_t>{nullptr, 0} : regionSpan<uint16_t>(Region::kNormalized, starts[e] * hidden);
      }
      args.blockStart[kEmbeddingSlots] = blocks;
      args.attentionNorm = weight<float>(Weight::kAttentionNorm, 0);
      args.epsilon = shape_.epsilon;
      timed(kernel, [&] { launchEmbed(hidden, args, blocks, stream_); });
      return;
    }
    case Kernel::kRelation: {
      RelationArgs args{};
      args.relation = section<uint8_t>(batch, Section::kRelation);
      args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
      args.tileRelation = weight<uint8_t>(Weight::kTileRelation, 0);
      args.codes = regionSpan<uint32_t>(Region::kRelationCode, 0);
      args.layout = tokens;
      args.rowWidth = header_.width(Section::kRowColumn);
      args.rowFirstEvent = header_.value(Schema::kRowFirstEvent);
      args.rowEvents = header_.value(Schema::kRowEvents);
      args.columns = RelationColumns{header_.width(Section::kRelation),       header_.value(Schema::kRelationTile),
                                     header_.value(Schema::kRelationOwner),   header_.value(Schema::kRelationEvent),
                                     header_.value(Schema::kRelationRiichi),  header_.value(Schema::kRelationLastDiscard),
                                     header_.value(Schema::kRelationLink),    header_.value(Schema::kRelationSafety),
                                     header_.value(Schema::kRelationSafeties)};
      timed(kernel, [&] { launchRelation(args, stream_); });
      return;
    }
    case Kernel::kAttention0:
    case Kernel::kAttention1: {
      AttentionArgs args{};
      args.normalized = regionSpan<const uint16_t>(Region::kNormalized, 0);
      args.o = regionSpan<uint16_t>(Region::kAttention, 0);
      args.codes = regionSpan<const uint32_t>(Region::kRelationCode, 0);
      args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
      args.qkv = qkvFragments<const uint16_t>(layer);
      args.qkvBias = weight<float>(Weight::kQkvBias, layer);
      args.bias = relationBias<const float>(layer);
      args.layout = tokens;
      args.rowWidth = header_.width(Section::kRowColumn);
      args.rowFirstEvent = header_.value(Schema::kRowFirstEvent);
      args.rowEvents = header_.value(Schema::kRowEvents);
      args.scale = scale;
      const bool last = kernel == Kernel::kAttention1;
      timed(kernel, [&] { launchAttention(hidden, last, args, shape_.heads() / 2, stream_); });
      return;
    }
    case Kernel::kMlp0: {
      MlpArgs args{};
      args.x = regionSpan<uint16_t>(Region::kResidual, 0);
      args.o = regionSpan<const uint16_t>(Region::kAttention, 0);
      args.normalized = regionSpan<uint16_t>(Region::kNormalized, 0);
      args.weights = mlpWeights(layer);
      args.nextNorm = weight<float>(Weight::kAttentionNorm, layer + 1);
      args.tokens = static_cast<int>(fixed + batch.events);
      const bool last = pass == Pass::kForward && layer == shape_.depth - 2;
      args.residualTokens = last ? static_cast<int>(fixed) : args.tokens;
      args.ffn = shape_.ffn;
      args.epsilon = shape_.epsilon;
      timed(kernel, [&] { launchMlp(hidden, args, stream_); });
      return;
    }
    case Kernel::kMlp1Kv: {
      MlpKvArgs args{};
      args.x = regionSpan<const uint16_t>(Region::kResidual, 0);
      args.o = regionSpan<const uint16_t>(Region::kAttention, 0);
      args.weights = mlpWeights(layer);
      args.finalNorm = weight<float>(Weight::kFinalNorm, 0);
      args.keyValue = mlpFragments<const uint16_t>(Weight::kKeyValue, 0);
      args.keyValueBias = weight<float>(Weight::kKeyValueBias, 0);
      args.state = regionSpan<uint16_t>(Region::kState, 0);
      args.memory = regionSpan<uint16_t>(Region::kMemory, 0);
      args.layout = tokens;
      args.tokens = static_cast<int>(fixed);
      args.ffn = shape_.ffn;
      args.epsilon = shape_.epsilon;
      timed(kernel, [&] { launchMlpKv(hidden, args, stream_); });
      return;
    }
    case Kernel::kCandidate: {
      CandidateArgs args{};
      args.statics = regionSpan<const uint16_t>(Region::kStatic, 0);
      args.state = regionSpan<const uint16_t>(Region::kState, 0);
      args.memory = regionSpan<const uint16_t>(Region::kMemory, 0);
      args.category = section<uint8_t>(batch, Section::kTransitionCategory);
      args.code = section<uint8_t>(batch, Section::kTransitionCode);
      args.discardExact = section<float>(batch, Section::kDiscardExact);
      args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
      args.tileEntry = weight<uint16_t>(Weight::kTileEntry, 0);
      args.queryNorm = weight<float>(Weight::kQueryNorm, 0);
      args.query = weight<uint16_t>(Weight::kQuery, 0);
      args.queryBias = weight<float>(Weight::kQueryBias, 0);
      args.codeKey = weight<float>(Weight::kCodeKey, 0);
      args.codeValue = weight<float>(Weight::kCodeValue, 0);
      args.weights = candidateWeights();
      args.discardNorm = weight<float>(Weight::kDiscardNorm, 0);
      args.discardHidden = weight<uint16_t>(Weight::kDiscardHidden, 0);
      args.discardHiddenBias = weight<float>(Weight::kDiscardHiddenBias, 0);
      args.discardOutput = weight<float>(Weight::kDiscardOutput, 0);
      args.discardWeights = weight<float>(Weight::kDiscardExact, 0);
      args.transition = regionSpan<uint16_t>(Region::kTransition, 0);
      args.value = regionSpan<float>(Region::kValue, 0);
      args.layout = tokens;
      args.transitions = batch.transitions;
      args.categoryWidth = header_.width(Section::kTransitionCategory);
      args.callDiscard = header_.value(Schema::kTransitionCallDiscard);
      args.mainTile = header_.value(Schema::kTransitionMainTile);
      args.codes = header_.value(Schema::kCodes);
      args.discardExactWidth = header_.width(Section::kDiscardExact);
      args.rowWidth = header_.width(Section::kRowColumn);
      args.rowFirstTransition = header_.value(Schema::kRowFirstTransition);
      args.ffn = shape_.ffn;
      args.gate = shape_.gate;
      args.epsilon = shape_.epsilon;
      args.scale = scale;
      timed(kernel, [&] { launchCandidate(hidden, args, stream_); });
      return;
    }
    case Kernel::kValueGraph: {
      ValueGraphArgs args{};
      args.transition = regionSpan<const uint16_t>(Region::kTransition, 0);
      args.value = regionSpan<const float>(Region::kValue, 0);
      args.state = regionSpan<const uint16_t>(Region::kState, 0);
      args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
      args.actionColumn = section<int32_t>(batch, Section::kActionColumn);
      args.actionExact = section<float>(batch, Section::kActionExact);
      args.gateNorm = weight<float>(Weight::kGateNorm, 0);
      args.gateHidden = weight<uint16_t>(Weight::kGateHidden, 0);
      args.gateHiddenBias = weight<float>(Weight::kGateHiddenBias, 0);
      args.kindOffset = weight<float>(Weight::kGateKindOffset, 0);
      args.exactProjection = weight<float>(Weight::kGateExactProjection, 0);
      args.gateOutput = weight<float>(Weight::kGateOutput, 0);
      args.gateOutputBias = weight<float>(Weight::kGateOutputBias, 0);
      args.exactOutput = weight<float>(Weight::kGateExactOutput, 0);
      args.scores = regionSpan<float>(Region::kScores, 0);
      args.rows = batch.rows;
      args.rowWidth = header_.width(Section::kRowColumn);
      args.rowFirstAction = header_.value(Schema::kRowFirstAction);
      args.rowActions = header_.value(Schema::kRowActions);
      args.actionWidth = header_.width(Section::kActionColumn);
      args.actionFirstTransition = header_.value(Schema::kActionFirstTransition);
      args.actionTransitions = header_.value(Schema::kActionTransitions);
      args.actionRole = header_.value(Schema::kActionRole);
      args.actionBase = header_.value(Schema::kActionBase);
      args.actionRepresentative = header_.value(Schema::kActionRepresentative);
      args.actionRow = header_.value(Schema::kActionRow);
      args.actionExactWidth = header_.width(Section::kActionExact);
      args.roles = header_.value(Schema::kRoles);
      for (int r = 0; r < kMaxRoles; ++r) {
        args.composition[r] = header_.composition[r];
        args.gateKind[r] = header_.gate[r];
      }
      args.gateKinds = header_.value(Schema::kGateKinds);
      args.gate = shape_.gate;
      args.epsilon = shape_.epsilon;
      args.tau = tau;
      timed(kernel, [&] { launchValueGraph(hidden, args, stream_); });
      return;
    }
    case Kernel::kCount:
      break;
  }
  throw std::invalid_argument("unknown kernel " + std::to_string(ordinal(kernel)));
}

void Plan::resetTiming() {
  launches_.fill(0);
}

template <class Launch>
void Plan::timed(Kernel kernel, Launch&& launch) {
  const int k = ordinal(kernel);
  const int index = launches_[k];
  const bool record = profile_ && index < kMaxLaunches;
  if (record) {
    PRISM_HIP(hipEventRecord(start_[k][index], stream_));
  }
  launch();
  if (record) {
    PRISM_HIP(hipEventRecord(stop_[k][index], stream_));
  }
  launches_[k] = index + 1;
}

void Plan::run(Kernel kernel, int layer, const BatchShape& batch) {
  validate(batch);
  const bool layered = kernel == Kernel::kAttention0 || kernel == Kernel::kMlp0;
  const bool last = kernel == Kernel::kAttention1 || kernel == Kernel::kMlp1Kv;
  require(layered ? layer >= 0 && layer < shape_.depth - 1 : last ? layer == shape_.depth - 1 : layer == 0,
          "kernel " + std::to_string(ordinal(kernel)) + " does not run layer " + std::to_string(layer));
  DeviceScope scope(device_);
  resetTiming();
  launch(kernel, layer, batch, Pass::kKernel);
}

void Plan::forward(const void* input, const BatchShape& batch, void* scores, int64_t version) {
  validate(batch);
  require(version == version_, "weight pack version " + std::to_string(version_) + " is stale; the network is at " +
                                   std::to_string(version));
  DeviceScope scope(device_);
  PRISM_HIP(hipMemcpyAsync(workspace_ + regionOffset_[ordinal(Region::kInput)], input, batch.bytes,
                           hipMemcpyHostToDevice, stream_));
  resetTiming();
  launch(Kernel::kEmbed, 0, batch, Pass::kForward);
  launch(Kernel::kRelation, 0, batch, Pass::kForward);
  for (int l = 0; l + 1 < shape_.depth; ++l) {
    launch(Kernel::kAttention0, l, batch, Pass::kForward);
    launch(Kernel::kMlp0, l, batch, Pass::kForward);
  }
  launch(Kernel::kAttention1, shape_.depth - 1, batch, Pass::kForward);
  launch(Kernel::kMlp1Kv, shape_.depth - 1, batch, Pass::kForward);
  launch(Kernel::kCandidate, 0, batch, Pass::kForward);
  launch(Kernel::kValueGraph, 0, batch, Pass::kForward);
  PRISM_HIP(hipMemcpyAsync(scores, workspace_ + regionOffset_[ordinal(Region::kScores)],
                           static_cast<size_t>(batch.actions) * sizeof(float), hipMemcpyDeviceToHost, stream_));
}

std::array<float, kKernels> Plan::kernelMillis() {
  DeviceScope scope(device_);
  std::array<float, kKernels> millis{};
  for (int k = 0; k < kKernels; ++k) {
    const int recorded = profile_ ? std::min(launches_[k], kMaxLaunches) : 0;
    if (recorded == 0) {
      millis[k] = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
    float total = 0.0f;
    for (int l = 0; l < recorded; ++l) {
      float elapsed = 0.0f;
      PRISM_HIP(hipEventSynchronize(stop_[k][l]));
      PRISM_HIP(hipEventElapsedTime(&elapsed, start_[k][l], stop_[k][l]));
      total += elapsed;
    }
    millis[k] = total;
  }
  return millis;
}

void Plan::synchronize() {
  DeviceScope scope(device_);
  PRISM_HIP(hipStreamSynchronize(stream_));
}

}  // namespace prism
