#include "prism_hip_plan.h"
#include "prism_hip_hot_cache.h"
#include "prism_hip_event_pages.h"
#include "prism_hip_v4.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "prism_hip_capabilities.h"
#include "prism_hip_kernels.h"
#include "prism_hip_relation.h"

namespace prism {

// 形を作るときに一度引き、起動時は選んだ起動関数だけを呼ぶ。
struct WidthKernels {
  WidthCapability capability;
  decltype(&launchEmbed) embed;
  decltype(&embedSharedBytes) embedBytes;
  decltype(&v4::launchWideEmbed) projectedEmbed;
  decltype(&launchQkvFragments) qkv;
  decltype(&launchAttention) attention;
  decltype(&launchV4CausalAttention) causal;
  decltype(&v4::launchSharedProjection) sharedProjection;
  decltype(&launchMlp) mlp;
  decltype(&launchMlpKv) lastMlp;
  decltype(&launchValueGraph) valueGraph;
  decltype(&launchCandidate) standard;
  decltype(&launchKeyValue) rereadKeys;
  decltype(&launchReread) reread;
  bool splitCandidate;  // 実収集で速くなった幅では、候補注意を分離する。
};

constexpr WidthKernels kernelsS{kWidthS,
                                launchEmbed,
                                embedSharedBytes,
                                nullptr,
                                launchQkvFragments,
                                launchAttention,
                                launchV4CausalAttention,
                                nullptr,
                                launchMlp,
                                launchMlpKv,
                                launchValueGraph,
                                launchCandidate,
                                launchKeyValue,
                                launchReread,
                                false};
constexpr WidthKernels kernelsM{kWidthM,
                                nullptr,
                                nullptr,
                                v4::launchWideEmbed,
                                launchV4WideQkvFragments,
                                launchV4WideAttention,
                                launchV4WideCausalAttention,
                                v4::launchSharedProjection,
                                launchV4WideMlp,
                                launchV4WideMlpKv,
                                launchV4WideValueGraph,
                                nullptr,
                                nullptr,
                                nullptr,
                                true};

const WidthKernels& widthKernels(int hidden) {
  switch (hidden) {
#define PRISM_WIDTH(name, width, ...) \
  case width:                         \
    return kernels##name;
#define PRISM_RULES(seats, tiles)
#define PRISM_EAGER_WIDTH(name, width)
#include "prism_hip_capabilities.def"
#undef PRISM_EAGER_WIDTH
#undef PRISM_RULES
#undef PRISM_WIDTH
    default:
      throw std::invalid_argument("no HIP kernels for hidden width " + std::to_string(hidden));
  }
}

namespace {

constexpr int kMaxCodes = kU8Codes - 1;  // コードの表の行の上限（遷移のコードは UINT8）
constexpr size_t kSlack =
    kPackAlignment;  // 領域ごとの余白（境界検査つきのビルドが、外れた読み書きを領域の先頭へ向けても外に出ない）
constexpr int kRelationRowsExpected = kRelationSegments * kRelationEntries;
constexpr int kMaxExact = 16;                  // K7 の LDS に置く、ゲートに直接入る特徴 x^G の数の上限
constexpr int kTileRelationClasses = 30;       // セグメント 1・2 の 4 bit に入る、牌の関係のクラスの上限
constexpr size_t kVectorAlignment = 16;        // 埋め込み表と入力をベクトルで読むときのバイト境界。
constexpr size_t kMaxSharedBytes = 64 * 1024;  // ワークグループの LDS の上限
constexpr size_t kBufferBytes =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());  // 生の buffer で読む領域の上限

size_t alignUp(size_t value) { return (value + kPackAlignment - 1) / kPackAlignment * kPackAlignment; }

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

template <class E>
int ordinal(E value) {
  return static_cast<int>(value);
}

// 領域の行が何の軸か（Java の HipSection.Axis）。
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

// FFN の重みのスロットの項目の幅（Java の HipArchitecture.ffnWidth）。
int ffnWidth(const Architecture& a, int entry) {
  return entry == kCandidateFfnEntry ? a.candidateFfn : a.layer[entry / kTokenGroups].ffn[entry % kTokenGroups];
}

// 重みのスロットの項目がこの形にあるか（Java の HipArchitecture.holds）。組ごとの項目は、その層の FFN の組の幅が 0 でないときだけある。
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

// 重みのスロットの項目の要素の数（Java の HipPack.Header.elements と同じ式）。
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

// 1 層の QKV をフラグメントの順に並べた写しの、BF16 の数（W_qkv を並べ替えただけなので、同じ 3A·H）。
size_t qkvFragmentElements(const PlanShape& shape) {
  return static_cast<size_t>(3) * shape.attention() * shape.hidden();
}

// BF16・FP32 のビットが NaN か（指数が全て 1 で、仮数が 0 でない）。
bool isNan(uint16_t bits) { return (bits & 0x7fffu) > 0x7f80u; }

bool isNan(uint32_t bits) { return (bits & 0x7fffffffu) > 0x7f800000u; }

// pack の offset バイトから始まる count 個の要素（Bits は BF16 なら uint16_t、FP32 なら uint32_t）に NaN があるか。
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

// WeightPack の BF16・FP32 の部分（重みのスロットの全ての項目と、埋め込みの表・W_d・b）に NaN がないことを確かめる（ヘッダーを照合した
// 後に呼ぶ）。NaN の重みは、それを使う判断のスコアを黙って NaN にするので、公開のときに名前を付けて拒む。これでネットワークの中の NaN は、
// 演算が作るものだけになる（活性の丸め bf16RoundActivation は NaN を特別に扱わない。届く NaN についての契約は prism_hip_common.h）。
void requireNoNan(const PackHeader& header, const PlanShape& shape, const uint8_t* pack) {
  for (int w = 0; w < kWeightSlots; ++w) {
    const Weight weight = static_cast<Weight>(w);
    const int bytes = weightElementBytes(weight);
    for (int e = 0; e < indexEntries(weightIndex(weight)); ++e) {
      const int32_t offset = header.offset(weight, e);
      if (offset < 0 || bytes == 1) {  // 1 バイトの要素は牌の関係のクラス（整数）
        continue;
      }
      const size_t count = weightElements(weight, e, header);
      const bool nan = bytes == 2 ? containsNan<uint16_t>(pack, static_cast<size_t>(offset), count)
                                  : containsNan<uint32_t>(pack, static_cast<size_t>(offset), count);
      require(!nan, weightName(weight, e) + " contains NaN");
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
                                     static_cast<size_t>(table.rows) * shape.hidden()),
              name + " table " + std::to_string(t) + " contains NaN");
    }
    require(!containsNan<uint16_t>(pack, static_cast<size_t>(spec.denseOffset),
                                   static_cast<size_t>(spec.dense) * shape.hidden()),
            name + " dense weights contain NaN");
    require(!containsNan<uint32_t>(pack, static_cast<size_t>(spec.biasOffset), shape.hidden()),
            name + " bias contains NaN");
    if (spec.foldOffset < 0) {
      continue;
    }
    const size_t width = static_cast<size_t>(3) * shape.attention();
    for (int t = 0; t < spec.tables; ++t) {
      const EmbeddingTable& table = spec.table[t];
      require(
          !containsNan<uint32_t>(pack, static_cast<size_t>(table.foldOffset), static_cast<size_t>(table.rows) * width),
          name + " folded table " + std::to_string(t) + " contains NaN");
    }
    require(
        !containsNan<uint32_t>(pack, static_cast<size_t>(spec.foldOffset), static_cast<size_t>(spec.dense + 2) * width),
        name + " folded dense weights or biases contain NaN");
  }
}

}  // namespace

void validateArchitecture(const Architecture& a) {
  const auto& kernels = widthKernels(a.hidden);
  require(!a.reread() || kernels.capability.reread, "the HIP width does not support reread");
  for (float value : {a.epsilon(), a.scale(), a.tau()}) {
    require(std::isfinite(value) && value > 0.0f, "epsilon, scale and tau must be positive and finite");
  }
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
  require(a.candidateFfn > 0 && a.candidateFfn % kFfnChunk == 0,
          "candidate ffn must be a positive multiple of " + std::to_string(kFfnChunk));
  require(a.stages == ordinal(CandidateStages::kOne) || a.stages == ordinal(CandidateStages::kReread),
          "unknown candidate stages " + std::to_string(a.stages));
  require(a.gate > 0 && a.gate % kGateStep == 0, "gate must be a positive multiple of " + std::to_string(kGateStep));
  require(a.gateKinds > 0 && a.gateKinds <= a.roles && a.roles <= kMaxRoles,
          "need 0 < gate kinds <= roles <= " + std::to_string(kMaxRoles));
  require(a.seats >= kMinSeats && a.seats <= kMaxSeats, "seats out of range: " + std::to_string(a.seats));
  require(a.tiles > 0 && a.tiles <= kMaxTiles, "tiles out of range: " + std::to_string(a.tiles));
  require(a.keyCodes > 0 && a.tileRelations > 0 && a.maxEvents >= 0, "counts must be positive");
  require(a.fixed() <= kMaxFixedTokens && a.rowTokens() <= kMaxRowTokens,
          "rows of " + std::to_string(a.fixed()) + " fixed and " + std::to_string(a.rowTokens()) + " tokens exceed " +
              std::to_string(kMaxFixedTokens) + " and " + std::to_string(kMaxRowTokens));
  require(a.rereadCodes() <= kMaxCodes, "code tables exceed " + std::to_string(kMaxCodes) + " rows");
  require(a.opponents() <= kRelationOpponents,
          "relation codes hold the safety of at most " + std::to_string(kRelationOpponents) + " opponents");
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

namespace {

PlanShape checkedPlanShape(const PlanShape& shape) {
  validateArchitecture(shape.architecture);
  require(shape.rows > 0 && shape.events >= 0 && shape.transitions >= shape.rows && shape.actions >= shape.rows,
          "invalid batch capacity");
  // K1 は入力の領域を、32 bit のバイトオフセットで読む。
  require(shape.inputBytes > 0 && shape.inputBytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "input capacity must be positive and stay below the 2 GiB that K1 addresses with 32-bit offsets");
  require(shape.weightBytes >= sizeof(PackHeader) &&
              shape.weightBytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "weight pack capacity must hold the header and stay below 2 GiB");
  return shape;
}

CandidatePackLayout describeCandidatePack(const PlanShape& shape) {
  const auto& a = shape.architecture;
  const auto& options = shape.options;
  const bool count = options.has(PlanOptions::kCount);
  const bool segments = options.has(PlanOptions::kSegments);
  constexpr int features = PlanOptions::kCausal | PlanOptions::kCount | PlanOptions::kSegments;
  require((options.features & ~features) == 0, "unknown V4 plan feature");
  require(count == segments, "count and segments must be enabled together");
  const auto& capability = widthKernels(a.hidden).capability;
  require(count ? capability.core : capability.standard, "unsupported candidate mode for this HIP width");
  require(!options.has(PlanOptions::kCausal) || !a.reread(),
          "V4 causal attention requires the layer-zero all-token path without reread");

  CandidatePackLayout pack;
  const auto requireSpan = [&](int32_t offset, size_t bytes, const char* name) {
    const size_t start = static_cast<size_t>(offset);
    require(start >= sizeof(PackHeader) && start % kPackAlignment == 0 && start <= shape.weightBytes &&
                bytes <= shape.weightBytes - start,
            name);
  };
  if (count) {
    require(options.countFirstCode >= 0 && options.countFirstCode < a.keyCodes, "invalid V4 counting codes");
    require(options.countWidth == v4::kCountWidth, "V4 count width does not match the kernel");
    pack.gate = static_cast<size_t>(a.keyCodes - options.countFirstCode) * v4::kCountWidth;
    pack.projection = static_cast<size_t>(a.hidden) * v4::kCountWidth;
    pack.countWorkspace =
        static_cast<size_t>(shape.rows) * a.tiles * v4::kCountWidth + static_cast<size_t>(shape.transitions) * a.hidden;
    requireSpan(options.countOffset, pack.countBytes(), "V4 count weights are outside the plan capacity");
  } else {
    require(options.countOffset == 0 && options.countFirstCode == 0 && options.countWidth == 0,
            "V4 count data requires counting");
  }
  if (segments) {
    require(!a.reread() && options.segments.valid(a.fixed(), a.tiles), "invalid V4 segment layout");
    pack.segments = static_cast<size_t>(options.segments.tableRows) * 2 * a.attention();
    requireSpan(options.segmentsOffset, pack.segmentsBytes(), "V4 segment weights are outside the plan capacity");
  } else {
    require(options.segmentsOffset == 0 && options.segments.count == 0 && options.segments.codeWidth == 0 &&
                options.segments.tableRows == 0,
            "V4 segment data requires segments");
  }
  if (count && segments) {
    require(
        static_cast<size_t>(options.countOffset) + pack.countBytes() <= static_cast<size_t>(options.segmentsOffset) ||
            static_cast<size_t>(options.segmentsOffset) + pack.segmentsBytes() <=
                static_cast<size_t>(options.countOffset),
        "V4 candidate weight ranges overlap");
  }
  return pack;
}

}  // namespace

Plan::Plan(int device, hipStream_t stream, const PlanShape& shape)
    : device_(device),
      stream_(stream),
      shape_(checkedPlanShape(shape)),
      kernels_(widthKernels(shape_.hidden())),
      candidatePack_(describeCandidatePack(shape_)) {
  const Architecture& a = shape_.architecture;
  planSteps(a);
  if (shape_.options.has(PlanOptions::kSegments)) {
    if (auto classifier = v4::hotCodeClassifier(a.tiles)) {
      v4HotOffsets_ = std::make_unique<v4::HotOffsetCache>(a, shape_.transitions, shape_.options.segments, classifier);
    }
  }
  const size_t pairWidth = static_cast<size_t>(2) * shape.attention();
  regionCapacity_[ordinal(Region::kInput)] = shape.inputBytes;
  regionCapacity_[ordinal(Region::kResidual)] = shape.tokens() * shape.hidden() * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kNormalized)] = shape.tokens() * shape.hidden() * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kQkv0)] = shape.tokens() * 3 * shape.attention() * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kAttention)] = shape.tokens() * shape.attention() * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kRelationCode)] = codeIndex(shape.rows, 0, 0, a.codeBlocks()) * sizeof(uint32_t);
  regionCapacity_[ordinal(Region::kState)] = shape.fixed() * shape.hidden() * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kMemory)] = shape.fixed() * pairWidth * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kStatic)] =
      static_cast<size_t>(shape.transitions) * shape.hidden() * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kTransition)] =
      static_cast<size_t>(shape.transitions) * shape.hidden() * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kValue)] = static_cast<size_t>(shape.transitions) * sizeof(float);
  regionCapacity_[ordinal(Region::kScores)] = static_cast<size_t>(shape.actions) * sizeof(float);
  regionCapacity_[ordinal(Region::kEventKeyValue)] =
      static_cast<size_t>(a.copiedEventLayers()) * shape.events * pairWidth * sizeof(uint16_t);
  regionCapacity_[ordinal(Region::kRereadMemory)] = a.reread() ? shape.tokens() * pairWidth * sizeof(uint16_t) : 0;
  // K2・K4 は n を、K6 は Cross-Attention のキーとバリューを、生の buffer の 32 bit のバイトオフセットで読む。
  require(regionCapacity_[ordinal(Region::kNormalized)] <= kBufferBytes,
          "the normalized region of " + std::to_string(regionCapacity_[ordinal(Region::kNormalized)]) +
              " bytes exceeds the 2 GiB that K2/K4 address with buffer offsets");
  require(regionCapacity_[ordinal(Region::kMemory)] <= kBufferBytes,
          "the memory region of " + std::to_string(regionCapacity_[ordinal(Region::kMemory)]) +
              " bytes exceeds the 2 GiB that K6 addresses with buffer offsets");
  require(regionCapacity_[ordinal(Region::kRereadMemory)] <= kBufferBytes,
          "the reread memory region exceeds the 2 GiB that K9 addresses with buffer offsets");
  planFragmentCopies();
  // K3・K5・K6・K8・K9 は、MLP の重みの写しを生の buffer の 32 bit のバイトオフセットで読む。
  const size_t mlpFragmentBytes = mlpFragmentElements_ * sizeof(uint16_t);
  require(mlpFragmentBytes <= kBufferBytes, "the MLP weight copies of " + std::to_string(mlpFragmentBytes) +
                                                " bytes exceed the 2 GiB that the kernels address with buffer offsets");

  DeviceScope scope(device_);
  hipDeviceProp_t properties;
  PRISM_HIP(hipGetDeviceProperties(&properties, device_));
  const std::string arch = properties.gcnArchName;
  const std::string expected = PRISM_HIP_ARCH;
  require(arch == expected || arch.rfind(expected + ":", 0) == 0,
          "GPU " + std::to_string(device_) + " is " + arch + " but libprism_hip.so was built for " + expected);

  size_t bytes = 0;
  // 因果の折り込み埋め込みは、入力と注意の一時領域を同じ 32 bit の相対位置で読む。両者を隣に置く。
  std::array<int, kRegions> order{};
  for (int r = 0; r < kRegions; ++r) order[r] = r;
  if (shape_.options.has(PlanOptions::kCausal) && saveEventProjection())
    std::swap(order[ordinal(Region::kResidual)], order[ordinal(Region::kAttention)]);
  for (int r : order) {
    regionOffset_[r] = bytes;
    bytes += alignUp(regionCapacity_[r]) + kSlack;
  }
  if (kernels_.sharedProjection && shape_.options.has(PlanOptions::kCausal)) {
    projectionIndexOffset_ = bytes;
    bytes += v4::ProjectionIndex::bytes(shape_.tokens());
  }
  try {
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&workspace_), bytes));
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&weights_), shape.weightBytes + kSlack));
    if (candidatePack_.countWorkspace != 0) {
      PRISM_HIP(
          hipMalloc(reinterpret_cast<void**>(&v4CountWorkspace_), candidatePack_.countWorkspace * sizeof(uint16_t)));
    }
    if (v4HotOffsets_ && kernels_.splitCandidate) {
      PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&v4CandidateAttention_),
                          static_cast<size_t>(shape_.transitions) * a.attention() * sizeof(uint16_t)));
    }
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&relationBias_),
                        a.depth * relationBiasFloats(a.headPairs()) * sizeof(float) + kSlack));
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&qkvFragments_),
                        a.depth * qkvFragmentElements(shape) * sizeof(uint16_t) + kSlack));
    PRISM_HIP(hipMalloc(reinterpret_cast<void**>(&mlpFragments_), mlpFragmentBytes + kSlack));
    PRISM_HIP(
        hipMalloc(reinterpret_cast<void**>(&rowOrder_), static_cast<size_t>(shape.rows) * sizeof(int32_t) + kSlack));
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
  const bool switched =
      hipGetDevice(&previous) == hipSuccess && previous != device_ && hipSetDevice(device_) == hipSuccess;
  release();
  if (switched) {
    (void)hipSetDevice(previous);
  }
}

void Plan::release() noexcept {
  // 解放の前に、この計画が積んだ処理が終わるのを待つ（作業領域を使うカーネルが残らないように）。
  if (workspace_ != nullptr || weights_ != nullptr || v4CountWorkspace_ != nullptr || relationBias_ != nullptr ||
      qkvFragments_ != nullptr || mlpFragments_ != nullptr || rowOrder_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  v4HotOffsets_.reset();
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
  if (pageStorage_ != nullptr) {
    (void)hipFree(pageStorage_);
    pageStorage_ = nullptr;
  }
  if (rowOrder_ != nullptr) {
    (void)hipFree(rowOrder_);
    rowOrder_ = nullptr;
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
  if (v4CountWorkspace_ != nullptr) {
    (void)hipFree(v4CountWorkspace_);
    v4CountWorkspace_ = nullptr;
  }
  if (v4CandidateAttention_ != nullptr) {
    (void)hipFree(v4CandidateAttention_);
    v4CandidateAttention_ = nullptr;
  }
  if (workspace_ != nullptr) {
    (void)hipFree(workspace_);
    workspace_ = nullptr;
  }
}

void Plan::planFragmentCopies() {
  const Architecture& a = shape_.architecture;
  const int h = a.hidden;
  const int att = a.attention();
  fragmentCopy_.fill(-1);
  size_t offset = 0;
  const auto add = [&](Weight weight, int entry, int firstRow, int rows, int columns) {
    fragmentCopy_[weightFirstEntry(weight) + entry] = static_cast<int>(fragmentCopies_.size());
    fragmentCopies_.push_back(FragmentCopy{weight, entry, firstRow, rows, columns, offset});
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
  // 2 つ目からの局・席・牌の層が使うイベントのキーとバリュー（作るのは最後の全トークンの層の K3）: W_qkv の k・v の行 [A, 3A)。
  int firstKv = a.allTokenLayers() + (shape_.options.has(PlanOptions::kCausal) ? 0 : 1);
  for (int l = firstKv; l < a.depth; ++l) {
    add(Weight::kQkv, l, att, 2 * att, h);
  }
  if (a.reread()) {
    add(Weight::kRereadKeyValue, 0, 0, 2 * att, h);
    add(Weight::kRereadOutput, 0, 0, h, att);
  }
  mlpFragmentElements_ = offset;
}

size_t Plan::regionBytes(Region region) const {
  require(ordinal(region) >= 0 && ordinal(region) < kRegions, "unknown workspace region");
  return regionCapacity_[ordinal(region)];
}

void Plan::validateHeader(const PackHeader& header, size_t bytes) const {
  const Architecture& a = shape_.architecture;
  require(std::memcmp(&header.architecture, &a, sizeof(Architecture)) == 0,
          "the weight pack was built for another network architecture than the plan");
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
              header.value(Schema::kRelationSafety) + a.opponents() <= relationWidth,
          "relation safety columns of the " + std::to_string(a.opponents()) + " opponents are out of range");
  const int transitionWidth = header.width(Section::kTransitionCategory);
  require(header.value(Schema::kTransitionCallDiscard) >= 0 &&
              header.value(Schema::kTransitionCallDiscard) < transitionWidth &&
              header.value(Schema::kTransitionMainTile) >= 0 &&
              header.value(Schema::kTransitionMainTile) < transitionWidth,
          "transition columns out of range");
  const int codeWidth = shape_.options.has(PlanOptions::kSegments) ? shape_.options.segments.codeWidth : a.fixed();
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
    // 局・席・牌・イベントは、層 0 の QKV を折り込んだ表を持つ。遷移は持たない。
    const bool folding = kernels_.capability.folded && static_cast<EmbeddingSlot>(e) != EmbeddingSlot::kTransition;
    const size_t foldRow = static_cast<size_t>(3) * shape_.attention() * sizeof(float);
    const auto inside = [&](int32_t offset, size_t size) {
      return offset >= 0 && static_cast<size_t>(offset) % kVectorAlignment == 0 &&
             static_cast<size_t>(offset) + size <= bytes;
    };
    for (int t = 0; t < spec.tables; ++t) {
      const EmbeddingTable& table = spec.table[t];
      require(table.members > 0 && table.members <= kMaxMembers && table.rows > 0, name + " table shape is invalid");
      require(
          table.offset >= 0 && static_cast<size_t>(table.offset) % kVectorAlignment == 0 &&
              static_cast<size_t>(table.offset) + static_cast<size_t>(table.rows) * shape_.hidden() * sizeof(float) <=
                  bytes,
          name + " table " + std::to_string(t) + " is outside the weight pack");
      require(
          folding ? inside(table.foldOffset, static_cast<size_t>(table.rows) * foldRow) : table.foldOffset == -1,
          name + " folded table " + std::to_string(t) + (folding ? " is outside the weight pack" : " must be absent"));
      for (int m = 0; m < table.members; ++m) {
        require(table.column[m] >= 0 && table.column[m] < categoryWidth && table.stride[m] >= 0,
                name + " table member out of range");
      }
    }
    require(spec.dense >= 0 && spec.dense <= kMaxDense, name + " dense count out of range");
    if (spec.dense > 0) {
      require(
          spec.denseOffset >= 0 && static_cast<size_t>(spec.denseOffset) % kVectorAlignment == 0 &&
              static_cast<size_t>(spec.denseOffset) + static_cast<size_t>(spec.dense) * shape_.hidden() * 2 <= bytes,
          name + " dense weights are outside the weight pack");
    }
    require(spec.biasOffset >= 0 && static_cast<size_t>(spec.biasOffset) % kVectorAlignment == 0 &&
                static_cast<size_t>(spec.biasOffset) + shape_.hidden() * sizeof(float) <= bytes,
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

void Plan::uploadWeights(const void* host, size_t bytes, int64_t version) {
  require(bytes <= shape_.weightBytes, "weight pack of " + std::to_string(bytes) + " bytes exceeds the plan capacity " +
                                           std::to_string(shape_.weightBytes));
  require(bytes >= sizeof(PackHeader), "weight pack is smaller than its header");
  require(version >= 0, "weights version must not be negative");
  PackHeader header;
  std::memcpy(&header, host, sizeof(PackHeader));
  const auto& options = shape_.options;
  size_t baseBytes = bytes;
  if (options.has(PlanOptions::kCount)) {
    const size_t offset = options.countOffset;
    require(offset <= bytes && candidatePack_.countBytes() <= bytes - offset, "V4 count weights are missing");
    baseBytes = std::min(baseBytes, offset);
    const char* data = static_cast<const char*>(host) + offset;
    for (size_t i = 0; i < candidatePack_.gate; ++i) {
      float value;
      std::memcpy(&value, data + i * sizeof(value), sizeof(value));
      require(std::isfinite(value), "non-finite V4 count gate");
    }
    data += candidatePack_.gate * sizeof(float);
    for (size_t i = 0; i < 2 * candidatePack_.projection; ++i) {
      uint16_t value;
      std::memcpy(&value, data + i * sizeof(value), sizeof(value));
      require((value & 0x7f80u) != 0x7f80u, "non-finite V4 count projection");
    }
  }
  if (options.has(PlanOptions::kSegments)) {
    const size_t offset = options.segmentsOffset;
    require(offset <= bytes && candidatePack_.segmentsBytes() <= bytes - offset, "V4 segment weights are missing");
    baseBytes = std::min(baseBytes, offset);
    const char* data = static_cast<const char*>(host) + offset;
    for (size_t i = 0; i < candidatePack_.segments; ++i) {
      float value;
      std::memcpy(&value, data + i * sizeof(value), sizeof(value));
      require(std::isfinite(value), "non-finite V4 segment weight");
    }
  }
  validateHeader(header, baseBytes);
  const int32_t tileRelation = header.offset(Weight::kTileRelation, 0);
  if (tileRelation >= 0) {
    const auto* classes = static_cast<const uint8_t*>(host) + tileRelation;
    const size_t count = weightElements(Weight::kTileRelation, 0, header);
    for (size_t i = 0; i < count; ++i) {
      require(classes[i] < kTileRelationClasses, "tile relation class out of range");
    }
  }
  requireNoNan(header, shape_, static_cast<const uint8_t*>(host));
  const size_t embedShared = kernels_.embedBytes ? kernels_.embedBytes(shape_.hidden(), header.embedding) : 0;
  require(embedShared + sizeof(EmbeddingSpec) <= kMaxSharedBytes,
          "K1 needs " + std::to_string(embedShared) + " bytes of LDS for the dense weights");
  DeviceScope scope(device_);
  // 写しと、重みから作るものが済むまでは、重みのない計画として扱う（途中で失敗しても、新しい重みと古い表・写しの組み合わせで
  // forward が回らない）。
  version_ = -1;
  standaloneRows_.resize(static_cast<size_t>(shape_.rows) * header.width(Section::kRowColumn) * sizeof(int32_t));
  header_ = header;
  weightsUsed_ = bytes;
  embedShared_ = embedShared;
  PRISM_HIP(hipMemcpyAsync(weights_, host, bytes, hipMemcpyHostToDevice, stream_));
  // バイアスの表と、QKV・MLP の重みをフラグメントの順に並べた写しは重みだけで決まる。重みを替える手段はここだけなので、公開のたびに
  // 作り直す（古いものは残らない）。pack にない層・重みの分は作らず、カーネルが引くときに止める（relationBias・qkvFragments・
  // mlpFragments）。層 0 の W_qkv は K1 の表に折り込んであり（K2 の層 0 は qkv0 を読む）、K2・K4 用の写しは作らない。
  const Architecture& a = shape_.architecture;
  for (int l = 0; l < a.depth; ++l) {
    if (header_.offset(Weight::kRelationBias, l) >= 0) {
      const RelationBiasArgs args{weight<float>(Weight::kRelationBias, l), relationBias<float>(l), a.heads()};
      launchRelationBias(args, a.headPairs(), stream_);
    }
    if ((l > 0 || !kernels_.capability.folded) && header_.offset(Weight::kQkv, l) >= 0) {
      const QkvFragmentArgs args{weight<uint16_t>(Weight::kQkv, l), qkvFragments<uint16_t>(l)};
      kernels_.qkv(a.hidden, args, a.headPairs(), stream_);
    }
  }
  for (const FragmentCopy& copy : fragmentCopies_) {
    if (header_.offset(copy.weight, copy.entry) < 0) {
      continue;
    }
    Span<const uint16_t> source = weight<uint16_t>(copy.weight, copy.entry);
    const size_t elements = static_cast<size_t>(copy.rows) * copy.columns;
    const size_t first = static_cast<size_t>(copy.firstRow) * copy.columns;
    require(first + elements <= source.limit, weightName(copy.weight, copy.entry) + " is smaller than its copy");
    const MlpFragmentArgs args{Span<const uint16_t>{source.data + first, elements},
                               Span<uint16_t>{mlpFragments_ + copy.offset, elements}, copy.rows, copy.columns};
    launchMlpFragments(args, stream_);
  }
  if (v4HotOffsets_ && v4HotOffsets_->active()) publishHotOffsets(*v4HotOffsets_);
  PRISM_HIP(hipStreamSynchronize(stream_));
  version_ = version;
}

void Plan::chooseHotOffsets(const void* input, const BatchShape& batch) {
  if (!v4HotOffsets_) return;
  const auto* codes = static_cast<const uint8_t*>(input) + batch.section[ordinal(Section::kTransitionCode)];
  if (v4HotOffsets_->observe(codes, batch.transitions, stream_)) publishHotOffsets(*v4HotOffsets_);
}

void Plan::publishHotOffsets(v4::HotOffsetCache& cache) {
  const auto& options = shape_.options;
  const auto* segments = reinterpret_cast<const float*>(weights_ + options.segmentsOffset);
  cache.publish(weight<float>(Weight::kCodeKey, 0), weight<float>(Weight::kCodeValue, 0),
                {{segments, candidatePack_.segments}, options.segments}, stream_);
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
  const int seats = shape_.architecture.seats;
  const int tiles = shape_.architecture.tiles;
  for (int s = 0; s < kSections; ++s) {
    const Section section = static_cast<Section>(s);
    const size_t extent =
        axisCount(sectionAxis(section), batch, seats, tiles) * header_.sectionWidth[s] * sectionElementBytes(section);
    if (batch.section[s] % kVectorAlignment != 0 || batch.section[s] > batch.bytes ||
        extent > batch.bytes - batch.section[s]) {
      throw std::invalid_argument("input section " + std::to_string(s) + " at " + std::to_string(batch.section[s]) +
                                  " (+" + std::to_string(extent) + ") is outside the blob of " +
                                  std::to_string(batch.bytes) + " bytes");
    }
  }
}

template <class T>
Span<T> Plan::regionSpan(Region region, size_t elementOffset) const {
  const size_t capacity = regionCapacity_[ordinal(region)] / sizeof(T);
  require(elementOffset <= capacity, "region offset out of range");
  return Span<T>{reinterpret_cast<T*>(workspace_ + regionOffset_[ordinal(region)]) + elementOffset,
                 capacity - elementOffset};
}

size_t Plan::weightBytes(Weight weight, int entry) const {
  return weightElements(weight, entry, header_) * weightElementBytes(weight);
}

template <class T>
Span<const T> Plan::weight(Weight weight, int entry) const {
  const int32_t offset = header_.offset(weight, entry);
  if (offset < 0) {
    throw std::invalid_argument(weightName(weight, entry) + " is absent from the weight pack");
  }
  return Span<const T>{reinterpret_cast<const T*>(weights_ + offset), weightBytes(weight, entry) / sizeof(T)};
}

template <class T>
Span<const T> Plan::section(const BatchShape& batch, Section section) const {
  const int s = ordinal(section);
  const size_t elements = axisCount(sectionAxis(section), batch, shape_.architecture.seats, shape_.architecture.tiles) *
                          header_.sectionWidth[s];
  return Span<const T>{
      reinterpret_cast<const T*>(workspace_ + regionOffset_[ordinal(Region::kInput)] + batch.section[s]), elements};
}

template <class T>
Span<T> Plan::relationBias(int layer) const {
  require(header_.offset(Weight::kRelationBias, layer) >= 0, "relation bias is absent from the weight pack");
  const size_t floats = relationBiasFloats(shape_.architecture.headPairs());
  return Span<T>{relationBias_ + layer * floats, floats};
}

template <class T>
Span<T> Plan::qkvFragments(int layer) const {
  require(layer > 0 || !kernels_.capability.folded,
          "the layer-0 QKV weights are folded into the K1 tables and have no fragment copy");
  require(header_.offset(Weight::kQkv, layer) >= 0, "QKV projection is absent from the weight pack");
  const size_t elements = qkvFragmentElements(shape_);
  return Span<T>{qkvFragments_ + layer * elements, elements};
}

Span<const uint16_t> Plan::mlpFragments(Weight weight, int entry) const {
  const int index = fragmentCopy_[weightFirstEntry(weight) + entry];
  if (index < 0 || header_.offset(weight, entry) < 0) {
    throw std::invalid_argument(weightName(weight, entry) + " has no MLP fragment copy");
  }
  const FragmentCopy& copy = fragmentCopies_[index];
  return Span<const uint16_t>{mlpFragments_ + copy.offset, static_cast<size_t>(copy.rows) * copy.columns};
}

MlpWeights Plan::mlpWeights(int layer, TokenGroup group) const {
  const int entry = ffnEntry(layer, group);
  return MlpWeights{mlpFragments(Weight::kOutput, layer),   weight<float>(Weight::kOutputBias, layer),
                    weight<float>(Weight::kFfnNorm, entry), mlpFragments(Weight::kUp, entry),
                    weight<float>(Weight::kUpBias, entry),  mlpFragments(Weight::kDown, entry),
                    weight<float>(Weight::kDownBias, entry)};
}

MlpWeights Plan::candidateWeights() const {
  return MlpWeights{
      mlpFragments(Weight::kCandidateOutput, 0),           weight<float>(Weight::kCandidateOutputBias, 0),
      weight<float>(Weight::kFfnNorm, kCandidateFfnEntry), mlpFragments(Weight::kUp, kCandidateFfnEntry),
      weight<float>(Weight::kUpBias, kCandidateFfnEntry),  mlpFragments(Weight::kDown, kCandidateFfnEntry),
      weight<float>(Weight::kDownBias, kCandidateFfnEntry)};
}

DiscardValueWeights Plan::discardWeights() const {
  return DiscardValueWeights{weight<float>(Weight::kDiscardNorm, 0), weight<uint16_t>(Weight::kDiscardHidden, 0),
                             weight<float>(Weight::kDiscardHiddenBias, 0), weight<float>(Weight::kDiscardOutput, 0),
                             weight<float>(Weight::kDiscardExact, 0)};
}

int Plan::segments(int layer, const BatchShape& batch, bool last, MlpSegment (&segment)[2]) const {
  const Architecture& a = shape_.architecture;
  const auto finalNorm = [&](TokenGroup group) {
    return last ? weight<float>(Weight::kFinalNorm, ordinal(group)) : Span<const float>{nullptr, 0};
  };
  const int fixed = a.fixed() * batch.rows;
  if (!a.typed(layer)) {
    segment[0] = MlpSegment{0, fixed, mlpWeights(layer, TokenGroup::kAll), a.ffn(layer, TokenGroup::kAll),
                            finalNorm(TokenGroup::kAll)};
    return 1;
  }
  const int context = (1 + a.seats) * batch.rows;
  segment[0] = MlpSegment{0, context, mlpWeights(layer, TokenGroup::kContext), a.ffn(layer, TokenGroup::kContext),
                          finalNorm(TokenGroup::kContext)};
  segment[1] = MlpSegment{context, fixed - context, mlpWeights(layer, TokenGroup::kTile),
                          a.ffn(layer, TokenGroup::kTile), finalNorm(TokenGroup::kTile)};
  return 2;
}

void Plan::eventKeyValues(const BatchShape& batch, MlpArgs& args) const {
  const Architecture& a = shape_.architecture;
  const size_t width = static_cast<size_t>(2) * shape_.attention();
  for (int l = a.allTokenLayers() + 1; l < a.depth; ++l) {
    const Span<const float> bias = weight<float>(Weight::kQkvBias, l);
    args.eventKeyValue[args.eventKeyValues++] = EventKeyValue{
        weight<float>(Weight::kAttentionNorm, l), mlpFragments(Weight::kQkv, l),
        Span<const float>{bias.data + shape_.attention(), bias.limit - shape_.attention()},
        regionSpan<uint16_t>(Region::kEventKeyValue, static_cast<size_t>(a.fixedIndex(l) - 1) * batch.events * width)};
  }
  if (a.stages == ordinal(CandidateStages::kReread)) {
    const size_t fixed = static_cast<size_t>(a.fixed()) * batch.rows;
    args.eventKeyValue[args.eventKeyValues++] = EventKeyValue{
        weight<float>(Weight::kRereadEventNorm, 0), mlpFragments(Weight::kRereadKeyValue, 0),
        weight<float>(Weight::kRereadKeyValueBias, 0), regionSpan<uint16_t>(Region::kRereadMemory, fixed * width)};
  }
}

TokenLayout Plan::layout(const BatchShape& batch) const {
  return TokenLayout{batch.rows, shape_.architecture.seats, shape_.architecture.tiles};
}

void Plan::countGroupRows(const char* rowColumn, const BatchShape& batch) {
  const int width = header_.width(Section::kRowColumn);
  const int countColumn = header_.value(Schema::kRowEvents);
  const int firstColumn = header_.value(Schema::kRowFirstEvent);
  const int fixed = shape_.architecture.fixed();
  std::fill(std::begin(groupRows_), std::end(groupRows_), 0);
  for (int r = 0; r < batch.rows; ++r) {
    const char* row = rowColumn + static_cast<size_t>(r) * width * sizeof(int32_t);
    int32_t first, events;
    std::memcpy(&first, row + firstColumn * sizeof(int32_t), sizeof(first));
    std::memcpy(&events, row + countColumn * sizeof(int32_t), sizeof(events));
    if (events < 0 || events > shape_.architecture.maxEvents || first < 0 || first > batch.events ||
        events > batch.events - first) {
      throw std::invalid_argument("event span outside row or batch at row " + std::to_string(r));
    }
    ++groupRows_[attentionGroup(fixed, events)];
  }
}

void Plan::launch(Kernel kernel, int layer, const BatchShape& batch) {
  const Architecture& a = shape_.architecture;
  const int hidden = shape_.hidden();
  const int attention = shape_.attention();
  const TokenLayout tokens = layout(batch);
  const int fixed = tokens.fixed() * batch.rows;
  const int allTokens = fixed + batch.events;
  switch (kernel) {
    case Kernel::kEmbed: {
      EmbedArgs args{};
      args.pack = Span<const uint8_t>{reinterpret_cast<const uint8_t*>(weights_), weightsUsed_};
      args.spec = reinterpret_cast<const EmbeddingSpec*>(weights_ + offsetof(PackHeader, embedding));
      args.input =
          Span<const uint8_t>{reinterpret_cast<const uint8_t*>(workspace_ + regionOffset_[ordinal(Region::kInput)]),
                              regionCapacity_[ordinal(Region::kInput)]};
      for (int s = 0; s < kSections; ++s) {
        args.section[s] = batch.section[s];
        args.sectionWidth[s] = header_.sectionWidth[s];
      }
      const size_t starts[kEmbeddingSlots] = {0, static_cast<size_t>(batch.rows),
                                              static_cast<size_t>(1 + tokens.seats) * batch.rows,
                                              static_cast<size_t>(fixed), 0};
      const int counts[kEmbeddingSlots] = {batch.rows, tokens.seats * batch.rows, tokens.tiles * batch.rows,
                                           batch.events, batch.transitions};
      int blocks = 0;
      for (int e = 0; e < kEmbeddingSlots; ++e) {
        if (header_.embedding[e].tables == 0) {
          throw std::invalid_argument("embedding slot " + std::to_string(e) + " is absent from the weight pack");
        }
        args.count[e] = counts[e];
        args.blockStart[e] = blocks;
        blocks += (counts[e] + 63) / 64;
        const bool transition = static_cast<EmbeddingSlot>(e) == EmbeddingSlot::kTransition;
        args.out[e] = regionSpan<uint16_t>(transition ? Region::kStatic : Region::kResidual, starts[e] * hidden);
        args.qkv0[e] =
            transition ? Span<uint16_t>{nullptr, 0} : regionSpan<uint16_t>(Region::kQkv0, starts[e] * 3 * attention);
      }
      args.blockStart[kEmbeddingSlots] = blocks;
      args.epsilon = a.epsilon();
      timed(kernel, [&] {
        if (paged_ && saveEventProjection()) gatherEventFeatures(batch, args);
        if (kernels_.projectedEmbed) {
          Span<uint16_t> normalized[kEmbeddingSlots]{};
          for (int e = 0; e < ordinal(EmbeddingSlot::kTransition); ++e) {
            normalized[e] = regionSpan<uint16_t>(Region::kNormalized, starts[e] * hidden);
          }
          v4::ProjectionIndex index{};
          if (paged_ && kernels_.sharedProjection) {
            index = {reinterpret_cast<int*>(workspace_ + projectionIndexOffset_), allTokens};
            v4::indexLayerInputs(args, header_.embedding, index, stream_);
          }
          kernels_.projectedEmbed(args, weight<float>(Weight::kAttentionNorm, 0), normalized, index, stream_);
        } else {
          kernels_.embed(hidden, args, blocks, embedShared_, stream_);
        }
        if (paged_ && saveEventProjection()) exchangePages(batch, true);
      });
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
      args.codeBlocks = a.codeBlocks();
      args.columns = RelationColumns{header_.width(Section::kRelation),
                                     header_.value(Schema::kRelationTile),
                                     header_.value(Schema::kRelationOwner),
                                     header_.value(Schema::kRelationEvent),
                                     header_.value(Schema::kRelationRiichi),
                                     header_.value(Schema::kRelationLastDiscard),
                                     header_.value(Schema::kRelationLink),
                                     header_.value(Schema::kRelationSafety),
                                     a.opponents()};
      // 同じ起動で、K2・K4 が読む組の順の行の番号も書く（どちらもバッチに 1 回で、K2・K4 の全ての層が読む）。
      RowOrderArgs order{};
      order.rowColumn = args.rowColumn;
      order.rowOrder = Span<int32_t>{rowOrder_, static_cast<size_t>(batch.rows)};
      order.rows = batch.rows;
      order.fixed = tokens.fixed();
      order.rowWidth = args.rowWidth;
      order.rowEvents = args.rowEvents;
      for (int g = 0, start = 0; g < kAttentionGroups; start += groupRows_[g++]) {
        order.groupStart[g] = start;
        order.groupRows[g] = groupRows_[g];
      }
      timed(kernel, [&] {
        if (paged_)
          launchPageRelation(args, pageMap_, stream_);
        else
          launchRelation(args, stream_);
        launchRowOrder(order, stream_);
      });
      return;
    }
    case Kernel::kAttentionAll:
    case Kernel::kAttentionFixed: {
      // 全トークンの層 0 は射影をせず、K1 が書いた qkv0 を読む。ほかの層は n と W_qkv の写しで射影する。
      const bool fixedLayer = kernel == Kernel::kAttentionFixed;
      const bool folded = layer == 0 && kernels_.capability.folded;
      AttentionArgs args{};
      args.o = regionSpan<uint16_t>(Region::kAttention, 0);
      args.codes = regionSpan<const uint32_t>(Region::kRelationCode, 0);
      args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
      args.rowOrder = Span<const int32_t>{rowOrder_, static_cast<size_t>(batch.rows)};
      if (folded) {
        args.qkv0 = regionSpan<const uint16_t>(Region::kQkv0, 0);
      } else {
        args.normalized = regionSpan<const uint16_t>(Region::kNormalized, 0);
        args.qkv = qkvFragments<const uint16_t>(layer);
        args.qkvBias = weight<float>(Weight::kQkvBias, layer);
      }
      args.bias = relationBias<const float>(layer);
      args.layout = tokens;
      args.rowWidth = header_.width(Section::kRowColumn);
      args.rowFirstEvent = header_.value(Schema::kRowFirstEvent);
      args.rowEvents = header_.value(Schema::kRowEvents);
      args.scale = a.scale();
      args.role = fixedLayer ? LayerRole::kFixedTokens : LayerRole::kAllTokens;
      args.codeBlocks = a.codeBlocks();
      // 最初の局・席・牌の層は、最後の全トークンの層の K3 が全てのトークンについて書いた n から、イベントの k・v も射影する。
      // 後の局・席・牌の層は、その K3 が作ったイベントの k・v を写す。
      args.eventKeys = !fixedLayer || a.fixedIndex(layer) == 0 ? EventKeys::kProjected : EventKeys::kCopied;
      if (paged_ && fixedLayer && a.fixedIndex(layer) == 0) {
        args.eventKeys = EventKeys::kCopied;
        args.eventKeyValue = regionSpan<const uint16_t>(Region::kQkv0, 0);
      } else if (args.eventKeys == EventKeys::kCopied) {
        args.eventKeyValue = regionSpan<const uint16_t>(
            Region::kEventKeyValue, static_cast<size_t>(a.fixedIndex(layer) - 1) * batch.events * 2 * attention);
      }
      timed(kernel, [&] {
        if (shape_.options.has(PlanOptions::kCausal) && layer == 0) {
          if (paged_) args.eventKeyValue = {reinterpret_cast<const uint16_t*>(pageMap_), size_t(batch.rows) * 2};
          if (paged_ && kernels_.sharedProjection) {
            v4::ProjectionIndex index{reinterpret_cast<int*>(workspace_ + projectionIndexOffset_), allTokens};
            kernels_.sharedProjection(args, index, regionSpan<uint16_t>(Region::kQkv0, 0), a.headPairs(), groupRows_,
                                      stream_);
          } else {
            kernels_.causal(args, a.headPairs(), groupRows_, stream_);
          }
        } else {
          kernels_.attention(hidden, folded, args, a.headPairs(), groupRows_, stream_);
        }
      });
      return;
    }
    case Kernel::kMlpAll: {
      MlpArgs args{};
      args.x = regionSpan<uint16_t>(Region::kResidual, 0);
      args.o = regionSpan<const uint16_t>(Region::kAttention, 0);
      args.normalized = regionSpan<uint16_t>(Region::kNormalized, 0);
      args.nextNorm = weight<float>(Weight::kAttentionNorm, layer + 1);
      args.segments = 1;
      args.segment[0] = MlpSegment{0, allTokens, mlpWeights(layer, TokenGroup::kAll), a.ffn(layer, TokenGroup::kAll),
                                   Span<const float>{nullptr, 0}};
      // 最後の全トークンの層は、x を局・席・牌の分だけ書く（局・席・牌の層はイベントの残差を読まない）。n は全てのトークンについて書く
      // （最初の局・席・牌の層の Attention がイベントの k・v も射影する）。後の局・席・牌の層と 2 段目が使うイベントのキーとバリューも、
      // その場で作る。
      args.normalizedTokens = allTokens;
      if (a.role(layer + 1) == LayerRole::kAllTokens) {
        args.residualTokens = allTokens;
      } else {
        args.residualTokens = fixed;
        args.firstEvent = fixed;
        eventKeyValues(batch, args);
        if (paged_) {
          // 最初の固定クエリ層の K/V も同じ丸めで作る。因果注意が読み終えた QKV0 に詰める。
          auto bias = weight<float>(Weight::kQkvBias, layer + 1);
          args.eventKeyValue[args.eventKeyValues++] = EventKeyValue{args.nextNorm,
                                                                    mlpFragments(Weight::kQkv, layer + 1),
                                                                    {bias.data + attention, bias.limit - attention},
                                                                    regionSpan<uint16_t>(Region::kQkv0, 0)};
          args.normalizedTokens = fixed;
        }
      }
      args.epsilon = a.epsilon();
      timed(kernel, [&] {
        if (paged_) compactEventInputs(batch);
        kernels_.mlp(hidden, args, stream_);
        if (paged_) exchangePages(batch, false);
      });
      return;
    }
    case Kernel::kMlpFixed: {
      MlpArgs args{};
      args.x = regionSpan<uint16_t>(Region::kResidual, 0);
      args.o = regionSpan<const uint16_t>(Region::kAttention, 0);
      args.normalized = regionSpan<uint16_t>(Region::kNormalized, 0);
      args.nextNorm = weight<float>(Weight::kAttentionNorm, layer + 1);
      args.segments = segments(layer, batch, false, args.segment);
      args.residualTokens = fixed;
      args.normalizedTokens = fixed;
      args.epsilon = a.epsilon();
      timed(kernel, [&] { kernels_.mlp(hidden, args, stream_); });
      return;
    }
    case Kernel::kMlpLast: {
      MlpKvArgs args{};
      args.x = regionSpan<const uint16_t>(Region::kResidual, 0);
      args.o = regionSpan<const uint16_t>(Region::kAttention, 0);
      args.segments = segments(layer, batch, true, args.segment);
      args.keyValue = mlpFragments(Weight::kKeyValue, 0);
      args.keyValueBias = weight<float>(Weight::kKeyValueBias, 0);
      args.state = regionSpan<uint16_t>(Region::kState, 0);
      args.memory = regionSpan<uint16_t>(Region::kMemory, 0);
      args.layout = tokens;
      args.epsilon = a.epsilon();
      timed(kernel, [&] { kernels_.lastMlp(hidden, args, stream_); });
      return;
    }
    case Kernel::kRereadKeyValue: {
      launchRereadKeys(batch);
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
      args.discard = discardWeights();
      args.transition = regionSpan<uint16_t>(Region::kTransition, 0);
      args.value = regionSpan<float>(Region::kValue, 0);
      args.layout = tokens;
      args.transitions = batch.transitions;
      args.categoryWidth = header_.width(Section::kTransitionCategory);
      args.callDiscard = header_.value(Schema::kTransitionCallDiscard);
      args.mainTile = header_.value(Schema::kTransitionMainTile);
      args.codes = a.keyCodes;
      args.discardExactWidth = header_.width(Section::kDiscardExact);
      args.rowWidth = header_.width(Section::kRowColumn);
      args.rowFirstTransition = header_.value(Schema::kRowFirstTransition);
      args.ffn = a.candidateFfn;
      args.gate = a.gate;
      args.keys = a.fixed();
      args.writeValue = !a.reread();
      args.epsilon = a.epsilon();
      args.scale = a.scale();
      const auto& options = shape_.options;
      const auto segmentWeights = [&] {
        const auto* offsets = reinterpret_cast<const float*>(weights_ + options.segmentsOffset);
        return v4::SegmentWeights{{offsets, candidatePack_.segments}, options.segments};
      };
      if (options.has(PlanOptions::kCount)) {
        const size_t gateCount = candidatePack_.gate;
        const size_t matrixCount = candidatePack_.projection;
        const auto* gate = reinterpret_cast<const float*>(weights_ + options.countOffset);
        const auto* tile = reinterpret_cast<const uint16_t*>(gate + gateCount);
        const v4::CountWeights extra{
            {gate, gateCount}, {tile, matrixCount}, {tile + matrixCount, matrixCount}, options.countFirstCode};
        timed(kernel, [&] {
          const auto hot = v4HotOffsets_ ? v4HotOffsets_->prepare(args, stream_) : v4::HotCandidateOffsets{};
          const Span<uint16_t> attentionState{v4CandidateAttention_,
                                              static_cast<size_t>(batch.transitions) * a.attention()};
          v4::launchCandidateCore(hidden, args, extra, segmentWeights(), v4CountWorkspace_, stream_, hot,
                                  attentionState);
        });
      } else {
        timed(kernel, [&] { kernels_.standard(hidden, args, stream_); });
      }
      return;
    }
    case Kernel::kReread: {
      RereadArgs args{};
      args.transition = regionSpan<uint16_t>(Region::kTransition, 0);
      args.value = regionSpan<float>(Region::kValue, 0);
      args.memory = regionSpan<const uint16_t>(Region::kRereadMemory, 0);
      args.category = section<uint8_t>(batch, Section::kTransitionCategory);
      args.code = section<uint8_t>(batch, Section::kTransitionCode);
      args.relation = section<uint8_t>(batch, Section::kRelation);
      args.discardExact = section<float>(batch, Section::kDiscardExact);
      args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
      args.tileRelation = weight<uint8_t>(Weight::kTileRelation, 0);
      args.queryNorm = weight<float>(Weight::kRereadQueryNorm, 0);
      args.query = weight<uint16_t>(Weight::kRereadQuery, 0);
      args.queryBias = weight<float>(Weight::kRereadQueryBias, 0);
      args.codeKey = weight<float>(Weight::kRereadCodeKey, 0);
      args.codeValue = weight<float>(Weight::kRereadCodeValue, 0);
      args.output = mlpFragments(Weight::kRereadOutput, 0);
      args.outputBias = weight<float>(Weight::kRereadOutputBias, 0);
      args.discard = discardWeights();
      args.layout = tokens;
      args.transitions = batch.transitions;
      args.categoryWidth = header_.width(Section::kTransitionCategory);
      args.callDiscard = header_.value(Schema::kTransitionCallDiscard);
      args.mainTile = header_.value(Schema::kTransitionMainTile);
      args.relationWidth = header_.width(Section::kRelation);
      args.relationTile = header_.value(Schema::kRelationTile);
      args.keyCodes = a.keyCodes;
      args.rowWidth = header_.width(Section::kRowColumn);
      args.rowFirstTransition = header_.value(Schema::kRowFirstTransition);
      args.rowFirstEvent = header_.value(Schema::kRowFirstEvent);
      args.rowEvents = header_.value(Schema::kRowEvents);
      args.discardExactWidth = header_.width(Section::kDiscardExact);
      args.gate = a.gate;
      args.epsilon = a.epsilon();
      args.scale = a.scale();
      timed(kernel, [&] { kernels_.reread(hidden, args, stream_); });
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
      args.roles = a.roles;
      for (int r = 0; r < kMaxRoles; ++r) {
        args.composition[r] = header_.composition[r];
        args.gateKind[r] = header_.gate[r];
      }
      args.gateKinds = a.gateKinds;
      args.gate = a.gate;
      args.epsilon = a.epsilon();
      args.tau = a.tau();
      timed(kernel, [&] { kernels_.valueGraph(hidden, args, stream_); });
      return;
    }
    case Kernel::kCount:
      break;
  }
  throw std::invalid_argument("unknown kernel " + std::to_string(ordinal(kernel)));
}

void Plan::launchRereadKeys(const BatchShape& batch) {
  KeyValueArgs args{regionSpan<const uint16_t>(Region::kState, 0), mlpFragments(Weight::kRereadKeyValue, 0),
                    weight<float>(Weight::kRereadKeyValueBias, 0), regionSpan<uint16_t>(Region::kRereadMemory, 0),
                    layout(batch)};
  timed(Kernel::kRereadKeyValue, [&] { kernels_.rereadKeys(shape_.hidden(), args, stream_); });
}

void Plan::planSteps(const Architecture& a) {
  steps_.push_back({Kernel::kEmbed, 0});
  steps_.push_back({Kernel::kRelation, 0});
  for (int l = 0; l < a.allTokenLayers(); ++l) {
    steps_.push_back({Kernel::kAttentionAll, l});
    steps_.push_back({Kernel::kMlpAll, l});
  }
  for (int l = a.allTokenLayers(); l < a.depth; ++l) {
    steps_.push_back({Kernel::kAttentionFixed, l});
    steps_.push_back({l + 1 < a.depth ? Kernel::kMlpFixed : Kernel::kMlpLast, l});
  }
  if (a.stages == ordinal(CandidateStages::kReread)) {
    steps_.push_back({Kernel::kRereadKeyValue, 0});
  }
  steps_.push_back({Kernel::kCandidate, 0});
  if (a.reread()) {
    steps_.push_back({Kernel::kReread, 0});
  }
  steps_.push_back({Kernel::kValueGraph, 0});
}

void Plan::resetTiming() { launches_.fill(0); }

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
    launches_[k] = index + 1;
  }
}

void Plan::run(Kernel kernel, int layer, const BatchShape& batch) {
  validate(batch);
  require(acceptsLayer(kernel, layer, shape_.architecture),
          "kernel " + std::to_string(ordinal(kernel)) + " does not run layer " + std::to_string(layer));
  DeviceScope scope(device_);
  // 行インデックスを読むカーネルは、イベントの区間を検証し、Attention の組ごとの行数を数える。
  if (kernel == Kernel::kRelation || kernel == Kernel::kAttentionAll || kernel == Kernel::kAttentionFixed ||
      kernel == Kernel::kReread) {
    const size_t bytes = static_cast<size_t>(batch.rows) * header_.width(Section::kRowColumn) * sizeof(int32_t);
    const char* device =
        workspace_ + regionOffset_[ordinal(Region::kInput)] + batch.section[ordinal(Section::kRowColumn)];
    PRISM_HIP(hipMemcpyAsync(standaloneRows_.data(), device, bytes, hipMemcpyDeviceToHost, stream_));
    PRISM_HIP(hipStreamSynchronize(stream_));
    countGroupRows(standaloneRows_.data(), batch);
  }
  resetTiming();
  launch(kernel, layer, batch);
}

bool Plan::saveEventProjection() const { return kernels_.capability.folded; }

void Plan::configureCandidateInput(const int* descriptor, size_t count) {
  const auto& segments = shape_.options.segments;
  require(shape_.options.has(PlanOptions::kCausal) && shape_.options.has(PlanOptions::kSegments) && version_ >= 0,
          "candidate input restoration requires published causal segment weights");
  require(count * sizeof(int) ==
              offsetof(CandidateInputLayout, slots) + segments.count * sizeof(CandidateInputLayout::Slot),
          "candidate input descriptor length mismatch");
  CandidateInputLayout layout{};
  std::memcpy(&layout, descriptor, count * sizeof(int));
  auto column = [&](int index, Section section) {
    require(index >= 0 && index < header_.width(section), "candidate input column out of range");
  };
  for (int index : {layout.kind, layout.author, layout.eventTile, layout.declaration})
    column(index, Section::kEventCategory);
  column(layout.river, Section::kTileCategory);
  column(layout.riichi, Section::kSeatCategory);
  for (auto digit : {layout.ukeire, layout.wait, layout.discarded})
    require(digit.weight > 0 && digit.radix > 0, "invalid candidate digit");
  for (int i = 0; i < segments.count; i++) {
    auto slot = layout.slots[i];
    require(slot.seat >= -1 && slot.seat < shape_.architecture.seats, "candidate input seat outside rule");
    require(slot.bonusTile >= -1 && slot.bonusTile < shape_.architecture.tiles, "candidate bonus tile outside rule");
    if (slot.seat != 0) column(slot.tileColumn, Section::kTileCategory);
  }
  candidateInput_ = layout;
  restoreCandidates_ = true;
}

size_t Plan::eventInputWidth() const {
  return header_.width(Section::kEventCategory) + sizeof(uint16_t) * header_.width(Section::kEventScalar) +
         header_.width(Section::kRelation);
}

size_t Plan::basicCodeBytes(int transitions) const {
  return restoreCandidates_ ? size_t(transitions) * shape_.architecture.fixed() : 0;
}

hipError_t Plan::allocateEventPages(int slots) {
  const size_t capacity = slots, batchCapacity = std::min(slots, shape_.events);
  size_t bytes = 0;
  auto reserve = [&](size_t count) {
    size_t at = bytes;
    bytes += alignUp(count);
    return at;
  };
  size_t indices = reserve((shape_.rows + 2 * batchCapacity) * sizeof(int) + batchCapacity * eventInputWidth() +
                           basicCodeBytes(shape_.transitions));
  size_t raw = reserve(capacity * eventInputWidth());
  size_t in = reserve(saveEventProjection() ? capacity * 3 * shape_.attention() * sizeof(uint16_t) : 0);
  size_t norm = reserve(capacity * shape_.hidden() * sizeof(uint16_t));
  size_t kv = reserve(capacity * shape_.hidden() * sizeof(uint16_t) * (shape_.architecture.depth - 2));
  size_t scratchWidth = saveEventProjection() ? shape_.attention() : shape_.hidden();
  size_t compact = reserve(batchCapacity * scratchWidth * sizeof(uint16_t));
  char* storage = nullptr;
  hipError_t status = hipMalloc(&storage, bytes);
  if (status != hipSuccess) return status;
  pageStorage_ = storage;
  pageSlots_ = slots;
  pageMap_ = reinterpret_cast<int*>(storage + indices);
  pageFeatures_ = reinterpret_cast<uint8_t*>(storage + raw);
  pageInput_ = saveEventProjection() ? storage + in : nullptr;
  pageFixedKeys_ = storage + norm;
  pageExtraKeys_ = storage + kv;
  compactAttention_ = reinterpret_cast<uint16_t*>(storage + compact);
  return hipSuccess;
}

std::array<int64_t, 4> Plan::configureEventPages(int remaining, int events, double reserveFraction) {
  const auto& a = shape_.architecture;
  require(shape_.options.has(PlanOptions::kCausal) && a.allTokenLayers() == 1 && !a.reread(),
          "event cache needs a prefix-independent event trunk");
  require(remaining > 0 && events >= 0 && reserveFraction >= 0 && reserveFraction <= 1 && !pageStorage_,
          "invalid event page allocation request");
  DeviceScope scope(device_);
  size_t before, after, total;
  PRISM_HIP(hipMemGetInfo(&before, &total));
  if (events > 0) {
    PRISM_HIP(allocateEventPages(events));
    PRISM_HIP(hipMemGetInfo(&after, &total));
    return {int64_t(before), int64_t(after), int64_t(total), pageSlots_};
  }
  size_t headroom = static_cast<size_t>(total * reserveFraction);
  size_t budget = before > headroom ? (before - headroom) / remaining : 0;
  size_t scratchWidth = saveEventProjection() ? shape_.attention() : shape_.hidden();
  size_t fixed = alignUp((shape_.rows + 2 * size_t(shape_.events)) * sizeof(int) +
                         size_t(shape_.events) * eventInputWidth() + basicCodeBytes(shape_.transitions)) +
                 alignUp(size_t(shape_.events) * scratchWidth * sizeof(uint16_t)) + 4 * kPackAlignment;
  size_t projectionWidth = saveEventProjection() ? 3 * shape_.attention() : 0;
  size_t perEvent = (projectionWidth + (a.depth - 1) * shape_.hidden()) * sizeof(uint16_t) + eventInputWidth();
  size_t slots = budget > fixed ? (budget - fixed) / perEvent : 0;
  slots = std::min(slots, size_t(std::numeric_limits<int>::max()));
  while (slots >= size_t(shape_.events) && slots > 0) {
    hipError_t status = allocateEventPages(static_cast<int>(slots));
    if (status == hipSuccess) break;
    if (status != hipErrorOutOfMemory) PRISM_HIP(status);
    hipGetLastError();
    if (slots == size_t(shape_.events)) break;
    slots = std::max(size_t(shape_.events), slots * 3 / 4);
  }
  PRISM_HIP(hipMemGetInfo(&after, &total));
  return {int64_t(before), int64_t(after), int64_t(total), pageSlots_};
}

void Plan::uploadPageMap(const BatchShape& batch, const int* map, int misses, int slotCount, size_t transferBytes) {
  const auto& a = shape_.architecture;
  require(shape_.options.has(PlanOptions::kCausal) && a.allTokenLayers() == 1 && !a.reread(),
          "event cache needs a prefix-independent event trunk");
  require(misses >= 0 && misses <= batch.events, "invalid event delta");
  require(slotCount > 0 && slotCount >= batch.events && (!pageStorage_ || slotCount == pageSlots_),
          "invalid event cache capacity");
  const size_t mapBytes = (size_t(batch.rows) + batch.events + misses) * sizeof(int);
  const int* slots = map + batch.rows;
  for (int e = 0; e < batch.events; ++e)
    if (static_cast<unsigned>(slots[e]) >= static_cast<unsigned>(slotCount))
      throw std::invalid_argument("event slot outside cache");
  for (int e = 0; e < misses; ++e)
    if (static_cast<unsigned>(slots[batch.events + e]) >= static_cast<unsigned>(batch.events))
      throw std::invalid_argument("event delta index outside batch");
  if (!pageStorage_) PRISM_HIP(allocateEventPages(slotCount));
  require(!transferBytes ||
              transferBytes == mapBytes + size_t(misses) * eventInputWidth() + basicCodeBytes(batch.transitions),
          "event transfer size mismatch");
  PRISM_HIP(hipMemcpyAsync(pageMap_, map, transferBytes ? transferBytes : mapBytes, hipMemcpyHostToDevice, stream_));
  auto* device = reinterpret_cast<uint8_t*>(workspace_ + regionOffset_[ordinal(Region::kInput)]);
  EventInputColumns columns{};
  columns.categoryWidth = header_.width(Section::kEventCategory);
  columns.scalarBytes = sizeof(uint16_t) * header_.width(Section::kEventScalar);
  columns.relationWidth = header_.width(Section::kRelation);
  columns.categories = device + batch.section[ordinal(Section::kEventCategory)];
  columns.scalars = device + batch.section[ordinal(Section::kEventScalar)];
  columns.relations = device + batch.section[ordinal(Section::kRelation)] +
                      size_t(batch.rows) * shape_.architecture.fixed() * columns.relationWidth;
  const int* deviceSlots = pageMap_ + batch.rows;
  const int* missing = deviceSlots + batch.events;
  const auto* raw = transferBytes ? reinterpret_cast<const uint8_t*>(missing + misses) : nullptr;
  restoreEventInputs(columns, pageFeatures_, deviceSlots, missing, batch.events, misses, raw, stream_);
  if (transferBytes && restoreCandidates_) {
    auto at = [&](Section section) { return device + batch.section[ordinal(section)]; };
    CandidateInput input{reinterpret_cast<const int*>(at(Section::kRowColumn)),
                         columns.categories,
                         at(Section::kTileCategory),
                         at(Section::kSeatCategory),
                         at(Section::kRelation) + size_t(batch.rows) * (a.fixed() - a.tiles) * columns.relationWidth,
                         raw + size_t(misses) * eventInputWidth(),
                         at(Section::kTransitionCode),
                         header_.width(Section::kRowColumn),
                         header_.value(Schema::kRowFirstEvent),
                         header_.value(Schema::kRowEvents),
                         header_.value(Schema::kRowFirstTransition),
                         header_.value(Schema::kRowTransitions),
                         columns.categoryWidth,
                         header_.width(Section::kTileCategory),
                         header_.width(Section::kSeatCategory),
                         columns.relationWidth,
                         header_.value(Schema::kRelationTile),
                         a.tiles,
                         a.seats,
                         a.fixed()};
    restoreCandidateCodes(input, candidateInput_, shape_.options.segments, batch.rows, stream_);
  }
  denseEvents_ = batch.events;
}

void Plan::uploadInputs(const void* input, const BatchShape& batch, bool sparse) {
  char* device = workspace_ + regionOffset_[ordinal(Region::kInput)];
  const auto* host = static_cast<const char*>(input);
  auto copy = [&](size_t first, size_t end) {
    if (end > first)
      PRISM_HIP(hipMemcpyAsync(device + first, host + first, end - first, hipMemcpyHostToDevice, stream_));
  };
  if (!sparse) {
    copy(0, batch.bytes);
    return;
  }
  copy(0, batch.section[ordinal(Section::kEventCategory)]);
  size_t relation = batch.section[ordinal(Section::kRelation)];
  copy(relation, relation + size_t(batch.rows) * shape_.architecture.fixed() * header_.width(Section::kRelation));
  size_t transitions = batch.section[ordinal(Section::kTransitionCategory)];
  if (!restoreCandidates_) {
    copy(transitions, batch.bytes);
    return;
  }
  size_t codes = batch.section[ordinal(Section::kTransitionCode)];
  // 候補の基本符号は配置表の後ろに連続している。遷移ごとに短い H2D を発行しない。
  copy(transitions, codes);
  copy(batch.section[ordinal(Section::kDiscardExact)], batch.bytes);
}

void Plan::gatherEventFeatures(const BatchShape& batch, EmbedArgs& args) {
  if (!batch.events) return;
  auto* input = reinterpret_cast<uint8_t*>(workspace_ + regionOffset_[ordinal(Region::kInput)]);
  auto* categories = input + batch.section[ordinal(Section::kEventCategory)];
  auto* scalars = reinterpret_cast<uint16_t*>(input + batch.section[ordinal(Section::kEventScalar)]);
  int categoryWidth = header_.width(Section::kEventCategory), scalarWidth = header_.width(Section::kEventScalar);
  size_t catBytes = size_t(batch.events) * categoryWidth, scalarBytes = size_t(batch.events) * scalarWidth * 2;
  // K1 の間はまだ読まない注意の出力領域に詰め、埋め込みに渡す入力のうちイベントの 2 列だけをそこに向ける。
  size_t scalarOffset = (catBytes + 1) & ~size_t(1);
  require(scalarOffset + scalarBytes <= regionBytes(Region::kAttention), "event features exceed scratch");
  size_t offset = regionOffset_[ordinal(Region::kAttention)] - regionOffset_[ordinal(Region::kInput)];
  require(offset + scalarOffset + scalarBytes <= std::numeric_limits<uint32_t>::max(),
          "embedding offset exceeds 32 bits");
  auto* scratch = input + offset;
  auto* packedScalars = reinterpret_cast<uint16_t*>(scratch + scalarOffset);
  gatherPageFeatures(categories, scalars, categoryWidth, scalarWidth, scratch, packedScalars,
                     pageMap_ + batch.rows + denseEvents_, batch.events, stream_);
  args.input.limit = offset + scalarOffset + scalarBytes;
  args.section[ordinal(Section::kEventCategory)] = offset;
  args.section[ordinal(Section::kEventScalar)] = offset + scalarOffset;
}

void Plan::exchangePages(const BatchShape& batch, bool embedding) {
  const size_t fixed = static_cast<size_t>(batch.rows) * shape_.architecture.fixed();
  const int hidden = shape_.hidden();
  PageColumns columns{};
  auto add = [&](void* current, void* pages, int bytes) { columns.column[columns.count++] = {current, pages, bytes}; };
  if (embedding) {
    int width = 3 * shape_.attention();
    add(regionSpan<uint16_t>(Region::kQkv0, fixed * width).data, pageInput_, width * 2);
  } else {
    add(regionSpan<uint16_t>(Region::kQkv0, 0).data, pageFixedKeys_, hidden * 2);
    if (shape_.architecture.depth > 2)
      add(regionSpan<uint16_t>(Region::kEventKeyValue, 0).data, pageExtraKeys_, hidden * 2);
  }
  const int* slots = pageMap_ + batch.rows;
  exchangeEventPages(slots, slots + denseEvents_, denseEvents_, batch.events, columns, stream_);
}

void Plan::compactEventInputs(const BatchShape& batch) {
  const int fixed = batch.rows * shape_.architecture.fixed();
  const int* misses = pageMap_ + batch.rows + denseEvents_;
  if (kernels_.sharedProjection) {
    // K2 は代表だけを読めるが、K3 は各行の残差を上書きする。計算する行だけを先に独立させる。
    const auto* owners = reinterpret_cast<const int*>(workspace_ + projectionIndexOffset_);
    v4::restoreEmbedRows(regionSpan<uint16_t>(Region::kResidual, 0), owners, misses, fixed, batch.events, stream_);
  }
  if (!batch.events) return;
  auto compact = [&](Region region, int width) {
    auto* dense = regionSpan<uint16_t>(region, static_cast<size_t>(fixed) * width).data;
    compactEventRows(dense, compactAttention_, misses, batch.events, width, stream_);
    PRISM_HIP(hipMemcpyAsync(dense, compactAttention_, size_t(batch.events) * width * sizeof(uint16_t),
                             hipMemcpyDeviceToDevice, stream_));
  };
  // K3 に渡す残差と注意出力を、同じ未計算イベントの順に詰める。
  if (!saveEventProjection()) compact(Region::kResidual, shape_.hidden());
  compact(Region::kAttention, shape_.attention());
}

void Plan::forward(const void* input, const BatchShape& batch, void* scores, int64_t version, const int* eventMap,
                   int misses, int slots, size_t transferBytes) {
  validate(batch);
  if (version != version_) {
    throw std::invalid_argument("weight pack version " + std::to_string(version_) + " is stale; the network is at " +
                                std::to_string(version));
  }
  DeviceScope scope(device_);
  countGroupRows(static_cast<const char*>(input) + batch.section[ordinal(Section::kRowColumn)], batch);
  chooseHotOffsets(input, batch);
  BatchShape compact = batch;
  uploadInputs(input, batch, transferBytes != 0);
  if (eventMap) {
    uploadPageMap(batch, eventMap, misses, slots, transferBytes);
    compact.events = misses;
  }
  resetTiming();
  paged_ = eventMap != nullptr;
  try {
    for (const auto& step : steps_) {
      bool delta = step.kernel == Kernel::kMlpAll || (step.kernel == Kernel::kEmbed && saveEventProjection());
      launch(step.kernel, step.layer, delta ? compact : batch);
    }
  } catch (...) {
    paged_ = false;
    throw;
  }
  paged_ = false;
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
