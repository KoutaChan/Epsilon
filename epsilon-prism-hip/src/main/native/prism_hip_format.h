// WeightPack の頭と、入力の区画・重みの置き場・埋め込みの約束（Java の HipPack・HipSchema・HipSection・HipWeight・HipEmbedding と同じ並び）。
// 頭は int32 の little-endian で、JNI は網を知らずにバイト列だけを受け取る。
#pragma once

#include <cstddef>
#include <cstdint>

namespace prism {

constexpr int32_t kPackMagic = 0x31574850;  // "PHW1"
constexpr size_t kPackAlignment = 256;
constexpr int kMaxDepth = 3;
constexpr int kMaxRoles = 8;
constexpr int kMaxTables = 8;
constexpr int kMaxMembers = 8;
constexpr int kMaxDense = 24;
constexpr int kMaxRowTokens = 160;  // K2・K4 の LDS の行（Java の HipShape.MAX_ROW_TOKENS）
constexpr int kMaxFixedTokens = 48;  // K4 の LDS の問い合わせの行（Java の HipShape.MAX_FIXED_TOKENS）

// 入力の約束（Java の HipSchema）。
enum class Schema : int {
  kSeats,
  kTiles,
  kRowFirstEvent,
  kRowEvents,
  kRowFirstAction,
  kRowActions,
  kRowFirstTransition,
  kRowTransitions,
  kActionRow,
  kActionFirstTransition,
  kActionTransitions,
  kActionRole,
  kActionBase,
  kActionRepresentative,
  kRelationTile,
  kRelationOwner,
  kRelationEvent,
  kRelationRiichi,
  kRelationLastDiscard,
  kRelationLink,
  kRelationSafety,
  kRelationSafeties,
  kTransitionCallDiscard,
  kTransitionMainTile,
  kCodes,
  kRelationRows,
  kGateKinds,
  kRoles,
  kTauBits,
  kScaleBits,
  kCount
};

// 入力の区画（Java の HipSection）。
enum class Section : int {
  kRoundCategory,
  kRoundScalar,
  kSeatCategory,
  kSeatScalar,
  kTileCategory,
  kEventCategory,
  kEventScalar,
  kRelation,
  kTransitionCategory,
  kTransitionScalar,
  kTransitionCode,
  kDiscardExact,
  kActionColumn,
  kActionExact,
  kRowColumn,
  kCount
};

// 重みの置き場（Java の HipWeight）。
enum class Weight : int {
  kTileRelation,
  kAttentionNorm,
  kQkv,
  kQkvBias,
  kRelationBias,
  kOutput,
  kOutputBias,
  kFfnNorm,
  kUp,
  kUpBias,
  kDown,
  kDownBias,
  kFinalNorm,
  kTileEntry,
  kQueryNorm,
  kQuery,
  kQueryBias,
  kKeyValue,
  kKeyValueBias,
  kCodeKey,
  kCodeValue,
  kCandidateOutput,
  kCandidateOutputBias,
  kCandidateFfnNorm,
  kCandidateUp,
  kCandidateUpBias,
  kCandidateDown,
  kCandidateDownBias,
  kDiscardNorm,
  kDiscardHidden,
  kDiscardHiddenBias,
  kDiscardOutput,
  kDiscardExact,
  kGateNorm,
  kGateHidden,
  kGateHiddenBias,
  kGateKindOffset,
  kGateExactProjection,
  kGateOutput,
  kGateOutputBias,
  kGateExactOutput,
  kCount
};

// 手の値の作り方（Java の HipComposition）。
enum class Composition : int { kDiscardValue, kRiichiGate, kResponseGate, kTurnGate, kZero, kCount };

// 埋め込みの置き場（Java の HipEmbedding.Slot）と特徴の元（HipEmbedding.Source）。
enum class EmbeddingSlot : int { kRound, kSeat, kTile, kEvent, kTransition, kCount };
enum class DenseSource : int { kFlag, kScalar, kExact };

constexpr int kSchemaInts = static_cast<int>(Schema::kCount);
constexpr int kSections = static_cast<int>(Section::kCount);
constexpr int kWeightSlots = static_cast<int>(Weight::kCount);
constexpr int kEmbeddingSlots = static_cast<int>(EmbeddingSlot::kCount);

// 畳んだ組の表 1 つ: FP32 の [rows, H]。行 = Σ 値(column[i])·stride[i]。
struct EmbeddingTable {
  int32_t offset;
  int32_t rows;
  int32_t members;
  int32_t column[kMaxMembers];
  int32_t stride[kMaxMembers];
};

// 1 つの型の埋め込み（Java の HipEmbedding.encode と同じ並び）。tables == 0 は置き場なし。
struct EmbeddingSpec {
  int32_t category;  // Section
  int32_t scalar;    // Section か −1
  int32_t exact;     // Section か −1
  int32_t tables;
  int32_t dense;
  int32_t denseOffset;  // W_d^T [dense, H] BF16
  int32_t biasOffset;   // [H] FP32
  int32_t reserved;
  EmbeddingTable table[kMaxTables];
  int32_t source[kMaxDense];  // DenseSource << 16 | 列
};

struct PackHeader {
  int32_t magic;
  int32_t ints;
  int32_t schema[kSchemaInts];
  int32_t sectionWidth[kSections];
  int32_t composition[kMaxRoles];
  int32_t gate[kMaxRoles];
  int32_t weight[kWeightSlots][kMaxDepth];  // バイトの位置、なしは −1
  EmbeddingSpec embedding[kEmbeddingSlots];

  int value(Schema field) const { return schema[static_cast<int>(field)]; }
  int width(Section section) const { return sectionWidth[static_cast<int>(section)]; }
};

constexpr int kHeaderInts = static_cast<int>(sizeof(PackHeader) / sizeof(int32_t));
static_assert(sizeof(PackHeader) % sizeof(int32_t) == 0, "the pack header is a sequence of int32");
static_assert(sizeof(EmbeddingSpec) == (8 + kMaxTables * (3 + 2 * kMaxMembers) + kMaxDense) * sizeof(int32_t),
              "EmbeddingSpec must match HipEmbedding.INTS");

// 区画の要素のバイト数（Java の HipSection.element）。
constexpr int sectionElementBytes(Section section) {
  switch (section) {
    case Section::kRoundCategory:
    case Section::kSeatCategory:
    case Section::kTileCategory:
    case Section::kEventCategory:
    case Section::kRelation:
    case Section::kTransitionCategory:
    case Section::kTransitionCode:
      return 1;
    case Section::kRoundScalar:
    case Section::kSeatScalar:
    case Section::kEventScalar:
    case Section::kTransitionScalar:
      return 2;
    case Section::kDiscardExact:
    case Section::kActionColumn:
    case Section::kActionExact:
    case Section::kRowColumn:
      return 4;
    case Section::kCount:
      break;
  }
  return 0;
}

// 重みの要素のバイト数（Java の HipWeight.element）。
constexpr int weightElementBytes(Weight weight) {
  switch (weight) {
    case Weight::kTileRelation:
      return 1;
    case Weight::kQkv:
    case Weight::kOutput:
    case Weight::kUp:
    case Weight::kDown:
    case Weight::kTileEntry:
    case Weight::kQuery:
    case Weight::kKeyValue:
    case Weight::kCandidateOutput:
    case Weight::kCandidateUp:
    case Weight::kCandidateDown:
    case Weight::kDiscardHidden:
    case Weight::kGateHidden:
      return 2;
    default:
      return 4;
  }
}

constexpr bool weightLayered(Weight weight) {
  switch (weight) {
    case Weight::kAttentionNorm:
    case Weight::kQkv:
    case Weight::kQkvBias:
    case Weight::kRelationBias:
    case Weight::kOutput:
    case Weight::kOutputBias:
    case Weight::kFfnNorm:
    case Weight::kUp:
    case Weight::kUpBias:
    case Weight::kDown:
    case Weight::kDownBias:
      return true;
    default:
      return false;
  }
}

}  // namespace prism
