// WeightPack のヘッダーと、ネットワークの形の記述・入力の領域・重みの種類・埋め込みの取り決め。並びは Java の HipPack・HipArchitecture・
// HipSchema・HipSection・HipWeight・HipEmbedding と同じ。ヘッダーは little-endian の int32 の並びで、JNI はネットワークの中身を知らずに
// バイト列だけを受け取る。
#pragma once

#include <cstddef>
#include <cstdint>

namespace prism {

constexpr size_t kPackAlignment = 256;
constexpr int kMaxDepth = 3;
constexpr int kMaxRoles = 16;
constexpr int kMaxTables = 8;
constexpr int kMaxMembers = 8;
constexpr int kMaxDense = 24;
constexpr int kMaxRowTokens = 160;   // 1 行のトークンの上限（K0・K2・K4 が LDS に置く。Java の HipArchitecture.MAX_ROW_TOKENS）
constexpr int kMaxFixedTokens = 48;  // 1 行の局・席・牌のトークンの上限（K4・K6 のタイル。Java の HipArchitecture.MAX_FIXED_TOKENS）

// バックボーンの層が、全てのトークンを扱うか局・席・牌のトークンだけを扱うか（Java の LayerRole）。
enum class LayerRole : int32_t { kAllTokens, kFixedTokens };

// FFN を通るトークンの組（Java の TokenGroup）。トークンの軸の上で、局・席は [0, (1 + 席の数)·R)、牌はその後ろの 牌の数·R 個。
enum class TokenGroup : int32_t { kAll, kContext, kTile, kCount };

// 候補層の段（Java の CandidateStages）。
enum class CandidateStages : int32_t { kOne, kReread };

constexpr int kTokenGroups = static_cast<int>(TokenGroup::kCount);

// バックボーンの 1 層。深さより後ろの層は role −1、typed 0、FFN の幅 0。
struct LayerSpec {
  int32_t role;               // LayerRole
  int32_t typed;              // 局・席と牌で別の FFN なら 1
  int32_t ffn[kTokenGroups];  // 組ごとの FFN の幅（通らない組は 0）
};

// ネットワークの形の記述（Java の HipArchitecture.encode と同じ並び）。Plan と WeightPack のヘッダーがそれぞれ持ち、ビットまで同じかを
// 照合する。
struct Architecture {
  int32_t hidden;
  int32_t epsilonBits;
  int32_t scaleBits;
  int32_t tauBits;
  int32_t depth;
  LayerSpec layer[kMaxDepth];
  int32_t candidateFfn;
  int32_t stages;  // CandidateStages
  int32_t gate;
  int32_t gateKinds;
  int32_t roles;
  int32_t seats;
  int32_t tiles;
  int32_t keyCodes;
  int32_t tileRelations;
  int32_t maxEvents;

  float epsilon() const { return __builtin_bit_cast(float, epsilonBits); }
  float scale() const { return __builtin_bit_cast(float, scaleBits); }
  float tau() const { return __builtin_bit_cast(float, tauBits); }
  int attention() const { return hidden / 2; }
  int heads() const { return attention() / 16; }
  int headPairs() const { return heads() / 2; }
  LayerRole role(int l) const { return static_cast<LayerRole>(layer[l].role); }
  bool reread() const { return stages == static_cast<int32_t>(CandidateStages::kReread); }
  // 行ごとの局・席・牌の数 F。
  int fixed() const { return 1 + seats + tiles; }
  int opponents() const { return seats - 1; }
  // 2 段目のコードの表の行数（遷移のコード、牌の関係なしの 1 行、牌の関係の種類）。
  int rereadCodes() const { return keyCodes + 1 + tileRelations; }
  int rowTokens() const { return fixed() + maxEvents; }
  // 1 行の関係のコードを、16 トークンずつのブロックに分けたときのブロックの数。
  int codeBlocks() const { return (rowTokens() + 15) / 16; }
  int allTokenLayers() const {
    int count = 0;
    while (count < depth && role(count) == LayerRole::kAllTokens) {
      ++count;
    }
    return count;
  }
  int fixedLayers() const { return depth - allTokenLayers(); }
  // 局・席・牌の層の中の番号（最初の局・席・牌の層が 0）。
  int fixedIndex(int l) const { return l - allTokenLayers(); }
  // 局・席・牌の層のうち、最後の全トークンの層の K3 が作ったイベントのキーとバリューを写して使う層の数。最初の局・席・牌の層は、
  // 自分で n から射影する。イベントのキーとバリューの領域は、層 l の分を fixedIndex(l) − 1 番目に置く。
  int copiedEventLayers() const { return fixedLayers() - 1; }
  bool typed(int l) const { return layer[l].typed != 0; }
  int ffn(int l, TokenGroup group) const { return layer[l].ffn[static_cast<int>(group)]; }
};

constexpr int kArchitectureInts = static_cast<int>(sizeof(Architecture) / sizeof(int32_t));
static_assert(sizeof(Architecture) == (5 + kMaxDepth * (2 + kTokenGroups) + 10) * sizeof(int32_t),
              "Architecture must match HipArchitecture.INTS");

// 入力の領域の中で、各値がどの列にあるか（Java の HipSchema）。
enum class Schema : int {
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
  kTransitionCallDiscard,
  kTransitionMainTile,
  kRelationRows,
  kCount
};

// 入力の領域（Java の HipSection）。
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

// 重みの種類（Java の HipWeight）。
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
  kRereadQueryNorm,
  kRereadQuery,
  kRereadQueryBias,
  kRereadEventNorm,
  kRereadKeyValue,
  kRereadKeyValueBias,
  kRereadCodeKey,
  kRereadCodeValue,
  kRereadOutput,
  kRereadOutputBias,
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

// 重みの項目の引き方（Java の HipWeight.Index）。LAYER は層ごと、FFN は (層, 組) ごとの ffnEntry と候補層の kCandidateFfnEntry、
// GROUP は組ごとに項目を持つ。
enum class WeightIndex : int { kSingle, kLayer, kFfn, kGroup };

constexpr int kCandidateFfnEntry = kMaxDepth * kTokenGroups;

constexpr int ffnEntry(int layer, TokenGroup group) { return layer * kTokenGroups + static_cast<int>(group); }

constexpr int indexEntries(WeightIndex index) {
  switch (index) {
    case WeightIndex::kSingle:
      return 1;
    case WeightIndex::kLayer:
      return kMaxDepth;
    case WeightIndex::kFfn:
      return kCandidateFfnEntry + 1;
    case WeightIndex::kGroup:
      return kTokenGroups;
  }
  return 0;
}

constexpr WeightIndex weightIndex(Weight weight) {
  switch (weight) {
    case Weight::kAttentionNorm:
    case Weight::kQkv:
    case Weight::kQkvBias:
    case Weight::kRelationBias:
    case Weight::kOutput:
    case Weight::kOutputBias:
      return WeightIndex::kLayer;
    case Weight::kFfnNorm:
    case Weight::kUp:
    case Weight::kUpBias:
    case Weight::kDown:
    case Weight::kDownBias:
      return WeightIndex::kFfn;
    case Weight::kFinalNorm:
      return WeightIndex::kGroup;
    default:
      return WeightIndex::kSingle;
  }
}

// 候補層の 2 段目の重みか。2 段目のない形は、これらの重みを持たない。
constexpr bool rereadWeight(Weight weight) {
  return static_cast<int>(weight) >= static_cast<int>(Weight::kRereadQueryNorm) &&
         static_cast<int>(weight) <= static_cast<int>(Weight::kRereadOutputBias);
}

// ヘッダーの位置の表で、その重みの最初の項目の番号。
constexpr int weightFirstEntry(Weight weight) {
  int first = 0;
  for (int w = 0; w < static_cast<int>(weight); ++w) {
    first += indexEntries(weightIndex(static_cast<Weight>(w)));
  }
  return first;
}

constexpr int kWeightEntries = weightFirstEntry(Weight::kCount);

// 手の値の作り方（Java の HipComposition）。
enum class Composition : int { kDiscardValue, kRiichiGate, kResponseGate, kTurnGate, kZero, kCount };

// 埋め込みを持つトークンの種類（Java の HipEmbedding.Slot）と、密な特徴の出どころ（HipEmbedding.Source）。
enum class EmbeddingSlot : int { kRound, kSeat, kTile, kEvent, kTransition, kCount };
enum class DenseSource : int { kFlag, kScalar, kExact };

constexpr int kSchemaInts = static_cast<int>(Schema::kCount);
constexpr int kSections = static_cast<int>(Section::kCount);
constexpr int kWeightSlots = static_cast<int>(Weight::kCount);
constexpr int kEmbeddingSlots = static_cast<int>(EmbeddingSlot::kCount);

// いくつかのカテゴリの列をまとめて引く表 1 つ。FP32 の [rows, H] で、行 = Σ 値(column[i])·stride[i]。foldOffset は、同じ行について
// 層 0 の QKV を折り込んだ表 T' = T·diag(g0)·W_qkv0ᵀ（FP32 の [rows, 3A]、列はヘッドの対の並び）の位置で、遷移の表では −1。
struct EmbeddingTable {
  int32_t offset;
  int32_t rows;
  int32_t members;
  int32_t column[kMaxMembers];
  int32_t stride[kMaxMembers];
  int32_t foldOffset;
};

// トークンの種類 1 つの埋め込み（Java の HipEmbedding.encode と同じ並び）。tables == 0 の種類は埋め込みを持たない。foldOffset は層 0 の
// QKV を折り込んだ W'_d = W_d·diag(g0)·W_qkv0ᵀ（FP32 の [dense, 3A]）の位置で、その後ろに b' = b·diag(g0)·W_qkv0ᵀ と b_qkv（FP32 の
// [2, 3A]）が続く。列はヘッドの対の並び（foldColumn）。遷移では −1。
struct EmbeddingSpec {
  int32_t category;  // Section
  int32_t scalar;    // Section か −1
  int32_t exact;     // Section か −1
  int32_t tables;
  int32_t dense;
  int32_t denseOffset;  // W_d^T [dense, H] BF16
  int32_t biasOffset;   // [H] FP32
  int32_t foldOffset;   // [dense + 2, 3A] FP32 か −1
  EmbeddingTable table[kMaxTables];
  int32_t source[kMaxDense];  // DenseSource << 16 | 列
};

// 層 0 の QKV の列（q・k・v を並べた [3A] の中の位置。column / A が q・k・v のどれかを表す）を、ヘッドの対ごとの並びでの位置に移す
// （Java の HipEmbedding.foldColumn と同じ）。ヘッドの対 p の q・k・v の 32 列ずつ（ヘッド 2p と 2p + 1 の 16 次元ずつ）が 96p から続く
// ので、K2 でヘッドの対を受け持つワークグループは、1 トークンにつき続いた 192 バイトを読めばよい。
constexpr int foldColumn(int column, int attention) {
  return 96 * ((column % attention) / 32) + 32 * (column / attention) + column % 32;
}

// ヘッダーは形の記述から始まり、マジックナンバーもフォーマットのバージョンも持たない。WeightPack は公開のたびにチェックポイントから作り、
// 保存しないため。ヘッダーの int の数は、ライブラリを読み込むときに HipLibrary が Java の値と照合する。
struct PackHeader {
  Architecture architecture;
  int32_t schema[kSchemaInts];
  int32_t sectionWidth[kSections];
  int32_t composition[kMaxRoles];
  int32_t gate[kMaxRoles];
  int32_t weight[kWeightEntries];  // 重みの項目ごとのバイト位置（weightFirstEntry + 項目の番号）。無い項目は −1
  EmbeddingSpec embedding[kEmbeddingSlots];

  int value(Schema field) const { return schema[static_cast<int>(field)]; }
  int width(Section section) const { return sectionWidth[static_cast<int>(section)]; }
  int32_t offset(Weight slot, int entry) const { return weight[weightFirstEntry(slot) + entry]; }
};

constexpr int kHeaderInts = static_cast<int>(sizeof(PackHeader) / sizeof(int32_t));
static_assert(sizeof(PackHeader) % sizeof(int32_t) == 0, "the pack header is a sequence of int32");
static_assert(sizeof(EmbeddingSpec) == (8 + kMaxTables * (4 + 2 * kMaxMembers) + kMaxDense) * sizeof(int32_t),
              "EmbeddingSpec must match HipEmbedding.INTS");

// 入力の領域の要素のバイト数（Java の HipSection.element）。
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
    case Weight::kRereadQuery:
    case Weight::kRereadKeyValue:
    case Weight::kRereadOutput:
    case Weight::kDiscardHidden:
    case Weight::kGateHidden:
      return 2;
    default:
      return 4;
  }
}

}  // namespace prism
