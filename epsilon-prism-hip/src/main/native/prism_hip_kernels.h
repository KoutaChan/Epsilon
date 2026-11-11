// カーネルの引数と起動の関数。呼ぶのは計画（prism_hip_plan.cc）だけ。行列は BF16 の [out, in]、バイアスとゲインは FP32。Span の limit は
// 指す先から領域の終わりまでの要素の数で、境界検査つきのビルドだけが読む。
#pragma once

#include "prism_hip_common.h"
#include "prism_hip_format.h"

namespace prism {

// グローバルメモリの 1 つの領域（境界検査つきのビルドが使う上限つき）。
template <class T>
struct Span {
  T* data;
  size_t limit;  // data から読み書きしてよい要素の数
};

// 1 層の MLP の重み（K3・K5・K6 の後半で共通）。行列はフラグメントの順に並べた写しで、W [N, K] を [N/16, K/16, 2, 16, 8] の形に置く
// （置き方は prism_hip_mlp.h の FragmentWeights。作るのは重みの公開のときで、引数は MlpFragmentArgs）。
struct MlpWeights {
  Span<const uint16_t> outWeight;   // W_o [H, A] の写し
  Span<const float> outBias;        // [H]
  Span<const float> normGain;       // [H]
  Span<const uint16_t> upWeight;    // W1 [F, H] の写し
  Span<const float> upBias;         // [F]
  Span<const uint16_t> downWeight;  // W2 [H, F] の写し
  Span<const float> downBias;       // [H]
};

// 関係の属性の列（入力の契約で決まる）。safeties は相手の数で、安全の列は safety から続く。
struct RelationColumns {
  int width;
  int tile;
  int owner;
  int event;
  int riichi;
  int lastDiscard;
  int link;
  int safety;
  int safeties;
};

// トークンの並び: 行 r の位置 p（局 0、席 1..S、牌 S+1..F−1、イベント F..）が、トークンの軸の何番目かを求める。軸は型ごとの領域を
// 続けたもの（局 R・席 S·R・牌 T·R・イベント N_E）で、入力の blob のトークンの軸と同じ。
struct TokenLayout {
  int rows;
  int seats;
  int tiles;

  __host__ __device__ __forceinline__ int fixed() const { return 1 + seats + tiles; }

  __device__ __forceinline__ int index(int row, int firstEvent, int position) const {
    if (position == 0) {
      return row;
    }
    if (position <= seats) {
      return rows + seats * row + position - 1;
    }
    if (position < fixed()) {
      return (1 + seats) * rows + tiles * row + position - 1 - seats;
    }
    return fixed() * rows + firstEvent + position - fixed();
  }

  // 位置の型（0 局、1 席、2 牌、3 イベント）。
  __device__ __forceinline__ int type(int position) const {
    return position == 0 ? 0 : position <= seats ? 1 : position < fixed() ? 2 : 3;
  }

  // 局・席・牌のトークンの番号 t（< F·R）から、行と行の中の位置を求める。
  __device__ __forceinline__ void locate(int token, int& row, int& position) const {
    if (token < rows) {
      row = token;
      position = 0;
    } else if (token < (1 + seats) * rows) {
      const int u = token - rows;
      row = u / seats;
      position = 1 + u % seats;
    } else {
      const int u = token - (1 + seats) * rows;
      row = u / tiles;
      position = 1 + seats + u % tiles;
    }
  }
};

// K1: 型ごとの埋め込み。ワークグループは 64 トークン（ウェーブごとに 8 トークン。lane は x の H/32 列と qkv0 の 3A/32 列を受け持つ）。
struct EmbedArgs {
  Span<const uint8_t> pack;             // WeightPack 全体（表・W_d・b・折り込んだ表はバイトの位置で引く）
  const EmbeddingSpec* spec;            // 埋め込みの記述（デバイスに写した pack のヘッダーの中）
  Span<const uint8_t> input;            // blob 全体
  size_t section[kSections];            // 領域の先頭（バイト）
  int sectionWidth[kSections];
  int count[kEmbeddingSlots];           // 型ごとのトークンの数
  int blockStart[kEmbeddingSlots + 1];  // 型ごとの最初のワークグループ
  Span<uint16_t> out[kEmbeddingSlots];  // 型ごとの出力 [count, H]
  Span<uint16_t> qkv0[kEmbeddingSlots];  // 型ごとの層 0 の q・k・v [count, 3A]（ヘッドの対の順。遷移の型には書かない）
  float epsilon;
};

// K0: 1 行。
struct RelationArgs {
  Span<const uint8_t> relation;      // 関係の属性 [トークン, 幅]
  Span<const int32_t> rowColumn;     // 行のインデックス [行, 幅]
  Span<const uint8_t> tileRelation;  // [牌²]
  Span<uint32_t> codes;              // 関係のコード [行, タイル, タイル, 256]
  TokenLayout layout;
  int rowWidth;
  int rowFirstEvent;
  int rowEvents;
  int codeBlocks;  // 1 行にある 16 トークンのタイルの数（Architecture::codeBlocks）
  RelationColumns columns;
};

// 関係のバイアスの表を作る（重みの公開のとき）。1 回の起動で 1 層のヘッドの対を受け持つ。
struct RelationBiasArgs {
  Span<const float> rho;  // [関係の行, ヘッド]
  Span<float> table;      // その層のバイアスの表 [ヘッドの対, kBiasEntries, 2]（prism_hip_relation.h）
  int heads;
};

// QKV をフラグメントの順に並べた写しを作る（重みの公開のとき）。1 回の起動で 1 層のヘッドの対を受け持つ。
struct QkvFragmentArgs {
  Span<const uint16_t> qkv;  // W_qkv [3A, H]
  Span<uint16_t> fragments;  // その層の写し [ヘッドの対, H/16, 2, 3, 2, 16, 8]（置き方は k2_attention.hip）
};

// MLP の重みをフラグメントの順に並べた写しを作る（重みの公開のとき）。1 回の起動で 1 つの行列。
struct MlpFragmentArgs {
  Span<const uint16_t> weight;  // W [rows, columns]
  Span<uint16_t> fragments;     // 写し [rows/16, columns/16, 2, 16, 8]（MlpWeights）
  int rows;
  int columns;
};

// K2・K4 の行の組。組 g のカーネルは attentionWaves(g) ウェーブで動き、LDS に 16·attentionWaves(g) トークン分を持つ。受け持つのは、
// 16 トークンのタイルの数 ⌈T/16⌉ が attentionWaves(g) 以下で、前の組に入らない行（1 ウェーブがキーのタイルを 1 つ作り、クエリのタイルを
// 1 つ回す）。計画は組ごとに起動し、起動の大きさは組の中の行の数で決まる（使われないウェーブを持たない）。行の算術は組に依らない
// （k2_attention.hip）。
constexpr int kAttentionGroups = 4;

__host__ __device__ constexpr int attentionWaves(int group) {
  return group == 0 ? 4 : group == 1 ? 6 : group == 2 ? 8 : 10;
}

// 行の組を返す。T = min(F + max(イベント数, 0), kMaxRowTokens) は、K2・K4 が行のトークンを切り詰めるのと同じ式。
__host__ __device__ constexpr int attentionGroup(int fixed, int events) {
  const int tokens = fixed + (events > 0 ? events : 0);
  const int blocks = ((tokens < kMaxRowTokens ? tokens : kMaxRowTokens) + 15) / 16;
  return blocks <= attentionWaves(0) ? 0 : blocks <= attentionWaves(1) ? 1 : blocks <= attentionWaves(2) ? 2 : 3;
}

static_assert(16 * attentionWaves(kAttentionGroups - 1) >= kMaxRowTokens, "the last row group holds the longest row");

// K0 の後: 行の番号を組の順に並べる（組の中は行の順）。組の先頭と行の数は、ホストが行のインデックスから数えたもの。
struct RowOrderArgs {
  Span<const int32_t> rowColumn;  // 行のインデックス [行, 幅]
  Span<int32_t> rowOrder;         // [行]: 組 g の i 番目の行が groupStart[g] + i
  int rows;
  int fixed;
  int rowWidth;
  int rowEvents;
  int groupStart[kAttentionGroups];
  int groupRows[kAttentionGroups];
};

// K4 がイベントのキーとバリューをどう作るか。最初の局・席・牌の層は、最後の全トークンの層の K3 が全てのトークンについて書いた n から
// 自分で射影する。後の局・席・牌の層は、最後の全トークンの層の K3 が作った eventKeyValue を写す。
enum class EventKeys : int { kProjected, kCopied };

// K2（全トークンの層）・K4（局・席・牌の層）: ヘッドの対 × 組の行（起動は組ごと）。K2 の層 0 は射影をせず qkv0 を読む（n・qkv・qkvBias
// は読まない）。K4 は、eventKeys が kProjected なら全てのタイルを n から射影する。kCopied なら局・席・牌のタイルだけを射影し、イベントの
// k・v は eventKeyValue から写す。
struct AttentionArgs {
  Span<const uint16_t> normalized;  // Attention の入力 n = ⌊(x·r)·g⌉ [トークン, H]（書くのは K3。写す側の K4 は局・席・牌だけを読む）
  Span<uint16_t> o;                 // [トークン, A]
  Span<const uint32_t> codes;       // 関係のコード [行, タイル, タイル, 256]（K0）
  Span<const int32_t> rowColumn;    // 行のインデックス [行, 幅]
  Span<const int32_t> rowOrder;     // 組の順に並べた行の番号 [行]（RowOrderArgs）
  Span<const uint16_t> qkv;         // W_qkv をフラグメントの順に並べた写し（重みの公開のときに作る。QkvFragmentArgs::fragments）
  Span<const float> qkvBias;        // [3A]
  Span<const float> bias;           // その層のバイアスの表 [ヘッドの対, kBiasEntries, 2]（重みの公開のときに作る）
  TokenLayout layout;
  int rowWidth;
  int rowFirstEvent;
  int rowEvents;
  int groupStart;  // この起動の組が rowOrder の何番目から始まるか（launchAttention が組ごとに入れる）
  float scale;
  Span<const uint16_t> qkv0;           // 層 0 の q・k・v [トークン, 3A]（ヘッドの対の順。K1 が書き、読むのは K2 の層 0 だけ）
  LayerRole role;                      // 全トークンの層は K2、局・席・牌の層は K4
  EventKeys eventKeys;                 // K4 がイベントのキーとバリューをどう作るか
  Span<const uint16_t> eventKeyValue;  // その層のイベントのキーとバリュー [イベント, 2A]（読むのは kCopied の K4 だけ）
  int codeBlocks;                      // 1 行にある関係のコードのタイルの数
};

// K3・K5 のトークンの軸の 1 区間と、その区間の重み。O 射影は層で共通、FFN は組ごと。
struct MlpSegment {
  int firstToken;
  int tokens;
  MlpWeights weights;
  int ffn;
  Span<const float> finalNorm;  // 最後の正規化のゲイン [H]（K5 だけ）
};

// 最後の全トークンの層の K3 が、イベントのトークンの x から作るキーとバリューの 1 組: [K V] = ⌊⌊(x·r)·g⌉·W_kvᵀ + b_kv⌉。読む側は、
// 2 つ目からの局・席・牌の層の Attention（その層の Attention の前のゲインと、W_qkv の k・v の行）と、2 段目のイベントの記憶（2 段目の
// イベントのゲインと、キーとバリューの射影）。
struct EventKeyValue {
  Span<const float> gain;       // [H]
  Span<const uint16_t> weight;  // W_kv [2A, H] をフラグメントの順に並べた写し
  Span<const float> bias;       // [2A]
  Span<uint16_t> destination;   // [イベント, 2A]
};

// 1 回の K3 が作るイベントのキーとバリューの組の上限（2 つ目からの局・席・牌の層の分が深さ − 2 まで、それに 2 段目の分）。
constexpr int kMaxEventKeyValues = kMaxDepth - 1;

// K3: kMlpRows トークン（k3_mlp.hip）。ワークグループ b は区間 s（blockStart[s] ≤ b < blockStart[s + 1]）の
// トークン firstToken + 48·(b − blockStart[s]) から受け持つ。
struct MlpArgs {
  Span<uint16_t> x;            // [トークン, H] 読んで上書きする
  Span<const uint16_t> o;      // [トークン, A]
  Span<uint16_t> normalized;   // 次の層の Attention の入力 n [トークン, H]
  Span<const float> nextNorm;  // 次の層の Attention のゲイン [H]
  int segments;                // 区間の数（1 か 2）
  MlpSegment segment[2];
  int blockStart[3];     // 区間ごとの最初のワークグループ（launchMlp が区間のトークンの数から入れる）
  int residualTokens;    // x を書き戻すのはトークンの軸の先頭からこの数まで。その先の x は前の値のまま残る
  int normalizedTokens;  // n を書くのはトークンの軸の先頭からこの数まで
  int eventKeyValues;    // イベントのキーとバリューの組の数（持つのは最後の全トークンの層だけで、ほかは 0）
  EventKeyValue eventKeyValue[kMaxEventKeyValues];
  int firstEvent;        // トークンの軸でイベントが始まる位置（F·R。組を持つときだけ読む）
  float epsilon;
};

// K5: 局・席・牌の kMlpRows トークン。最後の層の MLP → 最後の正規化 y → Cross-Attention のキーとバリュー。区間の選び方は K3 と同じ。
struct MlpKvArgs {
  Span<const uint16_t> x;  // [トークン, H]（先頭 F·R を読む）
  Span<const uint16_t> o;  // [トークン, A]
  int segments;
  MlpSegment segment[2];
  int blockStart[3];  // launchMlpKv が入れる
  Span<const uint16_t> keyValue;   // W_kv [2A, H] の写し（MlpWeights と同じ置き方）
  Span<const float> keyValueBias;  // [2A]
  Span<uint16_t> state;            // y [F·R, H]
  Span<uint16_t> memory;           // [R, F, 2A]
  TokenLayout layout;
  float epsilon;
};

// K8: 局・席・牌の 48 トークンのワークグループ。最後の正規化の後の y から、2 段目の局・席・牌のキーとバリュー
// [K V] = ⌊y·W_kvᵀ + b_kv⌉ を作る。
struct KeyValueArgs {
  Span<const uint16_t> state;   // y [F·R, H]
  Span<const uint16_t> weight;  // W_kv [2A, H] をフラグメントの順に並べた写し
  Span<const float> bias;       // [2A]
  Span<uint16_t> memory;        // [R, F, 2A]
  TokenLayout layout;
};

// B の重み（K6 と K9 が同じ式で計算する）。
struct DiscardValueWeights {
  Span<const float> norm;        // [H]
  Span<const uint16_t> hidden;   // W_B [G, H]
  Span<const float> hiddenBias;  // c_B [G]
  Span<const float> output;      // wB [G]
  Span<const float> exact;       // uBx [x^B の特徴]
};

// K6: 48 遷移。
struct CandidateArgs {
  Span<const uint16_t> statics;    // t_s [遷移, H]
  Span<const uint16_t> state;      // y [F·R, H]
  Span<const uint16_t> memory;     // [R, F, 2A]
  Span<const uint8_t> category;    // 遷移のフィールド [遷移, 幅]
  Span<const uint8_t> code;        // キーのコード [遷移, F]
  Span<const float> discardExact;  // B に直接足す特徴 x^B [遷移, 幅]
  Span<const int32_t> rowColumn;   // 行のインデックス [行, 幅]
  Span<const uint16_t> tileEntry;  // [H, H]
  Span<const float> queryNorm;     // [H]
  Span<const uint16_t> query;      // [A, H]
  Span<const float> queryBias;     // [A]
  Span<const float> codeKey;       // [コード, A]
  Span<const float> codeValue;     // [コード, A]
  MlpWeights weights;              // 候補の O 射影と FFN
  DiscardValueWeights discard;
  Span<uint16_t> transition;  // z [遷移, H]（value が false なら z1）
  Span<float> value;          // B [遷移]
  TokenLayout layout;
  int transitions;
  int categoryWidth;
  int callDiscard;
  int mainTile;
  int codes;
  int discardExactWidth;
  int rowWidth;
  int rowFirstTransition;
  int ffn;
  int gate;
  int keys;         // 行のキーの数 F（31 用と 39 用のどちらのカーネルを使うかを決める）
  bool writeValue;  // B を書くなら true。2 段目のある形は false で、z1 を書いて終わる
  float epsilon;
  float scale;
};

// K9: 48 遷移。z1 を読み、局・席・牌とイベントを読み直した z を書き、B を書く。
struct RereadArgs {
  Span<uint16_t> transition;           // z1 を読み z を書く [遷移, H]
  Span<float> value;                   // B [遷移]
  Span<const uint16_t> memory;         // REREAD_MEMORY（[R, F, 2A] の後に [イベント, 2A]）
  Span<const uint8_t> category;        // 遷移のフィールド [遷移, 幅]
  Span<const uint8_t> code;            // 遷移のキーのコード [遷移, F]
  Span<const uint8_t> relation;        // 関係の属性 [トークン, 幅]（イベントの牌）
  Span<const float> discardExact;      // B に直接足す特徴 x^B [遷移, 幅]
  Span<const int32_t> rowColumn;       // 行のインデックス [行, 幅]
  Span<const uint8_t> tileRelation;    // 牌の関係のクラス [牌²]
  Span<const float> queryNorm;         // [H]
  Span<const uint16_t> query;          // [A, H]
  Span<const float> queryBias;         // [A]
  Span<const float> codeKey;           // [2 段目のコード, A]
  Span<const float> codeValue;         // [2 段目のコード, A]
  Span<const uint16_t> output;         // 出力の射影 W_O [H, A] をフラグメントの順に並べた写し
  Span<const float> outputBias;        // [H]
  DiscardValueWeights discard;
  TokenLayout layout;
  int transitions;
  int categoryWidth;
  int callDiscard;
  int mainTile;
  int relationWidth;
  int relationTile;
  int keyCodes;  // 遷移のコードの数（イベントのキーのコードはこの後ろに続く）
  int rowWidth;
  int rowFirstTransition;
  int rowFirstEvent;
  int rowEvents;
  int discardExactWidth;
  int gate;
  float epsilon;
  float scale;
};

// K7: 16 行。
struct ValueGraphArgs {
  Span<const uint16_t> transition;      // z [遷移, H]
  Span<const float> value;              // B [遷移]
  Span<const uint16_t> state;           // y [F·R, H]（局の領域の行 r が h）
  Span<const int32_t> rowColumn;        // [行, 幅]
  Span<const int32_t> actionColumn;     // [手, 幅]
  Span<const float> actionExact;        // [手, 幅]
  Span<const float> gateNorm;           // [H]
  Span<const uint16_t> gateHidden;      // [G, 3H]
  Span<const float> gateHiddenBias;     // [G]
  Span<const float> kindOffset;         // [種類, G]
  Span<const float> exactProjection;    // U_k [種類, G, x^G の特徴]
  Span<const float> gateOutput;         // [種類, G]
  Span<const float> gateOutputBias;     // [種類]
  Span<const float> exactOutput;        // u_k [種類, x^G の特徴]
  Span<float> scores;                   // [手]
  int rows;
  int rowWidth;
  int rowFirstAction;
  int rowActions;
  int actionWidth;
  int actionRow;
  int actionFirstTransition;
  int actionTransitions;
  int actionRole;
  int actionBase;
  int actionRepresentative;
  int actionExactWidth;
  int roles;
  int composition[kMaxRoles];
  int gateKind[kMaxRoles];
  int gateKinds;
  int gate;
  float epsilon;
  float tau;
};

// 起動の関数（hidden は特化した幅。同期はしない）。
// sharedBytes はワークグループの動的な LDS（embedSharedBytes）。
void launchEmbed(int hidden, const EmbedArgs& args, int blocks, size_t sharedBytes, hipStream_t stream);
// K1 の動的な LDS のバイト数。型ごとに W_d [dense, H]（BF16）を置き、折り込む型では W'_d [dense, 3A]（FP32）も置く。8 特徴の組で
// まとめて読む W_d の行の分（組の外の行も読むが使わない）も含め、全ての型の最大を返す。
size_t embedSharedBytes(int hidden, const EmbeddingSpec (&spec)[kEmbeddingSlots]);
void launchRelation(const RelationArgs& args, hipStream_t stream);
void launchRelationBias(const RelationBiasArgs& args, int headPairs, hipStream_t stream);
void launchQkvFragments(int hidden, const QkvFragmentArgs& args, int headPairs, hipStream_t stream);
void launchRowOrder(const RowOrderArgs& args, hipStream_t stream);
// 組ごとに 1 回起動する（行の数が 0 の組は起動しない）。groupRows は RowOrderArgs と同じ、組ごとの行の数。folded は K2 の層 0（qkv0 を
// 読む）。カーネルは args.role で選ぶ。
void launchAttention(int hidden, bool folded, const AttentionArgs& args, int headPairs,
                     const int (&groupRows)[kAttentionGroups], hipStream_t stream);
void launchMlpFragments(const MlpFragmentArgs& args, hipStream_t stream);
void launchMlp(int hidden, const MlpArgs& args, hipStream_t stream);
void launchMlpKv(int hidden, const MlpKvArgs& args, hipStream_t stream);
void launchKeyValue(int hidden, const KeyValueArgs& args, hipStream_t stream);
void launchCandidate(int hidden, const CandidateArgs& args, hipStream_t stream);
void launchReread(int hidden, const RereadArgs& args, hipStream_t stream);
void launchValueGraph(int hidden, const ValueGraphArgs& args, hipStream_t stream);

}  // namespace prism
