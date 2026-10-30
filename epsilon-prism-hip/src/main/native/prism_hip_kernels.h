// カーネルの引数と起動の口。計画（prism_hip_plan.cc）だけが呼ぶ。行列は BF16 の [out, in]、偏りと利得は FP32。Span の limit はその
// 指し先から区画の終わりまでの要素の数で、検査つきの組み立てだけが読む。
#pragma once

#include "prism_hip_common.h"
#include "prism_hip_format.h"

namespace prism {

// 全体メモリの 1 つの区画（検査つきの組み立ての上限つき）。
template <class T>
struct Span {
  T* data;
  size_t limit;  // data から読み書きしてよい要素の数
};

// 1 層の MLP の重み（K3・K5・K6 の後半で共通）。行列は断片の順の写し（W [N, K] を [N/16, K/16, 2, 16, 8] に並べたもの。
// prism_hip_mlp.h の FragmentWeights の置き方。重みの公開のときに作る。MlpFragmentArgs）。
struct MlpWeights {
  Span<const uint16_t> outWeight;   // W_o [H, A] の写し
  Span<const float> outBias;        // [H]
  Span<const float> normGain;       // [H]
  Span<const uint16_t> upWeight;    // W1 [F, H] の写し
  Span<const float> upBias;         // [F]
  Span<const uint16_t> downWeight;  // W2 [H, F] の写し
  Span<const float> downBias;       // [H]
};

// 関係の属性の列（入力の約束から）。
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

// トークンの並び: 行 r の位置 p（局 0、席 1..S、牌 S+1..F−1、出来事 F..）のトークンの軸の番号。軸は型ごとの区画を続けたもの
// （局 R・席 S·R・牌 T·R・出来事 N_E）で、入力の blob のトークンの軸と同じ。
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

  // 位置の型（0 局、1 席、2 牌、3 出来事）。
  __device__ __forceinline__ int type(int position) const {
    return position == 0 ? 0 : position <= seats ? 1 : position < fixed() ? 2 : 3;
  }

  // 局・席・牌のトークンの番号 t（< F·R）の行と行の中の位置。
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

// K1: 型ごとの埋め込み。作業群は 64 トークン（波ごとに 8 トークン、lane は H/32 列）。
struct EmbedArgs {
  Span<const uint8_t> pack;             // WeightPack 全体（表・W_d・b はバイトの位置で引く）
  const EmbeddingSpec* spec;            // デバイスの頭の埋め込み（pack の中）
  Span<const uint8_t> input;            // blob 全体
  size_t section[kSections];            // 区画の先頭（バイト）
  int sectionWidth[kSections];
  int count[kEmbeddingSlots];           // 型ごとのトークンの数
  int blockStart[kEmbeddingSlots + 1];  // 型ごとの最初の作業群
  Span<uint16_t> out[kEmbeddingSlots];  // 型ごとの出力 [count, H]
  Span<uint16_t> normalized[kEmbeddingSlots];  // 型ごとの層 0 の注意の入力 n [count, H]（遷移は書かない）
  Span<const float> attentionNorm;             // 層 0 の注意の利得 [H]
  float epsilon;
};

// K0: 1 行。
struct RelationArgs {
  Span<const uint8_t> relation;      // 関係の属性 [トークン, 幅]
  Span<const int32_t> rowColumn;     // 行の索引 [行, 幅]
  Span<const uint8_t> tileRelation;  // [牌²]
  Span<uint32_t> codes;              // 関係の符号 [行, 塊, 塊, 256]
  TokenLayout layout;
  int rowWidth;
  int rowFirstEvent;
  int rowEvents;
  RelationColumns columns;
};

// 偏りの表（重みの公開のとき）: 1 層の頭の対。
struct RelationBiasArgs {
  Span<const float> rho;  // [関係の行, 頭]
  Span<float> table;      // その層の偏りの表 [頭の対, kBiasEntries, 2]（prism_hip_relation.h）
  int heads;
};

// QKV の断片の写し（重みの公開のとき）: 1 層の頭の対。
struct QkvFragmentArgs {
  Span<const uint16_t> qkv;  // W_qkv [3A, H]
  Span<uint16_t> fragments;  // その層の写し [頭の対, H/16, 2, 3, 2, 16, 8]（k2_attention.hip の置き方）
};

// MLP の重みの断片の順の写し（重みの公開のとき）: 1 つの行列。
struct MlpFragmentArgs {
  Span<const uint16_t> weight;  // W [rows, columns]
  Span<uint16_t> fragments;     // 写し [rows/16, columns/16, 2, 16, 8]（MlpWeights）
  int rows;
  int columns;
};

// K2・K4: 頭の対 × 行。
struct AttentionArgs {
  Span<const uint16_t> normalized;  // 注意の入力 n = ⌊(x·r)·g⌉ [トークン, H]（K1・K3 が書く）
  Span<uint16_t> o;                 // [トークン, A]
  Span<const uint32_t> codes;       // 関係の符号 [行, 塊, 塊, 256]（K0）
  Span<const int32_t> rowColumn;    // 行の索引 [行, 幅]
  Span<const uint16_t> qkv;         // W_qkv の断片の写し（重みの公開のときに作る。QkvFragmentArgs::fragments）
  Span<const float> qkvBias;        // [3A]
  Span<const float> bias;           // その層の偏りの表 [頭の対, kBiasEntries, 2]（重みの公開のときに作る）
  TokenLayout layout;
  int rowWidth;
  int rowFirstEvent;
  int rowEvents;
  float scale;
};

// K3: kMlpRows トークン（k3_mlp.hip）。
struct MlpArgs {
  Span<uint16_t> x;           // [トークン, H] 読んで上書きする
  Span<const uint16_t> o;     // [トークン, A]
  Span<uint16_t> normalized;  // 次の層の注意の入力 n [トークン, H]
  MlpWeights weights;
  Span<const float> nextNorm;  // 次の層の注意の利得 [H]
  int tokens;
  int ffn;
  float epsilon;
};

// K5: 局・席・牌の kMlpRows トークン。最後の層の MLP → 最後の正規化 y → 交差注意の鍵と値。
struct MlpKvArgs {
  Span<const uint16_t> x;  // [トークン, H]（先頭 F·R を読む）
  Span<const uint16_t> o;  // [トークン, A]
  MlpWeights weights;
  Span<const float> finalNorm;      // [H]
  Span<const uint16_t> keyValue;    // W_kv [2A, H] の写し（MlpWeights と同じ置き方）
  Span<const float> keyValueBias;   // [2A]
  Span<uint16_t> state;             // y [F·R, H]
  Span<uint16_t> memory;            // [R, F, 2A]
  TokenLayout layout;
  int tokens;  // F·R
  int ffn;
  float epsilon;
};

// K6: 48 遷移。
struct CandidateArgs {
  Span<const uint16_t> statics;         // t_s [遷移, H]
  Span<const uint16_t> state;           // y [F·R, H]
  Span<const uint16_t> memory;          // [R, F, 2A]
  Span<const uint8_t> category;         // 遷移の欄 [遷移, 幅]
  Span<const uint8_t> code;             // 鍵の符号 [遷移, F]
  Span<const float> discardExact;       // B の道 [遷移, 幅]
  Span<const int32_t> rowColumn;        // 行の索引 [行, 幅]
  Span<const uint16_t> tileEntry;       // [H, H]
  Span<const float> queryNorm;          // [H]
  Span<const uint16_t> query;           // [A, H]
  Span<const float> queryBias;          // [A]
  Span<const float> codeKey;            // [符号, A]
  Span<const float> codeValue;          // [符号, A]
  MlpWeights weights;                   // 候補の O 射影と FFN
  Span<const float> discardNorm;        // [H]
  Span<const uint16_t> discardHidden;   // [G, H]
  Span<const float> discardHiddenBias;  // [G]
  Span<const float> discardOutput;      // [G]
  Span<const float> discardWeights;     // uBx [B の道]
  Span<uint16_t> transition;            // z [遷移, H]
  Span<float> value;                    // B [遷移]
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
  float epsilon;
  float scale;
};

// K7: 16 行。
struct ValueGraphArgs {
  Span<const uint16_t> transition;      // z [遷移, H]
  Span<const float> value;              // B [遷移]
  Span<const uint16_t> state;           // y [F·R, H]（局の区画の行 r が h）
  Span<const int32_t> rowColumn;        // [行, 幅]
  Span<const int32_t> actionColumn;     // [手, 幅]
  Span<const float> actionExact;        // [手, 幅]
  Span<const float> gateNorm;           // [H]
  Span<const uint16_t> gateHidden;      // [G, 3H]
  Span<const float> gateHiddenBias;     // [G]
  Span<const float> kindOffset;         // [種類, G]
  Span<const float> exactProjection;    // [種類, G, 門の道]
  Span<const float> gateOutput;         // [種類, G]
  Span<const float> gateOutputBias;     // [種類]
  Span<const float> exactOutput;        // [種類, 門の道]
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

// 起動の口（hidden は特化した幅。同期はしない）。
void launchEmbed(int hidden, const EmbedArgs& args, int blocks, hipStream_t stream);
void launchRelation(const RelationArgs& args, hipStream_t stream);
void launchRelationBias(const RelationBiasArgs& args, int headPairs, hipStream_t stream);
void launchQkvFragments(int hidden, const QkvFragmentArgs& args, int headPairs, hipStream_t stream);
void launchAttention(int hidden, bool last, const AttentionArgs& args, int headPairs, hipStream_t stream);
void launchMlpFragments(const MlpFragmentArgs& args, hipStream_t stream);
void launchMlp(int hidden, const MlpArgs& args, hipStream_t stream);
void launchMlpKv(int hidden, const MlpKvArgs& args, hipStream_t stream);
void launchCandidate(int hidden, const CandidateArgs& args, hipStream_t stream);
void launchValueGraph(int hidden, const ValueGraphArgs& args, hipStream_t stream);

}  // namespace prism
