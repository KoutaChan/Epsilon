#pragma once

#include <cstdint>

namespace prism {
// 学習用の Attention が 1 行で扱えるトークン数の上限。
constexpr int kAttentionMaxTokens = kMaxRowTokens;
// Attention のヘッドあたりの特徴数。
constexpr int kAttentionFeatures = 16;
// 関係のコードのセグメントの数の上限と、セグメントあたりの値の数の上限（16 bit の 1 ワードに 4 ビットのコードを 4 つ詰める）。
constexpr int kAttentionMaxSegments = 8;
constexpr int kAttentionMaxEntries = 16;

// 関係のコードを作るときの形（配列は全て連続）。行の数は layout.rows。tokens は行のトークンの軸の長さ S で、どの行でも
// 局・席・牌・イベントの数の合計より小さくない。
//   relation [rows·layout.fixed() + events, columns.width]（U8、入力の関係の属性。トークンの軸は型ごとの範囲に
//   分かれる: 局・席・牌・イベント）、
//   rowColumn [rows, rowWidth]（INT32、行のインデックス）、tileRelation [tiles·tiles]（INT32、牌の関係のクラス）、
//   codes [rows, tokens, tokens]（UINT32。セグメント s のコードは bit 4s から。DJL の [rows, tokens, tokens, 2] の INT16 と同じバイト列）
struct RelationCodeShape {
  TokenLayout layout;
  int events;
  int tokens;
  int rowWidth;
  int rowFirstEvent;
  int rowEvents;
  RelationColumns columns;
};

// 関係のコードを作るときの配列。
struct RelationCodeArrays {
  const uint8_t* relation;
  const int32_t* rowColumn;
  const int32_t* tileRelation;
  uint32_t* codes;
};

// Attention の形（配列は全て連続）。
//   q [rows, queries, heads·16]、kv [rows, keys, 2·heads·16]（キーの後にバリュー）、mask [rows, keys]（0 でなければ有効）、
//   codes [rows, codeQueries, keys, words]（セグメント s のコードはワード s/4 の 4(s%4) ビット目から 4 ビット。クエリは先頭の queries 個を
//   読む）、
//   table [segments·entries, heads]（ρ、FP32）
// heads は偶数（順伝播のワークグループはヘッドの対を受け持つ）。words は 2（1 組のコードを 32 bit にまとめて読むので、codes は
// 4 バイトにそろっていること）。
struct AttentionShape {
  int rows;
  int queries;
  int keys;
  int codeQueries;
  int heads;
  int words;
  int segments;
  int entries;
  float scale;
};

// 順伝播の配列。out は BF16。outFloat [rows, queries, heads·16] と logSumExp [rows, heads, queries]（自然対数）は、逆伝播で使う FP32。
struct AttentionForward {
  const uint16_t* query;
  const uint16_t* keyValue;
  const float* mask;
  const int16_t* codes;
  const float* table;
  uint16_t* out;
  float* outFloat;
  float* logSumExp;
};

// 逆伝播の配列。gradOut は BF16。q・kv の勾配は FP32 で作り、BF16 に最近接偶数で 1 回丸める（c10::BFloat16 と
// 同じ）: gradQuery [rows, queries, heads·16]、gradKeyValue [rows, keys, 2·heads·16]。gradTable [rows·heads, segments·entries] は FP32 で、
// 行 × ヘッドごとの ρ の勾配の部分和（アトミックな加算を使わない決まった順で作り、呼び出し元が行について足し合わせる）。ヘッドの出力の勾配が
// 全て ±0 のクエリ（パディング）は寄与がちょうど 0 なので計算せず、その dq は +0。パディングの出力の勾配が 0 なら、行の勾配はパディングの
// 幅と q に依らず、ビットまで同じになる。
struct AttentionBackward {
  const uint16_t* query;
  const uint16_t* keyValue;
  const float* mask;
  const int16_t* codes;
  const float* table;
  const float* outFloat;
  const float* logSumExp;
  const uint16_t* gradOut;
  uint16_t* gradQuery;
  uint16_t* gradKeyValue;
  float* gradTable;
};

// RMSNorm の幅の上限と、ゲインの勾配の部分和 1 つが受け持つ行の数（Java の RmsNormKernels.MAX_WIDTH・GAIN_ROWS）。
// 16 バイトにそろう）。width は 8 の倍数で kNormMaxWidth 以下。
struct NormShape {
  int rows;
  int width;
  float epsilon;
};

// 順伝播の配列。inverseRms [rows] は逆伝播で使う r = rsqrt(Σx²/width + ε)（FP32）。
struct NormForward {
  const uint16_t* input;
  const float* gain;
  uint16_t* out;
  float* inverseRms;
};

// 逆伝播の配列。gradInput は FP32 で作り、BF16 に最近接偶数で 1 回丸める。gradGain [⌈rows / kNormGainRows⌉, width] は FP32 で、
// kNormGainRows 行のブロックごとの部分和（アトミックな加算を使わない決まった順で作り、呼び出し元がブロックについて足し合わせる）。
struct NormBackward {
  const uint16_t* input;
  const float* gain;
  const float* inverseRms;
  const uint16_t* gradOut;
  uint16_t* gradInput;
  float* gradGain;
};

}  // namespace prism
