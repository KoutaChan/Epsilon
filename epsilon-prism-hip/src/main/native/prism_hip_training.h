// 学習の自前の演算（DJL の配列のデバイスのメモリを直に読み書きする）: 関係の符号、関係の偏りつきの注意の順伝播と逆伝播、DJL の配列の
// アドレス。
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "prism_hip_kernels.h"

namespace prism {

// 学習の注意の行の上限（K2・K4 の kMaxRowTokens と同じ。Java の HipAttention.MAX_TOKENS）。
constexpr int kAttentionMaxTokens = 160;
// 頭の幅（Java の HipAttention.FEATURES）。
constexpr int kAttentionFeatures = 16;
// 関係の符号の区画と、区画あたりの値の上限（1 語に 4 ビットの符号を 4 つ詰める）。
constexpr int kAttentionMaxSegments = 8;
constexpr int kAttentionMaxEntries = 16;

// 関係の符号の数（全ての配列は連続）。行の数は layout.rows、tokens は行のトークンの軸の長さ S（どの行の局・席・牌と出来事の数より
// 小さくない）。
//   relation [rows·layout.fixed() + events, columns.width]（U8、入力の関係の属性。トークンの軸は型ごとの区画: 局・席・牌・出来事）、
//   rowColumn [rows, rowWidth]（INT32、行の索引）、tileRelation [tiles·tiles]（INT32、牌の関係の類）、
//   codes [rows, tokens, tokens]（UINT32、区画 s の符号を bit 4s から。DJL の [rows, tokens, tokens, 2] の INT16 と同じバイト）
struct RelationCodeShape {
  TokenLayout layout;
  int events;
  int tokens;
  int rowWidth;
  int rowFirstEvent;
  int rowEvents;
  RelationColumns columns;
};

// 関係の符号の配列。
struct RelationCodeArrays {
  const uint8_t* relation;
  const int32_t* rowColumn;
  const int32_t* tileRelation;
  uint32_t* codes;
};

// 注意の数（全ての配列は連続）。
//   q [rows, queries, heads·16]、kv [rows, keys, 2·heads·16]（鍵の後に値）、mask [rows, keys]（0 でなければ在る）、
//   codes [rows, codeQueries, keys, words]（区画 s の符号は語 s/4 の 4(s%4) ビット目から 4 ビット。問い合わせは先頭の queries 個を読む）、
//   table [segments·entries, heads]（ρ、FP32）
// heads は偶数（順伝播の作業群は頭の対）、words は 2（1 組の符号は 32 bit の 1 語で、codes は 4 バイトにそろうこと）。
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

// 順伝播の配列。out は BF16、outFloat [rows, queries, heads·16] と logSumExp [rows, heads, queries]（自然対数）は逆伝播のための FP32。
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

// 逆伝播の配列。gradOut は BF16。q・kv の勾配は FP32 で作って BF16 に最近接偶数で 1 回丸める（c10::BFloat16 と同じ）:
// gradQuery [rows, queries, heads·16]、gradKeyValue [rows, keys, 2·heads·16]。gradTable [rows·heads, segments·entries]
// は FP32（行 × 頭ごとの ρ の勾配の部分和。原子的な加算なしの決まった順で、呼び手が行で足す）。頭の出力の勾配が全て ±0 の問い合わせ
// （詰め物）は寄与がちょうど 0 なので回さず、その dq は +0。詰め物の出力の勾配が 0 なら、行の勾配は詰め物の幅と q にビットで依らない。
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

// 数を確かめる（範囲外は std::invalid_argument）。
void checkRelationCodeShape(const RelationCodeShape& shape);

// 関係の符号を作る（T0。codes は 4 バイトにそろうこと）。
void relationCodes(const RelationCodeShape& shape, const RelationCodeArrays& arrays, hipStream_t stream);

// 数を確かめる（範囲外は std::invalid_argument）。
void checkAttentionShape(const AttentionShape& shape);

void attentionForward(const AttentionShape& shape, const AttentionForward& arrays, hipStream_t stream);

void attentionBackward(const AttentionShape& shape, const AttentionBackward& arrays, hipStream_t stream);

// DJL の配列の要素の型（Java の HipAttention.Element と同じ順）。
enum class TensorElement : int { kBool = 0, kInt16 = 1, kInt32 = 2, kBfloat16 = 3, kFloat32 = 4, kUint8 = 5 };

// DJL の配列（PtNDArray.getHandle() の torch::Tensor*）のデータの先頭のアドレス。型が合わなければ libtorch が例外を投げる。
// アドレスが GPU device のメモリでなければ std::invalid_argument。
void* tensorData(int64_t handle, TensorElement element, int device);

}  // namespace prism
