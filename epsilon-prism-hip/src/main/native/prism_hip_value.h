// 打牌の値 B（eager の DiscardValue）。候補層が 1 段の形は K6 が、2 段目のある形は K9 が、候補層の出力 z から同じ式で作る。
//
//   B = Σ_g silu(⌊⌊N_B(z)⌉·W_Bᵀ + c_B⌉)·wB_g + Σ_k x^B_k·uBx_k   （FP32）
//
// G は実行時の値（16 の倍数）で、ウェーブ w は G の列のタイル n = w, w + 8, … を受け持つ（G が 512 なら受け持つ回数が倍になるだけで、
// LDS とレジスタの量は G に依らない）。和は lane の列 → 16 lane の木 → ウェーブの順に、x^B の項は k の順に足すので、遷移ごとの和の順は
// 遷移のタイルの組み方に依らない。
#pragma once

#include "prism_hip_mlp.h"

namespace prism {
namespace discard {

constexpr int kMaxExact = 8;  // B に直接足す特徴 x^B の数の上限（DiscardExact の 8 フィールド。起動のときに照合する）

// タイルの遷移（t0 から rows 個）の B を out [遷移] に書く。呼ぶ前: stream にタイルの z がある（同期済み）。呼んだ後: stream は ⌊N_B(z)⌉
// になる。ワークグループの全てのスレッドで呼ぶ（中でバリアを 2 回通る）。exact は x^B [遷移, exactWidth]、partial はウェーブごとの和の
// バッファ。W_B は行優先の [G, H] で、gemmTile で読む。
template <int H, int Rows>
__device__ __forceinline__ void value(uint16_t (&stream)[Rows][H + tile::kPad], float (&partial)[tile::kWaves][Rows],
                                      const DiscardValueWeights& weights, const Span<const float>& exact, int exactWidth,
                                      int gate, float epsilon, const Span<float>& out, int t0, int rows, Bounds& bounds) {
  constexpr int kRowTiles = Rows / 16;
  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int half = lane >> 4;
  const int column = lane & 15;

  tile::normalizeTile<H>(stream, weights.norm.data, weights.norm.limit, epsilon, tid, bounds);
  tile::barrier();
  float rowSum[kRowTiles][8];
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      rowSum[m][v] = 0.0f;
    }
  }
  for (int n = wave; n < gate / 16; n += tile::kWaves) {
    f32x8 u[kRowTiles][1];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
      u[m][0] = f32x8{};
    }
    const int units[1] = {n * 16 + column};
    tile::gemmTile<1, H>(u, stream, weights.hidden.data, weights.hidden.limit, H, 0, units, column, bounds);
    const float bias = tile::load(weights.hiddenBias.data, weights.hiddenBias.limit, units[0], bounds);
    const float weight = tile::load(weights.output.data, weights.output.limit, units[0], bounds);
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        rowSum[m][v] += siluFast(bf16Float(bf16RoundActivation(u[m][0][v] + bias))) * weight;
      }
    }
  }
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const float sum = halfSum(rowSum[m][v]);
      if (column == 0) {
        partial[wave][m * 16 + 2 * v + half] = sum;
      }
    }
  }
  tile::barrier();
  if (tid < rows) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < tile::kWaves; ++w) {
      total += partial[w][tid];
    }
    // Σ_k x^B_k·uBx_k（k の順）。kMaxExact 個をまとめて先に読み込む（幅の外は列 0 を読み、足さない）。
    const size_t exactAt = static_cast<size_t>(t0 + tid) * exactWidth;
    float features[kMaxExact];
    float coefficients[kMaxExact];
#pragma unroll
    for (int k = 0; k < kMaxExact; ++k) {
      const int at = k < exactWidth ? k : 0;
      features[k] = exact.data[bounds.index(exactAt + at, 1, exact.limit)];
      coefficients[k] = tile::load(weights.exact.data, weights.exact.limit, at, bounds);
    }
    float direct = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxExact; ++k) {
      if (k < exactWidth) {
        direct += features[k] * coefficients[k];
      }
    }
    out.data[bounds.index(t0 + tid, 1, out.limit)] = total + direct;
  }
}

}  // namespace discard
}  // namespace prism
