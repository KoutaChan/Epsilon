#pragma once

namespace prism {
namespace {

// 学習用の RMSNorm の順伝播と逆伝播（入出力は BF16、統計・ゲイン・勾配は FP32）。GPU で学習するとき、eager の EncoderLayer・RmsNorm が行う
// cast → rms_norm（FP32）→ cast の 3 段の代わりに使う。数式は同じ（⌊·⌉ は BF16 への最近接偶数の 1 回の丸め。c10::BFloat16 と同じ）:
//
//   r_i = rsqrt(Σ_j x_ij² / width + ε)、y_ij = ⌊(x_ij·r_i)·g_j⌉
//   x̂_ij = x_ij·r_i、m_i = Σ_j (dy_ij·g_j)·x̂_ij / width、dx_ij = ⌊r_i·(dy_ij·g_j − x̂_ij·m_i)⌉、dg_j = Σ_i dy_ij·x̂_ij
//
// 1 つのウェーブが 1 行を受け持ち、lane l は 8 列のチャンク l + 32k（k < kChunks。列 8(l + 32k) < width のものだけ）を
// 16 バイトずつ読み書きする。
// 行の和（Σx² と m）は、lane の中でチャンク・列の順に足してから、ウェーブの xor の木（waveSum）で足す。行の中で足す順が決まるので、行の
// 値（y・r・dx）はバッチの組み方と行の数に依らない。
// 逆伝播のワークグループは kNormGainRows 行のブロックを受け持ち、ウェーブ w がブロックの行 w, w + 8, … を順に回して、dg の lane ごとの
// 部分和を持つ。ウェーブの部分和は共有メモリでウェーブの順に足し、ブロックごとの部分和 [ブロック, width] に書く。呼び出し元はそれをブロックの
// 順に足す（アトミックな加算を使わない、順の決まった 2 段の和）。
// 検査つきのビルドでは、全てのグローバルメモリの読み書きが配列の範囲の中にあるかを確かめる。
constexpr int kThreads = 256;
constexpr int kWaves = kThreads / kWave;
constexpr int kChunk = 8;                                     // lane が一度に読み書きする列の数（BF16 の 16 バイト）
constexpr int kMaxChunks = kNormMaxWidth / (kWave * kChunk);  // lane あたりのチャンクの数の上限
static_assert(kMaxChunks * kWave * kChunk == kNormMaxWidth, "the widest row is a whole number of chunks per lane");
static_assert(kNormGainRows % kWaves == 0, "every wave walks the same number of rows of a full gain block");

// 検査つきのビルドで使う、上限つきの配列（NormForward・NormBackward と同じ並び）。
struct ForwardArgs {
  NormShape shape;
  Span<const uint16_t> input;
  Span<const float> gain;
  Span<uint16_t> out;
  Span<float> inverseRms;
};

struct BackwardArgs {
  NormShape shape;
  Span<const uint16_t> input;
  Span<const float> gain;
  Span<const float> inverseRms;
  Span<const uint16_t> gradOut;
  Span<uint16_t> gradInput;
  Span<float> gradGain;
};

// lane のチャンク k の先頭の列。
__device__ __forceinline__ int column(int lane, int k) { return kChunk * (lane + kWave * k); }

// lane のチャンク k が行の中にあれば true（width は 8 の倍数なので、チャンクは丸ごと中か丸ごと外）。
__device__ __forceinline__ bool owns(int lane, int k, int width) { return column(lane, k) < width; }

// 行の 8 列の BF16（16 バイト）を float にする。
__device__ __forceinline__ void load8(Span<const uint16_t> span, size_t at, float (&values)[8], Bounds& bounds) {
  unpack8(*reinterpret_cast<const u32x4*>(span.data + bounds.index(at, kChunk, span.limit)), values);
}

// 8 つの float を BF16 に丸めて行の 8 列に書く。
__device__ __forceinline__ void store8(Span<uint16_t> span, size_t at, const float (&values)[8], Bounds& bounds) {
  *reinterpret_cast<u32x4*>(span.data + bounds.index(at, kChunk, span.limit)) = pack8(values);
}

// ゲインの 8 列 [col, col + 8) を読む。
__device__ __forceinline__ void loadGain(Span<const float> gain, int col, float (&values)[8], Bounds& bounds) {
  const size_t at = bounds.index(col, kChunk, gain.limit);
  const f32x4 lo = *reinterpret_cast<const f32x4*>(gain.data + at);
  const f32x4 hi = *reinterpret_cast<const f32x4*>(gain.data + at + 4);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    values[e] = lo[e];
    values[e + 4] = hi[e];
  }
}

// 順伝播（ワークグループは 8 行、ウェーブは 1 行を受け持つ）。
template <int kChunks>
__global__ __launch_bounds__(kThreads) void normForwardKernel(ForwardArgs args) {
  Bounds bounds;
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const int row = static_cast<int>(blockIdx.x) * kWaves + static_cast<int>(threadIdx.x) / kWave;
  const int width = args.shape.width;
  if (row < args.shape.rows) {
    const size_t first = static_cast<size_t>(row) * width;
    float x[kChunks][kChunk];
    // 二乗の和は、2 列目の二乗を丸めた値に 1 列目の二乗を積和で足し、3 列目からは列の順に積和で足す。
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (owns(lane, k, width)) {
        load8(args.input, first + column(lane, k), x[k], bounds);
      } else {
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          x[k][e] = 0.0f;
        }
      }
      if (k == 0) {
        sum = fusedMultiplyAdd(x[0][0], x[0][0], roundedMultiply(x[0][1], x[0][1]));
      }
#pragma unroll
      for (int e = k == 0 ? 2 : 0; e < kChunk; ++e) {
        sum = fusedMultiplyAdd(x[k][e], x[k][e], sum);
      }
    }
    const float r = inverseSqrt(waveSum(sum) / static_cast<float>(width) + args.shape.epsilon);
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (owns(lane, k, width)) {
        float g[kChunk];
        loadGain(args.gain, column(lane, k), g, bounds);
        float y[kChunk];
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          y[e] = (x[k][e] * r) * g[e];
        }
        store8(args.out, first + column(lane, k), y, bounds);
      }
    }
    if (lane == 0) {
      args.inverseRms.data[bounds.index(row, 1, args.inverseRms.limit)] = r;
    }
  }
  bounds.trap();
}

// 逆伝播（ワークグループは kNormGainRows 行のブロックを受け持ち、ウェーブはブロックの行を 8 行おきに回す）。dy·g は m の和と dx の 2 か所で
// 同じ式で計算する（ビットで同じ値になる）。
template <int kChunks>
__global__ __launch_bounds__(kThreads) void normBackwardKernel(BackwardArgs args) {
  __shared__ float partial[kWaves][kChunks * kWave * kChunk];
  Bounds bounds;
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const int wave = static_cast<int>(threadIdx.x) / kWave;
  const int width = args.shape.width;
  const int begin = static_cast<int>(blockIdx.x) * kNormGainRows;
  const int end = min(begin + kNormGainRows, args.shape.rows);
  float g[kChunks][kChunk];
  float sums[kChunks][kChunk];
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    if (owns(lane, k, width)) {
      loadGain(args.gain, column(lane, k), g[k], bounds);
    } else {
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        g[k][e] = 0.0f;
      }
    }
#pragma unroll
    for (int e = 0; e < kChunk; ++e) {
      sums[k][e] = 0.0f;
    }
  }
  for (int row = begin + wave; row < end; row += kWaves) {
    const size_t first = static_cast<size_t>(row) * width;
    const float r = args.inverseRms.data[bounds.index(row, 1, args.inverseRms.limit)];
    float normalized[kChunks][kChunk];
    float dy[kChunks][kChunk];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (owns(lane, k, width)) {
        float x[kChunk];
        load8(args.input, first + column(lane, k), x, bounds);
        load8(args.gradOut, first + column(lane, k), dy[k], bounds);
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          normalized[k][e] = x[e] * r;
          dot = fusedMultiplyAdd(dy[k][e] * g[k][e], normalized[k][e], dot);
        }
      }
    }
    const float m = waveSum(dot) / static_cast<float>(width);
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (owns(lane, k, width)) {
        float dx[kChunk];
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          dx[e] = r * fusedMultiplyAdd(dy[k][e], g[k][e], -(normalized[k][e] * m));
          sums[k][e] = fusedMultiplyAdd(dy[k][e], normalized[k][e], sums[k][e]);
        }
        store8(args.gradInput, first + column(lane, k), dx, bounds);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    if (owns(lane, k, width)) {
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        partial[wave][column(lane, k) + e] = sums[k][e];
      }
    }
  }
  shareGroupWrites();
  for (int col = static_cast<int>(threadIdx.x); col < width; col += kThreads) {
    float total = partial[0][col];
#pragma unroll
    for (int w = 1; w < kWaves; ++w) {
      total += partial[w][col];
    }
    args.gradGain.data[bounds.index(static_cast<size_t>(blockIdx.x) * width + col, 1, args.gradGain.limit)] = total;
  }
  bounds.trap();
}
}  // namespace
}  // namespace prism
