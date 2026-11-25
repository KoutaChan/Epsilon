#pragma once

#include "accumulator_layout.cuh"
#include "wave_tile.h"

namespace prism::attention_tile {
using cuda_tile::bf16x16;
using cuda_tile::f32x8;
using cuda_tile::u32x4;

// 4 ワードの番号をレーンの 2 ビットに合わせて入れ替える。
__device__ __forceinline__ u32x4 permuteColumnWords(u32x4 words, int coordinate) {
  u32x4 pairs;
  u32x4 result;
#pragma unroll
  for (int i = 0; i < 4; ++i) pairs[i] = (coordinate & 1) ? words[i ^ 1] : words[i];
#pragma unroll
  for (int i = 0; i < 4; ++i) result[i] = (coordinate & 2) ? pairs[i ^ 2] : pairs[i];
  return result;
}

// 各レーンの 8 個の BF16 を、ワードごとに 1 回のシャッフルで左右の B に分ける。
__device__ __forceinline__ u32x4 columns(u32x4 words, int lane) {
  words = permuteColumnWords(words, lane >> 3);
  u32x4 result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int source = (lane >> 2) | ((i ^ (lane & 3)) << 3);
    result[i] = __shfl_sync(0xffffffffu, words[i], source);
  }
  return permuteColumnWords(result, lane & 3);
}

// native の行・列を、各クエリの偶数行または奇数行を持つ論理配置に戻す。
__device__ __forceinline__ f32x8 logical(const f32x8& values, int lane) {
  float native[2][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) native[i / 4][i % 4] = values[i];
  return cuda_tile::storeAccumulator(native, lane);
}

__device__ __forceinline__ f32x8 accumulator(const f32x8& values, int lane) {
  float native[2][4];
  cuda_tile::loadAccumulator(values, lane, native);
  f32x8 result;
#pragma unroll
  for (int i = 0; i < 8; ++i) result[i] = native[i / 4][i % 4];
  return result;
}

// 各レーンが一つの 8×8 部分行列の行頭を渡し、ワープ全体で 16×16 の A を読む。
__device__ __forceinline__ u32x4 loadRows(const uint16_t* row) {
  u32x4 a;
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(row));
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(address)
      : "memory");
  return a;
}

__device__ __forceinline__ void multiply(const u32x4& a, const u32x4& b, f32x8& c) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float half[4] = {c[4 * h], c[4 * h + 1], c[4 * h + 2], c[4 * h + 3]};
    cuda_tile::mma16x8(a[0], a[1], a[2], a[3], b[h], b[h + 2], half);
#pragma unroll
    for (int i = 0; i < 4; ++i) c[4 * h + i] = half[i];
  }
}

// 左右の 8 行について K の前半と後半から BF16 を 2 個ずつ読む。
__device__ __forceinline__ u32x4 rowPairs(cuda_tile::BufferResource buffer, uint32_t row0, uint32_t row8, int step,
                                          int secondK) {
  return {cuda_tile::bufferLoad4(buffer, row0, step), cuda_tile::bufferLoad4(buffer, row8, step),
          cuda_tile::bufferLoad4(buffer, row0, step + secondK), cuda_tile::bufferLoad4(buffer, row8, step + secondK)};
}

// 固定クエリの Q・K・V 射影を K の昇順で累積し、最後に論理配置へ戻す。
template <int H, int Blocks, int Queries>
__device__ __forceinline__ void project(f32x8 (&out)[Blocks][3], cuda_tile::BufferResource weights, int head,
                                        cuda_tile::BufferResource inputs, const uint32_t (&inputRows)[Blocks]) {
  constexpr int fragmentBytes = 16 * 16 * sizeof(uint16_t);
  constexpr int stepBytes = 2 * 3 * fragmentBytes;
  const int lane = cuda_tile::tileLane(), group = lane / 4, pairBytes = (lane % 4) * sizeof(uint32_t);
  const int weightRow = group * 8 * sizeof(uint16_t) + pairBytes;
  uint32_t row0[Blocks], row8[Blocks];
  f32x8 native[Blocks][3]{};
#pragma unroll
  for (int b = 0; b < Blocks; ++b) {
    row0[b] = __shfl_sync(0xffffffffu, inputRows[b], group) + pairBytes;
    row8[b] = __shfl_sync(0xffffffffu, inputRows[b], group + 8) + pairBytes;
  }
#pragma unroll 1
  for (int k = 0; k < H / 16; ++k) {
    u32x4 features[Blocks];
#pragma unroll
    for (int b = 0; b < Blocks; ++b) features[b] = rowPairs(inputs, row0[b], row8[b], 32 * k, 16);
#pragma unroll
    for (int j = Queries ? 0 : 1; j < 3; ++j) {
      const auto w = rowPairs(weights, weightRow, weightRow + fragmentBytes / 4,
                              k * stepBytes + (3 * head + j) * fragmentBytes, fragmentBytes / 2);
#pragma unroll
      for (int b = 0; b < Blocks; ++b) {
        if (j != 0)
          multiply(features[b], w, native[b][j]);
        else if (b < Queries)
          multiply(w, features[b], native[b][0]);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < Blocks; ++b) {
#pragma unroll
    for (int j = 0; j < 3; ++j) out[b][j] = logical(native[b][j], lane);
  }
}

// 共有メモリに並べた代表入力を、各ヘッドの Q・K・V に射影する。
template <int H, int Parts, int Rows, int Stride>
__device__ __forceinline__ void projectDistinct(f32x8 (&acc)[Parts], cuda_tile::BufferResource weights,
                                                const uint16_t (&input)[Rows][Stride], int projection) {
  constexpr int fragmentBytes = 16 * 16 * sizeof(uint16_t);
  constexpr int stepBytes = 6 * fragmentBytes;
  const int lane = cuda_tile::tileLane();
  const int row = (lane / 4) * 8 * sizeof(uint16_t) + (lane & 3) * sizeof(uint32_t);
  f32x8 native[Parts]{};
#pragma unroll 2
  for (int k = 0; k < H / 16; ++k) {
    const auto w = rowPairs(weights, row, row + fragmentBytes / 4, k * stepBytes, fragmentBytes / 2);
#pragma unroll
    for (int m = 0; m < Parts; ++m) {
      const auto x = loadRows(&input[16 * m + (lane & 15)][16 * k + 8 * (lane >> 4)]);
      if (projection == 0)
        multiply(w, x, native[m]);
      else
        multiply(x, w, native[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < Parts; ++m) acc[m] = logical(native[m], lane);
}

__device__ __forceinline__ f32x8 score(const uint16_t (*keys)[32], int firstKey, int head, const bf16x16& query) {
  const int lane = cuda_tile::tileLane();
  const int row = firstKey + (lane & 15);
  const int chunk = (2 * head + (lane >> 4)) ^ ((row >> 1) & 3);
  const u32x4 a = loadRows(&keys[row][8 * chunk]);
  u32x4 b;
#pragma unroll
  for (int i = 0; i < 4; ++i) b[i] = (lane & 16) ? query[4 + i] : query[i];
  f32x8 result{};
  multiply(a, columns(b, lane), result);
  return logical(result, lane);
}

__device__ __forceinline__ f32x8 accumulate(const uint16_t* value, int, const float (&e)[8], const f32x8& out) {
  const int lane = cuda_tile::tileLane();
  const u32x4 a = loadRows(value + 8 * (lane >> 4));
  u32x4 high;
  u32x4 low;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t x = __float_as_uint(e[2 * i]), y = __float_as_uint(e[2 * i + 1]);
    high[i] = cuda_tile::upperHalves(x, y);
    const float rx = e[2 * i] - __uint_as_float(x & 0xffff0000u);
    const float ry = e[2 * i + 1] - __uint_as_float(y & 0xffff0000u);
    low[i] = cuda_tile::upperHalves(__float_as_uint(rx), __float_as_uint(ry));
  }
  f32x8 result = accumulator(out, lane);
  multiply(a, columns(high, lane), result);
  multiply(a, columns(low, lane), result);
  return logical(result, lane);
}
}  // namespace prism::attention_tile
