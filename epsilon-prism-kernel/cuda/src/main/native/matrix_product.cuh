#pragma once

#include "accumulator_layout.cuh"
#include "inference_device.h"

namespace prism::matrix_product {

struct Accumulator {
  float native[2][4];
};

// A の 16 行を共有メモリから読み、B の配置に応じたロードで K 昇順に積和を求める。
template <int K, int Rows, bool Prefetch, int Columns, int Stride, class LoadWeight>
__device__ __forceinline__ void accumulateNative(Accumulator (&sums)[Rows / 16][Columns], const uint16_t (*a)[Stride],
                                                 LoadWeight load) {
  static_assert(K % 16 == 0 && Rows % 16 == 0 && Stride % 8 == 0);
  const int lane = threadIdx.x % 32;
  using WeightTile = uint32_t[Columns][2][2];
  const auto loadTile = [&](int k, WeightTile& b) {
#pragma unroll
    for (int j = 0; j < Columns; ++j) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        load(j, half, k, b[j][half]);
      }
    }
  };
  const auto multiplyTile = [&](int k, const WeightTile& b) {
#pragma unroll
    for (int m = 0; m < Rows / 16; ++m) {
      const uint32_t address =
          static_cast<uint32_t>(__cvta_generic_to_shared(&a[16 * m + lane % 16][16 * k + 8 * (lane / 16)]));
      uint32_t a0, a1, a2, a3;
      asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
          : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
          : "r"(address)
          : "memory");
#pragma unroll
      for (int j = 0; j < Columns; ++j) {
#pragma unroll
        for (int half = 0; half < 2; ++half)
          cuda_tile::mma16x8(a0, a1, a2, a3, b[j][half][0], b[j][half][1], sums[m][j].native[half]);
      }
    }
  };
  if constexpr (Prefetch) {
    // 現在のタイルで積和を始める前に次の重みを読む。
    WeightTile weights[2];
    loadTile(0, weights[0]);
#pragma unroll 1
    for (int first = 0; first < K / 16; first += 2) {
#pragma unroll
      for (int slot = 0; slot < 2; ++slot) {
        const int k = first + slot;
        if (k < K / 16) {
          if (k + 1 < K / 16) loadTile(k + 1, weights[1 - slot]);
          multiplyTile(k, weights[slot]);
        }
      }
    }
  } else {
#pragma unroll 2
    for (int k = 0; k < K / 16; ++k) {
      WeightTile b;
      loadTile(k, b);
      multiplyTile(k, b);
    }
  }
}

// 論理列で受け取った累積値を積和の前後で並べ替える。
template <int K, int Rows, bool Prefetch = false, int Columns, int Stride, class LoadWeight>
__device__ __forceinline__ void accumulate(f32x8 (&acc)[Rows / 16][Columns], const uint16_t (*a)[Stride],
                                           LoadWeight load) {
  const int lane = threadIdx.x % 32;
  Accumulator sums[Rows / 16][Columns];
#pragma unroll
  for (int m = 0; m < Rows / 16; ++m) {
#pragma unroll
    for (int j = 0; j < Columns; ++j) cuda_tile::loadAccumulator(acc[m][j], lane, sums[m][j].native);
  }
  accumulateNative<K, Rows, Prefetch>(sums, a, load);
#pragma unroll
  for (int m = 0; m < Rows / 16; ++m) {
#pragma unroll
    for (int j = 0; j < Columns; ++j) acc[m][j] = cuda_tile::storeAccumulator(sums[m][j].native, lane);
  }
}
}  // namespace prism::matrix_product
