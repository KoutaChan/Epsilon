#pragma once

#include "matrix_product.cuh"

namespace prism::packed_mlp_product {

using FfnAccumulator = matrix_product::Accumulator;

// 公開時に並べた重みから、MMA の担当列に対応する上下 8 要素の組を読む。
template <int Columns>
struct PackedWeight {
  device::Buffer weight;
  const int (&base)[Columns];

  __device__ __forceinline__ void operator()(int j, int half, int k, uint32_t (&pair)[2]) const {
    const int lane = threadIdx.x % 32;
    const uint32_t offset = (lane / 4) * 16 + (lane % 4) * 4;
    const int at = base[j] + k * 512 + half * 128;
    pair[0] = device::loadWord(weight, offset, at);
    pair[1] = device::loadWord(weight, offset, at + 256);
  }
};

template <bool, int Columns, int K, int Rows, int Stride>
__device__ __forceinline__ void multiply(f32x8 (&acc)[Rows / 16][Columns], const uint16_t (&a)[Rows][Stride],
                                         device::Buffer weight, const int (&base)[Columns]) {
  matrix_product::accumulate<K, Rows>(acc, a, PackedWeight<Columns>{weight, base});
}

template <int Columns, int K, int Rows, int Stride>
__device__ __forceinline__ void multiplyFfn(FfnAccumulator (&acc)[Rows / 16][Columns],
                                            const uint16_t (&a)[Rows][Stride], device::Buffer weight,
                                            const int (&base)[Columns]) {
  matrix_product::accumulateNative<K, Rows, true>(acc, a, PackedWeight<Columns>{weight, base});
}

// 各レーンが持つ隣り合う 2 列を、上半分と下半分の行へ書く。
template <int RowTiles, int Columns>
struct FfnActivation {
  uint32_t pairs[RowTiles][Columns][2][2];

  template <int Rows, int Stride>
  __device__ __forceinline__ void store(uint16_t (&inner)[Rows][Stride], int wave) const {
    static_assert(Rows == 16 * RowTiles && Stride % 2 == 0);
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int m = 0; m < RowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < Columns; ++j) {
#pragma unroll
        for (int side = 0; side < 2; ++side) {
          const int col = 16 * (wave * Columns + j) + 8 * side + 2 * (lane % 4);
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int row = 16 * m + lane / 4 + 8 * half;
            *reinterpret_cast<uint32_t*>(&inner[row][col]) = pairs[m][j][side][half];
          }
        }
      }
    }
  }
};

// 論理列ごとのバイアスを MMA の担当列に渡し、活性化した BF16 を共有メモリへ書く順に組む。
template <int RowTiles, int Columns, class Activate>
__device__ __forceinline__ FfnActivation<RowTiles, Columns> activateFfn(const FfnAccumulator (&up)[RowTiles][Columns],
                                                                        const float (&bias)[Columns],
                                                                        Activate activate) {
  const int lane = threadIdx.x % 32;
  FfnActivation<RowTiles, Columns> result;
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
#pragma unroll
    for (int side = 0; side < 2; ++side) {
      const int col = 8 * side + 2 * (lane % 4);
      const float lowBias = __shfl_sync(0xffffffffu, bias[j], col);
      const float highBias = __shfl_sync(0xffffffffu, bias[j], col + 1);
#pragma unroll
      for (int m = 0; m < RowTiles; ++m) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const auto& values = up[m][j].native[side];
          const uint16_t low = activate(values[2 * half], lowBias);
          const uint16_t high = activate(values[2 * half + 1], highBias);
          result.pairs[m][j][side][half] = low | (uint32_t(high) << 16);
        }
      }
    }
  }
  return result;
}

__device__ __forceinline__ f32x8 finishFfn(const FfnAccumulator& down) {
  return cuda_tile::storeAccumulator(down.native, threadIdx.x % 32);
}
}  // namespace prism::packed_mlp_product
