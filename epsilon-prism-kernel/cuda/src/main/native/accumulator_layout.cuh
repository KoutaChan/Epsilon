#pragma once

#include "wave_tile.h"

namespace prism::cuda_tile {

// 各 4 レジスタの位置とレーンの座標を XOR し、同じ位置の値を 1 回のシャッフルで渡せる順にする。
__device__ __forceinline__ void xorRegisterPairs(f32x8& values, int coordinate) {
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const float lo = values[i];
    const float hi = values[i + 1];
    values[i] = coordinate & 1 ? hi : lo;
    values[i + 1] = coordinate & 1 ? lo : hi;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int pair = 4 * (i / 2) + i % 2;
    const float lo = values[pair];
    const float hi = values[pair + 2];
    values[pair] = coordinate & 2 ? hi : lo;
    values[pair + 2] = coordinate & 2 ? lo : hi;
  }
}

// レジスタ番号の上位 2 ビットを入れ替え、左右の MMA と各 MMA 内の要素番号に合わせる。
__device__ constexpr int nativeAccumulatorIndex(int i) { return (i & 1) | ((i & 2) << 1) | ((i & 4) >> 1); }

// 論理列の偶奇と上下 8 列を組にしてから、MMA のレーンへ累積値を配る。
__device__ __forceinline__ void loadAccumulator(const f32x8& logical, int lane, float (&native)[2][4]) {
  f32x8 input = logical;
  xorRegisterPairs(input, (lane & 1) | ((lane >> 2) & 2));
  f32x8 output;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int coordinate = (i & 3) ^ (lane >> 3);
    const int source = ((lane & 4) << 2) | ((lane & 3) << 1) | ((coordinate & 2) << 2) | (coordinate & 1);
    output[i] = __shfl_sync(0xffffffffu, input[i], source);
  }
  xorRegisterPairs(output, lane >> 3);
#pragma unroll
  for (int i = 0; i < 8; ++i) native[nativeAccumulatorIndex(i) / 4][nativeAccumulatorIndex(i) % 4] = output[i];
}

// MMA の行と列のビットを逆の順で配り、後段の論理レーンへ戻す。
__device__ __forceinline__ f32x8 storeAccumulator(const float (&native)[2][4], int lane) {
  f32x8 input;
#pragma unroll
  for (int i = 0; i < 8; ++i) input[i] = native[nativeAccumulatorIndex(i) / 4][nativeAccumulatorIndex(i) % 4];
  xorRegisterPairs(input, lane >> 3);
  const int coordinate = (lane & 1) | ((lane >> 2) & 2);
  f32x8 output;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int source = (((i & 3) ^ coordinate) << 3) | ((lane >> 2) & 4) | ((lane >> 1) & 3);
    output[i] = __shfl_sync(0xffffffffu, input[i], source);
  }
  xorRegisterPairs(output, coordinate);
  return output;
}

}  // namespace prism::cuda_tile
