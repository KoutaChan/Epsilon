#pragma once

#include "matrix_product.cuh"

namespace prism::row_major_product {

// 論理タイルの出力行を MMA のレーンに配り、行優先の B から BF16 を二つずつ読む。
template <int K, int Rows, int Columns, int Stride>
__device__ __forceinline__ void accumulate(f32x8 (&acc)[Rows / 16][Columns], const uint16_t (*a)[Stride],
                                           const uint16_t* weight, size_t limit, int stride, int first,
                                           const int (&output)[Columns], Bounds& bounds) {
  const int lane = threadIdx.x % 32;
  const uint16_t* rows[Columns][2];
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = __shfl_sync(0xffffffffu, output[j], lane / 4 + 8 * half);
      const size_t base = bounds.index(static_cast<size_t>(row) * stride + first, K, limit);
      bounds.trap();
      rows[j][half] = weight + base + 2 * (lane % 4);
    }
  }
  matrix_product::accumulate<K, Rows>(acc, a, [&](int j, int half, int k, uint32_t (&pair)[2]) {
    const uint16_t* p = rows[j][half] + 16 * k;
    pair[0] = *reinterpret_cast<const uint32_t*>(p);
    pair[1] = *reinterpret_cast<const uint32_t*>(p + 8);
  });
}

template <int Columns, int K, int Rows, int Stride>
__device__ __forceinline__ void multiply(f32x8 (&acc)[Rows / 16][Columns], const uint16_t (&a)[Rows][Stride],
                                         const uint16_t* weight, size_t limit, int stride, int first,
                                         const int (&output)[Columns], Bounds& bounds) {
  accumulate<K, Rows>(acc, a, weight, limit, stride, first, output, bounds);
}

template <int K>
__device__ __forceinline__ f32x8 gate(const uint16_t (*input)[K + 8], const uint16_t* weight, size_t limit, int unit,
                                      Bounds& bounds) {
  const int output[] = {unit};
  f32x8 acc[1][1]{};
  accumulate<K, 16>(acc, input, weight, limit, K, 0, output, bounds);
  return acc[0][0];
}
}  // namespace prism::row_major_product
