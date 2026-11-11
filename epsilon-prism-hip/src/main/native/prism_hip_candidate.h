// 候補層の 1 段目（K6）と 2 段目（K9）が共有する部品。遷移の行と牌 τ を求める関数と、Cross-Attention のキーの読み込みと内積を持つ。
// 打牌の値 B は prism_hip_value.h にある。
#pragma once

#include "prism_hip_kernels.h"

namespace prism {
namespace candidate {

// 遷移 t が属する行を返す。バッチの rows 行から、行の最初の遷移を 2 分探索する。
__device__ __forceinline__ int transitionRow(const Span<const int32_t>& rowColumn, int rowWidth, int firstTransition,
                                             int rows, int t, Bounds& bounds) {
  int lo = 0;
  int hi = rows - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    const int first =
        rowColumn.data[bounds.index(static_cast<size_t>(mid) * rowWidth + firstTransition, 1, rowColumn.limit)];
    if (first <= t) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// 遷移 t の牌 τ を返す。鳴いた後に捨てる牌、なければ主牌（ID、0 はなし）。
__device__ __forceinline__ int discardTile(const Span<const uint8_t>& category, int width, int callDiscard, int mainTile,
                                           int t, Bounds& bounds) {
  const size_t at = static_cast<size_t>(t) * width;
  const int afterCall = category.data[bounds.index(at + callDiscard, 1, category.limit)];
  const int primary = category.data[bounds.index(at + mainTile, 1, category.limit)];
  return afterCall > 0 ? afterCall : primary;
}

// 16 バイトに詰めたコードのうち byte 番目（0〜15）について、コードの表 [コード, A]（FP32）でその行が始まるバイト位置を返す。
// 行ごとに範囲を確かめる。
template <int A>
__device__ __forceinline__ uint32_t codeTableRow(const u32x4& codes, int byte, size_t limit, Bounds& bounds) {
  const uint32_t code = (codes[byte >> 2] >> (8 * (byte & 3))) & 0xffu;
  return static_cast<uint32_t>(bounds.index(static_cast<size_t>(code) * A, A, limit) * sizeof(float));
}

// Cross-Attention のキー 1 つ分の読み込み。キーのヘッドの 16 次元（K か V、BF16、b128 × 2）と、そのコードの E_k か E_v のヘッドの
// 16 次元（FP32、b128 × 4）を持つ。
struct KeySlot {
  u32x4 key[2];
  f32x4 code[4];

  // 読み込みを発行する。lane ごとに違う位置（キーの memoryLane と E の行の codeRow。どちらも項目のヘッドの位置を含む）は VGPR で、
  // ウェーブで一様な位置（キーの中の memoryByte）は soffset で渡す。
  __device__ __forceinline__ void issue(__amdgpu_buffer_rsrc_t memory, uint32_t memoryLane, int memoryByte,
                                        __amdgpu_buffer_rsrc_t table, uint32_t codeRow) {
    key[0] = __builtin_amdgcn_raw_buffer_load_b128(memory, memoryLane, memoryByte, 0);
    key[1] = __builtin_amdgcn_raw_buffer_load_b128(memory, memoryLane, memoryByte + 16, 0);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      code[w] = __builtin_bit_cast(f32x4, __builtin_amdgcn_raw_buffer_load_b128(table, codeRow, 16 * w, 0));
    }
  }

  // q·K を返す。v_dot2_f32_bf16 で、BF16 の対の積を FP32 の 1 つの累積に順に足す。
  __device__ __forceinline__ float keyDot(const u32x4 (&query)[2]) const {
    float dot = 0.0f;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        dot = __builtin_amdgcn_fdot2_f32_bf16(asPair(query[half][w]), asPair(key[half][w]), dot, false);
      }
    }
    return dot;
  }

  // q·E_k を返す。FMA で 1 つの累積に順に足す。
  __device__ __forceinline__ float codeDot(const float (&query)[2][8]) const {
    float dot = 0.0f;
#pragma unroll
    for (int d = 0; d < 16; ++d) {
      dot = fmaf(query[d >> 3][d & 7], code[d >> 2][d & 3], dot);
    }
    return dot;
  }

  // value += p·(V + E_v)。16 次元で、次元ごとにキーの順に FMA で足す。
  __device__ __forceinline__ void accumulate(float p, float (&value)[2][8]) const {
    float v[2][8];
    unpack8(key[0], v[0]);
    unpack8(key[1], v[1]);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        value[half][d] = fmaf(p, v[half][d] + code[2 * half + (d >> 2)][d & 3], value[half][d]);
      }
    }
  }
};

}  // namespace candidate
}  // namespace prism
