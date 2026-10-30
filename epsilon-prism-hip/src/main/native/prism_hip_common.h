// 実行系の共通の部品: HIP の失敗の扱い、検査つきの組み立ての添字、BF16 の丸め、WMMA の断片、生の buffer の資源。
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace prism {

// HIP の呼び出しの失敗を英語の例外にする（JNI の口が Java の例外に替える）。
inline void hipCheck(hipError_t status, const char* call) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " + hipGetErrorString(status));
  }
}

#define PRISM_HIP(call) ::prism::hipCheck((call), #call)

// gfx11 の WMMA は wave32 で、A・B の断片を下と上の 16 lane に同じ値で持つ。
constexpr int kWave = 32;

using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
using f32x4 = float __attribute__((ext_vector_type(4)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using bf16x2 = __bf16 __attribute__((ext_vector_type(2)));
using bf16x16 = __bf16 __attribute__((ext_vector_type(16)));

// ext_vector の要素に __builtin_bit_cast を直にかけない: clang 22（ROCm 7.2.2）は bit_cast(T, v[i]) を添字に依らず要素 0 の読みに
// する。要素は一度スカラーとして値で受けてから写す（asPair など）。

// float を BF16 に最近接偶数で丸める（c10::BFloat16 と同じ。NaN は 0x7FC0）。NaN の判定を浮動小数の比較にすると比較と選択の 2 つで
// 済む（ビットで書くと and・比較・選択の 3 つ）。判定する集合は同じなので結果はビットで同じ。
// GPU が作る活性（K2〜K6）もこの関数（と pack8）で丸め、NaN の選択を残す。選択のない形 (bits + 0x7FFF + lsb) >> 16（lsb = bit 16）は
// NaN でない全ての入力でビットで同じだが、NaN（指数が全て 1 で仮数が 0 でない）は top7 = bit 22:16、low16 = bit 15:0 として次の
// 2 つだけが NaN でなくなる（それ以外は NaN のままで、中身は 0x7FC0 にそろわない）:
//   top7 = 0 で low16 が 1〜0x8000: 桁上がりがなく仮数が 0 になり、同じ符号の inf
//   top7 = 0x7F で low16 ≥ 0x8000: 桁上がりが指数を越えて符号の bit に入り、符号の反転した 0（+NaN は −0）
// 正規の 0x7FC00000 はどちらにも当たらない（top7 = 0x40）が、WMMA が返す NaN の payload は確かめられていないので選択を残す。
__host__ __device__ __forceinline__ uint16_t bf16Round(float value) {
  const uint32_t bits = __builtin_bit_cast(uint32_t, value);
  const uint16_t rounded = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  return __builtin_isnan(value) ? static_cast<uint16_t>(0x7fc0u) : rounded;
}

__host__ __device__ __forceinline__ float bf16Float(uint16_t value) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(value) << 16);
}

// 2 つの BF16 を詰めた 1 語（下位が先）を v_dot2 の被演算子にする。語は値で受ける（ext_vector の要素を直に写さない。上の注意）。
__device__ __forceinline__ bf16x2 asPair(uint32_t bits) {
  return __builtin_bit_cast(bf16x2, bits);
}

// SiLU を FP32 で（PyTorch の x / (1 + exp(-x)) と同じ式）。K7 はこちら（eager と約 1e-6 で合い、速い式にしても得がない）。
__device__ __forceinline__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

// FFN の h（K3・K5・K6）と K6 の B の頭の SiLU: u·rcp(1 + 2^(−u·log2e))（v_exp_f32・v_rcp_f32。expf の範囲の縮めと IEEE の割り算を
// 省く）。silu との差は FP32 の相対で約 2e-6（|u| ≤ 30。u·log2e の 1 回の丸めが主）、−88.7 < u < −87.3 では約 −5e-37 が −0 になる。
// 要素ごとに決まるので塊の組み方に依らずビットで同じで、BF16 に丸める位置も変わらない。
__device__ __forceinline__ float siluFast(float value) {
  return value * __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(value * -1.4426950408889634f));
}

// WMMA の A・B の断片: 1 行（A なら M、B なら N）の K 16 要素（32 バイト、16 バイトにそろう）を読む。
__device__ __forceinline__ bf16x16 fragment(const uint16_t* row) {
  const u32x4 lo = *reinterpret_cast<const u32x4*>(row);
  const u32x4 hi = *reinterpret_cast<const u32x4*>(row + 8);
  return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

// D = A·B + C（16×16×16、BF16 の積を FP32 で累積）。C・D は lane l が列 l & 15、行 2v + (l >> 4) を持つ。
__device__ __forceinline__ f32x8 mma(bf16x16 a, bf16x16 b, f32x8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

__device__ __forceinline__ u32x4 loadUnaligned16(const uint8_t* bytes) {
  u32x4 value;
  __builtin_memcpy(&value, bytes, sizeof(value));
  return value;
}

// 8 個の BF16（16 バイト）を float にする。
__device__ __forceinline__ void unpack8(u32x4 packed, float (&values)[8]) {
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    values[2 * w] = bf16Float(static_cast<uint16_t>(packed[w] & 0xffffu));
    values[2 * w + 1] = bf16Float(static_cast<uint16_t>(packed[w] >> 16));
  }
}

// 8 個の float を BF16 に丸めて 16 バイトに詰める。
__device__ __forceinline__ u32x4 pack8(const float (&values)[8]) {
  u32x4 packed;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    packed[w] = static_cast<uint32_t>(bf16Round(values[2 * w])) | (static_cast<uint32_t>(bf16Round(values[2 * w + 1])) << 16);
  }
  return packed;
}

// gfx11 の生の buffer の資源（形式の語は CK と同じ 32 bit の生の読み）。num_records は 2 GiB で範囲は実質照らさない: 計画が buffer で
// 読む区画を 2 GiB 未満に限り、検査つきの組み立ては lane の行の位置を Bounds で照らす。
template <class T>
__device__ __forceinline__ __amdgpu_buffer_rsrc_t bufferResource(const T* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(base), 0, 0x7fffffff, 0x31004000);
}

// 同じ波の lane の間の LDS の書き込みを、後の読み込みより先に見せる（波の中の待ち合わせ）。
__device__ __forceinline__ void waveSync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// 波の 32 lane の和（xor の木。全ての lane が同じ値を得る。和の順は lane の並びだけで決まる）。
__device__ __forceinline__ float waveSum(float value) {
#pragma unroll
  for (int stride = kWave / 2; stride > 0; stride >>= 1) {
    value += __shfl_xor(value, stride);
  }
  return value;
}

// 波の 32 lane の最大。
__device__ __forceinline__ float waveMax(float value) {
#pragma unroll
  for (int stride = kWave / 2; stride > 0; stride >>= 1) {
    value = fmaxf(value, __shfl_xor(value, stride));
  }
  return value;
}

// lane と相方（lane ^ 16）の値を入れ替える（v_permlanex16。波の全ての lane が動いていること）。
__device__ __forceinline__ uint32_t swapHalves(uint32_t value) {
  return __builtin_amdgcn_permlanex16(value, value, 0x76543210u, 0xfedcba98u, false, false);
}

__device__ __forceinline__ float swapHalves(float value) {
  return __builtin_bit_cast(float, swapHalves(__builtin_bit_cast(uint32_t, value)));
}

// 同じ 16 lane の組（C の断片の同じ行）の和。
__device__ __forceinline__ float halfSum(float value) {
#pragma unroll
  for (int stride = 8; stride > 0; stride >>= 1) {
    value += __shfl_xor(value, stride);
  }
  return value;
}

// 全体メモリの添字の検査（-DPRISM_HIP_CHECK の組み立てだけ）。[begin, begin + count) が上限 limit（要素の数）を外れたら、
// その読み書きを区画の先頭に向けて印を立てる（範囲外には一度も触れない）。印は段の終わりの trap() で止める。分岐を作らないので、
// 内側の輪の並べ替えを妨げない。通常の組み立てでは begin をそのまま返し、印は消える。
struct Bounds {
#ifdef PRISM_HIP_CHECK
  bool violated = false;

  __device__ __forceinline__ size_t index(size_t begin, size_t count, size_t limit) {
    const bool inside = begin + count <= limit;
    violated |= !inside;
    return inside ? begin : 0;
  }

  __device__ __forceinline__ void trap() const {
    if (violated) {
      __builtin_trap();
    }
  }
#else
  __device__ __forceinline__ size_t index(size_t begin, size_t, size_t) { return begin; }

  __device__ __forceinline__ void trap() const {}
#endif
};

}  // namespace prism
