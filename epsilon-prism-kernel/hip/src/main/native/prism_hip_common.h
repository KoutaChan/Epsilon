#pragma once

#include "device_arithmetic.h"
#include "hip_error.h"
#include "wave_tile.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace prism {

// float を BF16 に最近接偶数で丸める（c10::BFloat16 と同じ。NaN は 0x7FC0 にする）。NaN の判定を浮動小数点の比較で書くと、比較と選択の
// 2 命令で済む（ビット演算で書くと and・比較・選択の 3 命令）。NaN と判定する値は同じなので、結果はビットまで同じ。ホスト側の計算、埋め込み、価値ヘッド、
// 学習はこの関数で丸める。
__host__ __device__ __forceinline__ uint16_t bf16Round(float value) {
  const uint32_t bits = __builtin_bit_cast(uint32_t, value);
  const uint16_t rounded = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  return __builtin_isnan(value) ? static_cast<uint16_t>(0x7fc0u) : rounded;
}

// Attention と FFN の活性値を BF16 に丸める。bf16Round から NaN の選択を除いた (bits + 0x7FFF + lsb) >> 16（lsb = bit 16）で、NaN でない
// 入力では全て bf16Round とビットまで同じになる。NaN（指数が全て 1 で仮数が 0 でない）のうち、top7 = bit 22:16、low16 = bit 15:0 と
// すると、次の 2 つだけが NaN でなくなる（ほかは NaN のままだが、中身は 0x7FC0 にそろわない）:
//   top7 = 0 で low16 が 1〜0x8000: 桁上がりが起きずに仮数が 0 になり、同じ符号の inf になる
//   top7 = 0x7F で low16 ≥ 0x8000: 桁上がりが指数を越えて符号ビットに入り、符号が反転した 0 になる（+NaN は −0）
// この丸めに届く NaN は、このどちらにも当たらないことを前提とする。入力に含まれる NaN は、埋め込みで丸めた 0x7FC0 と、Attention または FFN の活性値に限られる
// （重みの公開時に NaN を拒む）。後続の演算はそこから quiet NaN（bit 22 が 1 なので top7 ≥ 0x40）しか作らない。bit 22:15 が
// 全て 1 の NaN も作らない。ほかの NaN が丸めに届く演算を足すときは、そこを bf16Round で丸めること。
__host__ __device__ __forceinline__ uint16_t bf16RoundActivation(float value) {
  const uint32_t bits = __builtin_bit_cast(uint32_t, value);
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// 2 つの float を bf16RoundActivation で丸めて 1 ワードに詰める（lo が下位、hi が上位）。丸めの和 bits + 0x7FFF + lsb の上位 16 bit を
// v_perm 1 命令で選ぶ。結果は bf16RoundActivation(lo) | bf16RoundActivation(hi) << 16 とビットまで同じで、シフトと and_or の 2 命令が
// 1 命令で済む。
__device__ __forceinline__ uint32_t bf16RoundActivationPair(float lo, float hi) {
  const uint32_t loBits = __builtin_bit_cast(uint32_t, lo);
  const uint32_t hiBits = __builtin_bit_cast(uint32_t, hi);
  return upperHalves(loBits + 0x7fffu + ((loBits >> 16) & 1u), hiBits + 0x7fffu + ((hiBits >> 16) & 1u));
}

__host__ __device__ __forceinline__ float bf16Float(uint16_t value) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(value) << 16);
}

// SiLU を FP32 で計算する（PyTorch の x / (1 + exp(-x)) と同じ式）。価値ヘッドではこの式を使う。
__device__ __forceinline__ float silu(float value) { return value / (1.0f + expf(-value)); }

// 幹と候補層の FFN、および候補層の打牌価値のヘッドで使う SiLU。u·rcp(1 + 2^(−u·log2e)) を v_exp_f32・v_rcp_f32 で計算し、expf の範囲縮小と
// IEEE の割り算を省く。silu との差は FP32 の相対誤差で約 2e-6（|u| ≤ 30。主に u·log2e の 1 回の丸めによる）。−88.7 < u < −87.3 では、
// 約 −5e-37 になる値が −0 になる。値は要素ごとに決まるので、バッチの組み方によらずビットまで同じで、BF16 に丸める位置も変わらない。
__device__ __forceinline__ float siluFast(float value) {
  return value * fastReciprocal(1.0f + fastExp2(value * -1.4426950408889634f));
}

// 8 個の BF16（16 バイト）を float にする。
__device__ __forceinline__ void unpack8(u32x4 packed, float (&values)[8]) {
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    values[2 * w] = bf16Float(static_cast<uint16_t>(packed[w] & 0xffffu));
    values[2 * w + 1] = bf16Float(static_cast<uint16_t>(packed[w] >> 16));
  }
}

// 8 個の float を BF16 に丸めて 16 バイトに詰める（bf16Round）。
__device__ __forceinline__ u32x4 pack8(const float (&values)[8]) {
  u32x4 packed;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    packed[w] =
        static_cast<uint32_t>(bf16Round(values[2 * w])) | (static_cast<uint32_t>(bf16Round(values[2 * w + 1])) << 16);
  }
  return packed;
}

// 8 個の float を活性値の丸め（bf16RoundActivation）で BF16 にして 16 バイトに詰める。
__device__ __forceinline__ u32x4 pack8Activation(const float (&values)[8]) {
  u32x4 packed;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    packed[w] = static_cast<uint32_t>(bf16RoundActivation(values[2 * w])) |
                (static_cast<uint32_t>(bf16RoundActivation(values[2 * w + 1])) << 16);
  }
  return packed;
}

// グローバルメモリのインデックスの範囲検査（PRISM_CHECKED=1 のビルドだけ）。[begin, begin + count) が上限 limit（要素の数）を
// 外れたら、その読み書きを領域の先頭に向け、違反のフラグを立てる（範囲外には一度も触れない）。フラグが立っていれば、カーネルの最後の
// trap() で止める。分岐を作らないので、内側のループの命令の並べ替えを妨げない。通常のビルドでは begin をそのまま返し、フラグは持たない。
struct Bounds {
#if PRISM_CHECKED
  bool violated = false;

  __device__ __forceinline__ size_t index(size_t begin, size_t count, size_t limit) {
    const bool inside = begin <= limit && count <= limit - begin;
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
