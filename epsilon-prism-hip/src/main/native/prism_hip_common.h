// HIP のコードで共通に使う部品。HIP のエラーの扱い、範囲検査つきビルドのインデックス、BF16 の丸め、WMMA のフラグメント、
// 生の buffer リソースを持つ。
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace prism {

// HIP の呼び出しが失敗したら、英語のメッセージで例外を投げる（JNI の実装が Java の例外に変える）。
inline void hipCheck(hipError_t status, const char* call) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " + hipGetErrorString(status));
  }
}

#define PRISM_HIP(call) ::prism::hipCheck((call), #call)

// gfx11 の WMMA は wave32 で動き、A・B のフラグメントは下と上の 16 lane が同じ値を持つ。
constexpr int kWave = 32;

using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
using f32x4 = float __attribute__((ext_vector_type(4)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using bf16x2 = __bf16 __attribute__((ext_vector_type(2)));
using bf16x16 = __bf16 __attribute__((ext_vector_type(16)));

// ext_vector の要素に __builtin_bit_cast を直接かけないこと。clang 22（ROCm 7.2.2）は bit_cast(T, v[i]) を、添字に関係なく要素 0 の
// 読み込みにしてしまう。要素はいったんスカラーの値として受けてから bit_cast する（asPair など）。

// float を BF16 に最近接偶数で丸める（c10::BFloat16 と同じ。NaN は 0x7FC0 にする）。NaN の判定を浮動小数点の比較で書くと、比較と選択の
// 2 命令で済む（ビット演算で書くと and・比較・選択の 3 命令）。NaN と判定する値は同じなので、結果はビットまで同じ。ホスト、K1、K7 と
// 学習はこの関数で丸める。
__host__ __device__ __forceinline__ uint16_t bf16Round(float value) {
  const uint32_t bits = __builtin_bit_cast(uint32_t, value);
  const uint16_t rounded = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  return __builtin_isnan(value) ? static_cast<uint16_t>(0x7fc0u) : rounded;
}

// GPU が作る活性値（K2〜K6）の丸め。bf16Round から NaN の選択を除いた (bits + 0x7FFF + lsb) >> 16（lsb = bit 16）で、NaN でない
// 入力では全て bf16Round とビットまで同じになる。NaN（指数が全て 1 で仮数が 0 でない）のうち、top7 = bit 22:16、low16 = bit 15:0 と
// すると、次の 2 つだけが NaN でなくなる（ほかは NaN のままだが、中身は 0x7FC0 にそろわない）:
//   top7 = 0 で low16 が 1〜0x8000: 桁上がりが起きずに仮数が 0 になり、同じ符号の inf になる
//   top7 = 0x7F で low16 ≥ 0x8000: 桁上がりが指数を越えて符号ビットに入り、符号が反転した 0 になる（+NaN は −0）
// K2〜K6 の丸めに届く NaN は、このどちらにも当たらないことを前提とする。入ってくる NaN は K1 が丸めた 0x7FC0 と K2〜K6 の活性値だけで
// （重みは公開のときに NaN を拒む）、K2〜K6 の演算はそこから quiet NaN（bit 22 が 1 なので top7 ≥ 0x40）しか作らない。bit 22:15 が
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
  return __builtin_amdgcn_perm(hiBits + 0x7fffu + ((hiBits >> 16) & 1u), loBits + 0x7fffu + ((loBits >> 16) & 1u),
                               0x07060302u);
}

__host__ __device__ __forceinline__ float bf16Float(uint16_t value) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(value) << 16);
}

// 2 つの BF16 を詰めた 1 ワード（下位が先）を v_dot2 のオペランドにする。ワードは値で受ける（ext_vector の要素を直接 bit_cast
// しない。上の注意を参照）。
__device__ __forceinline__ bf16x2 asPair(uint32_t bits) {
  return __builtin_bit_cast(bf16x2, bits);
}

// SiLU を FP32 で計算する（PyTorch の x / (1 + exp(-x)) と同じ式）。K7 はこちらを使う（eager と約 1e-6 で一致し、速い式にしても得が
// ない）。
__device__ __forceinline__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

// FFN の h（K3・K5・K6）と K6 の B のヘッドで使う SiLU。u·rcp(1 + 2^(−u·log2e)) を v_exp_f32・v_rcp_f32 で計算し、expf の範囲縮小と
// IEEE の割り算を省く。silu との差は FP32 の相対誤差で約 2e-6（|u| ≤ 30。主に u·log2e の 1 回の丸めによる）。−88.7 < u < −87.3 では、
// 約 −5e-37 になる値が −0 になる。値は要素ごとに決まるので、バッチの組み方によらずビットまで同じで、BF16 に丸める位置も変わらない。
__device__ __forceinline__ float siluFast(float value) {
  return value * __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(value * -1.4426950408889634f));
}

// WMMA の A・B のフラグメントを読む。1 行（A なら M、B なら N）の K 方向の 16 要素で、32 バイト（先頭は 16 バイト境界にそろうこと）。
__device__ __forceinline__ bf16x16 fragment(const uint16_t* row) {
  const u32x4 lo = *reinterpret_cast<const u32x4*>(row);
  const u32x4 hi = *reinterpret_cast<const u32x4*>(row + 8);
  return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

// D = A·B + C（16×16×16、BF16 の積を FP32 で累積）。C・D の要素 v には、lane l が列 l & 15・行 2v + (l >> 4) の値を持つ。
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

// 8 個の float を BF16 に丸めて 16 バイトに詰める（bf16Round）。
__device__ __forceinline__ u32x4 pack8(const float (&values)[8]) {
  u32x4 packed;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    packed[w] = static_cast<uint32_t>(bf16Round(values[2 * w])) | (static_cast<uint32_t>(bf16Round(values[2 * w + 1])) << 16);
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

// gfx11 の生の buffer リソースを作る。フォーマットのワードは CK と同じで、32 bit の生の読み込みになる。num_records を 2 GiB にするので、
// ハードウェアは範囲を実質的に検査しない。代わりに Plan が buffer で読む領域を 2 GiB 未満に限り、範囲検査つきのビルドでは lane の行の
// 位置を Bounds で確かめる。
template <class T>
__device__ __forceinline__ __amdgpu_buffer_rsrc_t bufferResource(const T* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(base), 0, 0x7fffffff, 0x31004000);
}

// 同じウェーブの lane が書いた LDS を、その後の読み込みから見えるようにする（ウェーブの中のバリア）。
__device__ __forceinline__ void waveSync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// ウェーブの 32 lane の和（xor のツリー。全ての lane が同じ値を得る。足す順は lane の並びだけで決まる）。
__device__ __forceinline__ float waveSum(float value) {
#pragma unroll
  for (int stride = kWave / 2; stride > 0; stride >>= 1) {
    value += __shfl_xor(value, stride);
  }
  return value;
}

// lane と相方（lane ^ 16）の値を入れ替える（v_permlanex16。ウェーブの全ての lane が有効であること）。
__device__ __forceinline__ uint32_t swapHalves(uint32_t value) {
  return __builtin_amdgcn_permlanex16(value, value, 0x76543210u, 0xfedcba98u, false, false);
}

__device__ __forceinline__ float swapHalves(float value) {
  return __builtin_bit_cast(float, swapHalves(__builtin_bit_cast(uint32_t, value)));
}

// 16 lane の組（C のフラグメントの同じ行）の中の和。
__device__ __forceinline__ float halfSum(float value) {
#pragma unroll
  for (int stride = 8; stride > 0; stride >>= 1) {
    value += __shfl_xor(value, stride);
  }
  return value;
}

// グローバルメモリのインデックスの範囲検査（-DPRISM_HIP_CHECK を付けたビルドだけ）。[begin, begin + count) が上限 limit（要素の数）を
// 外れたら、その読み書きを領域の先頭に向け、違反のフラグを立てる（範囲外には一度も触れない）。フラグが立っていれば、カーネルの最後の
// trap() で止める。分岐を作らないので、内側のループの命令の並べ替えを妨げない。通常のビルドでは begin をそのまま返し、フラグは持たない。
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
