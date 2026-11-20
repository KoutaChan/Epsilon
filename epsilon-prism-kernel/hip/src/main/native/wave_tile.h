#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace prism {

// gfx11 の WMMA は wave32 で動き、A・B のフラグメントは下と上の 16 lane が同じ値を持つ。
constexpr int kWave = 32;

using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
using f32x4 = float __attribute__((ext_vector_type(4)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using bf16x2 = __bf16 __attribute__((ext_vector_type(2)));
using bf16x16 = __bf16 __attribute__((ext_vector_type(16)));

// 行列積のタイルは 16 行 × 16 列で、K を 16 要素ずつ累積する。
constexpr int kMatrixTile = 16;
__device__ __forceinline__ int fragmentColumnOfLane(int lane) { return lane & (kMatrixTile - 1); }
__device__ __forceinline__ int fragmentParityOfLane(int lane) { return lane >> 4; }
__device__ __forceinline__ int accumulatorRow(int element, int parity) { return 2 * element + parity; }

// ext_vector の要素に __builtin_bit_cast を直接かけないこと。clang 22（ROCm 7.2.2）は bit_cast(T, v[i]) を、添字に関係なく要素 0 の
// 読み込みにしてしまう。要素はいったんスカラーの値として受けてから bit_cast する（asPair など）。

// ウェーブで一様な値を SGPR に移す。仕様のフィールドに使うと表・特徴のループの分岐がスカラーになり、トークンに使うと行の先頭をスカラー
// 命令で計算できる。
__device__ __forceinline__ int uniform(int value) { return __builtin_amdgcn_readfirstlane(value); }

// ウェーブの中で lane 番目の lane が持つ value を返す（結果はウェーブで一様）。
__device__ __forceinline__ uint32_t readLane(uint32_t value, int lane) {
  return __builtin_amdgcn_readlane(value, lane);
}

__device__ __forceinline__ float readLane(float value, int lane) {
  return __builtin_bit_cast(float, readLane(__builtin_bit_cast(uint32_t, value), lane));
}

// 2 つの FP32 の上位 16 bit を 1 ワードに詰める（a が下、b が上。v_perm 1 命令）。
__device__ __forceinline__ uint32_t upperHalves(uint32_t a, uint32_t b) {
  return __builtin_amdgcn_perm(b, a, 0x07060302u);
}

// lane ごとの 8 値（16 バイト）から B のフラグメントを作る。下の 16 lane が K の 0〜7、上の 16 lane が 8〜15 を持つとき、どの lane も
// [lane & 15 の 8 値, lane | 16 の 8 値] を得る。ビットマスクの形の ds_swizzle（and 0x0F、or 0x10）は LDS の crossbar を通るだけで、
// bank には触れない。
__device__ __forceinline__ bf16x16 joinHalves(u32x4 own) {
  u32x4 first;
  u32x4 second;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    first[w] = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(own[w]), 0x000F));
    second[w] = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(own[w]), 0x021F));
  }
  return __builtin_bit_cast(bf16x16, __builtin_shufflevector(first, second, 0, 1, 2, 3, 4, 5, 6, 7));
}

// 2 つの BF16 を詰めた 1 ワード（下位が先）を v_dot2 のオペランドにする。ワードは値で受ける（ext_vector の要素を直接 bit_cast
// しない。上の注意を参照）。
__device__ __forceinline__ bf16x2 asPair(uint32_t bits) { return __builtin_bit_cast(bf16x2, bits); }

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

using BufferResource = __amdgpu_buffer_rsrc_t;

// 先頭から読める要素数を受け取る。バイト数は実装選択時に確認した符号付き 32 ビットの範囲内にある。
// 通常ビルドではループ外の範囲検査を利用し、同じ定数の記述子を使う。検査ビルドではハードウェアにも実際の境界を渡す。
template <class T>
__device__ __forceinline__ BufferResource bufferResource(const T* base, size_t elements) {
#if PRISM_CHECKED
  const uint32_t bytes = static_cast<uint32_t>(elements * sizeof(T));
#else
  (void)elements;
  constexpr uint32_t bytes = 0x7fffffff;
#endif
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(base), 0, bytes, 0x31004000);
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

// 参加マスクのうち、実行中で述語が真のレーンを下位ビットから順に返す。
__device__ __forceinline__ uint32_t vote(bool predicate, uint32_t participants = 0xffffffffu) {
  return __builtin_amdgcn_ballot_w32(predicate) & participants;
}

// 命令の並べ替えをこの位置で区切る。メモリの可視性は変えない。
__device__ __forceinline__ void scheduleBoundary() { __builtin_amdgcn_sched_barrier(0); }

// ワークグループの全スレッドを待ち、各スレッドが書いた LDS を全員に公開する。
__device__ __forceinline__ void shareLocalWrites() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// ワークグループの全員を待ち、グローバルメモリと LDS への先行する書き込みを同じグループに公開する。
__device__ __forceinline__ void shareGroupWrites() { __syncthreads(); }

// 全レーンの到着だけを待つ。
__device__ __forceinline__ void waveArrive() { __builtin_amdgcn_wave_barrier(); }

// 位置はバイト単位。呼び出し側は、レーンごとの位置と共通の変位の和から 16 バイトを読めることを保証する。
__device__ __forceinline__ u32x4 bufferLoad16(BufferResource resource, uint32_t lane, int displacement) {
  return __builtin_amdgcn_raw_buffer_load_b128(resource, lane, displacement, 0);
}
// 先読みに使う 4 バイトを返す。
__device__ __forceinline__ uint32_t bufferLoad4(BufferResource resource, uint32_t lane, int displacement) {
  return __builtin_amdgcn_raw_buffer_load_b32(resource, lane, displacement, 0);
}

// 生の buffer から 32 バイト（WMMA のフラグメント 1 つ）を 16 バイトずつ読む。offset は lane ごとのバイト位置（VGPR）。low・high は
// 下と上の 16 バイトの、ウェーブで同じ位置（soffset。上の半分の差も soffset に入れて、ループの中に VGPR の足し算を作らない）。
__device__ __forceinline__ bf16x16 bufferFragment(BufferResource resource, uint32_t offset, int low, int high) {
  const u32x4 lo = bufferLoad16(resource, offset, low);
  const u32x4 hi = bufferLoad16(resource, offset, high);
  return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

__device__ __forceinline__ float dotPair(bf16x2 a, bf16x2 b, float sum) {
  return __builtin_amdgcn_fdot2_f32_bf16(a, b, sum, false);
}

// 値をコンパイラから隠す（命令は出ない）。キーの段とバリューの段は、同じキーのアドレスを同じ式で作る。隠さないと、コンパイラがキーの段で
// 作ったアドレスをバリューの段まで持ち続ける（キーごとに VGPR 2 本）。
template <class T>
__device__ __forceinline__ void opaque(T& value) {
  asm volatile("" : "+v"(value));
}

// 先読みした 3 ワードをこの位置までレジスタに残す。
__device__ __forceinline__ void retainPrefetch(const uint32_t (&words)[3]) {
  asm volatile("" ::"v"(words[0]), "v"(words[1]), "v"(words[2]));
}

// レーン 0 が参加している組で、その値を参加レーンに配る。
template <class T>
__device__ __forceinline__ T broadcastFirst(T value) {
  return __shfl(value, 0, kWave);
}
template <class T>
__device__ __forceinline__ T shuffleDown(T value, int distance) {
  return __shfl_down(value, distance, kWave);
}

}  // namespace prism
