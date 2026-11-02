// 学習の注意（関係の偏りつき、相対の値なし）の順伝播と逆伝播（WMMA、FlashAttention-2 の形）。GPU の学習で、eager の EncoderLayer の
// packedRelationScaledDotProductAttention（djl-rocm）の代わりに使う。数式は同じ:
//
//   ℓ_ij = (q_i·k_j)·scale + Σ_s ρ[entries·s + c_s(i, j), h]（鍵 j が在るときだけ）、p_i = softmax_j(ℓ_i)、o_i = Σ_j p_ij v_j
//
// 内側は ℓ₂ = ℓ·log₂e の単位で計算する（exp2）。偏りは区画 2 つずつ前もって足した組の表 τ_t[a + 16b] = (ρ(2t, a) + ρ(2t + 1, b))·log₂e
// を符号の byte t で 4 回引き、((τ₀ + τ₁) + τ₂) + τ₃ と足す。作業群は頭の ρ の列（区画 × 値の 128 個、範囲外は 0）をスレッドで分けて
// 1 回で LDS に置き、その LDS から表を作る（表の 1 項ごとに全体メモリを読まない）。順伝播と逆伝播は同じ式なので ℓ₂ はビットで同じ。
//
// WMMA に入る P・dS は FP32 を BF16 の 3 項 hi + mid + lo（どれも上位 16 bit の切り捨て）に分けて 3 回掛ける。3 項の和は元の FP32 と
// 等しく、BF16 の q・k・v・do との積も FP32 で正確なので、変わるのは FP32 の和の順だけ。q·k と do·v は BF16 どうしの積で、そのまま掛ける。
//
// 順伝播（作業群は頭の対 × 1 行、波は問い合わせの塊 16 個。K2 の核と同じ形）:
//   S^T = K·Qᵀ、ℓ₂ = S^T·scale·log₂e + τ、オンラインの softmax、O^T += V^T·P^T。o は BF16 に 1 回丸め、逆伝播のために FP32 の o と
//   log Σ exp = (m₂ + log₂ Σ)·ln 2 も書く。
// 逆伝播（作業群は 1 頭 × 1 行。問い合わせの塊ごとに 2 段で、段の間は作業群の待ち合わせ）:
//   段 A（波 = 鍵の塊 kt。lane が鍵、8 つの問い合わせ）: S = Q·Kᵀ、dP = dO·Vᵀ、P = 2^(ℓ₂ − lse₂)、dS = P·(dP − δ)、
//        dV^T += dO^T·P、dK^T += Q^T·dS。dS と符号を LDS の問い合わせの塊の行に置く。
//   段 B: 波 0〜3 が区画 2 つずつ ρ の勾配を lane ごとの部分和に足し（原子的な加算なし）、波 4 が dQ^T = K^T·dS^T（鍵の塊の順）を
//        書いて、次の問い合わせの塊の Q^T・dO^T を LDS に置く（全体メモリの読みは dQ の前に出し、待ちを dQ に重ねる）。
//   dS と符号の LDS の行の刻みは 64 を法として 36 語: 段 B の 16 行 × 2 つの半分の 16 バイトの読みが 2-way（最小）、段 A の 2 つの半分の
//   書き込みは bank が重ならない。
//   δ_i = Σ_d do_id·o_id（FP32 の o）。dq・dk・dv は FP32 で作り、ちょうど 0 の和を +0 にそろえて（positiveZero）BF16 に最近接偶数で 1 回
//   丸める（c10 の型の変換と同じビット）。ρ の勾配は作業群（行 × 頭）ごとの部分和を FP32 で書き、呼び手が行で足す（決まった順の縮約）。
//   生きた問い合わせ（i < queries で do_i の 16 値に 0 でない値がある）だけを回す: do_i = 0 なら dP_i = δ_i = 0、dS_i = 0 で、dk・dv・dq・
//   ρ の勾配への寄与はちょうど 0（近似ではない）。死んだ問い合わせは問い合わせの外と同じに扱い（Q^T・dO^T は 0、P = +0、dS = +0、
//   ρ の勾配に足さない、dq は +0）、生きた問い合わせを含む最後の塊より後ろの塊は回さない。
// 縮約は全て作業群の中で順が決まっている（WMMA の累積、鍵の塊の順、lane の部分和の和の順）。在る鍵の dk・dv に入る WMMA は、詰め物の
// 問い合わせ（出力の勾配がちょうど 0）を足しても同じ順で、A・B・C がビットで同じ（詰め物の q は WMMA に入らない。0 の積でも
// gfx1100 の WMMA は A の値で実の積の丸めを変えうる）。よって結果は行ごとに塊の組み方と詰め物の幅に依らずビットで同じ（詰め物の
// 出力の勾配がちょうど 0 であることが約束）。
// 全体メモリの位置は一様な行の先頭と lane の 32 bit の位置に分ける（読み書きが SGPR の先頭と VGPR の位置になる）。
// 検査つきの組み立てでは行の区画と lane の位置の両方を照らす（全ての全体メモリの読み書き）。
// 範囲外の符号（区画 s < segments の値が entries 以上）は止める。照らすのは在る鍵を 1 つでも含む鍵の塊の全ての符号（塊の中の在らない
// 鍵と寄せた問い合わせの外の行を含む。全く在らない塊は読まない）。T0 は詰め物の組にも範囲内の符号を書き、entries = 16 なら照らす処理は
// 一様な分岐で飛ぶ。
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "prism_hip_common.h"
#include "prism_hip_training.h"

namespace prism {
namespace {

constexpr int kF = kAttentionFeatures;
constexpr int kTables = 4;                   // 組の表（符号の byte t ごと、区画 2t と 2t + 1）
constexpr int kTableEntries = 256;           // 組の表の項（byte の値）
constexpr int kRhoEntries = kAttentionMaxSegments * kAttentionMaxEntries;  // LDS の頭の ρ の列（区画 × 値）
constexpr int kScatterWaves = 4;             // 段 B で ρ の勾配を足す波（区画 2 つずつ）
constexpr int kServiceWave = kScatterWaves;  // 段 B で dQ と次の Q^T・dO^T を受け持つ波
constexpr int kMinBackwardWaves = kServiceWave + 1;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
static_assert(2 * kScatterWaves == kAttentionMaxSegments, "the scatter waves cover every segment of a code word");
static_assert(2 * kTables == kAttentionMaxSegments, "the pair tables cover every segment of a code word");
static_assert(kAttentionMaxEntries == 16, "a segment code is one nibble");

// 鍵の塊の上限の段（LDS の大きさ。S ≤ 16·kBlocks）。
constexpr int kSmallBlocks = 5;
constexpr int kMediumBlocks = 8;
constexpr int kLargeBlocks = kAttentionMaxTokens / 16;

// 次元 d を A の断片の行 r(d) = 2(d & 7) + (d >> 3) に置くと、C の断片の行 2v + half が次元 v + 8·half になる（lane が続いた 8 次元を
// 持ち、16 バイトで書ける）。
__device__ __forceinline__ int featureRow(int d) {
  return 2 * (d & 7) + (d >> 3);
}

// C の断片の行 m = 2v + half の値を B の断片の K の位置 κ(m) = v + 8·half に置く（lane の 8 値と相方の 8 値を続けるだけで B の断片になる）。
__device__ __forceinline__ int fragmentColumn(int m) {
  return (m >> 1) + 8 * (m & 1);
}

// 作業群の待ち合わせ（LDS の順序だけ。t1 はカーネルの中で全体メモリに書いた値を読まない）。
__device__ __forceinline__ void barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// 2 つの FP32 の上位 16 bit を 1 語に詰める（a が下、b が上）。
__device__ __forceinline__ uint32_t upperHalves(uint32_t a, uint32_t b) {
  return __builtin_amdgcn_perm(b, a, 0x07060302u);
}

// B の断片を lane の 8 値（16 バイト）から作る: 下の 16 lane が K の 0〜7、上の 16 lane が 8〜15 を持つとき、どの lane も
// [lane & 15 の 8 値, lane | 16 の 8 値] を得る（ds_swizzle の bit の型は LDS の crossbar を通るだけで bank に触れない）。
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

// lane の 8 つの FP32 を BF16 の 3 項 hi + mid + lo（どれも上位 16 bit の切り捨て）に分け、相方の 8 値と合わせて B の断片 3 つにする。
// hi が仮数の上位 8 bit、mid が残りの上位 8 bit、lo が残り（8 bit 以下）を持つので、3 項の和は元の値と等しい（BF16 の正規数の範囲）。
__device__ __forceinline__ void splitFragments(const float (&x)[8], bf16x16 (&terms)[3]) {
  u32x4 own[3];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    float a = x[2 * w];
    float b = x[2 * w + 1];
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      const uint32_t ua = __builtin_bit_cast(uint32_t, a);
      const uint32_t ub = __builtin_bit_cast(uint32_t, b);
      own[t][w] = upperHalves(ua, ub);
      a -= __builtin_bit_cast(float, ua & 0xffff0000u);
      b -= __builtin_bit_cast(float, ub & 0xffff0000u);
    }
  }
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    terms[t] = joinHalves(own[t]);
  }
}

// acc += A·(lo + mid + hi)（小さい項から）。
__device__ __forceinline__ f32x8 mma3(bf16x16 a, const bf16x16 (&terms)[3], f32x8 acc) {
  acc = mma(a, terms[2], acc);
  acc = mma(a, terms[1], acc);
  return mma(a, terms[0], acc);
}

// 16 バイトの値 2 つを 8 値の配列に並べる。
template <class T, class V>
__device__ __forceinline__ void unpackEight(V first, V second, T (&values)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = first[i];
    values[4 + i] = second[i];
  }
}

// 頭 head の ρ の列を LDS の rho[16s + c] に置く（s < segments かつ c < entries のときだけ表の値、ほかは 0）。作業群のスレッドが
// 添字を分けて 1 回で読む。
__device__ __forceinline__ void stageRho(float* rho, const AttentionShape& shape, Span<const float> table, int head,
                                         int tid, int threads, Bounds& bounds) {
  for (int e = tid; e < kRhoEntries; e += threads) {
    const int s = e / kAttentionMaxEntries;
    const int c = e % kAttentionMaxEntries;
    float value = 0.0f;
    if (s < shape.segments && c < shape.entries) {
      const size_t at = (static_cast<size_t>(s) * shape.entries + c) * shape.heads + head;
      value = table.data[bounds.index(at, 1, table.limit)];
    }
    rho[e] = value;
  }
}

// 組の表の項 τ_t[a + 16b] = (ρ(2t, a) + ρ(2t + 1, b))·log₂e（e = 256t + a + 16b。rho は stageRho の列）。
__device__ __forceinline__ float pairEntry(const float* rho, int e) {
  const int t = e / kTableEntries;
  const float first = rho[kAttentionMaxEntries * (2 * t) + (e & 15)];
  const float second = rho[kAttentionMaxEntries * (2 * t + 1) + ((e >> 4) & 15)];
  return (first + second) * kLog2e;
}

// 符号 code の偏り ((τ₀ + τ₁) + τ₂) + τ₃（×log₂e）。
__device__ __forceinline__ float pairBias(uint32_t code, const float* table) {
  return ((table[code & 0xffu] + table[kTableEntries + ((code >> 8) & 0xffu)]) +
          table[2 * kTableEntries + ((code >> 16) & 0xffu)]) +
         table[3 * kTableEntries + (code >> 24)];
}

// 2 頭の表から（頭ごとに pairBias と同じ順）。
__device__ __forceinline__ float2 pairBias(uint32_t code, const float2* table) {
  const float2 t0 = table[code & 0xffu];
  const float2 t1 = table[kTableEntries + ((code >> 8) & 0xffu)];
  const float2 t2 = table[2 * kTableEntries + ((code >> 16) & 0xffu)];
  const float2 t3 = table[3 * kTableEntries + (code >> 24)];
  return make_float2(((t0.x + t1.x) + t2.x) + t3.x, ((t0.y + t1.y) + t2.y) + t3.y);
}

// 範囲外の符号（区画 s < segments の値が entries 以上）は djl-rocm の演算と同じく止める。4 bit の値を 8 bit の欄 2 つ（偶数・奇数の
// 区画）に分けて 128 − entries を足すと、欄の最上位 bit が「値 ≥ entries」になる。entries が 16 なら全て範囲内なので、呼び手は塊ごとに
// 1 回の一様な分岐で飛ばす（内側の輪を 1 つの基本塊のまま保つ）。
__device__ __forceinline__ void checkCodes(const uint32_t (&code)[8], int segments, int entries) {
  const uint32_t bias = static_cast<uint32_t>(128 - entries) * 0x01010101u;
  uint32_t even = 0;
  uint32_t odd = 0;
#pragma unroll
  for (int t = 0; t < kTables; ++t) {
    even |= 2 * t < segments ? 0x80u << (8 * t) : 0u;
    odd |= 2 * t + 1 < segments ? 0x80u << (8 * t) : 0u;
  }
  uint32_t outside = 0;
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    outside |= (((code[v] & 0x0f0f0f0fu) + bias) & even) | ((((code[v] >> 4) & 0x0f0f0f0fu) + bias) & odd);
  }
  if (outside != 0) {
    __builtin_trap();
  }
}

// 検査つきの組み立ての上限つきの配列（AttentionForward・AttentionBackward と同じ並び。codes は 1 組 1 語）。
struct ForwardArgs {
  AttentionShape shape;
  Span<const uint16_t> query;
  Span<const uint16_t> keyValue;
  Span<const float> mask;
  Span<const uint32_t> codes;
  Span<const float> table;
  Span<uint16_t> out;
  Span<float> outFloat;
  Span<float> logSumExp;
};

struct BackwardArgs {
  AttentionShape shape;
  Span<const uint16_t> query;
  Span<const uint16_t> keyValue;
  Span<const float> mask;
  Span<const uint32_t> codes;
  Span<const float> table;
  Span<const float> outFloat;
  Span<const float> logSumExp;
  Span<const uint16_t> gradOut;
  Span<uint16_t> gradQuery;
  Span<uint16_t> gradKeyValue;
  Span<float> gradTable;
};

// 行の区画: 一様な先頭のアドレスと、lane の 32 bit の位置の読み書き（全体メモリの読み書きが SGPR の先頭と VGPR の 32 bit のバイトの
// 位置になる）。検査つきの組み立てでは、rowOf が区画 [first, first + size) を span の中か照らし、at が読み書きのたびに lane の
// [offset, offset + count) を区画の中か照らす（lane の位置は寄せた添字から作るので、通常は収まる）。通常の組み立てでは size は消える。
template <class T>
struct Row {
  T* data;
  size_t size;

  __device__ __forceinline__ T* at(uint32_t offset, uint32_t count, Bounds& bounds) const {
    using Byte = std::conditional_t<std::is_const_v<T>, const char, char>;
    const uint32_t checked = static_cast<uint32_t>(bounds.index(offset, count, size));
    return reinterpret_cast<T*>(reinterpret_cast<Byte*>(data) + checked * static_cast<uint32_t>(sizeof(T)));
  }
};

// span の [first, first + size) の区画。
template <class T>
__device__ __forceinline__ Row<T> rowOf(Span<T> span, size_t first, size_t size, Bounds& bounds) {
  return Row<T>{span.data + bounds.index(first, size, span.limit), size};
}

// 鍵の塊 kt の在否の 16 bit（鍵 16kt + m が bit m。在否の 0 でない鍵で、keys より後ろは在らない）を LDS の keyBits[kt] に置く（波で
// 塊を分け、ballot の下の 16 bit）。
__device__ __forceinline__ void stageKeyBits(uint32_t* keyBits, Row<const float> maskRow, int keys, int wave, int waves,
                                             int lane, Bounds& bounds) {
  const int blocks = (keys + 15) / 16;
  for (int kt = wave; kt < blocks; kt += waves) {
    const int key = 16 * kt + (lane & 15);
    const bool present = key < keys && *maskRow.at(static_cast<uint32_t>(min(key, keys - 1)), 1, bounds) != 0.0f;
    const uint32_t bits = __builtin_amdgcn_ballot_w32(present) & 0xffffu;
    if (lane == 0) {
      keyBits[kt] = bits;
    }
  }
}

// LDS。V^T は K2 と同じ並び: 頭 h の行 16h + r(d)、鍵 16kt + m は列 16kt + κ(m)（在らない鍵は 0）。符号の写しは波ごとの鍵の塊 1 つで、
// 問い合わせ n の行の列 κ(m) に鍵 m の符号（lane が持つ 8 鍵 2v + half が 8·half から続く）。
template <int kBlocks>
struct alignas(16) ForwardShared {
  uint16_t valueT[2 * kF][16 * kBlocks + 8];
  float rho[2][kRhoEntries];             // 2 頭の ρ の列（stageRho）
  float2 bias[kTables * kTableEntries];  // 2 頭の組の表
  uint32_t codes[kBlocks][16][16];
  uint32_t keyBits[kBlocks];
};

template <int kBlocks>
__global__ __launch_bounds__(kWave * kBlocks) void forwardKernel(ForwardArgs args) {
  static_assert(sizeof(ForwardShared<kBlocks>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  __shared__ ForwardShared<kBlocks> shared;

  const AttentionShape shape = args.shape;
  const int pair = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int waves = threads / kWave;
  const int lane = tid % kWave;
  const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
  const int column = lane & 15;  // A・B の断片の行、C の断片の列
  const int half = lane >> 4;    // C の断片で奇数の行を持つ lane
  const int width = shape.heads * kF;
  const int keys = shape.keys;
  const int queries = shape.queries;
  const int blocks = (keys + 15) / 16;
  const float scale = shape.scale * kLog2e;
  Bounds bounds;
  // 行の区画（問い合わせ・鍵と値は頭の対の列から行の終わりまで）。
  const int columns = 2 * pair * kF;
  const Row<const uint16_t> queryRow =
      rowOf(args.query, static_cast<size_t>(row) * queries * width + columns, queries * width - columns, bounds);
  const Row<const uint16_t> keyRow =
      rowOf(args.keyValue, static_cast<size_t>(row) * keys * 2 * width + columns, keys * 2 * width - columns, bounds);
  const Row<const uint32_t> codeRow =
      rowOf(args.codes, static_cast<size_t>(row) * shape.codeQueries * keys,
            static_cast<size_t>(shape.codeQueries) * keys, bounds);
  const Row<const float> maskRow = rowOf(args.mask, static_cast<size_t>(row) * keys, keys, bounds);

  // 0. 2 頭の ρ の列、鍵の塊の在否、V^T（鍵の塊と頭の組を波で分ける）。待ち合わせの後に ρ の列から 2 頭の組の表。
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    stageRho(shared.rho[h], shape, args.table, 2 * pair + h, tid, threads, bounds);
  }
  stageKeyBits(shared.keyBits, maskRow, keys, wave, waves, lane, bounds);
  for (int unit = wave; unit < 2 * blocks; unit += waves) {
    const int kt = unit >> 1;
    const int h = unit & 1;
    const int key = 16 * kt + column;
    const bool present = key < keys && *maskRow.at(static_cast<uint32_t>(min(key, keys - 1)), 1, bounds) != 0.0f;
    u32x4 values = *reinterpret_cast<const u32x4*>(
        keyRow.at(static_cast<uint32_t>(min(key, keys - 1) * 2 * width + width + h * kF + 8 * half), 8, bounds));
    if (!present) {
      values = u32x4{0u, 0u, 0u, 0u};
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      shared.valueT[kF * h + featureRow(8 * half + i)][16 * kt + fragmentColumn(column)] =
          static_cast<uint16_t>(values[i >> 1] >> (16 * (i & 1)));
    }
  }
  barrier();
  for (int e = tid; e < kTables * kTableEntries; e += threads) {
    shared.bias[e] = make_float2(pairEntry(shared.rho[0], e), pairEntry(shared.rho[1], e));
  }
  barrier();

  // 1. 問い合わせの塊 qt = 波。lane は問い合わせ 16qt + column の鍵 16kt + 2v + half（v < 8）を持つ。符号（行 2t + half、鍵 column の
  //    16 lane で 64 バイト続く）と K の行は、次の鍵の塊の分を前もって読む。
  const int qt = wave;
  const int query = 16 * qt + column;
  bf16x16 queryFragment[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t at = static_cast<uint32_t>(min(query, queries - 1) * width + h * kF);
    queryFragment[h] = fragment(queryRow.at(at, kF, bounds));
  }
  auto loadBlock = [&](int kt, uint32_t (&code)[8], bf16x16 (&key)[2]) {
    const int j = min(16 * kt + column, keys - 1);
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      code[t] = *codeRow.at(static_cast<uint32_t>(min(16 * qt + 2 * t + half, queries - 1) * keys + j), 1, bounds);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      key[h] = fragment(keyRow.at(static_cast<uint32_t>(j * 2 * width + h * kF), kF, bounds));
    }
  };
  uint32_t nextCode[8];
  bf16x16 nextKey[2];
  loadBlock(0, nextCode, nextKey);
  float running[2] = {-INFINITY, -INFINITY};
  float total[2] = {0.0f, 0.0f};
  f32x8 out[2] = {f32x8{}, f32x8{}};
  for (int kt = 0; kt < blocks; ++kt) {
    uint32_t staged[8];
    bf16x16 keyFragment[2];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      staged[t] = nextCode[t];
    }
    keyFragment[0] = nextKey[0];
    keyFragment[1] = nextKey[1];
    if (kt + 1 < blocks) {
      loadBlock(kt + 1, nextCode, nextKey);
    }
    const uint32_t keyBits = shared.keyBits[kt];
    if (keyBits == 0) {
      continue;
    }
    // 符号を問い合わせの行に並べ替える（波の中の LDS の写し）。
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      shared.codes[wave][2 * t + half][fragmentColumn(column)] = staged[t];
    }
    waveSync();
    uint32_t code[8];
    unpackEight(*reinterpret_cast<const u32x4*>(&shared.codes[wave][column][8 * half]),
                *reinterpret_cast<const u32x4*>(&shared.codes[wave][column][8 * half + 4]), code);
    if (shape.entries < kAttentionMaxEntries) {
      checkCodes(code, shape.segments, shape.entries);
    }

    f32x8 score[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      score[h] = mma(keyFragment[h], queryFragment[h], f32x8{});
    }
    float logit[2][8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      // 在らない鍵は −∞。表を引く読みを条件の枝に沈めないよう、選ぶのは WMMA の値と定数だけで、−∞ は足して付ける。
      const bool present = ((keyBits >> (2 * v + half)) & 1u) != 0;
      const float floor = present ? 0.0f : -INFINITY;
      const float2 bias = pairBias(code[v], shared.bias);
      logit[0][v] = fmaf(present ? score[0][v] : 0.0f, scale, bias.x) + floor;
      logit[1][v] = fmaf(present ? score[1][v] : 0.0f, scale, bias.y) + floor;
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float block = logit[h][0];
#pragma unroll
      for (int v = 1; v < 8; ++v) {
        block = fmaxf(block, logit[h][v]);
      }
      block = fmaxf(block, swapHalves(block));
      const float next = fmaxf(running[h], block);
      const float alpha = __builtin_amdgcn_exp2f(running[h] - next);
      running[h] = next;
      float e[8];
      float sum = 0.0f;
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        e[v] = __builtin_amdgcn_exp2f(logit[h][v] - next);
        sum += e[v];
      }
      total[h] = fmaf(total[h], alpha, sum);
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        out[h][v] *= alpha;
      }
      bf16x16 terms[3];
      splitFragments(e, terms);
      out[h] = mma3(fragment(&shared.valueT[kF * h + column][16 * kt]), terms, out[h]);
    }
  }

  // 2. 出力: lane は問い合わせ query の次元 8·half〜8·half + 7（BF16 は 16 バイト、FP32 は 32 バイト）。半分どうしの Σ の和
  //    （permlanex16）は波の全 lane が在る分岐の外で取る。
  float sums[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sums[h] = total[h] + swapHalves(total[h]);
  }
  if (query < queries) {
    const size_t first = static_cast<size_t>(row) * queries * width + columns;
    const Row<uint16_t> outRow = rowOf(args.out, first, queries * width - columns, bounds);
    const Row<float> outFloatRow = rowOf(args.outFloat, first, queries * width - columns, bounds);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float sum = sums[h];
      const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
      float o[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        o[v] = out[h][v] * inverse;
      }
      const uint32_t at = static_cast<uint32_t>(query * width + h * kF + 8 * half);
      *reinterpret_cast<u32x4*>(outRow.at(at, 8, bounds)) = pack8(o);
      float* outFloat = outFloatRow.at(at, 8, bounds);
      *reinterpret_cast<f32x4*>(outFloat) = f32x4{o[0], o[1], o[2], o[3]};
      *reinterpret_cast<f32x4*>(outFloat + 4) = f32x4{o[4], o[5], o[6], o[7]};
      if (half == 0) {
        const size_t lse = (static_cast<size_t>(row) * shape.heads + 2 * pair + h) * queries + query;
        args.logSumExp.data[bounds.index(lse, 1, args.logSumExp.limit)] =
            sum > 0.0f ? (running[h] + log2f(sum)) * kLn2 : 0.0f;
      }
    }
  }
  bounds.trap();
}

// dS と符号の LDS の行の刻み（語）: 16·blocks 以上で 64 を法として 36（100 / 164 / 164）。64 bank のとき、段 B の 16 バイトの読み
// （lane が行 column の 8·half から）は 16 行 × 2 つの半分が 2-way（最小）、段 A の書き込み（行 2v + half の 16 列）は 2 つの半分の
// bank が重ならない。行は 16 バイトにそろう。
constexpr int scoreStride(int blocks) {
  return 16 * blocks + ((36 - 16 * blocks) % 64 + 64) % 64;
}
static_assert(scoreStride(kSmallBlocks) == 100 && scoreStride(kMediumBlocks) == 164 && scoreStride(kLargeBlocks) == 164,
              "the score rows are 36 words apart modulo 64 banks");

// LDS。dS と符号は問い合わせの塊の行 [問い合わせ][鍵]（行の刻みは scoreStride）。ρ の列は stageRho。K^T は鍵の塊ごとの
// A の断片（行 r(d)、鍵は自然の順）、Q^T・dO^T は問い合わせの塊の A の断片（行 r(d)、問い合わせ m は列 κ(m)）で、
// 在らない鍵と生きていない問い合わせは 0。lse₂・δ は問い合わせの塊ごとに列 κ(m)（生きていない問い合わせは 0）。queryBits は塊ごとの
// 生きた問い合わせの 16 bit（列 κ(m) の bit が問い合わせ m）。ρ の勾配の部分和は波 w の区画 2w + p の符号 c を lane ごとに
// （区画が segments 以上の欄は書かない）。
template <int kBlocks>
struct alignas(16) BackwardShared {
  float gradScore[16][scoreStride(kBlocks)];
  uint32_t codes[16][scoreStride(kBlocks)];
  uint16_t keyT[kBlocks][kF][16];
  uint16_t queryT[kF][16];
  uint16_t gradOutT[kF][16];
  float logSumExp[kBlocks][16];
  float delta[kBlocks][16];
  float rho[kRhoEntries];
  float bias[kTables * kTableEntries];
  float bins[kScatterWaves][2][kAttentionMaxEntries][kWave];
  uint32_t keyBits[kBlocks];
  uint32_t queryBits[kBlocks];
};

// 問い合わせの塊の生きた bits（queryBits）で、列 κ の問い合わせが生きているか。
__device__ __forceinline__ bool liveQuery(uint32_t bits, int k) {
  return ((bits >> k) & 1u) != 0;
}

// 波の 2 区画の部分和 bins[p][c][lane] に、lane の続いた 2 鍵の dS を足す。区画ごとに順に足し（同じ符号なら 2 つ目は 1 つ目の結果に
// 足す。1 つずつ読み書きするのとビットで同じ）、2 区画の 4 つの読みの待ちを重ねる。shift は区画 2w の符号の bit の位置。
__device__ __forceinline__ void accumulatePair(float (*bins)[kAttentionMaxEntries][kWave], int lane, int shift,
                                               uint32_t first, uint32_t second, float x0, float x1) {
  uint32_t c0[2];
  uint32_t c1[2];
  float v0[2];
  float v1[2];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    c0[p] = (first >> (shift + 4 * p)) & 15u;
    c1[p] = (second >> (shift + 4 * p)) & 15u;
    v0[p] = bins[p][c0[p]][lane];
    v1[p] = bins[p][c1[p]][lane];
  }
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    v0[p] += x0;
    v1[p] = (c1[p] == c0[p] ? v0[p] : v1[p]) + x1;
    bins[p][c0[p]][lane] = v0[p];
    bins[p][c1[p]][lane] = v1[p];
  }
}

// 勾配の 0 の符号を +0 にそろえる: x + (+0) は −0 だけを +0 に写し、ほかの値（非正規数も）は変えない（IEEE の和。nsz の最適化は
// 使っていないので畳まれない。積に付けると fma(x, scale, +0) の 1 命令）。在らない鍵の dk・dv は全ての積が 0 の和で、その 0 の符号は
// 和に入る 0 の符号（WMMA の結果）で決まる。そろえると、ちょうど 0 の和は常に +0 で書かれる。−0 が全く出ないわけではない: 0 でない
// 和の積 x·scale が下に溢れると fma は符号つきの 0 を返し、pack8 は負の小さな値を BF16 の −0 に丸める。その和は塊に依らず同じ値なので、
// 行の独立は保たれる。
__device__ __forceinline__ float positiveZero(float x) {
  return x + 0.0f;
}

// 段 A の 1 つの問い合わせの塊の全体メモリの入力（lane の鍵の 8 つの問い合わせの符号、Q・dO の行の断片）。次の塊の分を前もって読む。
struct QueryBlockInputs {
  uint32_t code[8];
  bf16x16 query;
  bf16x16 gradOut;
};

// 問い合わせの塊の Q^T・dO^T の元（波 1 つ。lane は問い合わせ 16qt + (lane & 15) の次元 8·half〜+7 を q と do で 1 つずつ持つ）。
struct QueryRows {
  u32x4 query;
  u32x4 gradOut;
};

// 問い合わせの塊 qt の q・do の行を読む（全体メモリ。q も do も一様な行の先頭と lane の 32 bit の位置で読む）。問い合わせの外は寄せた
// 最後の行を読み、生きていない問い合わせを 0 にするのは storeQueries なので、読みと LDS に置く間に挟んだ仕事に全体メモリの待ちが重なる。
__device__ __forceinline__ QueryRows loadQueries(Row<const uint16_t> queryRow, Row<const uint16_t> gradOutRow,
                                                 int queries, int width, int qt, int lane, Bounds& bounds) {
  const uint32_t at = static_cast<uint32_t>(min(16 * qt + (lane & 15), queries - 1) * width + 8 * (lane >> 4));
  return QueryRows{*reinterpret_cast<const u32x4*>(queryRow.at(at, 8, bounds)),
                   *reinterpret_cast<const u32x4*>(gradOutRow.at(at, 8, bounds))};
}

// loadQueries の行を LDS の Q^T・dO^T に置く（行 r(d)、問い合わせ m は列 κ(m)。生きていない問い合わせ（外を含む。queryBits の
// bit が 0）は q も do も 0 にする: dk^T += Q^T·dS の K の位置で、dS = +0 の相手の q がどの S でも同じ +0 になる）。
template <int kBlocks>
__device__ __forceinline__ void storeQueries(BackwardShared<kBlocks>& shared, QueryRows rows, int qt, int lane) {
  const int m = lane & 15;
  const int half = lane >> 4;
  if (!liveQuery(shared.queryBits[qt], fragmentColumn(m))) {
    rows.query = rows.gradOut = u32x4{0u, 0u, 0u, 0u};
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int shift = 16 * (i & 1);
    shared.queryT[featureRow(8 * half + i)][fragmentColumn(m)] = static_cast<uint16_t>(rows.query[i >> 1] >> shift);
    shared.gradOutT[featureRow(8 * half + i)][fragmentColumn(m)] = static_cast<uint16_t>(rows.gradOut[i >> 1] >> shift);
  }
}

template <int kBlocks>
__global__ __launch_bounds__(kWave * kBlocks) void backwardKernel(BackwardArgs args) {
  static_assert(kBlocks >= kMinBackwardWaves, "stage B needs the scatter waves and the service wave");
  static_assert(sizeof(BackwardShared<kBlocks>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  __shared__ BackwardShared<kBlocks> shared;

  const AttentionShape shape = args.shape;
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int lane = tid % kWave;
  const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
  const int column = lane & 15;
  const int half = lane >> 4;
  const int width = shape.heads * kF;
  const int keys = shape.keys;
  const int queries = shape.queries;
  const int keyBlocks = (keys + 15) / 16;
  const int queryBlocks = (queries + 15) / 16;
  const float scale = shape.scale * kLog2e;
  Bounds bounds;
  // 行の区画（問い合わせ・鍵と値は頭の列から行の終わりまで）。
  const int columns = head * kF;
  const size_t queryFirst = static_cast<size_t>(row) * queries * width + columns;
  const size_t querySize = static_cast<size_t>(queries) * width - columns;
  const size_t keyFirst = static_cast<size_t>(row) * keys * 2 * width + columns;
  const size_t keySize = static_cast<size_t>(keys) * 2 * width - columns;
  const Row<const uint16_t> queryRow = rowOf(args.query, queryFirst, querySize, bounds);
  const Row<const uint16_t> gradOutRow = rowOf(args.gradOut, queryFirst, querySize, bounds);
  const Row<const float> outFloatRow = rowOf(args.outFloat, queryFirst, querySize, bounds);
  const Row<const float> logSumExpRow =
      rowOf(args.logSumExp, (static_cast<size_t>(row) * shape.heads + head) * queries, queries, bounds);
  const Row<const uint16_t> keyRow = rowOf(args.keyValue, keyFirst, keySize, bounds);
  const Row<const uint32_t> codeRow =
      rowOf(args.codes, static_cast<size_t>(row) * shape.codeQueries * keys,
            static_cast<size_t>(shape.codeQueries) * keys, bounds);
  const Row<const float> maskRow = rowOf(args.mask, static_cast<size_t>(row) * keys, keys, bounds);
  const Row<uint16_t> gradQueryRow = rowOf(args.gradQuery, queryFirst, querySize, bounds);

  // 0. ρ の列、部分和の 0、生きた問い合わせの bits と lse₂・δ（生きていない問い合わせは 0）、鍵の塊の在否、K^T（波 = 鍵の塊）、
  //    最初の q・do の行の読み。待ち合わせの後に最初の Q^T・dO^T（生きた bits を使う）と ρ の列から組の表。
  stageRho(shared.rho, shape, args.table, head, tid, threads, bounds);
  for (int e = tid; e < kScatterWaves * 2 * kAttentionMaxEntries * kWave; e += threads) {
    (&shared.bins[0][0][0][0])[e] = 0.0f;
  }
  // e の lane は e mod 32（threads は 32 の倍数）なので、波の下の 16 lane が塊 e >> 4 の列 κ = 0〜15、上の 16 lane が次の塊を持つ。
  for (int e = tid; e < 16 * queryBlocks; e += threads) {
    const int qt = e >> 4;
    const int k = e & 15;
    const int query = 16 * qt + 2 * (k & 7) + (k >> 3);  // κ(m) = k の問い合わせ m
    float lse = 0.0f;
    float delta = 0.0f;
    uint32_t bits = 0;  // do の BF16 のビットの OR（符号を除いて 0 でなければ生きている。NaN・Inf も生きている）
    if (query < queries) {
      const uint16_t* gradOut = gradOutRow.at(static_cast<uint32_t>(query * width), kF, bounds);
      const float* outFloat = outFloatRow.at(static_cast<uint32_t>(query * width), kF, bounds);
#pragma unroll
      for (int d = 0; d < kF; ++d) {
        bits |= gradOut[d];
        delta = fmaf(bf16Float(gradOut[d]), outFloat[d], delta);
      }
      lse = *logSumExpRow.at(static_cast<uint32_t>(query), 1, bounds) * kLog2e;
    }
    const bool live = (bits & 0x7fffu) != 0;
    shared.logSumExp[qt][k] = live ? lse : 0.0f;
    shared.delta[qt][k] = live ? delta : 0.0f;
    const uint32_t ballot = __builtin_amdgcn_ballot_w32(live);
    if (k == 0) {
      shared.queryBits[qt] = (ballot >> (lane & 16)) & 0xffffu;
    }
  }
  stageKeyBits(shared.keyBits, maskRow, keys, wave, threads / kWave, lane, bounds);
  // 段 A の波（kt = 波）の鍵の行: S・dP の B の断片（レジスタに残す）と K^T。
  const int kt = wave;
  const int key = 16 * kt + column;
  const uint32_t keyAt = static_cast<uint32_t>(min(key, keys - 1) * 2 * width);
  bool keyPresent = false;
  bf16x16 keyFragment = {};
  bf16x16 valueFragment = {};
  if (kt < keyBlocks) {
    keyPresent = key < keys && *maskRow.at(static_cast<uint32_t>(min(key, keys - 1)), 1, bounds) != 0.0f;
    keyFragment = fragment(keyRow.at(keyAt, kF, bounds));
    valueFragment = fragment(keyRow.at(keyAt + width, kF, bounds));
    u32x4 values = *reinterpret_cast<const u32x4*>(keyRow.at(keyAt + 8 * half, 8, bounds));
    if (!keyPresent) {
      values = u32x4{0u, 0u, 0u, 0u};
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      shared.keyT[kt][featureRow(8 * half + i)][column] = static_cast<uint16_t>(values[i >> 1] >> (16 * (i & 1)));
    }
  }
  QueryRows firstRows = {};
  if (wave == kServiceWave) {
    firstRows = loadQueries(queryRow, gradOutRow, queries, width, 0, lane, bounds);
  }
  auto loadInputs = [&](int qt, QueryBlockInputs& in) {
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      in.code[v] = *codeRow.at(
          static_cast<uint32_t>(min(16 * qt + 2 * v + half, queries - 1) * keys + min(key, keys - 1)), 1, bounds);
    }
    const uint32_t at = static_cast<uint32_t>(min(16 * qt + column, queries - 1) * width);
    in.query = fragment(queryRow.at(at, kF, bounds));
    in.gradOut = fragment(gradOutRow.at(at, kF, bounds));
  };
  QueryBlockInputs next;
  if (kt < keyBlocks) {
    loadInputs(0, next);
  }
  barrier();
  if (wave == kServiceWave) {
    storeQueries(shared, firstRows, 0, lane);
  }
  for (int e = tid; e < kTables * kTableEntries; e += threads) {
    shared.bias[e] = pairEntry(shared.rho, e);
  }
  // 回す問い合わせの塊は生きた問い合わせを含む最後の塊まで（一様。後ろの塊を落とすだけなので、段 A の次の塊の前もった読みと段 B の
  // 次の Q^T・dO^T の流れはそのまま）。
  int liveBlocks = 0;
  for (int qt = 0; qt < queryBlocks; ++qt) {
    liveBlocks = shared.queryBits[qt] != 0 ? qt + 1 : liveBlocks;
  }
  liveBlocks = __builtin_amdgcn_readfirstlane(liveBlocks);
  barrier();
  const bool ownsKeys = kt < keyBlocks && shared.keyBits[kt] != 0;

  f32x8 gradKeyT = {};
  f32x8 gradValueT = {};
  for (int qt = 0; qt < liveBlocks; ++qt) {
    const uint32_t liveBits = shared.queryBits[qt];
    // 段 A: lane は鍵 key の問い合わせ 16qt + 2v + half（v < 8。列 κ = v + 8·half）を持つ。
    if (ownsKeys) {
      const QueryBlockInputs in = next;
      if (qt + 1 < liveBlocks) {
        loadInputs(qt + 1, next);
      }
      if (shape.entries < kAttentionMaxEntries) {
        checkCodes(in.code, shape.segments, shape.entries);
      }
      const bf16x16 queryT = fragment(&shared.queryT[column][0]);
      const bf16x16 gradOutT = fragment(&shared.gradOutT[column][0]);
      float lse[8];
      float delta[8];
      unpackEight(*reinterpret_cast<const f32x4*>(&shared.logSumExp[qt][8 * half]),
                  *reinterpret_cast<const f32x4*>(&shared.logSumExp[qt][8 * half + 4]), lse);
      unpackEight(*reinterpret_cast<const f32x4*>(&shared.delta[qt][8 * half]),
                  *reinterpret_cast<const f32x4*>(&shared.delta[qt][8 * half + 4]), delta);

      const f32x8 score = mma(in.query, keyFragment, f32x8{});
      const f32x8 gradProbability = mma(in.gradOut, valueFragment, f32x8{});
      float probability[8];
      float gradScore[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        // 在らない鍵と生きていない問い合わせは P = +0、dS = +0·(0 − 0) = +0（順伝播と同じく −∞ を足して付け、表を引く読みを枝に
        // 沈めない）。
        const bool valid = keyPresent && liveQuery(liveBits, v + 8 * half);
        const float floor = valid ? 0.0f : -INFINITY;
        const float logit = fmaf(valid ? score[v] : 0.0f, scale, pairBias(in.code[v], shared.bias));
        probability[v] = __builtin_amdgcn_exp2f((logit - lse[v]) + floor);
        gradScore[v] = probability[v] * ((valid ? gradProbability[v] : 0.0f) - delta[v]);
      }
      bf16x16 terms[3];
      splitFragments(probability, terms);
      gradValueT = mma3(gradOutT, terms, gradValueT);
      splitFragments(gradScore, terms);
      gradKeyT = mma3(queryT, terms, gradKeyT);
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        shared.gradScore[2 * v + half][key] = gradScore[v];
        shared.codes[2 * v + half][key] = in.code[v];
      }
    }
    barrier();

    // 段 B。
    if (wave < kScatterWaves) {
      // lane は問い合わせ 16qt + column の鍵 16k + 8·half〜+7 を、区画 2·波 と 2·波 + 1 の部分和（lane ごと）に足す（生きた
      // 問い合わせだけ）。区画が segments 以上の部分和も足すが書かない（本体を分岐なしに保つ）。
      if (liveQuery(liveBits, fragmentColumn(column))) {
        for (int k = 0; k < keyBlocks; ++k) {
          if (shared.keyBits[k] == 0) {
            continue;
          }
          const int first = 16 * k + 8 * half;
          float x[8];
          uint32_t c[8];
          unpackEight(*reinterpret_cast<const f32x4*>(&shared.gradScore[column][first]),
                      *reinterpret_cast<const f32x4*>(&shared.gradScore[column][first + 4]), x);
          unpackEight(*reinterpret_cast<const u32x4*>(&shared.codes[column][first]),
                      *reinterpret_cast<const u32x4*>(&shared.codes[column][first + 4]), c);
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            accumulatePair(shared.bins[wave], lane, 8 * wave, c[e], c[e + 1], x[e], x[e + 1]);
          }
        }
      }
    } else if (wave == kServiceWave) {
      // 次の問い合わせの塊の q・do の行は dQ の前に読み（全体メモリの待ちを dQ に重ねる）、dQ の後で LDS に置く。
      const bool stagesNext = qt + 1 < liveBlocks;
      QueryRows nextRows = {};
      if (stagesNext) {
        nextRows = loadQueries(queryRow, gradOutRow, queries, width, qt + 1, lane, bounds);
      }
      // dQ^T = K^T·dS^T（鍵の塊の順）: lane は問い合わせ 16qt + column の次元 8·half〜+7。
      f32x8 gradQueryT = {};
      for (int k = 0; k < keyBlocks; ++k) {
        if (shared.keyBits[k] == 0) {
          continue;
        }
        float own[8];
        unpackEight(*reinterpret_cast<const f32x4*>(&shared.gradScore[column][16 * k + 8 * half]),
                    *reinterpret_cast<const f32x4*>(&shared.gradScore[column][16 * k + 8 * half + 4]), own);
        bf16x16 terms[3];
        splitFragments(own, terms);
        gradQueryT = mma3(fragment(&shared.keyT[k][column][0]), terms, gradQueryT);
      }
      // 死んだ問い合わせの dq はちょうど 0（dS = 0）なので +0 を書く。
      const int query = 16 * qt + column;
      if (query < queries) {
        const bool live = liveQuery(liveBits, fragmentColumn(column));
        float values[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          values[v] = live ? positiveZero(gradQueryT[v] * shape.scale) : 0.0f;
        }
        *reinterpret_cast<u32x4*>(gradQueryRow.at(static_cast<uint32_t>(query * width + 8 * half), 8, bounds)) =
            pack8(values);
      }
      if (stagesNext) {
        storeQueries(shared, nextRows, qt + 1, lane);
      }
    }
    barrier();
  }
  // 回さなかった後ろの塊の問い合わせ（全て死んでいる）の dq は +0（1 つの問い合わせは 16 バイトの 2 つ）。
  for (int e = tid; e < 2 * (queries - 16 * liveBlocks); e += threads) {
    const int query = 16 * liveBlocks + (e >> 1);
    *reinterpret_cast<u32x4*>(gradQueryRow.at(static_cast<uint32_t>(query * width + 8 * (e & 1)), 8, bounds)) =
        u32x4{0u, 0u, 0u, 0u};
  }

  // 3. dk・dv（波 = 鍵の塊。lane は鍵 key の次元 8·half〜+7。在らない鍵の塊は 0 を書く）と ρ の勾配の部分和（区画 2·波 + half の
  //    符号 column を 32 lane の順に足す）。
  if (kt < keyBlocks && key < keys) {
    float gradKey[8];
    float gradValue[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      gradKey[v] = positiveZero(gradKeyT[v] * shape.scale);
      gradValue[v] = positiveZero(gradValueT[v]);
    }
    const Row<uint16_t> gradKeyRow = rowOf(args.gradKeyValue, keyFirst, keySize, bounds);
    const uint32_t at = static_cast<uint32_t>(key * 2 * width + 8 * half);
    *reinterpret_cast<u32x4*>(gradKeyRow.at(at, 8, bounds)) = pack8(gradKey);
    *reinterpret_cast<u32x4*>(gradKeyRow.at(at + width, 8, bounds)) = pack8(gradValue);
  }
  if (wave < kScatterWaves) {
    const int s = 2 * wave + half;
    if (s < shape.segments && column < shape.entries) {
      float sum = 0.0f;
#pragma unroll
      for (int l = 0; l < kWave; l += 4) {
        const f32x4 part = *reinterpret_cast<const f32x4*>(&shared.bins[wave][half][column][l]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sum += part[i];
        }
      }
      const int tableRows = shape.segments * shape.entries;
      const size_t at = (static_cast<size_t>(row) * shape.heads + head) * tableRows + s * shape.entries + column;
      args.gradTable.data[bounds.index(at, 1, args.gradTable.limit)] = sum;
    }
  }
  bounds.trap();
}

int blockBucket(int blocks) {
  return blocks <= kSmallBlocks ? kSmallBlocks : blocks <= kMediumBlocks ? kMediumBlocks : kLargeBlocks;
}

const uint32_t* codeWords(const int16_t* codes) {
  if (reinterpret_cast<uintptr_t>(codes) % alignof(uint32_t) != 0) {
    throw std::invalid_argument("relation codes must be aligned to 4 bytes");
  }
  return reinterpret_cast<const uint32_t*>(codes);
}

}  // namespace

void checkAttentionShape(const AttentionShape& shape) {
  auto require = [](bool condition, const std::string& message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  require(shape.rows > 0, "attention needs rows, got " + std::to_string(shape.rows));
  require(shape.queries > 0 && shape.queries <= kAttentionMaxTokens,
          "attention queries must be in 1.." + std::to_string(kAttentionMaxTokens) + ", got " +
              std::to_string(shape.queries));
  require(shape.keys > 0 && shape.keys <= kAttentionMaxTokens,
          "attention keys must be in 1.." + std::to_string(kAttentionMaxTokens) + ", got " +
              std::to_string(shape.keys));
  require(shape.codeQueries >= shape.queries, "relation codes must cover the queries");
  require(shape.heads > 0 && shape.heads % 2 == 0,
          "attention needs an even number of heads, got " + std::to_string(shape.heads));
  require(shape.words == 2,
          "relation codes must be one 32-bit word per pair, got " + std::to_string(shape.words) + " words");
  require(shape.segments > 0 && shape.segments <= kAttentionMaxSegments,
          "relation segments must be in 1.." + std::to_string(kAttentionMaxSegments));
  require(shape.entries > 0 && shape.entries <= kAttentionMaxEntries,
          "relation entries must be in 1.." + std::to_string(kAttentionMaxEntries));
  require(shape.scale > 0.0f, "attention scale must be positive");
}

void attentionForward(const AttentionShape& shape, const AttentionForward& arrays, hipStream_t stream) {
  checkAttentionShape(shape);
  const size_t rows = static_cast<size_t>(shape.rows);
  const size_t width = static_cast<size_t>(shape.heads) * kF;
  ForwardArgs args{};
  args.shape = shape;
  args.query = {arrays.query, rows * shape.queries * width};
  args.keyValue = {arrays.keyValue, rows * shape.keys * 2 * width};
  args.mask = {arrays.mask, rows * shape.keys};
  args.codes = {codeWords(arrays.codes), rows * shape.codeQueries * shape.keys};
  args.table = {arrays.table, static_cast<size_t>(shape.segments) * shape.entries * shape.heads};
  args.out = {arrays.out, rows * shape.queries * width};
  args.outFloat = {arrays.outFloat, rows * shape.queries * width};
  args.logSumExp = {arrays.logSumExp, rows * shape.heads * shape.queries};
  const int queryBlocks = (shape.queries + 15) / 16;
  const int keyBlocks = (shape.keys + 15) / 16;
  // 同じ行の頭の対が隣どうしで走る（行の符号と k・v を L2 で使い回す）。波は問い合わせの塊の数。
  const dim3 grid(shape.heads / 2, shape.rows);
  const dim3 block(kWave * queryBlocks);
  switch (blockBucket(std::max(queryBlocks, keyBlocks))) {
    case kSmallBlocks:
      hipLaunchKernelGGL(forwardKernel<kSmallBlocks>, grid, block, 0, stream, args);
      break;
    case kMediumBlocks:
      hipLaunchKernelGGL(forwardKernel<kMediumBlocks>, grid, block, 0, stream, args);
      break;
    default:
      hipLaunchKernelGGL(forwardKernel<kLargeBlocks>, grid, block, 0, stream, args);
      break;
  }
  PRISM_HIP(hipGetLastError());
}

void attentionBackward(const AttentionShape& shape, const AttentionBackward& arrays, hipStream_t stream) {
  checkAttentionShape(shape);
  const size_t rows = static_cast<size_t>(shape.rows);
  const size_t width = static_cast<size_t>(shape.heads) * kF;
  BackwardArgs args{};
  args.shape = shape;
  args.query = {arrays.query, rows * shape.queries * width};
  args.keyValue = {arrays.keyValue, rows * shape.keys * 2 * width};
  args.mask = {arrays.mask, rows * shape.keys};
  args.codes = {codeWords(arrays.codes), rows * shape.codeQueries * shape.keys};
  args.table = {arrays.table, static_cast<size_t>(shape.segments) * shape.entries * shape.heads};
  args.outFloat = {arrays.outFloat, rows * shape.queries * width};
  args.logSumExp = {arrays.logSumExp, rows * shape.heads * shape.queries};
  args.gradOut = {arrays.gradOut, rows * shape.queries * width};
  args.gradQuery = {arrays.gradQuery, rows * shape.queries * width};
  args.gradKeyValue = {arrays.gradKeyValue, rows * shape.keys * 2 * width};
  args.gradTable = {arrays.gradTable, rows * shape.heads * shape.segments * shape.entries};
  const int queryBlocks = (shape.queries + 15) / 16;
  const int keyBlocks = (shape.keys + 15) / 16;
  // 同じ行の頭が隣どうしで走る。波は鍵の塊の数（段 B の 5 波より少なくしない）。
  const dim3 grid(shape.heads, shape.rows);
  const dim3 block(kWave * std::max(keyBlocks, kMinBackwardWaves));
  switch (blockBucket(std::max(queryBlocks, keyBlocks))) {
    case kSmallBlocks:
      hipLaunchKernelGGL(backwardKernel<kSmallBlocks>, grid, block, 0, stream, args);
      break;
    case kMediumBlocks:
      hipLaunchKernelGGL(backwardKernel<kMediumBlocks>, grid, block, 0, stream, args);
      break;
    default:
      hipLaunchKernelGGL(backwardKernel<kLargeBlocks>, grid, block, 0, stream, args);
      break;
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
