// K2（attn0）と K4（attn1）: 作業群は 1 頭の対 × 1 行で、波の数 W は行の組の W（prism_hip_kernels.h の attentionWaves。行の 16
// トークンの塊の数が W 以下）。計画は組ごとに起動し、作業群は組の順の行の番号（rowOrderKernel）から自分の行を引く。eager の
// EncoderLayer の注意の半分（O 射影は K3・K5）。
//
//   0. 2 頭の偏りの表（×log2e）を LDS へ写す（重みの公開のときに relationBiasKernel が層ごとに作ったもの。prism_hip_relation.h）
//   1. 頭の対の q・k・v = ⌊n·W_qkvᵀ + b⌉（WMMA。K4 は q を局・席・牌の 39 個だけ）。n = ⌊(x·r)·g⌉ は前の段（K1・K3）が
//      NORMALIZED に書いたもので、断片としてそのまま読む。W_qkv は重みの公開のときに qkvFragmentKernel が断片の順に並べ替えた写し
//      （下）を読む。波 m が 16 トークンの塊 m を作る。k・v は LDS へ。q は転置の qᵀ = W_q·nᵀ で作り、問い合わせの B の断片のまま
//      レジスタに残す（波 m が問い合わせの塊 m を受け持つ）。ただし K4 の W ≥ 6 の組は 1 波が 2 塊の 1 頭を作り（重みの断片を 2 塊で
//      共有して読みを減らす）、q を LDS へ置く
//   2. 問い合わせの塊（16 個）ごとに 1 波が 2 頭を回す。K4 の W ≥ 6 の組は 1 波が 1 頭を回す（K4 の問い合わせの塊は 3 つ以下なので
//      6 波以下）。鍵の塊（16 個）ごとに
//        S^T = K·Qᵀ（WMMA、BF16 の積を FP32 で累積）、ℓ = S·scale + Σ_s ρ[16s + c_s(i, j)]（c は K0 の符号。ρ は区画 2 つずつ前もって
//        足した表を引く）、オンラインの softmax（exp2）、O^T += V^T·P^T（WMMA。P は FP32 の e を hi・lo の 2 つの BF16 に分けて 2 回掛け、
//        FP32 に近い精度を保つ）
//      最後に O / Σe を BF16 に 1 回丸める。
//
// 段 1 と段 2 の間の待ち合わせは LDS の順序だけ（phaseBarrier）。q・k・v・o の丸めは活性の丸め（bf16RoundActivation・
// pack8Activation。NaN の選択なし）。
//
// eager の packedRelationScaledDotProductAttention と式は同じで、和の順と exp だけが違う（相対 RMS ≤ 2e-2 の約束の内）。縮約は全て行の
// 中で順は固定、原子的な加算もない。WMMA の要素は互いに独立なので、同じ行の o は塊の組み方に依らずビットで同じ。W は塊と頭を受け
// 持つ波を決めるだけで、塊ごと・頭ごとの演算（被演算子・K の順・丸め・置く位置）はどの組の核でも同じなので、o は組にも依らない。
//
// 射影の重みの写し（qkvFragmentKernel。重みの公開のときに層ごとに 1 回）: 頭の対ごとに [歩 k][頭 h][j = q・k・v][半分][行 r][8]
// = W_qkv[unit(j, h, r)][16k + 8·半分 .. +8]。unit(q, h, r) = 32·対 + 16h + π(r)（π(r) = (r >> 1) + 8(r & 1)）、
// unit(k, h, r) = A + 32·対 + 16h + r、unit(v, h, r) = 2A + 32·対 + 16h + r。断片 1 つは 512 バイト、歩 1 つは 3 KB、頭の対は 48 KB。
// lane は行 column の 16 バイトを lane の位置 column·16 と波で同じ soffset k·3072 + (3h + j)·512（上の 8 要素は + 256）で読むので、
// b128 の 1 回は続いた 256 バイト（2 本の cache line。W_qkv をそのまま読むと 512 バイト離れた 16 行）。値と WMMA の断片は同じ。
#include <utility>

#include "prism_hip_relation.h"

namespace prism {
namespace {

constexpr int kHeadWidth = 16;
constexpr int kPairWidth = 2 * kHeadWidth;                 // 頭の対の q・k の列
constexpr int kFixedBlocks = (kMaxFixedTokens + 15) / 16;  // K4 の問い合わせの塊
constexpr int kBiasChunks = kBiasEntries / 2;              // 偏りの表の 16 バイトの塊
static_assert(kBiasEntries % 2 == 0, "the bias table is copied 16 bytes at a time");
static_assert(sizeof(float2) == kBiasEntryBytes, "a code field holds the byte offset of a two-head bias entry");
constexpr int kQkvThreads = 256;     // qkvFragmentKernel の作業群
constexpr int kOrderThreads = 1024;  // rowOrderKernel の作業群（1 つ）
constexpr int kOrderWaves = kOrderThreads / kWave;

// 射影の重みの写しの置き方（上）。
constexpr int kFragmentBytes = 512;                 // 断片 1 つ（K の下の 8 要素の 16 行、上の 8 要素の 16 行）
constexpr int kStepBytes = 2 * 3 * kFragmentBytes;  // 歩 1 つ（2 頭 × q・k・v）

// 問い合わせの q の LDS（k と同じ入れ替えで置く）。1 波が 1 頭を回す K4 の組だけが、問い合わせ（局・席・牌の 48 以下）の q を全て
// 置く。ほかの組の q はレジスタに残す。
template <bool kHeadWaves>
struct FixedQueries {};

template <>
struct FixedQueries<true> {
  uint16_t fixedQuery[kMaxFixedTokens][kPairWidth];
};

// LDS（kTokens = 16W トークン分）。k は [トークン][2 頭 × 16] の行の 16 バイトの塊を (トークン >> 1) & 3 で入れ替えて置く（8 lane の
// b128 の読みで bank が衝突しない）。V^T は [16h + r(d)][κ] で、r(d) = 2(d & 7) + (d >> 3)、鍵 16m + 2v + half は列 16m + v + 8·half。
// この並びで O^T の断片の lane が続いた 8 次元を持ち、P^T の B の断片は lane と相方の 8 値を続けるだけになる。V^T の行の刻み
// kTokens + 8 要素（8W + 4 語）は 8 語の倍数から 4 語ずれるので、16 lane が 16 バイトずつ読むとき bank が衝突しない。
template <int kTokens, bool kHeadWaves>
struct alignas(16) AttentionShared : FixedQueries<kHeadWaves> {
  static constexpr int kValueStride = kTokens + 8;
  uint16_t k[kTokens][kPairWidth];
  uint16_t valueT[kPairWidth][kValueStride];
  float2 bias[kBiasEntries];  // 2 頭の偏りの表（×log2e）
};

// q・k の行 token の列 column（< 32）の LDS の位置。
__device__ __forceinline__ int swizzled(int token, int column) {
  return 8 * ((column >> 3) ^ ((token >> 1) & 3)) + (column & 7);
}

// q・k の行 token の頭 head の 16 次元（WMMA の断片）。
__device__ __forceinline__ bf16x16 headFragment(const uint16_t (*m)[kPairWidth], int token, int head) {
  const int s = (token >> 1) & 3;
  const u32x4 lo = *reinterpret_cast<const u32x4*>(&m[token][8 * ((2 * head) ^ s)]);
  const u32x4 hi = *reinterpret_cast<const u32x4*>(&m[token][8 * ((2 * head + 1) ^ s)]);
  return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

// V^T の頭 head の行 column（= r(d)）の鍵の塊 block の 16 鍵（WMMA の A の断片の行）。
template <int kValueStride>
__device__ __forceinline__ const uint16_t* valueRow(const uint16_t (*valueT)[kValueStride], int head, int block,
                                                    int column) {
  return &valueT[kHeadWidth * head + column][16 * block];
}

__device__ __forceinline__ float fastExp2(float value) {
#if __has_builtin(__builtin_amdgcn_exp2f)
  return __builtin_amdgcn_exp2f(value);
#else
  return exp2f(value);
#endif
}

// 2 つの FP32 の上位 16 bit を 1 語に詰める（a が下、b が上。v_perm 1 つ）。
__device__ __forceinline__ uint32_t upperHalves(uint32_t a, uint32_t b) {
  return __builtin_amdgcn_perm(b, a, 0x07060302u);
}

// B の断片を lane の 8 値（16 バイト）から作る: 下の 16 lane が K の 0〜7、上の 16 lane が 8〜15 を持つとき、どの lane も
// [lane & 15 の 8 値, lane | 16 の 8 値] を得る。ds_swizzle の bit の型（and 0x0F、or 0x10）は LDS の crossbar を通るだけで bank に
// 触れない。
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

// P^T の B の断片（hi・lo の 2 つの BF16）。lane は鍵 2v + half（v < 8）の e を持ち、断片の K の順は V^T の列の並び κ = v + 8·half
// なので、下の 16 lane の 8 値が K の 0〜7、上の 16 lane の 8 値が 8〜15。hi は e の上位 16 bit、lo は e − hi の上位 16 bit（どちらも
// 切り捨て。hi + lo は e の仮数の 16 bit 以上を保つ）。
__device__ __forceinline__ void probabilityFragments(const float (&e)[8], bf16x16& high, bf16x16& low) {
  u32x4 ownHigh;
  u32x4 ownLow;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t a = __builtin_bit_cast(uint32_t, e[2 * w]);
    const uint32_t b = __builtin_bit_cast(uint32_t, e[2 * w + 1]);
    ownHigh[w] = upperHalves(a, b);
    const uint32_t ra = __builtin_bit_cast(uint32_t, e[2 * w] - __builtin_bit_cast(float, a & 0xffff0000u));
    const uint32_t rb = __builtin_bit_cast(uint32_t, e[2 * w + 1] - __builtin_bit_cast(float, b & 0xffff0000u));
    ownLow[w] = upperHalves(ra, rb);
  }
  high = joinHalves(ownHigh);
  low = joinHalves(ownLow);
}

// 生の buffer の 32 バイト（WMMA の断片 1 つ）を 16 バイトずつ読む。offset は lane のバイトの位置（VGPR）、low・high は下と上の
// 16 バイトの波で同じ位置（soffset。上の半分の差も soffset に入れて、輪の中に VGPR の足し算を作らない）。
__device__ __forceinline__ bf16x16 bufferFragment(__amdgpu_buffer_rsrc_t resource, uint32_t offset, int low, int high) {
  const u32x4 lo = __builtin_amdgcn_raw_buffer_load_b128(resource, offset, low, 0);
  const u32x4 hi = __builtin_amdgcn_raw_buffer_load_b128(resource, offset, high, 0);
  return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

// 射影の K の 1 歩の断片: 頭 h の q・k・v（j = 0・1・2）の重みの断片（写しの soffset k·3072 + (3h + j)·512）と、塊 b < kBlocks の
// n の行の 32k バイト。重みは塊で共有する。q を作るのは先頭の kQueries 個の塊だけで（K4 の局・席・牌より後の塊は作らない）、
// kQueries = 0 なら q の重みも読まない。
template <int kBlocks, int kQueries>
struct ProjectionStep {
  static_assert(0 <= kQueries && kQueries <= kBlocks, "the blocks with a query come first");
  bf16x16 weight[3];
  bf16x16 input[kBlocks];

  __device__ __forceinline__ void load(__amdgpu_buffer_rsrc_t weights, uint32_t fragmentRow, int h,
                                       __amdgpu_buffer_rsrc_t inputs, const uint32_t (&inputRow)[kBlocks], int k) {
#pragma unroll
    for (int j = kQueries > 0 ? 0 : 1; j < 3; ++j) {
      const int at = k * kStepBytes + (3 * h + j) * kFragmentBytes;
      weight[j] = bufferFragment(weights, fragmentRow, at, at + kFragmentBytes / 2);
    }
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
      input[b] = bufferFragment(inputs, inputRow[b], 32 * k, 32 * k + 16);
    }
  }

  // 塊 b の acc[b][0] += W_q·nᵀ（転置の q）、acc[b][1]・acc[b][2] += n·W_k,vᵀ。
  __device__ __forceinline__ void multiply(f32x8 (&acc)[kBlocks][3]) const {
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
      if (b < kQueries) {
        acc[b][0] = mma(weight[0], input[b], acc[b][0]);
      }
#pragma unroll
      for (int j = 1; j < 3; ++j) {
        acc[b][j] = mma(input[b], weight[j], acc[b][j]);
      }
    }
  }
};

// 頭 h の塊 kBlocks 個の射影 acc（ProjectionStep::multiply の式）を K の 16 歩で作る。断片の組 kSets 個を回して kSets − 1 歩先まで
// 読みながら掛ける。読みは全て無条件なので輪は 1 つの基本塊のまま（組を写す mov がない）、sched_barrier が次の読みを掛け算の後ろに
// 留める。
//   組 2 つ（K4 の q を作る塊 2 つ）: 展開しない 2 歩の輪と最後の 2 歩（読み直しはない）
//   組 3 つ（K2）: 展開しない 3 歩の輪と最後の 1 歩。最後の周の 1 つは歩 15 を読み直す（使わない）
//   組 4 つ（K4 の q を作らない塊 2 つと塊 1 つ）: 3 歩を先に読み、展開しない 4 歩の輪と最後の 4 歩（読み直しはない）
// 組はカーネルが 8 波/SIMD のレジスタ（192）に収まる最多にする。塊 1 つの組 2 つでは 1 歩の掛け算 3 つの間しか読みを隠せず、
// 展開すると断片が重なってレジスタが溢れる。K2 は組 4 つで占有が下がるので組 3 つ。K4 の q を作る塊 2 つは acc 6 つと組 1 つの断片
// 5 つが要るので組 2 つまでで、1 歩の掛け算 5〜6 つの間に読みを隠す。どれも各 acc の K の順は 0〜15 で同じ。
template <int H, int kSets, int kBlocks, int kQueries>
__device__ __forceinline__ void projectHead(f32x8 (&acc)[kBlocks][3], __amdgpu_buffer_rsrc_t weights,
                                            uint32_t fragmentRow, int h, __amdgpu_buffer_rsrc_t inputs,
                                            const uint32_t (&inputRow)[kBlocks]) {
  constexpr int kSteps = H / 16;
#pragma unroll
  for (int b = 0; b < kBlocks; ++b) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      acc[b][j] = f32x8{};
    }
  }
  ProjectionStep<kBlocks, kQueries> a;
  ProjectionStep<kBlocks, kQueries> b;
  a.load(weights, fragmentRow, h, inputs, inputRow, 0);
  if constexpr (kSets == 3) {
    static_assert(kSteps % 3 == 1, "the rolled loop covers the steps in threes plus the last one");
    ProjectionStep<kBlocks, kQueries> c;
    b.load(weights, fragmentRow, h, inputs, inputRow, 1);
#pragma unroll 1
    for (int k = 0; k + 1 < kSteps; k += 3) {
      c.load(weights, fragmentRow, h, inputs, inputRow, k + 2);
      a.multiply(acc);
      __builtin_amdgcn_sched_barrier(0);
      a.load(weights, fragmentRow, h, inputs, inputRow, k + 3);
      b.multiply(acc);
      __builtin_amdgcn_sched_barrier(0);
      b.load(weights, fragmentRow, h, inputs, inputRow, min(k + 4, kSteps - 1));
      c.multiply(acc);
      __builtin_amdgcn_sched_barrier(0);
    }
    a.multiply(acc);
  } else if constexpr (kSets == 2) {
    static_assert(kSteps % 2 == 0, "the rolled loop covers the steps in twos");
#pragma unroll 1
    for (int k = 0; k + 2 < kSteps; k += 2) {
      b.load(weights, fragmentRow, h, inputs, inputRow, k + 1);
      a.multiply(acc);
      __builtin_amdgcn_sched_barrier(0);
      a.load(weights, fragmentRow, h, inputs, inputRow, k + 2);
      b.multiply(acc);
      __builtin_amdgcn_sched_barrier(0);
    }
    b.load(weights, fragmentRow, h, inputs, inputRow, kSteps - 1);
    a.multiply(acc);
    __builtin_amdgcn_sched_barrier(0);
    b.multiply(acc);
  } else {
    static_assert(kSets == 4 && kSteps % 4 == 0, "the rolled loop covers the steps in fours");
    ProjectionStep<kBlocks, kQueries> c;
    ProjectionStep<kBlocks, kQueries> d;
    b.load(weights, fragmentRow, h, inputs, inputRow, 1);
    c.load(weights, fragmentRow, h, inputs, inputRow, 2);
#pragma unroll 1
    for (int k = 0; k + 4 < kSteps; k += 4) {
      d.load(weights, fragmentRow, h, inputs, inputRow, k + 3);
      a.multiply(acc);
      __builtin_amdgcn_sched_barrier(0);
      a.load(weights, fragmentRow, h, inputs, inputRow, k + 4);
      b.multiply(acc);
      __builtin_amdgcn_sched_barrier(0);
      b.load(weights, fragmentRow, h, inputs, inputRow, k + 5);
      c.multiply(acc);
      __builtin_amdgcn_sched_barrier(0);
      c.load(weights, fragmentRow, h, inputs, inputRow, k + 6);
      d.multiply(acc);
      __builtin_amdgcn_sched_barrier(0);
    }
    d.load(weights, fragmentRow, h, inputs, inputRow, kSteps - 1);
    a.multiply(acc);
    __builtin_amdgcn_sched_barrier(0);
    b.multiply(acc);
    __builtin_amdgcn_sched_barrier(0);
    c.multiply(acc);
    __builtin_amdgcn_sched_barrier(0);
    d.multiply(acc);
  }
}

// 頭の対 pair の偏りの表の 16 バイトの塊 e（項 2e・2e + 1）。
__device__ __forceinline__ f32x4 biasChunk(const AttentionArgs& args, Bounds& bounds, int pair, int e) {
  const size_t at = bounds.index((static_cast<size_t>(pair) * kBiasEntries + 2 * e) * 2, 4, args.bias.limit);
  return *reinterpret_cast<const f32x4*>(args.bias.data + at);
}

// 段 1 と段 2 の間の作業群の待ち合わせ（LDS の順序だけ。tile::barrier と同じ形）。__syncthreads は s_barrier の後に buffer_gl0_inv
// （WGP の型の取得で、同じ WGP の他の作業群が射影に使っている全体メモリの L0 まで捨てる）を出すが、K2・K4 はカーネルの中で書いた
// 全体メモリを読まない（o は書くだけ、n・重みの写し・符号・偏りの表は読むだけ）ので要らない。
__device__ __forceinline__ void phaseBarrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// 符号 code（推論の語）の 4 つの表の項の和 Σ_s ρ（×log2e）。表の項は (頭 2p, 頭 2p + 1) の値の対で、2 頭は対を 1 回で引く。語の欄は
// 項のバイトの位置なので、番地の式は欄の取り出しだけになる（項の番号に直す ÷8 と配列の ×8 は打ち消し合う）。
__device__ __forceinline__ float2 pairBias(const float2 (&bias)[kBiasEntries], uint32_t code) {
  const float2 b01 = bias[pairTableOffset(code, 0) / kBiasEntryBytes];
  const float2 b23 = bias[kPairEntries + pairTableOffset(code, 1) / kBiasEntryBytes];
  const float2 b45 = bias[2 * kPairEntries + pairTableOffset(code, 2) / kBiasEntryBytes];
  const float2 b6 = bias[3 * kPairEntries + lastTableOffset(code) / kBiasEntryBytes];
  return make_float2(((b01.x + b23.x) + b45.x) + b6.x, ((b01.y + b23.y) + b45.y) + b6.y);
}

// 同じ和の頭 head（0・1）の分だけ（対の片方を引く。足す順は pairBias と同じ）。
__device__ __forceinline__ float headBias(const float2 (&bias)[kBiasEntries], uint32_t code, int head) {
  const float* table = reinterpret_cast<const float*>(bias) + head;
  const float b01 = table[2 * (pairTableOffset(code, 0) / kBiasEntryBytes)];
  const float b23 = table[2 * (kPairEntries + pairTableOffset(code, 1) / kBiasEntryBytes)];
  const float b45 = table[2 * (2 * kPairEntries + pairTableOffset(code, 2) / kBiasEntryBytes)];
  const float b6 = table[2 * (3 * kPairEntries + lastTableOffset(code) / kBiasEntryBytes)];
  return ((b01 + b23) + b45) + b6;
}

// 1 頭のオンラインの softmax。lane は問い合わせ 16qt + column の鍵 16kt + 2v + half（v < 8）を持ち、total はその 8 鍵の Σe（相方の
// 分は最後に足す）、out は O^T の断片（lane は次元 8·half 〜 8·half + 7）。
struct HeadState {
  float running = -INFINITY;
  float total = 0.0f;
  f32x8 out{};
};

// 鍵の塊 1 つで 1 頭を進める: 塊の最大で running を上げ、e = 2^(ℓ − running)、Σe と O^T を α = 2^(前の running − running) で
// 縮めてから O^T += V^T·P^T（hi・lo）。sum は e[0] から始める（e ≥ +0 なので 0 + e[0] とビットで同じ）。value は V^T の行。
__device__ __forceinline__ void attendBlock(HeadState& s, const float (&logit)[8], const uint16_t* value) {
  float block = logit[0];
#pragma unroll
  for (int v = 1; v < 8; ++v) {
    block = fmaxf(block, logit[v]);
  }
  block = fmaxf(block, swapHalves(block));
  const float next = fmaxf(s.running, block);
  const float alpha = fastExp2(s.running - next);
  s.running = next;
  float e[8];
  e[0] = fastExp2(logit[0] - next);
  float sum = e[0];
#pragma unroll
  for (int v = 1; v < 8; ++v) {
    e[v] = fastExp2(logit[v] - next);
    sum += e[v];
  }
  s.total = fmaf(s.total, alpha, sum);
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    s.out[v] *= alpha;
  }
  bf16x16 high;
  bf16x16 low;
  probabilityFragments(e, high, low);
  const bf16x16 values = fragment(value);
  s.out = mma(values, high, s.out);
  s.out = mma(values, low, s.out);
}

// 問い合わせの塊 qt の頭 head 〜 head + kHeads − 1 を鍵の塊 kt < blocks で回す（頭 0・1 を 1 波で、または頭 head だけ）。符号の
// 8 語は 1 塊先を読む。頭ごとの演算と順は kHeads に依らない（pairBias と headBias は頭ごとに同じ順で足す）。
template <int kHeads, class Shared>
__device__ __forceinline__ void attendKeys(HeadState (&state)[kHeads], const bf16x16 (&query)[kHeads], int head,
                                           const Shared& shared, const AttentionArgs& args,
                                           Bounds& bounds, int row, int qt, int blocks, int lane, int column,
                                           float scale) {
  const size_t codes = codeIndex(row, qt, 0) + static_cast<size_t>(lane) * 8;
  u32x4 nextCodes[2];
  {
    const size_t at = bounds.index(codes, 8, args.codes.limit);
    nextCodes[0] = *reinterpret_cast<const u32x4*>(args.codes.data + at);
    nextCodes[1] = *reinterpret_cast<const u32x4*>(args.codes.data + at + 4);
  }
  for (int kt = 0; kt < blocks; ++kt) {
    const u32x4 code0 = nextCodes[0];
    const u32x4 code1 = nextCodes[1];
    if (kt + 1 < blocks) {
      const size_t at = bounds.index(codes + static_cast<size_t>(kt + 1) * kCodeBlockWords, 8, args.codes.limit);
      nextCodes[0] = *reinterpret_cast<const u32x4*>(args.codes.data + at);
      nextCodes[1] = *reinterpret_cast<const u32x4*>(args.codes.data + at + 4);
    }
    const int key = kt * 16 + column;
    f32x8 score[kHeads];
#pragma unroll
    for (int i = 0; i < kHeads; ++i) {
      score[i] = mma(headFragment(shared.k, key, head + i), query[i], f32x8{});
    }
    float logit[kHeads][8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const uint32_t code = v < 4 ? code0[v] : code1[v - 4];
      if constexpr (kHeads == 2) {
        const float2 bias = pairBias(shared.bias, code);
        logit[0][v] = fmaf(score[0][v], scale, bias.x);
        logit[1][v] = fmaf(score[1][v], scale, bias.y);
      } else {
        logit[0][v] = fmaf(score[0][v], scale, headBias(shared.bias, code, head));
      }
    }
#pragma unroll
    for (int i = 0; i < kHeads; ++i) {
      attendBlock(state[i], logit[i], valueRow(shared.valueT, head + i, kt, column));
    }
  }
}

// 1 頭の出力 O / Σe（相方の Σe を足してから割る）を BF16 に 1 回丸め、問い合わせが行の中なら o の at（lane の次元 8·half 〜
// 8·half + 7）へ 16 バイトを 1 回で書く。
__device__ __forceinline__ void storeHead(const HeadState& s, bool inside, size_t at, const AttentionArgs& args,
                                          Bounds& bounds) {
  const float inverse = 1.0f / (s.total + swapHalves(s.total));
  float values[8];
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    values[v] = s.out[v] * inverse;
  }
  const u32x4 words = pack8Activation(values);
  if (inside) {
    *reinterpret_cast<u32x4*>(args.o.data + bounds.index(at, 8, args.o.limit)) = words;
  }
}

// K4 の射影の波の仕事 1 つ: 頭 h の塊 first 〜 first + kBlocks − 1（先頭の kQueries 個が局・席・牌の q を作る）の q・k・v を作り、
// 偏りを足して丸めて LDS へ置く（q は lane の 8 次元を fixedQuery へ、k・v は K2 と同じ置き方）。重みの断片を塊で共有するので、
// 1 歩の読みは重み 3（q を作らなければ 2）と n の kBlocks 個。偏りも塊で共有して 1 回読む。値は塊ごとに 1 頭を回すのとビットで
// 同じ（各 acc の被演算子と K の順、偏りの足し方と丸めが同じ）。
template <int H, int kBlocks, int kQueries, class Shared>
__device__ __forceinline__ void projectLast(Shared& shared, const AttentionArgs& args, Bounds& bounds,
                                            __amdgpu_buffer_rsrc_t weights, int row, int firstEvent, int tokens,
                                            int pair, int h, int first, int lane) {
  constexpr int A = H / 2;
  constexpr int kSets = kBlocks == 2 && kQueries > 0 ? 2 : 4;  // projectHead
  const int column = lane & 15;
  const int half = lane >> 4;
  const __amdgpu_buffer_rsrc_t inputs = bufferResource(args.normalized.data);
  uint32_t inputRow[kBlocks];
  uint32_t touched[kBlocks][3];
#pragma unroll
  for (int b = 0; b < kBlocks; ++b) {
    const int token = min(16 * (first + b) + column, tokens - 1);
    inputRow[b] = static_cast<uint32_t>(bounds.index(static_cast<size_t>(args.layout.index(row, firstEvent, token)) * H,
                                                     H, args.normalized.limit)) *
                  static_cast<uint32_t>(sizeof(uint16_t));
    // n の行の 2〜4 本目の 128 バイトの行を 1 語ずつ先に読み、MALL から引いておく（K2 と同じ。値は射影の後の空の asm まで生かす）。
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      touched[b][t] = __builtin_amdgcn_raw_buffer_load_b32(inputs, inputRow[b], 128 * (t + 1), 0);
    }
  }
  f32x8 acc[kBlocks][3];
  projectHead<H, kSets, kBlocks, kQueries>(acc, weights, static_cast<uint32_t>(column) * 16, h, inputs, inputRow);
  const int base = kPairWidth * pair + kHeadWidth * h;
  if constexpr (kQueries > 0) {
    const size_t at = bounds.index(static_cast<size_t>(base + 8 * half), 8, args.qkvBias.limit);
    const f32x8 bias = *reinterpret_cast<const f32x8*>(args.qkvBias.data + at);
#pragma unroll
    for (int b = 0; b < kQueries; ++b) {
      float values[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        values[v] = acc[b][0][v] + bias[v];
      }
      const int t = 16 * (first + b) + column;
      *reinterpret_cast<u32x4*>(&shared.fixedQuery[t][8 * ((2 * h + half) ^ ((t >> 1) & 3))]) = pack8Activation(values);
    }
  }
  {
    const float bias = args.qkvBias.data[bounds.index(A + base + column, 1, args.qkvBias.limit)];
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const int key = 16 * (first + b) + 2 * v + half;
        shared.k[key][swizzled(key, kHeadWidth * h + column)] = bf16RoundActivation(acc[b][1][v] + bias);
      }
    }
  }
  {
    const float bias = args.qkvBias.data[bounds.index(2 * A + base + column, 1, args.qkvBias.limit)];
    const int dimension = 2 * (column & 7) + (column >> 3);
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
      float values[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        values[v] = acc[b][2][v] + bias;
      }
      *reinterpret_cast<u32x4*>(&shared.valueT[kHeadWidth * h + dimension][16 * (first + b) + 8 * half]) =
          pack8Activation(values);
    }
  }
#pragma unroll
  for (int b = 0; b < kBlocks; ++b) {
    asm volatile("" ::"v"(touched[b][0]), "v"(touched[b][1]), "v"(touched[b][2]));
  }
}

// 組の核: kWaves 波・LDS は 16·kWaves トークン分。作業群の行は組の順の行の番号 rowOrder[groupStart + blockIdx.y]。
template <int H, bool kLast, int kWaves>
__global__ __launch_bounds__(kWave * kWaves) void attentionKernel(AttentionArgs args) {
  constexpr int kThreads = kWave * kWaves;
  constexpr int kTokens = 16 * kWaves;
  constexpr int A = H / 2;
  constexpr int kPairElements = (H / 16) * kStepBytes / static_cast<int>(sizeof(uint16_t));  // 頭の対の重みの写し
  // K4 で問い合わせの塊と頭の組ごとに波があれば、射影は 1 波が 2 塊の 1 頭を作って q を LDS に置き、注意は 1 波が 1 頭を回す。
  // なければ（W = 4 の組。q の 3 KB を LDS に足すと占有が 8 から 7 に下がる）1 波が 1 塊の 2 頭を作り、q はレジスタに残す。
  constexpr bool kHeadWaves = kLast && 2 * kFixedBlocks <= kWaves;
  constexpr int kBiasCopies = (kBiasChunks + kThreads - 1) / kThreads;  // スレッドが写す偏りの表の塊
  using Shared = AttentionShared<kTokens, kHeadWaves>;
  static_assert(sizeof(Shared) <= 64 * 1024, "LDS of one work group is 64 KiB");
  static_assert(H * sizeof(uint16_t) == 4 * 128, "an n row spans four 128-byte lines");
  static_assert(!kLast || kFixedBlocks <= kWaves, "K4 gives every query block a wave");
  static_assert(!kHeadWaves || kWaves % 2 == 0, "K4 pairs the waves of the two heads in the projection");
  static_assert(kThreads <= kBiasChunks, "every thread copies at least one chunk of the bias table");
  __shared__ Shared shared;

  const int pair = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int column = lane & 15;  // A・B の断片の行、C の断片の列
  const int half = lane >> 4;    // C の断片で奇数の行を持つ lane
  const TokenLayout layout = args.layout;
  Bounds bounds;

  const int row = args.rowOrder.data[bounds.index(static_cast<size_t>(args.groupStart) + blockIdx.y, 1,
                                                  args.rowOrder.limit)];
  const int firstEvent = args.rowColumn.data[bounds.index(
      static_cast<size_t>(row) * args.rowWidth + args.rowFirstEvent, 1, args.rowColumn.limit)];
  const int events = args.rowColumn.data[bounds.index(
      static_cast<size_t>(row) * args.rowWidth + args.rowEvents, 1, args.rowColumn.limit)];
  const int fixed = layout.fixed();
  // 規則の上限で 149 以下、組の上限で 16W 以下。範囲外と組の取り違えは検査つきの組み立てで止め、通常の組み立てでも LDS の外には
  // 書かない（組は attentionGroup が同じ切り詰めの T で決めるので、通常は切り詰めない）。
  const int tokens = min(fixed + max(events, 0), kTokens);
  (void)bounds.index(static_cast<size_t>(fixed + events), 0, kMaxRowTokens);
  (void)bounds.index(static_cast<size_t>(fixed + max(events, 0)), 0, kTokens);
  const int queries = kLast ? fixed : tokens;
  const int blocks = (tokens + 15) / 16;  // 16 トークンの塊（鍵の塊。W 以下）

  // 0. 2 頭の偏りの表を 16 バイトの塊で写す。スレッドは塊 tid + c·32W（c < kBiasCopies）を受け持ち、読みを全て先に出して待ちを 1 回
  //    にする（表の外の塊のスレッドは塊 tid を読み直して同じ所へ書き直す。書きに分岐がないので、読みが分岐の中へ沈まない）。
  {
    int at[kBiasCopies];
    f32x4 chunk[kBiasCopies];
#pragma unroll
    for (int c = 0; c < kBiasCopies; ++c) {
      at[c] = tid + c * kThreads < kBiasChunks ? tid + c * kThreads : tid;
      chunk[c] = biasChunk(args, bounds, pair, at[c]);
    }
#pragma unroll
    for (int c = 0; c < kBiasCopies; ++c) {
      reinterpret_cast<f32x4*>(shared.bias)[at[c]] = chunk[c];
    }
  }

  // 1. q・k・v（WMMA）。n と重みの写しは生の buffer で読み、n は lane の行の位置（行の全ての歩を含む）、写しは作業群の頭の対の 48 KB
  //    を範囲の検査にかける。行の外の lane は行の最後のトークンを読む（有限。その鍵は符号 kPaddedKeyCode で −∞ になって P = 0、その
  //    問い合わせは書かない）。q の A の断片の行 r は W_q の行 π(r) なので（写しがそう並べてある）、C の断片の lane はトークン
  //    16m + column の次元 8·half + v（v < 8）を持つ。
  const __amdgpu_buffer_rsrc_t weights = bufferResource(
      args.qkv.data + bounds.index(static_cast<size_t>(pair) * kPairElements, kPairElements, args.qkv.limit));
  bf16x16 keptQuery[2];  // 波の問い合わせの塊 m の q（頭ごとの B の断片。K4 の W ≥ 6 の組は LDS に置く）
  if constexpr (kHeadWaves) {
    // K4（W ≥ 6）: 波 w は頭 w & 1 で塊の対 2(w >> 1)・2(w >> 1) + 1（行の最後の塊が対の前の側なら 1 つ）を作る（projectLast）。
    //    塊は W 以下なので、対は全ての塊を 1 回ずつ覆う。q を作るのは局・席・牌の塊（16m < fixed）だけで、対の中では前の塊から
    //    続く。波の番号は SGPR に置く（頭は重みの写しの soffset に入るので、VGPR のままだと読みごとに lane を回す輪になる。塊の枝も
    //    スカラーになる）。
    const int uniformWave = __builtin_amdgcn_readfirstlane(wave);
    const int h = uniformWave & 1;
    const int first = 2 * (uniformWave >> 1);
    if (first < blocks) {
      const int pairQueries = min(max((fixed - 16 * first + 15) / 16, 0), 2);
      if (first + 1 < blocks) {
        if (pairQueries == 2) {
          projectLast<H, 2, 2>(shared, args, bounds, weights, row, firstEvent, tokens, pair, h, first, lane);
        } else if (pairQueries == 1) {
          projectLast<H, 2, 1>(shared, args, bounds, weights, row, firstEvent, tokens, pair, h, first, lane);
        } else {
          projectLast<H, 2, 0>(shared, args, bounds, weights, row, firstEvent, tokens, pair, h, first, lane);
        }
      } else if (pairQueries > 0) {
        projectLast<H, 1, 1>(shared, args, bounds, weights, row, firstEvent, tokens, pair, h, first, lane);
      } else {
        projectLast<H, 1, 0>(shared, args, bounds, weights, row, firstEvent, tokens, pair, h, first, lane);
      }
    }
  } else if (wave < blocks) {
    // 波 m（< 塊の数）が 16 トークンの塊 m の頭ごとに q・k・v の 3 つの断片を作り、q は問い合わせの塊 qt = m のものとしてレジスタに
    // 残す。断片の組（projectHead）は K2 が 3 つ、K4 が 4 つ。
    constexpr int kSets = kLast ? 4 : 3;
    const uint32_t fragmentRow = static_cast<uint32_t>(column) * 16;  // 写しの半分の中の lane の行（lane 16〜31 は 0〜15 と同じ）
    const __amdgpu_buffer_rsrc_t inputs = bufferResource(args.normalized.data);
    const int m = wave;
    const int token = min(16 * m + column, tokens - 1);
    const uint32_t inputRow[1] = {
        static_cast<uint32_t>(bounds.index(static_cast<size_t>(layout.index(row, firstEvent, token)) * H, H,
                                           args.normalized.limit)) *
        static_cast<uint32_t>(sizeof(uint16_t))};
    // K4 は局・席・牌より後の塊で q を作らない。
    const bool hasQuery = !kLast || 16 * m < fixed;
    // n の行（512 バイト）の 2〜4 本目の 128 バイトの行（歩 4〜15 で読む）を 1 語ずつ先に読み、MALL から引いておく。値は使わないが
    // 2 頭の射影の後の空の asm まで生かす（読みを消させず、その待ちも射影の後にだけ置かせる）。
    uint32_t touched[3];
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      touched[t] = __builtin_amdgcn_raw_buffer_load_b32(inputs, inputRow[0], 128 * (t + 1), 0);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int base = kPairWidth * pair + kHeadWidth * h;
      f32x8 acc[1][3];
      if (hasQuery) {
        projectHead<H, kSets, 1, 1>(acc, weights, fragmentRow, h, inputs, inputRow);
      } else {
        projectHead<H, kSets, 1, 0>(acc, weights, fragmentRow, h, inputs, inputRow);
      }
      // 偏りを足して丸める。q は相方と合わせて問い合わせの B の断片に。k は入れ替えた位置へ 1 値ずつ、v は V^T の行へ 8 値を 1 回で
      // 書く。
      if (hasQuery) {
        const size_t at = bounds.index(static_cast<size_t>(base + 8 * half), 8, args.qkvBias.limit);
        const f32x8 bias = *reinterpret_cast<const f32x8*>(args.qkvBias.data + at);
        float values[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          values[v] = acc[0][0][v] + bias[v];
        }
        keptQuery[h] = joinHalves(pack8Activation(values));
      }
      {
        const float bias = args.qkvBias.data[bounds.index(A + base + column, 1, args.qkvBias.limit)];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int key = 16 * m + 2 * v + half;
          shared.k[key][swizzled(key, kHeadWidth * h + column)] = bf16RoundActivation(acc[0][1][v] + bias);
        }
      }
      {
        const float bias = args.qkvBias.data[bounds.index(2 * A + base + column, 1, args.qkvBias.limit)];
        float values[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          values[v] = acc[0][2][v] + bias;
        }
        const int dimension = 2 * (column & 7) + (column >> 3);
        *reinterpret_cast<u32x4*>(&shared.valueT[kHeadWidth * h + dimension][16 * m + 8 * half]) =
            pack8Activation(values);
      }
      // 次の頭の先読みを、この頭の丸めの一時のレジスタと重ねない（重ねるとレジスタが増えて占有が下がる）。
      __builtin_amdgcn_sched_barrier(0);
    }
    asm volatile("" ::"v"(touched[0]), "v"(touched[1]), "v"(touched[2]));
  }
  phaseBarrier();

  // 2. 注意。lane は問い合わせ query = 16qt + column の鍵 16kt + 2v + half（v < 8）を持つ。
  const int queryBlocks = (queries + 15) / 16;
  const float scale = args.scale * kLog2e;
  if constexpr (kHeadWaves) {
    // 波 w が問い合わせの塊 w >> 1 の頭 w & 1 を回す。頭ごとの演算と順は 1 波で 2 頭を回すのと同じなので、o もビットで同じ。
    const int qt = wave >> 1;
    const int h = wave & 1;
    if (qt < queryBlocks) {
      const int query = qt * 16 + column;
      const bf16x16 queryFragment[1] = {headFragment(shared.fixedQuery, query, h)};
      HeadState state[1];
      attendKeys(state, queryFragment, h, shared, args, bounds, row, qt, blocks, lane, column, scale);
      const size_t at = static_cast<size_t>(layout.index(row, firstEvent, query)) * A + kPairWidth * pair + 8 * half;
      storeHead(state[0], query < queries, at + kHeadWidth * h, args, bounds);
    }
  } else {
    // 波が問い合わせの塊 qt = 波 を受け持ち、段 1 でレジスタに残した q で 2 頭を同時に回す（問い合わせの塊は鍵の塊より多くないので、
    // その波は段 1 で塊 qt を作り、その q を持つ。K4 の問い合わせの塊はどれも局・席・牌に掛かる）。
    const int qt = wave;
    if (qt < queryBlocks) {
      const int query = qt * 16 + column;
      const bf16x16 queryFragment[2] = {keptQuery[0], keptQuery[1]};
      HeadState state[2];
      attendKeys(state, queryFragment, 0, shared, args, bounds, row, qt, blocks, lane, column, scale);
      const size_t at = static_cast<size_t>(layout.index(row, firstEvent, query)) * A + kPairWidth * pair + 8 * half;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        storeHead(state[h], query < queries, at + kHeadWidth * h, args, bounds);
      }
    }
  }
  bounds.trap();
}

// K0 の後: 行の番号を組の順に並べる（組の中は行の順なので、並びは塊だけで決まる）。作業群 1 つが 1024 行ずつ、波の ballot で組ごとの
// 順位を数えて書く。置いた行の数がホストの数えた組の行の数と違えば、検査つきの組み立てで止める（行が組の 16W を超えれば K2・K4 も
// 止める）。
__global__ __launch_bounds__(kOrderThreads) void rowOrderKernel(RowOrderArgs args) {
  __shared__ int waveRows[kOrderWaves][kAttentionGroups];  // 今の 1024 行の波ごと・組ごとの行の数
  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  Bounds bounds;
  int placed[kAttentionGroups] = {};  // 前の 1024 行までに置いた組ごとの行の数（どのスレッドも同じ）
  for (int first = 0; first < args.rows; first += kOrderThreads) {
    const int row = first + tid;
    int group = -1;
    if (row < args.rows) {
      const int events = args.rowColumn.data[bounds.index(
          static_cast<size_t>(row) * args.rowWidth + args.rowEvents, 1, args.rowColumn.limit)];
      group = attentionGroup(args.fixed, events);
    }
    int rank = 0;  // 波の中で前の lane にある同じ組の行の数
#pragma unroll
    for (int g = 0; g < kAttentionGroups; ++g) {
      const uint32_t mask = __builtin_amdgcn_ballot_w32(group == g);
      if (lane == 0) {
        waveRows[wave][g] = __builtin_popcount(mask);
      }
      rank = group == g ? __builtin_popcount(mask & ((1u << lane) - 1u)) : rank;
    }
    __syncthreads();
    int at = 0;
#pragma unroll
    for (int g = 0; g < kAttentionGroups; ++g) {
      int before = 0;
      int total = 0;
      for (int w = 0; w < kOrderWaves; ++w) {
        const int count = waveRows[w][g];
        before += w < wave ? count : 0;
        total += count;
      }
      at = group == g ? args.groupStart[g] + placed[g] + before + rank : at;
      placed[g] += total;
    }
    if (group >= 0) {
      args.rowOrder.data[bounds.index(static_cast<size_t>(at), 1, args.rowOrder.limit)] = row;
    }
    __syncthreads();
  }
  // 置いた数とホストの数が等しい（≤ と ≥ を範囲の検査で照らす）。
#pragma unroll
  for (int g = 0; g < kAttentionGroups; ++g) {
    (void)bounds.index(static_cast<size_t>(placed[g]), 0, static_cast<size_t>(args.groupRows[g]));
    (void)bounds.index(static_cast<size_t>(args.groupRows[g]), 0, static_cast<size_t>(placed[g]));
  }
  bounds.trap();
}

// 組 kGroup の行（rows 行、rowOrder の start から）を起動し、start を次の組の先頭へ進める。行の数 0 の組は起動しない。
template <int H, bool kLast, int kGroup>
void launchGroup(AttentionArgs args, int& start, int rows, int headPairs, hipStream_t stream) {
  constexpr int kWaves = attentionWaves(kGroup);
  if (rows > 0) {
    args.groupStart = start;
    // 同じ行の頭の対が隣どうしで走る（行の x と符号を L2 で使い回す）。
    hipLaunchKernelGGL((attentionKernel<H, kLast, kWaves>), dim3(headPairs, rows), dim3(kWave * kWaves), 0, stream,
                       args);
    PRISM_HIP(hipGetLastError());
  }
  start += rows;
}

template <int H, bool kLast, int... kGroups>
void launchGroups(const AttentionArgs& args, const int (&groupRows)[kAttentionGroups], int headPairs,
                  hipStream_t stream, std::integer_sequence<int, kGroups...>) {
  int start = 0;
  (launchGroup<H, kLast, kGroups>(args, start, groupRows[kGroups], headPairs, stream), ...);
}

// 重みの公開のとき: 1 層の W_qkv を射影の重みの写し（上）に並べ替える。作業群は 1 頭の対で、スレッドは写しの 16 バイト（W_qkv の
// 行 unit(j, h, r) の K の 8 要素）を 1 つずつ写す。
template <int H>
__global__ __launch_bounds__(kQkvThreads) void qkvFragmentKernel(QkvFragmentArgs args) {
  constexpr int A = H / 2;
  constexpr int kChunks = (H / 16) * kStepBytes / 16;  // 頭の対の写しの 16 バイトの数
  const int pair = blockIdx.x;
  Bounds bounds;

  for (int c = threadIdx.x; c < kChunks; c += kQkvThreads) {
    // c = (((k·2 + h)·3 + j)·2 + upper)·16 + r（upper は K の上の 8 要素）。
    const int r = c % 16;
    const int upper = (c / 16) % 2;
    const int j = (c / 32) % 3;
    const int h = (c / 96) % 2;
    const int k = c / 192;
    const int base = kPairWidth * pair + kHeadWidth * h;
    const int unit = j == 0 ? base + (r >> 1) + 8 * (r & 1) : j * A + base + r;
    const size_t from = bounds.index(static_cast<size_t>(unit) * H + 16 * k + 8 * upper, 8, args.qkv.limit);
    const size_t to = bounds.index((static_cast<size_t>(pair) * kChunks + c) * 8, 8, args.fragments.limit);
    *reinterpret_cast<u32x4*>(args.fragments.data + to) = *reinterpret_cast<const u32x4*>(args.qkv.data + from);
  }
  bounds.trap();
}

}  // namespace

void launchRowOrder(const RowOrderArgs& args, hipStream_t stream) {
  if (args.rows <= 0) {
    return;
  }
  hipLaunchKernelGGL(rowOrderKernel, dim3(1), dim3(kOrderThreads), 0, stream, args);
  PRISM_HIP(hipGetLastError());
}

void launchAttention(int hidden, bool last, const AttentionArgs& args, int headPairs,
                     const int (&groupRows)[kAttentionGroups], hipStream_t stream) {
  if (args.layout.rows <= 0) {
    return;
  }
  constexpr auto groups = std::make_integer_sequence<int, kAttentionGroups>{};
  switch (hidden) {
    case 256:
      if (last) {
        launchGroups<256, true>(args, groupRows, headPairs, stream, groups);
      } else {
        launchGroups<256, false>(args, groupRows, headPairs, stream, groups);
      }
      break;
    default:
      throw std::invalid_argument("K2/K4 have no specialisation for hidden " + std::to_string(hidden));
  }
}

void launchQkvFragments(int hidden, const QkvFragmentArgs& args, int headPairs, hipStream_t stream) {
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL((qkvFragmentKernel<256>), dim3(headPairs), dim3(kQkvThreads), 0, stream, args);
      break;
    default:
      throw std::invalid_argument("the QKV fragment copy has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
