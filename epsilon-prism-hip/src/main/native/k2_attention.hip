// K2（attn0）と K4（attn1）: 作業群は 1 頭の対 × 1 行（256 スレッド = 8 波）。eager の EncoderLayer の注意の半分（O 射影は K3・K5）。
//
//   0. 2 頭の偏りの表（×log2e）を LDS へ写す（重みの公開のときに relationBiasKernel が層ごとに作ったもの。prism_hip_relation.h）
//   1. 頭の対の q・k・v = ⌊n·W_qkvᵀ + b⌉（WMMA。K4 は q を局・席・牌の 39 個だけ）。n = ⌊(x·r)·g⌉ は前の段（K1・K3）が
//      NORMALIZED に書いたもので、断片としてそのまま読む。W_qkv は重みの公開のときに qkvFragmentKernel が断片の順に並べ替えた写し
//      （下）を読む。k・v は LDS へ。q は転置の qᵀ = W_q·nᵀ で作る。K2 は問い合わせの B の断片のままレジスタに残し（塊 m を作る波が
//      問い合わせの塊 m を受け持つ。波の 2 つ目の塊（T > 128）の q だけ LDS へ）、K4 は全て LDS へ置く
//   2. 問い合わせの塊（16 個）ごとに、K2 は 1 波が 2 頭を、K4 は 1 波が 1 頭を回す（K4 の問い合わせの塊は 3 つ以下なので 6 波以下）。
//      鍵の塊（16 個）ごとに
//        S^T = K·Qᵀ（WMMA、BF16 の積を FP32 で累積）、ℓ = S·scale + Σ_s ρ[16s + c_s(i, j)]（c は K0 の符号。ρ は区画 2 つずつ前もって
//        足した表を引く）、オンラインの softmax（exp2）、O^T += V^T·P^T（WMMA。P は FP32 の e を hi・lo の 2 つの BF16 に分けて 2 回掛け、
//        FP32 に近い精度を保つ）
//      最後に O / Σe を BF16 に 1 回丸める。
//
// 段 1 と段 2 の間の待ち合わせは LDS の順序だけ（phaseBarrier）。q・k・v・o の丸めは bf16Round（NaN の選択つき）。
//
// eager の packedRelationScaledDotProductAttention と式は同じで、和の順と exp だけが違う（相対 RMS ≤ 2e-2 の約束の内）。縮約は全て行の
// 中で順は固定、原子的な加算もない。WMMA の要素は互いに独立なので、同じ行の o は塊の組み方に依らずビットで同じ。
//
// 射影の重みの写し（qkvFragmentKernel。重みの公開のときに層ごとに 1 回）: 頭の対ごとに [歩 k][頭 h][j = q・k・v][半分][行 r][8]
// = W_qkv[unit(j, h, r)][16k + 8·半分 .. +8]。unit(q, h, r) = 32·対 + 16h + π(r)（π(r) = (r >> 1) + 8(r & 1)）、
// unit(k, h, r) = A + 32·対 + 16h + r、unit(v, h, r) = 2A + 32·対 + 16h + r。断片 1 つは 512 バイト、歩 1 つは 3 KB、頭の対は 48 KB。
// lane は行 column の 16 バイトを lane の位置 column·16 と波で同じ soffset k·3072 + (3h + j)·512（上の 8 要素は + 256）で読むので、
// b128 の 1 回は続いた 256 バイト（2 本の cache line。W_qkv をそのまま読むと 512 バイト離れた 16 行）。値と WMMA の断片は同じ。
#include "prism_hip_relation.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / kWave;
constexpr int kHeadWidth = 16;
constexpr int kPairWidth = 2 * kHeadWidth;                 // 頭の対の q・k の列
constexpr int kValueStride = kMaxRowTokens + 8;            // V^T の行（bank の衝突を避ける詰め物）
constexpr int kFixedBlocks = (kMaxFixedTokens + 15) / 16;  // K4 の問い合わせの塊
static_assert(2 * kFixedBlocks <= kWaves, "K4 gives every query block and head its own wave");
static_assert(kBiasEntries % 2 == 0, "the bias table is copied 16 bytes at a time");
static_assert(kThreads <= kBiasEntries / 2 && kBiasEntries / 2 <= 2 * kThreads,
              "every thread copies one or two 16-byte chunks of the bias table");

// 射影の重みの写しの置き方（上）。
constexpr int kFragmentBytes = 512;                 // 断片 1 つ（K の下の 8 要素の 16 行、上の 8 要素の 16 行）
constexpr int kStepBytes = 2 * 3 * kFragmentBytes;  // 歩 1 つ（2 頭 × q・k・v）

// 問い合わせの q の LDS（k と同じ入れ替えで置く）。K2 は波の 2 つ目の問い合わせの塊（m ≥ 8、T > 128）の q だけを置き、最初の塊の q は
// レジスタに残す。K4 は 1 波が 1 頭を回すので、問い合わせ（局・席・牌の 48 以下）の q を全て置く。
template <bool kLast>
struct QueryStash {
  uint16_t queryStash[kMaxRowTokens - 16 * kWaves][kPairWidth];
};

template <>
struct QueryStash<true> {
  uint16_t fixedQuery[kMaxFixedTokens][kPairWidth];
};

// LDS。k は [トークン][2 頭 × 16] の行の 16 バイトの塊を (トークン >> 1) & 3 で入れ替えて置く（8 lane の b128 の読みで bank が衝突
// しない）。V^T は [16h + r(d)][κ] で、r(d) = 2(d & 7) + (d >> 3)、鍵 16m + 2v + half は列 16m + v + 8·half。この並びで O^T の断片の
// lane が続いた 8 次元を持ち、P^T の B の断片は lane と相方の 8 値を続けるだけになる。
template <bool kLast>
struct alignas(16) AttentionShared : QueryStash<kLast> {
  uint16_t k[kMaxRowTokens][kPairWidth];
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

// 射影の K の 1 歩の断片: 頭 h の q・k・v（j = 0・1・2）の重みの断片（写しの soffset k·3072 + (3h + j)·512）と n の行の 32k バイト。
// kSkip より前（K4 の q を作らない塊の q）は読まず掛けない。
template <int kSkip>
struct ProjectionStep {
  bf16x16 weight[3];
  bf16x16 input;

  __device__ __forceinline__ void load(__amdgpu_buffer_rsrc_t weights, uint32_t fragmentRow, int h,
                                       __amdgpu_buffer_rsrc_t inputs, uint32_t inputRow, int k) {
#pragma unroll
    for (int j = kSkip; j < 3; ++j) {
      const int at = k * kStepBytes + (3 * h + j) * kFragmentBytes;
      weight[j] = bufferFragment(weights, fragmentRow, at, at + kFragmentBytes / 2);
    }
    input = bufferFragment(inputs, inputRow, 32 * k, 32 * k + 16);
  }

  // acc[0] += W_q·nᵀ（転置の q）、acc[1]・acc[2] += n·W_k,vᵀ。
  __device__ __forceinline__ void multiply(f32x8 (&acc)[3]) const {
    if constexpr (kSkip == 0) {
      acc[0] = mma(weight[0], input, acc[0]);
    }
#pragma unroll
    for (int j = 1; j < 3; ++j) {
      acc[j] = mma(input, weight[j], acc[j]);
    }
  }
};

// 頭 h の射影 acc（ProjectionStep::multiply の式）を K の 16 歩で作る。断片の組 kSets 個を回して kSets − 1 歩先まで読みながら掛ける。
// 読みは全て無条件なので輪は 1 つの基本塊のまま（組を写す mov がない）、sched_barrier が次の読みを掛け算の後ろに留める。
//   組 3 つ（K2）: 展開しない 3 歩の輪と最後の 1 歩。最後の周の 1 つは歩 15 を読み直す（使わない）
//   組 4 つ（K4）: 3 歩を先に読み、展開しない 4 歩の輪と最後の 4 歩（読み直しはない）
// 組 2 つでは待ちが 1 歩ぶんしか隠れず、展開すると断片が重なってレジスタが溢れる。K2 は組 4 つにするとレジスタが増えて占有が下がる
// ので組 3 つにする（K4 は組 4 つでも下がらない）。どちらも各 acc の K の順は 0〜15 で同じ。
template <int H, int kSkip, int kSets>
__device__ __forceinline__ void projectHead(f32x8 (&acc)[3], __amdgpu_buffer_rsrc_t weights, uint32_t fragmentRow,
                                            int h, __amdgpu_buffer_rsrc_t inputs, uint32_t inputRow) {
  constexpr int kSteps = H / 16;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    acc[j] = f32x8{};
  }
  ProjectionStep<kSkip> a;
  ProjectionStep<kSkip> b;
  ProjectionStep<kSkip> c;
  a.load(weights, fragmentRow, h, inputs, inputRow, 0);
  b.load(weights, fragmentRow, h, inputs, inputRow, 1);
  if constexpr (kSets == 3) {
    static_assert(kSteps % 3 == 1, "the rolled loop covers the steps in threes plus the last one");
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
  } else {
    static_assert(kSets == 4 && kSteps % 4 == 0, "the rolled loop covers the steps in fours");
    ProjectionStep<kSkip> d;
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

// 符号 code の 4 つの表の項の和 Σ_s ρ（×log2e）。表の項は (頭 2p, 頭 2p + 1) の値の対で、2 頭は対を 1 回で引く。
__device__ __forceinline__ float2 pairBias(const float2 (&bias)[kBiasEntries], uint32_t code) {
  const float2 b01 = bias[code & 0xffu];
  const float2 b23 = bias[kPairEntries + ((code >> 8) & 0xffu)];
  const float2 b45 = bias[2 * kPairEntries + ((code >> 16) & 0xffu)];
  const float2 b6 = bias[3 * kPairEntries + (code >> 24)];
  return make_float2(((b01.x + b23.x) + b45.x) + b6.x, ((b01.y + b23.y) + b45.y) + b6.y);
}

// 同じ和の頭 head（0・1）の分だけ（対の片方を引く。足す順は pairBias と同じ）。
__device__ __forceinline__ float headBias(const float2 (&bias)[kBiasEntries], uint32_t code, int head) {
  const float* table = reinterpret_cast<const float*>(bias) + head;
  const float b01 = table[2 * (code & 0xffu)];
  const float b23 = table[2 * (kPairEntries + ((code >> 8) & 0xffu))];
  const float b45 = table[2 * (2 * kPairEntries + ((code >> 16) & 0xffu))];
  const float b6 = table[2 * (3 * kPairEntries + (code >> 24))];
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

// 問い合わせの塊 qt の頭 head 〜 head + kHeads − 1 を鍵の塊 kt < blocks で回す（K2 は頭 0・1 を 1 波で、K4 は頭 head だけ）。符号の
// 8 語は 1 塊先を読む。頭ごとの演算と順は kHeads に依らない。
template <int kHeads, bool kLast>
__device__ __forceinline__ void attendKeys(HeadState (&state)[kHeads], const bf16x16 (&query)[kHeads], int head,
                                           const AttentionShared<kLast>& shared, const AttentionArgs& args,
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
  const u32x4 words = pack8(values);
  if (inside) {
    *reinterpret_cast<u32x4*>(args.o.data + bounds.index(at, 8, args.o.limit)) = words;
  }
}

template <int H, bool kLast>
__global__ __launch_bounds__(kThreads) void attentionKernel(AttentionArgs args) {
  constexpr int A = H / 2;
  constexpr int kPairElements = (H / 16) * kStepBytes / static_cast<int>(sizeof(uint16_t));  // 頭の対の重みの写し
  constexpr int kSets = kLast ? 4 : 3;  // 射影の断片の組（projectHead）
  static_assert(sizeof(AttentionShared<kLast>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  static_assert(H * sizeof(uint16_t) == 4 * 128, "an n row spans four 128-byte lines");
  __shared__ AttentionShared<kLast> shared;

  const int pair = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int column = lane & 15;  // A・B の断片の行、C の断片の列
  const int half = lane >> 4;    // C の断片で奇数の行を持つ lane
  const TokenLayout layout = args.layout;
  Bounds bounds;

  const int firstEvent = args.rowColumn.data[bounds.index(
      static_cast<size_t>(row) * args.rowWidth + args.rowFirstEvent, 1, args.rowColumn.limit)];
  const int events = args.rowColumn.data[bounds.index(
      static_cast<size_t>(row) * args.rowWidth + args.rowEvents, 1, args.rowColumn.limit)];
  const int fixed = layout.fixed();
  // 規則の上限で 149 以下。範囲外は検査つきの組み立てで止め、通常の組み立てでも LDS の外には書かない。
  const int tokens = min(fixed + max(events, 0), kMaxRowTokens);
  (void)bounds.index(static_cast<size_t>(fixed + events), 0, kMaxRowTokens);
  const int queries = kLast ? fixed : tokens;
  const int blocks = (tokens + 15) / 16;  // 16 トークンの塊（鍵の塊）

  // 0. 2 頭の偏りの表を 16 バイトの塊で写す。スレッドは塊 tid と tid + 256 を受け持ち、2 つの読みを先に出して待ちを 1 回にする
  //    （2 つ目の塊が表の外のスレッドは 1 つ目の塊を読み直して同じ所へ書き直す。書きに分岐がないので、読みが分岐の中へ沈まない）。
  {
    const int second = tid + kThreads < kBiasEntries / 2 ? tid + kThreads : tid;
    const f32x4 low = biasChunk(args, bounds, pair, tid);
    const f32x4 high = biasChunk(args, bounds, pair, second);
    reinterpret_cast<f32x4*>(shared.bias)[tid] = low;
    reinterpret_cast<f32x4*>(shared.bias)[second] = high;
  }

  // 1. q・k・v（WMMA）。波は 16 トークンの塊 m ≡ 波 (mod 8) を後ろから回し、頭ごとに q・k・v の 3 つの断片を作る。n と重みの写しは
  //    生の buffer で読み、n は lane の行の位置（行の全ての歩を含む）、写しは作業群の頭の対の 48 KB を範囲の検査にかける。行の外の
  //    lane は行の最後のトークンを読む（有限。その鍵は符号 kPaddedKeyCode で −∞ になって P = 0、その問い合わせは書かない）。q の
  //    A の断片の行 r は W_q の行 π(r) なので（写しがそう並べてある）、C の断片の lane はトークン 16m + column の次元 8·half + v
  //    （v < 8）を持つ。K2 が最後に作る q は問い合わせの塊 qt = 波 のもので、レジスタに残す。
  const __amdgpu_buffer_rsrc_t weights = bufferResource(
      args.qkv.data + bounds.index(static_cast<size_t>(pair) * kPairElements, kPairElements, args.qkv.limit));
  const uint32_t fragmentRow = static_cast<uint32_t>(column) * 16;  // 写しの半分の中の lane の行（lane 16〜31 は 0〜15 と同じ）
  const __amdgpu_buffer_rsrc_t inputs = bufferResource(args.normalized.data);
  bf16x16 keptQuery[2];  // K2 の波の最初の問い合わせの塊の q（頭ごとの B の断片）
  const int owned = (blocks - wave + kWaves - 1) / kWaves;  // この波の塊 m = 波 + 8i（i < owned）
  for (int i = owned - 1; i >= 0; --i) {
    const int m = wave + kWaves * i;
    const int token = min(16 * m + column, tokens - 1);
    const uint32_t inputRow =
        static_cast<uint32_t>(bounds.index(static_cast<size_t>(layout.index(row, firstEvent, token)) * H, H,
                                           args.normalized.limit)) *
        sizeof(uint16_t);
    // K4 は局・席・牌より後の塊で q を作らない。
    const bool hasQuery = !kLast || 16 * m < fixed;
    // n の行（512 バイト）の 2〜4 本目の 128 バイトの行（歩 4〜15 で読む）を 1 語ずつ先に読み、MALL から引いておく。値は使わないが
    // 2 頭の射影の後の空の asm まで生かす（読みを消させず、その待ちも射影の後にだけ置かせる）。
    uint32_t touched[3];
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      touched[t] = __builtin_amdgcn_raw_buffer_load_b32(inputs, inputRow, 128 * (t + 1), 0);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int base = kPairWidth * pair + kHeadWidth * h;
      f32x8 acc[3];
      if (hasQuery) {
        projectHead<H, 0, kSets>(acc, weights, fragmentRow, h, inputs, inputRow);
      } else {
        projectHead<H, 1, kSets>(acc, weights, fragmentRow, h, inputs, inputRow);
      }
      // 偏りを足して丸める。q は K2 なら相方と合わせて問い合わせの B の断片に（波の 2 つ目の塊は lane の 8 次元を LDS へ）、K4 なら
      // lane の 8 次元を LDS へ。k は入れ替えた位置へ 1 値ずつ、v は V^T の行へ 8 値を 1 回で書く。
      if (hasQuery) {
        const size_t at = bounds.index(static_cast<size_t>(base + 8 * half), 8, args.qkvBias.limit);
        const f32x8 bias = *reinterpret_cast<const f32x8*>(args.qkvBias.data + at);
        float values[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          values[v] = acc[0][v] + bias[v];
        }
        const u32x4 own = pack8(values);
        if constexpr (kLast) {
          const int t = 16 * m + column;
          *reinterpret_cast<u32x4*>(&shared.fixedQuery[t][8 * ((2 * h + half) ^ ((t >> 1) & 3))]) = own;
        } else if (m < kWaves) {
          keptQuery[h] = joinHalves(own);
        } else {
          const int t = 16 * (m - kWaves) + column;
          *reinterpret_cast<u32x4*>(&shared.queryStash[t][8 * ((2 * h + half) ^ ((t >> 1) & 3))]) = own;
        }
      }
      {
        const float bias = args.qkvBias.data[bounds.index(A + base + column, 1, args.qkvBias.limit)];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int key = 16 * m + 2 * v + half;
          shared.k[key][swizzled(key, kHeadWidth * h + column)] = bf16Round(acc[1][v] + bias);
        }
      }
      {
        const float bias = args.qkvBias.data[bounds.index(2 * A + base + column, 1, args.qkvBias.limit)];
        float values[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          values[v] = acc[2][v] + bias;
        }
        const int dimension = 2 * (column & 7) + (column >> 3);
        *reinterpret_cast<u32x4*>(&shared.valueT[kHeadWidth * h + dimension][16 * m + 8 * half]) =
            pack8(values);
      }
      // 次の頭の先読みを、この頭の丸めの一時のレジスタと重ねない（重ねるとレジスタが増えて K2 の占有が下がる）。
      __builtin_amdgcn_sched_barrier(0);
    }
    asm volatile("" ::"v"(touched[0]), "v"(touched[1]), "v"(touched[2]));
  }
  phaseBarrier();

  // 2. 注意。lane は問い合わせ query = 16qt + column の鍵 16kt + 2v + half（v < 8）を持つ。
  const int queryBlocks = (queries + 15) / 16;
  const float scale = args.scale * kLog2e;
  if constexpr (kLast) {
    // K4: 波 w が問い合わせの塊 w >> 1 の頭 w & 1 を回す。頭ごとの演算と順は 1 波で 2 頭を回すのと同じなので、o もビットで同じ。
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
    // K2: 波が問い合わせの塊 qt を受け持ち、2 頭を同時に回す。問い合わせの B の断片は、波の最初の塊（qt = 波）はレジスタに残した q、
    // 2 つ目の塊は LDS に置いた q。
    for (int qt = wave; qt < queryBlocks; qt += kWaves) {
      const int query = qt * 16 + column;
      bf16x16 queryFragment[2] = {keptQuery[0], keptQuery[1]};
      if (qt >= kWaves) {
        queryFragment[0] = headFragment(shared.queryStash, query - 16 * kWaves, 0);
        queryFragment[1] = headFragment(shared.queryStash, query - 16 * kWaves, 1);
      }
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

// 重みの公開のとき: 1 層の W_qkv を射影の重みの写し（上）に並べ替える。作業群は 1 頭の対で、スレッドは写しの 16 バイト（W_qkv の
// 行 unit(j, h, r) の K の 8 要素）を 1 つずつ写す。
template <int H>
__global__ __launch_bounds__(kThreads) void qkvFragmentKernel(QkvFragmentArgs args) {
  constexpr int A = H / 2;
  constexpr int kChunks = (H / 16) * kStepBytes / 16;  // 頭の対の写しの 16 バイトの数
  const int pair = blockIdx.x;
  Bounds bounds;

  for (int c = threadIdx.x; c < kChunks; c += kThreads) {
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

void launchAttention(int hidden, bool last, const AttentionArgs& args, int headPairs, hipStream_t stream) {
  if (args.layout.rows <= 0) {
    return;
  }
  // 同じ行の頭の対が隣どうしで走る（行の x と符号を L2 で使い回す）。
  const dim3 grid(headPairs, args.layout.rows);
  switch (hidden) {
    case 256:
      if (last) {
        hipLaunchKernelGGL((attentionKernel<256, true>), grid, dim3(kThreads), 0, stream, args);
      } else {
        hipLaunchKernelGGL((attentionKernel<256, false>), grid, dim3(kThreads), 0, stream, args);
      }
      break;
    default:
      throw std::invalid_argument("K2/K4 have no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

void launchQkvFragments(int hidden, const QkvFragmentArgs& args, int headPairs, hipStream_t stream) {
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL((qkvFragmentKernel<256>), dim3(headPairs), dim3(kThreads), 0, stream, args);
      break;
    default:
      throw std::invalid_argument("the QKV fragment copy has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
