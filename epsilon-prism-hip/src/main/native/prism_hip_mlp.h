// MLP の段の部品（K3・K5・K6・K8 で共通）。タイルの行数 Rows は LDS の配列の型が持つ（K3・K5 は k3_mlp.hip の kMlpRows、K6 は
// k6_candidate.hip の kTile、K8 は k8_key_value.hip の kKeyRows。どれも 48）。行列積は WMMA の BF16 で、累積は FP32。⌊⌉ は BF16 への
// 最近接偶数の丸め（NaN を特別に扱わない活性の丸め bf16RoundActivation。prism_hip_common.h）。
//
// mlpStage は K3 の式:
//   x1 = ⌊x + ⌊o·Woᵀ + bo⌉⌉ → n = ⌊(x1·r)·g⌉（r = rsqrt(mean(x1²) + ε)、FP32）
//   → F を 128 ずつ h = ⌊silu⌊n·W1ᵀ + b1⌉⌉（LDS）、acc += h·W2ᵀ（FP32）→ x = ⌊x1 + ⌊acc + b2⌉⌉
// リダクションは全て行の中で行い、K の向きの順は固定（アトミックな加算も K の分割もない）。同じ行の結果は、タイルの組み方にも
// タイルの行数にも依らずビットで同じ。
//
// 待ちを減らす工夫: 行列積は全てソフトウェアパイプラインにする。ワークグループのバリアは LDS だけにかける（barrier）。写し
// （stageTile・storeTile）は分岐なしで読み込みを一度に出す。h の silu は siluFast（prism_hip_common.h）。LDS に依らない読み込み
// （バイアスと down 射影の最初の B）はバリアと epilogue より前に出し、次の段の up 射影の最初の B は down 射影より前に出す。RMSNorm の
// 結果を LDS に戻さないときは、normalizeStore がレジスタからグローバルメモリへ直に書く。
//
// mlpStage とキー・バリューの射影（projectKeyValue）の重みは、計画が公開のときに作る、フラグメントの順に並べた写し（FragmentWeights）
// を生の buffer で読む。アドレスはウェーブで同じ soffset と lane の voffset だけで決まり（SALU で求める）、境界の照合は列のタイルごとに
// 1 回。B は 2 ステップ先に出す（gemmFragmentsFrom）。値と K の順は行の順の重みを読む形（gemmTile）と同じなので、結果はビットで同じ。
// gemmTile は K6 のほかの行列積（tileEntry・query・discardHidden）が使う。
#pragma once

#include "prism_hip_common.h"
#include "prism_hip_kernels.h"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE__) && (__AMDGCN_WAVEFRONT_SIZE__ != 32)
#error "the PRISM kernels need wave32 (gfx11 WMMA)"
#endif

namespace prism {
namespace tile {

constexpr int kThreads = 256;          // ワークグループのスレッド数
constexpr int kWaves = kThreads / kWave;
constexpr int kChunk = 128;            // F を区切る幅（隠れ層 h の LDS の列）
constexpr int kPad = 8;                // LDS の行のパディング（BF16 の要素数。行の間の bank の衝突を避ける）

// LDS。stream は x → x1 → n → x2 の順に、inner は o → h の順に使い回す。Rows は 16 の倍数で、RMSNorm（4 スレッドで 1 行）の受け持ちが
// ウェーブの境界で切れる数。
template <int H, int Rows>
struct alignas(16) MlpShared {
  static_assert(Rows % 16 == 0 && 4 * Rows <= kThreads && 4 * Rows % kWave == 0, "unsupported tile rows");
  uint16_t stream[Rows][H + kPad];
  uint16_t inner[Rows][kChunk + kPad];
};

// ワークグループのバリア（LDS の順序だけを守る）。__syncthreads は s_barrier の後に buffer_gl0_inv（WGP モードの acquire で、
// グローバルメモリの L0 を捨てる）を出すが、K3・K5・K6 はカーネルの中で読む領域と書く領域が別なので要らない（両方に使うのは K3 の x
// だけで、読むのは最初のバリアより前）。カーネルの中でグローバルメモリに書いてから別のウェーブが読む組を足すなら、その間は
// __syncthreads に戻すこと。
__device__ __forceinline__ void barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Rows 行 × Width 列の写しで、スレッド tid が pass 回目に受け持つ 16 バイトの行（列は (tid % kVectors)·8 で pass に依らない）。
// 「tid の行 + pass ごとの定数」と書くと、LDS とグローバルメモリのアドレスは 1 つの基底に定数を足すだけになる（ds の offset に
// 折り込まれる）。(pass·kThreads + tid) / kVectors の形だと、アドレスが段をまたいで同じ式としてまとめられ（K3 の x の写しと書き出し、
// K6 の段 1 と段 6）、pass ごとの値を長くレジスタに抱えて spill する。
template <int Width, int Rows>
__device__ __forceinline__ int tileRow(int pass, int tid) {
  constexpr int kVectors = Width / 8;
  static_assert(kThreads % kVectors == 0 && Rows * kVectors % kThreads == 0,
                "the tile must split evenly over the work group");
  return tid / kVectors + pass * (kThreads / kVectors);
}

// Rows 行 × Width 列の BF16 を、グローバルメモリから LDS へ 16 バイトずつ写す。タイルの外の行は 0 にする。読み込みは行をタイルの
// 最後の行に寄せて（rows ≥ 1）分岐をなくし、kPasses 個を一度に出してから LDS に書く。
template <int Width, int Rows, int Stride>
__device__ __forceinline__ void stageTile(uint16_t (&lds)[Rows][Stride], const uint16_t* __restrict__ global, size_t limit,
                                          int token0, int rows, int tid, Bounds& bounds) {
  constexpr int kVectors = Width / 8;
  constexpr int kPasses = Rows * kVectors / kThreads;
  const int col = (tid % kVectors) * 8;
  u32x4 values[kPasses];
#pragma unroll
  for (int pass = 0; pass < kPasses; ++pass) {
    const int row = min(tileRow<Width, Rows>(pass, tid), rows - 1);
    const size_t at = bounds.index(static_cast<size_t>(token0 + row) * Width + col, 8, limit);
    values[pass] = *reinterpret_cast<const u32x4*>(global + at);
  }
  const u32x4 zero = {0u, 0u, 0u, 0u};
#pragma unroll
  for (int pass = 0; pass < kPasses; ++pass) {
    const int row = tileRow<Width, Rows>(pass, tid);
    *reinterpret_cast<u32x4*>(&lds[row][col]) = row < rows ? values[pass] : zero;
  }
}

// LDS の Rows 行 × Width 列を、グローバルメモリへ 16 バイトずつ書く。タイルの外の行は書かない。LDS は先に全て読み、行が埋まった
// タイル（ワークグループで一様）は分岐なしで書く。
template <int Width, int Rows, int Stride>
__device__ __forceinline__ void storeTile(uint16_t* __restrict__ global, size_t limit, const uint16_t (&lds)[Rows][Stride],
                                          int token0, int rows, int tid, Bounds& bounds) {
  constexpr int kVectors = Width / 8;
  constexpr int kPasses = Rows * kVectors / kThreads;
  const int col = (tid % kVectors) * 8;
  u32x4 values[kPasses];
#pragma unroll
  for (int pass = 0; pass < kPasses; ++pass) {
    values[pass] = *reinterpret_cast<const u32x4*>(&lds[tileRow<Width, Rows>(pass, tid)][col]);
  }
  if (rows == Rows) {
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
      const int row = tileRow<Width, Rows>(pass, tid);
      const size_t at = bounds.index(static_cast<size_t>(token0 + row) * Width + col, 8, limit);
      *reinterpret_cast<u32x4*>(global + at) = values[pass];
    }
  } else {
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
      const int row = tileRow<Width, Rows>(pass, tid);
      if (row < rows) {
        const size_t at = bounds.index(static_cast<size_t>(token0 + row) * Width + col, 8, limit);
        *reinterpret_cast<u32x4*>(global + at) = values[pass];
      }
    }
  }
}

// LDS の行 [first, rows) を、グローバルメモリの行 row0 + 行 へ 16 バイトずつ書く（前の行を書かない storeTile。first は
// ワークグループで一様）。
template <int Width, int Rows, int Stride>
__device__ __forceinline__ void storeTileRows(uint16_t* __restrict__ global, size_t limit,
                                              const uint16_t (&lds)[Rows][Stride], int row0, int first, int rows, int tid,
                                              Bounds& bounds) {
  if (first == 0) {
    storeTile<Width>(global, limit, lds, row0, rows, tid, bounds);
    return;
  }
  constexpr int kVectors = Width / 8;
  constexpr int kPasses = Rows * kVectors / kThreads;
  const int col = (tid % kVectors) * 8;
  u32x4 values[kPasses];
#pragma unroll
  for (int pass = 0; pass < kPasses; ++pass) {
    values[pass] = *reinterpret_cast<const u32x4*>(&lds[tileRow<Width, Rows>(pass, tid)][col]);
  }
#pragma unroll
  for (int pass = 0; pass < kPasses; ++pass) {
    const int row = tileRow<Width, Rows>(pass, tid);
    if (row >= first && row < rows) {
      const size_t at = bounds.index(static_cast<size_t>(row0 + row) * Width + col, 8, limit);
      *reinterpret_cast<u32x4*>(global + at) = values[pass];
    }
  }
}

// 局・席・牌のタイルのトークン [token0, token0 + rows) の LDS の行（Width 要素）を、行ごとに局・席・牌の順に並べた [R, F, Width] へ
// 16 バイトずつ書く。pass ごとにウェーブが 1 トークンを受け持つので、行と位置（TokenLayout::locate の 2 つの割り算）は readfirstlane で
// ウェーブに 1 回、SALU で求める。
template <int Width, int Rows, int Stride>
__device__ __forceinline__ void storeRowMajorFixed(uint16_t* __restrict__ global, size_t limit,
                                                   const uint16_t (&lds)[Rows][Stride], const TokenLayout& layout,
                                                   int token0, int rows, int tid, Bounds& bounds) {
  constexpr int kVectors = Width / 8;
  static_assert(kVectors == kWave, "the store gives each wave one token per pass");
  const int fixed = layout.fixed();
#pragma unroll
  for (int pass = 0; pass < Rows * kVectors / kThreads; ++pass) {
    const int index = pass * kThreads + tid;
    const int row = __builtin_amdgcn_readfirstlane(index / kVectors);
    const int col = (index % kVectors) * 8;
    if (row < rows) {
      int owner;
      int position;
      layout.locate(token0 + row, owner, position);
      const size_t at = bounds.index((static_cast<size_t>(owner) * fixed + position) * Width + col, 8, limit);
      *reinterpret_cast<u32x4*>(global + at) = *reinterpret_cast<const u32x4*>(&lds[row][col]);
    }
  }
}

// 重み [rows, stride] の行 row の、K の向きの 16 要素（WMMA の B のフラグメント）。
__device__ __forceinline__ bf16x16 weightFragment(const uint16_t* __restrict__ weight, size_t limit, int stride, int row,
                                                  int k0, Bounds& bounds) {
  return fragment(weight + bounds.index(static_cast<size_t>(row) * stride + k0, 16, limit));
}

__device__ __forceinline__ float load(const float* __restrict__ values, size_t limit, int index, Bounds& bounds) {
  return values[bounds.index(index, 1, limit)];
}

// RMSNorm の 1/rms（4 スレッドで 1 行、統計は FP32）。スレッドは行の H/4 列を part に持つ。和の順は (c, w, lo, hi) と、隣の 2 スレッド
// との xor の木で、行の中で固定。K3・K5・K6 の全ての RMSNorm がこの和を使う。
template <int H>
__device__ __forceinline__ float inverseRms(const u32x4 (&part)[H / 32], float epsilon) {
  float sum = 0.0f;
#pragma unroll
  for (int c = 0; c < H / 32; ++c) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float lo = bf16Float(static_cast<uint16_t>(part[c][w] & 0xffffu));
      const float hi = bf16Float(static_cast<uint16_t>(part[c][w] >> 16));
      sum += lo * lo;
      sum += hi * hi;
    }
  }
  sum += __shfl_xor(sum, 1);
  sum += __shfl_xor(sum, 2);
  return rsqrtf(sum / static_cast<float>(H) + epsilon);
}

// RMSNorm の 8 列 [col, col + 8): ⌊(x·r)·g⌉（ゲインは FP32）。
__device__ __forceinline__ u32x4 normalize8(u32x4 packed, const float* __restrict__ gain, size_t gainLimit, int col, float r,
                                            Bounds& bounds) {
  const size_t at = bounds.index(col, 8, gainLimit);
  const f32x4 g0 = *reinterpret_cast<const f32x4*>(gain + at);
  const f32x4 g1 = *reinterpret_cast<const f32x4*>(gain + at + 4);
  u32x4 normalized;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const float gLo = w < 2 ? g0[2 * w] : g1[2 * w - 4];
    const float gHi = w < 2 ? g0[2 * w + 1] : g1[2 * w - 3];
    // 2 つの積を求めてから丸める（積ごとに丸めると、コンパイラが出す命令の並びが変わる）。
    const float lo = (bf16Float(static_cast<uint16_t>(packed[w] & 0xffffu)) * r) * gLo;
    const float hi = (bf16Float(static_cast<uint16_t>(packed[w] >> 16)) * r) * gHi;
    normalized[w] = bf16RoundActivationPair(lo, hi);
  }
  return normalized;
}

// RMSNorm を LDS のタイルにその場でかける。4 スレッドで 1 行、統計とゲインは FP32、出力は BF16。タイルの行がワークグループの 1/4 より
// 少ないとき、行を持たないウェーブ（48 行ならウェーブ 6・7）は何もしない（ウェーブで一様）。
template <int H, int Rows>
__device__ __forceinline__ void normalizeTile(uint16_t (&stream)[Rows][H + kPad], const float* __restrict__ gain,
                                              size_t gainLimit, float epsilon, int tid, Bounds& bounds) {
  const int row = tid >> 2;
  if (4 * Rows < kThreads && row >= Rows) {
    return;
  }
  const int col0 = (tid & 3) * (H / 4);
  u32x4 part[H / 32];
#pragma unroll
  for (int c = 0; c < H / 32; ++c) {
    part[c] = *reinterpret_cast<const u32x4*>(&stream[row][col0 + 8 * c]);
  }
  const float r = inverseRms<H>(part, epsilon);
#pragma unroll
  for (int c = 0; c < H / 32; ++c) {
    *reinterpret_cast<u32x4*>(&stream[row][col0 + 8 * c]) = normalize8(part[c], gain, gainLimit, col0 + 8 * c, r, bounds);
  }
}

// normalizeTile と同じ受け持ち・和の順・丸めで計算し、結果は LDS に戻さずに、タイルの中の行だけをグローバルメモリの [トークン, H] へ
// 直に書く。stream は読むだけなので、直前の storeTile との間にバリアは要らない。
template <int H, int Rows>
__device__ __forceinline__ void normalizeStore(const uint16_t (&stream)[Rows][H + kPad], const float* __restrict__ gain,
                                               size_t gainLimit, float epsilon, uint16_t* __restrict__ out, size_t outLimit,
                                               int token0, int rows, int tid, Bounds& bounds) {
  const int row = tid >> 2;
  if (4 * Rows < kThreads && row >= Rows) {
    return;
  }
  const int col0 = (tid & 3) * (H / 4);
  u32x4 part[H / 32];
#pragma unroll
  for (int c = 0; c < H / 32; ++c) {
    part[c] = *reinterpret_cast<const u32x4*>(&stream[row][col0 + 8 * c]);
  }
  const float r = inverseRms<H>(part, epsilon);
  if (row >= rows) {
    return;
  }
#pragma unroll
  for (int c = 0; c < H / 32; ++c) {
    const size_t at = bounds.index(static_cast<size_t>(token0 + row) * H + col0 + 8 * c, 8, outLimit);
    *reinterpret_cast<u32x4*>(out + at) = normalize8(part[c], gain, gainLimit, col0 + 8 * c, r, bounds);
  }
}

// acc[m][j] += A[m·16 + 行, 0:K) · W[out[j], k0:k0+K)ᵀ。A は LDS の [Rows][Stride]、W はグローバルメモリにある行の順の [行, ldw]
// （K6 の tileEntry・query・discardHidden）。ウェーブは列のフラグメント out[j]（lane の列を含む W の行）を Rows 行ぶん持つ。
// K のループは全て展開し、ソフトウェアパイプラインにする。重みのフラグメント（B）は、次のステップの分をそのステップの積より前に出す。
// LDS のフラグメント（A）は、行のフラグメントごとに、使った積の直後に次のステップの分を読む（使うまでに残りの行のフラグメントの積が入る）。
// sched_barrier で区切らないと、コンパイラが読み込みを使う直前へ寄せ、ステップごとに s_waitcnt vmcnt(0) lgkmcnt(0) で止まる。
template <int kColumns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmTile(f32x8 (&acc)[Rows / 16][kColumns], const uint16_t (&a)[Rows][Stride],
                                         const uint16_t* __restrict__ weight, size_t weightLimit, int ldw, int k0,
                                         const int (&out)[kColumns], int column, Bounds& bounds) {
  constexpr int kRowTiles = Rows / 16;
  constexpr int kSteps = K / 16;
  bf16x16 b[2][kColumns];
  bf16x16 fragments[kRowTiles];
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    b[0][j] = weightFragment(weight, weightLimit, ldw, out[j], k0, bounds);
  }
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
    fragments[m] = fragment(&a[m * 16 + column][0]);
  }
#pragma unroll
  for (int k = 0; k < kSteps; ++k) {
    if (k + 1 < kSteps) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        b[(k + 1) % 2][j] = weightFragment(weight, weightLimit, ldw, out[j], k0 + (k + 1) * 16, bounds);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = mma(fragments[m], b[k % 2][j], acc[m][j]);
      }
      if (k + 1 < kSteps) {
        fragments[m] = fragment(&a[m * 16 + column][(k + 1) * 16]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
  }
}

// MLP の重みをフラグメントの順に並べた写し（計画が公開のときに、k3_mlp.hip の mlpFragmentKernel で 1 回作る）。行列 W [N, K] の要素
// (n, k) を [列のタイル n/16][ステップ k/16][半分 (k/8) % 2][行 n % 16][k % 8] に置く。フラグメント（列のタイル c、ステップ s）は
// (c·K/16 + s)·512 バイトから続く 512 バイト。lane は自分の行 n % 16 = column の、K の下位 8 要素を voffset column·16 とウェーブで同じ
// soffset で読み、上位 8 要素は soffset + 256 で読む。b128 の 1 回の読み込みが連続の 256 バイトに触れる（行の順の W では 16 行に散った
// 16 バイトずつになる）。値は W と同じなので、同じフラグメントを同じ順で積めば結果はビットで同じ。
constexpr int kFragmentElements = 256;                  // フラグメント 1 つの BF16 の数（16 行 × K の 16 要素）
constexpr int kFragmentBytes = 2 * kFragmentElements;   // フラグメント 1 つのバイト数
constexpr int kFragmentHalfBytes = kFragmentBytes / 2;  // フラグメントの中で K の上位 8 要素（16 行ぶん）が始まるバイト

struct FragmentWeights {
  __amdgpu_buffer_rsrc_t resource;
  uint32_t lane;  // voffset = column·16
  int steps;      // K/16
  size_t limit;   // 写しの BF16 の数（境界検査つきのビルドだけが読む）
};

// 写し fragments（W は [N, k]）を lane column が読むための記述を作る。
__device__ __forceinline__ FragmentWeights fragmentWeights(const Span<const uint16_t>& fragments, int k, int column) {
  return {bufferResource(fragments.data), static_cast<uint32_t>(column) * 16u, k / 16, fragments.limit};
}

// 列のタイル tile のステップ s0 から kSteps ステップ分のフラグメントの先頭のバイト（tile・s0 がウェーブで同じなら soffset に置ける）。
// 境界検査つきのビルドは、kSteps ステップの範囲を列のタイルごとにここで 1 回だけ照合する（読むステップはこの範囲の中に限る）。
template <int kSteps>
__device__ __forceinline__ int fragmentBase(const FragmentWeights& w, int tile, int s0, Bounds& bounds) {
  const size_t first = (static_cast<size_t>(tile) * w.steps + s0) * kFragmentElements;
  return static_cast<int>(sizeof(uint16_t) * bounds.index(first, kSteps * kFragmentElements, w.limit));
}

// soffset at（バイト）にあるフラグメント（WMMA の B）。
__device__ __forceinline__ bf16x16 fragmentAt(const FragmentWeights& w, int at) {
  const u32x4 lo = __builtin_amdgcn_raw_buffer_load_b128(w.resource, w.lane, at, 0);
  const u32x4 hi = __builtin_amdgcn_raw_buffer_load_b128(w.resource, w.lane, at + kFragmentHalfBytes, 0);
  return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

// 列のフラグメント j の最初のステップの B（gemmFragmentsFrom の first）。LDS に依らないので、呼ぶ側がバリアや epilogue より前に出せる。
template <int kColumns>
__device__ __forceinline__ void firstFragmentsAt(bf16x16 (&first)[kColumns], const FragmentWeights& w,
                                                 const int (&base)[kColumns]) {
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    first[j] = fragmentAt(w, base[j]);
  }
}

// acc[m][j] += A[m·16 + 行, 0:K) · W_j[ステップ 0:K/16)ᵀ。W_j は写しの列のタイルで、base[j] はその最初のステップのバイト（fragmentBase）。
// 最初のステップの B は first（firstFragmentsAt で先に出したもの）。K のループは全て展開し、B は kFragmentAhead = 2 ステップ先に出す
// （3 組のリング。写しのアドレスは SGPR に置くので、空いたレジスタで 2 ステップ先まで持てる。up 射影では、読み込みから使うまでに
// 1 ステップの積が 3 つではなく 6 つ入る）。A は gemmTile と同じく使った積の直後に次のステップの分を読み、sched_barrier で区切る。
// 積の被演算子と K の順は gemmTile と同じなので、結果はビットで同じ。
template <int kColumns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmFragmentsFrom(f32x8 (&acc)[Rows / 16][kColumns], const uint16_t (&a)[Rows][Stride],
                                                  const FragmentWeights& w, const int (&base)[kColumns], int column,
                                                  const bf16x16 (&first)[kColumns]) {
  constexpr int kRowTiles = Rows / 16;
  constexpr int kSteps = K / 16;
  constexpr int kFragmentAhead = 2;
  constexpr int kRing = kFragmentAhead + 1;
  bf16x16 b[kRing][kColumns];
  bf16x16 fragments[kRowTiles];
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    b[0][j] = first[j];
  }
#pragma unroll
  for (int s = 1; s < kFragmentAhead && s < kSteps; ++s) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      b[s][j] = fragmentAt(w, base[j] + s * kFragmentBytes);
    }
  }
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
    fragments[m] = fragment(&a[m * 16 + column][0]);
  }
#pragma unroll
  for (int k = 0; k < kSteps; ++k) {
    if (k + kFragmentAhead < kSteps) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        b[(k + kFragmentAhead) % kRing][j] = fragmentAt(w, base[j] + (k + kFragmentAhead) * kFragmentBytes);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = mma(fragments[m], b[k % kRing][j], acc[m][j]);
      }
      if (k + 1 < kSteps) {
        fragments[m] = fragment(&a[m * 16 + column][(k + 1) * 16]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
  }
}

// gemmFragmentsFrom と同じ積を、最初のステップの B もその場で読んで行う。
template <int kColumns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmFragments(f32x8 (&acc)[Rows / 16][kColumns], const uint16_t (&a)[Rows][Stride],
                                              const FragmentWeights& w, const int (&base)[kColumns], int column) {
  bf16x16 first[kColumns];
  firstFragmentsAt(first, w, base);
  gemmFragmentsFrom<kColumns, K>(acc, a, w, base, column, first);
}

// gemmFragments を 2 ステップずつのループ（#pragma unroll 1）にしたもの。タイルに 1 度の長い行列積（K5 のキーとバリューの射影）で使い、
// 全てのステップを展開した一続きの命令をループに替えて、命令を小さくする。B は次のステップの分をそのステップの積より前に出し（偶と奇の
// 2 組）、A は使った直後に読む（最後の組の先読みは最後のステップに寄せて読み、使わない）。各 acc の積の被演算子と K の順は
// gemmFragments と同じなので、結果はビットで同じ。
template <int kColumns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmFragmentsLoop(f32x8 (&acc)[Rows / 16][kColumns], const uint16_t (&a)[Rows][Stride],
                                                  const FragmentWeights& w, const int (&base)[kColumns], int column) {
  constexpr int kRowTiles = Rows / 16;
  constexpr int kSteps = K / 16;
  static_assert(kSteps % 2 == 0, "the loop covers the steps in pairs");
  bf16x16 even[kColumns];
  bf16x16 odd[kColumns];
  bf16x16 fragments[kRowTiles];
  firstFragmentsAt(even, w, base);
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
    fragments[m] = fragment(&a[m * 16 + column][0]);
  }
#pragma unroll 1
  for (int k = 0; k < kSteps; k += 2) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      odd[j] = fragmentAt(w, base[j] + (k + 1) * kFragmentBytes);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = mma(fragments[m], even[j], acc[m][j]);
      }
      fragments[m] = fragment(&a[m * 16 + column][(k + 1) * 16]);
      __builtin_amdgcn_sched_barrier(0);
    }
    const int next = min(k + 2, kSteps - 1);
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      even[j] = fragmentAt(w, base[j] + next * kFragmentBytes);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = mma(fragments[m], odd[j], acc[m][j]);
      }
      fragments[m] = fragment(&a[m * 16 + column][next * 16]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
}

// タイルの行 t のキーとバリュー [K V] = ⌊t·W_kvᵀ + b_kv⌉ を t の上に書く（K3 のイベント、K5 の Cross-Attention、K8 の 2 段目のキーと
// バリュー）。前: stream に t（同期済み）。後: stream に [K V]（同期済み）。2A = H なので、ウェーブは O 射影と同じ列のフラグメント
// wave·kColumns + j を Rows 行ぶん持つ。タイルに 1 度の長い行列積なので、2 ステップずつのループの gemmFragmentsLoop を使い、バイアスを
// 足して 1 回丸める。重みはフラグメントの順に並べた写し（W_kv [2A, H]）。
template <int H, int Rows>
__device__ __forceinline__ void projectKeyValue(uint16_t (&stream)[Rows][H + kPad], const Span<const uint16_t>& weight,
                                                const Span<const float>& bias, Bounds& bounds) {
  constexpr int kColumns = H / 16 / kWaves;
  constexpr int kRowTiles = Rows / 16;
  static_assert(H % (16 * kWaves) == 0, "unsupported hidden width");
  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);  // ウェーブで同じ（写しの soffset を SGPR に置く）
  const int half = lane >> 4;
  const int column = lane & 15;
  f32x8 acc[kRowTiles][kColumns];
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      acc[m][j] = f32x8{};
    }
  }
  const FragmentWeights keyValue = fragmentWeights(weight, H, column);
  int output[kColumns];
  int base[kColumns];
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    output[j] = (wave * kColumns + j) * 16 + column;
    base[j] = fragmentBase<H / 16>(keyValue, wave * kColumns + j, 0, bounds);
  }
  gemmFragmentsLoop<kColumns, H>(acc, stream, keyValue, base, column);
  barrier();
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    const int col = output[j];
    const float b = load(bias.data, bias.limit, col, bounds);
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        stream[m * 16 + 2 * v + half][col] = bf16RoundActivation(acc[m][j][v] + b);
      }
    }
  }
  barrier();
}

// K3 の段（上の式）。前: stream に x、inner に o（同期済み）。後: stream に新しい x（同期済み）、inner は壊れる。重みの行列は
// フラグメントの順に並べた写し（MlpWeights）。ワークグループのバリアは 2·F/kChunk + 3 回。
template <int H, int Rows>
__device__ __forceinline__ void mlpStage(MlpShared<H, Rows>& shared, const MlpWeights& w, int ffn, float epsilon,
                                         Bounds& bounds) {
  constexpr int A = H / 2;                   // Attention の出力の幅
  constexpr int kColumns = H / 16 / kWaves;  // ウェーブごとの出力の列のフラグメント（O 射影と down 射影で同じ受け持ち）
  constexpr int kRowTiles = Rows / 16;
  static_assert(A <= kChunk, "the attention output is staged in the hidden buffer");
  static_assert(H % (16 * kWaves) == 0 && H % 32 == 0, "unsupported hidden width");

  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);  // ウェーブで同じ（写しの soffset を SGPR に置く）
  const int half = lane >> 4;     // C のフラグメントで奇数の行を持つ lane
  const int column = lane & 15;   // A・B のフラグメントの行、C のフラグメントの列
  const FragmentWeights outWeight = fragmentWeights(w.outWeight, A, column);
  const FragmentWeights upWeight = fragmentWeights(w.upWeight, H, column);
  const FragmentWeights downWeight = fragmentWeights(w.downWeight, ffn, column);

  // 1. O 射影: ウェーブ w は出力の列 [16·kColumns·w, 16·kColumns·(w + 1)) を Rows 行ぶん持つ。バイアスは積より前に読む。
  f32x8 acc[kRowTiles][kColumns];
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      acc[m][j] = f32x8{};
    }
  }
  // この lane が持つ出力の列（C のフラグメントの列）と、ウェーブの列のタイル wave·kColumns + j の写しの先頭（O 射影と down 射影で
  // 同じタイル）。
  int output[kColumns];
  float outBias[kColumns];
  int outBase[kColumns];
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    output[j] = (wave * kColumns + j) * 16 + column;
    outBias[j] = load(w.outBias.data, w.outBias.limit, output[j], bounds);
    outBase[j] = fragmentBase<A / 16>(outWeight, wave * kColumns + j, 0, bounds);
  }
  gemmFragments<kColumns, A>(acc, shared.inner, outWeight, outBase, column);

  // 2. x1 = ⌊x + ⌊o·Woᵀ + bo⌉⌉。LDS の x の上に書き、最後の残差のためにレジスタにも 2 つずつ詰めて残す。
  uint32_t keep[kRowTiles][kColumns][4];
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    const int col = output[j];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int v = 0; v < 8; v += 2) {
        uint32_t pair = 0;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int row = m * 16 + 2 * (v + e) + half;
          const float y = bf16Float(bf16RoundActivation(acc[m][j][v + e] + outBias[j]));
          const uint16_t x1 = bf16RoundActivation(bf16Float(shared.stream[row][col]) + y);
          shared.stream[row][col] = x1;
          pair |= static_cast<uint32_t>(x1) << (16 * e);
        }
        keep[m][j][v / 2] = pair;
      }
    }
  }
  barrier();

  // 3. RMSNorm: 4 スレッドで 1 行。統計とゲインは FP32、出力 n は BF16 で x1 の上に書く。和の順は行の中で固定。
  normalizeTile<H>(shared.stream, w.normGain.data, w.normGain.limit, epsilon, tid, bounds);
  barrier();

  // 4. FFN: F を kChunk ずつ処理する。up 射影ではウェーブ w が段の中の 16 列（W1 の列のタイル f0/16 + w）を受け持ち、down 射影では
  //    O 射影と同じ列（W2 の列のタイル wave·kColumns + j の、ステップ f0/16 から）を受け持つ。up 射影のバイアスと最初の B は前の段で
  //    読んだものを使う。次の段のバイアスと down 射影の最初の B は h の epilogue とバリアより前に、次の段の up 射影の最初の B は down 射影
  //    より前に出す（最後の段の「次の段」は F の最後のチャンクに寄せて読み、使わない）。
  //    inner のバリア: 段 c の h は、全てのウェーブが段 c − 1 の down 射影（inner を読む）を終えてから書く。そのバリアを段 c の up 射影と
  //    h の計算の後（h を書く直前）に置くので、速いウェーブは遅いウェーブの down 射影を待たずに次の up 射影へ進む。h を書いた後のバリアは、
  //    全てのウェーブが段 c の up 射影（stream を読む）を終えた後でもある。そのため最後の段の down 射影の後にバリアを足さずに、5 で stream
  //    を書ける（down 射影は inner だけを読む）。最初の段の前のバリアは要らない（o を読む O 射影は 2 のバリアの前に済む）が、条件付きに
  //    するとコンパイラが最初の段をループから切り出して命令が大きくなるので、常に置く。
  f32x8 down[kRowTiles][kColumns];
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      down[m][j] = f32x8{};
    }
  }
  float upBiasNext = load(w.upBias.data, w.upBias.limit, wave * 16 + column, bounds);
  float downBias[kColumns];
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    downBias[j] = load(w.downBias.data, w.downBias.limit, output[j], bounds);
  }
  int upBase[1] = {fragmentBase<H / 16>(upWeight, wave, 0, bounds)};
  bf16x16 upFirst[1];
  firstFragmentsAt(upFirst, upWeight, upBase);
  for (int f0 = 0; f0 < ffn; f0 += kChunk) {
    const float upBias = upBiasNext;
    f32x8 up[kRowTiles][1];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
      up[m][0] = f32x8{};
    }
    gemmFragmentsFrom<1, H>(up, shared.stream, upWeight, upBase, column, upFirst);
    upBiasNext = load(w.upBias.data, w.upBias.limit, min(f0 + kChunk + wave * 16 + column, ffn - 1), bounds);
    int downBase[kColumns];
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      downBase[j] = fragmentBase<kChunk / 16>(downWeight, wave * kColumns + j, f0 / 16, bounds);
    }
    bf16x16 downFirst[kColumns];
    firstFragmentsAt(downFirst, downWeight, downBase);
    uint16_t h[kRowTiles][8];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        h[m][v] = bf16RoundActivation(siluFast(bf16Float(bf16RoundActivation(up[m][0][v] + upBias))));
      }
    }
    barrier();
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        shared.inner[m * 16 + 2 * v + half][wave * 16 + column] = h[m][v];
      }
    }
    barrier();
    upBase[0] = fragmentBase<H / 16>(upWeight, min(f0 + kChunk, ffn - kChunk) / 16 + wave, 0, bounds);
    firstFragmentsAt(upFirst, upWeight, upBase);
    gemmFragmentsFrom<kColumns, kChunk>(down, shared.inner, downWeight, downBase, column, downFirst);
  }

  // 5. x = ⌊x1 + ⌊acc + b2⌉⌉ を LDS に置く。
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    const int col = output[j];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const uint16_t x1 = static_cast<uint16_t>(keep[m][j][v / 2] >> (16 * (v % 2)));
        const float y = bf16Float(bf16RoundActivation(down[m][j][v] + downBias[j]));
        shared.stream[m * 16 + 2 * v + half][col] = bf16RoundActivation(bf16Float(x1) + y);
      }
    }
  }
  barrier();
}

}  // namespace tile
}  // namespace prism
