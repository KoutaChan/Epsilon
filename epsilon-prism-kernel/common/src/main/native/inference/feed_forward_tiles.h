// MLP の段の部品（前段 FFN・最終 FFN・候補の Attention ・再読射影で共通）。タイルの行数 Rows は共有メモリの配列の型が持つ（前段 FFN・最終 FFNは feed_forward.cu の kMlpRows、候補の Attention は
// candidate_attention.cu の kTile、再読射影は reread_projection.cu の kKeyRows。どれも 48）。行列積はタイル積和の BF16 で、累積は FP32。⌊⌉ は BF16 への
// 最近接偶数の丸め（NaN を特別に扱わない活性の丸め bf16RoundActivation。inference_device.h）。
//
// mlpStage は前段 FFNの式:
//   x1 = ⌊x + ⌊o·Woᵀ + bo⌉⌉ → n = ⌊(x1·r)·g⌉（r = rsqrt(mean(x1²) + ε)、FP32）
//   → F を 128 ずつ h = ⌊silu⌊n·W1ᵀ + b1⌉⌉（共有メモリ）、acc += h·W2ᵀ（FP32）→ x = ⌊x1 + ⌊acc + b2⌉⌉
// リダクションは全て行の中で行い、K の向きの順は固定（アトミックな加算も K の分割もない）。同じ行の結果は、タイルの組み方にも
// タイルの行数にも依らずビットで同じ。
//
// 待ちを減らす工夫: 行列積は全てソフトウェアパイプラインにする。ワークグループのバリアは共有メモリだけにかける（barrier）。写し
// （stageTile・storeTile）は分岐なしで読み込みを一度に出す。h の silu は siluFast（inference_device.h）。共有メモリに依らない読み込み
// （バイアスと down 射影の最初の B）はバリアと epilogue より前に出し、次の段の up 射影の最初の B は down 射影より前に出す。RMSNorm の
// 結果を共有メモリに戻さないときは、normalizeStore がレジスタからグローバルメモリへ直に書く。
//
// mlpStage とキー・バリューの射影（projectKeyValue）の重みは、計画が公開のときに作る、フラグメントの順に並べた写し（FragmentWeights）
// を device::Buffer から読む。アドレスはウェーブで同じ変位と lane の位置だけで決まり（変位はウェーブに 1 回だけ求める）、境界の照合は列のタイルごとに
// 1 回。B は 2 ステップ先に出す（gemmFragmentsFrom）。値と K の順は行の順の重みを読む形（gemmTile）と同じなので、結果はビットで同じ。
// gemmTile は候補層内のほかの行列積（tileEntry・query・discardHidden）が使う。
#pragma once

#include "backend/packed_mlp_product.h"
#include "backend/row_major_product.h"
#include "bounded_buffer.h"
#include "inference_device.h"
#include "inference_kernels.h"

namespace prism {
namespace tile {

constexpr int kThreads = 256;  // ワークグループのスレッド数
constexpr int kWaves = kThreads / kWave;
constexpr int kChunk = kFfnChunk;  // F を区切る幅（隠れ層 h の共有メモリの列）
constexpr int kPad = 8;            // 共有メモリの行のパディング（BF16 の要素数。行の間の bank の衝突を避ける）

// 共有メモリ。stream は x → x1 → n → x2 の順に、inner は o → h の順に使い回す。Rows は 16 の倍数で、RMSNorm（4 スレッドで 1 行）の受け持ちが
// ウェーブの境界で切れる数。
template <int H, int Rows, int Chunks = 1>
struct alignas(16) MlpShared {
  static_assert(Chunks == 1 || Chunks == 2, "unsupported number of FFN chunks per tile");
  static_assert(Rows % 16 == 0 && 4 * Rows <= kThreads && 4 * Rows % kWave == 0, "unsupported tile rows");
  uint16_t stream[Rows][H + kPad];
  uint16_t inner[Rows][(H / 2 > Chunks * kChunk ? H / 2 : Chunks * kChunk) + kPad];
};

// ワークグループのバリア（共有メモリの順序だけを守る）。__syncthreads はバリアの後に、グローバルメモリのいちばん近い
// キャッシュを捨てる acquire も出すが、前段 FFN・最終 FFN・候補の Attention はカーネルの中で読む領域と書く領域が別なので要らない（両方に使うのは前段 FFNの x
// だけで、読むのは最初のバリアより前）。カーネルの中でグローバルメモリに書いてから別のウェーブが読む組を足すなら、その間は
// __syncthreads に戻すこと。
__device__ __forceinline__ void barrier() { device::sharedBarrier(); }

// 幅ごとのベクトルを、ワークグループで端数なく分けられる。
template <int Width>
constexpr bool kEvenTile = kThreads % (Width / 8) == 0;

// Rows 行 × Width 列の写しで、スレッド tid が pass 回目に受け持つ 16 バイトの行（列は (tid % kVectors)·8 で pass に依らない）。
// 「tid の行 + pass ごとの定数」と書くと、共有メモリとグローバルメモリのアドレスは 1 つの基底に定数を足すだけになる（ds の offset に
// 折り込まれる）。(pass·kThreads + tid) / kVectors の形だと、アドレスが段をまたいで同じ式としてまとめられ（前段 FFNの x の写しと書き出し、
// 候補層の入力と出力）、pass ごとの値を長くレジスタに抱えて spill する。
template <int Width, int Rows>
__device__ __forceinline__ int tileRow(int pass, int tid) {
  constexpr int kVectors = Width / 8;
  static_assert(kThreads % kVectors == 0 && Rows * kVectors % kThreads == 0,
                "the tile must split evenly over the work group");
  return tid / kVectors + pass * (kThreads / kVectors);
}

// Rows 行 × Width 列の BF16 を、グローバルメモリから共有メモリへ 16 バイトずつ写す。タイルの外の行は 0 にする。読み込みは行をタイルの
// 最後の行に寄せて（rows ≥ 1）分岐をなくし、kPasses 個を一度に出してから共有メモリに書く。
template <int Width, int Rows, int Stride>
__device__ __forceinline__ void stageTile(uint16_t (&lds)[Rows][Stride], const uint16_t* __restrict__ global,
                                          size_t limit, int token0, int rows, int tid, Bounds& bounds) {
  if constexpr (!kEvenTile<Width>) {
    for (int at = tid; at < Rows * (Width / 8); at += kThreads) {
      const int row = at / (Width / 8);
      const int c = at % (Width / 8) * 8;
      const u32x4 zero{};
      *reinterpret_cast<u32x4*>(&lds[row][c]) =
          row < rows ? *reinterpret_cast<const u32x4*>(
                           global + bounds.index(static_cast<size_t>(token0 + row) * Width + c, 8, limit))
                     : zero;
    }
  } else {
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
}

// 共有メモリの Rows 行 × Width 列を、グローバルメモリへ 16 バイトずつ書く。タイルの外の行は書かない。共有メモリは先に全て読み、行が埋まった
// タイル（ワークグループで一様）は分岐なしで書く。
template <int Width, int Rows, int Stride>
__device__ __forceinline__ void storeTile(uint16_t* __restrict__ global, size_t limit,
                                          const uint16_t (&lds)[Rows][Stride], int token0, int rows, int tid,
                                          Bounds& bounds) {
  if constexpr (!kEvenTile<Width>) {
    for (int at = tid; at < rows * (Width / 8); at += kThreads) {
      const int row = at / (Width / 8);
      const int c = at % (Width / 8) * 8;
      const size_t to = bounds.index(static_cast<size_t>(token0 + row) * Width + c, 8, limit);
      *reinterpret_cast<u32x4*>(global + to) = *reinterpret_cast<const u32x4*>(&lds[row][c]);
    }
  } else {
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
}

// 共有メモリの行 [first, rows) を、グローバルメモリの行 row0 + 行 へ 16 バイトずつ書く（前の行を書かない storeTile。first は
// ワークグループで一様）。
template <int Width, int Rows, int Stride>
__device__ __forceinline__ void storeTileRows(uint16_t* __restrict__ global, size_t limit,
                                              const uint16_t (&lds)[Rows][Stride], int row0, int first, int rows,
                                              int tid, Bounds& bounds) {
  if constexpr (!kEvenTile<Width>) {
    for (int at = first * (Width / 8) + tid; at < rows * (Width / 8); at += kThreads) {
      const int row = at / (Width / 8);
      const int c = at % (Width / 8) * 8;
      const size_t to = bounds.index(static_cast<size_t>(row0 + row) * Width + c, 8, limit);
      *reinterpret_cast<u32x4*>(global + to) = *reinterpret_cast<const u32x4*>(&lds[row][c]);
    }
  } else {
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
}

// 局・席・牌のタイルのトークン [token0, token0 + rows) の共有メモリの行（Width 要素）を、行ごとに局・席・牌の順に並べた [R, F, Width] へ
// 16 バイトずつ書く。pass ごとにウェーブが 1 トークンを受け持つので、行と位置（TokenLayout::locate の 2 つの割り算）は device::firstLane で
// ウェーブで一様な値にして、ウェーブに 1 回だけ求める。
template <int Width, int Rows, int Stride>
__device__ __forceinline__ void storeRowMajorFixed(uint16_t* __restrict__ global, size_t limit,
                                                   const uint16_t (&lds)[Rows][Stride], const TokenLayout& layout,
                                                   int token0, int rows, int tid, Bounds& bounds) {
  if constexpr (!kEvenTile<Width>) {
    for (int at = tid; at < rows * (Width / 8); at += kThreads) {
      const int row = at / (Width / 8);
      const int c = at % (Width / 8) * 8;
      int owner, position;
      layout.locate(token0 + row, owner, position);
      const size_t to = bounds.index((static_cast<size_t>(owner) * layout.fixed() + position) * Width + c, 8, limit);
      *reinterpret_cast<u32x4*>(global + to) = *reinterpret_cast<const u32x4*>(&lds[row][c]);
    }
  } else {
    constexpr int kVectors = Width / 8;
    static_assert(kVectors == kWave, "the store gives each wave one token per pass");
    const int fixed = layout.fixed();
#pragma unroll
    for (int pass = 0; pass < Rows * kVectors / kThreads; ++pass) {
      const int index = pass * kThreads + tid;
      const int row = device::firstLane(index / kVectors);
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
}

// 重み [rows, stride] の行 row の、K の向きの 16 要素（タイル積和の B のフラグメント）。
__device__ __forceinline__ bf16x16 weightFragment(const uint16_t* __restrict__ weight, size_t limit, int stride,
                                                  int row, int k0, Bounds& bounds) {
  return fragment(weight + bounds.index(static_cast<size_t>(row) * stride + k0, 16, limit));
}

__device__ __forceinline__ float load(const float* __restrict__ values, size_t limit, int index, Bounds& bounds) {
  return values[bounds.index(index, 1, limit)];
}

// 二乗和 sum の平均に ε を足す。H が 2 の冪なら 1/H は正確な値なので、sum·(1/H) + ε を 1 回で丸める。そうでなければ割り算と和を
// それぞれ丸める。
template <int H>
__device__ __forceinline__ float meanPlusEpsilon(float sum, float epsilon) {
  if constexpr ((H & (H - 1)) == 0) {
    return fusedMultiplyAdd(sum, 1.0f / static_cast<float>(H), epsilon);
  } else {
    return roundedAdd(sum / static_cast<float>(H), epsilon);
  }
}

// RMSNorm の 1/rms（4 スレッドで 1 行、統計は FP32）。スレッドは行の H/4 列を part に持つ。二乗は (c, w, lo, hi) の順に積和で足し
// （和は −0 から始めるので、最初の二乗はそのまま丸めた値になる）、隣の 2 スレッドとの xor の木で合わせる。順は行の中で固定。
// 前段 FFN・最終 FFN・候補の Attention の全ての RMSNorm がこの和を使う。
template <int H>
__device__ __forceinline__ float inverseRms(const u32x4 (&part)[H / 32], float epsilon) {
  float sum = -0.0f;
#pragma unroll
  for (int c = 0; c < H / 32; ++c) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float lo = bf16Float(static_cast<uint16_t>(part[c][w] & 0xffffu));
      const float hi = bf16Float(static_cast<uint16_t>(part[c][w] >> 16));
      sum = fusedMultiplyAdd(lo, lo, sum);
      sum = fusedMultiplyAdd(hi, hi, sum);
    }
  }
  sum += device::shuffleXor(sum, 1);
  sum += device::shuffleXor(sum, 2);
  return rsqrtf(meanPlusEpsilon<H>(sum, epsilon));
}

// RMSNorm の 8 列 [col, col + 8): ⌊(x·r)·g⌉（ゲインは FP32）。
__device__ __forceinline__ u32x4 normalize8(u32x4 packed, const float* __restrict__ gain, size_t gainLimit, int col,
                                            float r, Bounds& bounds) {
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

// RMSNorm を共有メモリのタイルにその場でかける。4 スレッドで 1 行、統計とゲインは FP32、出力は BF16。タイルの行がワークグループの 1/4 より
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
    *reinterpret_cast<u32x4*>(&stream[row][col0 + 8 * c]) =
        normalize8(part[c], gain, gainLimit, col0 + 8 * c, r, bounds);
  }
}

// normalizeTile と同じ受け持ち・和の順・丸めで計算し、結果は共有メモリに戻さずに、タイルの中の行だけをグローバルメモリの [トークン, H] へ
// 直に書く。stream は読むだけなので、直前の storeTile との間にバリアは要らない。
template <int H, int Rows>
__device__ __forceinline__ void normalizeStore(const uint16_t (&stream)[Rows][H + kPad], const float* __restrict__ gain,
                                               size_t gainLimit, float epsilon, uint16_t* __restrict__ out,
                                               size_t outLimit, int token0, int rows, int tid, Bounds& bounds) {
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

// acc[m][j] += A[m·16 + 行, 0:K) · W[out[j], k0:k0+K)ᵀ。A は共有メモリの [Rows][Stride]、W はグローバルメモリにある行の順の [行, ldw]
// （候補の Attention の tileEntry・query・discardHidden）。ウェーブは列のフラグメント out[j]（lane の列を含む W の行）を Rows 行ぶん持つ。
// K のループは全て展開し、ソフトウェアパイプラインにする。重みのフラグメント（B）は、次のステップの分をそのステップの積より前に出す。
// 共有メモリのフラグメント（A）は、行のフラグメントごとに、使った積の直後に次のステップの分を読む（使うまでに残りの行のフラグメントの積が入る）。
// device::scheduleLoads で区切らないと、コンパイラーが読み込みを使う直前へ寄せ、ステップごとに全ての読み込みの完了を待って止まる。
template <int kColumns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmTile(f32x8 (&acc)[Rows / 16][kColumns], const uint16_t (&a)[Rows][Stride],
                                         const uint16_t* __restrict__ weight, size_t weightLimit, int ldw, int k0,
                                         const int (&out)[kColumns], int column, Bounds& bounds) {
#ifdef PRISM_LOGICAL_ROW_MAJOR_FRAGMENTS
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
    device::scheduleLoads();
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = mma(fragments[m], b[k % 2][j], acc[m][j]);
      }
      if (k + 1 < kSteps) {
        fragments[m] = fragment(&a[m * 16 + column][(k + 1) * 16]);
      }
      device::scheduleLoads();
    }
  }
#else
  row_major_product::multiply<kColumns, K>(acc, a, weight, weightLimit, ldw, k0, out, bounds);
#endif
}

// MLP の重みをフラグメントの順に並べた写し（計画が公開のときに、feed_forward.cu の mlpFragmentKernel で 1 回作る）。行列 W [N, K] の要素
// (n, k) を [列のタイル n/16][ステップ k/16][半分 (k/8) % 2][行 n % 16][k % 8] に置く。フラグメント（列のタイル c、ステップ s）は
// (c·K/16 + s)·512 バイトから続く 512 バイト。lane は自分の行 n % 16 = column の、K の下位 8 要素を lane の位置 column·16 とウェーブで同じ
// 変位で読み、上位 8 要素は変位 + 256 で読む。ウェーブの 16 バイトずつの 1 回の読み込みが連続の 256 バイトに触れる（行の順の W では 16 行に散った
// 16 バイトずつになる）。値は W と同じなので、同じフラグメントを同じ順で積めば結果はビットで同じ。
constexpr int kFragmentElements = 256;                  // フラグメント 1 つの BF16 の数（16 行 × K の 16 要素）
constexpr int kFragmentBytes = 2 * kFragmentElements;   // フラグメント 1 つのバイト数
constexpr int kFragmentHalfBytes = kFragmentBytes / 2;  // フラグメントの中で K の上位 8 要素（16 行ぶん）が始まるバイト

struct FragmentWeights {
  device::Buffer resource;
  uint32_t lane;  // lane の位置 = column·16
  int steps;      // K/16
  size_t limit;   // 写しの BF16 の数（境界検査つきのビルドだけが読む）
};

// 写し fragments（W は [N, k]）を lane column が読むための記述を作る。
__device__ __forceinline__ FragmentWeights fragmentWeights(const Span<const uint16_t>& fragments, int k, int column) {
  return {device::boundedBuffer(fragments.data, fragments.limit), static_cast<uint32_t>(column) * 16u, k / 16,
          fragments.limit};
}

// 列のタイル tile のステップ s0 から kSteps ステップ分のフラグメントの先頭のバイト（tile・s0 がウェーブで同じなら、ウェーブで同じ変位に置ける）。
// 境界検査つきのビルドは、kSteps ステップの範囲を列のタイルごとにここで 1 回だけ照合する（読むステップはこの範囲の中に限る）。
template <int kSteps>
__device__ __forceinline__ int fragmentBase(const FragmentWeights& w, int tile, int s0, Bounds& bounds) {
  const size_t first = (static_cast<size_t>(tile) * w.steps + s0) * kFragmentElements;
  return static_cast<int>(sizeof(uint16_t) * bounds.index(first, kSteps * kFragmentElements, w.limit));
}

// ウェーブで同じ変位 at（バイト）にあるフラグメント（タイル積和の B）。
__device__ __forceinline__ bf16x16 fragmentAt(const FragmentWeights& w, int at) {
  const u32x4 lo = device::loadWords(w.resource, w.lane, at);
  const u32x4 hi = device::loadWords(w.resource, w.lane, at + kFragmentHalfBytes);
  return device::joinWords(lo, hi);
}

// 列のフラグメント j の最初のステップの B（gemmFragmentsFrom の first）。共有メモリに依らないので、呼ぶ側がバリアや epilogue より前に出せる。
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
// （3 組のリング。写しのアドレスはウェーブで一様なレジスタに置くので、空いたレジスタで 2 ステップ先まで持てる。up 射影では、読み込みから使うまでに
// 1 ステップの積が 3 つではなく 6 つ入る）。A は gemmTile と同じく使った積の直後に次のステップの分を読み、device::scheduleLoads で区切る。
// 積の被演算子と K の順は gemmTile と同じなので、結果はビットで同じ。
template <int kColumns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmFragmentsFrom(f32x8 (&acc)[Rows / 16][kColumns], const uint16_t (&a)[Rows][Stride],
                                                  const FragmentWeights& w, const int (&base)[kColumns], int column,
                                                  const bf16x16 (&first)[kColumns]) {
#ifdef PRISM_LOGICAL_MLP_FRAGMENTS
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
    device::scheduleLoads();
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = mma(fragments[m], b[k % kRing][j], acc[m][j]);
      }
      if (k + 1 < kSteps) {
        fragments[m] = fragment(&a[m * 16 + column][(k + 1) * 16]);
      }
      device::scheduleLoads();
    }
  }
#else
  packed_mlp_product::multiply<false, kColumns, K>(acc, a, w.resource, base);
#endif
}

// gemmFragmentsFrom と同じ積を、最初のステップの B もその場で読んで行う。
template <int kColumns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmFragments(f32x8 (&acc)[Rows / 16][kColumns], const uint16_t (&a)[Rows][Stride],
                                              const FragmentWeights& w, const int (&base)[kColumns], int column) {
  bf16x16 first[kColumns];
  firstFragmentsAt(first, w, base);
  gemmFragmentsFrom<kColumns, K>(acc, a, w, base, column, first);
}

// gemmFragments を 2 ステップずつのループ（#pragma unroll 1）にしたもの。タイルに 1 度の長い行列積（最終 FFNのキーとバリューの射影）で使い、
// 全てのステップを展開した一続きの命令をループに替えて、命令を小さくする。B は次のステップの分をそのステップの積より前に出し（偶と奇の
// 2 組）、A は使った直後に読む（最後の組の先読みは最後のステップに寄せて読み、使わない）。各 acc の積の被演算子と K の順は
// gemmFragments と同じなので、結果はビットで同じ。
template <int kColumns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmFragmentsLoop(f32x8 (&acc)[Rows / 16][kColumns], const uint16_t (&a)[Rows][Stride],
                                                  const FragmentWeights& w, const int (&base)[kColumns], int column) {
#ifdef PRISM_LOGICAL_MLP_FRAGMENTS
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
    device::scheduleLoads();
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = mma(fragments[m], even[j], acc[m][j]);
      }
      fragments[m] = fragment(&a[m * 16 + column][(k + 1) * 16]);
      device::scheduleLoads();
    }
    const int next = min(k + 2, kSteps - 1);
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      even[j] = fragmentAt(w, base[j] + next * kFragmentBytes);
    }
    device::scheduleLoads();
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = mma(fragments[m], odd[j], acc[m][j]);
      }
      fragments[m] = fragment(&a[m * 16 + column][next * 16]);
      device::scheduleLoads();
    }
  }
#else
  packed_mlp_product::multiply<true, kColumns, K>(acc, a, w.resource, base);
#endif
}

// タイルの行 t のキーとバリュー [K V] = ⌊t·W_kvᵀ + b_kv⌉ を t の上に書く（前段 FFNのイベント、最終 FFNの Cross-Attention、再読射影の 2 段目のキーと
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
  const int wave = device::firstLane(tid / kWave);  // ウェーブで同じ（写しの変位を一様なレジスタに置く）
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

struct FfnActivate {
  __device__ __forceinline__ uint16_t operator()(float sum, float bias) const {
    return bf16RoundActivation(siluFast(bf16Float(bf16RoundActivation(sum + bias))));
  }
};

#ifdef PRISM_LOGICAL_MLP_FRAGMENTS
using FfnAccumulator = f32x8;
#else
using packed_mlp_product::FfnAccumulator;

template <int Columns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmFragments(FfnAccumulator (&acc)[Rows / 16][Columns],
                                              const uint16_t (&a)[Rows][Stride], const FragmentWeights& w,
                                              const int (&base)[Columns], int) {
  packed_mlp_product::multiplyFfn<Columns, K>(acc, a, w.resource, base);
}

template <int Columns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmFragmentsFrom(FfnAccumulator (&acc)[Rows / 16][Columns],
                                                  const uint16_t (&a)[Rows][Stride], const FragmentWeights& w,
                                                  const int (&base)[Columns], int column, const bf16x16 (&)[Columns]) {
  gemmFragments<Columns, K>(acc, a, w, base, column);
}

template <int Columns, int K, int Rows, int Stride>
__device__ __forceinline__ void gemmFragmentsLoop(FfnAccumulator (&acc)[Rows / 16][Columns],
                                                  const uint16_t (&a)[Rows][Stride], const FragmentWeights& w,
                                                  const int (&base)[Columns], int column) {
  gemmFragments<Columns, K>(acc, a, w, base, column);
}

// 上げ射影にバイアスを足し、SiLU の前後で BF16 に丸めて下げ射影の入力を共有メモリに置く。
template <int RowTiles, int Columns, int Rows, int Stride>
__device__ __forceinline__ void publishFfn(const FfnAccumulator (&up)[RowTiles][Columns], const float (&bias)[Columns],
                                           uint16_t (&inner)[Rows][Stride], int wave) {
  auto h = packed_mlp_product::activateFfn(up, bias, FfnActivate{});
  barrier();
  h.store(inner, wave);
  barrier();
}

#endif

// 上げ射影の入力を複数の列タイルで共有する。受け渡しの同期は列タイルの組ごとに 2 回。
// 下げ射影の累積器は段をまたいで保持し、元の K の順と活性化の前後の BF16 丸めを保つ。
template <int UpTiles, int H, int Rows, int Chunks, int RowTiles, int Columns>
__device__ __forceinline__ void ffnChunk(MlpShared<H, Rows, Chunks>& shared, const MlpWeights& w,
                                         const FragmentWeights& upWeight, const FragmentWeights& downWeight,
                                         FfnAccumulator (&down)[RowTiles][Columns], int first, int wave, int column,
                                         int half, Bounds& bounds) {
  constexpr int width = UpTiles * kChunk;
  int upBase[UpTiles];
  float bias[UpTiles];
#pragma unroll
  for (int j = 0; j < UpTiles; ++j) {
    const int tile = wave * UpTiles + j;
    upBase[j] = fragmentBase<H / 16>(upWeight, first / 16 + tile, 0, bounds);
    bias[j] = load(w.upBias.data, w.upBias.limit, first + tile * 16 + column, bounds);
  }
  FfnAccumulator up[RowTiles][UpTiles]{};
  gemmFragments<UpTiles, H>(up, shared.stream, upWeight, upBase, column);
  int downBase[Columns];
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    downBase[j] = fragmentBase<width / 16>(downWeight, wave * Columns + j, first / 16, bounds);
  }
  bf16x16 next[Columns];
  firstFragmentsAt(next, downWeight, downBase);
#ifdef PRISM_LOGICAL_MLP_FRAGMENTS
  uint16_t activated[RowTiles][UpTiles][8];
#pragma unroll
  for (int m = 0; m < RowTiles; ++m) {
#pragma unroll
    for (int j = 0; j < UpTiles; ++j) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        activated[m][j][v] = FfnActivate{}(up[m][j][v], bias[j]);
      }
    }
  }
  barrier();
#pragma unroll
  for (int m = 0; m < RowTiles; ++m) {
#pragma unroll
    for (int j = 0; j < UpTiles; ++j) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        shared.inner[m * 16 + 2 * v + half][(wave * UpTiles + j) * 16 + column] = activated[m][j][v];
      }
    }
  }
  barrier();
#else
  publishFfn(up, bias, shared.inner, wave);
#endif
  gemmFragmentsFrom<Columns, width>(down, shared.inner, downWeight, downBase, column, next);
}

// 前段 FFNの段（上の式）。前: stream に x、inner に o（同期済み）。後: stream に新しい x（同期済み）、inner は壊れる。重みの行列は
// フラグメントの順に並べた写し（MlpWeights）。FFN は Chunks 個のチャンクをまとめて計算する。
template <int H, int Rows, bool kCount = false, bool kShortPrefetch = false, int Chunks>
__device__ __forceinline__ void mlpStage(MlpShared<H, Rows, Chunks>& shared, const MlpWeights& w, int ffn,
                                         float epsilon, Bounds& bounds, Span<const uint16_t> count = {}, int first = 0,
                                         int realRows = 0) {
  constexpr int A = H / 2;                   // Attention の出力の幅
  constexpr int kColumns = H / 16 / kWaves;  // ウェーブごとの出力の列のフラグメント（O 射影と down 射影で同じ受け持ち）
  constexpr int kRowTiles = Rows / 16;
  static_assert(A <= sizeof(shared.inner[0]) / sizeof(uint16_t) - kPad, "the attention output fits in shared memory");
  static_assert(H % (16 * kWaves) == 0 && H % 32 == 0, "unsupported hidden width");

  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = device::firstLane(tid / kWave);  // ウェーブで同じ（写しの変位を一様なレジスタに置く）
  const int half = lane >> 4;                       // C のフラグメントで奇数の行を持つ lane
  const int column = lane & 15;                     // A・B のフラグメントの行、C のフラグメントの列
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
  if constexpr (kShortPrefetch) {
    gemmFragmentsLoop<kColumns, A>(acc, shared.inner, outWeight, outBase, column);
  } else {
    gemmFragments<kColumns, A>(acc, shared.inner, outWeight, outBase, column);
  }

  // 2. x1 = ⌊x + ⌊o·Woᵀ + bo⌉⌉。共有メモリの x の上に書き、最後の残差のためにレジスタにも 2 つずつ詰めて残す。
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
          uint16_t x1 = bf16RoundActivation(bf16Float(shared.stream[row][col]) + y);
          if constexpr (kCount) {
            if (row < realRows) {
              const size_t at = bounds.index(static_cast<size_t>(first + row) * H + col, 1, count.limit);
              x1 = bf16RoundActivation(bf16Float(x1) + bf16Float(count.data[at]));
            }
          }
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
  FfnAccumulator down[kRowTiles][kColumns];
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      down[m][j] = FfnAccumulator{};
    }
  }
  float downBias[kColumns];
  if constexpr (Chunks == 2) {
    int first = 0;
    for (; first + 2 * kChunk <= ffn; first += 2 * kChunk) {
      ffnChunk<2>(shared, w, upWeight, downWeight, down, first, wave, column, half, bounds);
    }
    // FFN の幅が奇数個のチャンクなら、末尾では実在する 1 個だけを読む。
    if (first < ffn) ffnChunk<1>(shared, w, upWeight, downWeight, down, first, wave, column, half, bounds);
  } else {
    float upBiasNext = 0;
    int upBase[1];
    bf16x16 upFirst[1];
    if constexpr (!kShortPrefetch) {
      upBiasNext = load(w.upBias.data, w.upBias.limit, wave * 16 + column, bounds);
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        downBias[j] = load(w.downBias.data, w.downBias.limit, output[j], bounds);
      }
      upBase[0] = fragmentBase<H / 16>(upWeight, wave, 0, bounds);
      firstFragmentsAt(upFirst, upWeight, upBase);
    }
    for (int f0 = 0; f0 < ffn; f0 += kChunk) {
      float upBias = upBiasNext;
      FfnAccumulator up[kRowTiles][1];
#pragma unroll
      for (int m = 0; m < kRowTiles; ++m) {
        up[m][0] = FfnAccumulator{};
      }
      if constexpr (kShortPrefetch) {
        // 先読みは射影の中に収め、上げと下げをまたいで重みのフラグメントを保持しない。
        upBase[0] = fragmentBase<H / 16>(upWeight, f0 / 16 + wave, 0, bounds);
        gemmFragmentsLoop<1, H>(up, shared.stream, upWeight, upBase, column);
        upBias = load(w.upBias.data, w.upBias.limit, f0 + wave * 16 + column, bounds);
      } else {
        gemmFragmentsFrom<1, H>(up, shared.stream, upWeight, upBase, column, upFirst);
        upBiasNext = load(w.upBias.data, w.upBias.limit, min(f0 + kChunk + wave * 16 + column, ffn - 1), bounds);
      }
      int downBase[kColumns];
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        downBase[j] = fragmentBase<kChunk / 16>(downWeight, wave * kColumns + j, f0 / 16, bounds);
      }
      bf16x16 downFirst[kColumns];
      if constexpr (!kShortPrefetch) firstFragmentsAt(downFirst, downWeight, downBase);
#ifdef PRISM_LOGICAL_MLP_FRAGMENTS
      uint16_t h[kRowTiles][8];
#pragma unroll
      for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          h[m][v] = FfnActivate{}(up[m][0][v], upBias);
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
#else
      const float bias[1]{upBias};
      publishFfn(up, bias, shared.inner, wave);
#endif
      if constexpr (kShortPrefetch) {
        gemmFragmentsLoop<kColumns, kChunk>(down, shared.inner, downWeight, downBase, column);
      } else {
        upBase[0] = fragmentBase<H / 16>(upWeight, min(f0 + kChunk, ffn - kChunk) / 16 + wave, 0, bounds);
        firstFragmentsAt(upFirst, upWeight, upBase);
        gemmFragmentsFrom<kColumns, kChunk>(down, shared.inner, downWeight, downBase, column, downFirst);
      }
    }
  }

  // 5. x = ⌊x1 + ⌊acc + b2⌉⌉ を共有メモリに置く。
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    const int col = output[j];
    const float bias =
        (kShortPrefetch || Chunks == 2) ? load(w.downBias.data, w.downBias.limit, col, bounds) : downBias[j];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#ifdef PRISM_LOGICAL_MLP_FRAGMENTS
      const f32x8 result = down[m][j];
#else
      const f32x8 result = packed_mlp_product::finishFfn(down[m][j]);
#endif
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const uint16_t x1 = static_cast<uint16_t>(keep[m][j][v / 2] >> (16 * (v % 2)));
        const float y = bf16Float(bf16RoundActivation(result[v] + bias));
        shared.stream[m * 16 + 2 * v + half][col] = bf16RoundActivation(bf16Float(x1) + y);
      }
    }
  }
  barrier();
}

}  // namespace tile
}  // namespace prism
