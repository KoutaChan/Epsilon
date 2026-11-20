// 全トークンまたは局・席・牌だけをクエリにする Attention。ワークグループは 1 つのヘッドの対と 1 行を受け持ち、ウェーブの数
// W は行のグループとヘッドの分担で決まる（行の 16 トークンのタイルの数が W 以下）。計画はグループごとに起動し、
// ワークグループは、グループの順に並べた行の番号（rowOrderKernel）から自分の行を引く。eager の EncoderLayer の Attention の半分に当たる
// （O 射影は前段 FFN・最終 FFN）。全トークンをクエリにする層 0 は段 1 の射影をせず、埋め込みが折り込んだ表から作った q・k・v（qkv0）をコピーする（loadFolded）。段 2 は同じ。
//
//   0. 2 ヘッドのバイアスの表（×log2e）を共有メモリへコピーする（重みを公開するときに relationBiasKernel が層ごとに作ったもの。
//      relation_encoding.h）
//   1. ヘッドの対の q・k・v = ⌊n·W_qkvᵀ + b⌉（タイル積和）。n = ⌊(x·r)·g⌉ は前の段（前段 FFN）が NORMALIZED に書いたもので、フラグメントとして
//      そのまま読む。W_qkv は、重みを公開するときに qkvFragmentKernel がフラグメントの順に並べ替えたコピー（下）を読む。ウェーブ m が
//      16 トークンのタイル m を作る。k・v は共有メモリへ置く。q は転置の qᵀ = W_q·nᵀ で作り、クエリの B のフラグメントのままレジスタに残す
//      （ウェーブ m がクエリのタイル m を受け持つ）。
//      最初の局・席・牌の層の固定クエリの層は全てのタイルを射影する（W ≥ 6 のグループは 1 ウェーブが 2 タイルの 1 ヘッドを作り、q を共有メモリに
//      置く）。後の局・席・牌の層の固定クエリの層は局・席・牌のタイル（⌈F/16⌉ 個）だけを射影する。イベントの k・v は、最後の全トークンの層の前段 FFN
//      がイベントの x から作った eventKeyValue を同じ置き方でコピーする（loadEventKeys）。局・席・牌とイベントが混ざるタイルは、射影の
//      後に同じウェーブがイベントの位置をコピーし直す。
//   2. 各クエリタイルの 2 ヘッドを 1 ウェーブで回す。固定クエリの層で十分なウェーブがあればヘッドごとに分ける。16 キーのタイル
//      ごとに次を計算する。
//        S^T = K·Qᵀ（タイル積和、BF16 の積を FP32 で累積）、ℓ = S·scale + Σ_s ρ[16s + c_s(i, j)]（c は関係のコード。ρ は、領域 2 つ
//        ずつ前もって足した表を引く）、オンラインの softmax（exp2）、O^T += V^T·P^T（タイル積和。P は FP32 の e を hi・lo の 2 つの BF16 に
//        分けて 2 回掛け、FP32 に近い精度を保つ）
//      最後に O / Σe を BF16 に 1 回丸める。
//
// 段 1 と段 2 の間のバリアは共有メモリの順序だけをそろえる（phaseBarrier）。q・k・v・o は活性値の丸め（bf16RoundActivation・
// pack8Activation。NaN の選択なし）で丸める。
//
// eager の packedRelationScaledDotProductAttention と式は同じで、違うのは和の順と exp だけ（相対 RMS ≤ 2e-2 の保証に収まる）。リダクション
// は全て行の中で閉じ、順は固定で、アトミックな加算もない。タイル積和の要素は互いに独立なので、同じ行の o はタイルの組み方に依らずビットで
// 同じ。W はタイルとヘッドをどのウェーブが受け持つかを決めるだけで、タイルごと・ヘッドごとの演算（オペランド・K の順・丸め・置く位置）
// はどのグループのカーネルでも同じなので、o はグループにも依らない。前段 FFNが作るイベントの k・v は、固定クエリの層が n から射影するときと同じ n・
// 同じオペランド・同じ K の順・同じ丸めで作るので、射影した値とビットで同じ。
//
// 射影の重みのコピー（重みを公開するときに qkvFragmentKernel が層ごとに 1 回作る）: ヘッドの対ごとに
// [ステップ k][ヘッド h][j = q・k・v][半分][行 r][8] = W_qkv[unit(j, h, r)][16k + 8·半分 .. +8]。
// unit(q, h, r) = 32·対 + 16h + π(r)（π(r) = (r >> 1) + 8(r & 1)）、unit(k, h, r) = A + 32·対 + 16h + r、
// unit(v, h, r) = 2A + 32·対 + 16h + r。フラグメント 1 つは 512 バイト、1 ステップは 3 KB、ヘッドの対 1 つは 48 KB。lane は行 column
// の 16 バイトを、lane ごとの位置 column·16 と、ウェーブで同じ変位 k·3072 + (3h + j)·512（上の 8 要素は + 256）で読む。このため
// 16 バイトずつの 1 回の読み込みは連続した 256 バイト（cache line 2 本）になる（W_qkv をそのまま読むと 512 バイトずつ離れた 16 行になる）。
// 読んだ値とタイル積和のフラグメントは、W_qkv をそのまま読む場合と同じ。
#include "attention_tile.h"
#include "bounded_buffer.h"
#include "cache/projection_key.h"
#include "static_shared_memory.h"
#include <utility>

#include "relation_encoding.h"
#include "kernel_tuning.h"
#include "input_dimensions.h"
#include "feed_forward_tiles.h"
#include "causal_mask.h"
#include "event_submission.h"
#include "inference_stages.h"

namespace prism {
namespace {

constexpr int kHeadWidth = 16;
constexpr int kPairWidth = 2 * kHeadWidth;                 // ヘッドの対の q・k の列の数
constexpr int kFixedBlocks = (kMaxFixedTokens + 15) / 16;  // 固定クエリの層のクエリのタイルの数の上限
constexpr int kBiasChunks = kBiasEntries / 2;              // バイアスの表の 16 バイトのチャンクの数
static_assert(kBiasEntries % 2 == 0, "the bias table is copied 16 bytes at a time");
static_assert(sizeof(float2) == kBiasEntryBytes, "a code field holds the byte offset of a two-head bias entry");

constexpr int kQkvThreads = 256;     // qkvFragmentKernel のワークグループのスレッド数
constexpr int kOrderThreads = 1024;  // rowOrderKernel のワークグループ（1 つだけ）のスレッド数
constexpr int kOrderWaves = kOrderThreads / kWave;

// 射影の重みのコピーの並び（上の説明）。
constexpr int kFragmentBytes = 512;                 // フラグメント 1 つ（K の下の 8 要素が 16 行、上の 8 要素が 16 行）
constexpr int kStepBytes = 2 * 3 * kFragmentBytes;  // 1 ステップ（2 ヘッド × q・k・v）

// クエリの q を置く共有メモリ（k と同じ入れ替えで置く）。射影で 1 ウェーブが 2 タイルの 1 ヘッドを作る固定クエリの層のグループだけが、クエリ（局・席・牌
// の 48 以下）の q を全てここに置く（q を作るウェーブとクエリのタイルを回すウェーブが違うため）。ほかのグループの q はレジスタに残す。
template <bool kPairedBlocks>
struct FixedQueries {};

template <>
struct FixedQueries<true> {
  uint16_t fixedQuery[kMaxFixedTokens][kPairWidth];
};

// 共有メモリ（kTokens = 16W トークン分）。k は [トークン][2 ヘッド × 16] の行を 16 バイトのチャンクに分け、(トークン >> 1) & 3 で入れ替えて
// 置く（8 lane の 16 バイトずつの読み込みで bank が衝突しない）。V^T は [16h + r(d)][κ] で、r(d) = 2(d & 7) + (d >> 3)。キー 16m + 2v + half
// は列 16m + v + 8·half に置く。この並びにすると、O^T のフラグメントの lane が連続した 8 次元を持ち、P^T の B のフラグメントは lane と
// 相方の 8 値をつなげるだけで作れる。V^T の行のストライド kTokens + 8 要素（8W + 4 ワード）は 8 ワードの倍数から 4 ワードずれるので、
// 16 lane が 16 バイトずつ読むとき bank が衝突しない。
template <int kTokens, bool kPairedBlocks>
struct alignas(16) AttentionShared : FixedQueries<kPairedBlocks> {
  static constexpr int kValueStride = kTokens + 8;
  uint16_t k[kTokens][kPairWidth];
  uint16_t valueT[kPairWidth][kValueStride];
  float2 bias[kBiasEntries];  // 2 ヘッドのバイアスの表（×log2e）
};

// q・k の共有メモリで、行 token の列 column（< 32）の要素を置く位置（行の中で入れ替えた後の列）。
__device__ __forceinline__ int swizzled(int token, int column) {
  return 8 * ((column >> 3) ^ ((token >> 1) & 3)) + (column & 7);
}

// q・k の行 token のうち、ヘッド head の 16 次元（タイル積和のフラグメント）。
__device__ __forceinline__ bf16x16 headFragment(const uint16_t (*m)[kPairWidth], int token, int head) {
  const int s = (token >> 1) & 3;
  const u32x4 lo = *reinterpret_cast<const u32x4*>(&m[token][8 * ((2 * head) ^ s)]);
  const u32x4 hi = *reinterpret_cast<const u32x4*>(&m[token][8 * ((2 * head + 1) ^ s)]);
  return device::joinWords(lo, hi);
}

// V^T のヘッド head の行 column（= r(d)）のうち、キーのタイル block の 16 キー（タイル積和の A のフラグメントの行）。
template <int kValueStride>
__device__ __forceinline__ const uint16_t* valueRow(const uint16_t (*valueT)[kValueStride], int head, int block,
                                                    int column) {
  return &valueT[kHeadWidth * head + column][16 * block];
}

__device__ __forceinline__ float fastExp2(float value) { return device::exponent2(value); }

// 2 つの FP32 の上位 16 bit を 1 ワードに詰める（a が下、b が上。1 命令）。
__device__ __forceinline__ uint32_t upperHalves(uint32_t a, uint32_t b) { return device::upperPair(a, b); }

// lane ごとの 8 値（16 バイト）から B のフラグメントを作る。下の 16 lane が K の 0〜7、上の 16 lane が 8〜15 を持つとき、どの lane も
// [lane & 15 の 8 値, lane | 16 の 8 値] を得る。この lane の入れ替え（device::lowerHalfWord・upperHalfWord）は、
// 共有メモリの bank を使わない。
__device__ __forceinline__ bf16x16 joinHalves(u32x4 own) {
  u32x4 first;
  u32x4 second;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    first[w] = device::lowerHalfWord(own[w]);
    second[w] = device::upperHalfWord(own[w]);
  }
  return device::joinWords(first, second);
}

// P^T の B のフラグメント（hi・lo の 2 つの BF16）を作る。lane はキー 2v + half（v < 8）の e を持つ。フラグメントの K の順は V^T の
// 列の並び κ = v + 8·half なので、下の 16 lane の 8 値が K の 0〜7、上の 16 lane の 8 値が 8〜15 になる。hi は e の上位 16 bit、lo は
// e − hi の上位 16 bit（どちらも切り捨て。hi + lo で e の仮数の 16 bit 以上が残る）。
__device__ __forceinline__ void probabilityFragments(const float (&e)[8], bf16x16& high, bf16x16& low) {
  u32x4 ownHigh;
  u32x4 ownLow;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t a = device::bits<uint32_t>(e[2 * w]);
    const uint32_t b = device::bits<uint32_t>(e[2 * w + 1]);
    ownHigh[w] = upperHalves(a, b);
    const uint32_t ra = device::bits<uint32_t>(e[2 * w] - device::bits<float>(a & 0xffff0000u));
    const uint32_t rb = device::bits<uint32_t>(e[2 * w + 1] - device::bits<float>(b & 0xffff0000u));
    ownLow[w] = upperHalves(ra, rb);
  }
  high = joinHalves(ownHigh);
  low = joinHalves(ownLow);
}

// device::Buffer から 32 バイト（タイル積和のフラグメント 1 つ）を 16 バイトずつ読む。offset は lane ごとのバイト位置。low・high は
// 下と上の 16 バイトの、ウェーブで同じ変位（上の半分の差も変位に入れて、ループの中に lane ごとの足し算を作らない）。
__device__ __forceinline__ bf16x16 bufferFragment(device::Buffer resource, uint32_t offset, int low, int high) {
  const u32x4 lo = device::loadWords(resource, offset, low);
  const u32x4 hi = device::loadWords(resource, offset, high);
  return device::joinWords(lo, hi);
}

// 射影の K 方向の 1 ステップで使うフラグメント。ヘッド h の q・k・v（j = 0・1・2）の重み（コピーの中の変位 k·3072 + (3h + j)·512）
// と、タイル b < kBlocks ごとの n（行の 32k バイト目からの 32 バイト）を持つ。重みはタイルで共有する。q を作るのは先頭の kQueries 個の
// タイルだけで（固定クエリの層では局・席・牌より後のタイルは q を作らない）、kQueries = 0 なら q の重みも読まない。
template <int kBlocks, int kQueries>
struct ProjectionStep {
  static_assert(0 <= kQueries && kQueries <= kBlocks, "the blocks with a query come first");
  bf16x16 weight[3];
  bf16x16 input[kBlocks];

  __device__ __forceinline__ void load(device::Buffer weights, uint32_t fragmentRow, int h, device::Buffer inputs,
                                       const uint32_t (&inputRow)[kBlocks], int k) {
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

  // タイル b について acc[b][0] += W_q·nᵀ（転置の q）、acc[b][1]・acc[b][2] += n·W_k,vᵀ。
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

// ヘッド h について、kBlocks 個のタイルの射影 acc（ProjectionStep::multiply の式）を K 方向の 16 ステップで作る。フラグメントの組を
// kSets 個回し、kSets − 1 ステップ先まで読み込みながら掛ける。読み込みは全て無条件なので、ループは 1 つの基本ブロックのまま保たれ
// （組をコピーする命令が出ない）、device::scheduleLoads が次の読み込みを掛け算の後ろに留める。
//   2 組（固定クエリの層で q を作る 2 タイルの場合）: 展開しない 2 ステップのループと、最後の 2 ステップ（読み直しはない）
//   3 組（全トークンの層）: 展開しない 3 ステップのループと、最後の 1 ステップ。最後の周回の読み込みの 1 つはステップ 15 を読み直す（使わない）
//   4 組（固定クエリの層で q を作らない 2 タイルの場合と、1 タイルの場合）: 3 ステップを先に読み、展開しない 4 ステップのループと、最後の
//   4 ステップ（読み直しはない）
// 組の数は、カーネルが実行ユニットあたり 8 ウェーブの占有のレジスタ（192）に収まる最多にする。1 タイルで 2 組だと 1 ステップの掛け算 3 つの間しか
// 読み込みを隠せず、展開するとフラグメントが重なってレジスタが溢れる。全トークンの層は 4 組だと占有が下がるので 3 組にする。固定クエリの層で q を作る
// 2 タイルは、acc 6 つと 1 組のフラグメント 5 つが要るので 2 組までで、1 ステップの掛け算 5〜6 つの間に読み込みを隠す。どの場合も、
// 各 acc の K の順は 0〜15 で同じ。
template <int H, int kSets, int kBlocks, int kQueries>
__device__ __forceinline__ void projectHead(f32x8 (&acc)[kBlocks][3], device::Buffer weights, uint32_t fragmentRow,
                                            int h, device::Buffer inputs, const uint32_t (&inputRow)[kBlocks]) {
#ifdef PRISM_LOGICAL_ATTENTION_FRAGMENTS
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
      device::scheduleLoads();
      a.load(weights, fragmentRow, h, inputs, inputRow, k + 3);
      b.multiply(acc);
      device::scheduleLoads();
      b.load(weights, fragmentRow, h, inputs, inputRow, min(k + 4, kSteps - 1));
      c.multiply(acc);
      device::scheduleLoads();
    }
    a.multiply(acc);
  } else if constexpr (kSets == 2) {
    static_assert(kSteps % 2 == 0, "the rolled loop covers the steps in twos");
#pragma unroll 1
    for (int k = 0; k + 2 < kSteps; k += 2) {
      b.load(weights, fragmentRow, h, inputs, inputRow, k + 1);
      a.multiply(acc);
      device::scheduleLoads();
      a.load(weights, fragmentRow, h, inputs, inputRow, k + 2);
      b.multiply(acc);
      device::scheduleLoads();
    }
    b.load(weights, fragmentRow, h, inputs, inputRow, kSteps - 1);
    a.multiply(acc);
    device::scheduleLoads();
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
      device::scheduleLoads();
      a.load(weights, fragmentRow, h, inputs, inputRow, k + 4);
      b.multiply(acc);
      device::scheduleLoads();
      b.load(weights, fragmentRow, h, inputs, inputRow, k + 5);
      c.multiply(acc);
      device::scheduleLoads();
      c.load(weights, fragmentRow, h, inputs, inputRow, k + 6);
      d.multiply(acc);
      device::scheduleLoads();
    }
    d.load(weights, fragmentRow, h, inputs, inputRow, kSteps - 1);
    a.multiply(acc);
    device::scheduleLoads();
    b.multiply(acc);
    device::scheduleLoads();
    c.multiply(acc);
    device::scheduleLoads();
    d.multiply(acc);
  }
#else
  attention_tile::project<H, kBlocks, kQueries>(acc, weights, h, inputs, inputRow);
#endif
}

// ヘッドの対 pair のバイアスの表から、16 バイトのチャンク e（項 2e と 2e + 1）を読む。
__device__ __forceinline__ f32x4 biasChunk(const AttentionArgs& args, Bounds& bounds, int pair, int e) {
  const size_t at = bounds.index((static_cast<size_t>(pair) * kBiasEntries + 2 * e) * 2, 4, args.bias.limit);
  return *reinterpret_cast<const f32x4*>(args.bias.data + at);
}

// 段 1 と段 2 の間のワークグループのバリア（共有メモリの順序だけをそろえる。tile::barrier と同じ形）。__syncthreads はバリアの後に
// acquire も出す（同じ計算ユニットの他のワークグループが射影に使っているグローバルメモリの、いちばん近いキャッシュまで捨てる）。
// 全トークンの層と固定クエリの層はカーネルの中で書いたグローバルメモリを読まない（o は書くだけで、n・重みのコピー・イベントの k・v・コード・バイアスの表は
// 読むだけ）ので、それは要らない。
__device__ __forceinline__ void phaseBarrier() { device::sharedBarrier(); }

// 関係のコード code（推論用のワード）が指す、4 つの表の項の和 Σ_s ρ（×log2e）。表の項は (ヘッド 2p, ヘッド 2p + 1) の値の対で、
// 2 ヘッド分を 1 回で引く。ワードのフィールドは項のバイト位置なので、アドレスの計算はフィールドの取り出しだけで済む（項の番号に直す
// ÷8 と配列の ×8 が打ち消し合う）。
__device__ __forceinline__ float2 pairBias(const float2 (&bias)[kBiasEntries], uint32_t code) {
  const float2 b01 = bias[pairTableOffset(code, 0) / kBiasEntryBytes];
  const float2 b23 = bias[kPairEntries + pairTableOffset(code, 1) / kBiasEntryBytes];
  const float2 b45 = bias[2 * kPairEntries + pairTableOffset(code, 2) / kBiasEntryBytes];
  const float2 b6 = bias[3 * kPairEntries + lastTableOffset(code) / kBiasEntryBytes];
  return make_float2(((b01.x + b23.x) + b45.x) + b6.x, ((b01.y + b23.y) + b45.y) + b6.y);
}

// 1 ヘッドのオンラインの softmax の状態。lane はクエリ 16qt + column に対するキー 16kt + 2v + half（v < 8）を持つ。total はその 8 キー
// の Σe（相方の lane の分は最後に足す）、out は O^T のフラグメント（lane は次元 8·half 〜 8·half + 7 を持つ）。
struct HeadState {
  float running = -INFINITY;
  float total = 0.0f;
  f32x8 out{};
};

// キーのタイル 1 つ分、1 ヘッドを進める。タイルの最大で running を上げ、e = 2^(ℓ − running) を求める。Σe と O^T を
// α = 2^(前の running − running) で縮めてから、O^T += V^T·P^T（hi・lo）を足す。sum は e[0] から始める（e ≥ +0 なので 0 + e[0] と
// ビットで同じ）。value は V^T の行。
template <bool kCausal>
__device__ __forceinline__ void attendBlock(HeadState& s, const float (&logit)[8], const uint16_t* value,
                                            [[maybe_unused]] int valueStride) {
  float block = logit[0];
#pragma unroll
  for (int v = 1; v < 8; ++v) {
    block = fmaxf(block, logit[v]);
  }
  block = fmaxf(block, swapHalves(block));
  const float next = fmaxf(s.running, block);
  const float origin = kCausal && next == -INFINITY ? 0.0f : next;
  const float alpha = fastExp2(s.running - origin);
  s.running = next;
  float e[8];
  e[0] = fastExp2(logit[0] - origin);
  float sum = e[0];
#pragma unroll
  for (int v = 1; v < 8; ++v) {
    e[v] = fastExp2(logit[v] - origin);
    sum += e[v];
  }
  s.total = fusedMultiplyAdd(s.total, alpha, sum);
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    s.out[v] *= alpha;
  }
#ifdef PRISM_LOGICAL_ATTENTION_FRAGMENTS
  bf16x16 high;
  bf16x16 low;
  probabilityFragments(e, high, low);
  const bf16x16 values = fragment(value);
  s.out = mma(values, high, s.out);
  s.out = mma(values, low, s.out);
#else
  s.out = attention_tile::accumulate(value, valueStride, e, s.out);
#endif
}

// クエリのタイル qt で受け持つヘッドを、キーのタイル kt < blocks について回す。関係のコードの 8 ワードは 1 タイル先を読んでおく。
template <bool kCausal, int Heads, class Shared>
__device__ __forceinline__ void attendKeys(HeadState (&state)[Heads], const bf16x16 (&query)[Heads],
                                           const Shared& shared, const AttentionArgs& args, Bounds& bounds, int row,
                                           int qt, int blocks, int lane, int column, float scale, int firstHead = 0) {
  const size_t codes = codeIndex(row, qt, 0, args.codeBlocks) + static_cast<size_t>(lane) * 8;
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
    f32x8 score[Heads];
#pragma unroll
    for (int h = 0; h < Heads; ++h) {
#ifdef PRISM_LOGICAL_ATTENTION_FRAGMENTS
      score[h] = mma(headFragment(shared.k, 16 * kt + column, firstHead + h), query[h], f32x8{});
#else
      score[h] = attention_tile::score(shared.k, 16 * kt, firstHead + h, query[h]);
#endif
    }
    float logit[Heads][8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const uint32_t code = v < 4 ? code0[v] : code1[v - 4];
      const float2 bias = pairBias(shared.bias, code);
#pragma unroll
      for (int h = 0; h < Heads; ++h) {
        logit[h][v] = fusedMultiplyAdd(score[h][v], scale, firstHead + h == 0 ? bias.x : bias.y);
        if constexpr (kCausal) {
          const int keyToken = 16 * kt + 2 * v + (lane >> 4);
          if (!causalKey(true, args.layout.fixed(), 16 * qt + column, keyToken)) {
            logit[h][v] = -INFINITY;
          }
        }
      }
    }
#pragma unroll
    for (int h = 0; h < Heads; ++h) {
      attendBlock<kCausal>(state[h], logit[h], valueRow(shared.valueT, firstHead + h, kt, column),
                           Shared::kValueStride);
    }
  }
}

// 最初のイベントを含むタイルからクエリのタイルまでを公開順に足す。
// 全キーが見える中間タイルではタイル積和を使う。両端では各クエリが見えるキーだけを積和に入れる。両端の分母は、α 倍とタイルの最初の
// 指数の和を 1 回で丸め、残りの指数を順に足す。
template <class Shared>
__device__ __forceinline__ void attendEventPrefix(HeadState (&state)[2], const bf16x16 (&query)[2],
                                                  const Shared& shared, const AttentionArgs& args, Bounds& bounds,
                                                  int row, int qt, int lane, float scale) {
  const int fixed = args.layout.fixed();
  const int column = lane & 15;
  const int half = lane >> 4;
  const int q = 16 * qt + column;
  for (int kt = fixed / 16; kt <= qt; ++kt) {
    const size_t at = bounds.index(codeIndex(row, qt, kt, args.codeBlocks) + lane * 8, 8, args.codes.limit);
    const u32x4 code0 = *reinterpret_cast<const u32x4*>(args.codes.data + at);
    const u32x4 code1 = *reinterpret_cast<const u32x4*>(args.codes.data + at + 4);
    float2 bias[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) bias[v] = pairBias(shared.bias, v < 4 ? code0[v] : code1[v - 4]);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#ifdef PRISM_LOGICAL_ATTENTION_FRAGMENTS
      const f32x8 dot = mma(headFragment(shared.k, 16 * kt + column, h), query[h], f32x8{});
#else
      const f32x8 dot = attention_tile::score(shared.k, 16 * kt, h, query[h]);
#endif
      float logit[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const int key = 16 * kt + 2 * v + half;
        logit[v] = key >= fixed && key <= q ? fusedMultiplyAdd(dot[v], scale, h ? bias[v].y : bias[v].x) : -INFINITY;
      }
      if (16 * kt >= fixed && kt < qt) {
        attendBlock<true>(state[h], logit, valueRow(shared.valueT, h, kt, column), Shared::kValueStride);
        continue;
      }
      HeadState& s = state[h];
      float block = logit[0];
#pragma unroll
      for (int v = 1; v < 8; ++v) block = fmaxf(block, logit[v]);
      const float next = fmaxf(s.running, fmaxf(block, swapHalves(block)));
      const float origin = next == -INFINITY ? 0.0f : next;
      const float alpha = fastExp2(s.running - origin);
      s.running = next;
#pragma unroll
      for (int i = 0; i < 8; ++i) s.out[i] *= alpha;
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const float weight = fastExp2(logit[v] - origin);
        const float other = swapHalves(weight);
        s.total = v == 0 ? fusedMultiplyAdd(s.total, alpha, weight) : s.total + weight;
        const int even = 16 * kt + 2 * v;
#pragma unroll
        for (int odd = 0; odd < 2; ++odd) {
          if (even + odd >= fixed && even + odd <= q) {
            const float w = half == odd ? weight : other;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const float value = bf16Float(shared.valueT[16 * h + 2 * i + half][16 * kt + 8 * odd + v]);
              s.out[i] = fusedMultiplyAdd(w, value, s.out[i]);
            }
          }
        }
      }
    }
  }
}

// 1 ヘッドの出力 O / Σe（相方の lane の Σe を足してから割る）を BF16 に 1 回丸める。クエリが行の中なら、o の位置 at（lane の次元
// 8·half 〜 8·half + 7）へ 16 バイトを 1 回で書く。
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

// 射影済みの全トークンの層では、同じ入力を持つ代表トークンの Q・K・V を読む。
struct ProjectedAttentionArgs : AttentionArgs {
  Span<const int> owners{};
  Span<const uint16_t> cachedQkv{};
};

struct PageAttentionArgs : AttentionArgs {
  Span<const int> eventSlots;
};

// 全トークンをクエリにする層 0 の段 1。ウェーブ m が受け持つタイル m の q・k・v を、埋め込みが書いた qkv0 からコピーする（qkv0 の列はヘッドの対ごとの並び
// foldColumn で、トークンのヘッドの対 pair の 192 バイトが連続する）。lane（column, half）は、トークン 16m + column のヘッド h の次元
// 8·half 〜 8·half + 7 を、q・k・v ごとに 16 バイトずつ読む（6 回の読み込みで 1 往復）。q は相方の lane と合わせてクエリの B の
// フラグメントにする。k は入れ替えた位置へ 16 バイトを 1 回で、v は V^T の 8 行へ 1 値ずつ置く。行の外の lane は行の最後のトークンを
// 読む（射影する場合と同じ）。置く値と位置は射影する場合の q・k・v と同じ形なので、段 2 は変わらない（qkv0 は埋め込みで丸めてあるので、
// ここでは丸めない）。
template <int H, class Shared, class Args>
__device__ __forceinline__ void loadFolded(Shared& shared, bf16x16 (&query)[2], const Args& args, Bounds& bounds,
                                           int row, int firstEvent, int tokens, int pair, int m, int lane) {
  constexpr int A = H / 2;
  constexpr int kWidth = 3 * A;
  const int column = lane & 15;
  const int half = lane >> 4;
  const int key = 16 * m + column;
  const int token = min(key, tokens - 1);
  const int index = args.layout.index(row, firstEvent, token);
  int source = index;
  auto qkv = args.qkv0;
  if constexpr (std::is_same_v<Args, ProjectedAttentionArgs>) {
    const int owner = args.owners.data[bounds.index(index, 1, args.owners.limit)];
    source = owner < 0 ? ~owner : owner;
    if (owner < 0) qkv = args.cachedQkv;
  }
  const size_t base =
      bounds.index(static_cast<size_t>(source) * kWidth + foldColumn(kPairWidth * pair, A), 3 * kPairWidth, qkv.limit);
  u32x4 part[3][2];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int at =
          foldColumn(j * A + kPairWidth * pair + kHeadWidth * h + 8 * half, A) - foldColumn(kPairWidth * pair, A);
      part[j][h] = *reinterpret_cast<const u32x4*>(qkv.data + base + at);
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    query[h] = joinHalves(part[0][h]);
    *reinterpret_cast<u32x4*>(&shared.k[key][8 * ((2 * h + half) ^ ((key >> 1) & 3))]) = part[1][h];
  }
  // キー 16m + 2v + half' は V^T の列 16m + v + 8·half' に、次元 d = 8·half + i は行 r(d) = 2i + half に置く。
  const int value = 16 * m + (column >> 1) + 8 * (column & 1);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t word = part[2][h][i / 2];
      shared.valueT[kHeadWidth * h + 2 * i + half][value] = static_cast<uint16_t>(i % 2 == 0 ? word : word >> 16);
    }
  }
}

// 固定クエリの層の段 1 で、イベントのキーを置く。ウェーブ m が受け持つタイル m のうち、局・席・牌より後ろのキー（位置 fixed 以上）の k・v を、
// 前段 FFNが書いたその層の eventKeyValue（イベントごとに [k(A) | v(A)]）からコピーする。lane（column, half）は、キー 16m + column の
// ヘッド h の次元 8·half 〜 8·half + 7 を k・v ごとに 16 バイトずつ読み、loadFolded と同じ置き方で置く。行の外のキーは行の最後の
// イベントを読む（射影する場合と同じ）。局・席・牌とイベントが混ざるタイルでは、同じウェーブが射影の後に呼ぶので、イベントの位置に
// 射影した値を上書きする（同じウェーブの共有メモリの読み書きは順に行われる）。
template <int H, int Heads = 2, bool Paged = false, class Shared, class Args>
__device__ __forceinline__ void loadEventKeys(Shared& shared, const Args& args, Bounds& bounds, int firstEvent,
                                              int fixed, int tokens, int pair, int m, int lane, int firstHead = 0) {
  constexpr int A = H / 2;
  const int column = lane & 15;
  const int half = lane >> 4;
  const int key = 16 * m + column;
  if (key < fixed) {
    return;
  }
  int event = firstEvent + min(key, tokens - 1) - fixed;
  if constexpr (Paged) event = args.eventSlots.data[bounds.index(event, 1, args.eventSlots.limit)];
  const size_t base =
      bounds.index(static_cast<size_t>(event) * (2 * A) + kPairWidth * pair, A + kPairWidth, args.eventKeyValue.limit);
  u32x4 part[2][Heads];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int h = 0; h < Heads; ++h) {
      const int offset = j * A + kHeadWidth * (firstHead + h) + 8 * half;
      part[j][h] = *reinterpret_cast<const u32x4*>(args.eventKeyValue.data + base + offset);
    }
  }
#pragma unroll
  for (int h = 0; h < Heads; ++h) {
    *reinterpret_cast<u32x4*>(&shared.k[key][8 * ((2 * (firstHead + h) + half) ^ ((key >> 1) & 3))]) = part[0][h];
  }
  // キー 16m + 2v + half' は V^T の列 16m + v + 8·half' に、次元 d = 8·half + i は行 r(d) = 2i + half に置く。
  const int value = 16 * m + (column >> 1) + 8 * (column & 1);
#pragma unroll
  for (int h = 0; h < Heads; ++h) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t word = part[1][h][i / 2];
      shared.valueT[kHeadWidth * (firstHead + h) + 2 * i + half][value] =
          static_cast<uint16_t>(i % 2 == 0 ? word : word >> 16);
    }
  }
}

// 固定クエリの層の射影のうち、1 ウェーブが 1 回に受け持つ分。ヘッド h について、タイル first 〜 first + kBlocks − 1（先頭の kQueries 個が局・席・
// 牌の q を作る）の q・k・v を作り、バイアスを足して丸めて共有メモリへ置く（q は lane の 8 次元を fixedQuery へ、k・v は全トークンの層と同じ置き方）。
// 重みのフラグメントをタイルで共有するので、1 ステップの読み込みは重み 3 つ（q を作らなければ 2 つ）と n の kBlocks 個。バイアスも
// タイルで共有して 1 回だけ読む。値は、タイルごとに 1 ヘッドを回す場合とビットで同じ（各 acc のオペランドと K の順、バイアスの足し方と
// 丸めが同じ）。
template <int H, int kBlocks, int kQueries, bool keepQuery = false, class Shared>
__device__ __forceinline__ void projectLast(Shared& shared, const AttentionArgs& args, Bounds& bounds,
                                            device::Buffer weights, int row, int firstEvent, int tokens, int pair,
                                            int h, int first, int lane, bf16x16* query = nullptr) {
  constexpr int A = H / 2;
  constexpr int kSets = kBlocks == 2 && kQueries > 0 ? 2 : 4;  // projectHead
  const int column = lane & 15;
  const int half = lane >> 4;
  const device::Buffer inputs = device::boundedBuffer(args.normalized.data, args.normalized.limit);
  uint32_t inputRow[kBlocks];
  uint32_t touched[kBlocks][3];
#pragma unroll
  for (int b = 0; b < kBlocks; ++b) {
    const int token = min(16 * (first + b) + column, tokens - 1);
    inputRow[b] = static_cast<uint32_t>(bounds.index(static_cast<size_t>(args.layout.index(row, firstEvent, token)) * H,
                                                     H, args.normalized.limit)) *
                  static_cast<uint32_t>(sizeof(uint16_t));
    // n の行の 2〜4 本目の 128 バイトのラインを 1 ワードずつ先に読み、MALL から引いておく（全トークンの層と同じ。値は射影の後の空の asm まで
    // 生かしておく）。
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      touched[b][t] = device::loadWord(inputs, inputRow[b], 128 * (t + 1));
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
      const u32x4 rounded = pack8Activation(values);
      if constexpr (keepQuery) {
        static_assert(kQueries == 1);
        *query = joinHalves(rounded);
      } else {
        const int t = 16 * (first + b) + column;
        *reinterpret_cast<u32x4*>(&shared.fixedQuery[t][8 * ((2 * h + half) ^ ((t >> 1) & 3))]) = rounded;
      }
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
    device::keepPrefetched(touched[b][0], touched[b][1], touched[b][2]);
  }
}

// グループごとのカーネル。ウェーブは kWaves 個、共有メモリは 16·kWaves トークン分。ワークグループの行は、グループの順に並べた行の番号
// rowOrder[groupStart + blockIdx.y]。kLast は固定クエリの層（局・席・牌の層）を表し、kEvents はその層のイベントのキーとバリューの作り方。kFolded は
// 全トークンをクエリにする層 0 を表す（段 1 で qkv0 をコピーする）。
template <int H, bool kLast, bool kFolded, EventKeys kEvents, int kWaves, bool kCausalEvents, class Args>
__global__ __launch_bounds__(kWave* kWaves) void attentionKernel(Args args) {
  constexpr int kThreads = kWave * kWaves;
  constexpr int kTokens = 16 * kWaves;
  constexpr int A = H / 2;
  constexpr int kPairElements =
      (H / 16) * kStepBytes / static_cast<int>(sizeof(uint16_t));  // ヘッドの対 1 つ分の重みのコピーの要素数
  // イベントの k・v をコピーする固定クエリの層（kCopied）は、局・席・牌のタイルだけを射影する。
  constexpr bool kPaged = kEvents == EventKeys::kPaged;
  constexpr bool kCopied = kLast && (kEvents == EventKeys::kCopied || kPaged);
  // 固定タイルの各ヘッドにウェーブを割り当てられる場合、射影から Attention までを同じウェーブで行う。
  constexpr bool kSplitHeads = kCopied && !kCompiledWidth<H>.folded && kWaves >= 2 * kFixedBlocks;
  // 全てのタイルを射影する固定クエリの層のうち W ≥ 6 のグループは、射影で 1 ウェーブが 2 タイルの 1 ヘッドを作り、q を共有メモリに置く。W = 4 の
  // グループは 1 ウェーブが 1 タイルの 2 ヘッドを作り、q はレジスタに残す（q の 3 KB を共有メモリに足すと、そのグループの占有が 8 から 7 に
  // 下がる）。
  constexpr bool kPairedBlocks = kLast && !kCopied && kWaves > attentionWaves(0);
  constexpr int kBiasCopies =
      (kBiasChunks + kThreads - 1) / kThreads;  // 1 スレッドがコピーするバイアスの表のチャンクの数
  using Shared = AttentionShared<kTokens, kPairedBlocks>;
  static_assert(sizeof(Shared) <= device::kStaticSharedBytes,
                "the block exceeds the static shared memory of the device");
  static_assert(!kLast || kFixedBlocks <= kWaves, "fixed-token attention gives every query block a wave");
  static_assert(!kPairedBlocks || kWaves % 2 == 0,
                "fixed-token attention pairs the waves of the two heads in the projection");
  static_assert(kThreads <= kBiasChunks, "every thread copies at least one chunk of the bias table");
  static_assert(!(kLast && kFolded), "only all-token attention reads the folded layer-0 q, k and v");
  static_assert(kLast || kEvents == EventKeys::kProjected, "all-token attention projects every token");
  __shared__ Shared shared;

  const int pair = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int column = lane & 15;  // A・B のフラグメントの行、C のフラグメントの列
  const int half = lane >> 4;    // C のフラグメントで奇数の行を持つ lane なら 1
  const TokenLayout layout = args.layout;
  Bounds bounds;

  const int row =
      args.rowOrder.data[bounds.index(static_cast<size_t>(args.groupStart) + blockIdx.y, 1, args.rowOrder.limit)];
  const int firstEvent =
      args.rowColumn
          .data[bounds.index(static_cast<size_t>(row) * args.rowWidth + args.rowFirstEvent, 1, args.rowColumn.limit)];
  const int events =
      args.rowColumn
          .data[bounds.index(static_cast<size_t>(row) * args.rowWidth + args.rowEvents, 1, args.rowColumn.limit)];
  const int fixed = layout.fixed();
  // トークンの数は、ルールの上限で行のコードのタイル（16·codeBlocks トークン）以下、グループの上限で 16W 以下。範囲外とグループの
  // 取り違えは範囲検査つきのビルドで止め、通常のビルドでも共有メモリの外には書かない（グループは attentionGroup が同じ切り詰めた T で
  // 決めるので、通常は切り詰めが起きない）。
  const int tokens = min(fixed + max(events, 0), kTokens);
  (void)bounds.index(static_cast<size_t>(fixed + events), 0, static_cast<size_t>(16 * args.codeBlocks));
  (void)bounds.index(static_cast<size_t>(fixed + max(events, 0)), 0, kTokens);
  const int queries = kLast ? fixed : tokens;
  const int blocks = (tokens + 15) / 16;  // 16 トークンのタイルの数（キーのタイル。W 以下）

  // 0. 2 ヘッドのバイアスの表を 16 バイトのチャンクでコピーする。スレッドはチャンク tid + c·32W（c < kBiasCopies）を受け持ち、読み込みを
  //    全て先に出して待ちを 1 回にする（表の外を指すスレッドはチャンク tid を読み直して同じ所へ書き直す。書き込みに分岐がないので、
  //    読み込みが分岐の中へ沈まない）。
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

  // 1. q・k・v（タイル積和）。n と重みのコピーは device::Buffer から読む。範囲の検査は、n では lane の行の位置（行の全てのステップを含む）に、
  //    コピーではワークグループのヘッドの対の 48 KB にかける。行の外の lane は行の最後のトークン（イベントの k・v をコピーする固定クエリの層では
  //    最後の局・席・牌のトークン）を読む。読む値は有限で、そのキーはコード kPaddedKeyCode で −∞ になって P = 0 になるか、コピーする
  //    固定クエリの層のイベントの位置なら loadEventKeys が上書きする。そのクエリの出力は書かない。q の A のフラグメントの行 r は W_q の行 π(r)
  //    なので（コピーがそう並べてある）、C のフラグメントの lane はトークン 16m + column の次元 8·half + v（v < 8）を持つ。全トークンをクエリにする層 0 は
  //    射影をしないので、コピーを読まない。
  const device::Buffer weights = device::boundedBuffer(
      kFolded ? args.qkv.data
              : args.qkv.data + bounds.index(static_cast<size_t>(pair) * kPairElements, kPairElements, args.qkv.limit),
      kFolded ? args.qkv.limit : kPairElements);
  bf16x16 keptQuery
      [2];  // クエリのタイル m の q（ヘッドごとの B のフラグメント。2 タイルの 1 ヘッドを作る固定クエリの層のグループは共有メモリに置く）
  const auto* queryMask = reinterpret_cast<const int*>(args.eventKeyValue.data);
  const bool newEvents = !kCausalEvents || !queryMask || (queryMask[row] & (1 << wave));
  const bool needed = wave * 16 < fixed || newEvents;
  const int projected =
      kCopied ? (fixed + 15) / 16 : blocks;  // 射影するタイルの数（コピーする固定クエリの層は局・席・牌のタイルだけ）
  if constexpr (kSplitHeads) {
    const int w = device::firstLane(wave);
    const int m = w / 2;
    const int h = w % 2;
    if (m < projected) {
      projectLast<H, 1, 1, true>(shared, args, bounds, weights, row, firstEvent, fixed, pair, h, m, lane,
                                 &keptQuery[0]);
    }
    // 固定とイベントの混在タイルでは、同じヘッドを射影したウェーブがイベントの列を上書きする。
    for (int job = w; job < 2 * blocks; job += kWaves) {
      const int tile = job / 2;
      if (tokens > fixed && 16 * (tile + 1) > fixed) {
        loadEventKeys<H, 1, kPaged>(shared, args, bounds, firstEvent, fixed, tokens, pair, tile, lane, job % 2);
      }
    }
  } else if constexpr (kPairedBlocks) {
    // 全てのタイルを射影する固定クエリの層（W ≥ 6）: ウェーブ w はヘッド w & 1 について、タイルの対 2(w >> 1)・2(w >> 1) + 1 を作る
    //    （projectLast。行の最後のタイルが対の前の側なら 1 つだけ）。タイルは W 以下なので、対は全てのタイルを 1 回ずつ覆う。q を作るのは
    //    局・席・牌のタイル（16m < fixed）だけで、対の中では前のタイルから続く。ウェーブの番号は一様なレジスタに置く（ヘッドは重みのコピーの
    //    変位に入るので、lane ごとの値のままだと読み込みごとに lane を回すループになる。タイルの分岐もウェーブで 1 つになる）。
    const int uniformWave = device::firstLane(wave);
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
  } else if constexpr (kFolded) {
    // 全トークンをクエリにする層 0: ウェーブ m（< タイルの数）がタイル m の q・k・v を qkv0 からコピーする（q はレジスタ、k・v は共有メモリ。射影する場合と
    // 同じ置き方）。
    if (wave < blocks) {
      loadFolded<H>(shared, keptQuery, args, bounds, row, firstEvent, tokens, pair, wave, lane);
    }
  } else if (wave < projected) {
    // ウェーブ m（< 射影するタイルの数）が、16 トークンのタイル m についてヘッドごとに q・k・v の 3 つのフラグメントを作る。q はクエリの
    // タイル qt = m のものとしてレジスタに残す。フラグメントの組（projectHead）は全トークンの層が 3 つ、固定クエリの層が 4 つ。
    // 3 組で回す経路は、16 列のステップを 3 つずつ処理して最後に 1 つ残す。
    constexpr int kSets = !kLast && (H / 16) % 3 == 1 ? 3 : 4;
    const uint32_t fragmentRow =
        static_cast<uint32_t>(column) * 16;  // コピーの半分の中の lane の行（lane 16〜31 は 0〜15 と同じ）
    const device::Buffer inputs = device::boundedBuffer(args.normalized.data, args.normalized.limit);
    const int m = wave;
    const int token = min(16 * m + column, (kCopied ? fixed : tokens) - 1);
    const uint32_t inputRow[1] = {
        static_cast<uint32_t>(
            bounds.index(static_cast<size_t>(layout.index(row, firstEvent, token)) * H, H, args.normalized.limit)) *
        static_cast<uint32_t>(sizeof(uint16_t))};
    // 固定クエリの層のクエリは局・席・牌まで。キャッシュに命中した全トークンの層のタイルも K と V だけを作る。
    const bool hasQuery = needed && (!kLast || 16 * m < fixed);
    // n の行（512 バイト）の 2〜4 本目の 128 バイトのライン（ステップ 4〜15 で読む）を 1 ワードずつ先に読み、MALL から引いておく。値は
    // 使わないが、2 ヘッドの射影の後の空の asm まで生かしておく（読み込みを消させず、その待ちも射影の後にだけ置かせる）。
    uint32_t touched[3];
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      touched[t] = device::loadWord(inputs, inputRow[0], 128 * (t + 1));
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
      // バイアスを足して丸める。q は相方の lane と合わせてクエリの B のフラグメントにする。k は入れ替えた位置へ 1 値ずつ、v は V^T の
      // 行へ 8 値を 1 回で書く。
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
      // 次のヘッドの先読みを、このヘッドの丸めに使う一時レジスタと重ねない（重ねるとレジスタが増えて占有が下がる）。
      device::scheduleLoads();
    }
    device::keepPrefetched(touched[0], touched[1], touched[2]);
  }
  if constexpr (kCopied && !kSplitHeads) {
    // イベントの k・v をコピーする固定クエリの層: イベントの位置を含むタイル（局・席・牌と混ざるタイルと、イベントだけのタイル）のキーを、前段 FFNが
    // 作った k・v からコピーする。イベントのない行は、混ざるタイルのイベントの位置に、射影した値（最後の局・席・牌のトークン。有限）が
    // 残る。
    if (wave < blocks && 16 * (wave + 1) > fixed && tokens > fixed) {
      loadEventKeys<H, 2, kPaged>(shared, args, bounds, firstEvent, fixed, tokens, pair, wave, lane);
    }
  }
  phaseBarrier();

  // 2. Attention。キーのタイルは先頭から順に回す。射影したウェーブが同じなら q をレジスタから、異なるなら共有メモリから読む。
  const int queryBlocks = (queries + 15) / 16;
  const float scale = args.scale * kLog2e;
  const int qt = wave;
  if constexpr (kSplitHeads) {
    const int w = device::firstLane(wave);
    const int tile = w / 2;
    const int h = w % 2;
    if (tile < queryBlocks) {
      const int q = 16 * tile + column;
      HeadState state[1];
      const bf16x16 fragment[1] = {keptQuery[0]};
      attendKeys<false>(state, fragment, shared, args, bounds, row, tile, blocks, lane, column, scale, h);
      const size_t at =
          static_cast<size_t>(layout.index(row, firstEvent, q)) * A + kPairWidth * pair + kHeadWidth * h + 8 * half;
      storeHead(state[0], q < queries, at, args, bounds);
    }
  } else if (qt < queryBlocks && needed) {
    const int query = qt * 16 + column;
    bf16x16 queryFragment[2];
    if constexpr (kPairedBlocks) {
      queryFragment[0] = headFragment(shared.fixedQuery, query, 0);
      queryFragment[1] = headFragment(shared.fixedQuery, query, 1);
    } else {
      queryFragment[0] = keptQuery[0];
      queryFragment[1] = keptQuery[1];
    }
    if (kCausalEvents && 16 * (qt + 1) > fixed && newEvents) {
      HeadState events[2];
      attendEventPrefix(events, queryFragment, shared, args, bounds, row, qt, lane, scale);
      const size_t at = static_cast<size_t>(layout.index(row, firstEvent, query)) * A + kPairWidth * pair + 8 * half;
#pragma unroll
      for (int h = 0; h < 2; ++h)
        storeHead(events[h], query >= fixed && query < queries, at + kHeadWidth * h, args, bounds);
    }
    // 混在するタイルではタイル積和に全 lane が参加し、固定クエリの結果だけを書く。
    if (!kCausalEvents || qt * 16 < fixed) {
      HeadState state[2];
      attendKeys<kCausalEvents>(state, queryFragment, shared, args, bounds, row, qt, blocks, lane, column, scale);
      const size_t at = static_cast<size_t>(layout.index(row, firstEvent, query)) * A + kPairWidth * pair + 8 * half;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        storeHead(state[h], query < queries && (!kCausalEvents || query < fixed), at + kHeadWidth * h, args, bounds);
      }
    }
  }
  bounds.trap();
}

// 関係のコードを計算した後に、行の番号をグループの順に並べる（グループの中は行の順なので、並びはグループだけで決まる）。1 つのワークグループが 1024 行
// ずつ、ウェーブの ballot でグループごとの順位を数えて書く。置いた行の数が、ホストが数えたグループの行の数と違えば、範囲検査つきの
// ビルドで止める（行がグループの 16W を超えれば全トークンの層と固定クエリの層も止める）。
__global__ __launch_bounds__(kOrderThreads) void rowOrderKernel(RowOrderArgs args) {
  __shared__ int waveRows[kOrderWaves][kAttentionGroups];  // 今の 1024 行の中の、ウェーブごと・グループごとの行の数
  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  Bounds bounds;
  int placed[kAttentionGroups] = {};  // 前の 1024 行までに置いた、グループごとの行の数（どのスレッドも同じ）
  for (int first = 0; first < args.rows; first += kOrderThreads) {
    const int row = first + tid;
    int group = -1;
    if (row < args.rows) {
      const int events =
          args.rowColumn
              .data[bounds.index(static_cast<size_t>(row) * args.rowWidth + args.rowEvents, 1, args.rowColumn.limit)];
      group = attentionGroup(args.fixed, events);
    }
    int rank = 0;  // ウェーブの中で前の lane にある、同じグループの行の数
#pragma unroll
    for (int g = 0; g < kAttentionGroups; ++g) {
      const uint32_t mask = device::ballot(group == g);
      if (lane == 0) {
        waveRows[wave][g] = __popc(mask);
      }
      rank = group == g ? __popc(mask & ((1u << lane) - 1u)) : rank;
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
  // 置いた数とホストが数えた数が等しいことを確かめる（≤ と ≥ を範囲の検査で確かめる）。
#pragma unroll
  for (int g = 0; g < kAttentionGroups; ++g) {
    (void)bounds.index(static_cast<size_t>(placed[g]), 0, static_cast<size_t>(args.groupRows[g]));
    (void)bounds.index(static_cast<size_t>(args.groupRows[g]), 0, static_cast<size_t>(placed[g]));
  }
  bounds.trap();
}

// グループ kGroup の行（rowOrder の start から rows 行）について起動し、start を次のグループの先頭へ進める。行の数が 0 のグループは
// 起動しない。

template <int H, bool kLast, bool kFolded, EventKeys kEvents, bool kCausalEvents, int kGroup, bool kSplitShortHeads,
          class Args>
void launchGroup(Args args, int& start, int rows, int headPairs, gpu::Stream stream) {
  constexpr bool kSplitHeads =
      kSplitShortHeads && H == kWidthProjected.width && kLast && kEvents == EventKeys::kPaged && kGroup == 0;
  constexpr int kWaves = kSplitHeads ? 2 * kFixedBlocks : attentionWaves(kGroup);
  static_assert(kWaves >= attentionWaves(kGroup), "the block must cover every key tile in its row group");
  if (rows > 0) {
    args.groupStart = start;
    // 同じ行のヘッドの対が隣り合って走る（行の x とコードを L2 で使い回す）。
    attentionKernel<H, kLast, kFolded, kEvents, kWaves, kCausalEvents>
        <<<dim3(headPairs, rows), dim3(kWave * kWaves), 0, stream>>>(args);
    PRISM_GPU(gpu::lastError());
  }
  start += rows;
}

template <int H, bool kLast, bool kFolded, EventKeys kEvents, bool kCausalEvents = false, bool kSplitShortHeads = false,
          class Args, int... kGroups>
void launchGroups(const Args& args, const int (&groupRows)[kAttentionGroups], int headPairs, gpu::Stream stream,
                  std::integer_sequence<int, kGroups...>) {
  int start = 0;
  (launchGroup<H, kLast, kFolded, kEvents, kCausalEvents, kGroups, kSplitShortHeads>(args, start, groupRows[kGroups],
                                                                                     headPairs, stream),
   ...);
}

// 重みを公開するときに、1 層の W_qkv を射影の重みのコピー（先頭の説明）の並びに並べ替える。ワークグループは 1 つのヘッドの対を受け
// 持ち、スレッドはコピー先の 16 バイト（W_qkv の行 unit(j, h, r) の K 方向の 8 要素）を 1 つずつコピーする。
template <int H>
__global__ __launch_bounds__(kQkvThreads) void qkvFragmentKernel(QkvFragmentArgs args) {
  constexpr int A = H / 2;
  constexpr int kChunks = (H / 16) * kStepBytes / 16;  // ヘッドの対 1 つ分のコピーに含まれる 16 バイトの数
  const int pair = blockIdx.x;
  Bounds bounds;

  for (int c = threadIdx.x; c < kChunks; c += kQkvThreads) {
    // c = (((k·2 + h)·3 + j)·2 + upper)·16 + r（upper は K 方向の上の 8 要素なら 1）。
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

template <int H>
static void launchCausal(const AttentionArgs& args, int headPairs, const int (&groupRows)[kAttentionGroups],
                         gpu::Stream stream) {
  if (args.layout.rows <= 0) {
    return;
  }
  if (args.role != LayerRole::kAllTokens || args.layout.fixed() > kMaxFixedTokens || args.codeBlocks <= 0 ||
      args.codeBlocks > kMaxRowTokens / 16) {
    throw std::invalid_argument("invalid causal layer zero layout");
  }
  launchGroups<H, false, kCompiledWidth<H>.folded, EventKeys::kProjected, true>(
      args, groupRows, headPairs, stream, std::make_integer_sequence<int, kAttentionGroups>{});
}

void launchRowOrder(const RowOrderArgs& args, gpu::Stream stream) {
  if (args.rows <= 0) {
    return;
  }
  rowOrderKernel<<<dim3(1), dim3(kOrderThreads), 0, stream>>>(args);
  PRISM_GPU(gpu::lastError());
}

template <int H>
static void launchBackbone(int hidden, bool folded, const AttentionArgs& args, int headPairs,
                           const int (&groupRows)[kAttentionGroups], gpu::Stream stream) {
  if (args.layout.rows <= 0) {
    return;
  }
  const bool last = args.role == LayerRole::kFixedTokens;
  if (last && folded) {
    throw std::invalid_argument("fixed-token attention has no folded layer-0 path");
  }
  if (!last && args.eventKeys != EventKeys::kProjected) {
    throw std::invalid_argument("all-token attention projects the keys and values of every token");
  }
  if (args.layout.fixed() > kMaxFixedTokens || args.codeBlocks <= 0 || args.codeBlocks > kMaxRowTokens / 16) {
    throw std::invalid_argument("all-token attention/fixed-token attention take at most " +
                                std::to_string(kMaxFixedTokens) + " fixed tokens and " +
                                std::to_string(kMaxRowTokens / 16) + " code blocks per row, got " +
                                std::to_string(args.layout.fixed()) + " and " + std::to_string(args.codeBlocks));
  }
  constexpr auto groups = std::make_integer_sequence<int, kAttentionGroups>{};
  constexpr EventKeys kProjected = EventKeys::kProjected;
  constexpr EventKeys kCopied = EventKeys::kCopied;
  if (hidden != H || (folded && !kCompiledWidth<H>.folded)) {
    throw std::invalid_argument("unsupported all-token attention/fixed-token attention width or folded projection");
  }
  if (last && args.eventKeys == kCopied) {
    launchGroups<H, true, false, kCopied>(args, groupRows, headPairs, stream, groups);
  } else if (last) {
    launchGroups<H, true, false, kProjected>(args, groupRows, headPairs, stream, groups);
  } else {
    if constexpr (kCompiledWidth<H>.folded) {
      if (folded) {
        launchGroups<H, false, true, kProjected>(args, groupRows, headPairs, stream, groups);
        return;
      }
    }
    launchGroups<H, false, false, kProjected>(args, groupRows, headPairs, stream, groups);
  }
}

void launchPageAttention(const AttentionArgs& args, Span<const int> slots, int headPairs,
                         const int (&groupRows)[kAttentionGroups], bool splitShortHeads, gpu::Stream stream) {
  const auto groups = std::make_integer_sequence<int, kAttentionGroups>{};
  const PageAttentionArgs paged{args, slots};
  if constexpr (KernelTuning::kCanSplitShortPagedHeads) {
    if (splitShortHeads) {
      launchGroups<kWidthProjected.width, true, false, EventKeys::kPaged, false, true>(paged, groupRows, headPairs,
                                                                                       stream, groups);
      return;
    }
  }
  launchGroups<kWidthProjected.width, true, false, EventKeys::kPaged>(paged, groupRows, headPairs, stream, groups);
}

template <int H>
static void packQkv(int hidden, const QkvFragmentArgs& args, int headPairs, gpu::Stream stream) {
  if (hidden != H) {
    throw std::invalid_argument("the QKV fragment copy has no specialisation for hidden " + std::to_string(hidden));
  }
  qkvFragmentKernel<H><<<dim3(headPairs), dim3(kQkvThreads), 0, stream>>>(args);
  PRISM_GPU(gpu::lastError());
}

void launchAttention(int hidden, bool folded, const AttentionArgs& args, int headPairs,
                     const int (&groupRows)[kAttentionGroups], gpu::Stream stream) {
  launchBackbone<kWidthFolded.width>(hidden, folded, args, headPairs, groupRows, stream);
}

void launchFoldedCausalAttention(const AttentionArgs& args, int headPairs, const int (&groupRows)[kAttentionGroups],
                                 gpu::Stream stream) {
  launchCausal<kWidthFolded.width>(args, headPairs, groupRows, stream);
}

void launchQkvFragments(int hidden, const QkvFragmentArgs& args, int headPairs, gpu::Stream stream) {
  packQkv<kWidthFolded.width>(hidden, args, headPairs, stream);
}

void launchProjectedAttention(int hidden, bool folded, const AttentionArgs& args, int headPairs,
                              const int (&groupRows)[kAttentionGroups], gpu::Stream stream) {
  launchBackbone<kWidthProjected.width>(hidden, folded, args, headPairs, groupRows, stream);
}

void launchProjectedCausalAttention(const AttentionArgs& args, int headPairs, const int (&groupRows)[kAttentionGroups],
                                    gpu::Stream stream) {
  launchCausal<kWidthProjected.width>(args, headPairs, groupRows, stream);
}

void launchProjectedQkvFragments(int hidden, const QkvFragmentArgs& args, int headPairs, gpu::Stream stream) {
  packQkv<kWidthProjected.width>(hidden, args, headPairs, stream);
}

}  // namespace prism

namespace prism {
namespace {

constexpr int kProjectionTileRows = 32;

struct ProjectionTags {
  __device__ unsigned long long load(unsigned long long* tag) const { return device::loadTag(tag); }
  __device__ unsigned long long claim(unsigned long long* tag, unsigned long long value) const {
    return atomicCAS(tag, 0ull, value);
  }
};
#include "cache/projection_cache.cuh"

// 代表の正規化済み入力をまとめて共有メモリに読み、全ヘッドの射影に使う。Q と K/V では元の
// projectHead と同じ左右のオペランドを使う。K の順、バイアスを足す位置、BF16 への丸めもそろえる。
template <int H>
__global__ __launch_bounds__(tile::kThreads) void projectDistinctInputs(AttentionArgs args, const int* sources,
                                                                        const int* count, Span<uint16_t> output,
                                                                        ProjectionIndex index) {
  constexpr int A = H / 2;
  constexpr int parts = kProjectionTileRows / 16;
  const int first = blockIdx.x * kProjectionTileRows;
  const int rows = min(kProjectionTileRows, *count - first);
  if (rows <= 0) return;
  const int tid = threadIdx.x;
  const int wave = device::firstLane(tid / kWave);
  const int column = tid % 16;
  const int half = (tid % kWave) >> 4;
  __shared__ int source[kProjectionTileRows], destination[kProjectionTileRows];
  __shared__ __align__(16) uint16_t input[kProjectionTileRows][H + tile::kPad];
  Bounds bounds;
  if (tid < kProjectionTileRows) {
    source[tid] = sources[first + min(tid, rows - 1)];
    destination[tid] = index.owners[source[tid]];
  }
  tile::barrier();
  for (int i = tid; i < kProjectionTileRows * (H / 8); i += tile::kThreads) {
    const int row = i / (H / 8);
    const int col = (i % (H / 8)) * 8;
    const size_t at = bounds.index(static_cast<size_t>(source[row]) * H + col, 8, args.normalized.limit);
    *reinterpret_cast<u32x4*>(&input[row][col]) = *reinterpret_cast<const u32x4*>(args.normalized.data + at);
  }
  tile::barrier();
  for (int tileId = wave; tileId < 3 * A / kHeadWidth; tileId += tile::kWaves) {
    const int pair = tileId / 6;
    const int head = tileId % 2;
    const int projection = (tileId % 6) / 2;
    const int base = (pair * (H / 16) * kStepBytes + (3 * head + projection) * kFragmentBytes) / sizeof(uint16_t);
    const int extent = ((H / 16 - 1) * kStepBytes + kFragmentBytes) / sizeof(uint16_t);
    const auto weights = device::boundedBuffer(args.qkv.data + bounds.index(base, extent, args.qkv.limit), extent);
    f32x8 acc[parts]{};
#ifdef PRISM_LOGICAL_ATTENTION_FRAGMENTS
    bf16x16 weight[3];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      weight[k] = bufferFragment(weights, column * 16, k * kStepBytes, k * kStepBytes + kFragmentBytes / 2);
    }
#pragma unroll
    for (int k = 0; k < H / 16; ++k) {
      if (k + 2 < H / 16) {
        const int at = (k + 2) * kStepBytes;
        weight[(k + 2) % 3] = bufferFragment(weights, column * 16, at, at + kFragmentBytes / 2);
      }
      device::scheduleLoads();
#pragma unroll
      for (int m = 0; m < parts; ++m) {
        const bf16x16 values = fragment(&input[16 * m + column][k * 16]);
        acc[m] = projection == 0 ? mma(weight[k % 3], values, acc[m]) : mma(values, weight[k % 3], acc[m]);
      }
      device::scheduleLoads();
    }
#else
    attention_tile::projectDistinct<H>(acc, weights, input, projection);
#endif
    const int biasColumn = projection * A + pair * kPairWidth + head * kHeadWidth;
    const int outColumn = foldColumn(biasColumn, A);
    if (projection == 0) {
      const size_t at = bounds.index(biasColumn + 8 * half, 8, args.qkvBias.limit);
      const f32x8 bias = *reinterpret_cast<const f32x8*>(args.qkvBias.data + at);
#pragma unroll
      for (int m = 0; m < parts; ++m) {
        const int row = 16 * m + column;
        if (row < rows) {
          float values[8];
#pragma unroll
          for (int v = 0; v < 8; ++v) values[v] = acc[m][v] + bias[v];
          const int dst = destination[row];
          const auto target = dst < 0 ? index.cache.qkv : output;
          const size_t to =
              bounds.index(static_cast<size_t>(dst < 0 ? ~dst : dst) * (3 * A) + outColumn + 8 * half, 8, target.limit);
          *reinterpret_cast<u32x4*>(target.data + to) = pack8Activation(values);
        }
      }
    } else {
      const float bias = args.qkvBias.data[bounds.index(biasColumn + column, 1, args.qkvBias.limit)];
#pragma unroll
      for (int m = 0; m < parts; ++m) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int row = 16 * m + 2 * v + half;
          if (row < rows) {
            const int dst = destination[row];
            const auto target = dst < 0 ? index.cache.qkv : output;
            const size_t to =
                bounds.index(static_cast<size_t>(dst < 0 ? ~dst : dst) * (3 * A) + outColumn + column, 1, target.limit);
            target.data[to] = bf16RoundActivation(acc[m][v] + bias);
          }
        }
      }
    }
  }
  bounds.trap();
}

}  // namespace

void indexLayerInputs(const EmbedArgs& args, const EmbeddingSpec (&specs)[kEmbeddingSlots], ProjectionIndex index,
                      gpu::Stream stream) {
  if (index.tokens == 0) return;
  PRISM_GPU(gpu::clear(index.table(), 0, (static_cast<size_t>(index.tokens) + 1) * sizeof(int), stream));
  auto column = [&](int section) -> RowBytes {
    if (section < 0) return {args.input.data, 0};
    return {args.input.data + args.section[section],
            args.sectionWidth[section] * sectionElementBytes(static_cast<Section>(section))};
  };
  int first = 0;
  for (int slot = 0; slot < static_cast<int>(EmbeddingSlot::kTransition); ++slot) {
    const int rows = args.count[slot];
    if (rows == 0) continue;
    const auto& spec = specs[slot];
    const InputKey key{{column(spec.category), column(spec.scalar), column(spec.exact)}};
    const int blocks = (rows + tile::kThreads - 1) / tile::kThreads;
    indexProjectionInputs<<<dim3(blocks), dim3(tile::kThreads), 0, stream>>>(
        key, slot, first, rows, index.owners, index.sources(), index.table() + first, index.count(), index.cache);
    PRISM_GPU(gpu::lastError());
    first += rows;
  }
}

void launchSharedProjection(AttentionArgs args, ProjectionIndex index, Span<uint16_t> projections, int headPairs,
                            const int (&groupRows)[kAttentionGroups], gpu::Stream stream) {
  if (index.tokens == 0) return;
  const int blocks = (index.tokens + kProjectionTileRows - 1) / kProjectionTileRows;
  projectDistinctInputs<kWidthProjected.width>
      <<<dim3(blocks), dim3(tile::kThreads), 0, stream>>>(args, index.sources(), index.count(), projections, index);
  PRISM_GPU(gpu::lastError());
  args.qkv0 = {projections.data, projections.limit};
  const ProjectedAttentionArgs mapped{
      args, {index.owners, static_cast<size_t>(index.tokens)}, {index.cache.qkv.data, index.cache.qkv.limit}};
  launchGroups<kWidthProjected.width, false, true, EventKeys::kProjected, true>(
      mapped, groupRows, headPairs, stream, std::make_integer_sequence<int, kAttentionGroups>{});
}

}  // namespace prism
