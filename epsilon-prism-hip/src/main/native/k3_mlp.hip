// K3（MLP_ALL・MLP_FIXED）と K5（MLP_LAST）。どちらもワークグループ（256 スレッド = 8 ウェーブ）が kMlpRows トークンを受け持ち、
// MLP の段は prism_hip_mlp.h の mlpStage。重みを公開するときに、MLP の重みをフラグメントの順に並べ替えたコピーを作る
// mlpFragmentKernel もここに置く（K3・K5・K6・K8・K9 が読む）。
//
// 1 回の起動は、トークンの軸の 1 つか 2 つの区間（MlpSegment）にまたがる。ワークグループは自分の区間の重みと FFN の幅で mlpStage を
// 回す。区間が 1 つなら型で分けない FFN、2 つなら局・席 [0, (1+S)·R) と牌 [(1+S)·R, F·R) で型ごとの FFN（O 射影は層で共通）。区間の
// 境は 48 の倍数とは限らず、区間ごとに最後のタイルだけが半端になる。トークンの結果は区間の重みと自分の入力だけで決まり、区間の数にも
// タイルの中の位置にも依らずビットで同じ。
//
// K3: x ← mlpStage(x, o)。続けて、次の層の Attention の入力 n = ⌊N_attn(x)⌉ をレジスタから NORMALIZED へ書く。x をグローバルメモリへ
//     書き戻すのは先頭の residualTokens トークン、n を書くのは先頭の normalizedTokens トークン（次が局・席・牌の層なら局・席・牌だけ。
//     n は LDS の x から作る）。最後の全トークンの層は、イベントのトークンの x を書かない（局・席・牌の層はイベントの残差を読まない）。
//     代わりに、2 つ目以降の局・席・牌の層と 2 段目が読むイベントのキーとバリューを、読む側ごとにその場で作る（eventKeyValues）。
//     最初の局・席・牌の層の Attention は、全てのトークンの n からイベントの k・v も射影する。
// K5: 局・席・牌のトークン（トークンの軸の先頭 F·R）に最後の層の mlpStage をかけ、y = ⌊N_final(x)⌉（ゲインは区間ごと）を STATE へ、
//     [K V] = ⌊y·W_kvᵀ + b_kv⌉ を MEMORY の [行, 位置, 2A] へ書く（候補層の Cross-Attention が行ごとに読む並び）。
// 行列積はソフトウェアパイプラインで回す（キーとバリューの射影は、2 ステップずつ回すループの gemmFragmentsLoop）。バリアは全て
// tile::barrier で、LDS の順序だけをそろえる（読む領域と書く領域が別であることが前提）。重みの行列（W_o・W1・W2・W_kv）は、重みを
// 公開するときに計画が mlpFragmentKernel で作った、フラグメントの順のコピーから読む。
#include "prism_hip_mlp.h"

namespace prism {
namespace {

using tile::kThreads;

// ワークグループが受け持つトークンの数。64 より LDS が小さく、1 SIMD に載るウェーブが多い。
constexpr int kMlpRows = 48;

// ワークグループが受け持つ、区間の中のトークン [token0, token0 + rows)。
struct MlpBlock {
  const MlpSegment& segment;
  int token0;  // トークンの軸の番号
  int rows;    // 区間の最後のタイルだけ kMlpRows より少ない
};

// ワークグループの区間は、blockStart[s] ≤ ワークグループの番号 < blockStart[s + 1] となる s（ワークグループで一様なスカラー）。区間が
// 1 つなら blockStart[1] はワークグループの数なので、全てのワークグループが区間 0 を受け持つ。
template <class Args>
__device__ __forceinline__ MlpBlock mlpBlock(const Args& args) {
  const int block = static_cast<int>(blockIdx.x);
  const int s = block >= args.blockStart[1] ? 1 : 0;
  const MlpSegment& segment = args.segment[s];
  const int token0 = segment.firstToken + (block - args.blockStart[s]) * kMlpRows;
  return MlpBlock{segment, token0, min(kMlpRows, segment.firstToken + segment.tokens - token0)};
}

// 最後の全トークンの層で K3 の最後に行う処理。タイルの中のイベントのトークンについて、キーとバリューを組ごとに作り、組の [イベント, 2A]
// へ書く。n_j = ⌊(x·r)·g_j⌉ の r（1/rms）は x だけで決まるので組によらず共通で、normalizeStore と同じ受け持ち（4 スレッドで 1 行）と
// 同じ和のツリーで 1 回だけ求めて LDS に置く。⌊(x·r)·g_j⌉ は要素ごとの式なので、x は stageTile と同じ受け持ちでレジスタに残し、組ごとに
// n_j を stream へ書く（n_j は、normalizeStore が同じゲインで書く n とビットで同じ）。[K V] = ⌊n_j·W_jᵀ + b_j⌉ は projectKeyValue で
// 求める。局・席・牌とイベントが混ざるタイルは全ての行を射影し、イベントの行だけを書く。
template <int H>
__device__ __forceinline__ void eventKeyValues(tile::MlpShared<H, kMlpRows>& shared, const MlpArgs& args, int token0,
                                               int rows, int tid, Bounds& bounds) {
  constexpr int kVectors = H / 8;
  constexpr int kPasses = kMlpRows * kVectors / kThreads;
  __shared__ float inverse[kMlpRows];  // 行の r
  const int row = tid >> 2;
  if (row < kMlpRows) {  // 4 スレッドで 1 行（48 行ならウェーブ 6・7 は行を持たない。ウェーブで一様）
    const int col0 = (tid & 3) * (H / 4);
    u32x4 part[H / 32];
#pragma unroll
    for (int c = 0; c < H / 32; ++c) {
      part[c] = *reinterpret_cast<const u32x4*>(&shared.stream[row][col0 + 8 * c]);
    }
    const float r = tile::inverseRms<H>(part, args.epsilon);
    if ((tid & 3) == 0) {
      inverse[row] = r;
    }
  }
  const int col = (tid % kVectors) * 8;
  u32x4 x[kPasses];
#pragma unroll
  for (int pass = 0; pass < kPasses; ++pass) {
    x[pass] = *reinterpret_cast<const u32x4*>(&shared.stream[tile::tileRow<H, kMlpRows>(pass, tid)][col]);
  }
  const int first = max(args.firstEvent - token0, 0);  // タイルの中で最初のイベントの行
  for (int j = 0; j < args.eventKeyValues; ++j) {
    const EventKeyValue& keyValue = args.eventKeyValue[j];
    // stream を読む前の処理（x・n・r の読み込みと、前の組のキーとバリューの書き出し）が全てのウェーブで済み、r が LDS にそろってから
    // n_j を置く。
    tile::barrier();
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
      const int at = tile::tileRow<H, kMlpRows>(pass, tid);
      *reinterpret_cast<u32x4*>(&shared.stream[at][col]) =
          tile::normalize8(x[pass], keyValue.gain.data, keyValue.gain.limit, col, inverse[at], bounds);
    }
    tile::barrier();
    tile::projectKeyValue<H>(shared.stream, keyValue.weight, keyValue.bias, bounds);
    tile::storeTileRows<H>(keyValue.destination.data, keyValue.destination.limit, shared.stream,
                           token0 - args.firstEvent, first, rows, tid, bounds);
  }
}

template <int H>
__global__ __launch_bounds__(kThreads) void mlpKernel(MlpArgs args) {
  constexpr int A = H / 2;
  static_assert(sizeof(tile::MlpShared<H, kMlpRows>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  __shared__ tile::MlpShared<H, kMlpRows> shared;

  const MlpBlock block = mlpBlock(args);
  const int tid = threadIdx.x;
  const int token0 = block.token0;
  const int rows = block.rows;

  Bounds bounds;
  tile::stageTile<H>(shared.stream, args.x.data, args.x.limit, token0, rows, tid, bounds);
  tile::stageTile<A>(shared.inner, args.o.data, args.o.limit, token0, rows, tid, bounds);
  tile::barrier();
  tile::mlpStage<H>(shared, block.segment.weights, block.segment.ffn, args.epsilon, bounds);
  bounds.trap();
  // 書き戻す行の数はワークグループで一様。0 以下のタイルは LDS も読まずに飛ばす。
  const int stored = min(rows, args.residualTokens - token0);
  if (stored > 0) {
    tile::storeTile<H>(args.x.data, args.x.limit, shared.stream, token0, stored, tid, bounds);
  }
  // 次の層の Attention の入力 n = ⌊(x·r)·g⌉ をレジスタから直接書く（K2・K4 が A のフラグメントとしてそのまま読む）。
  const int normalized = min(rows, args.normalizedTokens - token0);
  if (normalized > 0) {
    tile::normalizeStore<H>(shared.stream, args.nextNorm.data, args.nextNorm.limit, args.epsilon,
                            args.normalized.data, args.normalized.limit, token0, normalized, tid, bounds);
  }
  // イベントのトークンを含むタイル（ワークグループで一様）は、最後の全トークンの層なら、後の層が読むイベントのキーとバリューを作る。
  if (args.eventKeyValues > 0 && token0 + rows > args.firstEvent) {
    eventKeyValues<H>(shared, args, token0, rows, tid, bounds);
  }
  bounds.trap();
}

template <int H>
__global__ __launch_bounds__(kThreads) void mlpKvKernel(MlpKvArgs args) {
  constexpr int A = H / 2;
  static_assert(sizeof(tile::MlpShared<H, kMlpRows>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  __shared__ tile::MlpShared<H, kMlpRows> shared;

  const MlpBlock block = mlpBlock(args);
  const int tid = threadIdx.x;
  const int token0 = block.token0;
  const int rows = block.rows;

  Bounds bounds;
  tile::stageTile<H>(shared.stream, args.x.data, args.x.limit, token0, rows, tid, bounds);
  tile::stageTile<A>(shared.inner, args.o.data, args.o.limit, token0, rows, tid, bounds);
  tile::barrier();
  tile::mlpStage<H>(shared, block.segment.weights, block.segment.ffn, args.epsilon, bounds);

  // y = ⌊N_final(x)⌉ を区間のゲインでその場で作り、STATE へ書く。
  tile::normalizeTile<H>(shared.stream, block.segment.finalNorm.data, block.segment.finalNorm.limit, args.epsilon, tid,
                         bounds);
  tile::barrier();
  tile::storeTile<H>(args.state.data, args.state.limit, shared.stream, token0, rows, tid, bounds);

  // [K V] = ⌊y·W_kvᵀ + b_kv⌉ を、行ごとに局・席・牌の順に並ぶ [R, F, 2A] へ書く。
  tile::projectKeyValue<H>(shared.stream, args.keyValue, args.keyValueBias, bounds);
  tile::storeRowMajorFixed<2 * A>(args.memory.data, args.memory.limit, shared.stream, args.layout, token0, rows, tid,
                                  bounds);
  bounds.trap();
}

// 重みを公開するときに、行列 W [rows, columns] をフラグメントの順に並べ替えてコピーする（prism_hip_mlp.h の FragmentWeights）。
// スレッドはコピー先の 16 バイト i（W の 1 行の K 方向の 8 要素）を 1 つずつ受け持つ。i = ((c·S + s)·2 + upper)·16 + r
// （S = columns/16）は、W の行 16c + r の要素 16s + 8·upper から始まる 8 要素で、書き込みは i の順に連続する。
__global__ __launch_bounds__(kThreads) void mlpFragmentKernel(MlpFragmentArgs args) {
  const int steps = args.columns / 16;
  const int vectors = args.rows * args.columns / 8;
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i >= vectors) {
    return;
  }
  const int r = i % 16;
  const int upper = (i / 16) % 2;
  const int s = (i / 32) % steps;
  const int c = i / 32 / steps;
  Bounds bounds;
  const size_t row = 16 * c + r;
  const size_t from = bounds.index(row * args.columns + 16 * s + 8 * upper, 8, args.weight.limit);
  const size_t to = bounds.index(static_cast<size_t>(i) * 8, 8, args.fragments.limit);
  *reinterpret_cast<u32x4*>(args.fragments.data + to) = *reinterpret_cast<const u32x4*>(args.weight.data + from);
  bounds.trap();
}

// 区間を確かめ、区間ごとの最初のワークグループの番号を区間のトークンの数から求めて blockStart に入れる。区間は 1 つか 2 つで、トークンの
// 軸の 0 から隙間なく続き、重みのコピーとバイアスは区間の FFN の幅に合った形でなければならない（違えば invalid_argument）。最後の区間の
// 終わり（全ての区間のトークンの数の合計）を返す。
template <class Args>
int segmentBlocks(int hidden, Args& args, const char* kernel) {
  if (args.segments < 1 || args.segments > 2) {
    throw std::invalid_argument(std::string(kernel) + " takes one or two FFN segments, got " +
                                std::to_string(args.segments));
  }
  const size_t h = static_cast<size_t>(hidden);
  const size_t a = h / 2;
  int blocks = 0;
  int end = 0;
  for (int s = 0; s < args.segments; ++s) {
    const MlpSegment& segment = args.segment[s];
    const std::string name = std::string(kernel) + " segment " + std::to_string(s);
    if (segment.tokens < 0 || segment.firstToken != end) {
      throw std::invalid_argument(name + " must continue the token axis at " + std::to_string(end));
    }
    if (segment.ffn <= 0 || segment.ffn % tile::kChunk != 0) {
      throw std::invalid_argument(name + " has FFN width " + std::to_string(segment.ffn));
    }
    const MlpWeights& w = segment.weights;
    const size_t f = static_cast<size_t>(segment.ffn);
    if (w.outWeight.limit != h * a || w.upWeight.limit != f * h || w.downWeight.limit != h * f || w.outBias.limit < h ||
        w.normGain.limit < h || w.upBias.limit < f || w.downBias.limit < h) {
      throw std::invalid_argument(name + " needs the weights of an FFN of width " + std::to_string(segment.ffn));
    }
    args.blockStart[s] = blocks;
    blocks += (segment.tokens + kMlpRows - 1) / kMlpRows;
    end += segment.tokens;
  }
  args.blockStart[args.segments] = blocks;
  return end;
}

}  // namespace

void launchMlpFragments(const MlpFragmentArgs& args, hipStream_t stream) {
  if (args.rows <= 0 || args.columns <= 0 || args.rows % 16 != 0 || args.columns % 16 != 0) {
    throw std::invalid_argument("the MLP fragment copy needs a positive multiple of 16 rows and columns, got " +
                                std::to_string(args.rows) + " x " + std::to_string(args.columns));
  }
  const size_t elements = static_cast<size_t>(args.rows) * args.columns;
  if (args.weight.limit != elements || args.fragments.limit != elements) {
    throw std::invalid_argument("the MLP fragment copy of a " + std::to_string(args.rows) + " x " +
                                std::to_string(args.columns) + " matrix got " + std::to_string(args.weight.limit) +
                                " weights and room for " + std::to_string(args.fragments.limit));
  }
  const int vectors = args.rows * args.columns / 8;
  hipLaunchKernelGGL(mlpFragmentKernel, dim3((vectors + kThreads - 1) / kThreads), dim3(kThreads), 0, stream, args);
  PRISM_HIP(hipGetLastError());
}

void launchMlp(int hidden, const MlpArgs& args, hipStream_t stream) {
  MlpArgs launched = args;
  const int end = segmentBlocks(hidden, launched, "K3");
  if (args.residualTokens < 0 || args.residualTokens > end || args.normalizedTokens < 0 ||
      args.normalizedTokens > end) {
    throw std::invalid_argument("K3 writes back " + std::to_string(args.residualTokens) + " residual and " +
                                std::to_string(args.normalizedTokens) + " normalized tokens of " +
                                std::to_string(end));
  }
  if (args.eventKeyValues < 0 || args.eventKeyValues > kMaxEventKeyValues) {
    throw std::invalid_argument("K3 makes zero to " + std::to_string(kMaxEventKeyValues) +
                                " sets of event keys and values, got " + std::to_string(args.eventKeyValues));
  }
  if (args.eventKeyValues > 0) {
    if (args.firstEvent < 0 || args.firstEvent > end) {
      throw std::invalid_argument("K3 events start at token " + std::to_string(args.firstEvent) + " of " +
                                  std::to_string(end));
    }
    const size_t width = static_cast<size_t>(hidden);  // 2A
    const size_t events = static_cast<size_t>(end - args.firstEvent);
    for (int j = 0; j < args.eventKeyValues; ++j) {
      const EventKeyValue& keyValue = args.eventKeyValue[j];
      if (keyValue.gain.limit < width || keyValue.weight.limit != width * hidden || keyValue.bias.limit < width ||
          keyValue.destination.limit < events * width) {
        throw std::invalid_argument("K3 event key and value set " + std::to_string(j) +
                                    " needs a [H] gain, a [2A, H] weight copy, a [2A] bias and room for " +
                                    std::to_string(events) + " events");
      }
    }
  }
  const int blocks = launched.blockStart[launched.segments];
  if (blocks <= 0) {
    return;
  }
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL(mlpKernel<256>, dim3(blocks), dim3(kThreads), 0, stream, launched);
      break;
    default:
      throw std::invalid_argument("K3 has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

void launchMlpKv(int hidden, const MlpKvArgs& args, hipStream_t stream) {
  MlpKvArgs launched = args;
  const int end = segmentBlocks(hidden, launched, "K5");
  if (end != args.layout.fixed() * args.layout.rows) {
    throw std::invalid_argument("K5 segments cover " + std::to_string(end) + " tokens, not the " +
                                std::to_string(args.layout.fixed() * args.layout.rows) + " fixed tokens");
  }
  for (int s = 0; s < args.segments; ++s) {
    const Span<const float>& gain = args.segment[s].finalNorm;
    if (gain.data == nullptr || gain.limit < static_cast<size_t>(hidden)) {
      throw std::invalid_argument("K5 segment " + std::to_string(s) + " needs the gain of the final norm");
    }
  }
  const int blocks = launched.blockStart[launched.segments];
  if (blocks <= 0) {
    return;
  }
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL(mlpKvKernel<256>, dim3(blocks), dim3(kThreads), 0, stream, launched);
      break;
    default:
      throw std::invalid_argument("K5 has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
