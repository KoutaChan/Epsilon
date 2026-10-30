// K3（mlp0）と K5（mlp1_kv）。どちらも kMlpRows トークンの作業群（256 スレッド = 8 波）で、MLP の段は prism_hip_mlp.h の mlpStage。
// 重みの公開のときに MLP の重みの断片の順の写しを作る mlpFragmentKernel もここに置く（K3・K5・K6 が読む）。
//
// K3: x ← mlpStage(x, o)（全てのトークン、最後でない層）。続けて次の層の注意の入力 n = ⌊N_attn(x)⌉ をレジスタから NORMALIZED へ。
// K5: 局・席・牌のトークン（トークンの軸の先頭 F·R）に最後の層の mlpStage → y = ⌊N_final(x)⌉ を STATE へ →
//     [K V] = ⌊y·W_kvᵀ + b_kv⌉ を MEMORY の [行, 位置, 2A] へ（候補層の交差注意が行ごとに読む並び）。
// 行列積は段ずらし（K5 の鍵と値の射影は 2 歩ずつの輪の gemmFragmentsLoop）、待ち合わせは全て tile::barrier（LDS だけ。読む区画と
// 書く区画が別という約束）。重みの行列（W_o・W1・W2・W_kv）は計画が公開のときに mlpFragmentKernel で作る断片の順の写しから読む。
#include "prism_hip_mlp.h"

namespace prism {
namespace {

using tile::kThreads;
using tile::kWaves;

// 作業群のトークン。64 より LDS が小さく、1 SIMD に載る波が多い。
constexpr int kMlpRows = 48;
constexpr int kRowTiles = kMlpRows / 16;

template <int H>
__global__ __launch_bounds__(kThreads) void mlpKernel(MlpArgs args) {
  constexpr int A = H / 2;
  static_assert(sizeof(tile::MlpShared<H, kMlpRows>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  __shared__ tile::MlpShared<H, kMlpRows> shared;

  const int tid = threadIdx.x;
  const int token0 = blockIdx.x * kMlpRows;
  const int rows = min(kMlpRows, args.tokens - token0);

  Bounds bounds;
  tile::stageTile<H>(shared.stream, args.x.data, args.x.limit, token0, rows, tid, bounds);
  tile::stageTile<A>(shared.inner, args.o.data, args.o.limit, token0, rows, tid, bounds);
  tile::barrier();
  tile::mlpStage<H>(shared, args.weights, args.ffn, args.epsilon, bounds);
  bounds.trap();
  tile::storeTile<H>(args.x.data, args.x.limit, shared.stream, token0, rows, tid, bounds);
  // 次の層の注意の入力 n = ⌊(x·r)·g⌉ をレジスタから直に書く（K2・K4 が A の断片として直に読む）。
  tile::normalizeStore<H>(shared.stream, args.nextNorm.data, args.nextNorm.limit, args.epsilon, args.normalized.data,
                          args.normalized.limit, token0, rows, tid, bounds);
  bounds.trap();
}

template <int H>
__global__ __launch_bounds__(kThreads) void mlpKvKernel(MlpKvArgs args) {
  constexpr int A = H / 2;
  constexpr int kColumns = H / 16 / kWaves;  // 2A = H なので鍵と値の列も O 射影と同じ受け持ち
  constexpr int kVectors = H / 8;            // トークンの鍵と値の 16 バイトの数
  static_assert(sizeof(tile::MlpShared<H, kMlpRows>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  static_assert(kVectors == kWave, "the memory store gives each wave one token per pass");
  __shared__ tile::MlpShared<H, kMlpRows> shared;

  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);  // 波で同じ（写しの soffset を SGPR に置く）
  const int half = lane >> 4;
  const int column = lane & 15;
  const int token0 = blockIdx.x * kMlpRows;
  const int rows = min(kMlpRows, args.tokens - token0);

  Bounds bounds;
  tile::stageTile<H>(shared.stream, args.x.data, args.x.limit, token0, rows, tid, bounds);
  tile::stageTile<A>(shared.inner, args.o.data, args.o.limit, token0, rows, tid, bounds);
  tile::barrier();
  tile::mlpStage<H>(shared, args.weights, args.ffn, args.epsilon, bounds);

  // y = ⌊N_final(x)⌉ をその場で作り、STATE へ書く。
  tile::normalizeTile<H>(shared.stream, args.finalNorm.data, args.finalNorm.limit, args.epsilon, tid, bounds);
  tile::barrier();
  tile::storeTile<H>(args.state.data, args.state.limit, shared.stream, token0, rows, tid, bounds);

  // [K V] = ⌊y·W_kvᵀ + b_kv⌉。
  f32x8 acc[kRowTiles][kColumns];
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      acc[m][j] = f32x8{};
    }
  }
  const tile::FragmentWeights keyValue = tile::fragmentWeights(args.keyValue, H, column);
  int output[kColumns];
  int base[kColumns];
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    output[j] = (wave * kColumns + j) * 16 + column;
    base[j] = tile::fragmentBase<H / 16>(keyValue, wave * kColumns + j, 0, bounds);
  }
  tile::gemmFragmentsLoop<kColumns, H>(acc, shared.stream, keyValue, base, column);
  tile::barrier();
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    const int col = output[j];
    const float bias = tile::load(args.keyValueBias.data, args.keyValueBias.limit, col, bounds);
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        shared.stream[m * 16 + 2 * v + half][col] = bf16Round(acc[m][j][v] + bias);
      }
    }
  }
  tile::barrier();

  // 行の鍵と値を [行, 位置, 2A] へ（トークンごとに 2A 要素、16 バイトずつ）。pass ごとに波が 1 トークンを受け持つので、行と位置
  // （TokenLayout::locate の 2 つの割り算）は readfirstlane で波に 1 回、SALU で求める。
  const int fixed = args.layout.fixed();
#pragma unroll
  for (int pass = 0; pass < kMlpRows * kVectors / kThreads; ++pass) {
    const int index = pass * kThreads + tid;
    const int row = __builtin_amdgcn_readfirstlane(index / kVectors);
    const int col = (index % kVectors) * 8;
    if (row < rows) {
      int owner;
      int position;
      args.layout.locate(token0 + row, owner, position);
      const size_t at =
          bounds.index((static_cast<size_t>(owner) * fixed + position) * (2 * A) + col, 8, args.memory.limit);
      *reinterpret_cast<u32x4*>(args.memory.data + at) = *reinterpret_cast<const u32x4*>(&shared.stream[row][col]);
    }
  }
  bounds.trap();
}

// 重みの公開のとき: 行列 W [rows, columns] を断片の順の写し（prism_hip_mlp.h の FragmentWeights）に並べ替える。スレッドは写しの
// 16 バイト i（W の行の K の 8 要素）を 1 つずつ写す。i = ((c·S + s)·2 + upper)·16 + r（S = columns/16）は W の行 16c + r の
// 要素 16s + 8·upper からの 8 要素で、書き込みは i の順に続く。
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
  if (args.tokens <= 0) {
    return;
  }
  const dim3 grid((args.tokens + kMlpRows - 1) / kMlpRows);
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL(mlpKernel<256>, grid, dim3(kThreads), 0, stream, args);
      break;
    default:
      throw std::invalid_argument("K3 has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

void launchMlpKv(int hidden, const MlpKvArgs& args, hipStream_t stream) {
  if (args.tokens <= 0) {
    return;
  }
  const dim3 grid((args.tokens + kMlpRows - 1) / kMlpRows);
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL(mlpKvKernel<256>, grid, dim3(kThreads), 0, stream, args);
      break;
    default:
      throw std::invalid_argument("K5 has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
