// K8（REREAD_KEY_VALUE）: ワークグループは局・席・牌のトークンを 48 個ずつ受け持つ（256 スレッド = 8 ウェーブ）。最後の正規化の後の
// y から、候補層の 2 段目が使う局・席・牌のキーとバリューを作る。書く先は行ごとに局・席・牌の順に並べた [R, F, 2A] で、K9 が行ごとに
// 読む並び:
//
//   [K V] = ⌊y·W_kvᵀ + b_kv⌉
//
// 射影と書き込み方は、K5 の Cross-Attention のキーとバリューと同じ（projectKeyValue・storeRowMajorFixed）。リダクションは全て
// トークンの中で閉じるので、同じトークンの結果はバッチの組み方に依らずビットまで同じ。グローバルメモリで読む領域と書く領域が別なので、
// バリアは全て LDS だけを守る tile::barrier で足りる。
#include "prism_hip_mlp.h"

namespace prism {
namespace {

using tile::kThreads;

constexpr int kKeyRows = 48;  // ワークグループが受け持つトークンの数（K5 と同じ）

template <int H>
struct alignas(16) KeyValueShared {
  uint16_t stream[kKeyRows][H + tile::kPad];
};

template <int H>
__global__ __launch_bounds__(kThreads) void keyValueKernel(KeyValueArgs args) {
  static_assert(sizeof(KeyValueShared<H>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  __shared__ KeyValueShared<H> shared;

  const int tid = threadIdx.x;
  const int token0 = static_cast<int>(blockIdx.x) * kKeyRows;
  const int rows = min(kKeyRows, args.layout.fixed() * args.layout.rows - token0);

  Bounds bounds;
  tile::stageTile<H>(shared.stream, args.state.data, args.state.limit, token0, rows, tid, bounds);
  tile::barrier();
  tile::projectKeyValue<H>(shared.stream, args.weight, args.bias, bounds);
  tile::storeRowMajorFixed<H>(args.memory.data, args.memory.limit, shared.stream, args.layout, token0, rows, tid,
                              bounds);
  bounds.trap();
}

}  // namespace

void launchKeyValue(int hidden, const KeyValueArgs& args, hipStream_t stream) {
  const size_t width = static_cast<size_t>(hidden);  // 2A
  const size_t tokens = static_cast<size_t>(args.layout.fixed()) * args.layout.rows;
  if (args.weight.limit != width * hidden || args.bias.limit < width || args.state.limit < tokens * width ||
      args.memory.limit < tokens * width) {
    throw std::invalid_argument("K8 needs a [2A, H] key and value weight copy, a [2A] bias and the " +
                                std::to_string(tokens) + " fixed tokens of y and of the memory");
  }
  const int blocks = static_cast<int>((tokens + kKeyRows - 1) / kKeyRows);
  if (blocks == 0) {
    return;
  }
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL(keyValueKernel<256>, dim3(blocks), dim3(kThreads), 0, stream, args);
      break;
    default:
      throw std::invalid_argument("K8 has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
