// K0（関係の符号）: 作業群は 1 行（256 スレッド）。行の全ての対（問い合わせ i, 鍵 j）の 7 区画の符号を推論の語（attentionCode）に
// 詰めて全体メモリへ書く（置き方は prism_hip_relation.h の codeIndex）。K2・K4 の全ての頭の対と層が同じ符号を読むので、1 回の順伝播で
// 1 回だけ作る。
//
// 行のトークンごとに属性を詰めた語（RelationToken、12 バイト）を LDS に 1 回作り、対の符号はトークンの語 2 つと牌の関係の表の 1 バイト
// から選択だけで作る（relationSegments）。問い合わせの塊 qt を外の輪にして、問い合わせの語は鍵の塊の間レジスタに置く。塊の対ごとに、
// スレッド t が語 t（1 KiB の連続）を書く。鍵が行の外なら kPaddedKeyCode、問い合わせだけが外なら 0（行の外の位置は行の最後の
// トークンの語で作って捨てる）。
//
// 塊の型（BlockKind: 固定の 39 トークンなら塊 0・1 が FIXED、塊 2 が MIXED、塊 3 から EVENT）は作業群で一様なので、塊の対ごとに
// スカラーの分岐で型の組の式を選ぶ。型の組の式は型から 0 と決まる項と、固定の塊では起きない行の外の選択を落とすだけなので、全ての語は
// 型を問わない式とビットで同じ。行の外の位置の語（行の最後のトークン）は出来事の塊では出来事なので、EVENT の式で作ってよい。
//
// 偏りの表（relationBiasKernel）: 重みの公開のときに層ごとに 1 回（Plan::uploadWeights）。作業群は 1 頭の対で、K2・K4 が引く
// 2 頭の表（prism_hip_relation.h の置き方）を ρ から作る。
#include "prism_hip_relation.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
static_assert(kThreads == kCodeBlockWords, "one thread per code word of a block pair");

// 塊 b（位置 16b〜16b + 15）の型。fixed は局・席・牌のトークンの数 F（位置 0〜F − 1）。
__device__ __forceinline__ BlockKind blockKind(int block, int fixed) {
  return 16 * block + 16 <= fixed ? BlockKind::FIXED : 16 * block >= fixed ? BlockKind::EVENT : BlockKind::MIXED;
}

// 問い合わせの位置 i と鍵の位置 j の対の語（i・j は塊の型が Query・Key の塊の中）。query は位置 min(i, S − 1) の語、tileRow はその
// 牌の関係の表の行の先頭。FIXED の塊は行の中（16b + 15 < F ≤ S）なので、行の外の選択と位置の切り詰めを落とす。
template <BlockKind Query, BlockKind Key>
__device__ __forceinline__ uint32_t pairCode(const RelationToken* token, const uint8_t* tileRelation,
                                             const RelationToken& query, int tileRow, int i, int j, int tokens,
                                             Bounds& bounds) {
  const bool queryInside = Query == BlockKind::FIXED || i < tokens;
  const bool keyInside = Key == BlockKind::FIXED || j < tokens;
  const RelationToken key = token[Key == BlockKind::FIXED ? j : min(j, tokens - 1)];
  int segments[kRelationSegments];
  relationSegments<Query, Key>(query, key, tileRelation[tileRow + key.tileIndex()], segments);
  // 区画の符号は 16 未満（推論の語の項の番号が組の表に収まる）で、区画 6 の 15 は行の外の鍵の印なので実の対では 14 以下
  // （検査つきの組み立てで止める）。
  const bool inside = queryInside && keyInside;
#pragma unroll
  for (int s = 0; s < kRelationSegments; ++s) {
    (void)bounds.index(static_cast<size_t>(inside ? segments[s] : 0), 1,
                       s == kRelationSegments - 1 ? kRelationEntries - 1 : kRelationEntries);
  }
  return !keyInside ? kPaddedKeyCode : queryInside ? attentionCode(segments) : 0u;
}

// 問い合わせの塊 qt（型 Query）と全ての鍵の塊の対の語を書く。鍵の塊の型は kt だけで決まるのでスカラーの分岐で選ぶ。
template <BlockKind Query>
__device__ __forceinline__ void writeQueryBlock(const RelationArgs& args, const RelationToken* token,
                                                const uint8_t* tileRelation, int row, int qt, int blocks, int tokens,
                                                Bounds& bounds) {
  const int tid = threadIdx.x;
  const int lane = tid >> 3;
  const int v = tid & 7;
  const int fixed = args.layout.fixed();
  const int i = 16 * qt + (lane & 15);
  const RelationToken query = token[Query == BlockKind::FIXED ? i : min(i, tokens - 1)];
  const int tileRow = query.tileIndex() * args.layout.tiles;
  for (int kt = 0; kt < blocks; ++kt) {
    const int j = 16 * kt + 2 * v + (lane >> 4);
    const BlockKind keyKind = blockKind(kt, fixed);
    const uint32_t code =
        keyKind == BlockKind::FIXED
            ? pairCode<Query, BlockKind::FIXED>(token, tileRelation, query, tileRow, i, j, tokens, bounds)
        : keyKind == BlockKind::EVENT
            ? pairCode<Query, BlockKind::EVENT>(token, tileRelation, query, tileRow, i, j, tokens, bounds)
            : pairCode<Query, BlockKind::MIXED>(token, tileRelation, query, tileRow, i, j, tokens, bounds);
    args.codes.data[bounds.index(codeIndex(row, qt, kt) + tid, 1, args.codes.limit)] = code;
  }
}

__global__ __launch_bounds__(kThreads) void relationKernel(RelationArgs args) {
  __shared__ RelationToken token[kMaxRowTokens];
  __shared__ uint8_t tileRelation[kMaxTiles * kMaxTiles];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const TokenLayout layout = args.layout;
  const RelationColumns columns = args.columns;
  Bounds bounds;

  const int firstEvent = args.rowColumn.data[bounds.index(
      static_cast<size_t>(row) * args.rowWidth + args.rowFirstEvent, 1, args.rowColumn.limit)];
  const int events = args.rowColumn.data[bounds.index(
      static_cast<size_t>(row) * args.rowWidth + args.rowEvents, 1, args.rowColumn.limit)];
  const int fixed = layout.fixed();
  // 規則の上限で 149 以下。範囲外は検査つきの組み立てで止め、通常の組み立てでも LDS の外には書かない。
  const int tokens = min(fixed + max(events, 0), kMaxRowTokens);
  (void)bounds.index(static_cast<size_t>(fixed + events), 0, kMaxRowTokens);

  // 行の局のトークン（位置 0）の番号。
  const int now = args.relation.data[bounds.index(
      static_cast<size_t>(layout.index(row, firstEvent, 0)) * columns.width + columns.event, 1, args.relation.limit)];
  for (int p = tid; p < tokens; p += kThreads) {
    const size_t at = static_cast<size_t>(layout.index(row, firstEvent, p)) * columns.width;
    uint8_t attributes[kMaxAttributes];
#pragma unroll
    for (int c = 0; c < kMaxAttributes; ++c) {
      attributes[c] = c < columns.width ? args.relation.data[bounds.index(at + c, 1, args.relation.limit)] : 0;
    }
    token[p] = relationToken(layout.type(p), attributes, now, columns);
  }
  for (int e = tid; e < layout.tiles * layout.tiles; e += kThreads) {
    tileRelation[e] = args.tileRelation.data[bounds.index(e, 1, args.tileRelation.limit)];
  }
  __syncthreads();

  const int blocks = (tokens + 15) / 16;
  for (int qt = 0; qt < blocks; ++qt) {
    switch (blockKind(qt, fixed)) {
      case BlockKind::FIXED:
        writeQueryBlock<BlockKind::FIXED>(args, token, tileRelation, row, qt, blocks, tokens, bounds);
        break;
      case BlockKind::MIXED:
        writeQueryBlock<BlockKind::MIXED>(args, token, tileRelation, row, qt, blocks, tokens, bounds);
        break;
      case BlockKind::EVENT:
        writeQueryBlock<BlockKind::EVENT>(args, token, tileRelation, row, qt, blocks, tokens, bounds);
        break;
    }
  }
  bounds.trap();
}

// 偏りの表の 1 頭の対: 項 e の (頭 2p, 頭 2p + 1) の値（×log2e）。組の表 τ の項 pairTableEntry(a, b) は ρ[32τ + a] + ρ[32τ + 16 + b]、
// 区画 6 の表の項 a は ρ[96 + a]（15 は行の外の鍵の −∞）。ρ は [関係の行, 頭]。
__global__ __launch_bounds__(kThreads) void relationBiasKernel(RelationBiasArgs args) {
  const int pair = blockIdx.x;
  Bounds bounds;

  for (int e = threadIdx.x; e < kBiasEntries; e += kThreads) {
    float value[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int head = 2 * pair + h;
      float bias;
      if (e < kPairTables * kPairEntries) {
        const int table = e / kPairEntries;
        const int entry = e % kPairEntries;
        const int first = kRelationEntries * 2 * table + pairTableFirst(entry);
        const int second = kRelationEntries * (2 * table + 1) + pairTableSecond(entry);
        bias = args.rho.data[bounds.index(static_cast<size_t>(first) * args.heads + head, 1, args.rho.limit)] +
               args.rho.data[bounds.index(static_cast<size_t>(second) * args.heads + head, 1, args.rho.limit)];
      } else {
        const int entry = e - kPairTables * kPairEntries;
        bias = entry == kRelationEntries - 1
                   ? -INFINITY
                   : args.rho.data[bounds.index(
                         static_cast<size_t>(kRelationEntries * (kRelationSegments - 1) + entry) * args.heads + head, 1,
                         args.rho.limit)];
      }
      value[h] = bias * kLog2e;
    }
    const size_t at = bounds.index((static_cast<size_t>(pair) * kBiasEntries + e) * 2, 2, args.table.limit);
    args.table.data[at] = value[0];
    args.table.data[at + 1] = value[1];
  }
  bounds.trap();
}

}  // namespace

void launchRelation(const RelationArgs& args, hipStream_t stream) {
  if (args.layout.rows <= 0) {
    return;
  }
  if (args.columns.safeties > kRelationOpponents) {
    throw std::invalid_argument("K0 holds the safety of at most " + std::to_string(kRelationOpponents) +
                                " opponents, got " + std::to_string(args.columns.safeties) + " safety columns");
  }
  hipLaunchKernelGGL(relationKernel, dim3(args.layout.rows), dim3(kThreads), 0, stream, args);
  PRISM_HIP(hipGetLastError());
}

void launchRelationBias(const RelationBiasArgs& args, int headPairs, hipStream_t stream) {
  hipLaunchKernelGGL(relationBiasKernel, dim3(headPairs), dim3(kThreads), 0, stream, args);
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
