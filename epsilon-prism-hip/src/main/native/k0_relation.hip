// K0（関係のコード）: 1 つのワークグループ（256 スレッド）が 1 行を受け持つ。行の中の全ての対（クエリ i、キー j）について、
// 7 つのセグメントのコードを推論のワード（attentionCode）に詰めて、グローバルメモリへ書く（置き方は prism_hip_relation.h の
// codeIndex）。K2・K4 は全てのヘッドの対と層で同じコードを読むので、1 回の順伝播で 1 回だけ作る。
//
// まず行のトークンごとに、属性を詰めた記述（RelationToken、12 バイト）を LDS に 1 回だけ作る。対のコードは、トークンの記述 2 つと
// 牌の関係の表の 1 バイトから、選択だけで作る（relationSegments）。クエリのタイル qt を外側のループにし、クエリの記述はキーのタイルを
// 回る間レジスタに置く。タイルの対ごとに、スレッド t がワード t を書く（合わせて 1 KiB の連続）。キーが行の外なら kPaddedKeyCode、
// クエリだけが行の外なら 0 を書く（行の外の位置は、行の最後のトークンの記述で作ってから捨てる）。
//
// タイルの型（BlockKind。局・席・牌が 39 トークンならタイル 0・1 が FIXED、タイル 2 が MIXED、タイル 3 から EVENT。31 トークンなら
// タイル 0 が FIXED、タイル 1 が MIXED）はワークグループの中で同じなので、タイルの対ごとにスカラーの分岐で型の組に合う式を選ぶ。
// 型の組ごとの式は、型から 0 と決まる項と、FIXED のタイルでは起きない行の外の選択を落としただけなので、どのワードも型を問わない式と
// ビットまで同じになる。行の外の位置のワードは行の最後のトークンで作り、そのトークンは EVENT のタイルではイベントなので、EVENT の式で
// 作ってよい。
//
// バイアスの表（relationBiasKernel）: 重みをデバイスへ写すときに、層ごとに 1 回作る（Plan::uploadWeights）。1 つのワークグループが
// 1 つのヘッドの対を受け持ち、K2・K4 が引く 2 ヘッド分の表（置き方は prism_hip_relation.h）を ρ から作る。
#include "prism_hip_relation.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
static_assert(kThreads == kCodeBlockWords, "one thread per code word of a block pair");

// タイル b（位置 16b〜16b + 15）の型。fixed は局・席・牌のトークンの数 F（位置 0〜F − 1）。
__device__ __forceinline__ BlockKind blockKind(int block, int fixed) {
  return 16 * block + 16 <= fixed ? BlockKind::FIXED : 16 * block >= fixed ? BlockKind::EVENT : BlockKind::MIXED;
}

// クエリの位置 i とキーの位置 j の対のワードを返す（i は型が Query のタイル、j は型が Key のタイルの中）。query は位置
// min(i, S − 1) のトークンの記述、tileRow は牌の関係の表でのその牌の行の先頭。FIXED のタイルは行の中にある（16b + 15 < F ≤ S）ので、
// 行の外の選択と位置の切り詰めを省く。
template <BlockKind Query, BlockKind Key>
__device__ __forceinline__ uint32_t pairCode(const RelationToken* token, const uint8_t* tileRelation,
                                             const RelationToken& query, int tileRow, int i, int j, int tokens,
                                             int seats, Bounds& bounds) {
  const bool queryInside = Query == BlockKind::FIXED || i < tokens;
  const bool keyInside = Key == BlockKind::FIXED || j < tokens;
  const RelationToken key = token[Key == BlockKind::FIXED ? j : min(j, tokens - 1)];
  int segments[kRelationSegments];
  relationSegments<Query, Key>(query, key, tileRelation[tileRow + key.tileIndex()], seats, segments);
  // セグメントのコードは 16 未満（推論のワードが指す項の番号が、組の表に収まる）。セグメント 6 の 15 は行の外のキーの印なので、
  // 実在する対では 14 以下（範囲外は、検査つきのビルドで止める）。
  const bool inside = queryInside && keyInside;
#pragma unroll
  for (int s = 0; s < kRelationSegments; ++s) {
    (void)bounds.index(static_cast<size_t>(inside ? segments[s] : 0), 1,
                       s == kRelationSegments - 1 ? kRelationEntries - 1 : kRelationEntries);
  }
  return !keyInside ? kPaddedKeyCode : queryInside ? attentionCode(segments) : 0u;
}

// クエリのタイル qt（型 Query）と、全てのキーのタイルとの対のワードを書く。キーのタイルの型は kt だけで決まるので、スカラーの分岐で
// 選ぶ。
template <BlockKind Query>
__device__ __forceinline__ void writeQueryBlock(const RelationArgs& args, const RelationToken* token,
                                                const uint8_t* tileRelation, int row, int qt, int blocks, int tokens,
                                                Bounds& bounds) {
  const int tid = threadIdx.x;
  const int lane = tid >> 3;
  const int v = tid & 7;
  const int fixed = args.layout.fixed();
  const int seats = args.layout.seats;
  const int i = 16 * qt + (lane & 15);
  const RelationToken query = token[Query == BlockKind::FIXED ? i : min(i, tokens - 1)];
  const int tileRow = query.tileIndex() * args.layout.tiles;
  for (int kt = 0; kt < blocks; ++kt) {
    const int j = 16 * kt + 2 * v + (lane >> 4);
    const BlockKind keyKind = blockKind(kt, fixed);
    const uint32_t code =
        keyKind == BlockKind::FIXED
            ? pairCode<Query, BlockKind::FIXED>(token, tileRelation, query, tileRow, i, j, tokens, seats, bounds)
        : keyKind == BlockKind::EVENT
            ? pairCode<Query, BlockKind::EVENT>(token, tileRelation, query, tileRow, i, j, tokens, seats, bounds)
            : pairCode<Query, BlockKind::MIXED>(token, tileRelation, query, tileRow, i, j, tokens, seats, bounds);
    args.codes.data[bounds.index(codeIndex(row, qt, kt, args.codeBlocks) + tid, 1, args.codes.limit)] = code;
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
  // 行のトークンの数は、ルールの上限により、行のコードのタイル全体（16·codeBlocks トークン）以下になる。範囲外は検査つきのビルドで
  // 止め、通常のビルドでも LDS と行のコードの外には書かない。
  const int rowLimit = min(16 * args.codeBlocks, kMaxRowTokens);
  const int tokens = min(fixed + max(events, 0), rowLimit);
  (void)bounds.index(static_cast<size_t>(fixed + events), 0, rowLimit);

  // 行の局のトークン（位置 0）が持つイベントの番号。
  const int now = args.relation.data[bounds.index(
      static_cast<size_t>(layout.index(row, firstEvent, 0)) * columns.width + columns.event, 1, args.relation.limit)];
  for (int p = tid; p < tokens; p += kThreads) {
    const size_t at = static_cast<size_t>(layout.index(row, firstEvent, p)) * columns.width;
    uint8_t attributes[kMaxAttributes];
#pragma unroll
    for (int c = 0; c < kMaxAttributes; ++c) {
      attributes[c] = c < columns.width ? args.relation.data[bounds.index(at + c, 1, args.relation.limit)] : 0;
    }
    // 持ち主は 0（なし）か 1〜S（relationSegments は、席のセグメントでこの範囲を前提にする）。
    (void)bounds.index(attributes[columns.owner], 1, static_cast<size_t>(layout.seats) + 1);
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

// バイアスの表の、1 つのヘッドの対の分を作る。項 e ごとに (ヘッド 2p, ヘッド 2p + 1) の値を書く（×log2e）。組の表 τ の項
// pairTableEntry(a, b) は ρ[32τ + a] + ρ[32τ + 16 + b]、セグメント 6 の表の項 a は ρ[96 + a]（15 は行の外のキーで −∞）。
// ρ は [関係の行, ヘッド]。
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
  if (args.layout.seats < 1 || 3 * args.layout.seats >= kRelationEntries) {
    throw std::invalid_argument("K0 codes the seat segment in 4 bits, which needs 3 * seats < 16, got " +
                                std::to_string(args.layout.seats) + " seats");
  }
  if (args.codeBlocks <= 0 || 16 * args.codeBlocks < args.layout.fixed() || args.codeBlocks > kMaxRowTokens / 16) {
    throw std::invalid_argument("K0 got " + std::to_string(args.codeBlocks) + " code blocks for " +
                                std::to_string(args.layout.fixed()) + " fixed tokens");
  }
  hipLaunchKernelGGL(relationKernel, dim3(args.layout.rows), dim3(kThreads), 0, stream, args);
  PRISM_HIP(hipGetLastError());
}

void launchRelationBias(const RelationBiasArgs& args, int headPairs, hipStream_t stream) {
  hipLaunchKernelGGL(relationBiasKernel, dim3(headPairs), dim3(kThreads), 0, stream, args);
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
