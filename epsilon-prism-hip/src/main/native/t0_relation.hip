// 学習用の関係のコード（T0）。GPU で学習するときの順伝播で、eager の RelationClass の代わりに使う。式は K0 と同じ relationSegments
// （prism_hip_relation.h）で、値は RelationClass とビットまで同じ:
//
//   行 r の全ての対（クエリ i, キー j。0 ≤ i, j < S）について、7 セグメントのコードを 1 ワードに詰め（セグメント s を bit 4s から）、
//   [行, S, S] の行優先で書く。これは RelationClass の [行, S, S, 2] の INT16（下位のワードにセグメント 0〜3、上位のワードにセグメント
//   4〜6）と同じバイト列になる。
//   パディングの位置（行の局・席・牌とイベントより後ろ）は、RelationClass が散らした属性と同じく属性を全て 0 にし、型は位置から決まる
//   イベントの型にして、同じ式で作る（K0 の行の外の印 kPaddedKeyCode や 0 は使わない）。パディングのキーは Attention のマスクが外す。
//
// ワークグループは 1 行を受け持つ（256 スレッド）。行の全ての位置（パディングは属性が全て 0）のトークンのワード（RelationToken）と、
// 牌の関係の表を LDS に置く。スレッド t は、行の中の対の通し番号 t, t + 256, … を受け持つ（グローバルメモリへ連続して書く）。
#include <stdexcept>
#include <string>

#include "prism_hip_relation.h"
#include "prism_hip_training.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
// 牌の関係のクラスの数の上限（セグメント 1 は c + 1 ≤ 15、セグメント 2 は c − 14 ≤ 15 なので c ≤ 29）。
constexpr int kTileRelationClasses = 2 * kRelationEntries - 2;

// T0 の引数: 形の数と、検査つきのビルドで範囲を確かめる上限つきの配列。
struct RelationCodeArgs {
  RelationCodeShape shape;
  Span<const uint8_t> relation;
  Span<const int32_t> rowColumn;
  Span<const int32_t> tileRelation;
  Span<uint32_t> codes;
};

__global__ __launch_bounds__(kThreads) void relationCodeKernel(RelationCodeArgs args) {
  __shared__ RelationToken token[kMaxRowTokens];
  __shared__ uint8_t tileRelation[kMaxTiles * kMaxTiles];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const TokenLayout layout = args.shape.layout;
  const RelationColumns columns = args.shape.columns;
  const int length = args.shape.tokens;  // S
  Bounds bounds;

  const size_t rowAt = static_cast<size_t>(row) * args.shape.rowWidth;
  const int firstEvent =
      args.rowColumn.data[bounds.index(rowAt + args.shape.rowFirstEvent, 1, args.rowColumn.limit)];
  const int events = args.rowColumn.data[bounds.index(rowAt + args.shape.rowEvents, 1, args.rowColumn.limit)];
  // 行のトークンは S 以下（超えたら検査つきのビルドで止める）。通常のビルドでも LDS の外には書かない。
  const int tokens = min(layout.fixed() + max(events, 0), length);
  (void)bounds.index(static_cast<size_t>(layout.fixed() + events), 0, length);

  // 行の局のトークン（位置 0）の番号。
  const int now = args.relation.data[bounds.index(
      static_cast<size_t>(layout.index(row, firstEvent, 0)) * columns.width + columns.event, 1, args.relation.limit)];
  for (int p = tid; p < length; p += kThreads) {
    const size_t at = static_cast<size_t>(layout.index(row, firstEvent, p)) * columns.width;
    uint8_t attributes[kMaxAttributes];
#pragma unroll
    for (int c = 0; c < kMaxAttributes; ++c) {
      attributes[c] =
          p < tokens && c < columns.width ? args.relation.data[bounds.index(at + c, 1, args.relation.limit)] : 0;
    }
    // 持ち主は 0（なし）か 1〜S（relationSegments が席のセグメントで前提にする範囲）。
    (void)bounds.index(attributes[columns.owner], 1, static_cast<size_t>(layout.seats) + 1);
    token[p] = relationToken(layout.type(p), attributes, now, columns);
  }
  for (int e = tid; e < layout.tiles * layout.tiles; e += kThreads) {
    const int relation = args.tileRelation.data[bounds.index(e, 1, args.tileRelation.limit)];
    (void)bounds.index(static_cast<size_t>(relation), 1, kTileRelationClasses);
    tileRelation[e] = static_cast<uint8_t>(relation);
  }
  __syncthreads();

  const size_t first = static_cast<size_t>(row) * length * length;
  for (int pair = tid; pair < length * length; pair += kThreads) {
    const int i = pair / length;
    const int j = pair - i * length;
    const RelationToken query = token[i];
    const RelationToken key = token[j];
    int segments[kRelationSegments];
    relationSegments<BlockKind::MIXED, BlockKind::MIXED>(
        query, key, tileRelation[query.tileIndex() * layout.tiles + key.tileIndex()], layout.seats, segments);
    // セグメントのコードは 4 bit に収まる（収まらなければ検査つきのビルドで止める）ので、packCode の論理和は RelationClass の
    // セグメントごとの和と同じになる。
#pragma unroll
    for (int s = 0; s < kRelationSegments; ++s) {
      (void)bounds.index(static_cast<size_t>(segments[s]), 1, kRelationEntries);
    }
    args.codes.data[bounds.index(first + pair, 1, args.codes.limit)] = packCode(segments);
  }
  bounds.trap();
}

}  // namespace

void checkRelationCodeShape(const RelationCodeShape& shape) {
  auto require = [](bool condition, const std::string& message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  const TokenLayout& layout = shape.layout;
  const RelationColumns& c = shape.columns;
  require(layout.rows > 0, "relation codes need rows, got " + std::to_string(layout.rows));
  require(layout.seats > 0 && 3 * layout.seats < kRelationEntries && layout.tiles > 0 && layout.tiles <= kMaxTiles,
          "seats " + std::to_string(layout.seats) + " and tiles " + std::to_string(layout.tiles) + " are out of range");
  require(shape.events >= 0, "event count must not be negative, got " + std::to_string(shape.events));
  require(shape.tokens >= layout.fixed() && shape.tokens <= kMaxRowTokens,
          "row tokens must be in " + std::to_string(layout.fixed()) + ".." + std::to_string(kMaxRowTokens) + ", got " +
              std::to_string(shape.tokens));
  require(shape.rowWidth > 0 && shape.rowFirstEvent >= 0 && shape.rowFirstEvent < shape.rowWidth &&
              shape.rowEvents >= 0 && shape.rowEvents < shape.rowWidth,
          "row column out of range");
  require(c.width > 0 && c.width <= kMaxAttributes,
          "relation attributes must be in 1.." + std::to_string(kMaxAttributes) + " columns, got " +
              std::to_string(c.width));
  for (int column : {c.tile, c.owner, c.event, c.riichi, c.lastDiscard, c.link}) {
    require(column >= 0 && column < c.width, "relation column out of range");
  }
  require(c.safety >= 0 && c.safeties >= 0 && c.safety + c.safeties <= c.width, "relation safety columns out of range");
  require(c.safeties <= kRelationOpponents, "relation codes hold the safety of at most " +
                                                std::to_string(kRelationOpponents) + " opponents, got " +
                                                std::to_string(c.safeties) + " safety columns");
}

void relationCodes(const RelationCodeShape& shape, const RelationCodeArrays& arrays, hipStream_t stream) {
  checkRelationCodeShape(shape);
  if (reinterpret_cast<uintptr_t>(arrays.codes) % alignof(uint32_t) != 0) {
    throw std::invalid_argument("relation codes must be aligned to 4 bytes");
  }
  const size_t rows = static_cast<size_t>(shape.layout.rows);
  const size_t tokens = static_cast<size_t>(shape.tokens);
  RelationCodeArgs args{};
  args.shape = shape;
  args.relation = {arrays.relation, (rows * shape.layout.fixed() + shape.events) * shape.columns.width};
  args.rowColumn = {arrays.rowColumn, rows * shape.rowWidth};
  args.tileRelation = {arrays.tileRelation, static_cast<size_t>(shape.layout.tiles) * shape.layout.tiles};
  args.codes = {arrays.codes, rows * tokens * tokens};
  hipLaunchKernelGGL(relationCodeKernel, dim3(shape.layout.rows), dim3(kThreads), 0, stream, args);
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
