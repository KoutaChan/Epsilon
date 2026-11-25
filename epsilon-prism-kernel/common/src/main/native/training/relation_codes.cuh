#pragma once

namespace prism {
namespace {

// 学習用の関係のコード。GPU で学習するときの順伝播で、eager の RelationClass の代わりに使う。式は推論と同じ relationSegments
// （relation_encoding.h）で、値は RelationClass とビットまで同じ:
//
//   行 r の全ての対（クエリ i, キー j。0 ≤ i, j < S）について、7 セグメントのコードを 1 ワードに詰め（セグメント s を bit 4s から）、
//   [行, S, S] の行優先で書く。これは RelationClass の [行, S, S, 2] の INT16（下位のワードにセグメント 0〜3、上位のワードにセグメント
//   4〜6）と同じバイト列になる。
//   パディングの位置（行の局・席・牌とイベントより後ろ）は、RelationClass が散らした属性と同じく属性を全て 0 にし、型は位置から決まる
//   イベントの型にして、同じ式で作る（推論で使う行の外の印 kPaddedKeyCode や 0 は使わない）。パディングのキーは Attention のマスクが外す。
//
// ワークグループは 1 行を受け持つ（256 スレッド）。行の全ての位置（パディングは属性が全て 0）のトークンのワード（RelationToken）と、
// 牌の関係の表を共有メモリに置く。スレッド t は、行の中の対の通し番号 t, t + 256, … を受け持つ（グローバルメモリへ連続して書く）。
constexpr int kThreads = 256;
// 牌の関係のクラスの数の上限（セグメント 1 は c + 1 ≤ 15、セグメント 2 は c − 14 ≤ 15 なので c ≤ 29）。
constexpr int kTileRelationClasses = 2 * kRelationEntries - 2;

// 関係コードの入力と出力: 形の数と、検査つきのビルドで範囲を確かめる上限つきの配列。
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
  const int firstEvent = args.rowColumn.data[bounds.index(rowAt + args.shape.rowFirstEvent, 1, args.rowColumn.limit)];
  const int events = args.rowColumn.data[bounds.index(rowAt + args.shape.rowEvents, 1, args.rowColumn.limit)];
  // 行のトークンは S 以下（超えたら検査つきのビルドで止める）。通常のビルドでも共有メモリの外には書かない。
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
#if PRISM_CHECKED
    checkRelationTile(token[p], layout.tiles, bounds);
#endif
  }
  for (int e = tid; e < layout.tiles * layout.tiles; e += kThreads) {
    const int relation = args.tileRelation.data[bounds.index(e, 1, args.tileRelation.limit)];
    (void)bounds.index(static_cast<size_t>(relation), 1, kTileRelationClasses);
    tileRelation[e] = static_cast<uint8_t>(relation);
  }
  shareGroupWrites();

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
}  // namespace prism
