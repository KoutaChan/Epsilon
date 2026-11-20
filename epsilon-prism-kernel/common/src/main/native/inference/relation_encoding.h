// 関係のバイアスのコード。eager の RelationClass と同じ式で、トークンの対（クエリ i, キー j）ごとに整数の演算だけで作る。
// セグメント s のコード c は、ρ の行 16s + c を引く。属性の値は入力のまま（0 はなし、番号は番号 + 1）。型は 0 が局、1 が席、2 が牌、
// 3 がイベント。
//
//   0 型の対:       4·型(i) + 型(j)
//   1・2 牌:        両方に牌があり、MahjongTileRelation のクラス c が 15 未満ならセグメント 1 に c + 1、15 以上ならセグメント 2 に c − 14
//   3 席:           両方に持ち主がいれば 1 + (持ち主(j) − 持ち主(i) + S) mod S、j だけなら S + 持ち主(j)、i だけなら 2S + 持ち主(i)
//                   （S は席の数）
//   4 立直:         キーがイベント: 1 + 前後(番号(j), 立直(j)) + 4·[最後の打牌(i) > 0 かつ 番号(j) > 最後の打牌(i)]
//                   クエリがイベントでキーが席: 9 + 前後(番号(i), 立直(j))。前後は、立直なし 0、宣言より前 1、宣言 2、宣言より後 3
//   5 時間:         両方がイベント: 5 + 符号(δ)·段(|δ|)（δ = 番号(j) − 番号(i)。段は 1・2・4・8 のうち |δ| 以下のものの数）
//                   クエリがイベントでなくキーがイベント: 10 + [d ≥ 5] + [d ≥ 9] + [d ≥ 17]（d = 番号(局) − 番号(j)）
//   6 繋がり・安全: 両方がイベント: i の繋がりが j なら 1、そうでなく j の繋がりが i なら 2
//                   席 → 牌で席が相手: 3 + その相手から見た牌の安全度、牌 → 席で席が相手: 7 + その相手から見た牌の安全度
//
// 推論と学習では同じ式で関係のコードを求める。どちらも、行のトークンごとに属性を詰めた 3 ワード（RelationToken）を
// 共有メモリに 1 回作る。対のコードは、2 つのトークンのワードと牌の関係の表の 1 バイトから、選択だけで作る（relationSegments。分岐は
// 書かない）。キーの側だけで決まる前後(番号(j), 立直(j)) と時間の段は、トークンのワードに入れておく。
//
// トークン対のコードは用途によって並びが異なる。学習では eager の RelationClass と同じワード（packCode: セグメント s を bit 4s から置く。eager の 2 ワードの
// low・high と同じ並び）を書き、Attention に渡す。推論では、行ごとに全ての対のコードを推論のワード（attentionCode: セグメントを
// 2 つずつ、バイアスの表の項のバイト位置に直した並び）に詰め、各層の全ヘッドが共有する。セグメント 6 は 10 以下なので、15 を
// 行の外のキーの印にする（全トークンの層と固定クエリの層のバイアスの表で −∞）。
//
// バイアスの表は ρ だけで決まる。そこで、重みを GPU に送るとき（Plan::uploadWeights）に関係のコードの隣の relationBiasKernel が層ごとに 1 回作り、
// 全トークンの層と固定クエリの層のワークグループは自分のヘッドの対の分を共有メモリへ写すだけにする。
#pragma once

#include "collection_capabilities.h"

#include "inference_kernels.h"
#include "relation_terms.h"

namespace prism {

// コードの置き方: 行ごとに、16 トークンのタイルの対 (qt, kt) ごとに kCodeBlockWords ワードずつ置く。1 行のタイルの数 blocks は、
// 形の記述の Architecture::codeBlocks。

// 行 row のタイルの対 (qt, kt) の先頭のワード。ワード t は、lane = t >> 3、v = t & 7 として、i = 16qt + (lane & 15)、
// j = 16kt + 2v + (lane >> 4) の対のコード（全トークンの層と固定クエリの層のタイル積和の S^T のフラグメントで、lane が持つ 8 つのキーの順）。
__host__ __device__ __forceinline__ size_t codeIndex(int row, int qt, int kt, int blocks) {
  return ((static_cast<size_t>(row) * blocks + qt) * blocks + kt) * kCodeBlockWords;
}

// バイアスの表の置き方: 1 層は [ヘッドの対, kBiasEntries, 2] の FP32 で、項は (ヘッド 2p, ヘッド 2p + 1) の値の対（exp2 で softmax を
// 計算するので ×log2e）。セグメントを 2 つずつ先に足した組の表 τ（< 3）の項 pairTableEntry(a, b) は ρ[32τ + a] + ρ[32τ + 16 + b]
// （a はセグメント 2τ、b はセグメント 2τ + 1 のコード）。セグメント 6 の表の項 a は ρ[96 + a]（15 は行の外のキーで −∞）。全トークンの層と固定クエリの層は、
// 推論のワードの 4 つのフィールドで引いて足す。
constexpr int kPairTables = 3;
constexpr int kPairEntries = kRelationEntries * kRelationEntries;
constexpr int kBiasEntries = kPairTables * kPairEntries + kRelationEntries;
constexpr float kLog2e = 1.4426950408889634f;
static_assert(2 * kPairTables + 1 == kRelationSegments,
              "the pair tables cover segments 0..5 and the last table segment 6");

// 推論のワードの置き方: 組の表 τ（< 3）のフィールドは bit 3 + 8τ から 8 bit、セグメント 6 のフィールドは bit 27 から 4 bit（bit 0〜2 と
// 31 は 0）。バイアスの表の項は 8 バイト（2 ヘッドの FP32）で、どのフィールドも表の中の項のバイト位置を持つ。そのため全トークンの層と固定クエリの層のアドレス
// 計算は、表 0 がワード & 0x7F8 の 1 命令、ほかの表は shift と and の 2 命令で済む。
constexpr int kBiasEntryShift = 3;  // log2(バイアスの表の項のバイト数)
constexpr uint32_t kBiasEntryBytes = 1u << kBiasEntryShift;
constexpr int kLastTableShift = 8 * kPairTables + kBiasEntryShift;  // セグメント 6 のフィールドの最下位の bit
constexpr uint32_t kPaddedKeyCode = (kRelationEntries - 1u)
                                    << kLastTableShift;  // セグメント 6 が 15、ほかが 0: 行の外のキー
static_assert(kLastTableShift + 4 < 32, "the last field must leave bit 31 clear");

// 推論のワード code で組の表 table（< 3）を引くときの、項のバイト位置（表の先頭から）。
__host__ __device__ constexpr uint32_t pairTableOffset(uint32_t code, int table) {
  return (code >> (8 * table)) & ((kPairEntries - 1u) << kBiasEntryShift);
}

// 推論のワード code でセグメント 6 の表を引くときの、項のバイト位置（表の先頭から）。
__host__ __device__ constexpr uint32_t lastTableOffset(uint32_t code) {
  return (code >> (8 * kPairTables)) & ((kRelationEntries - 1u) << kBiasEntryShift);
}

// 組の表で、セグメントの対 (a, b)（どちらも < 16）が引く項 (a ^ b) + 16b（16 × 16 の全単射）。表の 8 バイトの項 e が載る共有メモリのバンクは
// e mod 16 で決まる（全トークンの層の 8 バイトの読み込みは 2 バンクの対、固定クエリの層の 4 バイトの読み込みはヘッドごとに 1 バンク）。1 回の読み込みの 16 lane は同じキー・違うクエリで、
// セグメント 2τ（型の対・牌の関係の後半・立直）はほぼそろい、セグメント 2τ + 1 がばらつく。そこで下の 4 bit を a ^ b にして、b の違いを
// バンクの違いにする（a + 16b の並びでは、違う項が同じバンクに重なる）。
__host__ __device__ constexpr int pairTableEntry(int a, int b) { return (a ^ b) | b << 4; }

// 組の表の項 e（< 256）を引くセグメントの対 (a, b)（pairTableEntry の逆）。
__host__ __device__ constexpr int pairTableFirst(int e) { return (e ^ (e >> 4)) & 15; }
__host__ __device__ constexpr int pairTableSecond(int e) { return e >> 4; }

// 1 層のバイアスの表の FP32 の数。
__host__ __device__ __forceinline__ size_t relationBiasFloats(int headPairs) {
  return static_cast<size_t>(headPairs) * kBiasEntries * 2;
}

#if PRISM_CHECKED
// 範囲外の牌を検出したら、検査結果を通知するまで牌の関係表の先頭を参照する。
__device__ __forceinline__ void checkRelationTile(RelationToken& token, int tiles, Bounds& bounds) {
  constexpr uint32_t tileMask = 0xffu;
  const auto tile = bounds.index(token.a & tileMask, 1, static_cast<size_t>(tiles) + 1);
  token.a = (token.a & ~tileMask) | static_cast<uint32_t>(tile);
}
#endif

// 推論のワード（関係のコードが書き、全トークンの層と固定クエリの層が読む。置き方は上に書いた）: 組の表 τ（< 3）のフィールドに項 pairTableEntry(セグメント 2τ,
// セグメント 2τ + 1) を、セグメント 6 のフィールドにそのコードを、どれも項のバイト位置で置く（セグメントのコードはどれも 16 未満）。
__host__ __device__ constexpr uint32_t attentionCode(const int (&segments)[kRelationSegments]) {
  uint32_t code = static_cast<uint32_t>(segments[kRelationSegments - 1]) << kLastTableShift;
#pragma unroll
  for (int table = 0; table < kPairTables; ++table) {
    code |= static_cast<uint32_t>(pairTableEntry(segments[2 * table], segments[2 * table + 1]))
            << (8 * table + kBiasEntryShift);
  }
  return code;
}

// 推論のワード code を全トークンの層と固定クエリの層と同じアドレスの式（pairTableOffset・lastTableOffset）で引いた 4 つの項が、セグメントのコード segments
// から RelationClass が引くのと同じ ρ の行を指すか（組の表 τ の項は行 32τ + a と 32τ + 16 + b、セグメント 6 の表の項は行 96 + c）。
constexpr bool attentionCodeAddresses(uint32_t code, const int (&segments)[kRelationSegments]) {
  for (int table = 0; table < kPairTables; ++table) {
    const int e = static_cast<int>(pairTableOffset(code, table) / kBiasEntryBytes);
    if (pairTableFirst(e) != segments[2 * table] || pairTableSecond(e) != segments[2 * table + 1]) {
      return false;
    }
  }
  return static_cast<int>(lastTableOffset(code) / kBiasEntryBytes) == segments[kRelationSegments - 1];
}

// セグメントのコードのどの組でも attentionCodeAddresses が成り立つこと。セグメントはワードの別々のビットに入るので、引くセグメントの
// 対とコードを全て回し、ほかのセグメントを 0 と 15 にして漏れがないことを見れば、全てのワードを確かめたことになる。行の外のキーの印は、
// セグメント 6 の 15 と組の表の (0, 0) を指す。
constexpr bool attentionCodeMatchesRelationRows() {
  for (int other = 0; other < kRelationEntries; other += kRelationEntries - 1) {
    for (int table = 0; table < kPairTables; ++table) {
      for (int a = 0; a < kRelationEntries; ++a) {
        for (int b = 0; b < kRelationEntries; ++b) {
          int segments[kRelationSegments] = {other, other, other, other, other, other, other};
          segments[2 * table] = a;
          segments[2 * table + 1] = b;
          if (!attentionCodeAddresses(attentionCode(segments), segments)) {
            return false;
          }
        }
      }
    }
    for (int c = 0; c < kRelationEntries; ++c) {
      const int segments[kRelationSegments] = {other, other, other, other, other, other, c};
      if (!attentionCodeAddresses(attentionCode(segments), segments)) {
        return false;
      }
    }
  }
  const int padded[kRelationSegments] = {0, 0, 0, 0, 0, 0, kRelationEntries - 1};
  return attentionCodeAddresses(kPaddedKeyCode, padded);
}
static_assert(attentionCodeMatchesRelationRows(),
              "the attention code must address the relation rows of every segment code");

}  // namespace prism
