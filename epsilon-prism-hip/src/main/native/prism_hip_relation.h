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
// 式の実装はここ 1 か所だけで、K0（推論）と T0（学習）が使う。どちらも、行のトークンごとに属性を詰めた 3 ワード（RelationToken）を
// LDS に 1 回作る。対のコードは、2 つのトークンのワードと牌の関係の表の 1 バイトから、選択だけで作る（relationSegments。分岐は
// 書かない）。キーの側だけで決まる前後(番号(j), 立直(j)) と時間の段は、トークンのワードに入れておく。
//
// 対のコードのワードは 2 通りある。T0 は eager の RelationClass と同じワード（packCode: セグメント s を bit 4s から置く。eager の 2 ワードの
// low・high と同じ並び）を書き、学習の Attention（T1）が読む。K0 は、行ごとに全ての対のコードを推論のワード（attentionCode: セグメントを
// 2 つずつ、バイアスの表の項のバイト位置に直した並び）に詰め、K2・K4 が全てのヘッドの対と層で読む。セグメント 6 は 10 以下なので、15 を
// 行の外のキーの印にする（K2・K4 のバイアスの表で −∞）。
//
// バイアスの表は ρ だけで決まる。そこで、重みを GPU に送るとき（Plan::uploadWeights）に K0 の隣の relationBiasKernel が層ごとに 1 回作り、
// K2・K4 のワークグループは自分のヘッドの対の分を LDS へ写すだけにする。
#pragma once

#include "prism_hip_kernels.h"

namespace prism {

constexpr int kRelationSegments = 7;
constexpr int kRelationEntries = 16;
constexpr int kTileRelationFirstHalf = kRelationEntries - 1;
constexpr int kMaxAttributes = 12;  // 関係の属性の列の上限（K0・T0 はトークンの属性の行をレジスタに読む）
constexpr int kMaxTiles = 34;       // 牌の種類の上限（K0・T0 の LDS の牌の関係の表）

// コードの置き方: 行ごとに、16 トークンのタイルの対 (qt, kt) ごとに kCodeBlockWords ワードずつ置く。1 行のタイルの数 blocks は、
// 形の記述の Architecture::codeBlocks（Java の HipCounts.codeBlocks）。
constexpr int kCodeBlockWords = 256;

// 行 row のタイルの対 (qt, kt) の先頭のワード。ワード t は、lane = t >> 3、v = t & 7 として、i = 16qt + (lane & 15)、
// j = 16kt + 2v + (lane >> 4) の対のコード（K2・K4 の WMMA の S^T のフラグメントで、lane が持つ 8 つのキーの順）。
__host__ __device__ __forceinline__ size_t codeIndex(int row, int qt, int kt, int blocks) {
  return ((static_cast<size_t>(row) * blocks + qt) * blocks + kt) * kCodeBlockWords;
}

// バイアスの表の置き方: 1 層は [ヘッドの対, kBiasEntries, 2] の FP32 で、項は (ヘッド 2p, ヘッド 2p + 1) の値の対（exp2 で softmax を
// 計算するので ×log2e）。セグメントを 2 つずつ先に足した組の表 τ（< 3）の項 pairTableEntry(a, b) は ρ[32τ + a] + ρ[32τ + 16 + b]
// （a はセグメント 2τ、b はセグメント 2τ + 1 のコード）。セグメント 6 の表の項 a は ρ[96 + a]（15 は行の外のキーで −∞）。K2・K4 は、
// 推論のワードの 4 つのフィールドで引いて足す。
constexpr int kPairTables = 3;
constexpr int kPairEntries = kRelationEntries * kRelationEntries;
constexpr int kBiasEntries = kPairTables * kPairEntries + kRelationEntries;
constexpr float kLog2e = 1.4426950408889634f;
static_assert(2 * kPairTables + 1 == kRelationSegments, "the pair tables cover segments 0..5 and the last table segment 6");

// 推論のワードの置き方: 組の表 τ（< 3）のフィールドは bit 3 + 8τ から 8 bit、セグメント 6 のフィールドは bit 27 から 4 bit（bit 0〜2 と
// 31 は 0）。バイアスの表の項は 8 バイト（2 ヘッドの FP32）で、どのフィールドも表の中の項のバイト位置を持つ。そのため K2・K4 のアドレス
// 計算は、表 0 がワード & 0x7F8 の 1 命令、ほかの表は shift と and の 2 命令で済む。
constexpr int kBiasEntryShift = 3;  // log2(バイアスの表の項のバイト数)
constexpr uint32_t kBiasEntryBytes = 1u << kBiasEntryShift;
constexpr int kLastTableShift = 8 * kPairTables + kBiasEntryShift;  // セグメント 6 のフィールドの最下位の bit
constexpr uint32_t kPaddedKeyCode = (kRelationEntries - 1u) << kLastTableShift;  // セグメント 6 が 15、ほかが 0: 行の外のキー
static_assert(kLastTableShift + 4 < 32, "the last field must leave bit 31 clear");

// 推論のワード code で組の表 table（< 3）を引くときの、項のバイト位置（表の先頭から）。
__host__ __device__ constexpr uint32_t pairTableOffset(uint32_t code, int table) {
  return (code >> (8 * table)) & ((kPairEntries - 1u) << kBiasEntryShift);
}

// 推論のワード code でセグメント 6 の表を引くときの、項のバイト位置（表の先頭から）。
__host__ __device__ constexpr uint32_t lastTableOffset(uint32_t code) {
  return (code >> (8 * kPairTables)) & ((kRelationEntries - 1u) << kBiasEntryShift);
}

// 組の表で、セグメントの対 (a, b)（どちらも < 16）が引く項 (a ^ b) + 16b（16 × 16 の全単射）。表の 8 バイトの項 e が載る LDS のバンクは
// e mod 16 で決まる（K2 の b64 は 2 バンクの対、K4 の b32 はヘッドごとに 1 バンク）。1 回の読み込みの 16 lane は同じキー・違うクエリで、
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

// 立直との前後: 立直なし 0、宣言より前 1、宣言 2、宣言より後 3。
__device__ __forceinline__ int riichiPhase(int number, int riichi) {
  return riichi > 0 ? (number >= riichi) + (number > riichi) + 1 : 0;
}

// 相手の数の上限（持ち主の ID 2〜4）。トークンのワード c は相手ごとに安全度を 1 バイト持つので、安全度の列はこれ以下にする（K0・T0 を
// 起動する関数で確かめる）。
constexpr int kRelationOpponents = 3;

// 1 トークンの属性を詰めた 3 ワード（行ごとに 1 回作る。フィールドは属性のバイトそのままか、キーの側だけで決まる値）:
//   a: 牌（bit 0〜7）、持ち主（8〜15）、型（16〜17）、前後(番号, 立直)（18〜19）、時間の段 [d ≥ 5] + [d ≥ 9] + [d ≥ 17]（20〜21、
//      d = 番号(局) − 番号）
//   b: 番号（0〜7）、最後の打牌（8〜15）、立直（16〜23）、繋がり（24〜31）
//   c: 相手 o（持ち主 2 + o）から見た牌の安全度をバイト o に置く（安全度の列のない相手は 0）
struct RelationToken {
  uint32_t a;
  uint32_t b;
  uint32_t c;

  __device__ __forceinline__ int tile() const { return static_cast<int>(a & 0xffu); }
  __device__ __forceinline__ int owner() const { return static_cast<int>((a >> 8) & 0xffu); }
  __device__ __forceinline__ int type() const { return static_cast<int>((a >> 16) & 3u); }
  __device__ __forceinline__ int phase() const { return static_cast<int>((a >> 18) & 3u); }
  __device__ __forceinline__ int ageStep() const { return static_cast<int>((a >> 20) & 3u); }
  __device__ __forceinline__ int event() const { return static_cast<int>(b & 0xffu); }
  __device__ __forceinline__ int lastDiscard() const { return static_cast<int>((b >> 8) & 0xffu); }
  __device__ __forceinline__ int riichi() const { return static_cast<int>((b >> 16) & 0xffu); }
  __device__ __forceinline__ int link() const { return static_cast<int>(b >> 24); }

  // 牌の関係の表の行・列の番号（牌がなければ 0 を引くが、そのクラスはセグメント 1・2 の式が捨てる）。
  __device__ __forceinline__ int tileIndex() const { return max(tile() - 1, 0); }

  // 席 owner から見たこの牌の安全度（owner が相手でなければ 0）。
  __device__ __forceinline__ int safetyFrom(int owner) const {
    const int opponent = owner - 2;
    return opponent >= 0 && opponent < kRelationOpponents ? static_cast<int>((c >> (8 * opponent)) & 0xffu) : 0;
  }
};

// トークンのワードを作る。attributes はそのトークンの属性の行（列は c のとおり。パディングの位置は全て 0）、type は位置の型、now は
// 行の局のトークンの番号。
__device__ __forceinline__ RelationToken relationToken(int type, const uint8_t* attributes, int now,
                                                       const RelationColumns& c) {
  const int event = attributes[c.event];
  const int riichi = attributes[c.riichi];
  const int age = now - event;
  const uint32_t ageStep = (age >= 5) + (age >= 9) + (age >= 17);
  uint32_t safety = 0;
#pragma unroll
  for (int opponent = 0; opponent < kRelationOpponents; ++opponent) {
    if (opponent < c.safeties) {
      safety |= static_cast<uint32_t>(attributes[c.safety + opponent]) << (8 * opponent);
    }
  }
  RelationToken token;
  token.a = static_cast<uint32_t>(attributes[c.tile]) | static_cast<uint32_t>(attributes[c.owner]) << 8 |
            static_cast<uint32_t>(type) << 16 | static_cast<uint32_t>(riichiPhase(event, riichi)) << 18 | ageStep << 20;
  token.b = static_cast<uint32_t>(event) | static_cast<uint32_t>(attributes[c.lastDiscard]) << 8 |
            static_cast<uint32_t>(riichi) << 16 | static_cast<uint32_t>(attributes[c.link]) << 24;
  token.c = safety;
  return token;
}

// イベントのトークンの型。
constexpr int kEventType = 3;

// トークンのタイル（16 の位置）の中の型のそろい方。型は位置だけで決まる（局・席・牌は位置 0〜F − 1、イベントは F から）ので、
// ワークグループで一様。FIXED は局・席・牌だけ（型 < 3）、EVENT はイベントだけ（型 3）、MIXED はどちらもありうる（タイルで分けない T0 も
// これを使う）。
enum class BlockKind { FIXED, MIXED, EVENT };

// 対（クエリ q, キー k）の 7 セグメントのコードを segments に書く（選択だけで書き、分岐は書かない）。relation は牌の関係のクラス
// tileRelation[q.tileIndex()·tiles + k.tileIndex()]、seats は席の数 S。Query・Key は q・k のタイルの型で、型から 0 と決まる項を
// コンパイル時に落とす（局・席・牌どうしではセグメント 4・5 と繋がり、局・席・牌とイベントの間ではセグメント 6 の全て）。落とすのは
// 0 の項だけなので、コードは MIXED の式とビットで同じになる。
template <BlockKind Query, BlockKind Key>
__device__ __forceinline__ void relationSegments(const RelationToken& q, const RelationToken& k, int relation, int seats,
                                                 int (&segments)[kRelationSegments]) {
  const int typeQ = Query == BlockKind::EVENT ? kEventType : q.type();
  const int typeK = Key == BlockKind::EVENT ? kEventType : k.type();
  segments[0] = 4 * typeQ + typeK;

  const bool bothTiles = q.tile() > 0 && k.tile() > 0;
  segments[1] = bothTiles && relation < kTileRelationFirstHalf ? relation + 1 : 0;
  segments[2] = bothTiles && relation >= kTileRelationFirstHalf ? relation - (kTileRelationFirstHalf - 1) : 0;

  // 後の選択ほど優先する: i だけ、j だけ、両方の順に上書きする。持ち主は 1〜S なので、両方にいるときの差 + S は 1〜2S − 1 になり、
  // mod S は S 以上なら S を引くだけで済む（割り算を作らない）。
  const int ownerQ = q.owner();
  const int ownerK = k.owner();
  const int turn = ownerK - ownerQ + seats;
  int seat = ownerQ > 0 ? ownerQ + 2 * seats : 0;
  seat = ownerK > 0 ? ownerK + seats : seat;
  seat = ownerQ > 0 && ownerK > 0 ? (turn >= seats ? turn - seats : turn) + 1 : seat;
  segments[3] = seat;

  const int eventQ = q.event();
  const int eventK = k.event();
  const bool keyEvent = Key == BlockKind::MIXED ? typeK == kEventType : Key == BlockKind::EVENT;
  const bool queryEvent = Query == BlockKind::MIXED ? typeQ == kEventType : Query == BlockKind::EVENT;
  const int lastDiscardQ = q.lastDiscard();
  const bool afterDiscard = lastDiscardQ > 0 && eventK > lastDiscardQ;
  segments[4] = (keyEvent ? k.phase() + 1 + 4 * afterDiscard : 0) +
                (queryEvent && typeK == 1 ? riichiPhase(eventQ, k.riichi()) + 9 : 0);

  const bool bothEvents = queryEvent && keyEvent;
  const int delta = eventK - eventQ;
  const int magnitude = delta < 0 ? -delta : delta;
  const int step = (magnitude >= 1) + (magnitude >= 2) + (magnitude >= 4) + (magnitude >= 8);
  const int sign = (delta > 0) - (delta < 0);
  segments[5] = (bothEvents ? sign * step + 5 : 0) + (!queryEvent && keyEvent ? 10 + k.ageStep() : 0);

  const int linkQ = q.link();
  const int linkK = k.link();
  const bool forward = linkQ > 0 && linkQ == eventK;
  const bool backward = linkK > 0 && linkK == eventQ && !forward;
  const int link = bothEvents ? (forward ? 1 : 0) + (backward ? 2 : 0) : 0;
  const bool seatToTile = typeQ == 1 && typeK == 2 && ownerQ > 1;
  const bool tileToSeat = typeQ == 2 && typeK == 1 && ownerK > 1;
  segments[6] = link + (seatToTile ? k.safetyFrom(ownerQ) + 3 : 0) + (tileToSeat ? q.safetyFrom(ownerK) + 7 : 0);
}

// 学習のワード（T0）: 7 セグメントのコードを 1 ワードに詰める（セグメント s は bit 4s から 4 bit。eager の RelationClass と同じ並び）。
__device__ __forceinline__ uint32_t packCode(const int (&segments)[kRelationSegments]) {
  uint32_t code = 0;
#pragma unroll
  for (int s = 0; s < kRelationSegments; ++s) {
    code |= static_cast<uint32_t>(segments[s]) << (4 * s);
  }
  return code;
}

// 推論のワード（K0 が書き、K2・K4 が読む。置き方は上に書いた）: 組の表 τ（< 3）のフィールドに項 pairTableEntry(セグメント 2τ,
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

// 推論のワード code を K2・K4 と同じアドレスの式（pairTableOffset・lastTableOffset）で引いた 4 つの項が、セグメントのコード segments
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
static_assert(attentionCodeMatchesRelationRows(), "the attention code must address the relation rows of every segment code");

}  // namespace prism
