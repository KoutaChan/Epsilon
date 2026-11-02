// 関係の偏りの符号。eager の RelationClass と同じ式を、トークンの対 (問い合わせ i, 鍵 j) ごとに整数の演算だけで作る。
// 区画 s の符号 c は ρ の行 16s + c を引く。属性の値は書き方のまま（0 はなし、番号は番号 + 1）。型は 0 局、1 席、2 牌、3 出来事。
//
//   0 型の対:       4·型(i) + 型(j)
//   1・2 牌:        両方に牌があり、MahjongTileRelation の類 c が 15 未満なら区画 1 に c + 1、15 以上なら区画 2 に c − 14
//   3 席:           両方に持ち主: 1 + (持ち主(j) − 持ち主(i)) mod 4、j だけ: 4 + 持ち主(j)、i だけ: 8 + 持ち主(i)
//   4 立直:         鍵が出来事: 1 + 前後(番号(j), 立直(j)) + 4·[最後の打牌(i) > 0 かつ 番号(j) > 最後の打牌(i)]
//                   問い合わせが出来事で鍵が席: 9 + 前後(番号(i), 立直(j))。前後は立直なし 0、前 1、宣言 2、後 3
//   5 時間:         両方が出来事: 5 + 符号(δ)·段(|δ|)（δ = 番号(j) − 番号(i)、段は |δ| ≥ 1・2・4・8 の数）
//                   問い合わせが出来事でなく鍵が出来事: 10 + [d ≥ 5] + [d ≥ 9] + [d ≥ 17]（d = 番号(局) − 番号(j)）
//   6 繋がり・安全: 両方が出来事: i の繋がりが j なら 1、そうでなく j の繋がりが i なら 2
//                   席 → 牌で席が相手: 3 + その相手から見た牌の安全、牌 → 席で席が相手: 7 + その相手から見た牌の安全
//
// 式の実装はここの 1 つだけで、K0（推論）と T0（学習）が使う。どちらも行のトークンごとに属性を詰めた 3 語（RelationToken）を LDS に
// 1 回作り、対の符号は 2 つのトークンの語と牌の関係の表の 1 バイトから選択だけで作る（relationSegments。分岐は書かない）。鍵の側
// だけで決まる前後(番号(j), 立直(j)) と時間の段はトークンの語に入れておく。
//
// 対の符号の語は 2 通りある。T0 は eager の RelationClass と同じ語（packCode: 区画 s を bit 4s から、eager の 2 語の low・high と同じ
// 並び）を書き、学習の注意（T1）が読む。K0 は行ごとに全ての対の符号を推論の語（attentionCode: 区画 2 つずつを偏りの表の項の番号に
// 直した並び）に詰め、K2・K4 の全ての頭の対と層がそれを読む。区画 6 は 10 以下なので、15 を行の外の鍵の印にする（K2・K4 の偏りの表で
// −∞）。
//
// 偏りの表は ρ だけで決まるので、重みの公開のとき（Plan::uploadWeights）に K0 の隣の relationBiasKernel が層ごとに 1 回作り、K2・K4 の
// 作業群は自分の頭の対の分を LDS へ写すだけにする。
#pragma once

#include "prism_hip_kernels.h"

namespace prism {

constexpr int kRelationSegments = 7;
constexpr int kRelationEntries = 16;
constexpr int kTileRelationFirstHalf = kRelationEntries - 1;
constexpr int kMaxAttributes = 12;  // 関係の属性の列の上限（K0・T0 はトークンの属性の行をレジスタに読む）
constexpr int kMaxTiles = 34;       // 牌の種類の上限（K0・T0 の LDS の牌の関係の表）

// 符号の置き方: 行ごとに 16 トークンの塊の対 (qt, kt) を kCodeBlockWords 語ずつ（Java の HipShape.RELATION_CODE_WORDS）。
constexpr int kCodeBlocks = kMaxRowTokens / 16;
constexpr int kCodeBlockWords = 256;
constexpr uint32_t kPaddedKeyCode = 0x0F000000u;  // 推論の語で区画 6 が 15、他の区画が 0: 行の外の鍵

// 行 row の塊の対 (qt, kt) の先頭の語。語 t は lane = t >> 3、v = t & 7 として i = 16qt + (lane & 15)、j = 16kt + 2v + (lane >> 4)
// の符号（K2・K4 の WMMA の S^T の断片で lane が持つ 8 鍵の順）。
__host__ __device__ __forceinline__ size_t codeIndex(int row, int qt, int kt) {
  return ((static_cast<size_t>(row) * kCodeBlocks + qt) * kCodeBlocks + kt) * kCodeBlockWords;
}

// 偏りの表の置き方: 1 層は [頭の対, kBiasEntries, 2] の FP32 で、項は (頭 2p, 頭 2p + 1) の値の対（exp2 の softmax のため ×log2e）。
// 区画 2 つずつ前もって足した組の表 τ（< 3）の項 pairTableEntry(a, b) は ρ[32τ + a] + ρ[32τ + 16 + b]（a は区画 2τ、b は区画 2τ + 1 の
// 符号）、区画 6 の表の項 a は ρ[96 + a]（15 は行の外の鍵の −∞）。K2・K4 は推論の語を 8 bit ずつ 4 回引いて足す。
constexpr int kPairTables = 3;
constexpr int kPairEntries = kRelationEntries * kRelationEntries;
constexpr int kBiasEntries = kPairTables * kPairEntries + kRelationEntries;
constexpr float kLog2e = 1.4426950408889634f;
static_assert(2 * kPairTables + 1 == kRelationSegments, "the pair tables cover segments 0..5 and the last table segment 6");

// 組の表の区画の対 (a, b)（各 < 16）の項 (a ^ b) + 16b（16 × 16 の全単射）。表の 8 バイトの項 e が載る LDS の bank は e mod 16 で
// 決まる（K2 の b64 は 2 bank の対、K4 の b32 は頭ごとに 1 bank）。1 回の読みの 16 lane は同じ鍵・違う問い合わせで、区画 2τ（型の対・
// 牌の関係の後半・立直）はほぼ揃い、区画 2τ + 1 がばらつくので、下の 4 bit を a ^ b にして b の違いを bank に出す（a + 16b の並び
// では違う項が同じ bank に重なる）。
__host__ __device__ constexpr int pairTableEntry(int a, int b) { return (a ^ b) | b << 4; }

// 組の表の項 e（< 256）の区画の対 (a, b)（pairTableEntry の逆）。
__host__ __device__ constexpr int pairTableFirst(int e) { return (e ^ (e >> 4)) & 15; }
__host__ __device__ constexpr int pairTableSecond(int e) { return e >> 4; }

// 1 層の偏りの表の FP32 の数。
__host__ __device__ __forceinline__ size_t relationBiasFloats(int headPairs) {
  return static_cast<size_t>(headPairs) * kBiasEntries * 2;
}

// 前後: 立直なし 0、前 1、宣言 2、後 3。
__device__ __forceinline__ int riichiPhase(int number, int riichi) {
  return riichi > 0 ? (number >= riichi) + (number > riichi) + 1 : 0;
}

// 相手の数（持ち主の ID 2〜4）。トークンの語 c は相手ごとに安全を 1 バイト持つので、安全の列はこれ以下（K0・T0 の起動の口で確かめる）。
constexpr int kRelationOpponents = 3;

// 1 トークンの属性を詰めた 3 語（行ごとに 1 回作る。欄は属性のバイトのままか、鍵の側だけで決まる値）:
//   a: 牌（bit 0〜7）、持ち主（8〜15）、型（16〜17）、前後(番号, 立直)（18〜19）、時間の段 [d ≥ 5] + [d ≥ 9] + [d ≥ 17]（20〜21、
//      d = 番号(局) − 番号）
//   b: 番号（0〜7）、最後の打牌（8〜15）、立直（16〜23）、繋がり（24〜31）
//   c: 相手 o（持ち主 2 + o）から見た牌の安全をバイト o に（安全の列のない相手は 0）
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

  // 牌の関係の表の行・列の番号（牌がなければ 0 を引き、その類は区画 1・2 の式が捨てる）。
  __device__ __forceinline__ int tileIndex() const { return max(tile() - 1, 0); }

  // 席 owner から見たこの牌の安全（owner が相手でなければ 0）。
  __device__ __forceinline__ int safetyFrom(int owner) const {
    const int opponent = owner - 2;
    return opponent >= 0 && opponent < kRelationOpponents ? static_cast<int>((c >> (8 * opponent)) & 0xffu) : 0;
  }
};

// トークンの語。attributes はそのトークンの属性の行（列は c の通り。詰め物の位置は全て 0）、type は位置の型、now は行の局のトークンの
// 番号。
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

// 対 (問い合わせ q, 鍵 k) の 7 区画の符号を segments に書く（選択だけで書き、分岐は書かない）。relation は牌の関係の類
// tileRelation[q.tileIndex()·tiles + k.tileIndex()]。
__device__ __forceinline__ void relationSegments(const RelationToken& q, const RelationToken& k, int relation,
                                                 int (&segments)[kRelationSegments]) {
  const int typeQ = q.type();
  const int typeK = k.type();
  segments[0] = 4 * typeQ + typeK;

  const bool bothTiles = q.tile() > 0 && k.tile() > 0;
  segments[1] = bothTiles && relation < kTileRelationFirstHalf ? relation + 1 : 0;
  segments[2] = bothTiles && relation >= kTileRelationFirstHalf ? relation - (kTileRelationFirstHalf - 1) : 0;

  // 後の選択ほど強い: i だけ、j だけ、両方の順に上書きする。
  const int ownerQ = q.owner();
  const int ownerK = k.owner();
  int seat = ownerQ > 0 ? ownerQ + 8 : 0;
  seat = ownerK > 0 ? ownerK + 4 : seat;
  seat = ownerQ > 0 && ownerK > 0 ? (ownerK - ownerQ + 4) % 4 + 1 : seat;
  segments[3] = seat;

  const int eventQ = q.event();
  const int eventK = k.event();
  const bool keyEvent = typeK == 3;
  const bool queryEvent = typeQ == 3;
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

// 学習の語（T0）: 7 区画の符号を 1 語に詰める（区画 s は bit 4s から 4 bit。eager の RelationClass と同じ並び）。
__device__ __forceinline__ uint32_t packCode(const int (&segments)[kRelationSegments]) {
  uint32_t code = 0;
#pragma unroll
  for (int s = 0; s < kRelationSegments; ++s) {
    code |= static_cast<uint32_t>(segments[s]) << (4 * s);
  }
  return code;
}

// 推論の語（K0 が書き、K2・K4 が読む）: バイト τ（< 3）に組の表 τ の項 pairTableEntry(区画 2τ, 区画 2τ + 1)、バイト 3 に区画 6 の符号
// （区画の符号はどれも 16 未満）。
__host__ __device__ constexpr uint32_t attentionCode(const int (&segments)[kRelationSegments]) {
  uint32_t code = static_cast<uint32_t>(segments[kRelationSegments - 1]) << (8 * kPairTables);
#pragma unroll
  for (int table = 0; table < kPairTables; ++table) {
    code |= static_cast<uint32_t>(pairTableEntry(segments[2 * table], segments[2 * table + 1])) << (8 * table);
  }
  return code;
}

// 推論の語を K2・K4 と同じく 8 bit ずつ引いた項が、全ての区画の符号の組で RelationClass と同じ ρ の行を指すこと（組の表 τ の項は
// 行 32τ + a と 32τ + 16 + b、区画 6 の表の項は行 96 + c）。区画は語の別々のビットに入るので、引く区画の対・符号を全て回し、他の
// 区画を 0 と 15 にして漏れがないことを見れば全ての語を尽くす。行の外の鍵の印は区画 6 の 15 と組の表の (0, 0) を指す。
constexpr bool attentionCodeMatchesRelationRows() {
  for (int other = 0; other < kRelationEntries; other += kRelationEntries - 1) {
    for (int table = 0; table < kPairTables; ++table) {
      for (int a = 0; a < kRelationEntries; ++a) {
        for (int b = 0; b < kRelationEntries; ++b) {
          int segments[kRelationSegments] = {other, other, other, other, other, other, other};
          segments[2 * table] = a;
          segments[2 * table + 1] = b;
          const uint32_t code = attentionCode(segments);
          for (int t = 0; t < kPairTables; ++t) {
            const int e = static_cast<int>((code >> (8 * t)) & 0xffu);
            if (pairTableFirst(e) != segments[2 * t] || pairTableSecond(e) != segments[2 * t + 1]) {
              return false;
            }
          }
          if (static_cast<int>(code >> (8 * kPairTables)) != other) {
            return false;
          }
        }
      }
    }
    for (int c = 0; c < kRelationEntries; ++c) {
      int segments[kRelationSegments] = {other, other, other, other, other, other, c};
      if (static_cast<int>(attentionCode(segments) >> (8 * kPairTables)) != c) {
        return false;
      }
    }
  }
  for (int t = 0; t < kPairTables; ++t) {
    const int e = static_cast<int>((kPaddedKeyCode >> (8 * t)) & 0xffu);
    if (pairTableFirst(e) != 0 || pairTableSecond(e) != 0) {
      return false;
    }
  }
  return static_cast<int>(kPaddedKeyCode >> (8 * kPairTables)) == kRelationEntries - 1;
}
static_assert(attentionCodeMatchesRelationRows(), "the attention code must address the relation rows of every segment code");

}  // namespace prism
