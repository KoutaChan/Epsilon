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
// K0 が行ごとに全ての対の符号を作って 1 語に詰め（区画 s を bit 4s から、eager の 2 語の low・high と同じ並び）、K2・K4 の全ての頭の対と
// 層がそれを読む。区画 6 は 10 以下なので、15 を行の外の鍵の印にする（K2・K4 の偏りの表で −∞）。
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
constexpr uint32_t kPaddedKeyCode = 0x0F000000u;  // 区画 6 が 15: 行の外の鍵

// 行 row の塊の対 (qt, kt) の先頭の語。語 t は lane = t >> 3、v = t & 7 として i = 16qt + (lane & 15)、j = 16kt + 2v + (lane >> 4)
// の符号（K2・K4 の WMMA の S^T の断片で lane が持つ 8 鍵の順）。
__host__ __device__ __forceinline__ size_t codeIndex(int row, int qt, int kt) {
  return ((static_cast<size_t>(row) * kCodeBlocks + qt) * kCodeBlocks + kt) * kCodeBlockWords;
}

// 偏りの表の置き方: 1 層は [頭の対, kBiasEntries, 2] の FP32 で、項は (頭 2p, 頭 2p + 1) の値の対（exp2 の softmax のため ×log2e）。
// 区画 2 つずつ前もって足した組の表 τ（< 3）の項 a + 16b は ρ[32τ + a] + ρ[32τ + 16 + b]、区画 6 の表の項 a は ρ[96 + a]
// （15 は行の外の鍵の −∞）。K2・K4 は符号を 8 bit ずつ 4 回引いて足す。
constexpr int kPairTables = 3;
constexpr int kPairEntries = kRelationEntries * kRelationEntries;
constexpr int kBiasEntries = kPairTables * kPairEntries + kRelationEntries;
constexpr float kLog2e = 1.4426950408889634f;
static_assert(2 * kPairTables + 1 == kRelationSegments, "the pair tables cover segments 0..5 and the last table segment 6");

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

// 7 区画の符号を 1 語に詰める（区画 s は bit 4s から 4 bit）。
__device__ __forceinline__ uint32_t packCode(const int (&segments)[kRelationSegments]) {
  uint32_t code = 0;
#pragma unroll
  for (int s = 0; s < kRelationSegments; ++s) {
    code |= static_cast<uint32_t>(segments[s]) << (4 * s);
  }
  return code;
}

}  // namespace prism
