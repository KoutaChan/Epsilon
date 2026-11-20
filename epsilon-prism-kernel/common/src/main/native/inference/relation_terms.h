#pragma once
#include "device_inline.h"

#include "relation_layout.h"

namespace prism {

// 立直との前後: 立直なし 0、宣言より前 1、宣言 2、宣言より後 3。
PRISM_DEVICE_INLINE int phaseRelativeToRiichi(int number, int riichi) {
  return riichi > 0 ? (number >= riichi) + (number > riichi) + 1 : 0;
}

// 1 トークンの属性を詰めた 3 ワード（行ごとに 1 回作る。フィールドは属性のバイトそのままか、キーの側だけで決まる値）:
//   a: 牌（bit 0〜7）、持ち主（8〜15）、型（16〜17）、前後(番号, 立直)（18〜19）、時間の段 [d ≥ 5] + [d ≥ 9] + [d ≥ 17]（20〜21、
//      d = 番号(局) − 番号）
//   b: 番号（0〜7）、最後の打牌（8〜15）、立直（16〜23）、繋がり（24〜31）
//   c: 相手 o（持ち主 2 + o）から見た牌の安全度をバイト o に置く（安全度の列のない相手は 0）
struct RelationToken {
  uint32_t a;
  uint32_t b;
  uint32_t c;

  PRISM_DEVICE_INLINE int tile() const { return static_cast<int>(a & 0xffu); }
  PRISM_DEVICE_INLINE int owner() const { return static_cast<int>((a >> 8) & 0xffu); }
  PRISM_DEVICE_INLINE int type() const { return static_cast<int>((a >> 16) & 3u); }
  PRISM_DEVICE_INLINE int phase() const { return static_cast<int>((a >> 18) & 3u); }
  PRISM_DEVICE_INLINE int ageStep() const { return static_cast<int>((a >> 20) & 3u); }
  PRISM_DEVICE_INLINE int event() const { return static_cast<int>(b & 0xffu); }
  PRISM_DEVICE_INLINE int lastDiscard() const { return static_cast<int>((b >> 8) & 0xffu); }
  PRISM_DEVICE_INLINE int riichi() const { return static_cast<int>((b >> 16) & 0xffu); }
  PRISM_DEVICE_INLINE int link() const { return static_cast<int>(b >> 24); }

  // 牌の関係の表の行・列の番号（牌がなければ 0 を引くが、そのクラスはセグメント 1・2 の式が捨てる）。
  PRISM_DEVICE_INLINE int tileIndex() const { return tile() > 0 ? tile() - 1 : 0; }

  // 席 owner から見たこの牌の安全度（owner が相手でなければ 0）。
  PRISM_DEVICE_INLINE int safetyFrom(int owner) const {
    const int opponent = owner - 2;
    return opponent >= 0 && opponent < kRelationOpponents ? static_cast<int>((c >> (8 * opponent)) & 0xffu) : 0;
  }
};

// トークンのワードを作る。attributes はそのトークンの属性の行（列は c のとおり。パディングの位置は全て 0）、type は位置の型、now は
// 行の局のトークンの番号。
PRISM_DEVICE_INLINE RelationToken relationToken(int type, const uint8_t* attributes, int now,
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
            static_cast<uint32_t>(type) << 16 | static_cast<uint32_t>(phaseRelativeToRiichi(event, riichi)) << 18 |
            ageStep << 20;
  token.b = static_cast<uint32_t>(event) | static_cast<uint32_t>(attributes[c.lastDiscard]) << 8 |
            static_cast<uint32_t>(riichi) << 16 | static_cast<uint32_t>(attributes[c.link]) << 24;
  token.c = safety;
  return token;
}

// イベントのトークンの型。
constexpr int kEventType = 3;

// トークンのタイル（16 の位置）の中の型のそろい方。型は位置だけで決まる（局・席・牌は位置 0〜F − 1、イベントは F から）ので、
// ワークグループで一様。FIXED は局・席・牌だけ（型 < 3）、EVENT はイベントだけ（型 3）、MIXED はどちらもありうる（タイルを分けずに関係のコードを作る学習でも
// これを使う）。
enum class BlockKind { FIXED, MIXED, EVENT };

// 対（クエリ q, キー k）の 7 セグメントのコードを segments に書く（選択だけで書き、分岐は書かない）。relation は牌の関係のクラス
// tileRelation[q.tileIndex()·tiles + k.tileIndex()]、seats は席の数 S。Query・Key は q・k のタイルの型で、型から 0 と決まる項を
// コンパイル時に落とす（局・席・牌どうしではセグメント 4・5 と繋がり、局・席・牌とイベントの間ではセグメント 6 の全て）。落とすのは
// 0 の項だけなので、コードは MIXED の式とビットで同じになる。
template <BlockKind Query, BlockKind Key>
PRISM_DEVICE_INLINE void relationSegments(const RelationToken& q, const RelationToken& k, int relation, int seats,
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
                (queryEvent && typeK == 1 ? phaseRelativeToRiichi(eventQ, k.riichi()) + 9 : 0);

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

// 学習用のワード: 7 セグメントのコードを 1 ワードに詰める（セグメント s は bit 4s から 4 bit。eager の RelationClass と同じ並び）。
PRISM_DEVICE_INLINE uint32_t packCode(const int (&segments)[kRelationSegments]) {
  uint32_t code = 0;
#pragma unroll
  for (int s = 0; s < kRelationSegments; ++s) {
    code |= static_cast<uint32_t>(segments[s]) << (4 * s);
  }
  return code;
}

}  // namespace prism
