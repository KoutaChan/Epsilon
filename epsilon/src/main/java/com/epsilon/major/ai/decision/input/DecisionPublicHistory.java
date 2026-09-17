package com.epsilon.major.ai.decision.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.PublicObservation.VisibleHand;
import com.epsilon.core.River;

/** 公開イベントの時刻と各家の直近の判断時点との関係を、観測済み情報だけから符号化する。 */
public final class DecisionPublicHistory {

  public static final int QUERY_COUNT = 4;
  public static final int TOKEN_COUNT = 112;
  public static final int PACKED_WORDS = 2;
  public static final int SEGMENT_COUNT = 7;
  public static final int ENTRIES_PER_SEGMENT = 12;
  public static final int TABLE_SIZE = 76;
  public static final int NO_ANCHOR = 9;
  public static final int NO_SECOND_EVENT = 10;
  public static final int UNKNOWN = 11;
  private static final int ABSENT_EVENT = -2;
  private static final int UNKNOWN_EVENT = -1;
  private static final int RIVER_TOKEN_COUNT =
      GameState.NUM_PLAYERS * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER;

  private DecisionPublicHistory() {}

  /** スキーマの互換性確認に、時刻の定義・格納順・区間境界・不在と不明の優先順位も含める。 */
  public static String descriptor() {
    return "public-event-v1:discard,public-meld,public-kan-declaration-before-chankan;"
        + "origin-preserves-pon;kan-attempt-marker-does-not-tick;"
        + "queries=observer-relative4;tokens=river96,meld16;words=2x16;"
        + "nibbles=origin-riichi,origin-openmeld,origin-kan,update-riichi,update-openmeld,"
        + "update-kan,query-relative-author;"
        + "bins=le-8,-7to-4,-3to-2,-1,0,1,2to3,4to7,ge8,no-anchor9,no-update10,unknown11;"
        + "priority=no-update,no-anchor,unknown;unknown-time=-1;author=4;"
        + "table=6x12+4;padding=0-masked-by-present;not-state-embedding";
  }

  /** トークンの7種類の時刻関係を独立テーブルへ展開するときの辞書インデックスを返す。 */
  public static int tableIndex(short firstWord, short secondWord, int segment) {
    int word = segment < 4 ? firstWord & 0xffff : secondWord & 0xffff;
    return segment * ENTRIES_PER_SEGMENT + ((word >>> ((segment & 3) * 4)) & 15);
  }

  /** 4家のクエリ順と、既存river96・meld16 トークン順で未加工の関係を書き込む。 */
  static void encode(PublicObservation state, int observer, DecisionInputWriter writer) {
    for (int query = 0; query < QUERY_COUNT; query++) {
      int querySeat = (observer + query) % GameState.NUM_PLAYERS;
      River queryRiver = state.river(querySeat);
      int riichiIndex = queryRiver.riichiDeclarationIndex();
      int riichiAnchor =
          riichiIndex >= 0
              ? queryRiver.discard(riichiIndex).creationEvent()
              : state.isRiichi(querySeat) ? UNKNOWN_EVENT : ABSENT_EVENT;
      VisibleHand queryHand = state.hand(querySeat);
      int openMeldAnchor = lastMeldEvent(queryHand, false);
      int kanAnchor = lastMeldEvent(queryHand, true);

      for (int author = 0; author < GameState.NUM_PLAYERS; author++) {
        int seat = (observer + author) % GameState.NUM_PLAYERS;
        int relativeAuthor = (author - query + GameState.NUM_PLAYERS) % GameState.NUM_PLAYERS;
        River river = state.river(seat);
        for (int index = 0; index < DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER; index++) {
          int packed =
              index < river.size()
                  ? pack(
                      river.discard(index).creationEvent(),
                      ABSENT_EVENT,
                      riichiAnchor,
                      openMeldAnchor,
                      kanAnchor,
                      relativeAuthor)
                  : 0;
          writer.publicHistory(
              query, author * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + index, packed);
        }
        VisibleHand hand = state.hand(seat);
        for (int index = 0; index < DecisionInputSchema.MAX_MELDS_PER_PLAYER; index++) {
          int packed =
              index < hand.meldCount()
                  ? pack(
                      hand.meldCreationEvent(index),
                      hand.meld(index).type() == Meld.Type.KAKAN
                          ? hand.meldKakanEvent(index)
                          : ABSENT_EVENT,
                      riichiAnchor,
                      openMeldAnchor,
                      kanAnchor,
                      relativeAuthor)
                  : 0;
          writer.publicHistory(
              query,
              RIVER_TOKEN_COUNT + author * DecisionInputSchema.MAX_MELDS_PER_PLAYER + index,
              packed);
        }
      }
    }
  }

  private static int lastMeldEvent(VisibleHand hand, boolean kan) {
    int latest = ABSENT_EVENT;
    for (int index = 0; index < hand.meldCount(); index++) {
      Meld meld = hand.meld(index);
      if (kan ? !meld.isKan() : meld.preservesMenzen()) {
        continue;
      }
      int event =
          kan && meld.type() == Meld.Type.KAKAN
              ? hand.meldKakanEvent(index)
              : hand.meldCreationEvent(index);
      if (event == UNKNOWN_EVENT) {
        // 時刻不明の面子と既知面子のどちらが最後かは推測しない。
        return UNKNOWN_EVENT;
      }
      latest = Math.max(latest, event);
    }
    return latest;
  }

  private static int pack(
      int origin, int update, int riichi, int openMeld, int kan, int relativeAuthor) {
    return relation(origin, riichi)
        | relation(origin, openMeld) << 4
        | relation(origin, kan) << 8
        | relation(update, riichi) << 12
        | relation(update, openMeld) << 16
        | relation(update, kan) << 20
        | relativeAuthor << 24;
  }

  static int relation(int event, int anchor) {
    if (event == ABSENT_EVENT) {
      return NO_SECOND_EVENT;
    }
    if (anchor == ABSENT_EVENT) {
      return NO_ANCHOR;
    }
    if (event == UNKNOWN_EVENT || anchor == UNKNOWN_EVENT) {
      return UNKNOWN;
    }
    int distance = event - anchor;
    if (distance <= -8) {
      return 0;
    }
    if (distance <= -4) {
      return 1;
    }
    if (distance <= -2) {
      return 2;
    }
    if (distance <= 1) {
      return distance + 4;
    }
    if (distance <= 3) {
      return 6;
    }
    return distance <= 7 ? 7 : 8;
  }
}
