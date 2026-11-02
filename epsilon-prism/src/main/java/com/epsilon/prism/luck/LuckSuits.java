package com.epsilon.prism.luck;

import com.epsilon.core.AkaTileMask;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.prism.Limits;

/**
 * 観測 O° の萬子・筒子・索子の正準の順を決め、牌の写しを持つ（運の網 Ω の色の正準化）。O° ごとに 1 回決め、牌のトークンの順・プール・結果の
 * 全てに同じ写しをかける。字牌は動かさない。
 *
 * <p>色は次の鍵を順に比べて並べ、先の鍵で決まらなければ次の鍵を見る（{@link #DEFINITION}）。
 *
 * <ol>
 *   <li>相対席 0（行為者）・1・2・3 の手の、その色の枚数の和が多い色が先。同じなら 1 から 9 の枚数を順に比べて多い方、それも同じなら赤を持つ
 *       色が先。
 *   <li>その色の副露（全ての席）が早く成立した色が先（副露のない色は後）。
 *   <li>ドラ表示を表示の順に比べ、先の表示がその色の大きい数の色が先。同じなら表示に赤が出た色が先。
 *   <li>その色の打牌（全ての席）が早い色が先。
 *   <li>萬子・筒子・索子の順。
 * </ol>
 *
 * <p>鍵は全て O° の出来事の前の状態の関数で、結果の欄は読まない。色を入れ替えた O° では入れ替えた色の鍵になるので、正準の写しの後の入力は同じ
 * になる。割り当てをしない。
 */
final class LuckSuits {

  /** 定義の記述（入力の指紋に入れる）。 */
  static final String DEFINITION =
      "hands(seat0..3: total,counts1..9,aka) > melds(first event) > indicators(order,number,aka)"
          + " > discards(first event) > physical";

  /** 数牌の色の数。 */
  static final int SUITS = 3;

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int NUMBERS = Tile.TILE_TYPES_PER_SUIT;
  private static final int NUMBER_TILES = SUITS * NUMBERS;
  private static final int COUNT_BITS = 3;
  private static final int INDICATOR_BITS = 5;
  private static final int LATER = Integer.MAX_VALUE;

  private final long[][] handKey = new long[SEATS][SUITS];
  private final int[] firstMeld = new int[SUITS];
  private final long[] indicatorKey = new long[SUITS];
  private final int[] firstDiscard = new int[SUITS];
  private final int[] order = new int[SUITS];
  private final int[] canonical = new int[SUITS];
  private final int[] tileMap = new int[Tile.NUM_TILE_TYPES];
  private final int[] inverse = new int[Tile.NUM_TILE_TYPES];

  /**
   * 色の正準の順を決める。
   *
   * @param o 観測 O°（付け替えた読み手）
   */
  void load(LuckObservation o) {
    for (int seat = 0; seat < SEATS; seat++) {
      int aka = o.handAkaMask(seat);
      for (int suit = 0; suit < SUITS; suit++) {
        long total = 0;
        long counts = 0;
        for (int number = 0; number < NUMBERS; number++) {
          int count = o.handCount(seat, suit * NUMBERS + number);
          total += count;
          counts = counts << COUNT_BITS | count;
        }
        boolean red = AkaTileMask.containsTile(aka, suit * NUMBERS + Tile.M5);
        handKey[seat][suit] = total << (NUMBERS * COUNT_BITS + 1) | counts << 1 | (red ? 1 : 0);
      }
    }
    for (int suit = 0; suit < SUITS; suit++) {
      firstMeld[suit] = LATER;
      firstDiscard[suit] = LATER;
      indicatorKey[suit] = 0L;
      order[suit] = suit;
    }
    for (int seat = 0; seat < SEATS; seat++) {
      for (int index = 0, count = o.meldCount(seat); index < count; index++) {
        Meld meld = o.meld(seat, index);
        int suit = Tile.suitOf(meld.baseTileType());
        if (suit < SUITS) {
          firstMeld[suit] = Math.min(firstMeld[suit], o.meldEvent(seat, index));
        }
      }
      for (int index = 0, count = o.discardCount(seat); index < count; index++) {
        int suit = Tile.suitOf(o.discardTile(seat, index));
        if (suit < SUITS) {
          firstDiscard[suit] = Math.min(firstDiscard[suit], o.discardEvent(seat, index));
        }
      }
    }
    for (int index = 0, count = o.indicators(); index < count; index++) {
      int indicator = o.indicatorTile(index);
      int suit = Tile.suitOf(indicator);
      if (suit < SUITS) {
        int code = 1 << (INDICATOR_BITS - 1) | indicator % NUMBERS;
        indicatorKey[suit] |=
            (long) code << (INDICATOR_BITS * (Limits.MAX_INDICATORS - 1 - index) + 1);
        if (o.indicatorAka(index)) {
          indicatorKey[suit] |= 1L;
        }
      }
    }
    for (int index = 1; index < SUITS; index++) {
      int suit = order[index];
      int at = index;
      while (at > 0 && compare(suit, order[at - 1]) < 0) {
        order[at] = order[at - 1];
        at--;
      }
      order[at] = suit;
    }
    for (int position = 0; position < SUITS; position++) {
      canonical[order[position]] = position;
    }
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      int mapped =
          tile < NUMBER_TILES ? canonical[tile / NUMBERS] * NUMBERS + tile % NUMBERS : tile;
      tileMap[tile] = mapped;
      inverse[mapped] = tile;
    }
  }

  /** 物理の牌を正準の牌にする。 */
  int tile(int physicalTile) {
    return tileMap[physicalTile];
  }

  /** 正準の牌の物理の牌。 */
  int physical(int canonicalTile) {
    return inverse[canonicalTile];
  }

  /** 正準の色の位置（0〜2）の物理の色。 */
  int physicalSuit(int canonicalSuit) {
    return order[canonicalSuit];
  }

  /** a を b より先に置くなら負。鍵は {@link #DEFINITION} の順に比べる。 */
  private int compare(int a, int b) {
    for (int seat = 0; seat < SEATS; seat++) {
      int byHand = Long.compare(handKey[seat][b], handKey[seat][a]);
      if (byHand != 0) {
        return byHand;
      }
    }
    int byMeld = Integer.compare(firstMeld[a], firstMeld[b]);
    if (byMeld != 0) {
      return byMeld;
    }
    int byIndicator = Long.compare(indicatorKey[b], indicatorKey[a]);
    if (byIndicator != 0) {
      return byIndicator;
    }
    int byDiscard = Integer.compare(firstDiscard[a], firstDiscard[b]);
    return byDiscard != 0 ? byDiscard : Integer.compare(a, b);
  }
}
