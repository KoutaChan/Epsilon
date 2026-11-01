package com.epsilon.prism.input;

import com.epsilon.core.AkaTileMask;
import com.epsilon.core.Tile;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputSchema.PublicEvent;
import com.epsilon.prism.input.InputSchema.SuitKey;
import com.epsilon.prism.observe.Observation;

/**
 * 萬子・筒子・索子の正準の順を決め、牌・色・赤の写しを持つ。判断ごとに 1 回作り、牌の軸の全て（牌のトークンの順、出来事・遷移・鍵の牌、副露の色、
 * 赤）に同じ写しをかける。字牌は動かさない。
 *
 * <p>色は {@link SuitKey} の宣言の順に鍵を比べて並べ、先の鍵で決まらなければ次の鍵を見る。
 *
 * <ul>
 *   <li>{@link SuitKey#HAND}: 自分の手のその色の枚数の和が多い色が先。同じなら 1 から 9 の枚数を順に比べて多い方、それも同じなら赤を持つ色が先。
 *   <li>{@link SuitKey#MELDS}: その色の副露（全ての席）がある色が先。どちらにもあれば最初の副露が早い色が先。
 *   <li>{@link SuitKey#INDICATORS}: ドラ表示を表示の順に比べ、先の表示がその色の大きい数の色が先。同じなら表示に赤が出た色が先。
 *   <li>{@link SuitKey#EVENTS}: その色の牌の出来事（打牌・副露）が早い色が先。出来事は 1 つの色にしか属さないので、出来事の列の比べはこれで決まる。
 *   <li>{@link SuitKey#PHYSICAL}: 萬子・筒子・索子の順。ここまで来た判断は {@link #physicalTie} が true になる（色の入れ替えで入力が同じにならない）。
 * </ul>
 *
 * <p>鍵は全て O の純関数で、色を入れ替えた局面では入れ替えた色の鍵になる。だから正準の写しの後の入力は、物理の色で決まった判断を除いて同じになる。
 */
final class SuitCanonicalizer {

  /** 数牌の色の数。 */
  static final int SUITS = 3;

  private static final int NUMBERS = Tile.TILE_TYPES_PER_SUIT;
  private static final int NUMBER_TILES = SUITS * NUMBERS;
  private static final long SUIT_TILES = (1L << NUMBERS) - 1;
  private static final int COUNT_BITS = 3;
  private static final int INDICATOR_BITS = 5;
  private static final int LATER = Integer.MAX_VALUE;
  private static final SuitKey[] KEYS = SuitKey.values();

  private final long[] handKey = new long[SUITS];
  private final int[] firstMeld = new int[SUITS];
  private final long[] indicatorKey = new long[SUITS];
  private final int[] firstEvent = new int[SUITS];
  private final int[] order = new int[SUITS];
  private final int[] canonical = new int[SUITS];
  private final int[] tileMap = new int[Tile.NUM_TILE_TYPES];
  private final int[] inverse = new int[Tile.NUM_TILE_TYPES];
  private Observation observation;
  private EventTable events;
  private boolean deepKeys;
  private boolean physicalTie;

  /**
   * 色の正準の順を決める。
   *
   * @param o 判断の記録
   * @param events O の出来事
   * @param tiles 牌ごとの枚数
   */
  void load(Observation o, EventTable events, TileCounts tiles) {
    observation = o;
    this.events = events;
    deepKeys = false;
    physicalTie = false;
    int handAka = o.handAkaMask();
    for (int suit = 0; suit < SUITS; suit++) {
      long total = 0;
      long counts = 0;
      for (int number = 0; number < NUMBERS; number++) {
        int count = tiles.hand(suit * NUMBERS + number);
        total += count;
        counts = counts << COUNT_BITS | count;
      }
      boolean aka = AkaTileMask.containsTile(handAka, suit * NUMBERS + Tile.M5);
      handKey[suit] = total << (NUMBERS * COUNT_BITS + 1) | counts << 1 | (aka ? 1 : 0);
      order[suit] = suit;
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

  /** 物理の牌を正準の牌にする（負はそのまま）。 */
  int tile(int physicalTile) {
    return physicalTile < 0 ? physicalTile : tileMap[physicalTile];
  }

  /** 正準の牌の物理の牌。 */
  int physical(int canonicalTile) {
    return inverse[canonicalTile];
  }

  /** 物理の色（萬 0・筒 1・索 2）の正準の順の位置。 */
  int suit(int physicalSuit) {
    return canonical[physicalSuit];
  }

  /** 物理の牌のマスクを正準の牌のマスクにする（字牌のビットはそのまま）。 */
  long mask(long physicalTiles) {
    long tiles = physicalTiles & ~((1L << NUMBER_TILES) - 1);
    for (int suit = 0; suit < SUITS; suit++) {
      long numbers = physicalTiles >>> (suit * NUMBERS) & SUIT_TILES;
      tiles |= numbers << (canonical[suit] * NUMBERS);
    }
    return tiles;
  }

  /** 色の正準の順が、ほかの鍵で決まらず物理の色の順で決まった。 */
  boolean physicalTie() {
    return physicalTie;
  }

  /** a を b より先に置くなら負。鍵は {@link SuitKey} の順に比べる。 */
  private int compare(int a, int b) {
    for (SuitKey key : KEYS) {
      if (key != SuitKey.HAND) {
        loadDeepKeys();
      }
      int order =
          switch (key) {
            case HAND -> Long.compare(handKey[b], handKey[a]);
            case MELDS -> Integer.compare(firstMeld[a], firstMeld[b]);
            case INDICATORS -> Long.compare(indicatorKey[b], indicatorKey[a]);
            case EVENTS -> Integer.compare(firstEvent[a], firstEvent[b]);
            case PHYSICAL -> {
              physicalTie = true;
              yield Integer.compare(a, b);
            }
          };
      if (order != 0) {
        return order;
      }
    }
    throw new IllegalStateException("suits " + a + " and " + b + " are the same suit");
  }

  /** 手の鍵で決まらないときだけ、副露・ドラ表示・出来事の鍵を作る（判断ごとに 1 回）。 */
  private void loadDeepKeys() {
    if (deepKeys) {
      return;
    }
    for (int suit = 0; suit < SUITS; suit++) {
      firstMeld[suit] = LATER;
      firstEvent[suit] = LATER;
      indicatorKey[suit] = 0L;
    }
    for (int e = 0, count = events.events(); e < count; e++) {
      int suit = Tile.suitOf(events.tile(e));
      if (suit >= SUITS) {
        continue;
      }
      firstEvent[suit] = Math.min(firstEvent[suit], e);
      if (events.kind(e) != PublicEvent.DISCARD) {
        firstMeld[suit] = Math.min(firstMeld[suit], e);
      }
    }
    int indicators = observation.indicators();
    for (int index = 0; index < indicators; index++) {
      int indicator = observation.indicatorTile(index);
      int suit = Tile.suitOf(indicator);
      if (suit < SUITS) {
        int code = 1 << (INDICATOR_BITS - 1) | indicator % NUMBERS;
        indicatorKey[suit] |=
            (long) code << (INDICATOR_BITS * (Limits.MAX_INDICATORS - 1 - index) + 1);
      }
    }
    int indicatorAka = observation.indicatorAkaMask();
    for (int suit = 0; suit < SUITS; suit++) {
      if (AkaTileMask.containsTile(indicatorAka, suit * NUMBERS + Tile.M5)) {
        indicatorKey[suit] |= 1L;
      }
    }
    deepKeys = true;
  }
}
