package com.epsilon.prism.input;

import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.EventTable.PublicEvent;

/**
 * 数牌の色の正準の順を決め、牌・色・赤の写しを持つ。判断ごとに 1 回作り、牌を扱う全ての所（牌のトークンの順、イベント・遷移・キーの牌、
 * 副露の色、赤）に同じ写しをかける。入れ替えるのは、そのルールで入れ替えてよい色（{@link TileTokens#movableSuits}。四人打ちは萬子・筒子・
 * 索子、三人打ちは筒子と索子）だけで、ほかの色と字牌は動かさない。入れ替える色どうしでは、物理の色の順の位置を正準の順に配り直す。
 *
 * <p>色を並べるときは {@link SuitKey} の宣言の順にキーを比べ、先のキーで決まらなければ次のキーを見る。
 *
 * <ul>
 *   <li>{@link SuitKey#HAND}: 自分の手にあるその色の枚数の合計が多い色が先。同じなら 1 から 9 の枚数を順に比べて多い方が先、それも同じなら赤を持つ色が先。
 *   <li>{@link SuitKey#MELDS}: その色の副露（全ての席）がある色が先。どちらにもあれば、最初の副露が早い色が先。
 *   <li>{@link SuitKey#INDICATORS}: ドラ表示を表示の順に見て、先にその色の表示が出た色が先。表示の順で決まらなければ、表示に赤が出た色が先。
 *   <li>{@link SuitKey#EVENTS}: その色の牌のイベント（打牌・副露）が早い色が先。イベントは 1 つの色にしか属さないので、イベントの列での比べはこれで決まる。
 *   <li>{@link SuitKey#PHYSICAL}: 萬子・筒子・索子の順。ここまで来た判断では {@link #physicalTie} が true になる（色を入れ替えた局面と入力が同じにならない）。
 * </ul>
 *
 * <p>キーは全て判断の記録だけで決まり、色を入れ替えた局面では、キーも入れ替えた先の色に移る。だから正準の写しをかけた後の入力は、色を
 * 入れ替えた局面どうしで同じになる。ただし、物理の色の順で決まった判断は除く。
 */
public final class SuitCanonicalizer {

  /**
   * 色を正準の順に並べるときのキー。萬子・筒子・索子を並べ替えるときは、宣言の順にキーを比べ、先のキーで決まらなければ次のキーを見る。
   * フィンガープリントは宣言の順から作る。
   */
  public enum SuitKey {
    /** 自分の手。 */
    HAND,
    MELDS,
    INDICATORS,
    /** イベントの列。 */
    EVENTS,
    /** 物理の色（ほかのキーが全て同じときの決め手）。 */
    PHYSICAL
  }

  /** 数牌の色の数。 */
  public static final int SUITS = Tiles.NUMBER_SUITS;

  private static final int NUMBERS = Tiles.SUIT_SIZE;
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
  private final int[] tileMap = new int[Tiles.KINDS];
  private final int[] inverse = new int[Tiles.KINDS];

  /** 入れ替える色（物理の色の番号の昇順）。 */
  private final int[] movable;

  private Observation observation;
  private EventTable events;
  private boolean deepKeys;
  private boolean physicalTie;

  /**
   * そのルールで入れ替えてよい色だけを並べ替える正準化を作る。
   *
   * @param variant ルール（四人打ちか三人打ち）
   */
  public SuitCanonicalizer(Variant variant) {
    int suits = TileTokens.of(variant).movableSuits();
    movable = new int[Integer.bitCount(suits)];
    int index = 0;
    for (int suit = 0; suit < SUITS; suit++) {
      if ((suits >>> suit & 1) != 0) {
        movable[index++] = suit;
      }
    }
  }

  /**
   * 色の正準の順を決める。
   *
   * @param o 判断の記録
   * @param events 判断の記録のイベント
   * @param tiles 牌ごとの枚数
   */
  public void load(Observation o, EventTable events, TileCounts tiles) {
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
      boolean aka = (handAka >>> suit & 1) != 0;
      handKey[suit] = total << (NUMBERS * COUNT_BITS + 1) | counts << 1 | (aka ? 1 : 0);
      canonical[suit] = suit;
    }
    for (int index = 0; index < movable.length; index++) {
      order[index] = movable[index];
    }
    for (int index = 1; index < movable.length; index++) {
      int suit = order[index];
      int at = index;
      while (at > 0 && compare(suit, order[at - 1]) < 0) {
        order[at] = order[at - 1];
        at--;
      }
      order[at] = suit;
    }
    for (int position = 0; position < movable.length; position++) {
      canonical[order[position]] = movable[position];
    }
    for (int tile = 0; tile < Tiles.KINDS; tile++) {
      int mapped =
          tile < NUMBER_TILES ? canonical[tile / NUMBERS] * NUMBERS + tile % NUMBERS : tile;
      tileMap[tile] = mapped;
      inverse[mapped] = tile;
    }
  }

  /** 物理の牌を正準の牌にする（負の値はそのまま返す）。 */
  public int tile(int physicalTile) {
    return physicalTile < 0 ? physicalTile : tileMap[physicalTile];
  }

  /** 正準の牌を物理の牌に戻す。 */
  public int physical(int canonicalTile) {
    return inverse[canonicalTile];
  }

  /** 物理の色（萬子 0・筒子 1・索子 2）が正準の順で何番目かを返す（入れ替えない色はそのまま）。 */
  public int suit(int physicalSuit) {
    return canonical[physicalSuit];
  }

  /** 物理の牌のマスクを正準の牌のマスクにする（字牌のビットはそのまま）。 */
  public long mask(long physicalTiles) {
    long tiles = physicalTiles & ~((1L << NUMBER_TILES) - 1);
    for (int suit = 0; suit < SUITS; suit++) {
      long numbers = physicalTiles >>> (suit * NUMBERS) & SUIT_TILES;
      tiles |= numbers << (canonical[suit] * NUMBERS);
    }
    return tiles;
  }

  /** 色の正準の順が、ほかのキーで決まらず、物理の色の順で決まったか。 */
  public boolean physicalTie() {
    return physicalTie;
  }

  /** 色 a を色 b より先に置くなら負を返す。キーは {@link SuitKey} の順に比べる。 */
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

  /** 手のキーで決まらないときだけ、副露・ドラ表示・イベントのキーを作る（判断ごとに 1 回）。 */
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
      int suit = suitOf(events.tile(e));
      if (suit < 0) {
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
      int suit = suitOf(indicator);
      if (suit >= 0) {
        int code = 1 << (INDICATOR_BITS - 1) | Tiles.number(indicator);
        indicatorKey[suit] |=
            (long) code << (INDICATOR_BITS * (Limits.MAX_INDICATORS - 1 - index) + 1);
      }
    }
    int indicatorAka = observation.indicatorAkaMask();
    for (int suit = 0; suit < SUITS; suit++) {
      if ((indicatorAka >>> suit & 1) != 0) {
        indicatorKey[suit] |= 1L;
      }
    }
    deepKeys = true;
  }

  /** 数牌の色の番号（萬子 0・筒子 1・索子 2。字牌は −1）。 */
  private static int suitOf(int kind) {
    Tiles.Group group = Tiles.Group.of(kind);
    return group.suited() ? group.ordinal() : -1;
  }
}
