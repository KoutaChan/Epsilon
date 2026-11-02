package com.epsilon.prism.input;

import com.epsilon.core.AkaTileMask;
import com.epsilon.core.DoraState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.prism.input.InputSchema.PublicEvent;
import com.epsilon.prism.observe.Observation;
import java.util.Arrays;

/**
 * 判断の時点の牌ごとの枚数（自分の手・公開で見える・自分から見えない）、ドラの倍率と表ドラの列、見えた赤。判断ごとに 1 回作り、割り当てを
 * しない。物理の牌の番号で持つ（色の写しは書くときにかける）。
 *
 * <p>見える枚数はエンジンの公開の集計（{@code RoundPublicStateIndex}）と同じ数え方で、鳴かれていない打牌・全ての席の副露の牌・ドラ表示を数える。
 * 見えない枚数は 4 − 見える − 自分の手。見えた赤は、打牌・副露・ドラ表示・自分の手に出た色。
 */
final class TileCounts {

  private static final int TILES = Tile.NUM_TILE_TYPES;
  private static final int[] FIVES = {Tile.M5, Tile.P5, Tile.S5};
  private static final long FIVE_TILES = 1L << Tile.M5 | 1L << Tile.P5 | 1L << Tile.S5;

  private final int[] hand = new int[TILES];
  private final int[] visible = new int[TILES];
  private final int[] unseen = new int[TILES];
  private final int[] dora = new int[TILES];
  private final long[] unseenMore = new long[Tile.TILES_PER_TYPE];
  private int totalUnseen;
  private long unseenAka;
  private long doraTiles;
  private long doraPacked;
  private int indicators;

  /**
   * 枚数を数える。
   *
   * @param o 判断の記録
   * @param events O の出来事
   * @throws IllegalStateException 見える枚数と手が 4 枚を超える牌があるとき
   */
  void load(Observation o, EventTable events) {
    Arrays.fill(visible, 0);
    Arrays.fill(dora, 0);
    int akaSeen = o.handAkaMask() | o.indicatorAkaMask();
    for (int e = 0, count = events.events(); e < count; e++) {
      if (events.kind(e) == PublicEvent.DISCARD) {
        if (events.calledBy(e) < 0) {
          visible[events.tile(e)]++;
        }
        if (events.aka(e)) {
          akaSeen = AkaTileMask.includeTile(akaSeen, events.tile(e));
        }
        continue;
      }
      Meld meld = events.meld(e);
      if (meld.type() == Meld.Type.KAKAN) {
        // ポンの出来事が 3 枚を数えたので、加槓は足した 1 枚だけ。
        visible[meld.baseTileType()]++;
      } else {
        for (int index = 0, size = meld.size(); index < size; index++) {
          visible[meld.tileAt(index)]++;
        }
      }
      if (meld.containsAkaTile()) {
        akaSeen = AkaTileMask.includeTile(akaSeen, meld.akaTileType());
      }
    }
    unseenAka = FIVE_TILES & ~akaTiles(akaSeen);
    indicators = o.indicators();
    doraPacked = 0L;
    doraTiles = 0L;
    for (int index = 0; index < indicators; index++) {
      int indicator = o.indicatorTile(index);
      int doraTile = Tile.doraFrom(indicator);
      visible[indicator]++;
      dora[doraTile]++;
      doraTiles |= 1L << doraTile;
      doraPacked = DoraState.packTile(doraPacked, index, doraTile);
    }
    totalUnseen = 0;
    Arrays.fill(unseenMore, 0L);
    o.handCounts(hand);
    for (int tile = 0; tile < TILES; tile++) {
      int left = Tile.TILES_PER_TYPE - visible[tile] - hand[tile];
      if (left < 0) {
        throw new IllegalStateException("more than four copies of tile " + tile + " are known");
      }
      unseen[tile] = left;
      totalUnseen += left;
      for (int copy = 0; copy < left; copy++) {
        unseenMore[copy] |= 1L << tile;
      }
    }
  }

  /** 自分の手（副露を除き、ツモ牌を含む）の枚数。 */
  int hand(int tile) {
    return hand[tile];
  }

  /** 公開で見える枚数。 */
  int visible(int tile) {
    return visible[tile];
  }

  /** 自分から見えない枚数。 */
  int unseen(int tile) {
    return unseen[tile];
  }

  /** 見えない牌の総数。 */
  int totalUnseen() {
    return totalUnseen;
  }

  /** ドラの倍率（この牌を指すドラ表示の数）。 */
  int dora(int tile) {
    return dora[tile];
  }

  /** 表ドラの牌のマスク（倍率が 1 以上の牌）。 */
  long doraTiles() {
    return doraTiles;
  }

  /** 表ドラの牌を表示の順に詰めた列（{@link DoraState#doraTileTypesPacked} と同じ形）。 */
  long doraPacked() {
    return doraPacked;
  }

  int indicators() {
    return indicators;
  }

  /** この牌が赤になりうる 5 で、その色の赤がまだ見えていない（どこにも出ていない）。 */
  boolean akaUnseen(int tile) {
    return (unseenAka & 1L << tile) != 0L;
  }

  /** 赤がまだ見えていない 5 の牌のマスク。 */
  long unseenAka() {
    return unseenAka;
  }

  /**
   * 色ごとの赤のマスク（{@link AkaTileMask}）を、その色の 5 の牌のマスクにする。
   *
   * @param akaMask 色ごとの赤のマスク
   * @return 5 の牌のマスク
   */
  static long akaTiles(int akaMask) {
    long tiles = 0L;
    for (int five : FIVES) {
      if (AkaTileMask.containsTile(akaMask, five)) {
        tiles |= 1L << five;
      }
    }
    return tiles;
  }

  /** マスクの牌の見えない枚数の和（生きた枚数）。 */
  int liveCopies(long tiles) {
    int copies = 0;
    for (long more : unseenMore) {
      copies += Long.bitCount(tiles & more);
    }
    return copies;
  }
}
