package com.epsilon.prism.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.ScoreRanking;
import com.epsilon.core.Tile;
import com.epsilon.prism.input.InputSchema.PublicEvent;
import com.epsilon.prism.observe.Observation;
import java.util.Arrays;

/**
 * 席ごとの公開の事実（順位・副露の事実・流し満貫）。判断ごとに 1 回作り、割り当てをしない。席は観測席から見た相対席。
 *
 * <p>副露の色は物理の色で持ち、書くときに正準の色に写す（{@link #openSuit} の値は、副露がないか色が混ざれば {@link #MIXED}、字牌だけなら
 * {@link Tile#SUIT_HONOR}、1 つの色と字牌だけならその物理の色）。
 */
final class SeatFacts {

  /** 副露がないか、2 つ以上の色が混ざっている。 */
  static final int MIXED = -1;

  private static final int SEATS = GameState.NUM_PLAYERS;

  private final int[] absoluteScores = new int[SEATS];
  private final int[] absoluteRanks = new int[SEATS];
  private final int[] rank = new int[SEATS];
  private final int[] kans = new int[SEATS];
  private final int[] yakuhaiHan = new int[SEATS];
  private final int[] openSuit = new int[SEATS];
  private final int[] openDora = new int[SEATS];
  private final boolean[] menzen = new boolean[SEATS];
  private final boolean[] tanyao = new boolean[SEATS];
  private final boolean[] triplets = new boolean[SEATS];
  private final boolean[] nagashi = new boolean[SEATS];

  /**
   * 席の事実を作る。
   *
   * @param o 判断の記録
   * @param events O の出来事
   * @param tiles 牌ごとの枚数（ドラの倍率を読む）
   */
  void load(Observation o, EventTable events, TileCounts tiles) {
    for (int seat = 0; seat < SEATS; seat++) {
      absoluteScores[o.absoluteSeat(seat)] = o.score(seat);
    }
    ScoreRanking.byScoreThenSeat(absoluteScores, absoluteRanks);
    int roundWind = o.roundWindTile();
    for (int seat = 0; seat < SEATS; seat++) {
      rank[seat] = absoluteRanks[o.absoluteSeat(seat)];
      loadMelds(seat, events, tiles, seatWindTile(o, seat), roundWind);
    }
    Arrays.fill(nagashi, true);
    for (int e = 0, count = events.events(); e < count; e++) {
      if (events.kind(e) == PublicEvent.DISCARD
          && (events.calledBy(e) >= 0 || !Tile.isTerminalOrHonor(events.tile(e)))) {
        nagashi[events.author(e)] = false;
      }
    }
  }

  private void loadMelds(
      int seat, EventTable events, TileCounts tiles, int seatWind, int roundWind) {
    int melds = events.meldCount(seat);
    int kanCount = 0;
    int han = 0;
    int suits = 0;
    int dora = 0;
    boolean closed = true;
    boolean honors = false;
    boolean simples = true;
    boolean noChi = true;
    for (int index = 0; index < melds; index++) {
      Meld meld = events.meld(seat, index);
      int base = meld.baseTileType();
      kanCount += meld.isKan() ? 1 : 0;
      closed &= meld.preservesMenzen();
      simples &= (meld.tileTypeMask() & Tile.TERMINAL_OR_HONOR_TYPE_MASK) == 0L;
      if (meld.type() == Meld.Type.CHI) {
        noChi = false;
      } else {
        han += yakuhai(base, seatWind, roundWind);
      }
      int suit = Tile.suitOf(base);
      if (suit == Tile.SUIT_HONOR) {
        honors = true;
      } else {
        suits |= 1 << suit;
      }
      for (int tile = 0, size = meld.size(); tile < size; tile++) {
        dora += tiles.dora(meld.tileAt(tile));
      }
      dora += meld.containsAkaTile() ? 1 : 0;
    }
    kans[seat] = kanCount;
    yakuhaiHan[seat] = han;
    openDora[seat] = dora;
    menzen[seat] = closed;
    tanyao[seat] = melds > 0 && simples;
    triplets[seat] = melds > 0 && noChi;
    if (melds == 0 || Integer.bitCount(suits) > 1) {
      openSuit[seat] = MIXED;
    } else {
      openSuit[seat] = suits == 0 && honors ? Tile.SUIT_HONOR : Integer.numberOfTrailingZeros(suits);
    }
  }

  /**
   * 牌がその席にとって役牌なら翻を返す（三元牌は 1、風は自風と場風でそれぞれ 1）。
   *
   * @param tile 牌
   * @param seatWind 席の自風の牌
   * @param roundWind 場風の牌
   * @return 翻（0〜2）
   */
  static int yakuhai(int tile, int seatWind, int roundWind) {
    if (Tile.isDragon(tile)) {
      return 1;
    }
    return (tile == seatWind ? 1 : 0) + (tile == roundWind ? 1 : 0);
  }

  /**
   * 相対席の自風の牌を返す（親が東）。
   *
   * @param o 判断の記録
   * @param seat 相対席
   * @return 風の牌
   */
  static int seatWindTile(Observation o, int seat) {
    return Tile.TON + Math.floorMod(seat - o.dealer(), SEATS);
  }

  /** 今の順位（0 始まり、同点は絶対席の順）。 */
  int rank(int seat) {
    return rank[seat];
  }

  int kans(int seat) {
    return kans[seat];
  }

  /** 公開の役牌の刻子・槓子の翻の和。 */
  int yakuhaiHan(int seat) {
    return yakuhaiHan[seat];
  }

  /** 副露の色（{@link #MIXED}、{@link Tile#SUIT_HONOR}、または物理の色）。 */
  int openSuit(int seat) {
    return openSuit[seat];
  }

  /** 副露のドラ（表ドラと赤）。 */
  int openDora(int seat) {
    return openDora[seat];
  }

  /** 門前（暗槓のほかに副露がない）。 */
  boolean menzen(int seat) {
    return menzen[seat];
  }

  /** 副露があり、全て断么と両立する。 */
  boolean tanyao(int seat) {
    return tanyao[seat];
  }

  /** 副露があり、全て刻子か槓子。 */
  boolean triplets(int seat) {
    return triplets[seat];
  }

  /** 流し満貫の形が残っている（打牌が全て么九牌で、鳴かれていない）。 */
  boolean nagashi(int seat) {
    return nagashi[seat];
  }
}
