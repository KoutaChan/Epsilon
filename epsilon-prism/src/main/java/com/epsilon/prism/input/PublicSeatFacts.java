package com.epsilon.prism.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.input.EventTable.PublicEvent;
import java.util.Arrays;

/**
 * 席ごとの公開の事実のうち、PRISM の入力だけが使うもの（副露の色・公開の役牌の翻・副露のドラ・断么・対々・流し満貫）。順位・槓の数・門前と
 * 今の面子は {@link SeatFacts} から読む。判断ごとに 1 回読み込み、そのときメモリを確保しない。席は観測者から見た相対席。
 *
 * <p>副露の色は物理の色で持ち、書くときに正規化した色に置き換える。{@link #openSuit} の値は、副露がないか色が混ざれば {@link #MIXED}、
 * 字牌だけなら {@link #HONORS}、1 つの色と字牌だけならその物理の色の番号。
 */
public final class PublicSeatFacts {

  /** 副露がないか、2 つ以上の色が混ざっている。 */
  public static final int MIXED = -1;

  /** 副露が字牌だけ。 */
  public static final int HONORS = Tiles.NUMBER_SUITS;

  private final int[] yakuhaiHan;
  private final int[] openSuit;
  private final int[] openDora;
  private final boolean[] tanyao;
  private final boolean[] triplets;
  private final boolean[] nagashi;

  /**
   * 席の数だけの配列を確保して作る。
   *
   * @param seats 席の数
   */
  public PublicSeatFacts(int seats) {
    yakuhaiHan = new int[seats];
    openSuit = new int[seats];
    openDora = new int[seats];
    tanyao = new boolean[seats];
    triplets = new boolean[seats];
    nagashi = new boolean[seats];
  }

  /**
   * 判断の記録から、席ごとの事実を読み込む。前の判断の値は上書きする。
   *
   * @param o 判断の記録
   * @param events 判断の記録にある、公開されたイベント
   * @param seats 席ごとの公開情報（今の面子を読む）
   * @param tiles 種類ごとの枚数（ドラの重みを読む）
   */
  public void load(Observation o, EventTable events, SeatFacts seats, TileCounts tiles) {
    int roundWind = o.roundWindTile();
    for (int seat = 0; seat < openSuit.length; seat++) {
      loadMelds(seat, seats, tiles, seatWindTile(o, seat), roundWind);
    }
    Arrays.fill(nagashi, true);
    for (int e = 0, count = events.events(); e < count; e++) {
      if (events.kind(e) == PublicEvent.DISCARD
          && (events.calledBy(e) >= 0 || !Tiles.contains(Tiles.TERMINALS_AND_HONORS, events.tile(e)))) {
        nagashi[events.author(e)] = false;
      }
    }
  }

  private void loadMelds(int seat, SeatFacts seats, TileCounts tiles, int seatWind, int roundWind) {
    int melds = seats.meldCount(seat);
    int han = 0;
    int suits = 0;
    int dora = 0;
    boolean honors = false;
    boolean simples = true;
    boolean noChi = true;
    for (int index = 0; index < melds; index++) {
      Meld meld = seats.meld(seat, index);
      int base = meld.base();
      simples &= (meld.kinds() & Tiles.TERMINALS_AND_HONORS) == 0L;
      if (meld.type() == Meld.Type.CHI) {
        noChi = false;
      } else {
        han += yakuhai(base, seatWind, roundWind);
      }
      Tiles.Group group = Tiles.Group.of(base);
      if (group.suited()) {
        suits |= 1 << group.ordinal();
      } else {
        honors = true;
      }
      for (int tile = 0, size = meld.size(); tile < size; tile++) {
        dora += tiles.dora(meld.tile(tile));
      }
      dora += meld.red() ? 1 : 0;
    }
    yakuhaiHan[seat] = han;
    openDora[seat] = dora;
    tanyao[seat] = melds > 0 && simples;
    triplets[seat] = melds > 0 && noChi;
    if (melds == 0 || Integer.bitCount(suits) > 1) {
      openSuit[seat] = MIXED;
    } else {
      openSuit[seat] = suits == 0 && honors ? HONORS : Integer.numberOfTrailingZeros(suits);
    }
  }

  /**
   * 牌がその席にとって役牌なら翻を返す（三元牌は 1、風は自風と場風でそれぞれ 1）。
   *
   * @param tile 牌の種類
   * @param seatWind 席の自風の種類
   * @param roundWind 場風の種類
   * @return 翻（0〜2）
   */
  public static int yakuhai(int tile, int seatWind, int roundWind) {
    if (Tiles.Group.of(tile) == Tiles.Group.DRAGON) {
      return 1;
    }
    return (tile == seatWind ? 1 : 0) + (tile == roundWind ? 1 : 0);
  }

  /** 相対席の自風の種類（親が東）。 */
  public static int seatWindTile(Observation o, int seat) {
    return Tiles.TON + Math.floorMod(seat - o.dealer(), o.variant().seats());
  }

  /** 公開の役牌の刻子・槓子の翻の和。 */
  public int yakuhaiHan(int seat) {
    return yakuhaiHan[seat];
  }

  /** 副露の色（{@link #MIXED}、{@link #HONORS}、または物理の色の番号）。 */
  public int openSuit(int seat) {
    return openSuit[seat];
  }

  /** 副露のドラ（表ドラと赤）。 */
  public int openDora(int seat) {
    return openDora[seat];
  }

  /** 副露があり、どれも断么と両立する。 */
  public boolean tanyao(int seat) {
    return tanyao[seat];
  }

  /** 副露があり、全て刻子か槓子。 */
  public boolean triplets(int seat) {
    return triplets[seat];
  }

  /** 流し満貫の形が残っている（打牌が全て么九牌で、鳴かれていない）。 */
  public boolean nagashi(int seat) {
    return nagashi[seat];
  }
}
