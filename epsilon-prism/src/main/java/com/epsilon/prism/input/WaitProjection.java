package com.epsilon.prism.input;

import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.core.GameState;
import com.epsilon.core.ScoreRanking;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.WinPointDeltas;
import com.epsilon.prism.input.InputSchema.DiscardExact;
import com.epsilon.prism.input.WaitScoring.Stance;
import com.epsilon.prism.input.WaitScoring.WinningTile;
import com.epsilon.prism.observe.Observation;
import java.util.Arrays;

/**
 * 聴牌の遷移の B の道（{@link DiscardExact}）。待ちの採点から、ロン・ツモの和了それぞれで半荘が終わる割合・終わるときの効用・得点の平均を作る。
 *
 * <p>和了の 1 つ 1 つは、待ち ×（赤でない和了牌・赤の 5）× 元（ロンは他家 3 席、ツモは自分）で、重みはその和了牌の見えない枚数（ロンの元は等しく
 * 数える）。点の移動は {@link WinPointDeltas}（本場込み、包なし）、終局は {@link HanchanProgression#shouldFinishAfterSingleWin}（供託を
 * 受け取る前の点）、順位は供託を受け取った後の点の {@link ScoreRanking}（同点は絶対席の順）、効用は {@link InputSchema#UTILITY}。
 *
 * <p>得点は、判断の時点から和了の後までの自分の点の変化（本場・供託を含む。立直の遷移は立直棒の 1000 点を払って供託を 1 本足し、和了で取り戻す
 * ので、立直棒は差し引きで消える）。役のない和了と、振聴で決まってロンできない遷移のロンは数えない。数える和了がなければその方法の欄は 0。
 */
final class WaitProjection {

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final DiscardExact[] FIELDS = DiscardExact.values();
  private static final WinningTile[] WINNING_TILES = WinningTile.values();

  private final float[] values = new float[FIELDS.length];
  private final int[] scores = new int[SEATS];
  private final int[] deltas = new int[SEATS];
  private final int[] absolute = new int[SEATS];
  private final int[] ranks = new int[SEATS];
  private final long[] endsByRank = new long[SEATS];
  private int observer;
  private int dealer;
  private int dealerAbsolute;
  private int kyoku;
  private int honba;
  private int kyotaku;
  private long weight;
  private long points;

  /**
   * 判断の点と局を読む。
   *
   * @param o 判断の記録
   */
  void load(Observation o) {
    observer = o.observer();
    dealer = o.dealer();
    dealerAbsolute = o.absoluteSeat(dealer);
    kyoku = o.kyoku();
    honba = o.honba();
    kyotaku = o.kyotaku();
    for (int seat = 0; seat < SEATS; seat++) {
      scores[seat] = o.score(seat);
    }
  }

  /**
   * B の道を作る。
   *
   * @param waits 後の手の待ちの採点
   * @param stance その遷移の立直の構え
   * @param declaresRiichi その遷移で立直を宣言する（立直棒を払う）
   * @param ronBlocked 振聴でロンできない
   * @param tiles 牌ごとの枚数（見えない枚数と見えた赤を読む）
   * @param waitMask 形の待ち
   */
  void project(
      WaitScoring waits,
      Stance stance,
      boolean declaresRiichi,
      boolean ronBlocked,
      TileCounts tiles,
      long waitMask) {
    Arrays.fill(values, 0f);
    values[DiscardExact.TENPAI.ordinal()] = 1f;
    values[DiscardExact.LIVE_WAITS.ordinal()] = tiles.liveCopies(waitMask);
    if (!ronBlocked) {
      accumulate(waits, stance, WinMethod.RON, declaresRiichi, tiles);
      store(DiscardExact.RON_ENDS, DiscardExact.RON_UTILITY, DiscardExact.RON_POINTS);
    }
    accumulate(waits, stance, WinMethod.TSUMO, declaresRiichi, tiles);
    store(DiscardExact.TSUMO_ENDS, DiscardExact.TSUMO_UTILITY, DiscardExact.TSUMO_POINTS);
  }

  /** 作った欄の値（正規化の前）。 */
  float value(DiscardExact field) {
    return values[field.ordinal()];
  }

  /** 和了を重みつきで足す。和は整数で持つので、待ちの順（物理の色の順）によらず同じ値になる。 */
  private void accumulate(
      WaitScoring waits, Stance stance, WinMethod method, boolean declaresRiichi, TileCounts tiles) {
    weight = 0;
    points = 0;
    Arrays.fill(endsByRank, 0L);
    for (int wait = 0, count = waits.waits(); wait < count; wait++) {
      int tile = waits.waitTile(wait);
      int akaCopies = tiles.akaUnseen(tile) ? 1 : 0;
      for (WinningTile winning : WINNING_TILES) {
        int copies = winning == WinningTile.AKA ? akaCopies : tiles.unseen(tile) - akaCopies;
        int base = waits.basePoints(wait, stance, method, winning);
        if (copies == 0 || base == 0) {
          continue;
        }
        if (method == WinMethod.TSUMO) {
          outcome(base, 0, copies, declaresRiichi);
        } else {
          for (int from = 1; from < SEATS; from++) {
            outcome(base, from, copies, declaresRiichi);
          }
        }
      }
    }
  }

  /** 和了 1 つ（自分が base 点で、from から。from が 0 ならツモ）を重み copies で足す。 */
  private void outcome(int base, int from, int copies, boolean declaresRiichi) {
    WinPointDeltas.settleInto(base, 0, from, dealer, honba, 0, deltas);
    int stick = declaresRiichi ? HanchanProgression.RIICHI_COST : 0;
    for (int seat = 0; seat < SEATS; seat++) {
      absolute[(observer + seat) % SEATS] = scores[seat] + deltas[seat] - (seat == 0 ? stick : 0);
    }
    boolean finished =
        HanchanProgression.shouldFinishAfterSingleWin(
            kyoku, dealerAbsolute, observer, absolute[0], absolute[1], absolute[2], absolute[3]);
    int sticks = kyotaku + (declaresRiichi ? 1 : 0);
    absolute[observer] += sticks * HanchanProgression.RIICHI_COST;
    weight += copies;
    points += (long) copies * (absolute[observer] - scores[0]);
    if (finished) {
      ScoreRanking.byScoreThenSeat(absolute, ranks);
      endsByRank[ranks[observer]] += copies;
    }
  }

  private void store(DiscardExact endsField, DiscardExact utilityField, DiscardExact pointsField) {
    if (weight == 0) {
      return;
    }
    long ends = 0;
    double utility = 0;
    for (int rank = 0; rank < SEATS; rank++) {
      ends += endsByRank[rank];
      utility += endsByRank[rank] * InputSchema.UTILITY.utilityForRankDouble(rank);
    }
    values[endsField.ordinal()] = (float) ((double) ends / weight);
    values[utilityField.ordinal()] = (float) (utility / weight);
    values[pointsField.ordinal()] = (float) ((double) points / weight);
  }
}
