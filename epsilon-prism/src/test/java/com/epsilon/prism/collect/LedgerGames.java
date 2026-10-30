package com.epsilon.prism.collect;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameRecorder;
import com.epsilon.engine.RoundResult;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** 運の精算の試験の道具: engine を直接回す安い打ち方、仮の枠、何も記録しない記録処理、検定の近似。 */
final class LedgerGames {
  private LedgerGames() {}

  /** 帳簿の試験の仮の記録（判断の席だけを持つ）。 */
  record Seat(int seat) implements Pending {
    @Override
    public void accept(int action, double[] seatValues) {}
  }

  /** 4 席の仮の枠（帳簿つき）。 */
  static List<Slot> slots() {
    List<Slot> slots = new ArrayList<>();
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
      slots.add(new Slot(new Seat(seat), 0, new double[EventLedger.FEATURES]));
    return slots;
  }

  /**
   * 打ち方。
   *
   * @param callRate チー・ポンを選ぶ確率
   * @param kans 槓が合法なら必ず選ぶか
   */
  record Style(double callRate, boolean kans) {}

  static final Style PLAIN = new Style(0.05, false), KANS = new Style(0.3, true), CALLS = new Style(0.6, true);

  /**
   * 判断 point の手の位置: 和了は必ず、kans なら槓、立直は必ず、鳴きは確率 callRate、打牌は打った後の向聴が最小の手（同点は乱数）、九種九牌は
   * 選ばない。手牌と公開の状態と乱数だけで決まる（牌山は読まない）。
   */
  static int choose(
      GameState state, EngineDecisionPoint point, Style style, SplittableRandom random, HandShapeAnalyzer shapes) {
    List<Action> legal = point.legalActions();
    int n = legal.size();
    if (n == 1) return 0;
    for (int a = 0; a < n; a++) {
      Action.Type type = legal.get(a).type();
      if (type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI) return a;
    }
    if (style.kans())
      for (int a = 0; a < n; a++) {
        Action.Type type = legal.get(a).type();
        if (type == Action.Type.ANKAN || type == Action.Type.KAKAN || type == Action.Type.DAIMINKAN) return a;
      }
    for (int a = 0; a < n; a++) if (legal.get(a).type() == Action.Type.RIICHI_DAHAI) return a;
    int pass = -1;
    List<Integer> calls = new ArrayList<>();
    for (int a = 0; a < n; a++) {
      Action.Type type = legal.get(a).type();
      if (type == Action.Type.PASS) pass = a;
      if (type == Action.Type.CHI || type == Action.Type.PON) calls.add(a);
    }
    if (pass >= 0) {
      double coin = random.nextDouble();
      return !calls.isEmpty() && coin < style.callRate() ? calls.get(random.nextInt(calls.size())) : pass;
    }
    int best = Integer.MAX_VALUE;
    List<Integer> ties = new ArrayList<>();
    for (int a = 0; a < n; a++) {
      Action action = legal.get(a);
      if (action.type() != Action.Type.DAHAI) continue;
      int shanten = shapes.calculateMinimumAfterRemoving(state.hand(point.player()), action.tileType());
      if (shanten < best) {
        best = shanten;
        ties.clear();
      }
      if (shanten == best) ties.add(a);
    }
    if (!ties.isEmpty()) return ties.get(random.nextInt(ties.size()));
    for (int a = 0; a < n; a++) if (legal.get(a).type() != Action.Type.KYUSHU_KYUHAI) return a;
    return 0;
  }

  /** 境界の全ての判断を {@link #choose} で選んだ選択（chosen に手の位置も書く。null なら書かない）。 */
  static EngineSelectionBuffer select(
      GameState state,
      List<EngineDecisionPoint> points,
      Style style,
      SplittableRandom random,
      HandShapeAnalyzer shapes,
      int[] chosen) {
    var selections = new EngineSelectionBuffer();
    for (int index = 0; index < points.size(); index++) {
      EngineDecisionPoint point = points.get(index);
      int choice = choose(state, point, style, random, shapes);
      if (chosen != null) chosen[index] = choice;
      selections.add(point.id(), point.legalActions().get(choice));
    }
    return selections;
  }

  /** 何も記録しない記録処理（偶然の点のフックだけを上書きして使う）。 */
  static class SilentRecorder implements GameRecorder {
    @Override
    public void startRound(int roundIndex, int honba, int kyotakuCount, int[] startingScores) {}

    @Override
    public void recordHaipai(int player, int[] concealedTileCounts, int concealedAkaMask) {}

    @Override
    public void recordDoraIndicators(int[] physicalTileIds) {}

    @Override
    public void recordDraw(int player, int tileType, boolean isAkaTile) {}

    @Override
    public void recordDahai(int player, int tileType, boolean isAkaTile, boolean tsumogiri, boolean riichi) {}

    @Override
    public void recordRiichiAccepted(int player) {}

    @Override
    public void recordChi(
        int player, int[] chiTileTypes, int calledTileType, boolean calledTileIsAka, boolean consumedHandTileIsAka) {}

    @Override
    public void recordPon(
        int player, int tileType, int sourcePlayerOffset, boolean calledTileIsAka, boolean consumedHandTileIsAka) {}

    @Override
    public void recordDaiminkan(
        int player, int tileType, int sourcePlayerOffset, boolean calledTileIsAka, boolean consumedHandTileIsAka) {}

    @Override
    public void recordAnkan(int player, int tileType, boolean containsAkaTile) {}

    @Override
    public void recordKakan(int player, int tileType, boolean addedTileIsAka) {}

    @Override
    public void recordRoundResult(RoundResult result) {}

    @Override
    public void endRound(RoundEnd roundEnd) {}
  }

  /** 標準正規の上側確率。 */
  static double normalUpper(double z) {
    return 0.5 * erfc(z / Math.sqrt(2));
  }

  /** カイ二乗（自由度 degrees）の上側確率（Wilson–Hilferty の近似）。 */
  static double chiSquareUpper(double statistic, int degrees) {
    double k = degrees;
    double z = (Math.cbrt(statistic / k) - (1 - 2 / (9 * k))) / Math.sqrt(2 / (9 * k));
    return normalUpper(z);
  }

  /** カイ二乗の統計量（期待が 0 の区分は観測も 0 であること）。 */
  static double chiSquare(long[] observed, double[] expected) {
    double statistic = 0;
    for (int i = 0; i < observed.length; i++) {
      if (expected[i] == 0) {
        if (observed[i] != 0) return Double.POSITIVE_INFINITY;
        continue;
      }
      double d = observed[i] - expected[i];
      statistic += d * d / expected[i];
    }
    return statistic;
  }

  /** 期待が正の区分の数 − 1（自由度）。 */
  static int degrees(double[] expected) {
    int k = 0;
    for (double e : expected) if (e > 0) k++;
    return Math.max(1, k - 1);
  }

  /** erfc（Numerical Recipes の Chebyshev 近似、相対誤差 1.2e-7）。 */
  private static double erfc(double x) {
    double z = Math.abs(x);
    double t = 1 / (1 + 0.5 * z);
    double r =
        t
            * Math.exp(
                -z * z
                    - 1.26551223
                    + t
                        * (1.00002368
                            + t
                                * (0.37409196
                                    + t
                                        * (0.09678418
                                            + t
                                                * (-0.18628806
                                                    + t
                                                        * (0.27886807
                                                            + t
                                                                * (-1.13520398
                                                                    + t
                                                                        * (1.48851587
                                                                            + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
}
