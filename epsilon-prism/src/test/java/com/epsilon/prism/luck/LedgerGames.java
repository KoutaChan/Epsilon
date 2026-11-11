package com.epsilon.prism.luck;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RandomGames;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.noise.LedgerGroup;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * 運の精算のテスト用のヘルパー。新しいエンジンの合法手の番号で選ぶ軽い打ち方、席別ツモ列の半荘を続けて打つ歩き手、席ごとの仮の受け取り先、
 * 検定の近似式をまとめる。
 */
final class LedgerGames {
  private LedgerGames() {}

  /** 古典の 5 組を全て計算し、点数を 32 千点で、方策の s を 8 で切る帳簿（同梱の設定の既定値と同じ）。 */
  static final LedgerRecording RECORDING = new LedgerRecording(LedgerGroup.computed(), 32, 8);

  /**
   * 帳簿のテストの仮の受け取り先（席と帳簿だけを持つ）。
   *
   * @param seat 増分を足す向きの席
   * @param ledger 列ごとの帳簿
   */
  record Seat(int seat, double[] ledger) implements LedgerTarget {}

  /** 席ごとの仮の受け取り先（帳簿つき）。 */
  static List<Seat> seats(Variant variant) {
    List<Seat> seats = new ArrayList<>();
    for (int seat = 0; seat < variant.seats(); seat++) seats.add(new Seat(seat, new double[LedgerColumn.COUNT]));
    return seats;
  }

  /**
   * 打ち方。
   *
   * @param callRate チー・ポンを選ぶ確率
   * @param kans 槓と抜きが合法なら必ず選ぶか
   */
  record Style(double callRate, boolean kans) {}

  static final Style PLAIN = new Style(0.05, false), KANS = new Style(0.3, true), CALLS = new Style(0.6, true);

  /**
   * 打ち方に従って手を選ぶ。和了は必ず選び、{@code kans} なら槓と抜きも必ず選ぶ。立直は必ず選ぶ。鳴きは確率 {@code callRate} で選ぶ。
   * 打牌は、打った後の向聴が最小の手から選ぶ（同点は乱数）。九種九牌は選ばない。手牌と公開の状態と乱数だけで決まる（牌山は読まない）。
   */
  static RandomGames.Chooser chooser(Variant variant, Style style) {
    ShapeAnalyzer shapes = new ShapeAnalyzer(variant);
    HandShape hand = new HandShape(), after = new HandShape();
    List<Integer> picks = new ArrayList<>();
    return (engine, slot, random) -> {
      Decisions decisions = engine.decisions();
      int n = decisions.legalCount(slot);
      if (n == 1) return 0;
      for (int a = 0; a < n; a++) {
        Action.Type type = type(decisions, slot, a);
        if (type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI) return a;
      }
      if (style.kans())
        for (int a = 0; a < n; a++) {
          Action.Type type = type(decisions, slot, a);
          if (type == Action.Type.ANKAN || type == Action.Type.KAKAN || type == Action.Type.DAIMINKAN
              || type == Action.Type.NUKI) return a;
        }
      for (int a = 0; a < n; a++) if (type(decisions, slot, a) == Action.Type.RIICHI_DAHAI) return a;
      int pass = -1;
      picks.clear();
      for (int a = 0; a < n; a++) {
        Action.Type type = type(decisions, slot, a);
        if (type == Action.Type.PASS) pass = a;
        if (type == Action.Type.CHI || type == Action.Type.PON) picks.add(a);
      }
      if (pass >= 0) {
        double coin = random.nextDouble();
        return !picks.isEmpty() && coin < style.callRate() ? picks.get(random.nextInt(picks.size())) : pass;
      }
      engine.full().copyHand(decisions.seat(slot), hand);
      int best = Integer.MAX_VALUE;
      for (int a = 0; a < n; a++) {
        Action action = Action.of(decisions.legal(slot, a));
        if (action.type() != Action.Type.DAHAI) continue;
        after.copyFrom(hand);
        after.remove(action.tile());
        int shanten = shapes.shanten(after);
        if (shanten < best) {
          best = shanten;
          picks.clear();
        }
        if (shanten == best) picks.add(a);
      }
      if (!picks.isEmpty()) return picks.get(random.nextInt(picks.size()));
      for (int a = 0; a < n; a++) if (type(decisions, slot, a) != Action.Type.KYUSHU_KYUHAI) return a;
      return 0;
    };
  }

  static Action.Type type(Decisions decisions, int slot, int index) {
    return Action.of(decisions.legal(slot, index)).type();
  }

  /** 歩き手から、判断の時点と局末を受け取る。エンジンの状態を変えてはならない（写しは作ってよい）。 */
  interface Visitor {
    void decision(Engine engine, RuleWorkspace ws);

    /** 局末か終局（次の局を配る前）。 */
    default void roundEnd(Engine engine) {}
  }

  /** 席別ツモ列の半荘を乱数の種から続けて打ち、指定した局数を打ったら止める。 */
  static void walk(Variant variant, long seed, Style style, int rounds, Visitor visitor) {
    Random random = new Random(seed);
    RandomGames.Chooser chooser = chooser(variant, style);
    RuleWorkspace ws = new RuleWorkspace(variant);
    Engine engine = new Engine(variant);
    engine.startGame(random.nextLong(), DrawOrder.SEAT_STREAMS);
    Boundary boundary = engine.advance(ws);
    int[] choices = new int[variant.seats()];
    for (int played = 0; played < rounds; ) {
      switch (boundary) {
        case DECISIONS -> {
          visitor.decision(engine, ws);
          Decisions decisions = engine.decisions();
          for (int slot = 0; slot < decisions.size(); slot++) choices[slot] = chooser.choose(engine, slot, random);
          boundary = engine.commit(choices, ws);
        }
        case ROUND_END -> {
          visitor.roundEnd(engine);
          played++;
          boundary = engine.advance(ws);
        }
        case GAME_END -> {
          visitor.roundEnd(engine);
          played++;
          engine.startGame(random.nextLong(), DrawOrder.SEAT_STREAMS);
          boundary = engine.advance(ws);
        }
      }
    }
  }

  /**
   * 指定したイベントの直前のプール（局の始めの枚数から、それより前のイベントの牌を引いたもの）。帳簿の計算とは別のやり方で数える。
   *
   * @param pool 種類ごとの枚数を書く先
   */
  static void poolBefore(ChanceLog log, int index, Variant variant, int[] pool) {
    Arrays.fill(pool, 0);
    for (long bits = variant.tiles().kinds(); bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      pool[kind] = log.start(kind);
    }
    for (int i = 0; i < index; i++) pool[Tiles.kind(log.tile(i))]--;
  }

  /** ツモ（山・嶺上・抜きの補充）か。 */
  static boolean draw(ChanceKind kind) {
    return kind == ChanceKind.DRAW || kind == ChanceKind.RINSHAN;
  }

  /** 標準正規分布の上側確率。 */
  static double normalUpper(double z) {
    return 0.5 * erfc(z / Math.sqrt(2));
  }

  /** 指定した自由度のカイ二乗分布の上側確率（Wilson–Hilferty の近似）。 */
  static double chiSquareUpper(double statistic, int degrees) {
    double k = degrees;
    double z = (Math.cbrt(statistic / k) - (1 - 2 / (9 * k))) / Math.sqrt(2 / (9 * k));
    return normalUpper(z);
  }

  /** カイ二乗の統計量。期待度数が 0 の区分は観測も 0 でなければならず、0 でなければ正の無限大を返す。 */
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

  /** 自由度（期待度数が正の区分の数 − 1。最小 1）。 */
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
