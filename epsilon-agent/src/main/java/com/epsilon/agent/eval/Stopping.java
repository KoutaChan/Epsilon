package com.epsilon.agent.eval;

import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

/**
 * 止め方を計画の牌山の数に当てはめて作った、判定の時点の列。時点 k の判定は、牌山 0〜(その時点の牌山の数 − 1) の値が全てそろってから
 * 行う。渡す値は、候補が良いほど大きくなる向きにする（順位なら符号を反す）。
 */
public final class Stopping {

  private final StopRule rule;
  private final int[] walls;
  private final double[] critical;

  private Stopping(StopRule rule, int[] walls, double[] critical) {
    this.rule = rule;
    this.walls = walls;
    this.critical = critical;
  }

  /**
   * @param walls 計画の牌山の数。計画の牌山を全て打つ止め方でなければ、判定の間隔の正の倍数
   * @throws IllegalArgumentException 牌山の数が正でないか、間隔の倍数でないとき
   */
  public static Stopping of(StopRule rule, int walls) {
    Objects.requireNonNull(rule, "rule");
    if (walls < 1) {
      throw new IllegalArgumentException("Walls must be positive: " + walls);
    }
    return switch (rule) {
      case StopRule.Fixed fixed -> new Stopping(rule, new int[] {walls}, new double[] {Normal.upperPoint(fixed.alpha() / 2)});
      case StopRule.Precision precision -> {
        int checkEvery = precision.checkEvery();
        if (walls % checkEvery != 0) {
          throw new IllegalArgumentException(walls + " walls are not a multiple of the check interval " + checkEvery);
        }
        double[] critical = new double[walls / checkEvery];
        Arrays.fill(critical, Normal.upperPoint(precision.alpha() / 2));
        yield new Stopping(rule, schedule(critical.length, checkEvery), critical);
      }
      case StopRule.Test test -> {
        Sequential sequential = test.sequential();
        double[] critical = sequential.boundaries(walls);
        yield new Stopping(rule, schedule(critical.length, sequential.checkEvery()), critical);
      }
    };
  }

  /**
   * 計画に記録した臨界値で作る。判定の時点は止め方と牌山の数から決め、臨界値だけ記録の値を使う。記録した後に臨界値の求め方が変わっても、
   * 始めた決闘は同じ境界のまま続けられる。
   *
   * @param walls 計画の牌山の数。計画の牌山を全て打つ止め方でなければ、判定の間隔の正の倍数
   * @param critical 判定の時点ごとの臨界値（正の有限の数）
   * @throws IllegalArgumentException 牌山の数が間隔の正の倍数でないか、臨界値の数が判定の時点の数と違うか、正の有限の数でない臨界値が
   *     あるとき
   */
  public static Stopping recorded(StopRule rule, int walls, double[] critical) {
    Objects.requireNonNull(rule, "rule");
    if (walls < 1) {
      throw new IllegalArgumentException("Walls must be positive: " + walls);
    }
    int checkEvery = rule.checkEvery(walls);
    if (walls % checkEvery != 0) {
      throw new IllegalArgumentException(walls + " walls are not a multiple of the check interval " + checkEvery);
    }
    if (critical.length != walls / checkEvery) {
      throw new IllegalArgumentException(
          critical.length + " critical values were recorded for " + walls / checkEvery + " looks");
    }
    for (int look = 0; look < critical.length; look++) {
      if (!(critical[look] > 0 && Double.isFinite(critical[look]))) {
        throw new IllegalArgumentException("Invalid critical value at look " + look + ": " + critical[look]);
      }
    }
    return new Stopping(rule, schedule(critical.length, checkEvery), critical.clone());
  }

  private static int[] schedule(int looks, int checkEvery) {
    int[] walls = new int[looks];
    for (int k = 0; k < looks; k++) {
      walls[k] = (k + 1) * checkEvery;
    }
    return walls;
  }

  public StopRule rule() {
    return rule;
  }

  /** @return 判定の時点の数 */
  public int looks() {
    return walls.length;
  }

  /**
   * @param look 判定の時点（0 から）
   * @return その時点までに打ち終える牌山の数
   */
  public int wallsAt(int look) {
    return walls[look];
  }

  /**
   * @param look 判定の時点（0 から）
   * @return その時点の区間の臨界値
   */
  public double critical(int look) {
    return critical[look];
  }

  /**
   * @param look 判定の時点（0 から）
   * @param values 牌山 0〜(その時点の牌山の数 − 1) の値を番号の順に足したもの
   * @return 止めるなら判定。最後の時点では必ず返す
   * @throws IllegalArgumentException 足した牌山の数がその時点の牌山の数と違うとき
   */
  public Optional<Decision> decide(int look, WallMean values) {
    if (values.walls() != walls[look]) {
      throw new IllegalArgumentException("Look " + look + " needs " + walls[look] + " walls, have " + values.walls());
    }
    return decide(look, values.mean(), values.error());
  }

  /**
   * 良い・悪いを互角より先に見る。標準誤差が 0 なら区間は平均の 1 点になる。
   *
   * @param look 判定の時点（0 から）
   * @param mean 牌山ごとの値の平均
   * @param error 平均の標準誤差（有限で 0 以上）
   * @return 止めるなら判定。最後の時点では必ず返す
   */
  public Optional<Decision> decide(int look, double mean, double error) {
    if (look < 0 || look >= walls.length) {
      throw new IllegalArgumentException("Look " + look + " is not among " + walls.length);
    }
    if (!Double.isFinite(mean) || !(error >= 0 && Double.isFinite(error))) {
      throw new IllegalArgumentException("Invalid estimate: " + mean + " ± " + error);
    }
    Estimate estimate = Estimate.of(mean, error, critical[look]);
    Verdict sided = estimate.low() > 0 ? Verdict.BETTER : estimate.high() < 0 ? Verdict.WORSE : Verdict.UNDECIDED;
    boolean last = look == walls.length - 1;
    return switch (rule) {
      case StopRule.Fixed fixed -> Optional.of(new Decision(sided, Stopped.LIMIT, estimate));
      case StopRule.Precision precision -> {
        if (error <= precision.error()) {
          yield Optional.of(new Decision(sided, Stopped.PRECISE, estimate));
        }
        yield last ? Optional.of(new Decision(sided, Stopped.LIMIT, estimate)) : Optional.empty();
      }
      case StopRule.Test test -> {
        Verdict verdict = sided;
        if (verdict == Verdict.UNDECIDED && estimate.low() > -test.evenBelow() && estimate.high() < test.evenAbove()) {
          verdict = Verdict.EVEN;
        }
        if (verdict != Verdict.UNDECIDED) {
          yield Optional.of(new Decision(verdict, Stopped.DECIDED, estimate));
        }
        yield last ? Optional.of(new Decision(Verdict.UNDECIDED, Stopped.LIMIT, estimate)) : Optional.empty();
      }
    };
  }

  /**
   * 判定の時点で止めると決めた結果。
   *
   * @param estimate 止めた時点の平均と、止め方の区間
   */
  public record Decision(Verdict verdict, Stopped stopped, Estimate estimate) {}
}
