package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.grp.ScoreSmooth;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 点数のなめらかな関数の Φ（{@link ScoreSmooth}）で使う、最終の点のばらつきの係数を、半荘の記録から測った値。半荘が続く局末の
 * 席ごとに、局末から半荘の終わりまでの点の動き（最終の点 − 局末の点）を取る。係数は、点の動きを √残りの局の数で割った値の標準偏差。残りの局の数は Φ と同じ数え方で
 * 数える（{@link ScoreSmooth#remainingRounds}）。次局ごとにも、点の動きの標準偏差と、それを √残りの局の数で割った値を出す。
 * 標準偏差は全て、平均を引いた不偏分散の平方根。
 *
 * @param rounds 次局ごとの値（次局の順。席 × 局末が 2 つ未満の次局は入れない）
 * @param seatHands 席 × 局末の数
 * @param perHand 係数（点）。席 × 局末が 2 つ未満なら NaN
 */
public record SmoothSigma(Variant variant, List<Round> rounds, long seatHands, double perHand) {

  public SmoothSigma {
    Objects.requireNonNull(variant, "variant");
    rounds = List.copyOf(rounds);
  }

  /**
   * 次局 1 つの値。
   *
   * @param round 次局の番号（東 1 局が 0）
   * @param remaining 残りの局の数
   * @param seatHands 席 × 局末の数
   * @param spread 点の動きの標準偏差（点）
   */
  public record Round(int round, int remaining, long seatHands, double spread) {

    /** 点の動きの標準偏差を √残りの局の数で割った値（点）。 */
    public double perHand() {
      return spread / Math.sqrt(remaining);
    }
  }

  /** 半荘を足していき、最後に値を出すビルダーを返す。 */
  public static Builder builder(Variant variant) {
    return new Builder(variant);
  }

  /** 半荘をコホートごとに足す。足した半荘は持たない。 */
  public static final class Builder {

    private final Variant variant;

    /** 次局ごとの点の動き。 */
    private final Moments[] rounds;

    /** 点の動きを √残りの局の数で割った値。 */
    private final Moments scaled = new Moments();

    private Builder(Variant variant) {
      this.variant = Objects.requireNonNull(variant, "variant");
      rounds = new Moments[variant.finalRound() + 1];
      for (int round = 0; round < rounds.length; round++) {
        rounds[round] = new Moments();
      }
    }

    /** 半荘を足す。ルールの違う半荘があれば IllegalArgumentException を投げ、何も足さない。 */
    public Builder add(List<GameRecord> games) {
      for (GameRecord game : games) {
        if (game.variant() != variant) {
          throw new IllegalArgumentException("A " + game.variant() + " game among " + variant + " games");
        }
      }
      int seats = variant.seats();
      for (GameRecord game : games) {
        for (GrpState state : game.handEnds()) {
          int round = state.round();
          double scale = Math.sqrt(ScoreSmooth.remainingRounds(variant, round));
          for (int seat = 0; seat < seats; seat++) {
            double change = game.finalScore(seat) - state.score(seat);
            rounds[round].add(change);
            scaled.add(change / scale);
          }
        }
      }
      return this;
    }

    public SmoothSigma build() {
      List<Round> measured = new ArrayList<>();
      for (int round = 0; round < rounds.length; round++) {
        if (rounds[round].count >= 2) {
          measured.add(
              new Round(
                  round,
                  ScoreSmooth.remainingRounds(variant, round),
                  rounds[round].count,
                  rounds[round].deviation()));
        }
      }
      return new SmoothSigma(variant, measured, scaled.count, scaled.deviation());
    }
  }

  /** 値の数と平均と、平均からの差の二乗の和（Welford の方法で足す）。 */
  private static final class Moments {

    private long count;
    private double mean;
    private double squares;

    void add(double value) {
      count++;
      double step = value - mean;
      mean += step / count;
      squares += step * (value - mean);
    }

    /** 不偏分散の平方根。値が 2 つ未満なら NaN。 */
    double deviation() {
      return count < 2 ? Double.NaN : Math.sqrt(squares / (count - 1));
    }
  }
}
