package com.epsilon.agent.eval;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;

/** 決闘の止め方。どの区間も、牌山ごとの値の平均 ± 臨界値 × 標準誤差。 */
@Choice(key = "kind")
public sealed interface StopRule {

  /**
   * 計画の牌山を全て打つ。判定するのは最後の 1 回だけで、区間は固定の水準。
   *
   * @param alpha 両側の有意水準（0 より大きく 1 より小さい）
   */
  @Case("FIXED")
  record Fixed(double alpha) implements StopRule {
    public Fixed {
      checkAlpha(alpha);
    }
  }

  /**
   * 判定の時点で標準誤差が目標以下なら止める。止めるかどうかを平均を見ずに決めるので、区間は固定の水準のままでよい。
   *
   * @param error 目標の標準誤差（0 以上）
   * @param alpha 両側の有意水準（0 より大きく 1 より小さい）
   * @param checkEvery 判定の時点の間隔（牌山の数、1 以上）
   */
  @Case("PRECISION")
  record Precision(double error, double alpha, int checkEvery) implements StopRule {
    public Precision {
      if (!(error >= 0 && Double.isFinite(error))) {
        throw new IllegalArgumentException("Target error must be finite and non-negative: " + error);
      }
      checkAlpha(alpha);
      if (checkEvery < 1) {
        throw new IllegalArgumentException("Check interval must be positive: " + checkEvery);
      }
    }
  }

  /**
   * 良い・悪い・互角のどれかが出た時点で止める。区間は群逐次検定の同時区間。良い・悪いを互角より先に見る。互角になるのは、
   * 下端 > −（下側の幅）かつ 上端 < 上側の幅 のとき。
   *
   * @param sequential 群逐次検定の境界
   * @param evenBelow 互角の下側の幅（0 以上）
   * @param evenAbove 互角の上側の幅（0 以上）。両方 0 なら互角では止めない
   */
  @Case("TEST")
  record Test(Sequential sequential, double evenBelow, double evenAbove) implements StopRule {
    public Test {
      if (sequential == null) {
        throw new NullPointerException("sequential");
      }
      if (!(evenBelow >= 0 && Double.isFinite(evenBelow)) || !(evenAbove >= 0 && Double.isFinite(evenAbove))) {
        throw new IllegalArgumentException("Even margins must be finite and non-negative: " + evenBelow + ", " + evenAbove);
      }
    }
  }

  /** @return 判定の時点の間隔（牌山の数）。計画の牌山を全て打つ止め方では、計画の牌山の数 */
  default int checkEvery(int walls) {
    return switch (this) {
      case Fixed fixed -> walls;
      case Precision precision -> precision.checkEvery();
      case Test test -> test.sequential().checkEvery();
    };
  }

  private static void checkAlpha(double alpha) {
    if (!(alpha > 0 && alpha < 1)) {
      throw new IllegalArgumentException("Alpha must be in (0, 1): " + alpha);
    }
  }
}
