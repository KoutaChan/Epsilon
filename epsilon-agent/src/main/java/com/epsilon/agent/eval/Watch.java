package com.epsilon.agent.eval;

import java.util.Objects;

/**
 * 強さのリグレッションを監視する設定。
 *
 * @param alarmZ 基準からの低下と、ルールベースの打ち手の卓での低下をアラートにするしきい値（正）。同じ牌山で対にした差の平均を、その標準誤差で割った値と比べる
 * @param margin 基準からの低下のアラートに使う幅（0 以上）。対の差の平均がこの値にマイナスを付けた値以下のときだけアラートにする
 * @param onWorse 王者との決闘で「悪い」が続いたときの扱い（続いた回数は 1 以上）
 */
public record Watch(double alarmZ, double margin, OnWorseSettings onWorse) {

  public Watch {
    if (!(alarmZ > 0 && Double.isFinite(alarmZ))) {
      throw new IllegalArgumentException("Alarm z must be positive: " + alarmZ);
    }
    if (!(margin >= 0 && Double.isFinite(margin))) {
      throw new IllegalArgumentException("Margin must be finite and non-negative: " + margin);
    }
    Objects.requireNonNull(onWorse, "onWorse");
    if (onWorse.streak() < 1) {
      throw new IllegalArgumentException("Worse streak must be positive: " + onWorse.streak());
    }
  }

  /** 評価の節の監視の設定。 */
  public static Watch of(EvaluationSettings evaluation) {
    return new Watch(evaluation.alarmZ(), evaluation.dropMargin(), evaluation.onWorse());
  }
}
