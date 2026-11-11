package com.epsilon.classic.campaign;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.classic.learn.KlLearningRate;
import com.epsilon.classic.learn.SelfPlayLearner;
import java.util.Objects;

/**
 * 学習の段の報告。有限でない平均は書かない（数える行が無ければ無い）。
 *
 * @param version 学習器が学び始めたバージョン（公開したのは version + 1）
 * @param games 半荘の数
 * @param decisions 判断の数
 * @param actorRows 方策の損失に入れた判断の数
 * @param steps ステップの数
 * @param actorSteps 方策を更新したステップの数
 * @param kl 方策の行のあるステップの、更新の前に測った KL の平均
 * @param valueLoss 価値の交差エントロピーの、判断ごとの平均
 * @param entropy 方策のエントロピーの、方策の行ごとの平均
 * @param credit 方策の行の、選んだ手の補正の平均
 * @param advantage 方策の行の優位の平均
 * @param rate 方策の学習率の動き
 */
public record LearnReport(
    int version,
    int games,
    int decisions,
    int actorRows,
    int steps,
    int actorSteps,
    double kl,
    @Key(doc = "価値の交差エントロピーの、判断ごとの平均", presence = Presence.ABSENT) Double valueLoss,
    @Key(doc = "方策のエントロピーの、方策の行ごとの平均", presence = Presence.ABSENT) Double entropy,
    @Key(doc = "方策の行の、選んだ手の補正の平均", presence = Presence.ABSENT) Double credit,
    @Key(doc = "方策の行の優位の平均", presence = Presence.ABSENT) Double advantage,
    Rate rate) {

  public LearnReport {
    Objects.requireNonNull(rate, "rate");
  }

  /**
   * 方策の学習率の動き。
   *
   * @param action 決めた動き
   * @param applied この macro の学習率
   * @param next 次の macro の学習率
   */
  public record Rate(KlLearningRate.Action action, double applied, double next) {

    public Rate {
      Objects.requireNonNull(action, "action");
    }
  }

  /**
   * 1 回の macro の学習の記録から作る。
   *
   * @param version 学習器が学び始めたバージョン
   */
  static LearnReport of(int version, SelfPlayLearner.Macro macro) {
    double valueSum = 0;
    int rows = 0;
    double entropySum = 0;
    int actorRows = 0;
    for (SelfPlayLearner.Step step : macro.steps()) {
      valueSum += step.valueLoss() * step.rows();
      rows += step.rows();
      entropySum += step.entropy() * step.actorRows();
      actorRows += step.actorRows();
    }
    KlLearningRate.Adjustment adjustment = macro.rate();
    return new LearnReport(
        version,
        macro.games(),
        macro.decisions(),
        macro.actorRows(),
        macro.steps().size(),
        macro.actorSteps(),
        macro.kl(),
        finite(rows == 0 ? Double.NaN : valueSum / rows),
        finite(actorRows == 0 ? Double.NaN : entropySum / actorRows),
        finite(macro.credit()),
        finite(macro.advantage()),
        new Rate(adjustment.action(), adjustment.applied(), adjustment.next()));
  }

  private static Double finite(double value) {
    return Double.isFinite(value) ? value : null;
  }
}
