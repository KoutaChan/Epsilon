package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.ScoreRanking;
import com.epsilon.prism.collect.HandValue;
import java.util.List;

/**
 * 点数だけで決まる Φ（{@link PhiEntryKind#scoreBased}）。学習の JVM の局末の値と、記録した局末の評価の両方をここで作る。点数の順位は局末の
 * 精算後の点数をそのまま最終順位とみなし（{@link HandValue#scoreRank}）、なめらかな関数は {@link ScoreSmoothPhi}。
 */
public final class ScorePhi {
  private ScorePhi() {}

  /**
   * 学習の JVM の局末の値。
   *
   * @throws IllegalArgumentException kind が点数だけの Φ でないとき
   */
  public static HandValue handValue(PhiEntryKind kind, EpsilonUtilityProfile profile) {
    return switch (kind) {
      case SCORE_RANK -> HandValue.scoreRank(profile);
      case SCORE_SMOOTH -> ScoreSmoothPhi.handValue(profile);
      case FIXED_CHECKPOINT, SELF_CHECKPOINT ->
          throw new IllegalArgumentException("Not a score-only Φ: " + kind);
    };
  }

  /**
   * 記録した局末の評価（点数の順位の順位分布は 0/1）。
   *
   * @throws IllegalArgumentException kind が点数だけの Φ でないとき
   */
  public static HandEndPhi offline(PhiEntryKind kind, EpsilonUtilityProfile profile) {
    return switch (kind) {
      case SCORE_RANK -> inputs -> scoreRank(inputs, profile);
      case SCORE_SMOOTH -> inputs -> ScoreSmoothPhi.values(inputs, profile);
      case FIXED_CHECKPOINT, SELF_CHECKPOINT ->
          throw new IllegalArgumentException("Not a score-only Φ: " + kind);
    };
  }

  private static PhiValues scoreRank(List<HandEndInput> inputs, EpsilonUtilityProfile profile) {
    double[][] utilities = new double[inputs.size()][];
    double[][] ranks = new double[inputs.size()][];
    for (int i = 0; i < inputs.size(); i++) {
      int[] scores = inputs.get(i).next().scores();
      utilities[i] = HandValue.terminal(scores, profile);
      int[] order = ScoreRanking.byScoreThenSeat(scores);
      ranks[i] = new double[16];
      for (int seat = 0; seat < 4; seat++) ranks[i][4 * seat + order[seat]] = 1;
    }
    return new PhiValues(utilities, ranks);
  }
}
