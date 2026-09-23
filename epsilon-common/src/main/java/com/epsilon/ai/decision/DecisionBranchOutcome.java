package com.epsilon.ai.decision;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpRankPredictor;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.core.ScoreRanking;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.RoundTransition;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;

/** 終局時は実順位、続行時は直後の局境界の固定GRPで二枝を同じ尺度へ変換する。 */
public final class DecisionBranchOutcome {
  private DecisionBranchOutcome() {}

  /** 実順位または固定GRPの予測から、指定席の局境界効用を求める。 */
  public static CompletableFuture<Float> utility(
      RoundSettlement settlement,
      float[] prefix,
      int seat,
      EpsilonUtilityProfile profile,
      EpsilonGrpRankPredictor grp) {
    int[] scores = settlement.snapshotFinalScores();
    if (settlement.transition() instanceof RoundTransition.HanchanFinished) {
      return CompletableFuture.completedFuture(
          profile.utilityForRank(ScoreRanking.byScoreThenSeat(scores)[seat]));
    }
    RoundTransition.NextRound next = (RoundTransition.NextRound) settlement.transition();
    float[] feature =
        EpsilonGrpFeature.fromProgress(
            next.kyokuIndex(), next.honba(), next.kyotakuCount(), scores);
    float[] sequence = Arrays.copyOf(prefix, prefix.length + feature.length);
    System.arraycopy(feature, 0, sequence, prefix.length, feature.length);
    return grp.predictMarginalProbabilitiesAsync(sequence)
        .thenApply(marginals -> expectedUtility(marginals, seat, profile));
  }

  /** 席別順位確率を指定の順位効用へ変換する。 */
  public static float expectedUtility(float[] marginals, int seat, EpsilonUtilityProfile profile) {
    float utility = 0;
    for (int rank = 0; rank < 4; rank++) {
      utility += marginals[EpsilonGrpRanks.index(seat, rank)] * profile.utilityForRank(rank);
    }
    return utility;
  }
}
