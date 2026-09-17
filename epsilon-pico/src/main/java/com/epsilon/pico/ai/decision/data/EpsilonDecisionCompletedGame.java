package com.epsilon.pico.ai.decision.data;

import com.epsilon.ai.grp.EpsilonGrpRanks;
import java.util.List;
import java.util.Objects;

/** 完了した一対局の最終順位、局ごとの補助データ、各判断の学習データをまとめて受け渡す。 */
public record EpsilonDecisionCompletedGame(
    long gameId,
    int finalRanksCode,
    List<EpsilonDecisionGameBoundary> boundaries,
    List<EpsilonDecisionSampleRecord> samples) {

  /** 一局内識別情報、境界参照、終局順位を相互検証する。 */
  public EpsilonDecisionCompletedGame {
    if (!EpsilonGrpRanks.isValidCode(finalRanksCode)) {
      throw new IllegalArgumentException("completed game requires a valid final-ranks code");
    }
    boundaries = List.copyOf(Objects.requireNonNull(boundaries, "boundaries"));
    samples = List.copyOf(Objects.requireNonNull(samples, "samples"));
    for (int index = 0; index < boundaries.size(); index++) {
      if (boundaries.get(index).index() != index) {
        throw new IllegalArgumentException("game boundary indexes must be contiguous");
      }
    }
    int[] finalRanks = EpsilonGrpRanks.decode(finalRanksCode);
    for (EpsilonDecisionSampleRecord sample : samples) {
      if (sample.gameId() != gameId) {
        throw new IllegalArgumentException("Decision sample gameId does not match completed game");
      }
      if (sample.boundaryIndex() < 0 || sample.boundaryIndex() >= boundaries.size()) {
        throw new IllegalArgumentException("Decision sample references an unknown boundary");
      }
      if (sample.finalRank() != finalRanks[sample.playerSeat()]) {
        throw new IllegalArgumentException("Decision sample final rank does not match game result");
      }
      if (!boundaries
          .get(sample.boundaryIndex())
          .hasSameFeatureSequence(grpFeatureSequenceView(sample))) {
        throw new IllegalArgumentException(
            "Decision sample GRP prefix differs from boundary sidecar");
      }
    }
  }

  private static float[] grpFeatureSequenceView(EpsilonDecisionSampleRecord sample) {
    if (sample instanceof EpsilonDecisionDeferredSample deferred) {
      return deferred.grpFeatureSequenceView();
    }
    if (sample instanceof EpsilonDecisionSample materialized) {
      return materialized.grpFeatureSequenceView();
    }
    return sample.grpFeatureSequence();
  }
}
