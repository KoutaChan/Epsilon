package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import java.util.ArrayList;
import java.util.List;

/**
 * 同じ容量区分のサンプル参照と、学習ワーカーへの配分に必要な情報を保持する。
 *
 * <p>サンプル一覧はキューから引き継ぎ、入力パイプラインが読み終えるまで変更しない。追加の配列や変更禁止のラッパーは作らない。
 */
final class DecisionTrainingMicroBatch {

  private final int sourceOrdinal;
  private final ArrayList<EpsilonDecisionTrainingSampleDescriptor> samples;
  private final long estimatedCost;

  private DecisionTrainingMicroBatch(
      int sourceOrdinal,
      ArrayList<EpsilonDecisionTrainingSampleDescriptor> samples,
      long estimatedCost) {
    this.sourceOrdinal = sourceOrdinal;
    this.samples = samples;
    this.estimatedCost = estimatedCost;
  }

  /** キューが所有していたサンプルの参照情報リストをコピーせず引き取る。 */
  static DecisionTrainingMicroBatch take(
      int sourceOrdinal,
      ArrayList<EpsilonDecisionTrainingSampleDescriptor> samples,
      DecisionPolicySignalMultipliers multipliers) {
    if (samples.isEmpty()) {
      throw new IllegalArgumentException("Decision training microbatch must not be empty");
    }
    DecisionBucket bucket = samples.getFirst().bucket();
    boolean actorSignal = hasActorSignal(samples, multipliers);
    return new DecisionTrainingMicroBatch(
        sourceOrdinal, samples, estimateCost(samples.size(), bucket, actorSignal));
  }

  /**
   * 復号前に得られる形状だけから順伝播/逆伝播量を見積もる。
   *
   * <p>状態行、行動格納枠、遷移格納枠を順伝播量とし、価値逆伝播の行量と、有効方策があるバッチだけ方策
   * 逆伝播量を加える。絶対時間ではなく、同じ更新回数内で学習ワーカー負荷を比較するための単位である。
   */
  static long estimateCost(int rows, DecisionBucket bucket, boolean actorSignal) {
    long actionCells = (long) rows * bucket.legalActionCapacity();
    long transitionCells = actionCells * bucket.actionTransitionCapacity();
    long forward = rows + actionCells + transitionCells;
    return forward + rows + (actorSignal ? forward : 0L);
  }

  private static boolean hasActorSignal(
      List<EpsilonDecisionTrainingSampleDescriptor> samples,
      DecisionPolicySignalMultipliers multipliers) {
    for (EpsilonDecisionTrainingSampleDescriptor sample : samples) {
      if (sample.actorSnapshotId() > 0L
          && sample.legalActionCount() > 1
          && sample.learningRole().advancesActorClock()
          && multipliers.forKind(
                  EpsilonDecisionPointKind.ofLegalActions(
                      sample.legalActionIdsView(), sample.legalActionCount()))
              > 0.0f) {
        return true;
      }
    }
    return false;
  }

  int sourceOrdinal() {
    return sourceOrdinal;
  }

  List<EpsilonDecisionTrainingSampleDescriptor> samples() {
    return samples;
  }

  long estimatedCost() {
    return estimatedCost;
  }
}
