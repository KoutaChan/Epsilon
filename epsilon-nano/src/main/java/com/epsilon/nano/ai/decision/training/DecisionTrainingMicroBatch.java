package com.epsilon.nano.ai.decision.training;

import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import java.util.ArrayList;
import java.util.List;

/**
 * 同じ形状のサンプル参照情報と、学習ワーカーへの配分に使うメタデータを保持する。
 *
 * <p>{@code samples}は容量区分キューからこの型へ所有権を移す。以後は入力パイプラインが読み終えるまで変更せず、変更不可 包むオブジェクトや記述情報配列を追加生成しない。
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

  /** キューが所有していた記述情報リストをコピーせず引き取る。 */
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
   * <p>状態行、行動格納枠、遷移格納枠を順伝播量とし、価値逆伝播の行量と、有効方策モデルがあるバッチだけ方策
   * 逆伝播量を加える。絶対時間ではなく、同じ更新段階内で実行単位負荷を比較するための単位である。
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
