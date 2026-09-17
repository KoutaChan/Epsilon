package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.training.DecisionTrainingMetrics.AdvantageSigns;
import java.util.Objects;

/**
 * 入力の復号と同時に集計する、小バッチの行数と学習重み。
 *
 * <p>学習処理は完成したバッファを CPU から再走査せず、この集計値を方策・価値損失の重み付けと診断に使う。
 *
 * @param rows バッチ行数
 * @param bucket 行動形状を固定する容量区分
 * @param actorMass {@code actorWeight * sampleWeight}の総和
 * @param valueMass {@code sampleWeight}の総和
 * @param advantageSigns 有効方策行のスカラーアドバンテージ符号数
 */
record DecisionTrainingBatchSummary(
    int rows,
    DecisionBucket bucket,
    double actorMass,
    double valueMass,
    AdvantageSigns advantageSigns) {

  DecisionTrainingBatchSummary {
    if (rows <= 0) {
      throw new IllegalArgumentException("rows must be positive: " + rows);
    }
    Objects.requireNonNull(bucket, "bucket");
    Objects.requireNonNull(advantageSigns, "advantageSigns");
    if (!Double.isFinite(actorMass) || actorMass < 0.0) {
      throw new IllegalArgumentException("actorMass must be finite and non-negative: " + actorMass);
    }
    if (!Double.isFinite(valueMass) || valueMass < 0.0) {
      throw new IllegalArgumentException("valueMass must be finite and non-negative: " + valueMass);
    }
  }

  /** 方策損失へ寄与する行が一つ以上あるかを返す。 */
  boolean hasActorSignal() {
    return actorMass > 0.0;
  }
}
