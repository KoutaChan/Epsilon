package com.epsilon.nano.ai.decision.training;

import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.training.DecisionTrainingMetrics.AdvantageSigns;
import java.util.Objects;

/**
 * ホスト側の連続バッファの構築中に確定する小バッチ集計値。
 *
 * <p>Trainerは確定済み連続バッファをCPUから再走査せず、この値を方策モデル/価値損失の重み付けと診断に使う。集計はデータ本体の
 * 復号と同じループで行うため、追加の配列や行オブジェクトを作らない。
 *
 * @param rows バッチ行数
 * @param bucket 行動形状を固定する容量区分
 * @param actorMass {@code actorWeight * sampleWeight}の総和
 * @param valueMass {@code sampleWeight}の総和
 * @param advantageSigns 方策学習の対象行のスカラーアドバンテージ符号別の件数
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
