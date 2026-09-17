package com.epsilon.major.ai.decision.training;

import com.epsilon.major.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import java.util.List;
import java.util.Objects;

/**
 * 一つの学習ワーカーへ渡す小バッチと、その集計値をまとめる。
 *
 * <p>記述情報本体はコピーしない。呼び出し側は、この計画をパイプラインへ渡してから準備済みバッチが完成するまで、 {@code
 * samples}とその要素を変更してはならない。完成後は準備済みバッチが記述情報を保持しないため、収集済み学習データ片を解放できる。密な入力と方策
 * データ本体は記述情報が参照する外部ファイルに残り、ワーカーが最終ホスト側の連続バッファへ直接復号する。
 *
 * @param samples 同一容量区分に属するサンプル記述情報
 * @param policySignalMultipliers 判断種別ごとの方策係数
 */
record DecisionTrainingBatchPlan(
    List<EpsilonDecisionTrainingSampleDescriptor> samples,
    DecisionPolicySignalMultipliers policySignalMultipliers) {

  DecisionTrainingBatchPlan {
    Objects.requireNonNull(samples, "samples");
    Objects.requireNonNull(policySignalMultipliers, "policySignalMultipliers");
    if (samples.isEmpty()) {
      throw new IllegalArgumentException("training microbatch must not be empty");
    }
  }

  /** 小バッチの行数を返す。 */
  int rows() {
    return samples.size();
  }
}
