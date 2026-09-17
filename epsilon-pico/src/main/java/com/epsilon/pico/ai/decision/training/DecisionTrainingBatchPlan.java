package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import java.util.List;
import java.util.Objects;

/**
 * 学習ワーカーへ渡す小バッチの入力準備を指示する。
 *
 * <p>サンプルの参照情報は複製しない。入力の準備が完了するまで、呼び出し側はサンプル一覧とその要素を変更してはならない。ワーカーは参照先の外部ファイルから最終的なホスト側バッファへ直接復号し、準備完了後はサンプルの参照情報を保持しない。
 *
 * @param samples 同一容量区分に属するサンプルの参照情報
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
