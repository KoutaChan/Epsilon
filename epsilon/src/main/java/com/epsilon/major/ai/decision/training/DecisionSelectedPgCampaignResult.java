package com.epsilon.major.ai.decision.training;

import java.util.List;

/** 選択行動の方策勾配学習の実行全体の変更不可の結果。 */
public record DecisionSelectedPgCampaignResult(
    int maximumMacros,
    int completedMacros,
    int games,
    long samples,
    List<DecisionSelectedPgDuelResult> duels,
    DecisionSelectedPgDuelResult.Status status) {

  /** 件数と対戦評価列を検証し、変更不可の一覧へ固定する。 */
  public DecisionSelectedPgCampaignResult {
    if (maximumMacros <= 0
        || completedMacros < 0
        || completedMacros > maximumMacros
        || games < 0
        || samples < 0) {
      throw new IllegalArgumentException("selected PG campaign counters are invalid");
    }
    duels = List.copyOf(duels);
    java.util.Objects.requireNonNull(status, "status");
  }

  /** 検証条件違反ではなく、事前定義した対局実行処理状態へ終端したか。 */
  public boolean completed() {
    return status() != DecisionSelectedPgDuelResult.Status.GUARD_FAILED;
  }

  /** 一連の学習実行中に対局収集用の採用モデルを更新した回数。 */
  public long promotions() {
    return duels.stream().filter(DecisionSelectedPgDuelResult::promoted).count();
  }
}
