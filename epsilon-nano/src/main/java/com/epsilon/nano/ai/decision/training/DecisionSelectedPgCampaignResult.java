package com.epsilon.nano.ai.decision.training;

import java.util.List;

/** 選択行動の方策勾配学習と対戦評価を繰り返した全体の実行結果。 */
public record DecisionSelectedPgCampaignResult(
    int maximumMacros,
    int completedMacros,
    int games,
    long samples,
    List<DecisionSelectedPgDuelResult> duels,
    DecisionSelectedPgDuelResult.Status status) {

  /** 件数と対戦比較列を検証し、不変リストへ固定する。 */
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

  /** 検証条件違反ではなく、事前定義した対局状態へ終端したか。 */
  public boolean completed() {
    return status() != DecisionSelectedPgDuelResult.Status.GUARD_FAILED;
  }

  /** 学習と対戦評価の一連の実行中に対局収集に使う採用モデルを更新した回数。 */
  public long promotions() {
    return duels.stream().filter(DecisionSelectedPgDuelResult::promoted).count();
  }
}
