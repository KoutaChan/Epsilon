package com.epsilon.ai.decision;

import com.epsilon.config.settings.DecisionBranchComparisonSettings;

/** スケジューラー所有の追加推論予算。通常推論で獲得した行数だけを使う。 */
public final class DecisionBranchBudget {
  private final DecisionBranchComparisonSettings settings;
  private long mainRows;
  private long extraRows;
  private long admitted;
  private long completed;
  private int inFlight;

  public DecisionBranchBudget(DecisionBranchComparisonSettings settings) {
    this.settings = settings;
  }

  /** 通常対局の推論行数を追加予算の基準へ加える。 */
  public synchronized void recordMainRows(int rows) {
    mainRows += rows;
  }

  /** 対局ごとの上限と残り予算を満たす場合に、追加枝の実行枠を確保する。 */
  public synchronized boolean admit(int gameBranches) {
    if (gameBranches >= settings.maxBranchesPerGame()
        || inFlight >= settings.maxInFlightBranches()
        || extraRows >= Math.floor(mainRows * settings.maxExtraInferenceRatio())) {
      return false;
    }
    admitted++;
    inFlight++;
    return true;
  }

  /** 追加推論の行数を予算から消費する。予算不足なら消費しない。 */
  public synchronized boolean takeRows(int rows) {
    if (extraRows + rows > Math.floor(mainRows * settings.maxExtraInferenceRatio())) {
      return false;
    }
    extraRows += rows;
    return true;
  }

  /** 追加枝の実行枠を解放する。 */
  public synchronized void release() {
    inFlight--;
  }

  /** 教師値として確定した比較を数える。 */
  public synchronized void recordCompletedComparison() {
    completed++;
  }

  /** 推論行数と比較の完了数をログ用の文字列で返す。 */
  public synchronized String summary() {
    return "mainRows="
        + mainRows
        + " extraRows="
        + extraRows
        + " admitted="
        + admitted
        + " completed="
        + completed
        + " incomplete="
        + (admitted - completed);
  }
}
