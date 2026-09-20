package com.epsilon.ai.decision;

import java.util.concurrent.CompletableFuture;

/** 二枝の結果だけを保持する。未完成の比較を待つことも、局途中の値で補うこともしない。 */
public final class DecisionBranchComparison {
  private final DecisionBranchGate gate;
  private final float oldAcceptanceProbability;
  private final boolean mainAccepted;
  private final DecisionBranchBudget budget;
  private final CompletableFuture<Float> mainUtility = new CompletableFuture<>();
  private final CompletableFuture<Float> extraUtility = new CompletableFuture<>();
  private volatile boolean cancelled;

  public DecisionBranchComparison(
      DecisionBranchGate gate,
      float oldAcceptanceProbability,
      boolean mainAccepted,
      DecisionBranchBudget budget) {
    this.gate = gate;
    this.oldAcceptanceProbability = oldAcceptanceProbability;
    this.mainAccepted = mainAccepted;
    this.budget = budget;
  }

  /** 通常対局の効用が確定したときに結果を受け取る。 */
  public void completeMain(CompletableFuture<Float> utility) {
    completeFrom(mainUtility, utility);
  }

  /** 追加枝の効用が確定したときに結果を受け取る。 */
  public void completeExtra(CompletableFuture<Float> utility) {
    completeFrom(extraUtility, utility);
  }

  /** この比較を教師値の確定対象から外す。 */
  public void cancel() {
    cancelled = true;
  }

  /** この比較が打ち切られたか、教師値として確定済みかを返す。 */
  public boolean cancelled() {
    return cancelled;
  }

  /** 完成済みの比較だけを教師値に確定する。未完成の結果は待たない。 */
  public DecisionBranchTarget freeze() {
    if (cancelled || !mainUtility.isDone() || !extraUtility.isDone()) {
      cancelled = true;
      return gate == DecisionBranchGate.KYUSHU
          ? DecisionBranchTarget.KYUSHU_ONLY
          : DecisionBranchTarget.NONE;
    }
    cancelled = true;
    float mainValue = mainUtility.join();
    float extraValue = extraUtility.join();
    budget.recordCompletedComparison();
    return DecisionBranchTarget.completed(
        gate,
        oldAcceptanceProbability,
        mainAccepted ? mainValue : extraValue,
        mainAccepted ? extraValue : mainValue);
  }

  private static void completeFrom(
      CompletableFuture<Float> target, CompletableFuture<Float> source) {
    source.whenComplete(
        (value, failure) -> {
          if (failure == null) {
            target.complete(value);
          } else {
            target.completeExceptionally(failure);
          }
        });
  }
}
