package com.epsilon.nano.ai.decision.training;

import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator;
import java.nio.file.Path;
import java.util.Objects;

/** 選択行動の方策勾配学習における一区間の学習結果と、その直後の対戦評価結果。 */
public record DecisionSelectedPgDuelResult(
    int lineage,
    int duelRound,
    int checkpointIteration,
    int campaignMacros,
    int lineageMacros,
    int intervalMacros,
    int intervalGames,
    long intervalSamples,
    Status status,
    Path champion,
    Path candidate,
    String reason) {

  /** 件数、状態、チェックポイントパスを検証する。 */
  public DecisionSelectedPgDuelResult {
    if (lineage <= 0
        || duelRound <= 0
        || checkpointIteration < 0
        || campaignMacros < 0
        || lineageMacros < 0
        || intervalMacros < 0
        || intervalGames < 0
        || intervalSamples < 0) {
      throw new IllegalArgumentException("selected PG duel counters are invalid");
    }
    Objects.requireNonNull(status, "status");
    Objects.requireNonNull(champion, "champion");
    Objects.requireNonNull(candidate, "candidate");
    Objects.requireNonNull(reason, "reason");
  }

  /** 候補が対局収集に使う採用モデルへ昇格したか。 */
  public boolean promoted() {
    return status == Status.PROMOTED;
  }

  /** 固定牌山対戦比較の判定と学習と対戦評価の一連の実行位置から、対戦評価の間の学習区間の終端状態を決める。 */
  static Status resolveStatus(
      EpsilonDecisionWallDuelEvaluator.Decision decision, int completedMacros, int maximumMacros) {
    Objects.requireNonNull(decision, "decision");
    if (completedMacros <= 0 || maximumMacros <= 0 || completedMacros > maximumMacros) {
      throw new IllegalArgumentException(
          "invalid selected PG campaign position: completedMacros="
              + completedMacros
              + " maximumMacros="
              + maximumMacros);
    }
    return switch (decision) {
      case PROMOTED -> Status.PROMOTED;
      case HARMFUL -> Status.HARMFUL;
      case UNRESOLVED -> completedMacros < maximumMacros ? Status.UNRESOLVED : Status.PAUSED;
    };
  }

  /** 一つの学習対戦評価の間の学習区間後の対局判定。 */
  public enum Status {
    /** 固定した CI下限が0を超え、対局収集に使う採用モデルへ昇格した。 */
    PROMOTED(true),

    /** 固定した CI上限が0未満となり、学習と対戦評価の一連の実行を停止した。 */
    HARMFUL(false),

    /** CIが0をまたぎ、同じ継続する学習状態とオプティマイザーで次対戦評価の間の学習区間へ進む。 */
    UNRESOLVED(true),

    /** 一実行の予算を使い切り、学習器と未完継続する学習状態を保持して再開を待つ。 */
    PAUSED(true),

    /** 学習指標または最終方策監査の検証条件に失敗した。 */
    GUARD_FAILED(false);

    private final boolean preservesLearnerState;

    Status(boolean preservesLearnerState) {
      this.preservesLearnerState = preservesLearnerState;
    }

    /** 現在のモデル、AdamW、KL 制御器を次対戦評価の間の学習区間へ保持するか。 */
    public boolean preservesLearnerState() {
      return preservesLearnerState;
    }
  }
}
