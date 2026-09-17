package com.epsilon.pico.ai.decision.data;

import java.util.Objects;

/** 完了した対局の学習データと、収集条件・学習条件の識別情報をまとめる。 */
public record EpsilonDecisionFragment(
    long actorSnapshotId,
    int grpTeacherIteration,
    EpsilonDecisionCompletedGame game,
    EpsilonDecisionTrainingTargetIdentity trainingTargetIdentity) {

  /** 対局を再構築せずに保持し、収集・学習識別情報との整合性だけを検証する。 */
  public EpsilonDecisionFragment {
    if (actorSnapshotId <= 0L) {
      throw new IllegalArgumentException(
          "actorSnapshotId must identify the macro-start candidate: " + actorSnapshotId);
    }
    if (grpTeacherIteration < -1) {
      throw new IllegalArgumentException(
          "grpTeacherIteration must be -1 (disabled) or non-negative: " + grpTeacherIteration);
    }
    Objects.requireNonNull(game, "game");
    trainingTargetIdentity =
        Objects.requireNonNull(trainingTargetIdentity, "trainingTargetIdentity");
    for (EpsilonDecisionSampleRecord sample : game.samples()) {
      if (sample.actorSnapshotId() != actorSnapshotId) {
        throw new IllegalArgumentException(
            "Decision sample actor snapshot differs from fragment identity");
      }
    }
  }
}
