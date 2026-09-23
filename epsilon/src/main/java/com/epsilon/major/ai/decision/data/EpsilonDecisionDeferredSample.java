package com.epsilon.major.ai.decision.data;

import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.major.ai.decision.EpsilonDecisionReturns;
import com.epsilon.major.ai.decision.EpsilonUtilityTargets;
import com.epsilon.major.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore.PayloadRef;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import java.io.IOException;

/**
 * 収集中の入力と確率分布を外部ファイルに残し、学習時に必要になった段階で読み戻すサンプル。
 *
 * <p>半荘終了時には教師値とメタデータだけを確定する。学習データ片には入力や確率分布をコピーせず、ファイルのパス・オフセット・長さを保存する。
 *
 * <p>データ本体は圧縮しない。{@link EpsilonDecisionTrajectoryPayloadCodec}
 * の形式で保存した外部参照を引き継ぎ、学習器がサンプルを使うときだけ復元する。収集終了時の復号・再符号化・二重書き込みを省く。
 */
final class EpsilonDecisionDeferredSample implements EpsilonDecisionSampleRecord {

  private final EpsilonDecisionTrajectoryPayloadStore payloadStore;
  private final PayloadRef payloadRef;
  private final int[] legalActionIdBySlot;
  private final int legalActionCount;
  private final int playerSeat;
  private final int sourcePlayerRelativeSeat;
  private final int currentPlayerRelativeSeat;
  private final int chosenLegalSlot;
  private final int chosenActionId;
  private final float behaviorLogProb;
  private final float behaviorProb;
  private final float valueTarget;
  private final float advantage;
  private final int finalRank;
  private final long actorSnapshotId;
  private final int ruleProfile;
  private final long gameId;
  private final int boundaryIndex;
  private final int seatDecisionOrdinal;
  private final float[] grpFeatureSequence;
  private final int grpFinalRanksCode;
  private final DecisionLearningRole learningRole;
  private final DecisionBranchTarget branchTarget;

  EpsilonDecisionDeferredSample(
      EpsilonDecisionTrajectoryPayloadStore payloadStore,
      PayloadRef payloadRef,
      int[] legalActionIdBySlot,
      int legalActionCount,
      int playerSeat,
      int sourcePlayerRelativeSeat,
      int currentPlayerRelativeSeat,
      int chosenLegalSlot,
      int chosenActionId,
      float behaviorLogProb,
      float behaviorProb,
      float valueTarget,
      float advantage,
      int finalRank,
      long actorSnapshotId,
      int ruleProfile,
      long gameId,
      int boundaryIndex,
      int seatDecisionOrdinal,
      float[] grpFeatureSequence,
      int grpFinalRanksCode,
      DecisionLearningRole learningRole,
      DecisionBranchTarget branchTarget) {
    this.payloadStore = payloadStore;
    this.payloadRef = payloadRef;
    int compactCount = EpsilonDecisionLegalActions.compactCount(legalActionCount);
    if (compactCount == 0) {
      throw new EpsilonDecisionDataException("deferred sample requires at least one legal action");
    }
    this.legalActionIdBySlot =
        EpsilonDecisionLegalActions.compact(legalActionIdBySlot, compactCount);
    this.legalActionCount = compactCount;
    this.playerSeat = playerSeat;
    this.sourcePlayerRelativeSeat = sourcePlayerRelativeSeat;
    this.currentPlayerRelativeSeat = currentPlayerRelativeSeat;
    this.chosenLegalSlot = chosenLegalSlot;
    this.chosenActionId = chosenActionId;
    if (chosenLegalSlot < 0 || chosenLegalSlot >= compactCount) {
      throw new EpsilonDecisionDataException("chosenLegalSlot outside legal action range");
    }
    if (chosenActionId != this.legalActionIdBySlot[chosenLegalSlot]) {
      throw new EpsilonDecisionDataException("chosenActionId does not match chosenLegalSlot");
    }
    this.behaviorLogProb = behaviorLogProb;
    this.behaviorProb =
        EpsilonDecisionDataChecks.requireNonNegativeFinite(behaviorProb, "behaviorProb");
    if (!(this.behaviorProb > 0.0f)) {
      throw new EpsilonDecisionDataException("behaviorProb must be positive");
    }
    float expectedBehaviorLogProb = (float) Math.log(this.behaviorProb);
    if (Math.abs(behaviorLogProb - expectedBehaviorLogProb) > 1.0e-4f) {
      throw new EpsilonDecisionDataException(
          "behaviorLogProb does not match selected behavior probability");
    }
    this.valueTarget =
        EpsilonDecisionReturns.requireValueTarget(
            valueTarget, EpsilonUtilityTargets.profile(ruleProfile));
    this.advantage = EpsilonDecisionDataChecks.requireFinite(advantage, "advantage");
    this.finalRank = finalRank;
    this.actorSnapshotId = actorSnapshotId;
    this.ruleProfile = ruleProfile;
    this.gameId = gameId;
    if (boundaryIndex < 0 || seatDecisionOrdinal < -1) {
      throw new IllegalArgumentException("invalid boundary/seat decision identity");
    }
    this.boundaryIndex = boundaryIndex;
    this.seatDecisionOrdinal = seatDecisionOrdinal;
    this.grpFeatureSequence =
        EpsilonDecisionDataChecks.requireFinite(grpFeatureSequence, "grpFeatureSequence");
    this.grpFinalRanksCode = EpsilonDecisionSample.requireGrpFinalRanksCode(grpFinalRanksCode);
    if (learningRole == null) {
      throw new IllegalArgumentException("learningRole must not be null");
    }
    if (learningRole == DecisionLearningRole.FORCED && compactCount != 1) {
      throw new IllegalArgumentException("FORCED sample must have exactly one legal action");
    }
    if (learningRole == DecisionLearningRole.CAUSAL && compactCount == 1) {
      throw new IllegalArgumentException("single-action sample must be FORCED or PREEMPTED");
    }
    this.learningRole = learningRole;
    this.branchTarget = branchTarget;
  }

  EpsilonDecisionTrajectoryPayloadStore payloadStore() {
    return payloadStore;
  }

  @Override
  public DecisionBranchTarget branchTarget() {
    return branchTarget;
  }

  PayloadRef payloadRef() {
    return payloadRef;
  }

  /** 学習データ片符号化・復号処理向けの検証済みコンパクト合法手ビュー。 */
  int[] legalActionIdsView() {
    return legalActionIdBySlot;
  }

  @Override
  public DecisionHostBatch input() {
    return payload().input().materialize();
  }

  @Override
  public int legalActionCount() {
    return legalActionCount;
  }

  @Override
  public int legalActionIdBySlot(int slot) {
    return EpsilonDecisionLegalActions.actionIdAt(legalActionIdBySlot, slot);
  }

  @Override
  public int playerSeat() {
    return playerSeat;
  }

  @Override
  public int sourcePlayerRelativeSeat() {
    return sourcePlayerRelativeSeat;
  }

  @Override
  public int currentPlayerRelativeSeat() {
    return currentPlayerRelativeSeat;
  }

  @Override
  public int chosenLegalSlot() {
    return chosenLegalSlot;
  }

  @Override
  public int chosenActionId() {
    return chosenActionId;
  }

  @Override
  public float behaviorLogProb() {
    return behaviorLogProb;
  }

  @Override
  public float valueTarget() {
    return valueTarget;
  }

  @Override
  public float advantage() {
    return advantage;
  }

  @Override
  public int finalRank() {
    return finalRank;
  }

  @Override
  public float[] behaviorPolicy() {
    return payload().behaviorPolicy();
  }

  @Override
  public float[] rolloutPolicy() {
    return payload().rolloutPolicy();
  }

  @Override
  public long actorSnapshotId() {
    return actorSnapshotId;
  }

  @Override
  public int ruleProfile() {
    return ruleProfile;
  }

  @Override
  public long gameId() {
    return gameId;
  }

  @Override
  public int boundaryIndex() {
    return boundaryIndex;
  }

  @Override
  public int seatDecisionOrdinal() {
    return seatDecisionOrdinal;
  }

  @Override
  public float[] grpFeatureSequence() {
    return grpFeatureSequence.clone();
  }

  /** 学習データ片/境界検証向けの共有読み取り専用のビュー。パッケージ外へ公開しない。 */
  float[] grpFeatureSequenceView() {
    return grpFeatureSequence;
  }

  @Override
  public int grpFinalRanksCode() {
    return grpFinalRanksCode;
  }

  @Override
  public DecisionLearningRole learningRole() {
    return learningRole;
  }

  @Override
  public EpsilonDecisionSampleRecord withTrainingTargets(
      float nextValueTarget, float nextAdvantage) {
    return new EpsilonDecisionDeferredSample(
        payloadStore,
        payloadRef,
        legalActionIdBySlot,
        legalActionCount,
        playerSeat,
        sourcePlayerRelativeSeat,
        currentPlayerRelativeSeat,
        chosenLegalSlot,
        chosenActionId,
        behaviorLogProb,
        behaviorProb,
        nextValueTarget,
        nextAdvantage,
        finalRank,
        actorSnapshotId,
        ruleProfile,
        gameId,
        boundaryIndex,
        seatDecisionOrdinal,
        grpFeatureSequence,
        grpFinalRanksCode,
        learningRole,
        branchTarget);
  }

  @Override
  public float behaviorProb() {
    return behaviorProb;
  }

  @Override
  public EpsilonDecisionSample materialize() {
    try {
      return materializeChecked();
    } catch (IOException e) {
      throw new IllegalStateException("Failed to materialize deferred Decision sample", e);
    }
  }

  /** 学習データ片読み取り処理向け。外部データ本体のI/O失敗を隠さず伝播する。 */
  EpsilonDecisionSample materializeChecked() throws IOException {
    EpsilonDecisionTrajectoryPayload payload = payloadStore.read(payloadRef);
    return new EpsilonDecisionSample(
        payload.input().materialize(),
        chosenLegalSlot,
        chosenActionId,
        behaviorLogProb,
        valueTarget,
        advantage,
        finalRank,
        payload.behaviorPolicy(),
        payload.rolloutPolicy(),
        actorSnapshotId,
        ruleProfile,
        gameId,
        boundaryIndex,
        seatDecisionOrdinal,
        grpFeatureSequence,
        grpFinalRanksCode,
        learningRole,
        branchTarget);
  }

  private EpsilonDecisionTrajectoryPayload payload() {
    try {
      return payloadStore.read(payloadRef);
    } catch (IOException e) {
      throw new IllegalStateException("Failed to materialize deferred Decision sample", e);
    }
  }

  @Override
  public String toString() {
    return "EpsilonDecisionDeferredSample{"
        + "gameId="
        + gameId
        + ", legalActionCount="
        + legalActionCount
        + ", finalRank="
        + finalRank
        + ", payloadRef="
        + payloadRef
        + '}';
  }
}
