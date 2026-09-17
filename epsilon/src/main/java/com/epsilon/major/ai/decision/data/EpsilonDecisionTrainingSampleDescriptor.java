package com.epsilon.major.ai.decision.data;

import com.epsilon.core.DecisionLearningRole;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import java.io.IOException;
import java.nio.file.Path;

/**
 * 学習データの並べ替えとバッチ分割に必要なメタデータ、および外部ファイルへの参照を保持する。
 *
 * <p>密な入力と探索後と探索前の方策は外部データ本体に置いたままにし、並べ替え、容量区分キュー、パラメーター更新の
 * 分割に必要なメタデータとデータ本体参照だけを保持する。データ本体を読むのは、確定した小バッチの最終ホスト転送先へ 復号するときだけである。
 */
public final class EpsilonDecisionTrainingSampleDescriptor {

  private final EpsilonDecisionDeferredSample sample;
  private final FilePayloadRef payloadRef;

  private EpsilonDecisionTrainingSampleDescriptor(
      EpsilonDecisionDeferredSample sample, FilePayloadRef payloadRef) {
    this.sample = sample;
    this.payloadRef = payloadRef;
  }

  static EpsilonDecisionTrainingSampleDescriptor from(EpsilonDecisionDeferredSample sample)
      throws IOException {
    if (!(sample.payloadRef() instanceof FilePayloadRef fileRef)) {
      throw new IOException(
          "Decision training descriptor requires a resolved external payload reference");
    }
    return new EpsilonDecisionTrainingSampleDescriptor(sample, fileRef);
  }

  public DecisionBucket bucket() {
    return payloadRef.bucket();
  }

  FilePayloadRef payloadRef() {
    return payloadRef;
  }

  /** 外部データ本体の所在。診断と入力元破損の報告に使う。 */
  public Path payloadPath() {
    return Path.of(payloadRef.file());
  }

  public int legalActionCount() {
    return sample.legalActionCount();
  }

  public int legalActionIdBySlot(int slot) {
    return sample.legalActionIdBySlot(slot);
  }

  /** 検証済みメタデータの共有ビュー。呼出し側は配列を変更しない。 */
  public int[] legalActionIdsView() {
    return sample.legalActionIdsView();
  }

  public int playerSeat() {
    return sample.playerSeat();
  }

  public int sourcePlayerRelativeSeat() {
    return sample.sourcePlayerRelativeSeat();
  }

  public int currentPlayerRelativeSeat() {
    return sample.currentPlayerRelativeSeat();
  }

  public int chosenLegalSlot() {
    return sample.chosenLegalSlot();
  }

  public int chosenActionId() {
    return sample.chosenActionId();
  }

  public float behaviorLogProb() {
    return sample.behaviorLogProb();
  }

  public float behaviorProb() {
    return sample.behaviorProb();
  }

  public float valueTarget() {
    return sample.valueTarget();
  }

  public float advantage() {
    return sample.advantage();
  }

  public int finalRank() {
    return sample.finalRank();
  }

  public long actorSnapshotId() {
    return sample.actorSnapshotId();
  }

  public int ruleProfile() {
    return sample.ruleProfile();
  }

  public long gameId() {
    return sample.gameId();
  }

  public int boundaryIndex() {
    return sample.boundaryIndex();
  }

  public int seatDecisionOrdinal() {
    return sample.seatDecisionOrdinal();
  }

  /** 検証済みGRPの局履歴の共有ビュー。呼出し側は配列を変更しない。 */
  public float[] grpFeatureSequenceView() {
    return sample.grpFeatureSequenceView();
  }

  public int grpFinalRanksCode() {
    return sample.grpFinalRanksCode();
  }

  public DecisionLearningRole learningRole() {
    return sample.learningRole();
  }

  /** 移行中のヒープパスと一致性テストだけが使う明示的データ復元。 */
  public EpsilonDecisionSample materializeChecked() throws IOException {
    return sample.materializeChecked();
  }
}
