package com.epsilon.nano.ai.decision.data;

import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import java.io.IOException;
import java.nio.file.Path;

/**
 * 入力本体を読み込む前に、学習サンプルの選択とバッチ分割に使う情報。
 *
 * <p>入力と探索前後の方策分布は外部ファイルに置き、シャッフル、容量別の待ち行列、更新単位への分割に必要な情報とファイル参照だけを保持する。バッチの構成が確定した後、入力を CPU
 * 側の最終バッファへ直接読み込む。
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

  /** 検証済み局開始時点までの GRP 入力の共有ビュー。呼出し側は配列を変更しない。 */
  public float[] grpFeatureSequenceView() {
    return sample.grpFeatureSequenceView();
  }

  public int grpFinalRanksCode() {
    return sample.grpFinalRanksCode();
  }

  public DecisionBranchTarget branchTarget() {
    return sample.branchTarget();
  }

  public DecisionLearningRole learningRole() {
    return sample.learningRole();
  }

  /** 移行中のメモリ上での処理と一致性テスト向けに、参照先のサンプル本体を復元する。 */
  public EpsilonDecisionSample materializeChecked() throws IOException {
    return sample.materializeChecked();
  }
}
