package com.epsilon.major.ai.decision.training;

/**
 * 転送準備が終わった小バッチを保持し、演算側が一度だけ取得できるようにする。
 *
 * <p>{@code ON_DEMAND}では確定済みホスト側バッチ、{@code OVERLAPPED}では進行中または完了済みのH2D 受付記録を内部に持つ。
 * 呼び出し側は両者を区別せず、所有パイプラインの{@code acquire}へ渡す。取得せず閉じたバッチもパイプラインが安全に回収する。
 */
final class DecisionTrainingReadyBatch implements AutoCloseable {

  private final DecisionTrainingInputPipeline pipeline;
  private final DecisionTrainingBatchSummary summary;
  private DecisionTrainingSealedHostBatch hostBatch;
  private DecisionTrainingInputPipeline.PendingTransfer pendingTransfer;
  private boolean consumed;

  DecisionTrainingReadyBatch(
      DecisionTrainingInputPipeline pipeline,
      DecisionTrainingBatchSummary summary,
      DecisionTrainingSealedHostBatch hostBatch,
      DecisionTrainingInputPipeline.PendingTransfer pendingTransfer) {
    this.pipeline = pipeline;
    this.summary = summary;
    this.hostBatch = hostBatch;
    this.pendingTransfer = pendingTransfer;
  }

  /** バッチ行数と損失集計値を返す。 */
  DecisionTrainingBatchSummary summary() {
    return summary;
  }

  /** パイプラインだけが呼ぶ一回限りのデータ本体取得。 */
  synchronized Payload claim(DecisionTrainingInputPipeline expectedPipeline) {
    if (pipeline != expectedPipeline) {
      throw new IllegalArgumentException("ready batch belongs to another training lane");
    }
    if (consumed) {
      throw new IllegalStateException("ready batch is already consumed");
    }
    consumed = true;
    Payload payload = new Payload(hostBatch, pendingTransfer);
    hostBatch = null;
    pendingTransfer = null;
    return payload;
  }

  /** 未取得バッチをパイプラインへ返し、進行中のH2Dがあれば完了後に回収させる。 */
  @Override
  public void close() {
    pipeline.discard(this);
  }

  /** ON_DEMAND ホストまたはOVERLAPPED 転送のどちらか一方を持つ内部データ本体。 */
  record Payload(
      DecisionTrainingSealedHostBatch hostBatch,
      DecisionTrainingInputPipeline.PendingTransfer pendingTransfer) {}
}
