package com.epsilon.pico.ai.decision.training;

/**
 * 学習ワーカーが一度だけ取得できる、準備済みの小バッチ。
 *
 * <p>ON_DEMAND では復号済みのホスト側バッチ、OVERLAPPED では開始済みの転送を内部に保持する。呼び出し側はいずれもパイプラインの acquire
 * に渡す。取得前に閉じた場合も、パイプラインが資源を回収する。
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

  /** パイプラインだけが呼ぶ一回限りの入力と確率分布の本体取得。 */
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

  /** ON_DEMAND ホストまたはOVERLAPPED transferのどちらか一方を持つ内部入力と確率分布の本体。 */
  record Payload(
      DecisionTrainingSealedHostBatch hostBatch,
      DecisionTrainingInputPipeline.PendingTransfer pendingTransfer) {}
}
