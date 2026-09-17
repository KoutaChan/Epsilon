package com.epsilon.pico.ai.decision.training;

import ai.djl.ndarray.NDManager;
import com.epsilon.pico.ai.decision.input.DecisionDeviceBatch;

/**
 * 順伝播から逆伝播まで使うデバイス側バッチと、その専用リソース管理オブジェクトを保持する。
 *
 * <p>ホスト側の実行枠と転送完了の通知は含めない。これらはホストからデバイスへの転送が終わり次第回収し、順伝播や逆伝播の終了を待たずに再利用できる。
 */
final class DecisionTrainingDeviceBatchLease implements AutoCloseable {

  private final DecisionTrainingInputPipeline pipeline;
  private final DecisionTrainingBatchSummary summary;
  private NDManager manager;
  private DecisionDeviceBatch batch;

  DecisionTrainingDeviceBatchLease(
      DecisionTrainingInputPipeline pipeline,
      NDManager manager,
      DecisionDeviceBatch batch,
      DecisionTrainingBatchSummary summary) {
    this.pipeline = pipeline;
    this.manager = manager;
    this.batch = batch;
    this.summary = summary;
  }

  /** デバイステンソルを所有する専用のリソース管理オブジェクトを返す。 */
  NDManager manager() {
    requireOpen();
    return manager;
  }

  /** モデルへ渡す論理的な学習バッチを返す。 */
  DecisionDeviceBatch batch() {
    requireOpen();
    return batch;
  }

  /** ホスト復号時に確定した集計値を返す。 */
  DecisionTrainingBatchSummary summary() {
    return summary;
  }

  /** デバイステンソルを一括解放する。 */
  @Override
  public void close() {
    NDManager activeManager;
    synchronized (this) {
      activeManager = manager;
      if (activeManager == null) {
        return;
      }
      manager = null;
      batch = null;
    }
    try {
      activeManager.close();
    } finally {
      pipeline.releaseDeviceLease();
    }
  }

  private void requireOpen() {
    if (manager == null) {
      throw new IllegalStateException("device batch lease is closed");
    }
  }
}
