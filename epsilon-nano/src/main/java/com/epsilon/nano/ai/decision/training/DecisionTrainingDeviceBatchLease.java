package com.epsilon.nano.ai.decision.training;

import ai.djl.ndarray.NDManager;
import com.epsilon.nano.ai.decision.input.DecisionDeviceBatch;

/**
 * 順伝播から逆伝播まで必要なデバイス側バッチと一時管理元を保持し、使用後に解放する。
 *
 * <p>ホスト枠とH2D 完了通知はこの利用権へ含めない。それらは複製完了時にパイプラインが先に回収できるため、順伝播/逆伝播の寿命でページ固定の メモリを不必要に占有しない。
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

  /** デバイステンソルを所有する子管理元を返す。 */
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
