package com.epsilon.nano.ai.decision.training;

/**
 * 復号を終えたホストバッファを、転送処理へ一度だけ引き渡す。
 *
 * <p>この型は書き込み処理を公開しない。デバイス転送を開始すると所有権はパイプラインへ戻り、H2D完了時に枠が再利用可能になる。 転送前に閉じた場合は直ちに枠を返す。
 */
final class DecisionTrainingSealedHostBatch implements AutoCloseable {

  private final DecisionTrainingInputPipeline pipeline;
  private final DecisionTrainingInputPipeline.HostSlot slot;
  private final long generation;
  private final DecisionTrainingBatchSummary summary;
  private boolean claimed;

  DecisionTrainingSealedHostBatch(
      DecisionTrainingInputPipeline pipeline,
      DecisionTrainingInputPipeline.HostSlot slot,
      long generation,
      DecisionTrainingBatchSummary summary) {
    this.pipeline = pipeline;
    this.slot = slot;
    this.generation = generation;
    this.summary = summary;
  }

  /** バッチ集計値を返す。 */
  DecisionTrainingBatchSummary summary() {
    return summary;
  }

  /** パイプラインだけが呼ぶ、ホスト枠所有権の移動操作。 */
  synchronized DecisionTrainingInputPipeline.HostSlot claim(
      DecisionTrainingInputPipeline expectedPipeline) {
    if (pipeline != expectedPipeline) {
      throw new IllegalArgumentException("sealed host batch belongs to another training lane");
    }
    if (claimed) {
      throw new IllegalStateException("sealed host batch is already consumed");
    }
    claimed = true;
    pipeline.requireSealedHostSlot(slot, generation);
    return slot;
  }

  long generation() {
    return generation;
  }

  /** 未転送ならホスト枠をパイプラインへ返す。 */
  @Override
  public synchronized void close() {
    if (!claimed) {
      claimed = true;
      pipeline.releaseSealedHostSlot(slot, generation);
    }
  }
}
