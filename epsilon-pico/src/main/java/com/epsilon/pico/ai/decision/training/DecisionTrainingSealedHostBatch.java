package com.epsilon.pico.ai.decision.training;

/**
 * 復号が完了したホスト側バッファの利用権を、一度だけ引き渡す。
 *
 * <p>書き込み操作は公開しない。転送を開始すると管理をパイプラインへ戻し、転送完了時に実行枠を再利用可能にする。転送前に閉じた場合は、実行枠を直ちに返す。
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

  /** パイプラインだけが呼ぶ、ホスト側の実行枠所有権の移動操作。 */
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

  /** 未転送ならホスト側の実行枠をパイプラインへ返す。 */
  @Override
  public synchronized void close() {
    if (!claimed) {
      claimed = true;
      pipeline.releaseSealedHostSlot(slot, generation);
    }
  }
}
