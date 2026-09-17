package com.epsilon.nano.ai.decision.training;

import com.epsilon.nano.ai.decision.input.DecisionTrainingSlabWriter;

/**
 * 最終転送先のホストバッファへ書き込む間、そのバッファ枠の利用権を保持する。
 *
 * <p>{@link #seal(DecisionTrainingBatchSummary)}は書き込み処理の所有権を消費し、変更できない {@link
 * DecisionTrainingSealedHostBatch}へ移す。確定後の書き込み処理参照は使用してはならない。未完了利用権を閉じると枠を破棄してパイプラインへ返す。
 */
final class DecisionTrainingHostBatchLease implements AutoCloseable {

  private final DecisionTrainingInputPipeline pipeline;
  private final DecisionTrainingInputPipeline.HostSlot slot;
  private final long generation;
  private DecisionTrainingSlabWriter writer;

  DecisionTrainingHostBatchLease(
      DecisionTrainingInputPipeline pipeline,
      DecisionTrainingInputPipeline.HostSlot slot,
      long generation,
      DecisionTrainingSlabWriter writer) {
    this.pipeline = pipeline;
    this.slot = slot;
    this.generation = generation;
    this.writer = writer;
  }

  /** データ本体を直接書き込む最終連続バッファ書き込み処理を返す。 */
  DecisionTrainingSlabWriter writer() {
    if (writer == null) {
      throw new IllegalStateException("host batch lease is already consumed");
    }
    return writer;
  }

  /**
   * 全行を確定し、書込可能利用権を確定済みバッチへ一度だけ変換する。
   *
   * @param summary 復号と同時に集計したバッチ情報
   * @return 同じホスト枠を所有する確定済みバッチ
   */
  DecisionTrainingSealedHostBatch seal(DecisionTrainingBatchSummary summary) {
    DecisionTrainingSlabWriter activeWriter = writer();
    activeWriter.seal();
    pipeline.sealHostSlot(
        slot,
        generation,
        activeWriter.playerMemoryPresentCount(),
        activeWriter.transitionPresentCount());
    writer = null;
    return new DecisionTrainingSealedHostBatch(pipeline, slot, generation, summary);
  }

  /** 未確定の枠を破棄して再利用可能にする。 */
  @Override
  public void close() {
    if (writer != null) {
      writer = null;
      pipeline.abortHostSlot(slot, generation);
    }
  }
}
