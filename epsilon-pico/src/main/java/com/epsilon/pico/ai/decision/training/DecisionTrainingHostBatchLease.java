package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.ai.decision.input.DecisionTrainingSlabWriter;

/**
 * ホスト側の最終入力バッファへ書き込む間、実行枠を専有する。
 *
 * <p>{@link #seal(DecisionTrainingBatchSummary)}
 * で書き込みを終了し、完成したバッチへ管理を引き継ぐ。その後は書き込み用の参照を使ってはならない。完成前に閉じた場合は実行枠をパイプラインへ返す。
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

  /** 入力と確率分布の本体を直接書き込む最終連続バッファ書き込み処理を返す。 */
  DecisionTrainingSlabWriter writer() {
    if (writer == null) {
      throw new IllegalStateException("host batch lease is already consumed");
    }
    return writer;
  }

  /**
   * 全行を確定し、書込可能利用権を書き込みが完了したバッチへ一度だけ変換する。
   *
   * @param summary 復号と同時に集計したバッチ情報
   * @return 同じホスト側の実行枠を所有する書き込みが完了したバッチ
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

  /** 未書き込みの確定の位置を破棄して再利用可能にする。 */
  @Override
  public void close() {
    if (writer != null) {
      writer = null;
      pipeline.abortHostSlot(slot, generation);
    }
  }
}
