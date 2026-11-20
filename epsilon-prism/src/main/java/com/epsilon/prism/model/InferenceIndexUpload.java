package com.epsilon.prism.model;

import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.PinnedUploads;

/** 受付内の全チャンクの推論用インデックスを、既存の H2D ストリームへ一括で積む。 */
public final class InferenceIndexUpload {
  private InferenceIndexUpload() {}

  /** 転送先は受付の完了まで owner が保持する。計算側は呼び出し元の H2D 完了イベントを待つ。 */
  public static void enqueue(
      NetworkInputs[] chunks, NDManager owner, PinnedUploads uploads, PtStream h2d) {
    Staging staging = new Staging();
    for (int i = 0; i < chunks.length; i++) {
      NetworkInputs input = chunks[i];
      // 各チャンクを個別に送る場合と同じ境界にそろえる。
      staging.align();
      BatchIndex index = new BatchIndex(input, null, null, staging);
      chunks[i] =
          new NetworkInputs(
              input.format(),
              input.tensors(),
              input.rows(),
              input.events(),
              input.transitions(),
              input.actions(),
              input.rowColumns(),
              input.actionColumns(),
              null,
              index);
    }
    staging.write(uploads);
    staging.enqueue(owner, h2d);
  }
}
