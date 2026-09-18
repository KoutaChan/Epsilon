package com.epsilon.pico.ai.decision.arena;

import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.runtime.BatchEncodingExecutor;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/** 系列固有の作業領域と保存領域を使い、共通の並列実行処理から入力を符号化する。 */
public final class DecisionBatchEncoder implements AutoCloseable {

  private final BatchEncodingExecutor<DecisionBucket, DecisionBatchBuilder, DecisionHostBatch>
      executor;

  public DecisionBatchEncoder(int workers) {
    var workspaces = new DecisionBatchBuilder.EncodingWorkspacePool(workers);
    executor =
        new BatchEncodingExecutor<>(
            workers,
            (rows, bucket) -> DecisionBatchBuilder.inference(rows, bucket, workspaces),
            DecisionBatchBuilder::buildEncodedInferenceRows);
  }

  public <R> CompletableFuture<DecisionHostBatch> encodeAsync(
      List<R> rows,
      DecisionBucket bucket,
      BatchEncodingExecutor.RangeEncoder<? super R, DecisionBatchBuilder> encoder) {
    return executor.encodeAsync(rows, bucket, encoder);
  }

  @Override
  public void close() {
    executor.close();
  }
}
