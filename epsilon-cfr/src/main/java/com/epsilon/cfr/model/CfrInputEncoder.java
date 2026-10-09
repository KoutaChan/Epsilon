package com.epsilon.cfr.model;

import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionInferenceStoragePool;
import com.epsilon.spi.PolicyExecutionContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/** NanoのCPUエンコーダーで盤面入力を確定し、その入力から保存用観測を構築する。 */
final class CfrInputEncoder {
  private record InputBucket(DecisionBucket bucket, boolean forced) {}

  private record PendingGroup(DecisionBatchBuilder builder, int[] indices, boolean forced) {}

  private final DecisionBatchBuilder.EncodingWorkspacePool encodingWorkspaces =
      new DecisionBatchBuilder.EncodingWorkspacePool(1);

  CfrModel.EncodedBatch encode(List<DecisionContext> contexts, PolicyExecutionContext execution) {
    return encodeRows(contexts, execution, false, null).join();
  }

  CompletableFuture<CfrModel.EncodedBatch> encodeAsync(
      List<DecisionContext> contexts, PolicyExecutionContext execution) {
    return encodeRows(contexts, Objects.requireNonNull(execution), true, null);
  }

  CompletableFuture<CfrModel.EncodedBatch> encodeAsync(
      List<DecisionContext> contexts,
      PolicyExecutionContext execution,
      DecisionInferenceStoragePool.Lease storage) {
    return encodeRows(
        contexts, Objects.requireNonNull(execution), true, Objects.requireNonNull(storage));
  }

  private CompletableFuture<CfrModel.EncodedBatch> encodeRows(
      List<DecisionContext> contexts,
      PolicyExecutionContext execution,
      boolean asynchronous,
      DecisionInferenceStoragePool.Lease storage) {
    var buckets = new LinkedHashMap<InputBucket, List<Integer>>();
    for (int index = 0; index < contexts.size(); index++) {
      var legal = contexts.get(index).legalActions();
      var key =
          new InputBucket(DecisionBatchBuilder.selectInferenceBucket(legal), legal.size() == 1);
      buckets.computeIfAbsent(key, ignored -> new ArrayList<>()).add(index);
    }
    List<PendingGroup> pending = new ArrayList<>(buckets.size());
    buckets.forEach(
        (key, indices) ->
            pending.add(
                new PendingGroup(
                    storage == null
                        ? DecisionBatchBuilder.inference(
                            indices.size(), key.bucket(), encodingWorkspaces)
                        : storage.inference(indices.size(), key.bucket(), encodingWorkspaces),
                    indices.stream().mapToInt(Integer::intValue).toArray(),
                    key.forced())));
    PolicyExecutionContext.RangeEncoder range =
        (from, to) -> {
          int offset = 0;
          for (PendingGroup group : pending) {
            int first = Math.max(0, from - offset);
            int end = Math.min(group.indices.length, to - offset);
            if (first < end) {
              try (var session = group.builder.openEncoding()) {
                for (int row = first; row < end; row++) {
                  var context = contexts.get(group.indices[row]);
                  session.encodeObservedInferenceRow(
                      row, context.observation(), context.legalActions());
                }
              }
            }
            offset += group.indices.length;
          }
        };
    return encodeRange(execution, asynchronous, contexts.size(), range)
        .thenApply(
            ignored -> {
              // 全ワーカーの書き込み終了後にだけ確定し、以降は同じ入力を読み取る。
              List<CfrModel.EncodedGroup> groups =
                  pending.stream()
                      .map(
                          group ->
                              new CfrModel.EncodedGroup(
                                  group.builder.buildEncodedInferenceRows(),
                                  group.indices,
                                  group.forced))
                      .toList();
              return new CfrModel.EncodedBatch(groups, contexts.size());
            });
  }

  /** 確定済み入力だけを読み、推論と独立して保存用観測を構築する。 */
  static CompletableFuture<List<ObservationPayload>> encodePayloads(
      CfrModel.EncodedBatch batch, PolicyExecutionContext execution) {
    var payloads = new ObservationPayload[batch.size()];
    PolicyExecutionContext.RangeEncoder save =
        (from, to) -> {
          int offset = 0;
          for (CfrModel.EncodedGroup group : batch.groups) {
            int first = Math.max(0, from - offset);
            int end = Math.min(group.indices.length, to - offset);
            for (int row = first; row < end; row++)
              payloads[group.indices[row]] = CfrObservationCodec.encodePayload(group.host, row);
            offset += group.indices.length;
          }
        };
    return encodeRange(execution, true, batch.size(), save)
        .thenApply(ignored -> List.copyOf(Arrays.asList(payloads)));
  }

  private static CompletableFuture<Void> encodeRange(
      PolicyExecutionContext execution,
      boolean asynchronous,
      int rows,
      PolicyExecutionContext.RangeEncoder range) {
    if (execution == null) {
      range.encode(0, rows);
      return CompletableFuture.completedFuture(null);
    }
    return asynchronous
        ? execution.encodeRowsAsync(rows, range)
        : execution.encodeRows(rows, range);
  }

  /** 保存済みの観測も容量区分ごとにまとめ、直接構築した入力と同じ形式へ揃える。 */
  CfrModel.EncodedBatch decode(List<ObservationPayload> observations) {
    var rows = observations.stream().map(CfrObservationCodec::decode).toList();
    var buckets = new LinkedHashMap<InputBucket, List<Integer>>();
    for (int index = 0; index < rows.size(); index++) {
      var host = rows.get(index).host();
      var key = new InputBucket(host.bucket(), host.legalActionCount(0) == 1);
      buckets.computeIfAbsent(key, ignored -> new ArrayList<>()).add(index);
    }
    List<CfrModel.EncodedGroup> groups = new ArrayList<>(buckets.size());
    buckets.forEach(
        (key, indices) ->
            groups.add(
                new CfrModel.EncodedGroup(
                    CfrObservationCodec.batch(indices.stream().map(rows::get).toList()),
                    indices.stream().mapToInt(Integer::intValue).toArray(),
                    key.forced())));
    return new CfrModel.EncodedBatch(groups, observations.size());
  }
}
