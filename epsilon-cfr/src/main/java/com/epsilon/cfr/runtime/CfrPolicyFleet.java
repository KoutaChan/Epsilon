package com.epsilon.cfr.runtime;

import ai.djl.Device;
import com.epsilon.cfr.collection.CfrPolicy;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.network.NetworkDevices;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.spi.PolicyExecutionContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/** 同じ世代の凍結モデルを各デバイスへ複製し、要求順を保って並列推論する。 */
public final class CfrPolicyFleet implements CfrPolicy, AutoCloseable {
  private final List<CfrModel> models = new ArrayList<>();
  private final PolicyExecutionContext execution;
  private final boolean ownsExecution;
  private CfrInferenceCoordinator coordinator;
  private boolean closed;

  public CfrPolicyFleet(CfrModel source, NetworkDevices devices, CfrConfig.Collection settings) {
    this(
        source,
        devices,
        new PolicyExecutionContext(
            settings.arena().advanceWorkers(),
            settings.slotsPerDevice(),
            settings.readyBatchesPerDevice()),
        true);
  }

  /** 対戦用の実行コンテキストを借用する。呼び出し側は全推論器の処理完了後にコンテキストを解放する。 */
  public CfrPolicyFleet(CfrModel source, NetworkDevices devices, PolicyExecutionContext execution) {
    this(source, devices, execution, false);
  }

  private CfrPolicyFleet(
      CfrModel source,
      NetworkDevices devices,
      PolicyExecutionContext execution,
      boolean ownsExecution) {
    this.execution = Objects.requireNonNull(execution);
    this.ownsExecution = ownsExecution;
    try {
      for (Device device : devices) models.add(source.copyForInference(device));
      coordinator = new CfrInferenceCoordinator(models, execution);
    } catch (RuntimeException | Error failure) {
      try {
        close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  @Override
  public List<Prediction> evaluate(List<DecisionContext> batch) {
    if (batch.isEmpty()) return List.of();
    List<CompletableFuture<List<Prediction>>> pending = new ArrayList<>();
    int shards = Math.min(models.size(), batch.size());
    try {
      for (int shard = 0; shard < shards; shard++) {
        int from = batch.size() * shard / shards, to = batch.size() * (shard + 1) / shards;
        pending.add(coordinator.submit(batch.subList(from, to)));
      }
      CompletableFuture.allOf(pending.toArray(CompletableFuture[]::new)).join();
      List<Prediction> result = new ArrayList<>(batch.size());
      for (var future : pending) result.addAll(future.join());
      return result;
    } finally {
      // 入力生成やデバイス処理が失敗しても、借用モデルを使用中の処理が残ったまま復帰しない。
      for (var future : pending) future.handle((value, failure) -> null).join();
    }
  }

  @Override
  public boolean needsInferenceWork() {
    return coordinator.needsInferenceWork();
  }

  @Override
  public DecisionInferenceIngress.Attempt tryAcquireInferenceIngress() {
    return coordinator.tryAcquireInferenceIngress();
  }

  @Override
  public CompletableFuture<List<Prediction>> submitBatch(
      List<DecisionContext> batch, DecisionInferenceIngress ingress) {
    return coordinator.submit(batch, ingress);
  }

  @Override
  public synchronized void close() {
    if (closed) return;
    closed = true;
    // モデルを閉じる前に、共通ディスパッチャーで投入済みの処理と入力生成の予約をすべて回収する。
    if (coordinator != null) coordinator.close();
    Throwable failure = null;
    for (CfrModel model : models) {
      try {
        model.close();
      } catch (RuntimeException | Error problem) {
        if (failure == null) failure = problem;
        else failure.addSuppressed(problem);
      }
    }
    if (ownsExecution) {
      try {
        execution.close();
      } catch (RuntimeException | Error problem) {
        if (failure == null) failure = problem;
        else failure.addSuppressed(problem);
      }
    }
    if (failure instanceof RuntimeException problem) throw problem;
    if (failure instanceof Error problem) throw problem;
  }
}
