package com.epsilon.cfr.runtime;

import com.epsilon.cfr.collection.Prediction;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.cfr.model.CfrModel;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

/** 推論と保存用符号化を合流し、両方が終了してから受付予約を完了させる。 */
final class CfrPredictionCompletion {
  private final Supplier<CompletableFuture<List<ObservationPayload>>> encode;
  private final Runnable handoff;
  private final AtomicBoolean encodingStarted = new AtomicBoolean();
  private final CompletableFuture<List<ObservationPayload>> payloads = new CompletableFuture<>();
  private final CompletableFuture<CfrModel.Policies> policies = new CompletableFuture<>();
  private final CompletableFuture<List<Prediction>> result;

  CfrPredictionCompletion(
      Supplier<CompletableFuture<List<ObservationPayload>>> encode, Runnable handoff) {
    this.encode = encode;
    this.handoff = handoff;
    // 一方が失敗しても、他方が入力や実行資源を使い終えるまで完了させない。
    result =
        policies
            .handle(Outcome::new)
            .thenCombine(
                payloads.handle(Outcome::new),
                (computed, saved) -> {
                  Throwable failure = computed.failure != null ? computed.failure : saved.failure;
                  if (failure != null) {
                    if (computed.failure != null
                        && saved.failure != null
                        && computed.failure != saved.failure) failure.addSuppressed(saved.failure);
                    throw new CompletionException(failure);
                  }
                  return computed.value.withObservations(saved.value);
                });
  }

  /** 実行枠へ移譲されたときに符号化を始め、保存入力を読み終えてからホスト側の予約を返す。 */
  void handoff() {
    if (encodingStarted.compareAndSet(false, true)) {
      try {
        encode
            .get()
            .whenComplete(
                (value, failure) -> {
                  if (failure == null) payloads.complete(value);
                  else payloads.completeExceptionally(failure);
                });
      } catch (Throwable failure) {
        payloads.completeExceptionally(failure);
      }
    }
    payloads.whenComplete((ignored, failure) -> handoff.run());
  }

  void complete(CfrModel.Policies value) {
    policies.complete(value);
  }

  void fail(Throwable failure) {
    // 符号化を始める前の失敗では新しいタスクを投入せず、開始済みの処理だけを回収する。
    if (encodingStarted.compareAndSet(false, true)) payloads.completeExceptionally(failure);
    policies.completeExceptionally(failure);
  }

  CompletableFuture<List<Prediction>> result() {
    return result;
  }

  private record Outcome<T>(T value, Throwable failure) {}
}
