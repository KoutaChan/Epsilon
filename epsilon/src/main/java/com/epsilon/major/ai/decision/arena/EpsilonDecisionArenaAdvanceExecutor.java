package com.epsilon.major.ai.decision.arena;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 対局をワーカーごとに分担し、同時実行数を制限して進行させる。
 *
 * <p>1ゲームごとにタスクを作らず、ワーカー数ぶんの処理分担へ分ける。結果の適用と完了ゲームの公開は呼出側が行う。
 */
final class EpsilonDecisionArenaAdvanceExecutor implements AutoCloseable {

  private static final AtomicInteger THREAD_SEQUENCE = new AtomicInteger();

  private final int workers;
  private final ExecutorService executor;

  EpsilonDecisionArenaAdvanceExecutor(int workers) {
    if (workers <= 0) {
      throw new IllegalArgumentException("workers must be positive");
    }
    this.workers = workers;
    executor =
        Executors.newFixedThreadPool(
            workers,
            task -> {
              Thread thread =
                  new Thread(
                      task, "epsilon-decision-arena-advance-" + THREAD_SEQUENCE.getAndIncrement());
              thread.setDaemon(true);
              return thread;
            });
  }

  void invoke(int itemCount, IndexedAdvance advance) throws Exception {
    if (itemCount < 0) {
      throw new IllegalArgumentException("itemCount must be non-negative");
    }
    if (itemCount == 0) {
      return;
    }
    if (workers == 1 || itemCount == 1) {
      runSerial(itemCount, advance);
      return;
    }

    int taskCount = Math.min(workers, itemCount);
    ExecutorCompletionService<Void> completion = new ExecutorCompletionService<>(executor);
    ArrayList<Future<Void>> futures = new ArrayList<>(taskCount);
    try {
      for (int lane = 0; lane < taskCount; lane++) {
        int firstIndex = itemCount - 1 - lane;
        futures.add(
            completion.submit(
                () -> {
                  for (int index = firstIndex; index >= 0; index -= taskCount) {
                    requireRunning();
                    advance.run(index);
                  }
                  return null;
                }));
      }
      for (int completed = 0; completed < taskCount; completed++) {
        completion.take().get();
      }
    } catch (InterruptedException failure) {
      cancel(futures);
      executor.shutdownNow();
      Thread.currentThread().interrupt();
      throw failure;
    } catch (ExecutionException failure) {
      cancel(futures);
      executor.shutdownNow();
      throw workerFailure(failure.getCause());
    } catch (RuntimeException | Error failure) {
      cancel(futures);
      executor.shutdownNow();
      throw failure;
    }
  }

  <T> CompletableFuture<T> submit(Callable<T> task) {
    CompletableFuture<T> future = new CompletableFuture<>();
    executor.execute(
        () -> {
          try {
            future.complete(task.call());
          } catch (Throwable failure) {
            future.completeExceptionally(failure);
          }
        });
    return future;
  }

  private static void runSerial(int itemCount, IndexedAdvance advance) throws Exception {
    for (int index = itemCount - 1; index >= 0; index--) {
      requireRunning();
      advance.run(index);
    }
  }

  private static void requireRunning() throws InterruptedException {
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException("Decision arena advance worker was cancelled");
    }
  }

  private static void cancel(List<? extends Future<?>> futures) {
    for (Future<?> future : futures) {
      future.cancel(true);
    }
  }

  private static Exception workerFailure(Throwable cause) {
    if (cause instanceof Error error) {
      throw error;
    }
    if (cause instanceof Exception exception) {
      return exception;
    }
    return new IllegalStateException("Decision arena advance worker failed", cause);
  }

  @Override
  public void close() {
    executor.shutdown();
    boolean interrupted = false;
    while (!executor.isTerminated()) {
      try {
        if (!executor.awaitTermination(1L, TimeUnit.SECONDS)) {
          executor.shutdownNow();
        }
      } catch (InterruptedException failure) {
        interrupted = true;
        executor.shutdownNow();
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  @FunctionalInterface
  interface IndexedAdvance {
    void run(int index) throws Exception;
  }
}
