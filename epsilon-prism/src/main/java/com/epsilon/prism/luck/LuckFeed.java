package com.epsilon.prism.luck;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 運の網 Ω の塊を先に作る。標本の並びの区間を符号化のスレッドで分けて {@link LuckRows} に書き、学習のスレッドはデバイスの計算だけを回す。
 * 塊は決まった数を使い回す（借りて {@link #release} で返す）。符号化の道具はスレッドごとに持つ。
 */
final class LuckFeed implements AutoCloseable {

  /** 符号化のスレッドの道具。 */
  private static final class Tools {
    final LuckTokenizer tokenizer = new LuckTokenizer();
    final LuckObservation observation = new LuckObservation();
  }

  private static final ThreadLocal<Tools> TOOLS = ThreadLocal.withInitial(Tools::new);

  private final LuckSamples samples;
  private final ExecutorService workers;
  private final int threads;
  private final BlockingQueue<LuckRows> free;

  /**
   * 先読みを作る。
   *
   * @param samples 標本の表
   * @param capacity 塊の行の上限
   * @param buffers 塊の数（先読みの数 + 1 以上）
   * @param threads 符号化のスレッドの数
   */
  LuckFeed(LuckSamples samples, int capacity, int buffers, int threads) {
    this.samples = samples;
    this.threads = Math.max(1, threads);
    workers =
        Executors.newFixedThreadPool(
            this.threads,
            runnable -> {
              Thread thread = new Thread(runnable, "luck-feed");
              thread.setDaemon(true);
              return thread;
            });
    free = new ArrayBlockingQueue<>(buffers);
    for (int buffer = 0; buffer < buffers; buffer++) {
      free.add(new LuckRows(capacity));
    }
  }

  /**
   * 標本の並びの区間を塊に書き始める。
   *
   * @param order 標本の番号の並び
   * @param from 区間の始め
   * @param count 区間の長さ（塊の行の上限以下）
   * @return 書き終えた塊（使い終えたら {@link #release} で返す）
   */
  CompletableFuture<LuckRows> prepare(int[] order, int from, int count) {
    LuckRows rows;
    try {
      rows = free.take();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Interrupted while waiting for a luck batch", interrupted);
    }
    rows.resize(count);
    int chunk = Math.ceilDiv(count, threads);
    List<CompletableFuture<Void>> parts = new ArrayList<>(threads);
    for (int begin = 0; begin < count; begin += chunk) {
      int start = begin;
      int end = Math.min(count, begin + chunk);
      parts.add(
          CompletableFuture.runAsync(
              () -> {
                Tools tools = TOOLS.get();
                for (int row = start; row < end; row++) {
                  samples.put(order[from + row], rows, row, tools.tokenizer, tools.observation);
                }
              },
              workers));
    }
    return CompletableFuture.allOf(parts.toArray(CompletableFuture[]::new)).thenApply(done -> rows);
  }

  /**
   * 使い終えた塊を返す。
   *
   * @param rows 塊
   */
  void release(LuckRows rows) {
    free.add(rows);
  }

  @Override
  public void close() {
    workers.shutdownNow();
  }
}
