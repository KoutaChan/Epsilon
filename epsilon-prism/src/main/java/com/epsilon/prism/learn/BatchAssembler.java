package com.epsilon.prism.learn;

import com.epsilon.prism.rows.PrismRow;
import com.epsilon.prism.rows.ObservationCodec;
import com.epsilon.prism.rows.RowSource;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/** 行を展開する。展開（zstd と入力の復号）は専用のスレッド群（prism-lrn）で並列に行い、GPU の学習・採点と重ねる。 */
public final class BatchAssembler implements AutoCloseable {
  private final ExecutorService decoders;
  private final int threads;

  /** 展開の時間（報告だけ）: 行の読み出しと伸長、入力の復号、行数。スレッドごとの時間の和。 */
  private final LongAdder readNanos = new LongAdder(), codecNanos = new LongAdder();
  private final LongAdder decodedRows = new LongAdder();

  /** これまでの展開の {読み出し秒, 復号秒, 行数}（スレッドごとの時間の和）。 */
  public double[] stats() {
    return new double[] {readNanos.sum() / 1e9, codecNanos.sum() / 1e9, decodedRows.sum()};
  }

  /** 展開した行。hosts[i] は rows[i] の payload を復号した入力。 */
  public record Decoded(PrismRow[] rows, DecisionHostBatch[] hosts) {}

  public BatchAssembler(int threads) {
    this.threads = threads;
    var index = new AtomicInteger();
    decoders =
        Executors.newFixedThreadPool(
            threads,
            task -> {
              Thread thread = new Thread(task, "prism-lrn-" + index.getAndIncrement());
              thread.setDaemon(true);
              return thread;
            });
  }

  /** 並べ替えの塊の長さ。塊の中は元の順（貯蔵庫の索引の順 = shard の位置の順）のまま。 */
  static final int BLOCK = 64;

  /**
   * rows を BLOCK 行ずつの塊で seed で並べ替えた写し。塊の中は元の順なので、1 バッチの読み出しが shard の連続した位置の読みになり、
   * 1 行ずつのランダム読み（NVMe で 1 ms 近く）が先読みの効く順次読みになる。塊は同じ半荘の判断を含むので、バッチの独立な半荘の数は
   * 行数 / BLOCK 程度に減る（歩数が多いので平均で薄まる）。
   */
  public static int[] shuffle(int[] rows, long seed) {
    int blocks = Math.ceilDiv(rows.length, BLOCK);
    int[] starts = new int[blocks];
    for (int b = 0; b < blocks; b++) starts[b] = b * BLOCK;
    var random = new SplittableRandom(seed);
    for (int i = blocks - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int swap = starts[i];
      starts[i] = starts[j];
      starts[j] = swap;
    }
    int[] order = new int[rows.length];
    int at = 0;
    for (int start : starts) {
      int end = Math.min(rows.length, start + BLOCK);
      System.arraycopy(rows, start, order, at, end - start);
      at += end - start;
    }
    return order;
  }

  /** order の [from, from + size) の部分（末尾で切る）。 */
  public static int[] batch(int[] order, int from, int size) {
    return Arrays.copyOfRange(order, from, Math.min(order.length, from + size));
  }

  /** source の行 rows を非同期に展開する。返り値の順は rows の順。 */
  public CompletableFuture<Decoded> decode(RowSource source, int[] rows) {
    int size = rows.length;
    var decoded = new Decoded(new PrismRow[size], new DecisionHostBatch[size]);
    List<CompletableFuture<Void>> parts = new ArrayList<>(threads);
    int chunk = Math.max(1, Math.ceilDiv(size, threads));
    for (int from = 0; from < size; from += chunk) {
      int start = from, end = Math.min(size, from + chunk);
      parts.add(
          CompletableFuture.runAsync(
              () -> {
                long reading = 0, decoding = 0;
                for (int i = start; i < end; i++) {
                  long began = System.nanoTime();
                  try {
                    decoded.rows()[i] = source.read(rows[i]);
                  } catch (IOException failure) {
                    throw new UncheckedIOException(failure);
                  }
                  long read = System.nanoTime();
                  decoded.hosts()[i] = ObservationCodec.decode(decoded.rows()[i].payload());
                  reading += read - began;
                  decoding += System.nanoTime() - read;
                }
                readNanos.add(reading);
                codecNanos.add(decoding);
                decodedRows.add(end - start);
              },
              decoders));
    }
    return CompletableFuture.allOf(parts.toArray(CompletableFuture[]::new))
        .thenApply(ignored -> decoded);
  }

  @Override
  public void close() {
    decoders.shutdownNow();
  }
}
