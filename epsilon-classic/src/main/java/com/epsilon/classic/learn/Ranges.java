package com.epsilon.classic.learn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** 行の範囲を等分し、並列に処理する。 */
final class Ranges {

  /** スレッドの数の上限。 */
  private static final int THREADS = 8;

  /** 1 つのスレッドに渡す行の数の下限。 */
  private static final int MIN_ROWS = 1 << 16;

  /** 行の範囲 [from, to) を処理する。 */
  @FunctionalInterface
  interface Task<T> {

    T run(int from, int to);
  }

  private Ranges() {}

  /** 行の範囲 [0, count) を等分して並列に処理し、範囲の順に結果を返す。行が少なければ呼んだスレッドだけで処理する。 */
  static <T> List<T> split(int count, Task<T> task) {
    int parts = Math.max(1, Math.min(THREADS, count / MIN_ROWS));
    if (parts == 1) {
      return Collections.singletonList(task.run(0, count));
    }
    try (ExecutorService pool = Executors.newFixedThreadPool(parts)) {
      List<Future<T>> futures = new ArrayList<>(parts);
      for (int part = 0; part < parts; part++) {
        int from = (int) ((long) count * part / parts);
        int to = (int) ((long) count * (part + 1) / parts);
        futures.add(pool.submit(() -> task.run(from, to)));
      }
      List<T> results = new ArrayList<>(parts);
      for (Future<T> future : futures) {
        results.add(future.get());
      }
      return results;
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Interrupted while scanning the rows", interrupted);
    } catch (ExecutionException failed) {
      if (failed.getCause() instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (failed.getCause() instanceof Error error) {
        throw error;
      }
      throw new IllegalStateException("Could not scan the rows", failed.getCause());
    }
  }
}
