package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrainingSampleDescriptorReader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/** 対局単位の学習データの順序を保ち、同時実行数を制限して先読みする。 */
final class EpsilonDecisionFragmentPrefetch implements AutoCloseable {

  private static final AtomicInteger THREAD_IDS = new AtomicInteger();

  private final List<Path> paths;
  private final EpsilonDecisionTrainingSampleDescriptorReader reader;
  private final int depth;
  private final ExecutorService executor;
  private final ArrayDeque<PendingFragment> pending = new ArrayDeque<>();
  private int nextIndex;

  EpsilonDecisionFragmentPrefetch(
      List<Path> paths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int depth,
      int workers) {
    this.paths = paths;
    this.reader = reader;
    this.depth = depth;
    executor =
        depth <= 1
            ? null
            : Executors.newFixedThreadPool(
                Math.min(workers, depth),
                task -> {
                  Thread thread =
                      new Thread(
                          task, "decision-fragment-prefetch-" + THREAD_IDS.incrementAndGet());
                  thread.setDaemon(true);
                  return thread;
                });
    fill();
  }

  Fragment next() throws IOException {
    if (executor == null) {
      if (nextIndex >= paths.size()) {
        return null;
      }
      Path path = paths.get(nextIndex++);
      return new Fragment(path, reader.read(path));
    }
    PendingFragment next = pending.pollFirst();
    if (next == null) {
      return null;
    }
    try {
      Fragment loaded = new Fragment(next.path(), next.future().get());
      fill();
      return loaded;
    } catch (InterruptedException failure) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while reading Decision fragment " + next.path(), failure);
    } catch (ExecutionException failure) {
      Throwable cause = failure.getCause();
      if (cause instanceof IOException io) {
        throw io;
      }
      if (cause instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw new IOException("Failed to read Decision fragment " + next.path(), cause);
    }
  }

  private void fill() {
    if (executor == null) {
      return;
    }
    while (pending.size() < depth && nextIndex < paths.size()) {
      Path path = paths.get(nextIndex++);
      Future<ArrayList<EpsilonDecisionTrainingSampleDescriptor>> future =
          executor.submit(() -> reader.read(path));
      pending.addLast(new PendingFragment(path, future));
    }
  }

  @Override
  public void close() {
    while (!pending.isEmpty()) {
      pending.removeFirst().future().cancel(true);
    }
    if (executor != null) {
      executor.close();
    }
  }

  record Fragment(Path path, ArrayList<EpsilonDecisionTrainingSampleDescriptor> samples) {}

  private record PendingFragment(
      Path path, Future<ArrayList<EpsilonDecisionTrainingSampleDescriptor>> future) {}
}
