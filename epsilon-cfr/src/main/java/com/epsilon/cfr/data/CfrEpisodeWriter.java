package com.epsilon.cfr.data;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;

/** 完了した学習経路を半荘ごとの順序で保存する。待機中と保存中の経路数を制限し、半荘完了時だけ同期して公開する。 */
public final class CfrEpisodeWriter implements AutoCloseable {
  @FunctionalInterface
  interface Write {
    void accept(CfrEpisodeFrame frame) throws IOException;

    default void finish() throws IOException {}
  }

  private final ArrayDeque<CfrEpisodeFrame> queue = new ArrayDeque<>();
  private final HashSet<Long> writingGames = new HashSet<>();
  private final Thread[] workers;
  private final Write write;
  private final int capacity;
  private int pending;
  private boolean closing;
  private Throwable failure;
  private CompletableFuture<Void> available = CompletableFuture.completedFuture(null);

  public CfrEpisodeWriter(Path directory) {
    this(4, 16, new MatchWrites(directory));
  }

  CfrEpisodeWriter(int writerCount, int capacity, Write write) {
    if (writerCount <= 0 || capacity < writerCount)
      throw new IllegalArgumentException("Invalid episode writer capacity");
    this.capacity = capacity;
    this.write = Objects.requireNonNull(write);
    workers = new Thread[writerCount];
    for (int i = 0; i < workers.length; i++) {
      workers[i] = new Thread(this::run, "cfr-episode-writer-" + i);
      workers[i].setDaemon(true);
      workers[i].start();
    }
  }

  /** 待機中と書き込み中の経路を合わせて{@code capacity}件まで受け付ける。上限に達した場合は、空きができるまで収集側を待機させる。 */
  public synchronized void accept(CfrEpisodeFrame frame) {
    Objects.requireNonNull(frame);
    try {
      while (pending >= capacity && failure == null && !closing) wait();
      checkFailure();
      if (closing) throw new IllegalStateException("Episode writer is closed");
      enqueue(frame);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new UncheckedIOException(
          new IOException("Interrupted enqueueing episode", interrupted));
    }
  }

  /** 待機中と書き込み中の件数に空きがある場合だけ受け付ける。満杯の場合は経路を保持せずに戻る。 */
  public synchronized boolean tryAccept(CfrEpisodeFrame frame) {
    Objects.requireNonNull(frame);
    checkFailure();
    if (closing) throw new IllegalStateException("Episode writer is closed");
    if (pending == capacity) return false;
    enqueue(frame);
    return true;
  }

  /** 満杯の場合は、空き・保存失敗・終了のいずれかが発生したときに通知する。通知を得てから登録しても、空きの発生を取りこぼさない。 */
  public synchronized CompletionStage<Void> whenAvailable() {
    if (failure != null)
      return CompletableFuture.failedStage(
          new UncheckedIOException(new IOException("Episode persistence failed", failure)));
    if (closing)
      return CompletableFuture.failedStage(new IllegalStateException("Episode writer is closed"));
    return available.minimalCompletionStage();
  }

  private void enqueue(CfrEpisodeFrame frame) {
    if (++pending == capacity) available = new CompletableFuture<>();
    queue.addLast(frame);
    notifyAll();
  }

  /** 先行局を書き込み中の半荘だけを飛ばし、最も早く受け付けた処理可能な局を取り出す。 */
  private CfrEpisodeFrame takeReady() {
    var iterator = queue.iterator();
    while (iterator.hasNext()) {
      CfrEpisodeFrame frame = iterator.next();
      long gameId = frame.episode().episodeId();
      if (writingGames.add(gameId)) {
        iterator.remove();
        return frame;
      }
    }
    return null;
  }

  private void run() {
    try {
      while (true) {
        CfrEpisodeFrame frame;
        synchronized (this) {
          while (true) {
            if (failure != null) return;
            frame = takeReady();
            if (frame != null) break;
            if (closing && queue.isEmpty()) return;
            wait();
          }
        }
        long gameId = frame.episode().episodeId();
        write.accept(frame);
        frame = null;
        CompletableFuture<Void> ready;
        synchronized (this) {
          writingGames.remove(gameId);
          pending--;
          ready = available;
          notifyAll();
        }
        // 収集側の通知処理を保存先のロックの外で実行する。
        ready.complete(null);
      }
    } catch (Throwable error) {
      CompletableFuture<Void> ready;
      Throwable firstFailure;
      synchronized (this) {
        if (failure == null) failure = error;
        firstFailure = failure;
        queue.clear();
        ready = available;
        notifyAll();
      }
      ready.completeExceptionally(
          new UncheckedIOException(new IOException("Episode persistence failed", firstFailure)));
    }
  }

  @Override
  public void close() throws IOException {
    CompletableFuture<Void> ready;
    synchronized (this) {
      closing = true;
      ready = available;
      notifyAll();
    }
    ready.completeExceptionally(new IllegalStateException("Episode writer is closed"));
    // 受理した経路の保存が終わるまで待つ。半荘が未完了なら.partだけを残し、公開しない。
    boolean interrupted = false;
    for (Thread worker : workers) {
      while (worker.isAlive()) {
        try {
          worker.join();
        } catch (InterruptedException ignored) {
          interrupted = true;
        }
      }
    }
    if (interrupted) Thread.currentThread().interrupt();
    synchronized (this) {
      if (failure != null) throw new IOException("Episode persistence failed", failure);
    }
    write.finish();
    if (interrupted) throw new IOException("Interrupted draining episode writers");
  }

  private static final class MatchWrites implements Write {
    private final Path directory;
    private final ConcurrentHashMap<Long, CfrMatchFile.Appender> matches =
        new ConcurrentHashMap<>();

    private MatchWrites(Path directory) {
      this.directory = Objects.requireNonNull(directory);
    }

    @Override
    public void accept(CfrEpisodeFrame frame) throws IOException {
      long gameId = frame.episode().episodeId();
      CfrMatchFile.Appender match =
          matches.computeIfAbsent(
              gameId,
              id ->
                  new CfrMatchFile.Appender(
                      directory.resolve(String.format(Locale.ROOT, "%08d.bin", id)), id));
      match.append(frame);
      if (frame.last()) matches.remove(gameId, match);
    }

    @Override
    public void finish() throws IOException {
      if (!matches.isEmpty())
        throw new IOException("CFR writer has unfinished matches: " + matches.size());
    }
  }

  private void checkFailure() {
    if (failure != null)
      throw new UncheckedIOException(new IOException("Episode persistence failed", failure));
  }
}
