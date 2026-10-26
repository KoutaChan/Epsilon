package com.epsilon.prism.rows;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * 1コホートの行を、書き手ごとのシャードへ追記する。圧縮は書き手のスレッドで行い、収集のスレッドを止めない。
 *
 * <p>{@link #seal} で全シャードを同期してから {@value #MANIFEST} を原子的に置き、以後このディレクトリは変更しない。学習側は manifest のある
 * ディレクトリだけを読む。封印せずに閉じたら（収集が途中で失敗した）manifest は置かないので、再開時に途中のコホートとして消されて集め直す。
 */
public final class RowWriter implements RowSink {
  public static final String MANIFEST = "rows.json";
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  public record Shard(String file, long rows, long bytes) {}

  /**
   * 1 行の索引の長さ（[i32 枠の長さ][u8 purpose][u8 group][u8 source][u8 class][u8 measured]。{@link RowFormat.Index}）。shard と同じ名前の
   * .idx に行の順で並ぶ。
   */
  public static final int INDEX_ENTRY = 9;

  /** shard の索引のファイル（shard-00.rows → shard-00.idx）。 */
  public static Path index(Path shard) {
    String name = shard.getFileName().toString();
    return shard.resolveSibling(name.substring(0, name.lastIndexOf('.')) + ".idx");
  }

  /**
   * コホートの目録。
   *
   * @param rows 全シャードの行数
   * @param bytes 全シャードのバイト数
   */
  public record Manifest(int format, int iteration, long rows, long bytes, List<Shard> shards) {}

  private final Path directory;
  private final int iteration;
  private final List<Worker> workers = new ArrayList<>();
  private final AtomicInteger next = new AtomicInteger();
  private final AtomicLong backlog = new AtomicLong();
  private final LongAdder written = new LongAdder(), bytes = new LongAdder();
  private volatile Throwable failure;
  private boolean stopped, sealed;

  public RowWriter(Path directory, int iteration, int writers) throws IOException {
    this.directory = directory;
    this.iteration = iteration;
    Files.createDirectories(directory);
    if (Files.exists(directory.resolve(MANIFEST)))
      throw new IOException("Rows are already sealed: " + directory);
    for (int index = 0; index < writers; index++) workers.add(new Worker(index));
    for (Worker worker : workers) worker.thread.start();
  }

  /** 封印済みのコホートの目録を読む。manifest がなければ封印されていない（途中で失敗した）コホート。 */
  public static Manifest readManifest(Path directory) throws IOException {
    Path path = directory.resolve(MANIFEST);
    if (!Files.exists(path)) throw new IOException("Cohort is not sealed: " + directory);
    Manifest manifest = GSON.fromJson(Files.readString(path), Manifest.class);
    if (manifest.format() != RowFormat.VERSION)
      throw new IOException("Row format " + manifest.format() + " is not " + RowFormat.VERSION);
    return manifest;
  }

  /** 目録を同期して原子的に置く。 */
  private static void writeManifest(Path directory, Manifest manifest) throws IOException {
    Path temporary = directory.resolve(MANIFEST + ".tmp");
    try (FileChannel channel =
        FileChannel.open(
            temporary,
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING,
            StandardOpenOption.WRITE)) {
      channel.write(ByteBuffer.wrap(GSON.toJson(manifest).getBytes(StandardCharsets.UTF_8)));
      channel.force(true);
    }
    Files.move(temporary, directory.resolve(MANIFEST), StandardCopyOption.ATOMIC_MOVE);
  }

  @Override
  public void accept(PrismRow row) {
    Throwable problem = failure;
    if (problem != null) throw new IllegalStateException("Row writer failed", problem);
    backlog.incrementAndGet();
    workers.get(Math.floorMod(next.getAndIncrement(), workers.size())).queue.add(row);
  }

  @Override
  public long backlog() {
    return backlog.get();
  }

  @Override
  public RowStats stats() {
    return new RowStats(written.sum(), bytes.sum());
  }

  /** 残りを書き終え、同期して封印する。 */
  @Override
  public synchronized void seal() throws IOException {
    if (sealed) return;
    if (stopped) throw new IOException("Row writer was closed before sealing: " + directory);
    List<Shard> shards = stop();
    if (failure != null) throw new IOException("Row writer failed", failure);
    long rows = shards.stream().mapToLong(Shard::rows).sum();
    writeManifest(directory, new Manifest(RowFormat.VERSION, iteration, rows, bytes.sum(), shards));
    sealed = true;
  }

  /** 書き手を止める。封印していなければ manifest を置かない（途中のコホートとして残る）。 */
  @Override
  public synchronized void close() throws IOException {
    if (!stopped) stop();
  }

  /** 書き手に残りを書かせて止め、シャードの一覧を返す。 */
  private List<Shard> stop() throws IOException {
    stopped = true;
    for (Worker worker : workers) worker.queue.add(Worker.STOP);
    List<Shard> shards = new ArrayList<>();
    for (Worker worker : workers) {
      try {
        worker.thread.join();
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while stopping row writers", interrupted);
      }
      shards.add(new Shard(worker.file.getFileName().toString(), worker.rows, worker.bytes));
    }
    return shards;
  }

  private final class Worker {
    static final Object STOP = new Object();
    final LinkedBlockingQueue<Object> queue = new LinkedBlockingQueue<>();
    final Path file;
    final Thread thread;
    long rows, bytes;

    Worker(int index) {
      file = directory.resolve(String.format("shard-%02d.rows", index));
      thread = new Thread(this::run, "prism-row-" + index);
      thread.setDaemon(true);
    }

    void run() {
      try (FileChannel channel =
              FileChannel.open(
                  file,
                  StandardOpenOption.CREATE,
                  StandardOpenOption.TRUNCATE_EXISTING,
                  StandardOpenOption.WRITE);
          FileChannel indexChannel =
              FileChannel.open(
                  index(file),
                  StandardOpenOption.CREATE,
                  StandardOpenOption.TRUNCATE_EXISTING,
                  StandardOpenOption.WRITE)) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(1 << 20);
        ByteBuffer index =
            ByteBuffer.allocateDirect(INDEX_ENTRY << 13).order(ByteOrder.LITTLE_ENDIAN);
        while (true) {
          Object next = queue.take();
          if (next == STOP) break;
          PrismRow row = (PrismRow) next;
          byte[] frame = RowFormat.encodeFrame(row);
          if (buffer.remaining() < frame.length) flush(channel, buffer);
          if (frame.length > buffer.capacity()) writeFully(channel, ByteBuffer.wrap(frame));
          else buffer.put(frame);
          if (index.remaining() < INDEX_ENTRY) flush(indexChannel, index);
          RowFormat.Index entry = RowFormat.index(row, frame.length);
          index
              .putInt(frame.length)
              .put((byte) entry.purpose().ordinal())
              .put((byte) entry.group().ordinal())
              .put((byte) entry.source().ordinal())
              .put((byte) entry.decisionClass().ordinal())
              .put((byte) entry.measured());
          rows++;
          bytes += frame.length;
          RowWriter.this.bytes.add(frame.length);
          written.increment();
          backlog.decrementAndGet();
        }
        flush(channel, buffer);
        flush(indexChannel, index);
        channel.force(true);
        indexChannel.force(true);
      } catch (Throwable problem) {
        failure = problem;
      }
    }

    private void flush(FileChannel channel, ByteBuffer buffer) throws IOException {
      buffer.flip();
      writeFully(channel, buffer);
      buffer.clear();
    }

    private void writeFully(FileChannel channel, ByteBuffer source) throws IOException {
      while (source.hasRemaining()) channel.write(source);
    }
  }
}
