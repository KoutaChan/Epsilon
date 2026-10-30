package com.epsilon.prism.rows;

import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.observe.ObservationSchema;
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

/**
 * 1 コホートの行（{@link Row}）を、書き手ごとのシャードへ追記する（rows-v2/ の下に置く）。圧縮は書き手のスレッドで行い、収集の
 * スレッドを止めない。行のフレームは {@link RowCodec}、シャードと同じ名前の索引のファイル（{@link #index}）には行の順に
 * {@link #INDEX_ENTRY} バイトの索引を並べる（学習側はシャードの本体に触らずに行を選べる）。
 *
 * <p>目録（{@value #MANIFEST}）には行数・バイト数・シャードの一覧に加えて、帳簿の欄の数（全ての行で同じ）と記録の指紋（{@link Fingerprints}）
 * を書く。読む側は {@link #readManifest} で指紋を今の形式と照らし、合わなければ止める（版を持たず、読み分けもしない）。
 *
 * <p>{@link #seal} で全シャードを同期してから目録を原子的に置き、以後このディレクトリは変えない。封印せずに閉じたら（収集が途中で失敗した）
 * 目録を置かないので、再開時に途中のコホートとして消して集め直す。
 */
public final class CohortWriter implements AutoCloseable {

  /** 目録のファイル名。 */
  public static final String MANIFEST = "rows.json";

  /** run の下のコホートの置き場。 */
  public static final String ROOT = "rows-v2";

  /**
   * 1 行の索引の長さ（[i32 フレームの長さ][u8 purpose][u8 group][u8 source][u8 class][u8 measured]。{@link RowCodec.Index}）。
   */
  public static final int INDEX_ENTRY = 9;

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  /**
   * run のコホート t の行のディレクトリを返す（番号は 5 桁）。
   *
   * @param run run のディレクトリ
   * @param cohort コホートの番号
   * @return ディレクトリ
   */
  public static Path directory(Path run, int cohort) {
    return run.resolve(ROOT).resolve(String.format("%05d", cohort));
  }

  /**
   * シャードの索引のファイル（shard-00.rows → shard-00.idx）を返す。
   *
   * @param shard シャードのファイル
   * @return 索引のファイル
   */
  public static Path index(Path shard) {
    String name = shard.getFileName().toString();
    return shard.resolveSibling(name.substring(0, name.lastIndexOf('.')) + ".idx");
  }

  /**
   * シャード 1 つの目録。
   *
   * @param file シャードのファイル名
   * @param rows 行数
   * @param bytes バイト数
   */
  public record Shard(String file, long rows, long bytes) {}

  /**
   * 記録の指紋。
   *
   * @param observation 記録 O（{@link ObservationSchema#fingerprint}）
   * @param outcome 局末の記録（{@link OutcomeCodec#fingerprint}）
   * @param oracle 神託（{@link OracleCodec#fingerprint}）
   */
  public record Fingerprints(String observation, String outcome, String oracle) {

    /** この組み立ての記録の指紋。 */
    public static Fingerprints current() {
      return new Fingerprints(
          ObservationSchema.fingerprint(), OutcomeCodec.fingerprint(), OracleCodec.fingerprint());
    }
  }

  /**
   * コホートの目録。
   *
   * @param iteration 行を集めたコホート
   * @param rows 全シャードの行数
   * @param bytes 全シャードのバイト数
   * @param ledgerFeatures 帳簿の欄の数
   * @param fingerprints 記録の指紋
   * @param shards シャードの一覧
   */
  public record Manifest(
      int iteration,
      long rows,
      long bytes,
      int ledgerFeatures,
      Fingerprints fingerprints,
      List<Shard> shards) {}

  private final Path directory;
  private final int iteration;
  private final int ledgerFeatures;
  private final List<Worker> workers = new ArrayList<>();
  private final AtomicInteger next = new AtomicInteger();
  private final AtomicLong backlog = new AtomicLong();
  private final AtomicLong bytes = new AtomicLong();
  private volatile Throwable failure;
  private boolean stopped;
  private boolean sealed;

  /**
   * 書き手を始める。
   *
   * @param directory コホートのディレクトリ
   * @param iteration コホート
   * @param ledgerFeatures 帳簿の欄の数（全ての行で同じ）
   * @param writers 書き手のスレッドの数（シャードの数）
   * @throws IllegalArgumentException 書き手の数か帳簿の欄の数が範囲の外のとき
   * @throws IOException 既に封印されているか、ディレクトリを作れないとき
   */
  public CohortWriter(Path directory, int iteration, int ledgerFeatures, int writers)
      throws IOException {
    if (writers < 1) {
      throw new IllegalArgumentException("writer count must be positive: " + writers);
    }
    this.directory = directory;
    this.iteration = iteration;
    this.ledgerFeatures = ledgerFeatures;
    Files.createDirectories(directory);
    if (Files.exists(directory.resolve(MANIFEST))) {
      throw new IOException("Rows are already sealed: " + directory);
    }
    for (int index = 0; index < writers; index++) {
      workers.add(new Worker(index));
    }
    for (Worker worker : workers) {
      worker.thread.start();
    }
  }

  /**
   * 封印済みのコホートの目録を読み、記録の指紋を今の形式と照らす。
   *
   * @param directory コホートのディレクトリ
   * @return 目録
   * @throws IOException 封印されていないか、コホートの目録でないか、記録の指紋が合わないとき
   */
  public static Manifest readManifest(Path directory) throws IOException {
    Path path = directory.resolve(MANIFEST);
    if (!Files.exists(path)) {
      throw new IOException("Cohort is not sealed: " + directory);
    }
    Manifest manifest = GSON.fromJson(Files.readString(path), Manifest.class);
    if (manifest == null || manifest.fingerprints() == null || manifest.shards() == null) {
      throw new IOException("Not a cohort manifest: " + path);
    }
    Fingerprints current = Fingerprints.current();
    if (!current.equals(manifest.fingerprints())) {
      throw new IOException(
          "Cohort "
              + directory
              + " has record fingerprints "
              + manifest.fingerprints()
              + " but this build reads "
              + current);
    }
    return manifest;
  }

  /**
   * 行を書き手に渡す。どのスレッドからでも呼べ、行を捨てない。
   *
   * @param row 行
   * @throws IllegalStateException 書き手が失敗しているとき
   */
  public void accept(Row row) {
    Throwable problem = failure;
    if (problem != null) {
      throw new IllegalStateException("Row writer failed", problem);
    }
    backlog.incrementAndGet();
    workers.get(Math.floorMod(next.getAndIncrement(), workers.size())).queue.add(row);
  }

  /** まだ書き終えていない行数。 */
  public long backlog() {
    return backlog.get();
  }

  /**
   * 残りを書き終え、同期して封印する。
   *
   * @throws IOException 書き手が失敗したか、封印の前に閉じたとき
   */
  public synchronized void seal() throws IOException {
    if (sealed) {
      return;
    }
    if (stopped) {
      throw new IOException("Row writer was closed before sealing: " + directory);
    }
    List<Shard> shards = stop();
    if (failure != null) {
      throw new IOException("Row writer failed", failure);
    }
    long rows = 0;
    for (Shard shard : shards) {
      rows += shard.rows();
    }
    writeManifest(
        new Manifest(
            iteration, rows, bytes.get(), ledgerFeatures, Fingerprints.current(), shards));
    sealed = true;
  }

  /** 書き手を止める。封印していなければ目録を置かない（途中のコホートとして残る）。 */
  @Override
  public synchronized void close() throws IOException {
    if (!stopped) {
      stop();
    }
  }

  /** 書き手に残りを書かせて止め、シャードの一覧を返す。 */
  private List<Shard> stop() throws IOException {
    stopped = true;
    for (Worker worker : workers) {
      worker.queue.add(Worker.STOP);
    }
    List<Shard> shards = new ArrayList<>();
    for (Worker worker : workers) {
      try {
        worker.thread.join();
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while stopping row writers", interrupted);
      }
      String name = worker.file.getFileName().toString();
      shards.add(new Shard(name, worker.rows, worker.bytes));
    }
    return shards;
  }

  /** 目録を同期して原子的に置く。 */
  private void writeManifest(Manifest manifest) throws IOException {
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

  /** シャード 1 つの書き手。行の直列化はこのスレッドで行う。 */
  private final class Worker {

    static final Object STOP = new Object();

    final LinkedBlockingQueue<Object> queue = new LinkedBlockingQueue<>();
    final Path file;
    final Thread thread;
    final RowCodec codec = new RowCodec(ledgerFeatures);
    long rows;
    long bytes;

    Worker(int index) {
      file = directory.resolve(String.format("shard-%02d.rows", index));
      thread = new Thread(this::run, "prism-row-" + index);
      thread.setDaemon(true);
    }

    void run() {
      try (FileChannel channel = open(file);
          FileChannel indexChannel = open(index(file))) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(1 << 20);
        ByteBuffer index =
            ByteBuffer.allocateDirect(INDEX_ENTRY << 13).order(ByteOrder.LITTLE_ENDIAN);
        while (true) {
          Object next = queue.take();
          if (next == STOP) {
            break;
          }
          Row row = (Row) next;
          byte[] frame = codec.encode(row);
          if (buffer.remaining() < frame.length) {
            flush(channel, buffer);
          }
          if (frame.length > buffer.capacity()) {
            writeFully(channel, ByteBuffer.wrap(frame));
          } else {
            buffer.put(frame);
          }
          if (index.remaining() < INDEX_ENTRY) {
            flush(indexChannel, index);
          }
          RowCodec.Index entry = RowCodec.index(row, frame.length);
          index
              .putInt(frame.length)
              .put((byte) entry.purpose().ordinal())
              .put((byte) entry.group().ordinal())
              .put((byte) entry.source().ordinal())
              .put((byte) entry.decisionClass().ordinal())
              .put((byte) entry.measured());
          rows++;
          bytes += frame.length;
          CohortWriter.this.bytes.addAndGet(frame.length);
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

    private FileChannel open(Path path) throws IOException {
      return FileChannel.open(
          path,
          StandardOpenOption.CREATE,
          StandardOpenOption.TRUNCATE_EXISTING,
          StandardOpenOption.WRITE);
    }

    private void flush(FileChannel channel, ByteBuffer buffer) throws IOException {
      buffer.flip();
      writeFully(channel, buffer);
      buffer.clear();
    }

    private void writeFully(FileChannel channel, ByteBuffer source) throws IOException {
      while (source.hasRemaining()) {
        channel.write(source);
      }
    }
  }
}
