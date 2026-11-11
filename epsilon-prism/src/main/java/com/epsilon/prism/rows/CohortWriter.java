package com.epsilon.prism.rows;

import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
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
import java.util.Objects;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 1 コホートの行（{@link Row}）を、ライターごとのシャードに追記する。シャードは run の rows の下に置く。圧縮はライターのスレッドで行うので、
 * 収集のスレッドを止めない。行のフレームの形は {@link RowCodec} が決める。シャードと同じ名前のインデックスのファイル（{@link #index}）には、
 * 行の順に {@link #INDEX_ENTRY} バイトずつインデックスを並べる。これで学習側は、シャードの本体を読まずに行を選べる。
 * 行は届いた順に書くので、並びは収集のたびに変わる。中身は変わらず、学習は種で並べ替えてから使う。
 *
 * <p>マニフェスト（{@value #MANIFEST}）には、行数、バイト数、シャードの一覧のほかに、帳簿の列の数（全ての行で同じ）と記録の
 * フィンガープリント（{@link Fingerprints}）を書く。読む側は {@link #readManifest} で、O のフィンガープリントからコホートのルールを決め、
 * フィンガープリントをそのルールの今の形式と照合する。合わなければ止める。形式のバージョンは持たず、古い形式を読み分けることもしない。
 *
 * <p>{@link #seal} は全てのシャードをディスクに同期してから、マニフェストをアトミックに置く。その後はこのディレクトリを変えない。確定せずに
 * 閉じたとき（収集が途中で失敗したとき）はマニフェストを置かない。そのコホートは再開のときに書きかけとして消し、集め直す。
 *
 * <p>書いた行は、ライターのスレッドから、シャードごとに書いた順に {@link Tap} へ渡す。行を読み直さずに集計したい側のためにある。
 */
public final class CohortWriter implements AutoCloseable {

  /** マニフェストのファイル名。 */
  public static final String MANIFEST = "rows.json";

  /**
   * 1 行分のインデックスのバイト数。並びは [i32 フレームの長さ][u8 purpose][u8 source][u8 class][u8 measured]（{@link RowCodec.Index}）。
   */
  public static final int INDEX_ENTRY = 8;

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  /**
   * run の中で、確定したコホートの行を置くディレクトリを返す（{@link RunDirectory#rows}）。
   *
   * @param run run のディレクトリ
   * @param cohort コホートの番号
   * @return 行のディレクトリ
   */
  public static Path directory(Path run, int cohort) {
    return new RunDirectory(run).rows(cohort);
  }

  /**
   * シャードのインデックスのファイル（shard-00.rows → shard-00.idx）を返す。
   *
   * @param shard シャードのファイル
   * @return インデックスのファイル
   */
  public static Path index(Path shard) {
    String name = shard.getFileName().toString();
    return shard.resolveSibling(name.substring(0, name.lastIndexOf('.')) + ".idx");
  }

  /** 書いた行の受け取り先。 */
  public interface Tap {
    /** 何もしない受け取り先。 */
    Tap NONE = (shard, row, index) -> {};

    /**
     * シャードに行を書いたことを知らせる。そのシャードのライターのスレッドから、シャードの中で書いた順に呼ぶ。ここで投げた例外は、
     * ライターの失敗として扱う。
     *
     * @param shard シャードの番号
     * @param row 書いた行
     * @param index 行のインデックス（.idx に書いたもの）
     */
    void written(int shard, Row row, RowCodec.Index index);
  }

  /**
   * マニフェストに書く、シャード 1 つ分の情報。
   *
   * @param file シャードのファイル名
   * @param rows 行数
   * @param bytes バイト数
   */
  public record Shard(String file, long rows, long bytes) {}

  /**
   * 記録のフィンガープリント。
   *
   * @param observation 記録 O のフィンガープリント（{@link ObservationSchema#fingerprint}）
   * @param outcome 局末の記録のフィンガープリント（{@link OutcomeCodec#fingerprint}）
   * @param oracle Oracle の記録のフィンガープリント（{@link OracleCodec#fingerprint}）
   */
  public record Fingerprints(String observation, String outcome, String oracle) {

    /** このビルドが書く、そのルールでの記録のフィンガープリント。 */
    public static Fingerprints of(Variant variant) {
      return new Fingerprints(
          ObservationSchema.of(variant).fingerprint(),
          OutcomeCodec.of(variant).fingerprint(),
          OracleCodec.of(variant).fingerprint());
    }
  }

  /**
   * コホートのマニフェスト。
   *
   * @param iteration 行を集めたコホートの番号
   * @param rows 全シャードの行数
   * @param bytes 全シャードのバイト数
   * @param ledgerFeatures 帳簿の列の数
   * @param fingerprints 記録のフィンガープリント
   * @param shards シャードの一覧
   */
  public record Manifest(
      int iteration,
      long rows,
      long bytes,
      int ledgerFeatures,
      Fingerprints fingerprints,
      List<Shard> shards) {

    /**
     * コホートの O のルールを返す。O のフィンガープリントから決める。
     *
     * @throws IllegalStateException どのルールの O のフィンガープリントとも合わないとき
     */
    public Variant variant() {
      return ObservationSchema.variantOf(fingerprints.observation())
          .orElseThrow(
              () -> new IllegalStateException("No variant has observation fingerprint " + fingerprints.observation()));
    }
  }

  private final Path directory;
  private final Variant variant;
  private final int iteration;
  private final int ledgerFeatures;
  private final Tap tap;
  private final List<Worker> workers = new ArrayList<>();
  private final AtomicInteger next = new AtomicInteger();
  private final AtomicLong backlog = new AtomicLong();
  private final AtomicLong bytes = new AtomicLong();
  private volatile Throwable failure;
  private boolean stopped;
  private boolean sealed;

  /**
   * ライターのスレッドを始める。
   *
   * @param directory コホートのディレクトリ
   * @param variant 行の O のルール
   * @param iteration コホートの番号
   * @param ledgerFeatures 帳簿の列の数（全ての行で同じ）
   * @param writers ライターのスレッドの数（シャードの数）
   * @param tap 書いた行の受け取り先
   * @throws IllegalArgumentException ライターの数か帳簿の列の数が範囲の外のとき
   * @throws IOException 既に確定しているか、ディレクトリを作れないとき
   */
  public CohortWriter(
      Path directory, Variant variant, int iteration, int ledgerFeatures, int writers, Tap tap)
      throws IOException {
    if (writers < 1) {
      throw new IllegalArgumentException("writer count must be positive: " + writers);
    }
    this.directory = directory;
    this.variant = Objects.requireNonNull(variant, "variant");
    this.iteration = iteration;
    this.ledgerFeatures = ledgerFeatures;
    this.tap = Objects.requireNonNull(tap, "tap");
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
   * 確定済みのコホートのマニフェストを読む。O のフィンガープリントからルールを決め、記録のフィンガープリントをそのルールの今の形式と照合する。
   *
   * @param directory コホートのディレクトリ
   * @return マニフェスト
   * @throws IOException 確定していないか、コホートのマニフェストでないか、どのルールの O のフィンガープリントとも合わないか、記録の
   *     フィンガープリントが合わないとき
   */
  public static Manifest readManifest(Path directory) throws IOException {
    Path path = directory.resolve(MANIFEST);
    if (!Files.exists(path)) {
      throw new IOException("Cohort is not sealed: " + directory);
    }
    Manifest manifest;
    try {
      manifest = RecordFile.read(Manifest.class, path).value();
    } catch (SettingsException broken) {
      throw new IOException("Not a cohort manifest: " + path + ": " + broken.getMessage(), broken);
    }
    Variant variant = ObservationSchema.variantOf(manifest.fingerprints().observation()).orElse(null);
    if (variant == null) {
      throw new IOException(
          "Cohort " + directory + " has observation fingerprint " + manifest.fingerprints().observation()
              + " of no variant");
    }
    Fingerprints current = Fingerprints.of(variant);
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
   * 行をライターに渡す。どのスレッドからでも呼べて、行を捨てない。
   *
   * @param row 行
   * @throws IllegalStateException ライターが失敗しているとき
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
   * 残りの行を書き終え、同期してコホートを確定する。
   *
   * @throws IOException ライターが失敗したか、確定の前に閉じたとき
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
            iteration, rows, bytes.get(), ledgerFeatures, Fingerprints.of(variant), shards));
    sealed = true;
  }

  /** ライターを止める。確定していなければマニフェストを置かないので、書きかけのコホートとして残る。 */
  @Override
  public synchronized void close() throws IOException {
    if (!stopped) {
      stop();
    }
  }

  /** ライターに残りの行を書かせてから止め、シャードの一覧を返す。 */
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

  /** マニフェストを同期してから、アトミックに置く。 */
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

  /** シャード 1 つのライター。行の直列化はこのスレッドで行う。 */
  private final class Worker {

    static final Object STOP = new Object();

    final LinkedBlockingQueue<Object> queue = new LinkedBlockingQueue<>();
    final int shard;
    final Path file;
    final Thread thread;
    final RowCodec codec = new RowCodec(ledgerFeatures, variant);
    long rows;
    long bytes;

    Worker(int index) {
      shard = index;
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
              .put((byte) entry.source().ordinal())
              .put((byte) entry.decisionClass().ordinal())
              .put((byte) entry.measured());
          tap.written(shard, row, entry);
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
