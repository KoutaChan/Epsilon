package com.epsilon.prism.grp;

import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.PrismConfig;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Φ の切り替えの記録 grp/phi.jsonl（追記だけ、1 行 1 {@link PhiEntry}）。反復 t の Φ は {@link #effectiveAt} で決まり、時計に依らない。
 * 追記はファイルの錠の中で、effectiveFrom が {@link #firstOpenIteration} より前なら拒む（走っている反復の過去を書き換えない）。
 *
 * <p>学習の JVM が動いているかは、{@link PhiResolver} が開いているあいだ持つ錠 grp/{@value #TRAINING_LOCK}.lock で決める（プロセスが死ねば
 * カーネルが外す）。最後の改行のない行は書きかけとして読まず、次の追記で切り捨てる。refits.jsonl（{@link RefitRecord}）も同じ書き方。
 *
 * <p>ファイルの錠は JVM ごとなので、同じ JVM の中の追記はクラスの監視で 1 つずつにする。
 */
public final class PhiHistory {
  private PhiHistory() {}

  /** 学習の JVM が開いているあいだ持つ錠の名前（{@link GrpFiles#lock}）。 */
  static final String TRAINING_LOCK = "training";

  /** phi.jsonl の錠の名前。 */
  private static final String PHI_LOCK = "phi";

  /** 学習の JVM の印の錠を取るまで待つ上限と間隔（grp-switch などの別のプロセスの探り {@link #trainingRunning} は一瞬で外す）。 */
  static final Duration TRAINING_LOCK_WAIT = Duration.ofSeconds(30), TRAINING_LOCK_POLL = Duration.ofMillis(100);

  /** jsonl の 1 行（NaN と無限大はそのまま書く）。 */
  static final Gson LINES = new GsonBuilder().serializeSpecialFloatingPointValues().create();

  /** この JVM が持つ学習の錠のファイル（同じ JVM では錠を試さずに動いているとみなす。試すと POSIX の錠が外れる）。 */
  private static final Set<Path> HELD = ConcurrentHashMap.newKeySet();

  /**
   * phi.jsonl の全ての行（書いた順）。ファイルがなければ空。
   *
   * @throws IOException 読めないか、行が壊れているとき
   */
  public static List<PhiEntry> read(GrpFiles files) throws IOException {
    List<PhiEntry> entries = new ArrayList<>();
    List<String> lines = lines(files.phiHistory());
    for (int i = 0; i < lines.size(); i++) {
      try {
        entries.add(LINES.fromJson(lines.get(i), PhiEntry.class));
      } catch (RuntimeException broken) {
        throw new IOException("Broken line " + (i + 1) + " of " + files.phiHistory(), broken);
      }
    }
    return entries;
  }

  /** 錠の中で phi.jsonl の全ての行を読む（学習の JVM が反復の始めに読む。書き手の効き始めの確かめと重ならない）。 */
  static List<PhiEntry> readLocked(GrpFiles files) throws IOException {
    synchronized (PhiHistory.class) {
      try (Locked locked = lock(files, PHI_LOCK)) {
        return read(files);
      }
    }
  }

  /** 反復 t に効く項目（effectiveFrom ≤ t の行のうち effectiveFrom が最も大きい行、同じなら後の行。なければ空）。 */
  public static Optional<PhiEntry> effectiveAt(List<PhiEntry> entries, int t) {
    PhiEntry effective = null;
    for (PhiEntry entry : entries)
      if (entry.effectiveFrom() <= t && (effective == null || entry.effectiveFrom() >= effective.effectiveFrom()))
        effective = entry;
    return Optional.ofNullable(effective);
  }

  /**
   * SELF_GRP の最初の項目 g000 を phi.jsonl の 1 行目に書く（既にあれば読むだけで、設定と食い違えば拒む）。種類は phiStart、GRP_CHECKPOINT なら
   * grpCheckpoint の正確なディレクトリ（一番新しい子。以後は書いたパスだけを読む）・sha・phiTemperatures・phiCalibration を持ち、effectiveFrom は
   * 1（それまでの反復が同じ Φ で集めたとみなす。効き始めの決まりは掛けない）。
   *
   * @return g000
   * @throws IOException 書けないか、既にある g000 が設定と食い違うとき
   */
  public static PhiEntry initialize(GrpFiles files, PrismConfig config) throws IOException {
    PhiEntryKind kind = config.phiStart().kind();
    synchronized (PhiHistory.class) {
      try (Locked locked = lock(files, PHI_LOCK)) {
        List<PhiEntry> entries = read(files);
        if (!entries.isEmpty()) return requireInitialMatches(files, entries.getFirst(), config);
        PhiEntry initial;
        if (kind.scoreBased()) {
          initial = new PhiEntry(PhiEntry.INITIAL_ID, kind, "", "", "", "", 1, "", "phiStart = " + config.phiStart());
        } else {
          Path exact = EpsilonGrpCheckpointManager.resolveExisting(grpRoot(config));
          if (exact == null) throw new IOException("GRP checkpoint does not exist at " + grpRoot(config));
          exact = exact.toAbsolutePath().normalize();
          initial =
              new PhiEntry(
                  PhiEntry.INITIAL_ID,
                  kind,
                  exact.toString(),
                  EpsilonGrpCheckpointManager.checkpointSha256(exact),
                  config.phiTemperatures(),
                  config.phiCalibration(),
                  1,
                  "",
                  "phiStart = " + config.phiStart());
        }
        write(files.phiHistory(), LINES.toJson(initial));
        return initial;
      }
    }
  }

  /**
   * phi.jsonl の 1 行目が設定の g000（phiStart の種類、GRP なら grpCheckpoint の下の checkpoint・phiTemperatures・phiCalibration）と合うかを
   * 確かめる（学習の JVM の起動と check-config で同じ判定）。
   *
   * @return first
   * @throws IOException 食い違うとき
   */
  static PhiEntry requireInitialMatches(GrpFiles files, PhiEntry first, PrismConfig config) throws IOException {
    PhiEntryKind kind = config.phiStart().kind();
    if (!first.id().equals(PhiEntry.INITIAL_ID)
        || first.kind() != kind
        || (!kind.scoreBased()
            && (!Path.of(first.path()).startsWith(grpRoot(config))
                || !first.temperatures().equals(config.phiTemperatures())
                || !first.calibration().equals(config.phiCalibration()))))
      throw new IOException(
          "The first entry of " + files.phiHistory() + " does not match phiStart = " + config.phiStart()
              + " and the GRP settings: " + first);
    return first;
  }

  /**
   * 項目を 1 行足す（錠の中で effectiveFrom を確かめてから fsync する）。
   *
   * @return 足した項目
   * @throws IllegalArgumentException effectiveFrom が既に始まった反復以前のとき（{@link #firstOpenIteration}）、同じ id の行が別の Φ を持つとき、
   *     まだ g000 がないとき
   * @throws IOException 書けないとき
   */
  public static PhiEntry append(GrpFiles files, PhiEntry entry) throws IOException {
    synchronized (PhiHistory.class) {
      try (Locked locked = lock(files, PHI_LOCK)) {
        int open = firstOpenIteration(files);
        if (entry.effectiveFrom() < open)
          throw new IllegalArgumentException(
              "Φ entry " + entry.id() + " effectiveFrom " + entry.effectiveFrom()
                  + " is not after the last started iteration (the first open iteration is " + open + ")");
        appendChecked(files, entry);
        return entry;
      }
    }
  }

  /**
   * 学習の JVM が反復 t の始めに、反復 t から効く行を足す（AUTO の切り替えと、境目の点検で拒んだ候補の代わりに前の項目を効かせ直す行）。
   * 反復 t はまだ何も集めていないので効き始めの決まり（{@link #firstOpenIteration}）は掛けない。
   */
  static void appendAtIterationStart(GrpFiles files, PhiEntry entry) throws IOException {
    synchronized (PhiHistory.class) {
      try (Locked locked = lock(files, PHI_LOCK)) {
        appendChecked(files, entry);
      }
    }
  }

  /**
   * 新しく効かせられる最初の反復: state.json の次のコホート t（なければ 1）。学習の JVM が動いているか、コホート t が封印済み（当てはめの途中で
   * 落ちた反復。再開は前の Φ で集めた行を使い回す）なら、反復 t は始まっているので + 1。
   *
   * @throws IOException state.json が読めないとき
   */
  public static int firstOpenIteration(GrpFiles files) throws IOException {
    int next = nextIteration(files);
    boolean started = trainingRunning(files) || Files.exists(files.cohort(next).resolve(CohortWriter.MANIFEST));
    return next + (started ? 1 : 0);
  }

  /** state.json の次に集めるコホート（なければ 1）。 */
  static int nextIteration(GrpFiles files) throws IOException {
    if (!Files.exists(files.state())) return 1;
    try {
      return JsonParser.parseString(Files.readString(files.state())).getAsJsonObject().get("next").getAsInt();
    } catch (RuntimeException broken) {
      throw new IOException("Unreadable " + files.state(), broken);
    }
  }

  /** SELF_GRP の学習の JVM が run を開いているなら true（その錠が取れない）。 */
  static boolean trainingRunning(GrpFiles files) throws IOException {
    Path path = files.lock(TRAINING_LOCK).toAbsolutePath().normalize();
    if (HELD.contains(path)) return true;
    if (!Files.exists(path)) return false;
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.WRITE)) {
      FileLock lock = channel.tryLock();
      if (lock == null) return true;
      lock.release();
      return false;
    } catch (OverlappingFileLockException held) {
      return true;
    }
  }

  /**
   * 学習の JVM の印の錠を取る（{@link PhiResolver} が開いているあいだ持つ）。grp-switch などの別のプロセスの探り（{@link #trainingRunning}）が一瞬
   * 持っていることがあるので、{@link #TRAINING_LOCK_POLL} ごとに {@link #TRAINING_LOCK_WAIT} まで試す（学習の JVM が 1 つだけなのは走らせ役の
   * 錠が守る）。
   *
   * @throws IOException 待っても取れないとき（ほかの学習の JVM が同じ run を開いている）
   */
  static Locked holdTraining(GrpFiles files) throws IOException {
    Path path = files.lock(TRAINING_LOCK).toAbsolutePath().normalize();
    if (!HELD.add(path)) throw new IOException("Another training JVM holds " + path);
    try {
      Files.createDirectories(path.getParent());
      FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
      try {
        long deadline = System.nanoTime() + TRAINING_LOCK_WAIT.toNanos();
        while (true) {
          FileLock lock;
          try {
            lock = channel.tryLock();
          } catch (OverlappingFileLockException probing) {
            lock = null;
          }
          if (lock != null) return new Locked(channel, () -> HELD.remove(path));
          if (System.nanoTime() >= deadline) throw new IOException("Another training JVM holds " + path);
          Thread.sleep(TRAINING_LOCK_POLL.toMillis());
        }
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        channel.close();
        throw new IOException("Interrupted while waiting for " + path, interrupted);
      } catch (IOException | RuntimeException failure) {
        channel.close();
        throw failure;
      }
    } catch (IOException | RuntimeException failure) {
      HELD.remove(path);
      throw failure;
    }
  }

  private static void appendChecked(GrpFiles files, PhiEntry entry) throws IOException {
    List<PhiEntry> entries = read(files);
    if (entries.isEmpty())
      throw new IllegalArgumentException("No g000 in " + files.phiHistory() + " yet: the training JVM writes it first");
    for (PhiEntry existing : entries)
      if (existing.id().equals(entry.id()) && !existing.sameValue(entry))
        throw new IllegalArgumentException("Φ entry id " + entry.id() + " already names another Φ: " + existing);
    write(files.phiHistory(), LINES.toJson(entry));
  }

  /** file の改行で終わる行（空行を除く）。最後の改行のない部分は書きかけとして読まない。ファイルがなければ空。 */
  static List<String> lines(Path file) throws IOException {
    if (!Files.exists(file)) return List.of();
    String text = Files.readString(file, StandardCharsets.UTF_8);
    int end = text.lastIndexOf('\n');
    List<String> lines = new ArrayList<>();
    if (end < 0) return lines;
    for (String line : text.substring(0, end).split("\n")) if (!line.isBlank()) lines.add(line);
    return lines;
  }

  /** file に 1 行足して fsync する（呼び手が錠を持つ）。書きかけの最後の行があれば先に切り捨てる。 */
  static void write(Path file, String line) throws IOException {
    Files.createDirectories(file.getParent());
    try (FileChannel channel =
        FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
      long size = channel.size();
      long keep = size;
      if (size > 0) {
        byte[] bytes = Files.readAllBytes(file);
        int last = bytes.length - 1;
        while (last >= 0 && bytes[last] != '\n') last--;
        keep = last + 1;
      }
      if (keep < size) channel.truncate(keep);
      ByteBuffer buffer = ByteBuffer.wrap((line + "\n").getBytes(StandardCharsets.UTF_8));
      long position = keep;
      while (buffer.hasRemaining()) position += channel.write(buffer, position);
      channel.force(true);
    }
  }

  /** grp/&lt;name&gt;.lock の排他の錠（待つ）。 */
  static Locked lock(GrpFiles files, String name) throws IOException {
    Path path = files.lock(name);
    Files.createDirectories(path.getParent());
    FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    try {
      channel.lock();
    } catch (IOException | RuntimeException failure) {
      channel.close();
      throw failure;
    }
    return new Locked(channel, () -> {});
  }

  private static Path grpRoot(PrismConfig config) {
    return Path.of(config.grpCheckpoint()).resolve("grp").toAbsolutePath().normalize();
  }

  /** 持っている錠（閉じると外す）。 */
  static final class Locked implements AutoCloseable {
    private final FileChannel channel;
    private final Runnable released;

    private Locked(FileChannel channel, Runnable released) {
      this.channel = channel;
      this.released = released;
    }

    @Override
    public void close() throws IOException {
      try {
        channel.close();
      } finally {
        released.run();
      }
    }
  }
}
