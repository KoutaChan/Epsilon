package com.epsilon.prism.grp;

import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.TrainLock;
import com.epsilon.prism.rows.CohortWriter;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Φ の切り替えの記録 grp/phi.jsonl と、モードの切り替えの記録 grp/mode.jsonl を読み書きする。どちらも追記だけのファイルで、phi.jsonl は
 * 1 行に 1 つの {@link PhiEntry}、mode.jsonl は 1 行に 1 つの {@link ModeLine} を書く。反復 t の Φ は、{@link #effectiveAt} で選んだ
 * 項目に {@link ModeLine#effectiveAt} のモードを当てたもので、時刻には依らない。
 *
 * <p>2 つのファイルは同じロックの中で書き、学習の JVM も同じロックの中で続けて読む。追記はファイルのロックの中で行い、効き始めの反復が
 * {@link #firstOpenIteration} より前なら拒む（走っている反復の過去は書き換えない）。
 *
 * <p>学習の JVM が動いているかは、run の学習のロック（{@link TrainLock}）で決める。プロセスが死ねば OS がロックを外す。改行で終わっていない
 * 最後の行は書きかけとみなして読まず、次の追記で切り捨てる。refits.jsonl（{@link RefitRecord}）も同じ書き方をする。
 *
 * <p>ファイルのロックは JVM ごとに効くので、同じ JVM の中の追記はクラスで {@code synchronized} して 1 つずつ行う。
 */
public final class PhiHistory {
  private PhiHistory() {}

  /** phi.jsonl と mode.jsonl のロックの名前。 */
  private static final String PHI_LOCK = "phi";

  /** jsonl の 1 行を読み書きする Gson。NaN と無限大もそのまま書く。 */
  static final Gson LINES = new GsonBuilder().serializeSpecialFloatingPointValues().create();

  /**
   * phi.jsonl と mode.jsonl の全ての行。どちらも書いた順に並ぶ。
   *
   * @param entries phi.jsonl の行
   * @param modes mode.jsonl の行
   */
  record Lines(List<PhiEntry> entries, List<ModeLine> modes) {}

  /**
   * phi.jsonl の全ての行を、書いた順に読む。
   *
   * @return 読んだ項目。ファイルが無ければ空
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

  /**
   * mode.jsonl の全ての行を、書いた順に読む。
   *
   * @return 読んだモードの行。ファイルが無ければ空
   * @throws IOException 読めないか、行が壊れているとき
   */
  public static List<ModeLine> readModes(GrpFiles files) throws IOException {
    List<ModeLine> modes = new ArrayList<>();
    List<String> lines = lines(files.modeHistory());
    for (int i = 0; i < lines.size(); i++) {
      try {
        modes.add(LINES.fromJson(lines.get(i), ModeLine.class));
      } catch (RuntimeException broken) {
        throw new IOException("Broken line " + (i + 1) + " of " + files.modeHistory(), broken);
      }
    }
    return modes;
  }

  /**
   * ロックの中で、phi.jsonl と mode.jsonl の全ての行を続けて読む。学習の JVM が反復の始めに呼ぶ。ロックの中で読むので、追記する側が
   * 効き始めの反復を確かめている間とは重ならない。grp のディレクトリが無ければ、ロックを取らずに空を返す。
   */
  static Lines readLocked(GrpFiles files) throws IOException {
    if (!Files.isDirectory(files.directory())) return new Lines(List.of(), List.of());
    synchronized (PhiHistory.class) {
      try (Locked locked = lock(files, PHI_LOCK)) {
        return new Lines(read(files), readModes(files));
      }
    }
  }

  /**
   * 反復 t で使った Φ の前に使っていた、別の値の Φ を返す。反復 t − 1 から 1 つずつさかのぼり、反復 t の項目と
   * {@link PhiEntry#sameValue} で値が違う最初の項目を返す。反復 t に効く項目が無いとき、さかのぼる途中で効く項目が無くなったとき、
   * 見つからないときは空。
   */
  public static Optional<PhiEntry> previousAt(List<PhiEntry> entries, int t) {
    PhiEntry used = effectiveAt(entries, t).orElse(null);
    if (used == null) return Optional.empty();
    for (int earlier = t - 1; earlier >= 1; earlier--) {
      PhiEntry before = effectiveAt(entries, earlier).orElse(null);
      if (before == null) return Optional.empty();
      if (!before.sameValue(used)) return Optional.of(before);
    }
    return Optional.empty();
  }

  /** 反復 t に効く項目を返す。効き始めが t 以下の行のうち、効き始めが最も遅い行を選び、同じなら後に書いた行を選ぶ。無ければ空。 */
  public static Optional<PhiEntry> effectiveAt(List<PhiEntry> entries, int t) {
    PhiEntry effective = null;
    for (PhiEntry entry : entries)
      if (entry.effectiveFrom() <= t && (effective == null || entry.effectiveFrom() >= effective.effectiveFrom()))
        effective = entry;
    return Optional.ofNullable(effective);
  }

  /**
   * 自分の GRP の最初の項目 g000 を、phi.jsonl の 1 行目に書く。既にあれば読むだけで、設定と食い違えば拒む。g000 は反復 1 から効く。
   * 元の GRP から始めるときは、GRP と q のディレクトリの絶対パスとそれぞれの sha、PhiTable のパスとその sha を持つ。続けて、
   * 初めの LATE の行を書く（{@link #initializeModes} と同じ）。
   *
   * @return g000
   * @throws IOException 書けないか、既にある g000 が設定と食い違うとき
   */
  public static PhiEntry initialize(GrpFiles files, PhiSetup setup) throws IOException {
    synchronized (PhiHistory.class) {
      try (Locked locked = lock(files, PHI_LOCK)) {
        List<PhiEntry> entries = read(files);
        PhiEntry initial;
        if (entries.isEmpty()) {
          initial = initialEntry(setup);
          write(files.phiHistory(), LINES.toJson(initial));
        } else {
          initial = requireInitialMatches(files, entries.getFirst(), setup);
        }
        writeInitialMode(files, setup.selfGrp().late());
        return initial;
      }
    }
  }

  /**
   * LATE のモードを使い始める反復（{@link LateSettings#from}）が正で、mode.jsonl にまだ行が無ければ、その反復から効く LATE の行を書く。
   * 学習の JVM が run を開くときと、固定の GRP の run でモードの行を手で足す前に呼ぶ。こうして設定から作る行が、手で足す行より先に並ぶ。
   * 効き始めの反復の確かめ（{@link #firstOpenIteration}）はしない。
   *
   * @throws IOException 読めないか書けないとき
   */
  public static void initializeModes(GrpFiles files, LateSettings late) throws IOException {
    if (late.from() == 0) return;
    synchronized (PhiHistory.class) {
      try (Locked locked = lock(files, PHI_LOCK)) {
        writeInitialMode(files, late);
      }
    }
  }

  /** 初めの LATE の行を書く。呼び出し元がロックを持つこと。使い始める反復が 0 か、既に行があれば何もしない。 */
  private static void writeInitialMode(GrpFiles files, LateSettings late) throws IOException {
    if (late.from() == 0 || !readModes(files).isEmpty()) return;
    write(files.modeHistory(), LINES.toJson(ModeLine.late(late.cutoff(), late.from(), "phi.late.from = " + late.from())));
  }

  /**
   * モードの行を 1 行足す。ロックの中で効き始めの反復を確かめてから書き、fsync する。
   *
   * @return 足した行
   * @throws IllegalArgumentException 効き始めの反復が、既に始まった反復以前のとき（{@link #firstOpenIteration}）
   * @throws IOException 書けないとき
   */
  public static ModeLine appendMode(GrpFiles files, ModeLine line) throws IOException {
    synchronized (PhiHistory.class) {
      try (Locked locked = lock(files, PHI_LOCK)) {
        int open = firstOpenIteration(files);
        if (line.effectiveFrom() < open)
          throw new IllegalArgumentException(
              "Mode line " + line.mode() + " effectiveFrom " + line.effectiveFrom()
                  + " is not after the last started iteration (the first open iteration is " + open + ")");
        write(files.modeHistory(), LINES.toJson(line));
        return line;
      }
    }
  }

  /** 設定から g000 を作る。GRP から始めるときは、チェックポイント・表・q の sha を今のファイルから求める。 */
  static PhiEntry initialEntry(PhiSetup setup) throws IOException {
    PhiStartSettings start = setup.selfGrp().start();
    String reason = "phi start = " + start.kind();
    if (start instanceof PhiStartSettings.GrpCheckpoint checkpoint)
      return HandEndPhi.checkpointEntry(
          PhiEntry.INITIAL_ID, setup.variant(), checkpoint.grp(), table(checkpoint), checkpoint.outcomes(), 1, "", reason);
    return new PhiEntry(PhiEntry.INITIAL_ID, start.kind(), setup.variant(), "", "", "", "", "", "", 1, "", reason);
  }

  /** g000 で元の GRP に重ねる表のパス。表が無ければ空文字列。 */
  private static String table(PhiStartSettings.GrpCheckpoint checkpoint) {
    return checkpoint.table() == null ? "" : checkpoint.table().toString();
  }

  /** g000 の q のディレクトリの絶対パス。無ければ空文字列。 */
  private static String outcomes(PhiStartSettings.GrpCheckpoint checkpoint) {
    return checkpoint.outcomes() == null ? "" : checkpoint.outcomes().toAbsolutePath().normalize().toString();
  }

  /**
   * phi.jsonl の 1 行目が設定の g000 と合うかを確かめる。確かめるのは、id が g000 であることと、種類とルール（四人打ちか三人打ちか）。
   * GRP から始めるときは、元の GRP のディレクトリ・PhiTable・q のディレクトリも比べる。学習の JVM の起動と設定の確かめで、同じ判定を使う。
   *
   * @return 渡した 1 行目の項目
   * @throws IOException 食い違うとき
   */
  static PhiEntry requireInitialMatches(GrpFiles files, PhiEntry first, PhiSetup setup) throws IOException {
    PhiStartSettings start = setup.selfGrp().start();
    if (!first.id().equals(PhiEntry.INITIAL_ID)
        || first.kind() != start.kind()
        || first.variant() != setup.variant()
        || (start instanceof PhiStartSettings.GrpCheckpoint checkpoint
            && (!Path.of(first.path()).equals(checkpoint.grp().toAbsolutePath().normalize())
                || !first.table().equals(table(checkpoint))
                || !first.outcomes().equals(outcomes(checkpoint)))))
      throw new IOException(
          "The first entry of " + files.phiHistory() + " does not match the phi start " + start.kind()
              + " and the GRP settings: " + first);
    return first;
  }

  /**
   * 項目を 1 行足す。ロックの中で効き始めの反復を確かめてから書き、fsync する。
   *
   * @return 足した項目
   * @throws IllegalArgumentException 効き始めの反復が既に始まった反復以前のとき（{@link #firstOpenIteration}）、同じ id の行が別の Φ を
   *     持つとき、まだ g000 が無いとき
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
   * 学習の JVM が、まだ始めていない反復から効く行を足す。足すのは AUTO の切り替え、監視による戻し、開けなかった候補の代わりに前の項目を
   * 効かせ直す行。書くのは次の反復を始める前に止まっている学習の JVM だけなので、効き始めの反復の確かめ（{@link #firstOpenIteration}）は
   * しない。
   */
  static void appendUnstarted(GrpFiles files, PhiEntry entry) throws IOException {
    synchronized (PhiHistory.class) {
      try (Locked locked = lock(files, PHI_LOCK)) {
        appendChecked(files, entry);
      }
    }
  }

  /**
   * 新しい行を効かせられる最初の反復を返す。基本は state.json の次のコホート t（無ければ 1）。ただし反復 t が既に始まっていれば t + 1
   * を返す。始まっているとみなすのは、学習の JVM が動いているとき、反復 t の設定を固定したファイルがあるとき、コホート t が確定済みのとき。
   * コホートが確定済みなのは学習の途中で落ちた反復で、再開すると前の Φ で集めた行を使い回す。
   *
   * @throws IOException state.json が読めないとき
   */
  public static int firstOpenIteration(GrpFiles files) throws IOException {
    int next = nextIteration(files);
    boolean started =
        trainingRunning(files)
            || Files.exists(new RunDirectory(files.run()).iterationFile(next))
            || Files.exists(files.cohort(next).resolve(CohortWriter.MANIFEST));
    return next + (started ? 1 : 0);
  }

  /** state.json に書かれた、次に集めるコホートの番号。ファイルが無ければ 1。 */
  static int nextIteration(GrpFiles files) throws IOException {
    if (!Files.exists(files.state())) return 1;
    try {
      return JsonParser.parseString(Files.readString(files.state())).getAsJsonObject().get("next").getAsInt();
    } catch (RuntimeException broken) {
      throw new IOException("Unreadable " + files.state(), broken);
    }
  }

  /** 段とジョブを回す JVM が run を開いていれば true。run の学習のロックが取れないことで見分ける。 */
  static boolean trainingRunning(GrpFiles files) throws IOException {
    return TrainLock.running(new RunDirectory(files.run()).trainLock());
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

  /** ファイルの中の、改行で終わる行を返す（空行は除く）。改行で終わっていない最後の部分は書きかけとみなして読まない。ファイルが無ければ空。 */
  static List<String> lines(Path file) throws IOException {
    if (!Files.exists(file)) return List.of();
    String text = Files.readString(file, StandardCharsets.UTF_8);
    int end = text.lastIndexOf('\n');
    List<String> lines = new ArrayList<>();
    if (end < 0) return lines;
    for (String line : text.substring(0, end).split("\n")) if (!line.isBlank()) lines.add(line);
    return lines;
  }

  /** ファイルに 1 行足して fsync する。呼び出し元がロックを持つこと。書きかけの最後の行があれば、先に切り捨てる。 */
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

  /** grp/&lt;name&gt;.lock の排他ロックを取る。取れるまで待つ。 */
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
    return new Locked(channel);
  }

  /** 取ったロック。閉じると外れる。 */
  static final class Locked implements AutoCloseable {
    private final FileChannel channel;

    private Locked(FileChannel channel) {
      this.channel = channel;
    }

    @Override
    public void close() throws IOException {
      channel.close();
    }
  }
}
