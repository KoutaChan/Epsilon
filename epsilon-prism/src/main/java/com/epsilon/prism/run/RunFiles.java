package com.epsilon.prism.run;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.SeekableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * run のディレクトリのファイル（train の系譜 {@link Lineage}）: reports/ の 1 行 1 記録の jsonl、途中で止まっても壊れない
 * 書き込み（config.json・state.json・当てはめの記録）、終わった反復の途中の網（work/&lt;t&gt;）の後始末。再開で同じ反復をやり直したときに書き終えた
 * 記録を書き直さないよう、記録を番号で引ける（{@link #recorded}）。
 */
public final class RunFiles {
  private static final Logger LOG = LoggerFactory.getLogger(RunFiles.class);

  /** 反復ごとの記録の名前。 */
  public static final String ITERATIONS = "iterations.jsonl";

  /** バージョンの評価の記録の名前。 */
  public static final String EVALUATIONS = "evaluations.jsonl";

  /** 公開したバージョンを並べるディレクトリの名前（root/snapshots/&lt;5 桁のバージョン番号&gt;）。 */
  public static final String SNAPSHOTS = "snapshots";

  /** jsonl は 1 行 1 記録。 */
  private static final Gson LINES = new GsonBuilder().serializeSpecialFloatingPointValues().create();

  private final Path reports;

  /**
   * @param root run のディレクトリ（記録は root/reports）
   */
  public RunFiles(Path root) {
    reports = root.resolve("reports");
  }

  /** run のバージョン version のディレクトリ。 */
  public static Path snapshot(Path root, int version) {
    return root.resolve(SNAPSHOTS).resolve(String.format("%05d", version));
  }

  /** run の最新のバージョン: root/snapshots の、{@value Snapshot#POLICY} まで書き終わったバージョンのうち番号が最も大きいもの。 */
  public static Path latestSnapshot(Path root) throws IOException {
    Path snapshots = root.resolve(SNAPSHOTS);
    if (!Files.isDirectory(snapshots)) {
      throw new IOException("Not a snapshot or a run directory: " + root);
    }
    try (Stream<Path> versions = Files.list(snapshots)) {
      return versions
          .filter(path -> path.getFileName().toString().matches("\\d+"))
          .filter(path -> Files.isRegularFile(path.resolve(Snapshot.POLICY)))
          .max(Comparator.comparingInt(path -> Integer.parseInt(path.getFileName().toString())))
          .orElseThrow(() -> new IOException("No snapshot in " + snapshots));
    }
  }

  /**
   * reports/name に鍵 key の整数が value の記録があれば true（ファイルがなければ false）。書きかけで止まった行は読み飛ばす。再開で同じ反復の記録と
   * 評価を書き直さないために使う。
   */
  public boolean recorded(String name, String key, int value) throws IOException {
    Path path = reports.resolve(name);
    if (!Files.exists(path)) return false;
    for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
      if (line.isBlank()) continue;
      try {
        JsonElement number = JsonParser.parseString(line).getAsJsonObject().get(key);
        if (number != null && !number.isJsonNull() && number.getAsInt() == value) return true;
      } catch (RuntimeException truncated) {
        // 書きかけで止まった行
      }
    }
    return false;
  }

  /** reports/name に記録を 1 行足す。前の行が改行で終わっていなければ（書きかけで止まった行）、改行を足してから書く。 */
  public void append(String name, Map<String, Object> record) throws IOException {
    Files.createDirectories(reports);
    Path path = reports.resolve(name);
    String line = LINES.toJson(record) + "\n";
    if (!endsWithNewline(path)) line = "\n" + line;
    Files.writeString(
        path,
        line,
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE,
        StandardOpenOption.APPEND);
  }

  /** ファイルがないか空か、最後のバイトが改行なら true。 */
  private static boolean endsWithNewline(Path path) throws IOException {
    if (!Files.exists(path)) return true;
    try (SeekableByteChannel channel = Files.newByteChannel(path)) {
      long size = channel.size();
      if (size == 0) return true;
      ByteBuffer last = ByteBuffer.allocate(1);
      channel.position(size - 1).read(last);
      return last.get(0) == '\n';
    }
  }

  /** path.tmp に書いてから path に差し替える（途中で止まっても前の中身か新しい中身が残る）。 */
  public static void writeAtomically(Path path, String content) throws IOException {
    Files.createDirectories(path.getParent());
    Path temporary = path.resolveSibling(path.getFileName() + ".tmp");
    Files.writeString(temporary, content, StandardCharsets.UTF_8);
    Files.move(
        temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
  }

  /**
   * 終わった反復の work/&lt;c&gt;（c &lt; next）を消す。反復の終わりで消す前に止まったときの残りで、再開では使わない。work/&lt;next&gt; は続きに
   * 使うので残す。
   *
   * @param work 途中の網の置き場（run の work）
   */
  public static void removeFinishedWork(Path work, int next) throws IOException {
    if (!Files.isDirectory(work)) return;
    List<Path> entries;
    try (Stream<Path> listed = Files.list(work)) {
      entries = listed.toList();
    }
    for (Path entry : entries) {
      int iteration;
      try {
        iteration = Integer.parseInt(entry.getFileName().toString());
      } catch (NumberFormatException other) {
        continue;
      }
      if (iteration < next) {
        Snapshot.deleteTree(entry);
        LOG.info("removed the leftover work of finished iteration {}", iteration);
      }
    }
  }
}
