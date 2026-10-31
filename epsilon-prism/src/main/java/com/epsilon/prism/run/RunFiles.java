package com.epsilon.prism.run;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import java.io.IOException;
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
 * 記録を書き直さないよう、最後の記録の鍵を読める（{@link #lastRecorded}）。
 */
public final class RunFiles {
  private static final Logger LOG = LoggerFactory.getLogger(RunFiles.class);

  /** 反復ごとの記録の名前。 */
  public static final String ITERATIONS = "iterations.jsonl";

  /** 版の評価の記録の名前。 */
  public static final String EVALUATIONS = "evaluations.jsonl";

  /** 公開した版を並べるディレクトリの名前（root/snapshots/&lt;5 桁の版の番号&gt;）。 */
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

  /** run の版 version のディレクトリ。 */
  public static Path snapshot(Path root, int version) {
    return root.resolve(SNAPSHOTS).resolve(String.format("%05d", version));
  }

  /** run の最新の版: root/snapshots の、{@value Snapshot#POLICY} まで書き終わった版のうち番号が最も大きいもの。 */
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

  /** reports/name の最後の記録の鍵 key の整数（ファイルか記録か鍵がなければ 0）。再開で同じ反復の記録と評価を書き直さないために使う。 */
  public int lastRecorded(String name, String key) throws IOException {
    Path path = reports.resolve(name);
    if (!Files.exists(path)) return 0;
    List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
    for (int i = lines.size() - 1; i >= 0; i--) {
      String line = lines.get(i).trim();
      if (line.isEmpty()) continue;
      JsonElement value;
      try {
        value = JsonParser.parseString(line).getAsJsonObject().get(key);
      } catch (RuntimeException truncated) {
        // 書きかけで止まった行は読み飛ばす
        continue;
      }
      return value == null || value.isJsonNull() ? 0 : value.getAsInt();
    }
    return 0;
  }

  /** reports/name に記録を 1 行足す。 */
  public void append(String name, Map<String, Object> record) throws IOException {
    Files.createDirectories(reports);
    Files.writeString(
        reports.resolve(name),
        LINES.toJson(record) + "\n",
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE,
        StandardOpenOption.APPEND);
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
