package com.epsilon.agent.run;

import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 反復の報告（reports/iterations.jsonl の 1 行）。使った設定は、行の番号だけを持つ（設定の中身は settings.jsonl のその行にあり、行は
 * 書き換えない）。
 *
 * @param iteration 反復の番号
 * @param version 反復が公開したバージョン（反復の番号 + 1）
 * @param settingsRow 反復に固定した設定の行
 * @param started 反復を初めて固定した時刻
 * @param seconds 段にかかった時間の合計（秒）
 * @param steps 段の名前ごとの報告（宣言の順）
 * @param carry 次の反復へ持ち越す値（state に入れたもの）
 * @param resources 反復を閉じたときの JVM のリソースの使用量
 * @param addedState この反復の間に、jar が初期値で state に足したフィールド（無ければ空）
 */
public record IterationReport(
    int iteration,
    int version,
    int settingsRow,
    Instant started,
    double seconds,
    Map<String, StepReport> steps,
    Record carry,
    Resources resources,
    List<String> addedState) {

  private static final String ORIGIN = "iteration report";

  public IterationReport {
    Objects.requireNonNull(started, "started");
    steps = Collections.unmodifiableMap(new LinkedHashMap<>(steps));
    Objects.requireNonNull(carry, "carry");
    Objects.requireNonNull(resources, "resources");
    addedState = List.copyOf(addedState);
  }

  /**
   * JVM のリソースの使用量。
   *
   * @param rssMiB 常駐メモリ（RSS）の大きさ（MiB。読めなければ 0）
   * @param heapUsedMiB ヒープの使用量（MiB）
   * @param vramMiB GPU ごとの使用量（MiB、デバイスの順。GPU を使わなければ空）
   */
  public record Resources(long rssMiB, long heapUsedMiB, List<Long> vramMiB) {

    public Resources {
      vramMiB = List.copyOf(vramMiB);
    }

    /** 今の JVM のリソースの使用量を測る。 */
    public static Resources measure(List<Long> vramMiB) {
      long heap = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed() >> 20;
      return new Resources(residentMiB(), heap, vramMiB);
    }

    /** /proc/self/status の VmRSS（MiB）。読めなければ 0。 */
    private static long residentMiB() {
      try {
        for (String line : Files.readAllLines(Path.of("/proc/self/status"), StandardCharsets.US_ASCII)) {
          if (line.startsWith("VmRSS:")) {
            return Long.parseLong(line.substring("VmRSS:".length()).replace("kB", "").strip()) >> 10;
          }
        }
      } catch (IOException | NumberFormatException unavailable) {
        return 0;
      }
      return 0;
    }
  }

  /** 報告の木（1 行の JSON に書く）。 */
  public Node.Table tree() {
    Map<String, Node> entries = new LinkedHashMap<>();
    entries.put("iteration", numeral(iteration));
    entries.put("version", numeral(version));
    entries.put("settingsRow", numeral(settingsRow));
    entries.put("started", new Node.Text(started.toString(), ORIGIN));
    entries.put("seconds", new Node.Numeral(Double.toString(seconds), ORIGIN));
    Map<String, Node> stepTrees = new LinkedHashMap<>();
    steps.forEach((name, report) -> stepTrees.put(name, report.tree()));
    entries.put("steps", new Node.Table(stepTrees, ORIGIN));
    entries.put("carry", RecordFile.tree(carry));
    entries.put("resources", RecordFile.tree(resources));
    if (!addedState.isEmpty()) {
      List<Node> keys = addedState.stream().map(key -> (Node) new Node.Text(key, ORIGIN)).toList();
      entries.put("addedState", new Node.Array(keys, ORIGIN));
    }
    return new Node.Table(entries, ORIGIN);
  }

  /**
   * 報告の 1 行を読む。
   *
   * @param metrics 段の名前ごとの、報告の中身の型
   * @param carry 持ち越しの型
   * @throws SettingsException フィールドが欠けているか、知らないフィールドや段があるか、値が型に合わないとき
   */
  public static IterationReport of(
      Node.Table line, Map<String, Class<? extends Record>> metrics, Class<? extends Record> carry) {
    List<String> problems = new ArrayList<>();
    Map<String, Node> head = new LinkedHashMap<>(line.entries());
    Node steps = head.remove("steps");
    Node carried = head.remove("carry");
    Node resources = head.remove("resources");
    if (!(steps instanceof Node.Table stepTable)) {
      throw new SettingsException(List.of("The iteration report at " + line.origin() + " has no steps table"));
    }
    if (!(carried instanceof Node.Table carryTable) || !(resources instanceof Node.Table resourceTable)) {
      throw new SettingsException(List.of("The iteration report at " + line.origin() + " needs carry and resources"));
    }
    Head read = RecordFile.read(Head.class, new Node.Table(head, line.origin())).value();
    Map<String, StepReport> reports = new LinkedHashMap<>();
    for (Map.Entry<String, Node> entry : stepTable.entries().entrySet()) {
      Class<? extends Record> type = metrics.get(entry.getKey());
      if (type == null || !(entry.getValue() instanceof Node.Table stepTree)) {
        problems.add("Unknown step " + entry.getKey() + " in the iteration report at " + line.origin());
        continue;
      }
      reports.put(entry.getKey(), StepReport.of(stepTree, type));
    }
    if (!problems.isEmpty()) {
      throw new SettingsException(problems);
    }
    return new IterationReport(
        read.iteration(),
        read.version(),
        read.settingsRow(),
        read.started(),
        read.seconds(),
        reports,
        RecordFile.read(carry, carryTable).value(),
        RecordFile.read(Resources.class, resourceTable).value(),
        read.addedState() == null ? List.of() : read.addedState());
  }

  /** 報告のファイルの全ての行を読む。 */
  public static List<IterationReport> read(
      Path file, Map<String, Class<? extends Record>> metrics, Class<? extends Record> carry) throws IOException {
    List<String> lines = DurableFiles.lines(file);
    List<IterationReport> reports = new ArrayList<>(lines.size());
    for (int index = 0; index < lines.size(); index++) {
      reports.add(of(Json.read(lines.get(index), file + ":" + (index + 1)), metrics, carry));
    }
    return List.copyOf(reports);
  }

  /** 段・持ち越し・リソースを除いたフィールド。 */
  private record Head(
      int iteration,
      int version,
      int settingsRow,
      Instant started,
      double seconds,
      @Key(doc = "jar が state に足したフィールド", presence = Presence.ABSENT) List<String> addedState) {}

  private static Node numeral(long value) {
    return new Node.Numeral(Long.toString(value), ORIGIN);
  }
}
