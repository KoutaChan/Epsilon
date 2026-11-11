package com.epsilon.agent.run;

import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 段の報告。段は完了マーカーと一緒にこれを保存し、使い回すときは保存したものを読む（秒・起動の行・jar は回したときの値のまま）。
 *
 * <p>JSON では、状態・秒・起動の行・jar に続けて、中身の record のフィールドを同じ階層に並べる。中身のフィールドが 1 つも無ければ、
 * 中身は null として読む。
 *
 * @param status 状態
 * @param seconds 段を回した時間（秒）
 * @param launchRow 段を回した JVM が起動のときに読んだ設定の行
 * @param jar 段を回した jar の ID
 * @param metrics 段ごとの数値。無ければ null（SKIPPED では必ず null）
 */
public record StepReport(Status status, double seconds, int launchRow, String jar, Record metrics) {

  /** 段の報告の状態。 */
  public enum Status {
    /** この反復の中で回した。 */
    RAN,
    /** 完了マーカーがあったので回さず、保存した報告を使った。 */
    REUSED,
    /** 反復に固定した設定では、この段を行わない。 */
    SKIPPED
  }

  /** 状態・秒・起動の行・jar だけの record。中身のフィールドと分けて読むのに使う。 */
  private record Head(Status status, double seconds, int launchRow, String jar) {}

  private static final String ORIGIN = "step report";

  public StepReport {
    Objects.requireNonNull(status, "status");
    Objects.requireNonNull(jar, "jar");
    if (!(seconds >= 0) || Double.isInfinite(seconds)) {
      throw new IllegalArgumentException("Step seconds must be finite and not negative: " + seconds);
    }
    if (status == Status.SKIPPED && metrics != null) {
      throw new IllegalArgumentException("A skipped step reports no metrics");
    }
  }

  /**
   * 回した段の報告を作る。
   *
   * @param startedNanos 段を始めたときの {@link System#nanoTime}
   * @param metrics 段ごとの数値（無ければ null）
   */
  public static StepReport ran(Iteration iteration, long startedNanos, Record metrics) {
    return new StepReport(
        Status.RAN, (System.nanoTime() - startedNanos) / 1e9, iteration.launchRow(), iteration.jar(), metrics);
  }

  /** 行わない段の報告。 */
  public static StepReport skipped(Iteration iteration) {
    return new StepReport(Status.SKIPPED, 0, iteration.launchRow(), iteration.jar(), null);
  }

  /** 完了マーカーと一緒に保存した報告を、使い回すときの報告。 */
  public StepReport reused() {
    return new StepReport(Status.REUSED, seconds, launchRow, jar, metrics);
  }

  /**
   * 報告を木にしたもの。
   *
   * @throws IllegalStateException 中身の record が、状態・秒・起動の行・jar と同じ名前のコンポーネントを持つ
   */
  public Node.Table tree() {
    Node.Table head = RecordFile.tree(new Head(status, seconds, launchRow, jar));
    Map<String, Node> entries = new LinkedHashMap<>(head.entries());
    if (metrics != null) {
      for (Map.Entry<String, Node> entry : RecordFile.tree(metrics).entries().entrySet()) {
        if (entries.containsKey(entry.getKey())) {
          throw new IllegalStateException(
              "The step metrics " + metrics.getClass().getName() + " have a component named " + entry.getKey());
        }
        entries.put(entry.getKey(), entry.getValue());
      }
    }
    return new Node.Table(entries, ORIGIN);
  }

  /**
   * 報告の木を読む。
   *
   * @param metrics 中身の record の型
   * @throws SettingsException フィールドが欠けているか、知らないフィールドがあるか、値が型に合わないか、秒が負か、行わない段に中身の
   *     フィールドがある
   */
  public static StepReport of(Node.Table tree, Class<? extends Record> metrics) {
    Map<String, Node> head = new LinkedHashMap<>();
    Map<String, Node> rest = new LinkedHashMap<>();
    for (Map.Entry<String, Node> entry : tree.entries().entrySet()) {
      boolean headField = entry.getKey().equals("status") || entry.getKey().equals("seconds")
          || entry.getKey().equals("launchRow") || entry.getKey().equals("jar");
      (headField ? head : rest).put(entry.getKey(), entry.getValue());
    }
    Head read = RecordFile.read(Head.class, new Node.Table(head, tree.origin())).value();
    Record value = rest.isEmpty() ? null : RecordFile.read(metrics, new Node.Table(rest, tree.origin())).value();
    try {
      return new StepReport(read.status(), read.seconds(), read.launchRow(), read.jar(), value);
    } catch (IllegalArgumentException invalid) {
      throw new SettingsException(List.of("Invalid step report at " + tree.origin() + ": " + invalid.getMessage()));
    }
  }

  /** 報告をファイルに書く（段の出力の中に、完了マーカーを置く前に書く）。 */
  public void write(Path file) throws IOException {
    DurableFiles.write(file, Json.write(tree()) + "\n");
  }

  /** 完了マーカーと一緒に保存した報告を読む。 */
  public static StepReport read(Path file, Class<? extends Record> metrics) throws IOException {
    return of(Json.read(Files.readString(file, StandardCharsets.UTF_8), file.toString()), metrics);
  }
}
