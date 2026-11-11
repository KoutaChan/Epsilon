package com.epsilon.agent.app;

import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.IterationReport;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.RunKind;
import com.epsilon.agent.run.StepKind;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.lang.reflect.RecordComponent;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.function.Function;

/**
 * run の報告。reports/iterations.jsonl から反復ごとの 1 行を、reports/evaluations.jsonl から卓ごとに最も新しいバージョンの結果を読み、
 * 列をそろえた表か 1 行の JSON にする。
 *
 * <p>run のファイルは読むだけで、ロックも取らない。止まっている run も、回っている run も同じに読む。jsonl の最後の、改行で終わって
 * いない部分は書きかけとして読まない。ほかに読めない行があれば、誤りにする。
 *
 * <p>反復の報告の行を run の種類の型（段の報告の中身と持ち越し）で読むのは、出す反復だけにする。出さない反復の行は、反復の番号と設定の
 * 行だけを読む。run は反復の区切りで jar を替えて続けられ、前の jar が書いた行は今の jar の型に合わないことがある。
 *
 * <p>反復の表には、どの run の種類にもある列（反復、バージョン、反復に固定した設定の行、反復と段ごとの秒）に続けて、run の種類の列を
 * 並べる。設定の行が前の反復と違う反復には印を付ける。卓の表には、卓の名前と測ったバージョンに続けて、run の種類の列を並べる。run の
 * 種類の列は {@link Layout} で決める。
 */
public final class RunReport {

  /** 表で、設定の行が前の反復と違う反復の、行の番号に付ける印。 */
  public static final String CHANGED = "*";

  private static final String ORIGIN = "report";

  /** 反復の表の、どの run の種類にもある列。この後に段ごとの秒と、run の種類の列が並ぶ。 */
  private static final List<String> ITERATION_HEADER = List.of("iteration", "version", "row", "seconds");

  /** JSON の反復のキーのうち、表の見出しに無いもの。 */
  private static final List<String> ITERATION_KEYS = List.of("settingsRow", "rowChanged", "steps");

  /** 卓の表の、どの run の種類にもある列。この後に run の種類の列が並ぶ。 */
  private static final List<String> TABLE_HEADER = List.of("table", "version");

  /**
   * 表と JSON の 1 つの値。
   *
   * @param value JSON に書く値。null なら、JSON にはキーごと書かず、表には {@code -} を書く
   * @param text 表に書く文字列
   */
  public record Cell(Node value, String text) {

    /** 値の無いセル。 */
    public static final Cell NONE = new Cell(null, "-");

    public Cell {
      Objects.requireNonNull(text, "text");
    }

    public static Cell integer(long value) {
      String literal = Long.toString(value);
      return new Cell(new Node.Numeral(literal, ORIGIN), literal);
    }

    /**
     * 小数。JSON には丸めずに書き、表には桁をそろえて書く。
     *
     * @param value 値。null か、有限でなければ {@link #NONE}
     * @param decimals 表に書く、小数点より下の桁の数
     */
    public static Cell decimal(Double value, int decimals) {
      if (value == null || !Double.isFinite(value)) {
        return NONE;
      }
      return new Cell(
          new Node.Numeral(Double.toString(value), ORIGIN), String.format(Locale.ROOT, "%." + decimals + "f", value));
    }

    public static Cell text(String value) {
      return new Cell(new Node.Text(value, ORIGIN), value);
    }
  }

  /**
   * run の種類が足す列。
   *
   * @param names 列の名前（表の見出しと JSON のキー）
   * @param cells 1 つの行から、名前と同じ順に値を作る。行が型に合わなければ {@link SettingsException} を投げる
   * @param <T> 行の型
   */
  public record Columns<T>(List<String> names, Function<? super T, List<Cell>> cells) {

    public Columns {
      names = List.copyOf(names);
      Objects.requireNonNull(cells, "cells");
    }

    /** @throws IllegalStateException 値の数が列の数と違う */
    List<Cell> of(T row) {
      List<Cell> values = cells.apply(row);
      if (values.size() != names.size()) {
        throw new IllegalStateException("The columns " + names + " got " + values.size() + " values");
      }
      return values;
    }
  }

  /**
   * run の種類の報告の形。
   *
   * @param steps 段の名前ごとの、段の報告の中身の型（宣言の順）
   * @param carry 反復の報告の持ち越しの型
   * @param iterations 反復の報告から作る列
   * @param tables ジョブの結果の行（evaluations.jsonl の 1 行。ジョブの名前・バージョン・卓と、結果のフィールド）から作る列
   */
  public record Layout(
      Map<String, Class<? extends Record>> steps,
      Class<? extends Record> carry,
      Columns<IterationReport> iterations,
      Columns<Node.Table> tables) {

    /** @throws IllegalArgumentException 列かキーの名前が重なる */
    public Layout {
      steps = Collections.unmodifiableMap(new LinkedHashMap<>(steps));
      Objects.requireNonNull(carry, "carry");
      Objects.requireNonNull(iterations, "iterations");
      Objects.requireNonNull(tables, "tables");
      List<String> iterationNames = new ArrayList<>(ITERATION_HEADER);
      iterationNames.addAll(ITERATION_KEYS);
      iterationNames.addAll(steps.keySet());
      iterationNames.addAll(iterations.names());
      distinct(iterationNames);
      List<String> tableNames = new ArrayList<>(TABLE_HEADER);
      tableNames.addAll(tables.names());
      distinct(tableNames);
    }

    /** run の種類の段と持ち越しの型を使う形。 */
    public static Layout of(RunKind<?, ?, ?, ?> kind, Columns<IterationReport> iterations, Columns<Node.Table> tables) {
      return new Layout(metrics(kind.steps()), kind.carry(), iterations, tables);
    }

    private static <K extends Enum<K> & StepKind<?>> Map<String, Class<? extends Record>> metrics(Class<K> declared) {
      Map<String, Class<? extends Record>> steps = new LinkedHashMap<>();
      for (K step : declared.getEnumConstants()) {
        steps.put(step.name(), step.metrics());
      }
      return steps;
    }

    private static void distinct(List<String> names) {
      if (names.stream().distinct().count() != names.size()) {
        throw new IllegalArgumentException("The report repeats a column name: " + names);
      }
    }
  }

  /**
   * 反復の 1 行。
   *
   * @param report 反復の報告
   * @param rowChanged 設定の行が、前の反復に固定した行と違う（ファイルの最初の反復では false）
   * @param cells run の種類の列の値
   */
  public record IterationLine(IterationReport report, boolean rowChanged, List<Cell> cells) {

    public IterationLine {
      Objects.requireNonNull(report, "report");
      cells = List.copyOf(cells);
    }
  }

  /**
   * 卓の最も新しい結果。
   *
   * @param table 卓の名前
   * @param version 測ったバージョン
   * @param cells run の種類の列の値
   */
  public record TableLine(String table, int version, List<Cell> cells) {

    public TableLine {
      Objects.requireNonNull(table, "table");
      cells = List.copyOf(cells);
    }
  }

  /** 反復の報告の行のうち、出さない反復でも読むキー。 */
  private record Fixed(int iteration, int settingsRow) {}

  /** ジョブの結果の行で、結果のフィールドより前に並ぶキー。 */
  private record Head(String job, int version, String table) {}

  private final RunDirectory run;
  private final Layout layout;
  private final int closed;
  private final List<IterationLine> iterations;
  private final List<TableLine> tables;

  private RunReport(RunDirectory run, Layout layout, int closed, List<IterationLine> iterations, List<TableLine> tables) {
    this.run = run;
    this.layout = layout;
    this.closed = closed;
    this.iterations = List.copyOf(iterations);
    this.tables = List.copyOf(tables);
  }

  /**
   * run の報告を読む。
   *
   * @param last 最も新しい反復から数えて出す反復の数。null なら全て
   * @throws SettingsException run のディレクトリでないか、読めない行がある（反復の番号が前の行より大きくない行と、出す反復のうち
   *     run の種類の型に合わない行も）
   */
  public static RunReport read(RunDirectory run, Layout layout, Integer last) throws IOException {
    if (last != null && last < 1) {
      throw new IllegalArgumentException("The number of iterations to show must be at least 1: " + last);
    }
    if (!Files.isRegularFile(run.settings())) {
      throw new SettingsException(
          List.of("The run " + run.root() + " has no settings rows: create it with settings create"));
    }
    Path file = run.report(RunDirectory.ITERATIONS);
    List<String> written = DurableFiles.lines(file);
    int from = last == null ? 0 : Math.max(0, written.size() - last);
    List<IterationLine> shown = new ArrayList<>(written.size() - from);
    Fixed previous = null;
    for (int index = 0; index < written.size(); index++) {
      Node.Table line = Json.read(written.get(index), file + ":" + (index + 1));
      Fixed fixed = keys(Fixed.class, line);
      if (previous != null && fixed.iteration() <= previous.iteration()) {
        throw new SettingsException(
            List.of(
                "Iteration " + fixed.iteration() + " at " + line.origin() + " does not follow iteration "
                    + previous.iteration()));
      }
      if (index >= from) {
        IterationReport report = IterationReport.of(line, layout.steps(), layout.carry());
        boolean changed = previous != null && fixed.settingsRow() != previous.settingsRow();
        shown.add(new IterationLine(report, changed, layout.iterations().of(report)));
      }
      previous = fixed;
    }
    return new RunReport(
        run, layout, written.size(), shown, latest(run.report(RunDirectory.EVALUATIONS), layout.tables()));
  }

  /** 閉じた反復の数（reports/iterations.jsonl の行の数）。 */
  public int closed() {
    return closed;
  }

  /** 出す反復（古い順）。 */
  public List<IterationLine> iterations() {
    return iterations;
  }

  /** 卓ごとの最も新しい結果（卓の名前の順）。 */
  public List<TableLine> tables() {
    return tables;
  }

  /**
   * 表。1 行目に run のディレクトリと閉じた反復の数を書き、反復の表、空の行、卓の表を続ける。設定の行が前の反復と違う反復は、行の番号に
   * {@value #CHANGED} を付け、反復の表の下にその説明を書く。数の列は右に、ほかの列は左に寄せる。値の無いセルは {@code -}。
   */
  public String text() {
    StringBuilder out = new StringBuilder();
    out.append(run.root()).append(": ").append(closed).append(closed == 1 ? " closed iteration" : " closed iterations");
    if (iterations.size() < closed) {
      out.append(" (the last ").append(iterations.size()).append(" shown)");
    }
    out.append('\n');
    if (!iterations.isEmpty()) {
      List<String> header = new ArrayList<>(ITERATION_HEADER);
      header.addAll(layout.steps().keySet());
      header.addAll(layout.iterations().names());
      List<List<Cell>> rows = new ArrayList<>(iterations.size());
      for (IterationLine line : iterations) {
        IterationReport report = line.report();
        List<Cell> row = new ArrayList<>(header.size());
        row.add(Cell.integer(report.iteration()));
        row.add(Cell.integer(report.version()));
        Cell settingsRow = Cell.integer(report.settingsRow());
        row.add(line.rowChanged() ? new Cell(settingsRow.value(), settingsRow.text() + CHANGED) : settingsRow);
        row.add(Cell.decimal(report.seconds(), 1));
        for (String step : layout.steps().keySet()) {
          StepReport reported = report.steps().get(step);
          row.add(
              reported == null || reported.status() == StepReport.Status.SKIPPED
                  ? Cell.NONE
                  : Cell.decimal(reported.seconds(), 1));
        }
        row.addAll(line.cells());
        rows.add(row);
      }
      table(out, header, rows);
      if (iterations.stream().anyMatch(IterationLine::rowChanged)) {
        out.append(CHANGED).append(" the settings row differs from the iteration before\n");
      }
    }
    out.append('\n');
    if (tables.isEmpty()) {
      out.append("No finished jobs\n");
      return out.toString();
    }
    List<String> header = new ArrayList<>(TABLE_HEADER);
    header.addAll(layout.tables().names());
    List<List<Cell>> rows = new ArrayList<>(tables.size());
    for (TableLine line : tables) {
      List<Cell> row = new ArrayList<>(header.size());
      row.add(Cell.text(line.table()));
      row.add(Cell.integer(line.version()));
      row.addAll(line.cells());
      rows.add(row);
    }
    table(out, header, rows);
    return out.toString();
  }

  /**
   * 表と同じ中身の JSON。キーは {@code run}（ディレクトリ）・{@code closed}（閉じた反復の数）・{@code iterations}（出す反復の列。
   * 古い順）・{@code tables}（卓の名前から最も新しい結果へ。名前の順）。反復は {@code iteration}・{@code version}・
   * {@code settingsRow}・{@code rowChanged}（設定の行が前の反復と違う）・{@code seconds}・{@code steps}（段の名前から {@code status}
   * と {@code seconds} へ）と run の種類の列を、卓は {@code version} と run の種類の列を持つ。数は丸めずに書き、値の無い列はキーごと
   * 書かない。
   */
  public Node.Table json() {
    List<Node> lines = new ArrayList<>(iterations.size());
    for (IterationLine line : iterations) {
      IterationReport report = line.report();
      Map<String, Node> entries = new LinkedHashMap<>();
      put(entries, "iteration", Cell.integer(report.iteration()));
      put(entries, "version", Cell.integer(report.version()));
      put(entries, "settingsRow", Cell.integer(report.settingsRow()));
      entries.put("rowChanged", new Node.Bool(line.rowChanged(), ORIGIN));
      put(entries, "seconds", Cell.decimal(report.seconds(), 1));
      Map<String, Node> steps = new LinkedHashMap<>();
      for (String step : layout.steps().keySet()) {
        StepReport reported = report.steps().get(step);
        if (reported != null) {
          Map<String, Node> fields = new LinkedHashMap<>();
          put(fields, "status", Cell.text(reported.status().name()));
          put(fields, "seconds", Cell.decimal(reported.seconds(), 1));
          steps.put(step, new Node.Table(fields, ORIGIN));
        }
      }
      entries.put("steps", new Node.Table(steps, ORIGIN));
      put(entries, layout.iterations().names(), line.cells());
      lines.add(new Node.Table(entries, ORIGIN));
    }
    Map<String, Node> results = new LinkedHashMap<>();
    for (TableLine line : tables) {
      Map<String, Node> entries = new LinkedHashMap<>();
      put(entries, "version", Cell.integer(line.version()));
      put(entries, layout.tables().names(), line.cells());
      results.put(line.table(), new Node.Table(entries, ORIGIN));
    }
    Map<String, Node> top = new LinkedHashMap<>();
    put(top, "run", Cell.text(run.root().toString()));
    put(top, "closed", Cell.integer(closed));
    top.put("iterations", new Node.Array(lines, ORIGIN));
    top.put("tables", new Node.Table(results, ORIGIN));
    return new Node.Table(top, ORIGIN);
  }

  /** 卓ごとに、バージョンが最も大きい結果の行から列を作る（卓の名前の順）。ファイルが無ければ空。 */
  private static List<TableLine> latest(Path file, Columns<Node.Table> columns) throws IOException {
    Map<String, Head> heads = new TreeMap<>();
    Map<String, Node.Table> lines = new TreeMap<>();
    List<String> read = DurableFiles.lines(file);
    for (int index = 0; index < read.size(); index++) {
      Node.Table line = Json.read(read.get(index), file + ":" + (index + 1));
      Head head = keys(Head.class, line);
      Head kept = heads.get(head.table());
      if (kept == null || head.version() > kept.version()) {
        heads.put(head.table(), head);
        lines.put(head.table(), line);
      }
    }
    List<TableLine> latest = new ArrayList<>(heads.size());
    for (Map.Entry<String, Head> entry : heads.entrySet()) {
      latest.add(new TableLine(entry.getKey(), entry.getValue().version(), columns.of(lines.get(entry.getKey()))));
    }
    return latest;
  }

  /** 行から、record の成分と同じ名前のキーだけを読む。ほかのキーは見ない。 */
  private static <R extends Record> R keys(Class<R> type, Node.Table line) {
    Map<String, Node> kept = new LinkedHashMap<>();
    for (RecordComponent component : type.getRecordComponents()) {
      Node node = line.get(component.getName());
      if (node != null) {
        kept.put(component.getName(), node);
      }
    }
    return RecordFile.read(type, new Node.Table(kept, line.origin())).value();
  }

  /** 列をそろえた表を書く。列の間は空白 2 つで、行の終わりの空白は書かない。 */
  private static void table(StringBuilder out, List<String> header, List<List<Cell>> rows) {
    int[] widths = new int[header.size()];
    boolean[] left = new boolean[header.size()];
    for (int column = 0; column < header.size(); column++) {
      widths[column] = header.get(column).length();
    }
    for (List<Cell> row : rows) {
      for (int column = 0; column < header.size(); column++) {
        Cell cell = row.get(column);
        widths[column] = Math.max(widths[column], cell.text().length());
        left[column] |= cell.value() != null && !(cell.value() instanceof Node.Numeral);
      }
    }
    line(out, header, widths, left);
    for (List<Cell> row : rows) {
      line(out, row.stream().map(Cell::text).toList(), widths, left);
    }
  }

  private static void line(StringBuilder out, List<String> texts, int[] widths, boolean[] left) {
    StringBuilder line = new StringBuilder();
    for (int column = 0; column < texts.size(); column++) {
      if (column > 0) {
        line.append("  ");
      }
      String text = texts.get(column);
      String padding = " ".repeat(widths[column] - text.length());
      line.append(left[column] ? text + padding : padding + text);
    }
    out.append(line.toString().stripTrailing()).append('\n');
  }

  private static void put(Map<String, Node> entries, String name, Cell cell) {
    if (cell.value() != null) {
      entries.put(name, cell.value());
    }
  }

  private static void put(Map<String, Node> entries, List<String> names, List<Cell> cells) {
    for (int index = 0; index < names.size(); index++) {
      put(entries, names.get(index), cells.get(index));
    }
  }
}
