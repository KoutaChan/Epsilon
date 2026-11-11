package com.epsilon.agent.app;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.app.RunReport.Cell;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.IterationReport;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.run.StepReport.Status;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * run の報告のテスト: 作り物の run のディレクトリから作る表と JSON の形、最も新しい反復だけを出すこと、設定の行が前の反復と違う反復の印、
 * ジョブも反復も無い run、読めない行を誤りにすること、run のファイルに何も書かないこと。
 */
public class RunReportTest {

  /** 収集の段の報告の中身。 */
  record Collected(long rows) {}

  /** 公開の段の報告の中身。 */
  record Published(int version) {}

  /** 反復の報告の持ち越し。 */
  record Carry(long value) {}

  /** ジョブの結果。 */
  record Result(long games, long sum) {}

  /** 反復の表の列の幅と寄せ方（反復・バージョン・設定の行・秒・COLLECT・PUBLISH・rows）。 */
  private static final String ITERATIONS = "%9s  %7s  %3s  %7s  %7s  %7s  %4s";

  /** 卓の表の列の幅と寄せ方（卓・バージョン・games・sum）。 */
  private static final String TABLES = "%-5s  %7s  %5s  %3s";

  private static final String CHANGED_LINE = "* the settings row differs from the iteration before";

  private static final RunReport.Layout LAYOUT = layout();

  private Path root;
  private RunDirectory run;

  @BeforeMethod
  public void createRun() throws IOException {
    root = Files.createTempDirectory("run-report");
    run = new RunDirectory(root);
    Node.Table settings = new Node.Table(Map.of(), "test");
    try (SettingsRows.Locked locked = new SettingsRows(run).lock()) {
      locked.append(1, "test0001", SettingsRow.CREATED, List.of(), settings);
      locked.append(3, "test0001", "raise the rate", List.of(), settings);
    }
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    DurableFiles.delete(root);
  }

  @Test
  public void theTableShowsEveryIterationAndTheNewestResultOfEachTable() throws IOException {
    writeIterations();
    writeEvaluations();
    String expected =
        lines(
            run.root() + ": 4 closed iterations",
            row(ITERATIONS, "iteration", "version", "row", "seconds", "COLLECT", "PUBLISH", "rows"),
            row(ITERATIONS, "1", "2", "1", "10.5", "10.0", "0.5", "100"),
            row(ITERATIONS, "2", "3", "1", "12.2", "12.2", "-", "120"),
            row(ITERATIONS, "3", "4", "2*", "11.7", "11.0", "0.7", "110"),
            row(ITERATIONS, "4", "5", "2", "13.5", "13.0", "0.5", "1300"),
            CHANGED_LINE,
            "",
            row(TABLES, "table", "version", "games", "sum"),
            row(TABLES, "a", "3", "5", "250"),
            row(TABLES, "b", "4", "10", "300"));
    assertEquals(RunReport.read(run, LAYOUT, null).text(), expected);
  }

  @Test
  public void lastShowsTheNewestIterationsAndMarksAgainstTheIterationBefore() throws IOException {
    writeIterations();
    RunReport two = RunReport.read(run, LAYOUT, 2);
    assertEquals(two.closed(), 4);
    assertEquals(two.iterations().stream().map(line -> line.report().iteration()).toList(), List.of(3, 4));
    assertEquals(
        two.text(),
        lines(
            run.root() + ": 4 closed iterations (the last 2 shown)",
            row(ITERATIONS, "iteration", "version", "row", "seconds", "COLLECT", "PUBLISH", "rows"),
            row(ITERATIONS, "3", "4", "2*", "11.7", "11.0", "0.7", "110"),
            row(ITERATIONS, "4", "5", "2", "13.5", "13.0", "0.5", "1300"),
            CHANGED_LINE,
            "",
            "No finished jobs"));
    String one = RunReport.read(run, LAYOUT, 1).text();
    assertTrue(one.contains(row(ITERATIONS, "4", "5", "2", "13.5", "13.0", "0.5", "1300") + "\n"), one);
    assertFalse(one.contains(CHANGED_LINE), one);
    assertEquals(RunReport.read(run, LAYOUT, 10).iterations().size(), 4);
  }

  @Test
  public void theJsonHoldsTheSameLinesWithUnroundedNumbers() throws IOException {
    writeIterations();
    writeEvaluations();
    String expected =
        "{\"run\":"
            + Json.write(new Node.Text(run.root().toString(), "test"))
            + ",\"closed\":4,\"iterations\":["
            + "{\"iteration\":2,\"version\":3,\"settingsRow\":1,\"rowChanged\":false,\"seconds\":12.24,\"steps\":"
            + "{\"COLLECT\":{\"status\":\"RAN\",\"seconds\":12.24},\"PUBLISH\":{\"status\":\"SKIPPED\",\"seconds\":0.0}},"
            + "\"rows\":120},"
            + "{\"iteration\":3,\"version\":4,\"settingsRow\":2,\"rowChanged\":true,\"seconds\":11.7,\"steps\":"
            + "{\"COLLECT\":{\"status\":\"REUSED\",\"seconds\":11.0},\"PUBLISH\":{\"status\":\"RAN\",\"seconds\":0.7}},"
            + "\"rows\":110},"
            + "{\"iteration\":4,\"version\":5,\"settingsRow\":2,\"rowChanged\":false,\"seconds\":13.5,\"steps\":"
            + "{\"COLLECT\":{\"status\":\"RAN\",\"seconds\":13.0},\"PUBLISH\":{\"status\":\"RAN\",\"seconds\":0.5}},"
            + "\"rows\":1300}],"
            + "\"tables\":{\"a\":{\"version\":3,\"games\":5,\"sum\":250},\"b\":{\"version\":4,\"games\":10,\"sum\":300}}}";
    assertEquals(Json.write(RunReport.read(run, LAYOUT, 3).json()), expected);
  }

  @Test
  public void aRunWithoutIterationsOrJobsShowsWhatItHas() throws IOException {
    RunReport empty = RunReport.read(run, LAYOUT, null);
    assertEquals(empty.text(), lines(run.root() + ": 0 closed iterations", "", "No finished jobs"));
    assertEquals(
        Json.write(empty.json()),
        "{\"run\":"
            + Json.write(new Node.Text(run.root().toString(), "test"))
            + ",\"closed\":0,\"iterations\":[],\"tables\":{}}");
    writeIterations();
    RunReport noJobs = RunReport.read(run, LAYOUT, null);
    assertEquals(noJobs.tables(), List.of());
    assertTrue(noJobs.text().endsWith(CHANGED_LINE + "\n\nNo finished jobs\n"), noJobs.text());
    assertEquals(noJobs.json().get("tables"), new Node.Table(Map.of(), "test"));
  }

  @Test
  public void brokenLinesAreErrorsButAnUnfinishedLastLineIsNotRead() throws IOException {
    writeIterations();
    writeEvaluations();
    Path iterations = run.report(RunDirectory.ITERATIONS);
    Path evaluations = run.report(RunDirectory.EVALUATIONS);
    List<String> written = DurableFiles.lines(iterations);
    List<String> results = DurableFiles.lines(evaluations);

    // 落ちた JVM が書きかけた最後の行は読まない
    Files.writeString(iterations, "{\"iteration\":5,\"vers", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    Files.writeString(evaluations, "{\"job\":\"v00009-a\"", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    RunReport report = RunReport.read(run, LAYOUT, null);
    assertEquals(report.closed(), 4);
    assertEquals(report.tables().getFirst().version(), 3);

    assertBroken(iterations, List.of(written.get(0), "{\"iteration\":2,", written.get(2)), iterations + ":2");
    assertBroken(
        iterations,
        List.of(written.get(0), written.get(2), written.get(1)),
        "Iteration 2 at " + iterations + ":3 does not follow iteration 3");
    assertBroken(iterations, List.of(written.get(0).replace("\"PUBLISH\"", "\"LEARN\"")), "Unknown step LEARN");
    Files.writeString(iterations, lines(written.toArray(String[]::new)), StandardCharsets.UTF_8);
    assertBroken(evaluations, List.of(results.get(0), "not a line"), evaluations + ":2");
    assertBroken(evaluations, List.of(results.get(0).replace(",\"table\":\"b\"", "")), "Missing");
    assertBroken(evaluations, List.of(results.get(0).replace("\"sum\"", "\"total\"")), "total");
  }

  /** 反復の行は、設定の行の番号だけで設定を指し、読んで木に戻すと同じ報告になる。設定の中身の SHA-256 の欄がある行は読まない。 */
  @Test
  public void anIterationLineNamesItsSettingsByTheRowAlone() throws IOException {
    writeIterations();
    Path iterations = run.report(RunDirectory.ITERATIONS);
    List<String> written = DurableFiles.lines(iterations);
    for (String line : written) {
      assertFalse(line.contains("settingsSha"), line);
    }
    for (IterationReport report : IterationReport.read(iterations, LAYOUT.steps(), Carry.class)) {
      assertEquals(IterationReport.of(Json.read(Json.write(report.tree()), "test"), LAYOUT.steps(), Carry.class), report);
    }
    assertBroken(
        iterations,
        List.of(written.get(0).replace("\"settingsRow\":1,", "\"settingsRow\":1,\"settingsSha\":\"ab\",")),
        "settingsSha");
  }

  @Test
  public void lastReadsOnlyTheShownIterationsWithTheTypesOfThisJar() throws IOException {
    writeIterations();
    Path iterations = run.report(RunDirectory.ITERATIONS);
    List<String> written = DurableFiles.lines(iterations);
    // 反復 1 と 2 は前の jar が書いた行: 収集の段の報告に今の型の rows が無く、今の型に無い decisions がある
    Files.writeString(
        iterations,
        lines(
            written.get(0).replace("\"rows\"", "\"decisions\""),
            written.get(1).replace("\"rows\"", "\"decisions\""),
            written.get(2),
            written.get(3)),
        StandardCharsets.UTF_8);
    RunReport two = RunReport.read(run, LAYOUT, 2);
    assertEquals(two.closed(), 4);
    assertEquals(
        two.text(),
        lines(
            run.root() + ": 4 closed iterations (the last 2 shown)",
            row(ITERATIONS, "iteration", "version", "row", "seconds", "COLLECT", "PUBLISH", "rows"),
            row(ITERATIONS, "3", "4", "2*", "11.7", "11.0", "0.7", "110"),
            row(ITERATIONS, "4", "5", "2", "13.5", "13.0", "0.5", "1300"),
            CHANGED_LINE,
            "",
            "No finished jobs"));

    // 出す反復の行は、今の型で読めなければ誤り
    SettingsException error = expectThrows(SettingsException.class, () -> RunReport.read(run, LAYOUT, 3));
    assertTrue(error.getMessage().contains(iterations + ":2"), error.getMessage());
  }

  @Test
  public void aStepThatBreaksTheRulesOfAStepReportIsAnErrorOfItsLine() throws IOException {
    writeIterations();
    Path iterations = run.report(RunDirectory.ITERATIONS);
    List<String> written = DurableFiles.lines(iterations);
    // 秒が負の段
    assertBroken(
        iterations, List.of(written.get(0).replace("\"seconds\":10.0", "\"seconds\":-10.0")), iterations + ":1");
    // 行わなかった段に中身のフィールドがある
    assertBroken(
        iterations,
        List.of(
            written.get(0), written.get(1).replace("\"status\":\"SKIPPED\"", "\"status\":\"SKIPPED\",\"version\":3")),
        iterations + ":2");
  }

  @Test
  public void readingWritesNothingToTheRun() throws IOException {
    writeIterations();
    writeEvaluations();
    Map<String, String> before = contents();
    RunReport report = RunReport.read(run, LAYOUT, 2);
    report.text();
    report.json();
    assertEquals(contents(), before);
  }

  @Test
  public void aDirectoryWithoutSettingsRowsIsNotARun() throws IOException {
    Path other = Files.createTempDirectory("not-a-run");
    try {
      SettingsException error =
          expectThrows(SettingsException.class, () -> RunReport.read(new RunDirectory(other), LAYOUT, null));
      assertTrue(error.getMessage().contains("has no settings rows"), error.getMessage());
    } finally {
      DurableFiles.delete(other);
    }
  }

  @Test
  public void aKindCannotReuseTheNameOfACommonColumn() {
    expectThrows(
        IllegalArgumentException.class,
        () ->
            new RunReport.Layout(
                LAYOUT.steps(),
                Carry.class,
                new RunReport.Columns<IterationReport>(List.of("seconds"), report -> List.of(Cell.NONE)),
                LAYOUT.tables()));
    expectThrows(
        IllegalArgumentException.class,
        () ->
            new RunReport.Layout(
                LAYOUT.steps(),
                Carry.class,
                LAYOUT.iterations(),
                new RunReport.Columns<Node.Table>(List.of("version"), line -> List.of(Cell.NONE))));
  }

  private static RunReport.Layout layout() {
    Map<String, Class<? extends Record>> steps = new LinkedHashMap<>();
    steps.put("COLLECT", Collected.class);
    steps.put("PUBLISH", Published.class);
    return new RunReport.Layout(
        steps,
        Carry.class,
        new RunReport.Columns<>(List.of("rows"), RunReportTest::iterationCells),
        new RunReport.Columns<>(List.of("games", "sum"), RunReportTest::tableCells));
  }

  private static List<Cell> iterationCells(IterationReport report) {
    return List.of(
        report.steps().get("COLLECT").metrics() instanceof Collected collected
            ? Cell.integer(collected.rows())
            : Cell.NONE);
  }

  private static List<Cell> tableCells(Node.Table line) {
    Result result = Jobs.result(line, Result.class);
    return List.of(Cell.integer(result.games()), Cell.integer(result.sum()));
  }

  /** 4 つの反復。設定の行は 1・1・2・2 で、反復 2 は公開の段を行わず、反復 3 は収集の段の報告を使い回した。 */
  private void writeIterations() throws IOException {
    write(
        RunDirectory.ITERATIONS,
        iterationLine(1, 1, 10.5, step(Status.RAN, 10.0, new Collected(100)), step(Status.RAN, 0.5, new Published(2))),
        iterationLine(2, 1, 12.24, step(Status.RAN, 12.24, new Collected(120)), step(Status.SKIPPED, 0, null)),
        iterationLine(3, 2, 11.7, step(Status.REUSED, 11.0, new Collected(110)), step(Status.RAN, 0.7, new Published(4))),
        iterationLine(4, 2, 13.5, step(Status.RAN, 13.0, new Collected(1300)), step(Status.RAN, 0.5, new Published(5))));
  }

  /** 卓 a と b の結果。a のバージョン 2 は、バージョン 3 の後に足した（過去のバージョンを後から測ったジョブ）。 */
  private void writeEvaluations() throws IOException {
    write(
        RunDirectory.EVALUATIONS,
        resultLine(2, "b", 5, 200),
        resultLine(3, "a", 5, 250),
        resultLine(4, "b", 10, 300),
        resultLine(2, "a", 5, 210));
  }

  private static StepReport step(Status status, double seconds, Record metrics) {
    return new StepReport(status, seconds, 1, "test0001", metrics);
  }

  private static String iterationLine(int t, int row, double seconds, StepReport collect, StepReport publish) {
    Map<String, StepReport> steps = new LinkedHashMap<>();
    steps.put("COLLECT", collect);
    steps.put("PUBLISH", publish);
    IterationReport report =
        new IterationReport(
            t,
            t + 1,
            row,
            Instant.parse("2026-11-10T00:00:00Z"),
            seconds,
            steps,
            new Carry(t),
            new IterationReport.Resources(0, 0, List.of()),
            List.of());
    return Json.write(report.tree());
  }

  private static String resultLine(int version, String table, long games, long sum) {
    return "{\"job\":\"" + RunDirectory.jobName(version, table) + "\",\"version\":" + version + ",\"table\":\"" + table
        + "\",\"games\":" + games + ",\"sum\":" + sum + "}";
  }

  private void write(String name, String... lines) throws IOException {
    Files.createDirectories(run.reports());
    Files.writeString(run.report(name), lines(lines), StandardCharsets.UTF_8);
  }

  /** 行を書き換えたファイルを読むと、その文を含む誤りになる。 */
  private void assertBroken(Path file, List<String> lines, String problem) throws IOException {
    Files.writeString(file, lines(lines.toArray(String[]::new)), StandardCharsets.UTF_8);
    SettingsException error = expectThrows(SettingsException.class, () -> RunReport.read(run, LAYOUT, null));
    assertTrue(error.getMessage().contains(problem), error.getMessage());
  }

  /** run のディレクトリの全てのファイルとディレクトリの、中身と更新の時刻。 */
  private Map<String, String> contents() throws IOException {
    Map<String, String> contents = new TreeMap<>();
    try (Stream<Path> walk = Files.walk(root)) {
      for (Path path : walk.toList()) {
        String content = Files.isDirectory(path) ? "directory" : Files.readString(path, StandardCharsets.UTF_8);
        contents.put(root.relativize(path).toString(), content + " " + Files.getLastModifiedTime(path));
      }
    }
    return contents;
  }

  private static String row(String format, String... cells) {
    return String.format(Locale.ROOT, format, (Object[]) cells);
  }

  /** 行を改行でつなぎ、最後にも改行を付ける。 */
  private static String lines(String... lines) {
    return String.join("\n", lines) + "\n";
  }
}
