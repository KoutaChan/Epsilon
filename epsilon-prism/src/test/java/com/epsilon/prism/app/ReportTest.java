package com.epsilon.prism.app;

import com.epsilon.agent.app.App;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.PlayStats;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.Stopped;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.IterationReport;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.collect.CollectionMetrics;
import com.epsilon.prism.eval.EvaluationJobs;
import com.epsilon.prism.grp.PhiEntry;
import com.epsilon.prism.grp.PhiEntryKind;
import com.epsilon.prism.grp.PhiValue;
import com.epsilon.prism.luck.LedgerReport;
import com.epsilon.prism.luck.UraAudit;
import com.epsilon.prism.run.CollectReport;
import com.epsilon.prism.run.LearnReport;
import com.epsilon.prism.run.PhiReport;
import com.epsilon.prism.run.PrismCarry;
import com.epsilon.prism.run.PrismStep;
import com.epsilon.prism.run.PublishReport;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * report のテスト: PRISM の段の報告とジョブの結果から作る列（行の数・帳簿の R²・ゲインと、相手・半荘の数・平均順位・δ・判定）、
 * --last と --json、読めない行と run を読むだけの引数の誤りで終了コード 2、run のファイルに何も書かないこと。
 */
public class ReportTest {

  private Path root;
  private RunDirectory run;

  @BeforeMethod
  public void createRun() throws IOException {
    root = Files.createTempDirectory("prism-report");
    run = new RunDirectory(root);
    Files.writeString(run.settings(), "", StandardCharsets.UTF_8);
    Files.createDirectories(run.reports());
    Files.writeString(
        run.report(RunDirectory.ITERATIONS),
        lines(
            iterationLine(1, 1, 622_715, 0.0627, 0.9),
            iterationLine(2, 1, 692_014, null, 0.7867),
            iterationLine(3, 2, 742_764, 0.0511, 0.7002)),
        StandardCharsets.UTF_8);
    Files.writeString(
        run.report(RunDirectory.EVALUATIONS),
        lines(
            resultLine(2, "champ", new EvaluationJobs.Crowned("prism:00002")),
            resultLine(2, "x1n3", duel(2.961)),
            resultLine(3, "x1n3", duel(2.95))),
        StandardCharsets.UTF_8);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    DurableFiles.delete(root);
  }

  @Test
  public void theTableShowsThePrismColumns() {
    Ran ran = report("--last", "2");
    Assert.assertEquals(ran.exit(), Exit.OK.code(), ran.out());
    List<String> lines = ran.out().lines().toList();
    Assert.assertEquals(lines.get(0), run.root() + ": 3 closed iterations (the last 2 shown)");
    Assert.assertEquals(
        tokens(lines.get(1)),
        List.of("iteration", "version", "row", "seconds", "PHI", "COLLECT", "LEARN", "PUBLISH", "rows", "r2", "gain"));
    Assert.assertEquals(
        tokens(lines.get(2)), List.of("2", "3", "1", "480.3", "0.2", "418.0", "59.0", "3.1", "692014", "-", "0.7867"));
    Assert.assertEquals(
        tokens(lines.get(3)),
        List.of("3", "4", "2*", "480.3", "0.2", "418.0", "59.0", "3.1", "742764", "0.0511", "0.7002"));
    Assert.assertEquals(lines.get(4), "* the settings row differs from the iteration before");
    Assert.assertEquals(lines.get(5), "");
    Assert.assertEquals(
        tokens(lines.get(6)),
        List.of(
            "table", "version", "opponent", "games", "meanRank", "meanRankError", "delta", "deltaError", "verdict",
            "stopped"));
    Assert.assertEquals(tokens(lines.get(7)), List.of("champ", "2", "-", "-", "-", "-", "-", "-", "CROWNED", "-"));
    Assert.assertEquals(
        tokens(lines.get(8)),
        List.of("x1n3", "3", "nano:latest", "131072", "2.9500", "0.0030", "-0.0390", "0.0030", "UNDECIDED", "LIMIT"));
    Assert.assertEquals(lines.size(), 9, ran.out());
  }

  @Test
  public void theJsonHoldsTheSameValuesOnOneLine() {
    Ran ran = report("--json", "true");
    Assert.assertEquals(ran.exit(), Exit.OK.code(), ran.out());
    Assert.assertEquals(ran.out().lines().count(), 1, ran.out());
    Node.Table json = Json.read(ran.out(), "report");
    Assert.assertEquals(json.get("closed"), numeral("3"));
    List<Node> iterations = ((Node.Array) json.get("iterations")).items();
    Assert.assertEquals(iterations.size(), 3);
    Node.Table second = (Node.Table) iterations.get(1);
    Assert.assertEquals(second.get("rows"), numeral("692014"));
    Assert.assertNull(second.get("r2"), second.toString());
    Assert.assertEquals(second.get("gain"), numeral("0.7867"));
    Node.Table third = (Node.Table) iterations.get(2);
    Assert.assertEquals(third.get("rowChanged"), new Node.Bool(true, "test"));
    Assert.assertEquals(third.get("r2"), numeral("0.0511"));
    Assert.assertEquals(
        ((Node.Table) third.get("steps")).get("LEARN"), Json.read("{\"status\":\"RAN\",\"seconds\":59.0}", "test"));
    Node.Table tables = (Node.Table) json.get("tables");
    Assert.assertEquals(List.copyOf(tables.entries().keySet()), List.of("champ", "x1n3"));
    Assert.assertEquals(tables.get("champ"), Json.read("{\"version\":2,\"verdict\":\"CROWNED\"}", "test"));
    Node.Table x1n3 = (Node.Table) tables.get("x1n3");
    Assert.assertEquals(x1n3.get("version"), numeral("3"));
    Assert.assertEquals(x1n3.get("opponent"), new Node.Text("nano:latest", "test"));
    Assert.assertEquals(x1n3.get("meanRank"), numeral("2.95"));
    Assert.assertEquals(x1n3.get("delta"), numeral("-0.039"));
    Assert.assertEquals(x1n3.get("stopped"), new Node.Text("LIMIT", "test"));
  }

  @Test
  public void mistakesExitWithUsageAndNothingIsWrittenToTheRun() throws IOException {
    Map<String, String> before = contents();
    Assert.assertEquals(report().exit(), Exit.OK.code());
    Assert.assertEquals(report("--json", "true", "--last", "1").exit(), Exit.OK.code());
    Assert.assertEquals(report("--last", "0").exit(), Exit.USAGE.code());
    Assert.assertEquals(report("--set", "fit.lrPeak=0.002").exit(), Exit.USAGE.code());
    Assert.assertEquals(contents(), before);
    Files.writeString(
        run.report(RunDirectory.EVALUATIONS),
        "{\"job\":\"v00004-x1n3\",\"version\":4,\"table\":\"x1n3\",\"walls\":1}\n",
        StandardCharsets.UTF_8,
        StandardOpenOption.APPEND);
    Assert.assertEquals(report().exit(), Exit.USAGE.code());
  }

  @Test
  public void lastShowsTheIterationsOfThisJarAfterTheRowsOfAnOlderJar() throws IOException {
    // 反復 1 は、収集の段の報告に Φ の数（phi）がまだ無かった jar が書いた行
    Node.Table first = Json.read(iterationLine(1, 1, 622_715, 0.0627, 0.9), "test");
    Node.Table steps = (Node.Table) first.get("steps");
    Map<String, Node> older = new LinkedHashMap<>(steps.entries());
    older.put(PrismStep.COLLECT.name(), ((Node.Table) steps.get(PrismStep.COLLECT.name())).without("phi"));
    Map<String, Node> entries = new LinkedHashMap<>(first.entries());
    entries.put("steps", new Node.Table(older, "test"));
    Files.writeString(
        run.report(RunDirectory.ITERATIONS),
        lines(
            Json.write(new Node.Table(entries, "test")),
            iterationLine(2, 1, 692_014, null, 0.7867),
            iterationLine(3, 2, 742_764, 0.0511, 0.7002)),
        StandardCharsets.UTF_8);
    Ran ran = report("--last", "2");
    Assert.assertEquals(ran.exit(), Exit.OK.code(), ran.out());
    Assert.assertEquals(
        ran.out().lines().findFirst().orElseThrow(), run.root() + ": 3 closed iterations (the last 2 shown)");
    Assert.assertEquals(report("--json", "true", "--last", "1").exit(), Exit.OK.code());
    // 出す反復の行は、今の型で読めなければ誤り
    Assert.assertEquals(report().exit(), Exit.USAGE.code());
  }

  @Test
  public void aDuelResultWithoutTheOpponentsSlotIsAMistakeOfItsLine() throws IOException {
    Node.Table result = RecordFile.tree(duel(2.94));
    Map<String, Node> entries = new LinkedHashMap<>(result.entries());
    entries.put("slots", new Node.Array(List.of(((Node.Array) result.get("slots")).items().get(0)), "test"));
    Files.writeString(
        run.report(RunDirectory.EVALUATIONS),
        lines(resultLine(4, "x1n3", new Node.Table(entries, "test"))),
        StandardCharsets.UTF_8,
        StandardOpenOption.APPEND);
    Assert.assertEquals(report().exit(), Exit.USAGE.code());
  }

  /** 1 回の呼び出しの終了コードと、標準出力に書いたもの。 */
  private record Ran(int exit, String out) {}

  private Ran report(String... arguments) {
    List<String> args = new ArrayList<>(List.of("report", "--run", root.toString()));
    args.addAll(List.of(arguments));
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    int exit;
    try (PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
      exit = App.run(PrismApp.application(), args.toArray(String[]::new), out);
    }
    return new Ran(exit, bytes.toString(StandardCharsets.UTF_8));
  }

  /** 反復 t の報告の行。秒は Φ 0.2・収集 418.0・学習 59.0・公開 3.1 で、帳簿の R² が無ければ帳簿を当てはめなかった反復。 */
  private static String iterationLine(int t, int row, long rows, Double r2, double gain) {
    PhiValue phi = PhiValue.grp(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, Variant.TENHOU_4P));
    Map<String, StepReport> steps = new LinkedHashMap<>();
    steps.put(PrismStep.PHI.name(), ran(0.2, new PhiReport(phi, 0)));
    steps.put(PrismStep.COLLECT.name(), ran(418.0, collected(t, rows)));
    steps.put(PrismStep.LEARN.name(), ran(59.0, learned(t, r2, gain)));
    steps.put(PrismStep.PUBLISH.name(), ran(3.1, new PublishReport(t + 1, null)));
    IterationReport report =
        new IterationReport(
            t,
            t + 1,
            row,
            Instant.parse("2026-11-10T00:00:00Z"),
            480.3,
            steps,
            new PrismCarry(r2 == null ? 0 : r2, null),
            new IterationReport.Resources(31_200, 14_100, List.of(9_270L, 10_307L)),
            List.of());
    return Json.write(report.tree());
  }

  private static StepReport ran(double seconds, Record metrics) {
    return new StepReport(StepReport.Status.RAN, seconds, 1, "test0001", metrics);
  }

  private static CollectReport collected(int start, long rows) {
    return new CollectReport(
        start,
        64,
        512,
        40_000,
        2_000,
        1_000.0,
        rows,
        rows * 100,
        rows / 2,
        10,
        300,
        5,
        50,
        7,
        3,
        11,
        new CollectionMetrics.Breakdown(Map.of(), Map.of(), Map.of(), Map.of(), Map.of(), Map.of()),
        new UraAudit.Summary(0, 0, 0, List.of(), 0, List.of()),
        new Phi.Stats(0, 0, 0, 0, 0, new Phi.LateStats(0, 0, 0, 0, 0, 0, 0, 0, 0)));
  }

  private static LearnReport learned(int start, Double r2, double gain) {
    LedgerReport.Summary ledger =
        r2 == null ? null : new LedgerReport.Summary(1_000, 500, 0, r2, 1.5, Map.of(), 2.0, List.of(), 3.0, true);
    return new LearnReport(
        start,
        1_000,
        100,
        null,
        new LearnReport.Tempo(gain, 0, 1, gain, 0.05),
        ledger,
        r2 == null ? 0 : r2,
        new LearnReport.Fit(1_000, 150, 0, 0.5, 0.6, 0.4, 10.0));
  }

  /** x1n3 の決闘の結果（候補の平均順位だけを変える）。 */
  private static DuelResult duel(double meanRank) {
    Estimate rank = Estimate.of(-0.039, 0.003, 2.576);
    Estimate utility = Estimate.of(0.05, 0.01, 2.576);
    List<Double> shares = List.of(0.25, 0.25, 0.25, 0.25);
    return new DuelResult(
        65_536,
        131_072,
        Verdict.UNDECIDED,
        Stopped.LIMIT,
        Score.RANK,
        2.576,
        rank,
        utility,
        List.of(new DuelResult.SplitValue("1v3", rank, utility)),
        List.of(
            new DuelResult.SlotValue("prism:00003", meanRank, 0.003, 0.05, 0.01, shares),
            new DuelResult.SlotValue("nano:latest", 2.513, 0.001, -0.02, 0.003, shares)),
        new PlayStats.Rates(0.01, -0.005, 0.02, -0.01, 0),
        0,
        0,
        597.0);
  }

  /** ジョブの結果の行（ジョブの名前・バージョン・卓に続けて、結果のフィールド）。 */
  private static String resultLine(int version, String table, Record result) {
    return resultLine(version, table, RecordFile.tree(result));
  }

  /** 結果のフィールドを木で渡すジョブの結果の行（結果の record では作れない、壊れた行のため）。 */
  private static String resultLine(int version, String table, Node.Table result) {
    Map<String, Node> entries = new LinkedHashMap<>();
    entries.put("job", new Node.Text(RunDirectory.jobName(version, table), "test"));
    entries.put("version", numeral(Integer.toString(version)));
    entries.put("table", new Node.Text(table, "test"));
    entries.putAll(result.entries());
    return Json.write(new Node.Table(entries, "test"));
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

  private static Node numeral(String literal) {
    return new Node.Numeral(literal, "test");
  }

  private static List<String> tokens(String line) {
    return List.of(line.strip().split(" +"));
  }

  private static String lines(String... lines) {
    return String.join("\n", lines) + "\n";
  }
}
