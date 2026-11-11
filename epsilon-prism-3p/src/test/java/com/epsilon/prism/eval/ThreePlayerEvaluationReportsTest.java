package com.epsilon.prism.eval;

import static com.epsilon.prism.eval.ThreePlayerFrozenLadderTest.concat;
import static com.epsilon.prism.eval.ThreePlayerFrozenLadderTest.table;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSizing;
import com.epsilon.agent.eval.Sequential;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.Main;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.run.PrismCarry;
import com.epsilon.prism.run.PrismState;
import com.epsilon.prism.run.Regression;
import com.epsilon.prism.settings.PrismSettings;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 三人打ちの系譜の評価の報告（{@link EvaluationReports}）と、duel-sigma のコマンドを確かめる。答えの分かる合成の決闘（{@link SyntheticDuels}）
 * で、次を確かめる。
 *
 * <ul>
 *   <li>baseline.tsv: バージョンごとの δ・標準誤差・1v2 の δ・jar・同値な打牌の切り方と、jar か切り方が変わるたびに 1 増える区切りの
 *       番号。2 つ目の基準の卓は、両方で測ったバージョンでの差の平均（offset）と、差ごとの標準誤差の平均（SE）で、1 つ目の尺度に直す。
 *   <li>ratings.tsv: 基準の卓と三人打ちのラダーの卓の辺から求めた θ が、真の θ と標準誤差の範囲内で合う。
 *   <li>duel-sigma: 直近の王者との決闘の σ の中央値と、半荘の上限の中で勧める牌山の数・互角の幅が手計算と合う。
 * </ul>
 */
public class ThreePlayerEvaluationReportsTest {

  private Path root;
  private RunDirectory run;

  @BeforeMethod
  public void createRun() throws Exception {
    root = Files.createTempDirectory("prism-3p-evaluation-reports");
    run = new RunDirectory(root);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws Exception {
    DurableFiles.delete(root);
  }

  private PrismSettings lineage(int versions, String... assignments) throws Exception {
    PrismSettings settings = ThreePlayerSettings.create(root, List.of(), assignments);
    for (int version = 1; version <= versions; version++) {
      Files.createDirectories(run.snapshot(version));
      Files.writeString(run.snapshot(version).resolve("weights.bin"), "version " + version);
    }
    DurableFiles.write(
        run.state(), new PrismState(versions + 1, 0, PrismCarry.NONE, versions + 1, Regression.NONE));
    return settings;
  }

  private Jobs jobs() {
    return new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS));
  }

  /** バージョンを予定どおりの卓で測るジョブを作り、全て実行する。 */
  private void measure(
      PrismSettings settings, int version, Function<JobFile, DuelResult> champion, SyntheticDuels.Places places)
      throws Exception {
    measure(settings, 1, version, champion, places);
  }

  /** バージョンを予定で測る卓の仕事を設定の行 row で作って、全て回す。 */
  private void measure(
      PrismSettings settings,
      int row,
      int version,
      Function<JobFile, DuelResult> champion,
      SyntheticDuels.Places places)
      throws Exception {
    int order = 0;
    for (Map.Entry<String, TableSpec> entry : settings.evaluation().scheduled(version).entrySet()) {
      jobs().create(JobFile.pending(version, entry.getKey(), order++, row, RecordFile.tree(entry.getValue())));
    }
    SyntheticDuels.Runner runner =
        new SyntheticDuels.Runner(new EvaluationJobs(run, settings, null, null), champion, places);
    Assert.assertTrue(jobs().runAll(runner, RunControl.detached(() -> false)).isEmpty());
  }

  private void settleAgain(PrismSettings settings, String name) throws Exception {
    SyntheticDuels.Runner runner =
        new SyntheticDuels.Runner(
            new EvaluationJobs(run, settings, null, null), SyntheticDuels::noChampion, (job, wall, game) -> 0);
    Assert.assertTrue(jobs().run(jobs().read(name), runner, RunControl.detached(() -> false)).isEmpty());
  }

  private DuelResult result(int version, String table) throws Exception {
    return Jobs.result(run.job(RunDirectory.jobName(version, table)), DuelResult.class);
  }

  /** 報告の表を読む。# で始まる行を除き、ヘッダーの行にある列の名前から値への対応を、行ごとに返す。 */
  private List<Map<String, String>> rows(String report) throws Exception {
    List<String> lines =
        Files.readAllLines(run.report(report), StandardCharsets.UTF_8).stream()
            .filter(line -> !line.startsWith("#"))
            .toList();
    String[] header = lines.getFirst().split("\t", -1);
    List<Map<String, String>> rows = new ArrayList<>();
    for (String line : lines.subList(1, lines.size())) {
      String[] fields = line.split("\t", -1);
      Assert.assertEquals(fields.length, header.length, line);
      Map<String, String> row = new HashMap<>();
      for (int i = 0; i < header.length; i++) {
        row.put(header[i], fields[i]);
      }
      rows.add(row);
    }
    return rows;
  }

  /** 真の θ（v1 が 0 で、1 バージョンごとに 0.1 良くなる）。 */
  private static double theta(int version) {
    return -0.1 * (version - 1);
  }

  /** 候補と計画の相手の θ の差が、δ の期待値になる着順。 */
  private static SyntheticDuels.Places truth() {
    return SyntheticDuels.drawn(
        job -> {
          TableSpec table = RecordFile.read(TableSpec.class, job.definition()).value();
          return theta(job.version()) - theta(table.opponentVersion(job.version()));
        });
  }

  /** 基準の卓 2 つ（version:1 をバージョン 2 から、version:3 をバージョン 4 から）と、1 つ前のバージョンが相手のラダーの卓。尺度の基準は reference。 */
  private PrismSettings chain() throws Exception {
    return lineage(
        6,
        concat(
            table("reference", "version:1", "FIXED", 240, "FIXED", 1, 2, false),
            table("second", "version:3", "FIXED", 240, "FIXED", 1, 4, false),
            table("ladder", "version:-1", "FRESH", 240, "FIXED", 1, 2, null),
            List.of("evaluation.anchor=reference")));
  }

  // baseline.tsv と、2 つ目の基準のつなぎ

  @Test(timeOut = 600_000)
  public void theBaselineReportLinksTheSecondBaselineByTheOverlap() throws Exception {
    PrismSettings settings = chain();
    // ラダーは、相手が基準の卓と重なる回（v2 は reference と、v4 は second と）は打たない
    Assert.assertEquals(settings.evaluation().scheduled(2).keySet().toString(), "[reference]");
    Assert.assertEquals(settings.evaluation().scheduled(4).keySet().toString(), "[reference, second]");
    Assert.assertEquals(settings.evaluation().coveredLadders("second", 4), List.of("ladder"));
    for (int version = 2; version <= 6; version++) {
      measure(settings, version, SyntheticDuels::noChampion, truth());
    }

    // 手計算: 両方で測った v4〜v6 での 1 つ目 − 2 つ目の平均と、差ごとの標準誤差 √(e1² + e2²) の平均
    double offset = 0;
    double linkError = 0;
    for (int version = 4; version <= 6; version++) {
      DuelResult first = result(version, "reference");
      DuelResult second = result(version, "second");
      offset += first.rank().mean() - second.rank().mean();
      linkError += Math.hypot(first.rank().error(), second.rank().error());
    }
    offset /= 3;
    linkError /= 3;
    // 真の offset は θ(1) − θ(3) = −0.2
    Assert.assertEquals(offset, theta(3) - theta(1), 4 * linkError, "offset " + offset + " ± " + linkError);

    List<Map<String, String>> rows = rows(EvaluationReports.BASELINE);
    Assert.assertEquals(rows.size(), 8);
    List<String> order = new ArrayList<>();
    for (Map<String, String> row : rows) {
      int version = Integer.parseInt(row.get("version"));
      String table = row.get("table");
      order.add(version + table);
      DuelResult result = result(version, table);
      double value = result.rank().mean();
      double error = result.rank().error();
      Assert.assertEquals(row.get("opponent"), table.equals("reference") ? "v00001" : "v00003");
      Assert.assertEquals(Double.parseDouble(row.get("rank")), value, 1e-6, row.toString());
      Assert.assertEquals(Double.parseDouble(row.get("error")), error, 1e-6, row.toString());
      Assert.assertEquals(Double.parseDouble(row.get("oneVsRest")), value, 1e-6, "only 1v2 is played");
      Assert.assertEquals(row.get("segment"), "0");
      Assert.assertEquals(
          row.get("jar"), RecordFile.read(EvaluationPlan.class, jobs().read(RunDirectory.jobName(version, table)).plan())
              .value()
              .jar());
      double scaled = table.equals("reference") ? value : value + offset;
      double scaledError = table.equals("reference") ? error : Math.hypot(error, linkError);
      Assert.assertEquals(Double.parseDouble(row.get("scaled")), scaled, 1e-6, row.toString());
      Assert.assertEquals(Double.parseDouble(row.get("scaledError")), scaledError, 1e-6, row.toString());
    }
    // バージョンの順。同じバージョンでは 1 つ目の基準が先
    Assert.assertEquals(
        order,
        List.of("2reference", "3reference", "4reference", "4second", "5reference", "5second", "6reference", "6second"));
  }

  @Test(timeOut = 600_000)
  public void theBaselineReportCountsTheJarSegments() throws Exception {
    PrismSettings settings =
        lineage(4, concat(table("reference", "version:1", "FIXED", 30, "FIXED", 1, 2, false)));
    for (int version = 2; version <= 4; version++) {
      measure(settings, version, SyntheticDuels::noChampion, truth());
    }
    // v3 の計画を別の jar で決めたことにする: 区切りは v2 が 0、v3 が 1、v4 は jar が戻るので 2
    JobFile done = jobs().read("v00003-reference");
    EvaluationPlan plan = RecordFile.read(EvaluationPlan.class, done.plan()).value();
    DurableFiles.write(
        run.job(done.name()).resolve(Jobs.JOB),
        new JobFile(
            done.name(), done.version(), done.table(), done.order(), done.row(), done.definition(),
            RecordFile.tree(new EvaluationPlan(plan.candidate(), plan.duel(), "elsewhere", plan.twinDiscard())), done.state(),
            done.attempts(), done.cleanStops(), done.walls(), done.gamesBytes(), done.looksBytes(), done.created(),
            done.reason()));
    settleAgain(settings, "v00004-reference");
    List<Map<String, String>> rows = rows(EvaluationReports.BASELINE);
    Assert.assertEquals(rows.stream().map(row -> row.get("segment")).toList(), List.of("0", "1", "2"));
    Assert.assertEquals(rows.get(1).get("jar"), "elsewhere");
    Assert.assertEquals(rows.get(0).get("jar"), plan.jar());
    Assert.assertEquals(rows.get(2).get("jar"), plan.jar());
  }

  @Test(timeOut = 600_000)
  public void theBaselineReportCountsTheTwinDiscardSegments() throws Exception {
    List<String> tables = table("reference", "version:1", "FIXED", 30, "FIXED", 1, 2, false);
    PrismSettings settings = lineage(4, concat(tables));
    measure(settings, 2, SyntheticDuels::noChampion, truth());
    // 設定の 2 行目で切り方を手出しに替える: 1 行目で作った v2 は 0、2 行目で作った v3・v4 は 1
    PrismSettings tedashi =
        ThreePlayerSettings.append(root, List.of(), concat(tables, List.of("behaviour.twinDiscard=TEDASHI")));
    for (int version = 3; version <= 4; version++) {
      measure(tedashi, 2, version, SyntheticDuels::noChampion, truth());
    }
    List<Map<String, String>> rows = rows(EvaluationReports.BASELINE);
    Assert.assertEquals(rows.stream().map(row -> row.get("segment")).toList(), List.of("0", "1", "1"));
    Assert.assertEquals(
        rows.stream().map(row -> row.get("twinDiscard")).toList(), List.of("TSUMOGIRI", "TEDASHI", "TEDASHI"));
  }
  // ratings.tsv

  @Test(timeOut = 600_000)
  public void theRatingsGiveTheTrueThetaWithinTheirErrors() throws Exception {
    PrismSettings settings = chain();
    for (int version = 2; version <= 6; version++) {
      measure(settings, version, SyntheticDuels::noChampion, truth());
    }
    // 辺は reference が 5 本（v2〜v6）、second が 3 本（v4〜v6）、ladder が 3 本（v3・v5・v6）
    List<String> lines = Files.readAllLines(run.report(EvaluationReports.RATINGS), StandardCharsets.UTF_8);
    Assert.assertTrue(lines.getFirst().contains("from 11 edges of [ladder, reference, second]"), lines.getFirst());
    Assert.assertTrue(lines.getFirst().contains("(left out: 0 edges not connected to it yet, 0 edges"), lines.getFirst());
    Assert.assertTrue(lines.getFirst().contains("against v00001"), lines.getFirst());
    List<Map<String, String>> rows = rows(EvaluationReports.RATINGS);
    Assert.assertEquals(
        rows.stream().map(row -> row.get("label")).toList(),
        List.of("v00001", "v00002", "v00003", "v00004", "v00005", "v00006"));
    for (Map<String, String> row : rows) {
      int version = Integer.parseInt(row.get("label").substring(1));
      double value = Double.parseDouble(row.get("theta"));
      double error = Double.parseDouble(row.get("error"));
      if (version == 1) {
        Assert.assertEquals(value, 0.0, 1e-12);
        continue;
      }
      Assert.assertTrue(error > 0 && error < 0.05, row.toString());
      Assert.assertEquals(value, theta(version), 3.5 * error, row.toString());
    }
  }

  // duel-sigma

  /** 済んだ王者との決闘（バージョン 2〜20）の、δ の標準誤差。牌山の数は、v8 は 1,024 で止まり、ほかは 2,048。 */
  private static double error(int version) {
    return 0.004 + 0.0003 * ((version * 7) % 19);
  }

  private static int walls(int version) {
    return version == 8 ? 1024 : 2048;
  }

  private String sigma(String... arguments) throws Exception {
    List<String> all = new ArrayList<>(List.of("duel-sigma", "--run", root.toString()));
    all.addAll(List.of(arguments));
    PrintStream saved = System.out;
    ByteArrayOutputStream captured = new ByteArrayOutputStream();
    int exit;
    System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8));
    try {
      exit = Main.run(all.toArray(String[]::new));
    } finally {
      System.setOut(saved);
    }
    String out = captured.toString(StandardCharsets.UTF_8);
    Assert.assertEquals(exit, Exit.OK.code(), out);
    return out;
  }

  @Test(timeOut = 600_000)
  public void duelSigmaGivesTheMedianSigmaAndTheSuggestedSizeByHand() throws Exception {
    PrismSettings settings = lineage(20, ThreePlayerFrozenLadderTest.championTable().toArray(String[]::new));
    Champion first = Champion.of(EvaluationJobs.label(1), run.snapshot(1), 1);
    Assert.assertTrue(ChampionStore.open(run.champion()).append(new StoreEvent.Initial(first)));
    for (int version = 2; version <= 20; version++) {
      measure(
          settings,
          version,
          job -> SyntheticDuels.decided(Verdict.EVEN, walls(job.version()), error(job.version())),
          (job, wall, game) -> 0);
    }

    // 手計算: 直近 16 回（v5〜v20）の σ = SE × √牌山の数 の中央値（偶数個なので、真ん中の 2 つの平均）
    double[] sigmas = new double[16];
    for (int version = 5; version <= 20; version++) {
      sigmas[version - 5] = error(version) * Math.sqrt(walls(version));
    }
    Arrays.sort(sigmas);
    double median = (sigmas[7] + sigmas[8]) / 2;
    // 王者の卓は 1v2 と 2v1 で牌山 1 つにつき 6 半荘を打ち、512 牌山ごとに判定する。25,000 半荘以内なら判定は 8 回（4,096 牌山、24,576 半荘）
    Sequential sequential = new Sequential(0.01, 3.0, 512);
    double[] critical = sequential.boundaries(4096);
    double margin = (critical[critical.length - 1] + 1.6448536269514722) * median / (1.5 * Math.sqrt(4096));
    double rounded = Math.ceil(margin / 1e-4) * 1e-4;
    String out = sigma();
    Assert.assertTrue(
        out.contains(String.format(Locale.ROOT, "median sigma %.4f over 16 duels (v00005-champion .. v00020-champion)", median)),
        out);
    Assert.assertTrue(
        out.contains(String.format(Locale.ROOT, "suggested within 25000 games: walls=4096 games=24576 evenMargin=%.4f", rounded)),
        out);
    Assert.assertEquals(DuelSizing.choose(sequential, median, 6, 25_000).margin(), margin, 1e-9);
    // 切り上げた幅でも、真の差が幅の 1.5 倍なら 0.95 以上の確率で見つける
    String found = out.lines().filter(line -> line.startsWith("suggested")).findFirst().orElseThrow();
    double power = Double.parseDouble(found.substring(found.lastIndexOf(' ') + 1));
    Assert.assertTrue(power >= 0.95, found);
    Assert.assertFalse(out.contains("v00004-champion"), "only the last 16 duels");

    // 直近 4 回（v17〜v20）と、12,288 半荘の上限（判定 4 回、2,048 牌山）
    double[] four = new double[4];
    for (int version = 17; version <= 20; version++) {
      four[version - 17] = error(version) * Math.sqrt(walls(version));
    }
    Arrays.sort(four);
    double fourMedian = (four[1] + four[2]) / 2;
    double[] fewer = sequential.boundaries(2048);
    double fourMargin = (fewer[fewer.length - 1] + 1.6448536269514722) * fourMedian / (1.5 * Math.sqrt(2048));
    String short4 = sigma("--duels", "4", "--games", "12288");
    Assert.assertTrue(
        short4.contains(
            String.format(
                Locale.ROOT,
                "suggested within 12288 games: walls=2048 games=12288 evenMargin=%.4f",
                Math.ceil(fourMargin / 1e-4) * 1e-4)),
        short4);
    Assert.assertTrue(short4.contains(String.format(Locale.ROOT, "median sigma %.4f over 4 duels", fourMedian)), short4);

    // 王者との決闘が無い卓を指定するのは、使い方の誤り
    Assert.assertEquals(
        Main.run(new String[] {"duel-sigma", "--run", root.toString(), "--table", "nothing"}), Exit.USAGE.code());
  }
}
