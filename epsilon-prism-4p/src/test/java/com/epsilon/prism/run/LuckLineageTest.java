package com.epsilon.prism.run;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.prism.Main;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.app.FitCohorts;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 帳簿のモードごとに小さな系譜を回すテスト（CPU、Φ は点数の順位、1 コホート 4 半荘、収集はバージョン 1 の初期化したままのネットワーク、
 * 学習のステップ数 0、TEMPO なし）。
 *
 * <ul>
 *   <li>OFF は帳簿の列を 0 列にして最後まで回り、行は RECORD の行から帳簿を外したものと同じ。
 *   <li>AUDIT と APPLY は報告の全部を reports の ledger.jsonl に残し、段の報告の要約とノイズの表（noise.jsonl）はその中身と合う。
 *       fit-cohorts の LINEAGE は、同じコホートから同じ報告を書く。
 *   <li>APPLY で掛けた R² は state に持ち越され、補正に使う組を変えた次の反復で捨てられる。
 *   <li>生徒の行をラベル行だけにしても、帳簿を当てはめる系譜は回る。
 *   <li>裏ドラの監査の割合は行を変えず、収集の段の報告の ura だけを変える。
 * </ul>
 */
public class LuckLineageTest {

  /** 帳簿のモードを指定した系譜の設定。当てはめるモードでは、列 1 つに要る行の数を 1 にする（小さなコホートでも係数が付くように）。 */
  private static String[] lineage(String mode, String... more) {
    List<String> settings =
        new ArrayList<>(
            List.of(
                "phi.source=SCORE_RANK",
                "schedule.cohorts=[{from = 1, games = 4}]",
                "schedule.iterations=1",
                "fit.maxSteps=0",
                "noise.tempo.enabled=false",
                "noise.ledger.mode=" + mode));
    if (mode.equals("AUDIT") || mode.equals("APPLY")) {
      settings.add("noise.ledger.minRowsPerColumn=1");
    }
    settings.addAll(List.of(more));
    return TestSettings.cpuLineage(settings.toArray(String[]::new));
  }

  /** 新しい run を作って最後まで回す。 */
  private static Path finished(String prefix, String... settings) throws Exception {
    Path run = Files.createTempDirectory(prefix);
    TestSettings.create(run, settings);
    Assert.assertEquals(PrismRun.runner(run, null, () -> false).train(), Ending.FINISHED);
    return run;
  }

  @Test(groups = "slow", timeOut = 3_600_000)
  public void offRunsWithoutLedgerColumnsAndRecordOnlyAddsThem() throws Exception {
    Path off = null, record = null;
    try {
      off = finished("prism-luck-off", lineage("OFF"));
      record = finished("prism-luck-record", lineage("RECORD"));
      Cohort offCohort = cohort(off, 1), recordCohort = cohort(record, 1);
      Assert.assertEquals(offCohort.manifest().ledgerFeatures(), 0);
      Assert.assertEquals(recordCohort.manifest().ledgerFeatures(), LedgerColumn.COUNT);
      Assert.assertFalse(offCohort.rows().isEmpty());
      for (Row row : offCohort.rows()) Assert.assertFalse(row.hasLedger());
      for (Row row : recordCohort.rows()) Assert.assertTrue(row.hasLedger());
      Assert.assertEquals(stripped(recordCohort), stripped(offCohort));
      for (Path run : List.of(off, record)) {
        JsonObject learn = learn(run, 1);
        Assert.assertFalse(learn.has("ledger"), run.toString());
        Assert.assertEquals(learn.get("appliedR2").getAsDouble(), 0.0);
        Assert.assertFalse(Files.exists(new RunDirectory(run).report(LearnReport.LEDGER_LINES)), run.toString());
        JsonObject noise = lines(new RunDirectory(run).report(NoiseLine.FILE)).get(0);
        Assert.assertFalse(noise.has("maxAbsZ"), noise.toString());
        Assert.assertTrue(!noise.has("biased") || noise.getAsJsonArray("biased").size() == 0, noise.toString());
        Assert.assertEquals(carry(run).get("ledgerR2").getAsDouble(), 0.0);
      }
    } finally {
      delete(off);
      delete(record);
    }
  }

  /**
   * 裏ドラの監査の割合は行を変えない。割合 0 と 1 の系譜で、コホートの行（帳簿つき）はビットまで同じ。収集の段の報告の ura では、局末と
   * 立直の和了の数が同じ。割合 0 は期待値を求めず、割合 1 は全ての立直の和了で求める（基本点の組ごとのビンの数の和も、求めた数と同じ）。
   */
  @Test(groups = "slow", timeOut = 3_600_000)
  public void uraSampleOnlyChangesTheAudit() throws Exception {
    Path none = null, all = null;
    try {
      none = finished("prism-ura-none", lineage("RECORD", "noise.ura.sample=0"));
      all = finished("prism-ura-all", lineage("RECORD", "noise.ura.sample=1"));
      Assert.assertEquals(encoded(cohort(all, 1)), encoded(cohort(none, 1)));
      JsonObject off = collect(none, 1).getAsJsonObject("ura"), on = collect(all, 1).getAsJsonObject("ura");
      Assert.assertTrue(off.get("handEnds").getAsLong() > 0, off.toString());
      Assert.assertEquals(on.get("handEnds"), off.get("handEnds"));
      Assert.assertEquals(on.get("riichiWins"), off.get("riichiWins"));
      Assert.assertEquals(off.get("averaged").getAsLong(), 0);
      Assert.assertEquals(off.getAsJsonArray("cells").size(), 0);
      Assert.assertEquals(on.get("averaged"), on.get("riichiWins"));
      long binned = 0;
      for (JsonElement bin : on.getAsJsonArray("groups")) binned += bin.getAsLong();
      Assert.assertEquals(binned, on.get("averaged").getAsLong());
      Assert.assertEquals(on.getAsJsonArray("groups").size(), off.getAsJsonArray("groups").size());
    } finally {
      delete(none);
      delete(all);
    }
  }

  /**
   * AUDIT は収集のライターが作った和で当てはめ、報告の全部を ledger.jsonl に 1 行残す。段の報告の要約（行・R²・z の絶対値の最大・
   * しきい値 3 を越えた列）とノイズの表はその中身と合い、掛けた R² と持ち越しは 0。fit-cohorts の LINEAGE は行を読み直して、秒数の
   * ほかは同じ報告を書く。fit-cohorts は --ledger-r2 を受け付けず、--applied-r2 を受け付ける。
   */
  @Test(groups = "slow", timeOut = 3_600_000)
  public void auditReportsTheFullLedgerAndFitCohortsRefitsTheSame() throws Exception {
    Path run = null, out = null;
    try {
      run = finished("prism-luck-audit", lineage("AUDIT"));
      out = Files.createTempDirectory("prism-luck-refit");
      JsonObject learn = learn(run, 1);
      Assert.assertEquals(learn.get("appliedR2").getAsDouble(), 0.0);
      JsonObject full = ledgerLine(run, 1);
      Assert.assertTrue(full.get("rows").getAsLong() > 0, full.toString());
      assertSummaryMatches(learn.getAsJsonObject("ledger"), full, 3.0);
      JsonObject noise = lines(new RunDirectory(run).report(NoiseLine.FILE)).get(0);
      Assert.assertEquals(noise.get("maxAbsZ"), learn.getAsJsonObject("ledger").get("maxAbsZ"));
      Assert.assertEquals(noise.get("biased"), learn.getAsJsonObject("ledger").get("biased"));
      Assert.assertEquals(carry(run).get("ledgerR2").getAsDouble(), 0.0);

      Path refit = out.resolve("refit");
      Assert.assertEquals(Main.run(fitCohorts(run, refit, 1, "--ledger-r2", "0.1")), Exit.USAGE.code());
      Assert.assertEquals(Main.run(fitCohorts(run, refit, 1, "--applied-r2", "0")), Exit.OK.code());
      Assert.assertEquals(refitLedger(refit, 1), withoutSeconds(full));
    } finally {
      delete(run);
      delete(out);
    }
  }

  /**
   * APPLY の反復 1 では、掛けた R² が報告の全体の R²（負か、無ければ 0）になり、state の持ち越しも同じ値になる。補正に使う組を POINTS と
   * POLICY に変えた反復 2 では、持ち越しを捨てた理由に補正の組を書き、K′ の元にする R² は 0。反復 2 の報告では、ほかの組の列の係数は 0。
   * fit-cohorts の LINEAGE は、最新の設定でコホート 2 から同じ報告を書く。
   */
  @Test(groups = "slow", timeOut = 3_600_000)
  public void applyCarriesTheAppliedR2AndDropsItWhenTheLedgerChanges() throws Exception {
    Path run = null, out = null;
    try {
      run = Files.createTempDirectory("prism-luck-apply");
      TestSettings.create(run, lineage("APPLY", "schedule.iterations=2"));
      Assert.assertEquals(PrismRun.runner(run, 1, () -> false).train(), Ending.LIMIT);
      JsonObject first = learn(run, 1);
      JsonObject full = ledgerLine(run, 1);
      JsonElement r2 = full.getAsJsonObject("all").get("r2");
      double expected = r2 == null ? 0 : Math.max(0, r2.getAsDouble());
      Assert.assertEquals(first.get("appliedR2").getAsDouble(), expected);
      Assert.assertEquals(carry(run).get("ledgerR2").getAsDouble(), expected);
      Assert.assertTrue(first.getAsJsonObject("ledger").get("streamed").getAsBoolean());
      Assert.assertFalse(first.has("carryReset"), first.toString());

      Assert.assertEquals(change(run, "noise.ledger.apply=[POINTS, POLICY]"), Exit.OK.code());
      Assert.assertEquals(PrismRun.runner(run, null, () -> false).train(), Ending.FINISHED);
      JsonObject second = learn(run, 2);
      Assert.assertTrue(second.has("carryReset"), second.toString());
      Assert.assertTrue(second.get("carryReset").getAsString().startsWith("the ledger apply changed"), second.toString());
      Assert.assertEquals(second.getAsJsonObject("tempo").get("r2").getAsDouble(), 0.0);
      JsonObject later = ledgerLine(run, 2);
      JsonObject beta = later.getAsJsonObject("all").getAsJsonObject("beta");
      Set<String> applied = Set.of("POINTS", "POLICY");
      for (LedgerColumn column : LedgerColumn.values())
        if (!applied.contains(column.group().name()))
          Assert.assertEquals(beta.get(column.reportName()).getAsDouble(), 0.0, 0.0, column.reportName());

      out = Files.createTempDirectory("prism-luck-refit");
      Path refit = out.resolve("refit");
      Assert.assertEquals(Main.run(fitCohorts(run, refit, 2)), Exit.OK.code());
      Assert.assertEquals(refitLedger(refit, 2), withoutSeconds(later));
    } finally {
      delete(run);
      delete(out);
    }
  }

  /**
   * 生徒の行をラベル行だけ（fit.rows = LABELS）にした AUDIT の系譜も最後まで回り、帳簿はラベル行だけで当てはめる。先に、コホートに
   * 学習用の根の行（測った手が 2 つ以上）があることを確かめる。根の行は、ロールインの率を上げて作る。
   */
  @Test(groups = "slow", timeOut = 3_600_000)
  public void labelStudentRowsSettleTheLedger() throws Exception {
    Path run = null;
    try {
      run = Files.createTempDirectory("prism-luck-labels");
      String[] rollIn = {
        "labels.rollin.THREAT=0.2",
        "labels.rollin.RIICHI=0.2",
        "labels.rollin.CALL=0.2",
        "labels.rollin.KAN=0.2",
        "labels.rollin.END=0.2",
        "labels.rollin.FINAL_END=0.2"
      };
      TestSettings.create(run, TestSettings.concat(lineage("AUDIT", "fit.rows=LABELS"), rollIn));
      Ending ending = null;
      Exception failure = null;
      try {
        ending = PrismRun.runner(run, null, () -> false).train();
      } catch (Exception thrown) {
        failure = thrown;
      }
      long roots = 0, labels = 0;
      for (Row row : cohort(run, 1).rows()) {
        if (row.purpose() != Row.Purpose.TRAIN || measured(row) < 2) continue;
        if (row.source() == Row.Source.ROOT) roots++;
        else labels++;
      }
      Assert.assertTrue(roots > 0, "no training roots to tell the selections apart");
      Assert.assertNull(failure, "the lineage failed: " + failure);
      Assert.assertEquals(ending, Ending.FINISHED);
      JsonObject learn = learn(run, 1);
      Assert.assertEquals(learn.get("rows").getAsLong(), labels);
      JsonObject summary = learn.getAsJsonObject("ledger");
      Assert.assertEquals(summary.get("rows").getAsLong() + summary.get("skippedRows").getAsLong(), labels);
    } finally {
      delete(run);
    }
  }

  /** 段の報告の要約が、報告の全部の行・R²・z と合うことを確かめる。 */
  private static void assertSummaryMatches(JsonObject summary, JsonObject full, double biasZ) {
    Assert.assertTrue(summary.get("streamed").getAsBoolean(), summary.toString());
    Assert.assertEquals(summary.get("rows"), full.get("rows"));
    Assert.assertEquals(summary.get("skippedRows"), full.get("skippedRows"));
    Assert.assertEquals(summary.get("r2"), full.getAsJsonObject("all").get("r2"));
    Assert.assertEquals(summary.get("r2ByGroup"), full.getAsJsonObject("all").get("r2ByGroup"));
    Double max = null;
    List<String> above = new ArrayList<>();
    for (JsonElement element : full.getAsJsonObject("bias").getAsJsonArray("all")) {
      JsonObject feature = element.getAsJsonObject();
      if (!feature.has("z")) continue;
      double z = Math.abs(feature.get("z").getAsDouble());
      if (max == null || z > max) max = z;
      if (z > biasZ) above.add(feature.get("name").getAsString());
    }
    if (max == null) Assert.assertFalse(summary.has("maxAbsZ"), summary.toString());
    else Assert.assertEquals(summary.get("maxAbsZ").getAsDouble(), max, 0.0);
    List<String> biased = new ArrayList<>();
    for (JsonElement name : summary.getAsJsonArray("biased")) biased.add(name.getAsString());
    Assert.assertEquals(biased, above);
  }

  private static String[] fitCohorts(Path run, Path out, int cohort, String... more) {
    List<String> arguments =
        new ArrayList<>(
            List.of(
                "fit-cohorts",
                "--run",
                run.toString(),
                "--cohorts",
                cohort + ".." + cohort,
                "--teacher",
                "LINEAGE",
                "--out",
                out.toString()));
    arguments.addAll(List.of(more));
    return arguments.toArray(String[]::new);
  }

  /** fit-cohorts が書いたコホートの帳簿の報告から、秒数を除いたもの。 */
  private static JsonObject refitLedger(Path out, int cohort) throws Exception {
    Path file = out.resolve(FitCohorts.LEDGER).resolve(String.format("%05d.json", cohort));
    Assert.assertTrue(Files.exists(file), file.toString());
    return withoutSeconds(JsonParser.parseString(Files.readString(file)).getAsJsonObject());
  }

  /** reports の ledger.jsonl から、指定した反復の行を反復の番号を外して返す。 */
  private static JsonObject ledgerLine(Path run, int t) throws Exception {
    for (JsonObject line : lines(new RunDirectory(run).report(LearnReport.LEDGER_LINES))) {
      if (line.get("iteration").getAsInt() != t) continue;
      JsonObject copy = line.deepCopy();
      copy.remove("iteration");
      return copy;
    }
    throw new AssertionError("no ledger line for iteration " + t);
  }

  private static JsonObject withoutSeconds(JsonObject report) {
    JsonObject copy = report.deepCopy();
    copy.remove("seconds");
    return copy;
  }

  private static int change(Path run, String... assignments) {
    List<String> arguments = new ArrayList<>(List.of("settings", "change", "--run", run.toString()));
    for (String assignment : assignments) {
      arguments.add("--change");
      arguments.add(assignment);
    }
    arguments.addAll(List.of("--reason", "test"));
    return Main.run(arguments.toArray(String[]::new));
  }

  /**
   * 確定したコホートのマニフェストと全ての行。
   *
   * @param manifest コホートのマニフェスト
   */
  private record Cohort(CohortWriter.Manifest manifest, List<Row> rows) {}

  private static Cohort cohort(Path run, int t) throws Exception {
    Path directory = CohortWriter.directory(run, t);
    CohortWriter.Manifest manifest = CohortWriter.readManifest(directory);
    RowCodec codec = new RowCodec(manifest.ledgerFeatures(), manifest.variant());
    List<Row> rows = new ArrayList<>();
    for (CohortWriter.Shard shard : manifest.shards()) {
      var frames = ByteBuffer.wrap(Files.readAllBytes(directory.resolve(shard.file())));
      while (frames.hasRemaining()) rows.add(codec.decode(frames));
    }
    Assert.assertEquals(rows.size(), manifest.rows());
    return new Cohort(manifest, rows);
  }

  /** 行を帳簿を外して直列化し、並べ替えた一覧。 */
  private static List<String> stripped(Cohort cohort) {
    RowCodec codec = new RowCodec(0, cohort.manifest().variant());
    List<String> result = new ArrayList<>();
    for (Row row : cohort.rows())
      result.add(
          Base64.getEncoder()
              .encodeToString(
                  codec.encode(
                      new Row(
                          row.gameId(),
                          row.decisionId(),
                          row.decisionClass(),
                          row.purpose(),
                          row.source(),
                          row.played(),
                          row.values(),
                          row.prior(),
                          row.support(),
                          row.deviation(),
                          row.deviationProbability(),
                          row.propensity(),
                          row.rootDecisionId(),
                          new float[0],
                          row.outcomes(),
                          row.oracle(),
                          row.observation()))));
    Collections.sort(result);
    return result;
  }

  private static int measured(Row row) {
    int count = 0;
    for (int a = 0; a < row.actions(); a++) if (row.measured(a)) count++;
    return count;
  }

  /** 行を帳簿つきのまま直列化し、並べ替えた一覧。 */
  private static List<String> encoded(Cohort cohort) {
    RowCodec codec = new RowCodec(cohort.manifest().ledgerFeatures(), cohort.manifest().variant());
    List<String> result = new ArrayList<>();
    for (Row row : cohort.rows()) result.add(Base64.getEncoder().encodeToString(codec.encode(row)));
    Collections.sort(result);
    return result;
  }

  /** 指定した反復の、学習の段の報告。 */
  private static JsonObject learn(Path run, int t) throws Exception {
    return step(run, t, PrismStep.LEARN);
  }

  /** 指定した反復の、収集の段の報告。 */
  private static JsonObject collect(Path run, int t) throws Exception {
    return step(run, t, PrismStep.COLLECT);
  }

  private static JsonObject step(Path run, int t, PrismStep step) throws Exception {
    for (JsonObject line : lines(new RunDirectory(run).report(RunDirectory.ITERATIONS)))
      if (line.get("iteration").getAsInt() == t) return line.getAsJsonObject("steps").getAsJsonObject(step.name());
    throw new AssertionError("no iteration " + t);
  }

  /** state.json の持ち越し。 */
  private static JsonObject carry(Path run) throws Exception {
    JsonObject state = JsonParser.parseString(Files.readString(run.resolve("state.json"))).getAsJsonObject();
    return state.getAsJsonObject("carry");
  }

  private static List<JsonObject> lines(Path path) throws Exception {
    List<JsonObject> result = new ArrayList<>();
    for (String line : Files.readAllLines(path, StandardCharsets.UTF_8))
      if (!line.isBlank()) result.add(JsonParser.parseString(line).getAsJsonObject());
    return result;
  }

  private static void delete(Path directory) throws Exception {
    if (directory != null) Snapshot.deleteTree(directory);
  }
}
