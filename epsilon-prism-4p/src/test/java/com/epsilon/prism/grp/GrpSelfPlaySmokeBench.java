package com.epsilon.prism.grp;

import com.epsilon.agent.run.Ending;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.run.PrismRun;
import com.epsilon.prism.run.Snapshot;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * GRP の自己対局のスモークテスト（CPU。{@code -Dprism.grp.smoke=true} のときだけ走る）。小さな系譜を、本物の収集・再学習・
 * コホートの区切りの手続きで回す。
 *
 * <ol>
 *   <li>点数の順位から始め、SCHEDULE・PROPOSE、信頼する半荘 1 で反復 1〜5 を回す。反復 t は区切り t − 1 を待つ（待った時間は
 *       iterations.jsonl の Φ の段の waitSeconds）。切り替えの後の監視は、区切りごとに 1 行を書く。M(n) はコホート n で学習し、
 *       n ＋ 1 で関門に通す（refits.jsonl では TRAINED の行の後に READY か REJECTED の行が続く）。
 *   <li>小さなコホートでは関門のしきい値に届かない。そこで関門に通した最後の候補を READY にして、grp-switch で次の反復から使う。
 *       その反復は候補の NETWORK で集め、記録のヘッダーも候補の項目になり、帳簿の R² は持ち越さない。最後に grp-switch で g000 に戻す。
 * </ol>
 */
public class GrpSelfPlaySmokeBench {
  private static final Logger LOG = LoggerFactory.getLogger(GrpSelfPlaySmokeBench.class);
  private static final String PROPERTY = "prism.grp.smoke";

  @Test(timeOut = 7_200_000)
  public void scoreRankStartRefitsOnTheSlotAndSwitchesToANetwork() throws Exception {
    if (!Boolean.getBoolean(PROPERTY)) throw new SkipException("set -D" + PROPERTY + "=true to run the GRP self-play smoke");
    Path run = Files.createTempDirectory("prism-grp-smoke");
    try {
      TestSettings.create(run, settings());
      Assert.assertEquals(PrismRun.runner(run, 5, () -> false).train(), Ending.LIMIT);
      var files = new GrpFiles(run);
      List<PhiEntry> history = PhiHistory.read(files);
      Assert.assertEquals(history.getFirst().id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(history.getFirst().kind(), PhiEntryKind.SCORE_RANK);
      List<JsonObject> iterations = iterations(run);
      for (JsonObject iteration : iterations) Assert.assertTrue(step(iteration, "PHI").has("waitSeconds"), iteration.toString());
      List<SwitchWatchRecord> watch = SwitchWatchRecord.readAll(files);
      Assert.assertTrue(watch.size() >= 4, watch.size() + " switch watch lines");
      for (int index = 0; index < watch.size(); index++) Assert.assertEquals(watch.get(index).cohort(), index + 1);
      List<RefitRecord> records = RefitRecord.readAll(files);
      line("refits %s", records.stream().map(record -> record.id() + ":" + record.status()).toList());
      RefitRecord trained = RefitRecord.latestAt(records, 3).orElseThrow();
      Assert.assertTrue(
          trained.status() == RefitStatus.READY || trained.status() == RefitStatus.REJECTED, trained.toString());
      Assert.assertNotNull(trained.gate());
      line("M(3) %s at c4: %s", trained.status(), trained.detail());

      // 関門に通した候補を READY にして、次の反復から使う
      if (trained.status() != RefitStatus.READY) trained.with(RefitStatus.READY, trained.gate(), "smoke", 0).append(files);
      Assert.assertEquals(PhiControlTest.prism("grp-switch", "--run", run.toString(), "--to", trained.id()), 0);
      PhiEntry switched = PhiHistory.read(files).getLast();
      Assert.assertEquals(switched.id(), trained.id());
      int t = switched.effectiveFrom();
      Assert.assertEquals(PrismRun.runner(run, 1, () -> false).train(), Ending.LIMIT);
      JsonObject used = iterations(run).get(t - 1);
      Assert.assertEquals(phiId(used), trained.id());
      Assert.assertTrue(step(used, "LEARN").has("carryReset"), "a score-only to network switch resets the ledger R²");
      Assert.assertEquals(GameRecordCodec.read(files.records(t)).phiId(), trained.id());
      // g000 に巻き戻す
      Assert.assertEquals(
          PhiControlTest.prism("grp-switch", "--run", run.toString(), "--to", PhiEntry.INITIAL_ID, "--reason", "smoke"), 0);
      Assert.assertEquals(PrismRun.runner(run, 1, () -> false).train(), Ending.LIMIT);
      Assert.assertEquals(phiId(iterations(run).get(t)), PhiEntry.INITIAL_ID);
      line(
          "switched to %s at %d and rolled back at %d; switch watch %d lines",
          trained.id(),
          t,
          t + 1,
          SwitchWatchRecord.readAll(files).size());
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /**
   * CPU の小さな系譜（{@link TestSettings#cpuLineage}）に重ねる上書き。1 コホートは 4 半荘で、自前の GRP を点数の順位から始め、
   * SCHEDULE・PROPOSE、信頼する半荘は 1 にする。確約のクラス（打牌の 3 クラス以外）のラベル行の率とロールインの率、候補の数も下げる。評価は回さない。
   */
  private static String[] settings() {
    return TestSettings.cpuLineage(
        "memory.scheme=EMA", "memory.from=2", "schedule.cohorts=[{from = 1, games = 4}]", "schedule.iterations=20",
        "phi.source=SELF_GRP",
        "phi.start.kind=SCORE_RANK", "phi.grp.monitor=SCORE", "phi.grp.refit=SCHEDULE", "phi.grp.switching=PROPOSE",
        "phi.grp.trustGames=1", "phi.grp.threads=1", "noise.rates.table.THREAT=0.005", "noise.rates.table.RIICHI=0.005",
        "noise.rates.table.CALL=0.005", "noise.rates.table.KAN=0.005", "noise.rates.table.END=0.005",
        "labels.maxCandidates=2", "labels.rollin.THREAT=0.005", "labels.rollin.RIICHI=0.005", "labels.rollin.CALL=0.005",
        "labels.rollin.KAN=0.005", "labels.rollin.END=0.005", "labels.rollin.FINAL_END=0.005");
  }

  private static String phiId(JsonObject iteration) {
    return step(iteration, "PHI").getAsJsonObject("value").getAsJsonObject("entry").get("id").getAsString();
  }

  /** 反復の報告から、指定した段の報告を返す。 */
  private static JsonObject step(JsonObject iteration, String step) {
    return iteration.getAsJsonObject("steps").getAsJsonObject(step);
  }

  private static List<JsonObject> iterations(Path run) throws IOException {
    List<JsonObject> result = new ArrayList<>();
    try (Stream<String> lines = Files.lines(run.resolve("reports/iterations.jsonl"), StandardCharsets.UTF_8)) {
      for (String line : lines.toList())
        if (!line.isBlank()) result.add(JsonParser.parseString(line).getAsJsonObject());
    }
    return result;
  }

  private static void line(String format, Object... arguments) {
    LOG.info("PRISM-GRP-SMOKE " + String.format(format, arguments));
  }
}
