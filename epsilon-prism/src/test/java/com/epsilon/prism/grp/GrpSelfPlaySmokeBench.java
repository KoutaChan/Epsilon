package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Lineage;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.run.StopSignal;
import com.epsilon.workflow.WorkflowArguments;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * GRP の自己対局の煙（CPU。{@code -Dprism.grp.smoke=true} のときだけ走る）。小さな系譜（1 コホート 2 局、ラベル行は少なめ）を本物の収集・学習・境目の点検で回す。
 *
 * <ol>
 *   <li>GRP なしの出発（SELF_GRP＋phiStart = SCORE_RANK と SCORE_SMOOTH）: 信頼の半荘を 1 にして反復 1〜4 を集め、運用を予定表・AUTO に
 *       替えて裏の作業の順（予定表 → 学習）で白紙の GRP を学び、反復 5 の始めの AUTO の切り替えで入れる。反復 5 は学習の JVM が候補を境目の
 *       点検で確かめて読み、R² を持ち越さない。止めて再開した反復 6 も同じ候補で、R² を持ち越す。
 *   <li>手動の当てはめ直し（点数の順位の run の続き）: grp-refit の要求 → 温めの学習 → grp-switch で反復 7 から入れ、grp-switch の巻き戻しで
 *       反復 8 から前の候補に戻す。始まった反復への切り替えは拒む。
 *   <li>裏の作業（{@link GrpBackground}）の 1 回が 8 コホートを採点して phimon.jsonl と drift.jsonl を書き、締め切り 0 の学習は TIMEOUT を
 *       書くだけで phi.jsonl・state.json・行に触れず、反復 9 はそのまま回る。
 * </ol>
 *
 * 関門の数値の線はコードの定数で、4 局のコホートでは届かないので、学習の部品（{@link GrpRefitter.Steps}）の関門だけを合格に差し替える（本物の
 * 関門の結果は {@code PRISM-GRP-SMOKE} の行に書く）。そのため学習は煙が裏の作業に代わって呼び、学習の JVM の裏の作業には学ばせない（反復 1〜4
 * は MANUAL、予定表に替えた後は間隔 8 で反復 5・6 には時期が来ない）。学習・候補の公開・境目の見本・切り替え・学習の JVM の読み込みは本番の道。
 */
public class GrpSelfPlaySmokeBench {
  private static final Logger LOG = LoggerFactory.getLogger(GrpSelfPlaySmokeBench.class);
  private static final String PROPERTY = "prism.grp.smoke";

  @Test(timeOut = 7_200_000)
  public void scoreRankStartSwitchesRefitsRollsBackAndScores() throws Exception {
    requireProperty();
    Path run = Files.createTempDirectory("prism-grp-smoke-rank");
    try {
      RefitRecord grown = grownSwitch(run, "SCORE_RANK", PhiEntryKind.SCORE_RANK);
      manualRoundTrip(run, grown);
      backgroundAndTimeout(run, grown);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test(timeOut = 7_200_000)
  public void scoreSmoothStartSwitchesToAGrownGrp() throws Exception {
    requireProperty();
    Path run = Files.createTempDirectory("prism-grp-smoke-smooth");
    try {
      grownSwitch(run, "SCORE_SMOOTH", PhiEntryKind.SCORE_SMOOTH);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /** 点数だけの Φ で反復 1〜4、運用を予定表・AUTO に替えて白紙の GRP を学び、反復 5 の始めに入れ、止めて反復 6 を再開する。 */
  private static RefitRecord grownSwitch(Path run, String start, PhiEntryKind kind) throws Exception {
    Assert.assertEquals(Lineage.run(run, 4, EpsilonSettings.of(settings(start)), StopSignal.none()), 0);
    var files = new GrpFiles(run);
    List<PhiEntry> history = PhiHistory.read(files);
    Assert.assertEquals(history.size(), 1);
    Assert.assertEquals(history.getFirst().id(), PhiEntry.INITIAL_ID);
    Assert.assertEquals(history.getFirst().kind(), kind);
    for (int t = 1; t <= 4; t++) {
      Assert.assertEquals(GameLog.read(files.games(t)).header().phiId(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(phiId(iterations(run).get(t - 1)), PhiEntry.INITIAL_ID);
    }
    Assert.assertTrue(RefitRecord.readAll(files).isEmpty(), "the training JVM does not train while MANUAL");
    rewriteConfig(run, Map.of("grpRefit", "SCHEDULE", "grpSwitch", "AUTO"));
    PrismConfig config = config(run);
    int sealed = GrpBackground.lastSealed(files);
    Assert.assertEquals(sealed, 4);
    Assert.assertTrue(AutoSwitch.apply(files, config, 5).isEmpty(), "no candidate yet");
    RefitScheduler.Decision decision = RefitScheduler.decide(files, config, sealed).orElseThrow();
    Assert.assertEquals(decision, new RefitScheduler.Decision(4, RefitCause.SCHEDULE));
    RefitRecord grown =
        GrpRefitter.refit(files, config, 4, decision.cause(), () -> false, new Forced(null)).orElseThrow();
    line("%s grown %s %s init %s parent %s window %s %.1fs", start, grown.id(), grown.status(), grown.init(),
        grown.parent(), grown.window(), grown.seconds());
    Assert.assertEquals(grown.status(), RefitStatus.READY, grown.detail());
    Assert.assertEquals(grown.init(), GrpInit.SCRATCH);
    Assert.assertEquals(grown.parent(), PhiEntry.INITIAL_ID);
    Assert.assertTrue(Files.exists(files.boundary(grown.id())));
    // 反復 5 の始め（学習の JVM が Φ を決める前）の AUTO の切り替え
    PhiEntry entry = AutoSwitch.apply(files, config, 5).orElseThrow();
    Assert.assertEquals(entry.id(), grown.id());
    Assert.assertEquals(entry.kind(), PhiEntryKind.SELF_CHECKPOINT);
    Assert.assertEquals(entry.effectiveFrom(), 5);
    Assert.assertEquals(entry.reason(), "AUTO: " + grown.id() + " at iteration 5");
    // 反復 5: 学習の JVM が候補を境目の点検で確かめて読む（点数だけの Φ から GRP に変わるので R² を持ち越さない）
    Assert.assertEquals(Lineage.run(run, 1, EpsilonSettings.defaults(), StopSignal.none()), 0);
    JsonObject fifth = iterations(run).get(4);
    Assert.assertEquals(phiId(fifth), grown.id());
    Assert.assertTrue(fifth.has("ledgerR2Reset"), "iteration 5 resets the ledger R²");
    GameLogHeader header = GameLog.read(files.games(5)).header();
    Assert.assertEquals(header.phiId(), grown.id());
    Assert.assertEquals(header.phiSha256(), grown.sha256());
    // 再開: 同じ候補のまま、R² は持ち越す
    Assert.assertEquals(Lineage.run(run, 1, EpsilonSettings.defaults(), StopSignal.none()), 0);
    JsonObject sixth = iterations(run).get(5);
    Assert.assertEquals(phiId(sixth), grown.id());
    Assert.assertFalse(sixth.has("ledgerR2Reset"), "iteration 6 keeps the ledger R²");
    Assert.assertEquals(PhiHistory.read(files).size(), 2, "no second AUTO line and no fallback line");
    Assert.assertEquals(
        RefitRecord.readAll(files).stream().map(RefitRecord::status).toList(),
        List.of(RefitStatus.READY),
        "no boundary rejection and no refit by the training JVM before the interval");
    Assert.assertEquals(state(run).get("next").getAsInt(), 7);
    line("%s switched at 5 to %s (sha %s), ledgerR2Reset %s, resumed at 6 with the same entry", start, grown.id(),
        grown.sha256().substring(0, 16), fifth.get("ledgerR2Reset"));
    return grown;
  }

  /** grp-refit の要求 → 温めの学習 → grp-switch で反復 7、巻き戻しで反復 8。 */
  private static void manualRoundTrip(Path run, RefitRecord grown) throws Exception {
    rewriteConfig(run, Map.of("grpRefit", "MANUAL", "grpSwitch", "PROPOSE"));
    var files = new GrpFiles(run);
    PrismConfig config = config(run);
    Assert.assertEquals(new GrpRefitWorkflow().execute(arguments("--run", run.toString(), "--holdout", "6")), 0);
    Assert.assertTrue(Files.exists(files.refitRequest()));
    RefitScheduler.Decision decision =
        RefitScheduler.decide(files, config, GrpBackground.lastSealed(files)).orElseThrow();
    Assert.assertEquals(decision, new RefitScheduler.Decision(6, RefitCause.MANUAL));
    Assert.assertFalse(Files.exists(files.refitRequest()), "the request is consumed");
    RefitRecord warm =
        GrpRefitter.refit(files, config, 6, RefitCause.MANUAL, () -> false, new Forced(null)).orElseThrow();
    line("manual refit %s %s init %s parent %s %.1fs", warm.id(), warm.status(), warm.init(), warm.parent(),
        warm.seconds());
    Assert.assertEquals(warm.status(), RefitStatus.READY, warm.detail());
    Assert.assertEquals(warm.init(), GrpInit.WARM);
    Assert.assertEquals(warm.parent(), grown.id());
    Assert.assertTrue(AutoSwitch.apply(files, config, 7).isEmpty(), "PROPOSE switches nothing by itself");
    Assert.assertEquals(new GrpSwitchWorkflow().execute(arguments("--run", run.toString(), "--to", warm.id())), 0);
    PhiEntry switched = PhiHistory.read(files).getLast();
    Assert.assertEquals(switched.id(), warm.id());
    Assert.assertEquals(switched.effectiveFrom(), 7);
    Assert.assertEquals(Lineage.run(run, 1, EpsilonSettings.defaults(), StopSignal.none()), 0);
    JsonObject seventh = iterations(run).get(6);
    Assert.assertEquals(phiId(seventh), warm.id());
    Assert.assertFalse(seventh.has("ledgerR2Reset"), "a GRP to GRP switch keeps the ledger R²");
    Assert.assertEquals(GameLog.read(files.games(7)).header().phiId(), warm.id());
    // 巻き戻し: 前の候補の行を反復 8 から（同じ id、理由は rollback:）
    Assert.assertEquals(
        new GrpSwitchWorkflow()
            .execute(arguments("--run", run.toString(), "--to", grown.id(), "--reason", "smoke")),
        0);
    PhiEntry back = PhiHistory.read(files).getLast();
    Assert.assertEquals(back.id(), grown.id());
    Assert.assertEquals(back.effectiveFrom(), 8);
    Assert.assertEquals(back.reason(), "rollback: smoke");
    Assert.assertEquals(Lineage.run(run, 1, EpsilonSettings.defaults(), StopSignal.none()), 0);
    Assert.assertEquals(phiId(iterations(run).get(7)), grown.id());
    Assert.assertEquals(GameLog.read(files.games(8)).header().phiId(), grown.id());
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new GrpSwitchWorkflow().execute(arguments("--run", run.toString(), "--to", warm.id(), "--from", "8")));
    Assert.assertEquals(PhiHistory.read(files).size(), 4);
    line("manual round trip: %s from 7, rollback to %s from 8, phi.jsonl %d lines", warm.id(), grown.id(),
        PhiHistory.read(files).size());
  }

  /** 裏の作業の 1 回の採点と、締め切り 0 の学習の TIMEOUT。 */
  private static void backgroundAndTimeout(Path run, RefitRecord grown) throws Exception {
    rewriteConfig(run, Map.of("phiMonitor", "SCORE", "phiDrift", "REPORT"));
    var files = new GrpFiles(run);
    // 学習の JVM が封印の後に頼む 1 回をここで回す（MANUAL で要求がないので学ばない）
    try (var background = GrpBackground.start(files, config(run))) {
      background.pass();
    }
    List<PhiMonitorLine> lines = PhiMonitorLine.readAll(files);
    List<Integer> cohorts = lines.stream().map(PhiMonitorLine::cohort).toList();
    Assert.assertEquals(cohorts, List.of(1, 2, 3, 4, 5, 6, 7, 8));
    for (PhiMonitorLine scored : lines) {
      Assert.assertEquals(scored.phiId(), GameLog.read(files.games(scored.cohort())).header().phiId());
      Assert.assertTrue(scored.active().seatHands() > 0);
      line("%s", scored.summary());
    }
    Assert.assertTrue(lines.get(4).shadows().size() > 0, "READY candidates are scored as shadows");
    int drift = PhiHistory.lines(files.drift()).size();
    Assert.assertEquals(drift, 8);

    // 締め切り 0 の学習: TIMEOUT を書くだけで系譜に触れない
    byte[] phi = Files.readAllBytes(files.phiHistory());
    byte[] state = Files.readAllBytes(run.resolve("state.json"));
    RowHash.Digest rows = RowHash.of(CohortWriter.directory(run, 8));
    RefitRecord timedOut =
        GrpRefitter.refit(files, config(run), 8, RefitCause.MANUAL, () -> false, new Forced(Duration.ZERO))
            .orElseThrow();
    line("timeout refit %s %s: %s", timedOut.id(), timedOut.status(), timedOut.detail());
    Assert.assertEquals(timedOut.status(), RefitStatus.TIMEOUT);
    Assert.assertEquals(RefitRecord.readAll(files).getLast().status(), RefitStatus.TIMEOUT);
    Assert.assertEquals(Files.readAllBytes(files.phiHistory()), phi);
    Assert.assertEquals(Files.readAllBytes(run.resolve("state.json")), state);
    Assert.assertEquals(RowHash.of(CohortWriter.directory(run, 8)), rows);
    Assert.assertFalse(Files.exists(files.candidate(timedOut.id())));
    Assert.assertFalse(Files.exists(files.staging().resolve(timedOut.id())));
    Assert.assertFalse(Files.exists(files.boundary(timedOut.id())));
    Assert.assertEquals(Lineage.run(run, 1, EpsilonSettings.defaults(), StopSignal.none()), 0);
    Assert.assertEquals(phiId(iterations(run).get(8)), grown.id());
    Assert.assertEquals(state(run).get("next").getAsInt(), 10);
    line("background scored %d cohorts, drift %d lines; timeout left phi.jsonl, state.json and rows alone;"
        + " iteration 9 ran", lines.size(), drift);
  }

  /** 本番の学習の部品に、関門だけ合格を返し（本物の結果は書く）、締め切りを替えられるようにしたもの。 */
  private static final class Forced implements GrpRefitter.Steps {
    private static final GrpRefitter.Steps REAL = GrpRefitter.Steps.PRODUCTION;
    private final Duration deadline;

    /** @param deadline 学習の締め切り（null なら本番の計画のまま） */
    Forced(Duration deadline) {
      this.deadline = deadline;
    }

    @Override
    public CohortGames games(Path file) throws IOException {
      return REAL.games(file);
    }

    @Override
    public GrpTrainingPlan plan(GrpInit init, long seed) {
      GrpTrainingPlan plan = REAL.plan(init, seed);
      return deadline == null ? plan : plan.within(deadline);
    }

    @Override
    public GrpGameTrainer.Trained train(
        Path start,
        List<RecordedGame> training,
        List<RecordedGame> validation,
        GrpTrainingPlan plan,
        BooleanSupplier stop,
        Path out)
        throws IOException {
      return REAL.train(start, training, validation, plan, stop, out);
    }

    @Override
    public HandEndPhi open(PhiEntry entry, EpsilonUtilityProfile profile) throws IOException {
      return REAL.open(entry, profile);
    }

    @Override
    public boolean grpEpsilonV3(Path checkpoint) {
      return REAL.grpEpsilonV3(checkpoint);
    }

    @Override
    public String sha256(Path checkpoint) throws IOException {
      return REAL.sha256(checkpoint);
    }

    @Override
    public GateReport gate(GateInput input, EpsilonUtilityProfile profile) {
      try {
        GateReport report = REAL.gate(input, profile);
        line("real %s gate %s failures %s", input.mode(), report.result(), report.failures());
        return new GateReport(
            input.mode(), GateResult.PASS, List.of(), report.measures(), report.active(), report.candidate());
      } catch (IOException | RuntimeException failure) {
        line("real %s gate could not run on the tiny window: %s", input.mode(), failure);
        return new GateReport(input.mode(), GateResult.PASS, List.of(), Map.of(), null, null);
      }
    }
  }

  private static void requireProperty() {
    if (!Boolean.getBoolean(PROPERTY)) throw new SkipException("set -D" + PROPERTY + "=true to run the GRP self-play smoke");
  }

  /**
   * CPU の小さな系譜（{@code GameLogRowsTest} の形で、確約の類のラベル行・ロールインと候補集合を減らしたもの）を SELF_GRP・点数だけの出発・
   * MANUAL・PROPOSE・予定表「1:8」（運用を予定表に替えた後に効く）・窓 1・信頼の半荘 1 にしたもの。評価は回さない。
   */
  private static Map<String, String> settings(String start) {
    String[] pairs = {
      "phiSource", "SELF_GRP", "phiStart", start, "phiMonitor", "RECORD", "phiDrift", "OFF",
      "grpRefit", "MANUAL", "grpSwitch", "PROPOSE", "grpRefitSchedule", "1:8",
      "grpWindow", "1", "grpTrustGames", "1", "gamesPerCohort", "2",
      "gamesSchedule", "", "iterations", "20", "epsilonB", "0.005",
      "collectDevices", "cpu", "learnerDevice", "cpu", "maxActiveGames", "8", "advanceWorkers", "2",
      "encoders", "2", "slots", "1", "readyBatches", "2", "inferBatch", "256",
      "batchWaitMicros", "1000", "rowWriters", "1", "labelRateDiscardEarly", "0.003",
      "labelRateDiscardMid", "0.003", "labelRateDiscardLate", "0.003", "labelRateCommit", "0.005",
      "maxCandidates", "2", "rollinRateDiscard", "0.001", "rollinRateCommit", "0.005", "rhoBranch", "0.2",
      "averaging", "ema", "averageFrom", "2", "batchSize", "16", "decodeThreads", "2",
      "evaluation.3xH.every", "0"
    };
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put("epsilon.prism." + pairs[i], pairs[i + 1]);
    return values;
  }

  private static PrismConfig config(Path run) throws IOException {
    return PrismConfig.read(Files.readString(run.resolve("config.json"), StandardCharsets.UTF_8));
  }

  /** run の config.json の値を替える（反復の境目の運用の変更）。 */
  private static void rewriteConfig(Path run, Map<String, String> values) throws IOException {
    Path path = run.resolve("config.json");
    JsonObject config = JsonParser.parseString(Files.readString(path, StandardCharsets.UTF_8)).getAsJsonObject();
    values.forEach(config::addProperty);
    Files.writeString(path, config.toString(), StandardCharsets.UTF_8);
    config(run).validateTraining();
  }

  private static WorkflowArguments arguments(String... values) {
    return GrpRefitterTest.arguments(values);
  }

  private static String phiId(JsonObject iteration) {
    return iteration.getAsJsonObject("phi").get("id").getAsString();
  }

  private static JsonObject state(Path run) throws IOException {
    return JsonParser.parseString(Files.readString(run.resolve("state.json"))).getAsJsonObject();
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
