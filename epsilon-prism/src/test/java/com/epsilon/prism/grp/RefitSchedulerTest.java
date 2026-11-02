package com.epsilon.prism.grp;

import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.testng.Assert;
import org.testng.annotations.Test;

public class RefitSchedulerTest {
  private static final String[] SCHEDULED = {
    "phiSource", "SELF_GRP", "grpCheckpoint", "/grp", "grpRefit", "SCHEDULE", "grpSwitch", "AUTO"
  };
  private static final String[] DRIFT = {
    "phiSource", "SELF_GRP", "grpCheckpoint", "/grp", "grpRefit", "SCHEDULE_AND_DRIFT",
    "phiDrift", "REFIT", "grpSwitch", "AUTO"
  };
  private static final PhiEntry GRP =
      new PhiEntry("g000", PhiEntryKind.FIXED_CHECKPOINT, "/grp/a", "sha", "", "", 1, "", "g000");
  private static final PhiEntry SCORE =
      new PhiEntry("g000", PhiEntryKind.SCORE_RANK, "", "", "", "", 1, "", "g000");

  static PrismConfig config(String... pairs) {
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put(PrismConfig.SETTINGS + pairs[i], pairs[i + 1]);
    PrismConfig config = PrismConfig.fromSettings(EpsilonSettings.of(values));
    config.validate();
    return config;
  }

  private static RefitScheduler.Inputs inputs(
      int h, PhiEntry active, boolean driftOn, double absorbed, RefitScheduler.Attempt... attempts) {
    return new RefitScheduler.Inputs(
        h, List.of(attempts), active, false, driftOn, absorbed, 1L << 20, true);
  }

  private static RefitScheduler.Attempt attempt(int holdout, RefitStatus status) {
    return new RefitScheduler.Attempt("g" + holdout + "-a1", holdout, status);
  }

  private static Optional<RefitScheduler.Decision> decision(int holdout, RefitCause cause) {
    return Optional.of(new RefitScheduler.Decision(holdout, cause));
  }

  @Test
  public void absorptionFollowsTheActualGains() {
    Map<Integer, Double> gains = new LinkedHashMap<>();
    for (int t = 300; t < 308; t++) gains.put(t, 0.06);
    Assert.assertEquals(RefitScheduler.absorbed(gains, 300, 307), 1 - Math.pow(0.94, 8), 1e-12);
    // 記録のない反復は 0 と数える
    Assert.assertEquals(RefitScheduler.absorbed(gains, 300, 310), 1 - Math.pow(0.94, 8), 1e-12);
    Assert.assertEquals(RefitScheduler.absorbed(gains, 310, 309), 0, 0);
  }

  @Test
  public void attemptsKeepTheLastStatusPerId() {
    RefitWindow window = new RefitWindow(92, 99, 100);
    List<RefitRecord> records = new ArrayList<>();
    records.add(record("g100-a1", RefitStatus.READY, window));
    records.add(record("g108-a1", RefitStatus.REJECTED, new RefitWindow(100, 107, 108)));
    records.add(record("g100-a1", RefitStatus.SKIPPED_NOT_READY, window));
    List<RefitScheduler.Attempt> attempts = RefitScheduler.attempts(records);
    Assert.assertEquals(
        attempts,
        List.of(
            new RefitScheduler.Attempt("g100-a1", 100, RefitStatus.SKIPPED_NOT_READY),
            new RefitScheduler.Attempt("g108-a1", 108, RefitStatus.REJECTED)));
    Assert.assertEquals(RefitScheduler.failureStreak(attempts), 1);
    Assert.assertEquals(
        RefitScheduler.failureStreak(
            List.of(
                attempt(1, RefitStatus.FAILED),
                attempt(2, RefitStatus.READY),
                attempt(3, RefitStatus.TIMEOUT),
                attempt(4, RefitStatus.REJECTED))),
        2);
  }

  @Test
  public void scheduleFollowsTheIntervalOfTheHoldout() {
    PrismConfig config = config(SCHEDULED);
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(10, GRP, false, 1)), decision(10, RefitCause.SCHEDULE));
    // 反復 40 以降は 8 ごと
    RefitScheduler.Attempt ready = attempt(100, RefitStatus.READY);
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(107, GRP, false, 1, ready)), Optional.empty());
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(108, GRP, false, 1, ready)), decision(108, RefitCause.SCHEDULE));
    // 反復 120 以降は 48 ごと
    RefitScheduler.Attempt late = attempt(500, RefitStatus.READY);
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(547, GRP, false, 1, late)), Optional.empty());
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(548, GRP, false, 1, late)), decision(548, RefitCause.SCHEDULE));
    // 同じ取り置きでは二度学ばない、取り置きの記録がなければ学ばない
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(100, GRP, false, 1, ready)), Optional.empty());
    var unrecorded = new RefitScheduler.Inputs(10, List.of(), GRP, false, false, 1, 1L << 20, false);
    Assert.assertEquals(RefitScheduler.schedule(config, unrecorded), Optional.empty());
    // 窓のない始めのコホート
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(3, GRP, false, 1)), Optional.empty());
    // 手動だけの設定では予定表は動かない
    Assert.assertEquals(
        RefitScheduler.schedule(config("phiSource", "SELF_GRP", "grpCheckpoint", "/grp"), inputs(10, GRP, false, 1)),
        Optional.empty());
  }

  @Test
  public void failuresWaitForTheCooldownAndStopTheSchedule() {
    PrismConfig config = config(DRIFT);
    RefitScheduler.Attempt failed = attempt(100, RefitStatus.REJECTED);
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(107, GRP, true, 1, failed)), Optional.empty());
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(108, GRP, true, 1, failed)), decision(108, RefitCause.SCHEDULE));
    RefitScheduler.Attempt[] three = {
      attempt(100, RefitStatus.REJECTED), attempt(108, RefitStatus.TIMEOUT), attempt(116, RefitStatus.FAILED)
    };
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(300, GRP, true, 1, three)), Optional.empty());
    PrismConfig noCooldown = config(with(SCHEDULED, "grpRetryCooldown", "0"));
    Assert.assertEquals(
        RefitScheduler.schedule(noCooldown, inputs(101, GRP, false, 1, failed)), decision(101, RefitCause.SCHEDULE));
  }

  @Test
  public void driftBringsTheRefitForwardOnlyWhenSettled() {
    PrismConfig config = config(DRIFT);
    RefitScheduler.Attempt ready = attempt(100, RefitStatus.READY);
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(103, GRP, true, 0.7, ready)), decision(103, RefitCause.DRIFT));
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(103, GRP, true, 0.5, ready)), Optional.empty());
    Assert.assertEquals(RefitScheduler.schedule(config, inputs(103, GRP, false, 0.7, ready)), Optional.empty());
    var pending = new RefitScheduler.Inputs(103, List.of(ready), GRP, true, true, 0.7, 1L << 20, true);
    Assert.assertEquals(RefitScheduler.schedule(config, pending), Optional.empty());
    // SCHEDULE だけなら警報は前倒ししない
    Assert.assertEquals(RefitScheduler.schedule(config(SCHEDULED), inputs(103, GRP, true, 0.7, ready)), Optional.empty());
  }

  @Test
  public void aScoreStartWaitsForTheTrustGames() {
    PrismConfig config =
        config("phiSource", "SELF_GRP", "phiStart", "SCORE_RANK", "grpRefit", "SCHEDULE", "grpSwitch", "AUTO");
    var few = new RefitScheduler.Inputs(20, List.of(), SCORE, false, true, 1, 131071, true);
    Assert.assertEquals(RefitScheduler.schedule(config, few), Optional.empty());
    var enough = new RefitScheduler.Inputs(20, List.of(), SCORE, false, true, 1, 131072, true);
    Assert.assertEquals(RefitScheduler.schedule(config, enough), decision(20, RefitCause.SCHEDULE));
  }

  @Test
  public void manualRequestsAreConsumedOnceTheirHoldoutIsSealed() throws IOException {
    Path run = Files.createTempDirectory("refit-request");
    GrpFiles files = new GrpFiles(run);
    Files.createDirectories(files.directory());
    PrismConfig manual = config("phiSource", "SELF_GRP", "grpCheckpoint", "/grp");
    Files.writeString(files.refitRequest(), "{\"holdout\": 9}", StandardCharsets.UTF_8);
    Assert.assertEquals(RefitScheduler.decide(files, manual, 7), Optional.empty());
    Assert.assertTrue(Files.exists(files.refitRequest()));
    Assert.assertEquals(RefitScheduler.decide(files, manual, 9), decision(9, RefitCause.MANUAL));
    Assert.assertFalse(Files.exists(files.refitRequest()));
    Assert.assertEquals(RefitScheduler.decide(files, manual, 9), Optional.empty());
    Files.writeString(files.refitRequest(), "{}", StandardCharsets.UTF_8);
    Assert.assertEquals(RefitScheduler.decide(files, manual, 12), decision(12, RefitCause.MANUAL));
  }

  @Test
  public void requestedHoldoutDefaultsToTheLatestCohort() {
    Assert.assertEquals(RefitScheduler.requestedHoldout("{\"holdout\": 12}", 30), 12);
    Assert.assertEquals(RefitScheduler.requestedHoldout("{\"holdout\": null}", 30), 30);
    Assert.assertEquals(RefitScheduler.requestedHoldout("{\"requested\": \"now\"}", 30), 30);
    Assert.assertEquals(RefitScheduler.requestedHoldout("not json {", 30), 30);
  }

  @Test
  public void gainsAreReadFromIterations() throws IOException {
    Path run = Files.createTempDirectory("gains");
    Files.createDirectories(run.resolve("reports"));
    Files.writeString(
        run.resolve("reports").resolve("iterations.jsonl"),
        "{\"iteration\":5,\"phi\":{\"id\":\"g000\"},\"filter\":{\"gain\":0.07,\"r2\":0.2}}\n"
            + "{\"iteration\":6,\"collection\":\"reused the sealed cohort\"}\n"
            + "{\"iteration\":7,\"filter\":{\"gain\":0.05}}\n"
            + "{\"iteration\":8,\"filt",
        StandardCharsets.UTF_8);
    Assert.assertEquals(RefitScheduler.gains(new GrpFiles(run)), Map.of(5, 0.07, 7, 0.05));
  }

  static RefitRecord record(String id, RefitStatus status, RefitWindow window) {
    return new RefitRecord(
        id, status, RefitCause.SCHEDULE, window, GrpInit.WARM, "g000", "", "", "jar", null, "", 1);
  }

  static String[] with(String[] pairs, String... more) {
    String[] result = new String[pairs.length + more.length];
    System.arraycopy(pairs, 0, result, 0, pairs.length);
    System.arraycopy(more, 0, result, pairs.length, more.length);
    return result;
  }
}
