package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.workflow.WorkflowArguments;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * GRP の学習 1 回（偽の学習器と関門で）: 合格すれば試行の名前で staging → candidates へ rename し、見本と READY の行を書く。関門に落ちれば
 * REJECTED、学習器の失敗は FAILED、締め切りは TIMEOUT で、どれも候補を残さず系譜を止めない。止める合図で止まった試行は捨て、行も書かない。
 * 使用中が点数だけの Φ なら白紙から学び、同じ関門の最初の置き換えの列（FIRST_REPLACEMENT）を通す。窓は clamp(⌈h/4⌉, 4, 8)、試行の番号は
 * 続き、候補の形（順位分布の和）が崩れれば関門に知らせる。grp-refit は要求を書くだけ。
 */
public class GrpRefitterTest {
  private static final int NEXT = 10, HOLDOUT = 9;

  /** 偽の部品（呼ばれた引数を覚える）。 */
  private static final class FakeSteps implements GrpRefitter.Steps {
    GateResult result = GateResult.PASS;
    boolean failTraining, timeOut, stopped, malformed;
    Path start;
    List<RecordedGame> training, validation;
    BooleanSupplier stop;
    GateInput gateInput;
    int trained;

    @Override
    public CohortGames games(Path file) {
      int cohort = Integer.parseInt(file.getFileName().toString().substring(1, 6));
      return new CohortGames(
          new GameLogHeader(cohort, "g000", "", 0.005, 0, 0, 0), GrpFixtures.games(cohort * 1000L, 40, cohort));
    }

    @Override
    public GrpTrainingPlan plan(GrpInit init, long seed) {
      return new GrpTrainingPlan(init, 128, 2e-4, 3, seed, Duration.ofMinutes(10));
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
      trained++;
      this.start = start;
      this.training = training;
      this.validation = validation;
      this.stop = stop;
      Assert.assertFalse(Files.exists(out), "the trainer writes a new directory");
      if (failTraining) throw new IOException("trainer exploded");
      if (timeOut) return unfinished(GrpGameTrainer.Ending.TIMED_OUT);
      if (stopped) return unfinished(GrpGameTrainer.Ending.STOPPED);
      Files.createDirectories(out);
      Files.writeString(out.resolve("grp-0000.params"), "weights " + out.getFileName());
      return new GrpGameTrainer.Trained(out, sha256(out), 3, 30, 0.5, 0.6, 100, GrpGameTrainer.Ending.COMPLETED);
    }

    /** 締め切りか止める合図で止めた学習（本物と同じく何も書かない）。 */
    private static GrpGameTrainer.Trained unfinished(GrpGameTrainer.Ending ending) {
      return new GrpGameTrainer.Trained(null, "", 1, 3, 0.5, Double.NaN, 10, ending);
    }

    @Override
    public HandEndPhi open(PhiEntry entry, EpsilonUtilityProfile profile) {
      return inputs -> {
        double[][] utilities = new double[inputs.size()][4];
        double[][] ranks = new double[inputs.size()][16];
        for (int i = 0; i < inputs.size(); i++) {
          for (int seat = 0; seat < 4; seat++) utilities[i][seat] = 0.1 * seat - inputs.get(i).prefix().length * 1e-4;
          Arrays.fill(ranks[i], 0.25);
          if (malformed) ranks[i][0] = 0.3;
        }
        return new PhiValues(utilities, ranks);
      };
    }

    @Override
    public boolean grpEpsilonV3(Path checkpoint) {
      return true;
    }

    @Override
    public String sha256(Path checkpoint) throws IOException {
      return "sha-" + Files.readString(checkpoint.resolve("grp-0000.params"));
    }

    @Override
    public GateReport gate(GateInput input, EpsilonUtilityProfile profile) {
      gateInput = input;
      return report(input.mode(), input.candidateValid() ? result : GateResult.INVALID_OUTPUT);
    }
  }

  private static GateReport report(GateMode mode, GateResult result) {
    return new GateReport(
        mode,
        result,
        result == GateResult.PASS ? List.of() : List.of(result),
        Map.of("mseDifference", -1e-4, "eceShift", Double.NaN),
        new CellReport(10, 100, 0.1, 1e-4, 2e-4, new double[] {0.01, Double.NaN}, new double[] {1, 1}, new double[] {0.1, 0.1}, new double[] {0, 0}, 1.2, 0.3, 0.01),
        null);
  }

  /** 取り置き 9 までのコホート 1〜9 が封印済みで記録があり、g000 が start の run。 */
  private static GrpFiles run(PhiEntry start) throws IOException {
    Path run = Files.createTempDirectory("prism-refit");
    var files = new GrpFiles(run);
    GrpFixtures.writeState(run, NEXT);
    PhiHistory.write(files.phiHistory(), PhiHistory.LINES.toJson(start));
    for (int c = 1; c <= HOLDOUT; c++) {
      Files.createDirectories(files.cohort(c));
      Files.writeString(files.cohort(c).resolve(CohortWriter.MANIFEST), "{}");
      Files.createDirectories(files.games(c).getParent());
      Files.writeString(files.games(c), "");
    }
    return files;
  }

  private static PhiEntry grpStart() {
    return new PhiEntry(
        PhiEntry.INITIAL_ID, PhiEntryKind.FIXED_CHECKPOINT, "/human/grp/latest", "sha0", "", "/table.json", 1, "", "g000");
  }

  private static PhiEntry scoreStart() {
    return new PhiEntry(PhiEntry.INITIAL_ID, PhiEntryKind.SCORE_RANK, "", "", "", "", 1, "", "g000");
  }

  @Test
  public void aPassingWarmRefitPublishesAReadyCandidate() throws Exception {
    GrpFiles files = run(grpStart());
    try {
      var steps = new FakeSteps();
      PrismConfig config = GrpFixtures.grpStart(Path.of("/human"));
      RefitRecord record =
          GrpRefitter.refit(files, config, HOLDOUT, RefitCause.MANUAL, () -> false, steps).orElseThrow();
      Assert.assertEquals(record.status(), RefitStatus.READY, record.detail());
      Assert.assertEquals(record.id(), "g009-a1");
      Assert.assertEquals(record.window(), new RefitWindow(5, 8, 9));
      Assert.assertEquals(record.init(), GrpInit.WARM);
      Assert.assertEquals(record.parent(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(steps.start, Path.of("/human/grp/latest"));
      // 窓は 5..8（取り置き 9 は学習にも検証にも使わない）、半荘 ID を 1 つずらした値の 1/16 が検証
      Assert.assertEquals(steps.training.size() + steps.validation.size(), 4 * 40);
      for (RecordedGame game : steps.validation) Assert.assertEquals((game.gameId() >> 1) % 16, 15);
      for (RecordedGame game : steps.training) {
        Assert.assertNotEquals((game.gameId() >> 1) % 16, 15);
        Assert.assertTrue(game.gameId() >= 5000 && game.gameId() < 9000);
      }
      Assert.assertEquals(steps.gateInput.mode(), GateMode.REFIT);
      Assert.assertTrue(steps.gateInput.candidateValid());
      Assert.assertEquals(steps.gateInput.windowGames(), 160);
      Assert.assertEquals(steps.gateInput.holdout().size(), 40);
      // 公開: candidates に現れ、staging に残らず、見本と READY の行がある
      Path candidate = files.candidate("g009-a1");
      Assert.assertEquals(record.path(), candidate.toAbsolutePath().normalize().toString());
      Assert.assertEquals(record.sha256(), steps.sha256(candidate));
      Assert.assertFalse(Files.exists(files.staging().resolve("g009-a1")));
      BoundaryCheck boundary = BoundaryCheck.read(files.boundary("g009-a1"));
      Assert.assertTrue(boundary.inputs().size() > 0 && boundary.inputs().size() <= BoundaryCheck.SAMPLES);
      List<RefitRecord> records = RefitRecord.readAll(files);
      Assert.assertEquals(records.size(), 1);
      Assert.assertEquals(records.getFirst().status(), RefitStatus.READY);
      Assert.assertEquals(records.getFirst().gate().result(), GateResult.PASS);
      Assert.assertTrue(Double.isNaN(records.getFirst().gate().measures().get("eceShift")));
      Assert.assertTrue(Double.isNaN(records.getFirst().gate().active().b4()[1]));
      PhiEntry entry = records.getFirst().entry(12, "grp-switch");
      Assert.assertEquals(entry.kind(), PhiEntryKind.SELF_CHECKPOINT);
      Assert.assertEquals(entry.path(), record.path());
      Assert.assertEquals(entry.temperatures(), "");
      Assert.assertEquals(entry.calibration(), "");
      // 同じ取り置きの次の試行は次の番号
      Assert.assertEquals(
          GrpRefitter.refit(files, config, HOLDOUT, RefitCause.SCHEDULE, () -> false, steps).orElseThrow().id(),
          "g009-a2");
    } finally {
      Snapshot.deleteTree(files.run());
    }
  }

  /** 検証の分け方は偶奇に依らない: 偶数だけ・奇数だけの半分の窓でも、それぞれ約 1/16 が検証になる。 */
  @Test
  public void eachParityHalfKeepsAValidationShare() {
    List<RecordedGame> games = GrpFixtures.games(0, 3200, 7);
    for (int parity = 0; parity < 2; parity++) {
      int p = parity;
      GrpRefitter.Split split = GrpRefitter.Split.of(games.stream().filter(game -> game.gameId() % 2 == p).toList());
      Assert.assertEquals(split.training().size() + split.validation().size(), 1600);
      Assert.assertEquals(split.validation().size(), 100, "parity " + parity);
    }
  }

  @Test
  public void failedAttemptsLeaveNoCandidateAndDoNotThrow() throws Exception {
    GrpFiles files = run(grpStart());
    try {
      PrismConfig config = GrpFixtures.grpStart(Path.of("/human"));
      var rejected = new FakeSteps();
      rejected.result = GateResult.NOT_BETTER;
      RefitRecord gate =
          GrpRefitter.refit(files, config, HOLDOUT, RefitCause.MANUAL, () -> false, rejected).orElseThrow();
      Assert.assertEquals(gate.status(), RefitStatus.REJECTED);
      Assert.assertEquals(gate.gate().result(), GateResult.NOT_BETTER);
      var failing = new FakeSteps();
      failing.failTraining = true;
      RefitRecord failed =
          GrpRefitter.refit(files, config, HOLDOUT, RefitCause.MANUAL, () -> false, failing).orElseThrow();
      Assert.assertEquals(failed.status(), RefitStatus.FAILED);
      Assert.assertTrue(failed.detail().contains("trainer exploded"), failed.detail());
      var slow = new FakeSteps();
      slow.timeOut = true;
      Assert.assertEquals(
          GrpRefitter.refit(files, config, HOLDOUT, RefitCause.MANUAL, () -> false, slow).orElseThrow().status(),
          RefitStatus.TIMEOUT);
      var malformed = new FakeSteps();
      malformed.malformed = true;
      RefitRecord invalid =
          GrpRefitter.refit(files, config, HOLDOUT, RefitCause.MANUAL, () -> false, malformed).orElseThrow();
      Assert.assertEquals(invalid.status(), RefitStatus.REJECTED);
      Assert.assertFalse(malformed.gateInput.candidateValid());
      Assert.assertEquals(invalid.gate().result(), GateResult.INVALID_OUTPUT);
      // 記録のない取り置きは学習しない
      var unrecorded = new FakeSteps();
      Assert.assertEquals(
          GrpRefitter.refit(files, config, 12, RefitCause.MANUAL, () -> false, unrecorded).orElseThrow().status(),
          RefitStatus.FAILED);
      Assert.assertEquals(unrecorded.trained, 0);
      List<RefitRecord> records = RefitRecord.readAll(files);
      Assert.assertEquals(
          records.stream().map(RefitRecord::id).toList(), List.of("g009-a1", "g009-a2", "g009-a3", "g009-a4", "g012-a1"));
      for (String id : List.of("g009-a1", "g009-a2", "g009-a3", "g009-a4")) {
        Assert.assertFalse(Files.exists(files.candidate(id)), id);
        Assert.assertFalse(Files.exists(files.staging().resolve(id)), id);
        Assert.assertFalse(Files.exists(files.boundary(id)), id);
      }
    } finally {
      Snapshot.deleteTree(files.run());
    }
  }

  /** 止める合図: 学習が止まっても、学習の後・関門の前に立っても、試行を捨てて行も候補・書きかけ・見本も残さない。 */
  @Test
  public void aStopSignalDiscardsTheAttempt() throws Exception {
    GrpFiles files = run(grpStart());
    try {
      PrismConfig config = GrpFixtures.grpStart(Path.of("/human"));
      RefitRecord ready =
          GrpRefitter.refit(files, config, HOLDOUT, RefitCause.MANUAL, () -> false, new FakeSteps()).orElseThrow();
      BooleanSupplier stop = () -> true;
      var stopped = new FakeSteps();
      stopped.stopped = true;
      Assert.assertTrue(GrpRefitter.refit(files, config, HOLDOUT, RefitCause.SCHEDULE, stop, stopped).isEmpty());
      Assert.assertSame(stopped.stop, stop, "the trainer sees the stop signal");
      Assert.assertNull(stopped.gateInput, "no gate after a stopped training");
      // 学習を終えた後に立った合図は関門の前で見る
      var finished = new FakeSteps();
      Assert.assertTrue(GrpRefitter.refit(files, config, HOLDOUT, RefitCause.SCHEDULE, stop, finished).isEmpty());
      Assert.assertEquals(finished.trained, 1);
      Assert.assertNull(finished.gateInput, "no gate after the stop signal");
      Assert.assertEquals(RefitRecord.readAll(files).stream().map(RefitRecord::id).toList(), List.of(ready.id()));
      Assert.assertTrue(Files.exists(files.candidate(ready.id())), "the published candidate stays");
      Assert.assertFalse(Files.exists(files.candidate("g009-a2")));
      Assert.assertFalse(Files.exists(files.staging().resolve("g009-a2")));
      Assert.assertFalse(Files.exists(files.boundary("g009-a2")));
    } finally {
      Snapshot.deleteTree(files.run());
    }
  }

  @Test
  public void aScoreOnlyPhiTrainsFromScratchThroughTheFirstReplacementGate() throws Exception {
    GrpFiles files = run(scoreStart());
    try {
      var steps = new FakeSteps();
      PrismConfig config = GrpFixtures.scoreStart();
      RefitRecord record =
          GrpRefitter.refit(files, config, HOLDOUT, RefitCause.SCHEDULE, () -> false, steps).orElseThrow();
      Assert.assertEquals(record.status(), RefitStatus.READY, record.detail());
      Assert.assertEquals(record.init(), GrpInit.SCRATCH);
      Assert.assertNull(steps.start);
      // 同じ関門の最初の置き換えの列に、窓の全ての半荘の数と信頼の半荘の数を渡す
      Assert.assertEquals(steps.gateInput.mode(), GateMode.FIRST_REPLACEMENT);
      Assert.assertEquals(steps.gateInput.windowGames(), 160);
      Assert.assertEquals(steps.gateInput.trustGames(), config.grpTrustGames());
      Assert.assertEquals(steps.gateInput.holdout().size(), 40);
      Assert.assertEquals(record.gate().mode(), GateMode.FIRST_REPLACEMENT);
      // 最初の置き換えでも形が崩れていれば関門が拒む
      var malformed = new FakeSteps();
      malformed.malformed = true;
      RefitRecord invalid =
          GrpRefitter.refit(files, config, HOLDOUT, RefitCause.SCHEDULE, () -> false, malformed).orElseThrow();
      Assert.assertEquals(invalid.status(), RefitStatus.REJECTED);
      Assert.assertFalse(malformed.gateInput.candidateValid());
      Assert.assertEquals(invalid.gate().mode(), GateMode.FIRST_REPLACEMENT);
      Assert.assertEquals(invalid.gate().result(), GateResult.INVALID_OUTPUT);
      Assert.assertFalse(Files.exists(files.candidate(invalid.id())));
    } finally {
      Snapshot.deleteTree(files.run());
    }
  }

  @Test
  public void theWindowIsClampedAndSkipsTheFirstCohorts() {
    Assert.assertTrue(RefitWindow.ending(3, 8).isEmpty());
    Assert.assertEquals(RefitWindow.ending(4, 8).orElseThrow(), new RefitWindow(3, 3, 4));
    Assert.assertEquals(RefitWindow.ending(9, 8).orElseThrow(), new RefitWindow(5, 8, 9));
    Assert.assertEquals(RefitWindow.ending(40, 8).orElseThrow(), new RefitWindow(32, 39, 40));
    Assert.assertEquals(RefitWindow.ending(560, 8).orElseThrow(), new RefitWindow(552, 559, 560));
  }

  @Test
  public void grpRefitOnlyWritesTheRequest() throws Exception {
    GrpFiles files = run(grpStart());
    try {
      GrpFixtures.writeConfig(files.run(), GrpFixtures.grpStart(Path.of("/human")));
      var workflow = new GrpRefitWorkflow();
      Assert.assertEquals(workflow.execute(arguments("--run", files.run().toString())), 0);
      Assert.assertEquals(holdout(files), HOLDOUT);
      Assert.assertEquals(workflow.execute(arguments("--run", files.run().toString(), "--holdout", "7")), 0);
      Assert.assertEquals(holdout(files), 7);
      Assert.assertThrows(
          IllegalArgumentException.class, () -> workflow.execute(arguments("--run", files.run().toString(), "--holdout", "12")));
      Assert.assertThrows(
          IllegalArgumentException.class, () -> workflow.execute(arguments("--run", files.run().toString(), "--holdout", "3")));
      Assert.assertTrue(RefitRecord.readAll(files).isEmpty(), "grp-refit does not train");
      GrpFixtures.writeConfig(files.run(), GrpFixtures.config());
      Assert.assertThrows(IllegalArgumentException.class, () -> workflow.execute(arguments("--run", files.run().toString())));
    } finally {
      Snapshot.deleteTree(files.run());
    }
  }

  private static int holdout(GrpFiles files) throws IOException {
    return JsonParser.parseString(Files.readString(files.refitRequest()))
        .getAsJsonObject()
        .get(GrpRefitWorkflow.HOLDOUT)
        .getAsInt();
  }

  static WorkflowArguments arguments(String... values) {
    return new WorkflowArguments("grp", values, EpsilonSettings.defaults());
  }

  @Test
  public void attemptIdsArePadded() {
    Assert.assertEquals(GrpRefitter.attemptId(9, 1), "g009-a1");
    Assert.assertEquals(GrpRefitter.attemptId(1024, 12), "g1024-a12");
  }
}
