package com.epsilon.prism.grp;

import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpTensor;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GateFailure;
import com.epsilon.grp.train.GateKind;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 三人打ちの run で、点数の順位の Φ（g000）からネットワークへ最初に切り替える流れを確かめる。切り替えは、学び直した候補が置き換えの関門
 * （使用中より −2 SE 良く、単調）を通ったときだけ起き、反復 h ＋ 1 から効く。通らなければ点数だけの Φ のまま。範囲の半荘が
 * trustGames に届くまでは学び直さない。切り替えた後の 8 反復は τ が 1。
 *
 * <p>候補は重みが 0 のネットワーク（全ての順列が同じ確率で、どの席の Φ も効用の平均 0）。最終の順位が卓に依らない半荘では、
 * 点数の順位の Φ より MSE がはっきり良い（点数の順位は 16/3、候補は 8/3）。最終の順位が局末の順位のままの半荘では、点数の順位の
 * Φ の MSE が 0 なので通らない。
 */
public class ThreePlayerFirstSwitchTest {

  private static final Variant THREE = Variant.TENHOU_3P;

  /** 最終の点の組（着順の順）。 */
  private static final int[] FINALS = {50_000, 35_000, 20_000};

  private static final int GAMES = 60;

  private Path run;
  private GrpFiles files;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-3p-first-switch");
    files = new GrpFiles(run);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  /** 点数の順位で始め、学び直しは予定どおり、切り替えは AUTO にした三人打ちの run の設定（引数の上書きを後ろに重ねる）。 */
  static PhiSetup setup(String... more) {
    List<String> assignments =
        new ArrayList<>(
            List.of(
                "phi.source=SELF_GRP",
                "phi.start.kind=SCORE_RANK",
                "phi.grp.monitor=RECORD",
                "phi.grp.refit=SCHEDULE",
                "phi.grp.switching=AUTO",
                "phi.grp.threads=1"));
    assignments.addAll(List.of(more));
    PrismSettings settings = ThreePlayerSettings.of(assignments.toArray(String[]::new));
    return new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed());
  }

  /** g000 を書き、state.json の次のコホートを {@code next} にする。 */
  private PhiSetup start(int next, String... more) throws IOException {
    PhiSetup setup = setup(more);
    writeState(run, next);
    PhiEntry initial = PhiHistory.initialize(files, setup);
    Assert.assertEquals(initial.kind(), PhiEntryKind.SCORE_RANK);
    Assert.assertEquals(initial.variant(), THREE);
    return setup;
  }

  static void writeState(Path run, int next) throws IOException {
    Files.createDirectories(run);
    Files.writeString(run.resolve("state.json"), "{\"format\": 1, \"next\": " + next + "}");
  }

  /** コホート t を確定済みにし、Φ の項目 {@code entry} で集めた半荘の記録を書く。 */
  static void record(GrpFiles files, int t, PhiEntry entry, List<GameRecord> games) throws IOException {
    Files.createDirectories(files.cohort(t));
    Files.writeString(files.cohort(t).resolve(CohortWriter.MANIFEST), "{}");
    Files.createDirectories(files.records(t).getParent());
    GameRecordCodec.write(files.records(t), new GameCohort(THREE, t, entry.id(), games));
  }

  /** コホート t に、その反復に効く Φ の項目で集めた記録を書く。 */
  private void record(int t, boolean ordered) throws IOException {
    PhiEntry used = PhiHistory.effectiveAt(PhiHistory.read(files), t).orElseThrow();
    record(files, t, used, games(10_000L * t, GAMES, t, ordered));
  }

  /**
   * 三人打ちの半荘。局末の数は 2〜8 で、局末 k の次局は k ＋ 1、局末の点は 35,000 点のまわり。{@code ordered} なら最終の順位は全ての
   * 局末での順位と同じで、そうでなければ最終の点を卓に依らず並べ替える。
   */
  static List<GameRecord> games(long first, int count, long seed, boolean ordered) {
    SplittableRandom random = new SplittableRandom(seed);
    List<GameRecord> games = new ArrayList<>(count);
    for (int game = 0; game < count; game++) {
      int[] order = {0, 1, 2};
      for (int index = 2; index > 0; index--) {
        int other = random.nextInt(index + 1);
        int held = order[index];
        order[index] = order[other];
        order[other] = held;
      }
      int[] finals = new int[3];
      for (int seat = 0; seat < 3; seat++) {
        finals[seat] = FINALS[order[seat]];
      }
      int hands = 2 + random.nextInt(7);
      List<GrpState> handEnds = new ArrayList<>(hands);
      for (int hand = 0; hand < hands; hand++) {
        int[] scores = new int[3];
        if (ordered) {
          int step = 1_000 * (1 + random.nextInt(10));
          for (int seat = 0; seat < 3; seat++) {
            scores[seat] = 35_000 + step * (1 - order[seat]);
          }
        } else {
          scores[0] = 35_000 + 100 * (random.nextInt(201) - 100);
          scores[1] = 35_000 + 100 * (random.nextInt(201) - 100);
          scores[2] = 105_000 - scores[0] - scores[1];
        }
        handEnds.add(new GrpState(THREE, hand + 1, 0, 0, scores));
      }
      games.add(new GameRecord(THREE, first + game, handEnds, finals));
    }
    return games;
  }

  /** 学び直しと同じ形の三人打ちの GrpNetwork を、重みの標準偏差 {@code scale} で作り、ディレクトリに書く。 */
  static Path network(Path directory, long seed, double scale) throws IOException {
    float[] weights = new float[GrpTensor.starts(GrpRefitter.shape(THREE))[GrpTensor.values().length]];
    SplittableRandom random = new SplittableRandom(seed);
    for (int index = 0; index < weights.length; index++) {
      weights[index] = (float) (random.nextGaussian() * scale);
    }
    GrpCheckpoint.write(GrpNetwork.of(GrpRefitter.shape(THREE), weights), directory);
    return directory.toAbsolutePath().normalize();
  }

  /** 時点 {@code slot} の TRAINED の候補を作る（checkpoint を候補のディレクトリに写す）。 */
  private RefitRecord trained(int slot, Path checkpoint) throws IOException {
    String id = GrpRefitter.attemptId(slot, 1);
    Path candidate = files.candidate(id);
    Files.createDirectories(candidate);
    for (String name : List.of(GrpCheckpoint.JSON_FILE, GrpCheckpoint.WEIGHTS_FILE)) {
      Files.copy(checkpoint.resolve(name), candidate.resolve(name));
    }
    RefitRecord record =
        new RefitRecord(
            id, slot, RefitStatus.TRAINED, RefitCause.SCHEDULE, new RecencyWindow(slot, 2, 64), GrpInit.SCRATCH,
            PhiEntry.INITIAL_ID, candidate.toAbsolutePath().normalize().toString(), "", "", null, null, "", 1);
    record.append(files);
    return record;
  }

  private RefitRecord latest(String id) throws IOException {
    return RefitRecord.latest(RefitRecord.readAll(files), id).orElseThrow();
  }

  @Test
  public void aCandidateThatPassesTheReplacementGateSwitchesFromTheNextIteration() throws Exception {
    PhiSetup setup = start(3);
    GrpLifecycle lifecycle = new GrpLifecycle(files, setup);
    record(1, false);
    record(2, false);
    lifecycle.boundary(1);
    trained(1, network(run.resolve("flat"), 1, 0));
    lifecycle.boundary(2);
    RefitRecord gated = latest("g001-a1");
    Assert.assertEquals(gated.status(), RefitStatus.READY, gated.detail());
    Assert.assertEquals(gated.gate().kind(), GateKind.REPLACEMENT);
    Assert.assertTrue(gated.gate().passed(), gated.gate().toString());
    Assert.assertTrue(gated.gate().versusUsed().mean() < -2 * gated.gate().versusUsed().standardError());
    List<PhiEntry> history = PhiHistory.read(files);
    Assert.assertEquals(history.size(), 2);
    PhiEntry switched = history.getLast();
    Assert.assertEquals(switched.id(), "g001-a1");
    Assert.assertEquals(switched.kind(), PhiEntryKind.NETWORK);
    Assert.assertEquals(switched.variant(), THREE);
    Assert.assertEquals(switched.parent(), PhiEntry.INITIAL_ID);
    Assert.assertEquals(switched.effectiveFrom(), 3);
    Assert.assertTrue(switched.reason().startsWith("AUTO"), switched.reason());
    Assert.assertEquals(PhiHistory.effectiveAt(history, 2).orElseThrow().kind(), PhiEntryKind.SCORE_RANK);
    Assert.assertEquals(PhiHistory.effectiveAt(history, 3).orElseThrow().id(), "g001-a1");
    // ネットワークの形を替えたので、切り替えから 8 反復は τ が 1。その後は設定の値
    GrpOperations operations = setup.selfGrp().grp();
    SwitchWatchRecord.State quiet = SwitchWatchRecord.state(SwitchWatchRecord.readAll(files));
    Assert.assertFalse(quiet.widened());
    Assert.assertEquals(GrpRefitter.halfLife(operations, history, quiet, 2), operations.halfLife());
    for (int slot = 3; slot <= 10; slot++) {
      Assert.assertEquals(GrpRefitter.halfLife(operations, history, quiet, slot), 1.0, "slot " + slot);
    }
    Assert.assertEquals(GrpRefitter.halfLife(operations, history, quiet, 11), operations.halfLife());
  }

  @Test
  public void aCandidateThatFailsTheReplacementGateKeepsTheScoreOnlyPhi() throws Exception {
    PhiSetup setup = start(3);
    GrpLifecycle lifecycle = new GrpLifecycle(files, setup);
    record(1, true);
    record(2, true);
    lifecycle.boundary(1);
    trained(1, network(run.resolve("flat"), 1, 0));
    lifecycle.boundary(2);
    RefitRecord rejected = latest("g001-a1");
    Assert.assertEquals(rejected.status(), RefitStatus.REJECTED, rejected.detail());
    Assert.assertEquals(rejected.gate().kind(), GateKind.REPLACEMENT);
    Assert.assertTrue(rejected.gate().failures().contains(GateFailure.NOT_BETTER), rejected.gate().toString());
    List<PhiEntry> history = PhiHistory.read(files);
    Assert.assertEquals(history.size(), 1);
    Assert.assertEquals(PhiHistory.effectiveAt(history, 3).orElseThrow().kind(), PhiEntryKind.SCORE_RANK);
    Assert.assertEquals(PhiHistory.effectiveAt(history, 3).orElseThrow().id(), PhiEntry.INITIAL_ID);
  }

  /** 使用中が点数だけの Φ なら、範囲の半荘が trustGames に 1 つでも足りないうちは待ち、何も書かない。 */
  @Test
  public void aScoreOnlyPhiWaitsUntilTheWindowHoldsTrustGames() throws Exception {
    PhiSetup counting = start(6);
    for (int t = 1; t <= 5; t++) {
      record(files, t, PhiHistory.read(files).getFirst(), games(1_000L * t, 30, t, false));
    }
    GrpOperations operations = counting.selfGrp().grp();
    RecencyWindow window =
        new RecencyWindow(
            5,
            GrpRefitter.halfLife(operations, PhiHistory.read(files), SwitchWatchRecord.state(List.of()), 5),
            operations.window());
    int games = 30 * (5 - window.first() + 1);
    Assert.assertTrue(games > 0 && window.first() > 1, window.toString());
    Assert.assertTrue(
        GrpRefitter.refit(files, setup("phi.grp.trustGames=" + (games + 1)), 5, RefitCause.SCHEDULE, () -> false)
            .isEmpty());
    Assert.assertTrue(RefitRecord.readAll(files).isEmpty());
    Assert.assertFalse(Files.exists(files.candidate("x").getParent()));
    // 待っている間は関門も切り替えも起きない
    GrpLifecycle lifecycle = new GrpLifecycle(files, setup("phi.grp.trustGames=" + (games + 1)));
    lifecycle.boundary(5);
    Assert.assertEquals(PhiHistory.read(files).size(), 1);
  }

  /** 範囲の半荘がちょうど trustGames なら、点数だけの Φ からゼロから学ぶ。 */
  @Test(groups = "native", timeOut = 600_000)
  public void exactlyTrustGamesInTheWindowTrainsFromScratch() throws Exception {
    PhiSetup counting = start(6);
    for (int t = 1; t <= 5; t++) {
      record(files, t, PhiHistory.read(files).getFirst(), games(1_000L * t, 30, t, false));
    }
    GrpOperations operations = counting.selfGrp().grp();
    RecencyWindow window =
        new RecencyWindow(
            5,
            GrpRefitter.halfLife(operations, PhiHistory.read(files), SwitchWatchRecord.state(List.of()), 5),
            operations.window());
    int games = 30 * (5 - window.first() + 1);
    RefitRecord record =
        GrpRefitter.refit(files, setup("phi.grp.trustGames=" + games), 5, RefitCause.SCHEDULE, () -> false)
            .orElseThrow();
    Assert.assertEquals(record.status(), RefitStatus.TRAINED, record.detail());
    Assert.assertEquals(record.init(), GrpInit.SCRATCH);
    Assert.assertEquals(record.parent(), PhiEntry.INITIAL_ID);
    Assert.assertTrue(record.detail().contains("games=" + games), record.detail());
  }

  /** 三人打ちで点数のなめらかな関数の Φ を使うと、settings check と同じ確かめで問題になり、理由は英語で出る。点数の順位なら問題はない。 */
  @Test
  public void threePlayerScoreSmoothIsAProblemWithAnEnglishReason() {
    String reason = "The score-smooth sigma is not measured for TENHOU_3P";
    PrismSettings direct = ThreePlayerSettings.of("phi.source=SCORE_SMOOTH");
    List<String> problems =
        PhiControl.problems(new PhiSetup(direct.phi(), direct.run().variant(), direct.run().seed()));
    Assert.assertEquals(problems.size(), 1, problems.toString());
    Assert.assertTrue(problems.getFirst().startsWith("The Φ of the settings cannot be opened: "), problems.toString());
    Assert.assertTrue(problems.getFirst().endsWith(reason), problems.toString());
    Assert.assertEquals(PhiControl.problems(setup("phi.start.kind=SCORE_SMOOTH")).size(), 1);
    Assert.assertTrue(PhiControl.problems(setup("phi.start.kind=SCORE_SMOOTH")).getFirst().endsWith(reason));
    Assert.assertEquals(PhiControl.problems(setup()), List.of());
    PrismSettings rank = ThreePlayerSettings.of("phi.source=SCORE_RANK");
    Assert.assertEquals(PhiControl.problems(new PhiSetup(rank.phi(), rank.run().variant(), rank.run().seed())), List.of());
    IllegalArgumentException opened =
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> HandEndPhi.open(PhiEntry.scoreOnly(PhiEntryKind.SCORE_SMOOTH, THREE), setup().utility()));
    Assert.assertEquals(opened.getMessage(), reason);
  }
}
