package com.epsilon.prism.eval;

import com.epsilon.agent.eval.Alarm;
import com.epsilon.agent.eval.Arena;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.PlayStats;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.Stopped;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.run.ControlRequest;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.Jars;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.run.PrismCarry;
import com.epsilon.prism.run.PrismState;
import com.epsilon.prism.run.Regression;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 評価のジョブの結果を反映する処理（{@link EvaluationJobs#settle}）のテスト。王者の保存先が空なら、王者の卓は候補を最初の王者にする。
 * 王者との決闘で良ければ候補を昇格させ、悪い結果が続けば回数に応じて、アラート、王者への巻き戻し、一時停止の要求を置く。どの結果も、
 * 2 度反映しても 1 度と同じになる（保存先・state・アラート・要求）。
 */
public class EvaluationJobsTest {

  private static final String TABLE = "duel";

  private Path root;
  private RunDirectory run;
  private PrismSettings settings;

  @BeforeMethod
  public void createRun() throws Exception {
    root = Files.createTempDirectory("prism-evaluation-jobs");
    run = new RunDirectory(root);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws Exception {
    Snapshot.deleteTree(root);
  }

  @Test
  public void anEmptyStoreCrownsTheCandidateOnce() throws Exception {
    EvaluationJobs jobs = jobs("evaluation.onWorse={kind = \"REPORT\", streak = 2}");
    JobFile job = duel(2, false);
    Node.Table crowned = RecordFile.tree(new EvaluationJobs.Crowned("v00002"));
    jobs.settle(job, crowned);
    jobs.settle(job, crowned);
    ChampionStore store = ChampionStore.open(run.champion());
    Assert.assertEquals(store.history().size(), 1);
    Assert.assertEquals(store.current().orElseThrow().version(), 2);
    Assert.assertEquals(state(), initial(), "crowning does not touch the regression count");
  }

  @Test
  public void aBetterDuelPromotesTheCandidateAndSettlingAgainChangesNothing() throws Exception {
    EvaluationJobs jobs = jobs("evaluation.onWorse={kind = \"REPORT\", streak = 2}");
    crown(2);
    JobFile job = duel(3, true);
    jobs.settle(job, measured(Verdict.BETTER));
    String settled = Files.readString(run.state(), StandardCharsets.UTF_8);
    jobs.settle(job, measured(Verdict.BETTER));
    Assert.assertEquals(Files.readString(run.state(), StandardCharsets.UTF_8), settled);
    ChampionStore store = ChampionStore.open(run.champion());
    Assert.assertEquals(store.history().size(), 2);
    Assert.assertEquals(store.current().orElseThrow().version(), 3);
    Assert.assertEquals(state().regression(), new Regression(0, "v00003-duel", List.of()));
    Assert.assertFalse(Files.exists(run.report(Alarm.FILE)));
  }

  @Test
  public void worseDuelsReportThenRollBackThenPause() throws Exception {
    EvaluationJobs jobs = jobs("evaluation.onWorse={kind = \"ROLLBACK\", streak = 2, maxRollbacks = 1, span = 10}");
    crown(2);
    // 1 回目の悪い結果: アラートだけ
    settleTwice(jobs, duel(3, true), Verdict.WORSE);
    Assert.assertEquals(state().regression(), new Regression(1, "v00003-duel", List.of()));
    Assert.assertEquals(state().start(), 5);
    Assert.assertEquals(alarms(), List.of("v00003-duel:CHAMPION_WORSE/REPORT"));
    // 悪い結果が 2 回続いた: 次の反復を王者のバージョン 2 から始める
    settleTwice(jobs, duel(4, true), Verdict.WORSE);
    Assert.assertEquals(state().regression(), new Regression(2, "v00004-duel", List.of(4)));
    Assert.assertEquals(state().start(), 2);
    Assert.assertEquals(
        alarms(),
        List.of(
            "v00003-duel:CHAMPION_WORSE/REPORT", "v00004-duel:CHAMPION_WORSE/REPORT", "v00004-duel:WORSE_STREAK/ROLLBACK"));
    ChampionStore store = ChampionStore.open(run.champion());
    Assert.assertTrue(store.history().getLast().event() instanceof StoreEvent.RolledBack, store.history().toString());
    Assert.assertEquals(store.current().orElseThrow().version(), 2, "a rollback keeps the champion");
    Assert.assertEquals(new ControlRequests(run).pending(), List.of());
    // 決めた反復の数（span）の中で巻き戻しの上限を使い切った後の悪い結果: 一時停止を要求する
    settleTwice(jobs, duel(5, true), Verdict.WORSE);
    Assert.assertEquals(state().regression(), new Regression(3, "v00005-duel", List.of(4)));
    Assert.assertEquals(alarms().getLast(), "v00005-duel:WORSE_STREAK/PAUSE");
    Assert.assertTrue(ChampionStore.open(run.champion()).history().getLast().event() instanceof StoreEvent.Paused);
    List<ControlRequest> pending = new ControlRequests(run).pending();
    Assert.assertEquals(pending.size(), 1, pending.toString());
    Assert.assertTrue(pending.getFirst().request() instanceof Request.Pause, pending.toString());
    // 悪くない結果（ここでは互角）が出れば、悪い結果の連続は 0 に戻る
    settleTwice(jobs, duel(6, true), Verdict.EVEN);
    Assert.assertEquals(state().regression(), new Regression(0, "v00006-duel", List.of(4)));
  }

  /**
   * 後のバージョンの決闘を反映した後に決着した、前のバージョンの決闘（FAILED から戻したジョブ）は、良くても候補を昇格させず、悪くても
   * 悪い結果の連続に数えない。
   */
  @Test
  public void aDuelSettledAfterTheDuelOfALaterVersionChangesNothing() throws Exception {
    EvaluationJobs jobs = jobs("evaluation.onWorse={kind = \"ROLLBACK\", streak = 2, maxRollbacks = 1, span = 10}");
    crown(2);
    // バージョン 3 と 4 の決闘は、王者がバージョン 2 の間に計画を決めたまま、決着していない
    JobFile third = duel(3, true);
    JobFile fourth = duel(4, true);
    jobs.settle(duel(5, true), measured(Verdict.BETTER));
    Assert.assertEquals(ChampionStore.open(run.champion()).current().orElseThrow().version(), 5);
    jobs.settle(third, measured(Verdict.BETTER));
    ChampionStore store = ChampionStore.open(run.champion());
    Assert.assertEquals(store.current().orElseThrow().version(), 5, "an older version does not replace the champion");
    Assert.assertEquals(store.history().size(), 2, store.history().toString());
    Assert.assertEquals(state().regression(), new Regression(0, "v00005-duel", List.of()));

    jobs.settle(duel(6, true), measured(Verdict.WORSE));
    Assert.assertEquals(state().regression(), new Regression(1, "v00006-duel", List.of()));
    jobs.settle(fourth, measured(Verdict.WORSE));
    Assert.assertEquals(
        state().regression(), new Regression(1, "v00006-duel", List.of()), "an older duel does not extend the streak");
    Assert.assertEquals(state().start(), 5, "an older duel does not roll back");
    Assert.assertEquals(alarms(), List.of("v00006-duel:CHAMPION_WORSE/REPORT"));
    Assert.assertEquals(ChampionStore.open(run.champion()).history().size(), 2);
  }

  /** 結果を 2 度反映し、2 度目が保存先・state・アラート・要求を変えないことを確かめる。 */
  private void settleTwice(EvaluationJobs jobs, JobFile job, Verdict verdict) throws Exception {
    jobs.settle(job, measured(verdict));
    String state = Files.readString(run.state(), StandardCharsets.UTF_8);
    int history = ChampionStore.open(run.champion()).history().size();
    List<String> alarms = alarms();
    int requests = new ControlRequests(run).pending().size();
    jobs.settle(job, measured(verdict));
    Assert.assertEquals(Files.readString(run.state(), StandardCharsets.UTF_8), state, job.name());
    Assert.assertEquals(ChampionStore.open(run.champion()).history().size(), history, job.name());
    Assert.assertEquals(alarms(), alarms, job.name());
    Assert.assertEquals(new ControlRequests(run).pending().size(), requests, job.name());
  }

  /**
   * 王者と決闘する卓（全ての割り当てで打って判定し、良ければ昇格させる）を持つ run を作る。悪い結果のときの扱いは、渡した設定のとおりにする。
   * state は反復 5 を始める前の状態にする。
   */
  private EvaluationJobs jobs(String onWorse) throws Exception {
    settings =
        TestSettings.create(
            root,
            "phi.source=SCORE_RANK",
            "evaluation.tables." + TABLE + ".opponent=champion",
            "evaluation.tables." + TABLE + ".seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v3\", \"3v1\"]}",
            "evaluation.tables." + TABLE + ".draws=SEAT_STREAMS",
            "evaluation.tables." + TABLE + ".wallSet=FRESH",
            "evaluation.tables." + TABLE + ".walls=1024",
            "evaluation.tables." + TABLE
                + ".stop={kind = \"TEST\", evenMargin = 0.05, alpha = 0.05, spending = 1.0, checkEvery = 512,"
                + " promote = true}",
            "evaluation.tables." + TABLE + ".score=RANK",
            "evaluation.tables." + TABLE + ".every=1",
            "evaluation.tables." + TABLE + ".from=2",
            onWorse);
    DurableFiles.write(run.state(), initial());
    for (int version = 2; version <= 6; version++) {
      Files.createDirectories(run.snapshot(version));
      Files.writeString(run.snapshot(version).resolve("weights.bin"), "version " + version);
    }
    return new EvaluationJobs(run, settings, null, null);
  }

  private static PrismState initial() {
    return new PrismState(5, 0, PrismCarry.NONE, 5, Regression.NONE);
  }

  private void crown(int version) throws Exception {
    Champion champion = Champion.of(label(version), run.snapshot(version), version);
    Assert.assertTrue(ChampionStore.open(run.champion()).append(new StoreEvent.Initial(champion)));
  }

  /**
   * 卓のジョブに、始めたときの計画を入れたものを作る。
   *
   * @param againstChampion 今の王者と打つ計画なら true。false なら、保存先が空で、候補を最初の王者にする計画
   */
  private JobFile duel(int version, boolean againstChampion) throws Exception {
    TableSpec table = settings.evaluation().tables().get(TABLE);
    JobFile pending = JobFile.pending(version, TABLE, 0, 1, RecordFile.tree(table));
    DuelSetup.Entrant candidate =
        new DuelSetup.Entrant("prism:" + run.snapshot(version), label(version), version, null);
    EvaluationPlan plan = new EvaluationPlan(candidate, null, Jars.CLASSES, settings.behaviour().twinDiscard());
    if (againstChampion) {
      Champion champion = ChampionStore.open(run.champion()).current().orElseThrow();
      DuelSetup.Entrant opponent =
          new DuelSetup.Entrant("prism:" + champion.checkpoint(), champion.label(), champion.version(), null);
      plan =
          new EvaluationPlan(
              candidate,
              DuelSetup.of(
                  settings.run().variant(),
                  candidate,
                  opponent,
                  table.seating().seating(),
                  table.seating().splits(settings.run().variant()),
                  table.draws(),
                  table.wallSet(),
                  settings.run().seed(),
                  Arena.WALL_BASE,
                  table.walls(),
                  EvaluationJobs.rule(table.stop()),
                  table.score(),
                  settings.utility()),
              Jars.CLASSES,
              settings.behaviour().twinDiscard());
    }
    return new JobFile(
        pending.name(),
        pending.version(),
        pending.table(),
        pending.order(),
        pending.row(),
        pending.definition(),
        RecordFile.tree(plan),
        pending.state(),
        pending.attempts(),
        pending.cleanStops(),
        pending.walls(),
        pending.gamesBytes(),
        pending.looksBytes(),
        pending.created(),
        pending.reason());
  }

  /** 渡した判定になる決闘の結果を作る。順位の区間は、候補が良いほど正になる区間の符号を反転したもの。 */
  private static Node.Table measured(Verdict verdict) {
    Estimate better =
        switch (verdict) {
          case BETTER -> new Estimate(0.2, 0.05, 0.1, 0.3);
          case WORSE -> new Estimate(-0.2, 0.05, -0.3, -0.1);
          default -> new Estimate(0.0, 0.01, -0.02, 0.02);
        };
    List<Double> shares = List.of(0.25, 0.25, 0.25, 0.25);
    DuelResult result =
        new DuelResult(
            512,
            4096,
            verdict,
            Stopped.DECIDED,
            Score.RANK,
            3.0,
            better.negate(),
            new Estimate(0, 0, 0, 0),
            List.of(),
            List.of(
                new DuelResult.SlotValue("candidate", 2.5 - better.mean(), better.error(), 0, 0, shares),
                new DuelResult.SlotValue("v00002", 2.5, 0, 0, 0, shares)),
            new PlayStats.Rates(0, 0, 0, 0, 0),
            0,
            0,
            1.0);
    return RecordFile.tree(result);
  }

  private PrismState state() throws Exception {
    return RecordFile.read(PrismState.class, run.state()).value();
  }

  /** アラートの行の「名前/扱い」を、書いた順に返す。 */
  private List<String> alarms() throws Exception {
    List<String> lines = new ArrayList<>();
    for (String line : DurableFiles.lines(run.report(Alarm.FILE))) {
      Alarm alarm = RecordFile.read(Alarm.class, Json.read(line, "alarm")).value();
      lines.add(alarm.alarm() + "/" + alarm.handling());
    }
    return lines;
  }

  private static String label(int version) {
    return String.format("v%05d", version);
  }
}
