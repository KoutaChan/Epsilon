package com.epsilon.prism.eval;

import static com.epsilon.prism.eval.ThreePlayerFrozenLadderTest.concat;
import static com.epsilon.prism.eval.ThreePlayerFrozenLadderTest.table;

import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.Duel;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Signal;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobRunner;
import com.epsilon.agent.run.JobWork;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.run.PrismCarry;
import com.epsilon.prism.run.PrismState;
import com.epsilon.prism.run.Regression;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの小さな系譜のスモークテスト（CPU）。バージョン 2〜10 を、予定の卓のジョブ
 * （{@link com.epsilon.agent.eval.EvaluationSettings#scheduled}、{@code PrismRun.jobs} と同じ卓）として作り、{@link Jobs#runAll} で実行する。
 * 打ち手はルールベースの打ち手と、1 割の判断で合法手を乱数で選ぶルールベースの打ち手で、卓は実際に打つ。
 *
 * <p>v2・v3 は混ぜた打ち手（v2 が最初の王者）、v4〜v9 はルールベースの打ち手（v4 が昇格）、v10 は混ぜた打ち手。次が全て書かれることを
 * 確かめる: 王者との決闘の行（最初の王者・昇格・悪い）、昇格した v4 の基準のジョブ、8 つ前のラダー（v9 は打ち、v10 は基準の卓と相手が
 * 重なるので打たない）、baseline.tsv・ratings.tsv・alarms.tsv、王者より悪い・2×R の低下・基準の低下のアラートの行。
 */
public class ThreePlayerLadderSmokeTest {

  private static final Variant THREE = Variant.TENHOU_3P;
  private static final String TEST =
      "{kind = \"TEST\", evenMargin = 0.01, alpha = 0.01, spending = 3.0, checkEvery = 512, promote = true}";

  /** 混ぜた打ち手のバージョン（ほかはルールベースの打ち手。v1 は 8 つ前のラダーの相手）。 */
  private static final Set<Integer> MIXED = Set.of(2, 3, 10);

  /**
   * 判断の時点の観測から一様乱数を作り、確率 0.9 でルールベースの打ち手の手を打ち、残りは合法手から一様に選んで打つ。同じ判断の時点では
   * 同じ値を引くので、同じ打ち手どうしの決闘は全ての牌山で差が 0 になる。
   */
  record Mixed(InlinePolicy rules) implements InlinePolicy {

    @Override
    public Variant variant() {
      return THREE;
    }

    @Override
    public String label() {
      return "mixed";
    }

    @Override
    public int act(Observation o) {
      long state = mix(SeedMixer.mix64(23), o.observer());
      state = mix(state, o.kyoku() * 64L + o.honba());
      state = mix(state, o.remainingWall());
      state = mix(state, o.events());
      state = mix(state, o.drawnTile());
      for (int seat = 0; seat < THREE.seats(); seat++) {
        state = mix(state, o.score(seat));
      }
      int n = o.legalActions();
      for (int a = 0; a < n; a++) {
        state = mix(state, o.action(a).code());
      }
      double u = (state >>> 11) * 0x1.0p-53;
      return u < 0.9 ? rules.act(o) : Math.min(n - 1, (int) ((u - 0.9) * 10 * n));
    }

    private static long mix(long state, long value) {
      return SeedMixer.mix64(state ^ SeedMixer.mix64(value));
    }

    @Override
    public Evaluation evaluate(Observation o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {
      rules.close();
    }
  }

  /** 計画の打ち手を開く。系譜のバージョンが {@code MIXED} にあれば混ぜた打ち手を、ほかはルールベースの打ち手を返す。 */
  private static Policy player(DuelSetup.Entrant entrant) throws Exception {
    Policy rules =
        EvaluationTable.open(EvaluationTable.HEURISTIC, THREE, null, TwinDiscard.TSUMOGIRI, PolicyOptions.defaults(), null);
    if (entrant.version() != null && MIXED.contains(entrant.version())) {
      return new Mixed((InlinePolicy) rules);
    }
    return rules;
  }

  /**
   * 計画の作り方と結果の反映は評価のジョブのまま使い、打ち手だけを置き換えて実際に打つ。打たずに候補を最初の王者にするジョブは、評価のジョブの
   * まま開く。
   */
  record Runner(EvaluationJobs jobs, TableLoop table) implements JobRunner {

    @Override
    public Node.Table plan(JobFile job) throws Exception {
      return jobs.plan(job);
    }

    @Override
    public JobWork open(JobFile job, Path directory) throws Exception {
      EvaluationPlan plan = RecordFile.read(EvaluationPlan.class, job.plan()).value();
      if (plan.crowns()) {
        return jobs.open(job, directory);
      }
      DuelSetup duel = plan.duel();
      Policy candidate = player(duel.candidate());
      return Duel.open(job.name(), table, duel, candidate, player(duel.opponent()), directory);
    }

    @Override
    public void settle(JobFile job, Node.Table result) throws Exception {
      jobs.settle(job, result);
    }
  }

  @Test(timeOut = 1_800_000)
  public void aSmallLineageWritesTheChampionLadderBaselineAndAlarmLines() throws Exception {
    Path root = Files.createTempDirectory("prism-3p-ladder-smoke");
    RunDirectory run = new RunDirectory(root);
    try {
      String champion = "evaluation.tables.champion.";
      PrismSettings settings =
          ThreePlayerSettings.create(
              root,
              List.of(),
              concat(
                  List.of(
                      champion + "opponent=champion",
                      champion + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v2\", \"2v1\"]}",
                      champion + "draws=SEAT_STREAMS",
                      champion + "wallSet=FRESH",
                      champion + "walls=512",
                      champion + "stop=" + TEST,
                      champion + "score=RANK",
                      champion + "every=1",
                      champion + "from=2"),
                  table("rx2", "heuristic", "FIXED", 30, "FIXED", 1, 2, null),
                  table("reference", "version:2", "FIXED", 30, "FIXED", 5, 3, true),
                  table("ladder8", "version:-8", "FRESH", 30, "FIXED", 1, 9, null),
                  List.of("evaluation.anchor=reference")));
      for (int version = 1; version <= 10; version++) {
        Files.createDirectories(run.snapshot(version));
        Files.writeString(run.snapshot(version).resolve("weights.bin"), "version " + version);
      }
      DurableFiles.write(run.state(), new PrismState(11, 0, PrismCarry.NONE, 11, Regression.NONE));
      Jobs jobs = new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS));
      try (TableLoop loop = new TableLoop(new TableSettings(16, 4, 1000L, 4))) {
        for (int version = 2; version <= 10; version++) {
          int order = 0;
          for (Map.Entry<String, TableSpec> entry : settings.evaluation().scheduled(version).entrySet()) {
            jobs.create(JobFile.pending(version, entry.getKey(), order++, 1, RecordFile.tree(entry.getValue())));
          }
          Runner runner = new Runner(new EvaluationJobs(run, settings, null, null), loop);
          Assert.assertTrue(jobs.runAll(runner, RunControl.detached(() -> false)).isEmpty(), "v" + version);
        }
      }
      Set<String> names = new TreeSet<>();
      for (JobFile job : jobs.list().jobs()) {
        Assert.assertEquals(job.state(), JobFile.State.DONE, job.name() + " " + job.reason());
        names.add(job.name());
      }
      // 予定の卓と、昇格で足した v4 の基準。v10 の 8 つ前のラダーは、基準の卓（v10、相手 v2）と重なるので打たない
      Assert.assertTrue(
          names.containsAll(
              List.of(
                  "v00002-champion", "v00004-champion", "v00004-reference", "v00005-reference", "v00009-ladder8",
                  "v00010-reference", "v00010-champion", "v00010-rx2")),
          names.toString());
      Assert.assertFalse(names.contains("v00010-ladder8"), names.toString());
      Assert.assertFalse(names.contains("v00003-reference"), "v3 is not promoted: " + names);

      // 王者: v2 を最初の王者にし、v4 を昇格させ、v10 は悪い
      List<ChampionStore.Entry> history = ChampionStore.open(run.champion()).history();
      Assert.assertEquals(history.size(), 2, history.toString());
      Assert.assertTrue(history.get(0).event() instanceof StoreEvent.Initial initial && initial.champion().version() == 2);
      Assert.assertTrue(history.get(1).event() instanceof StoreEvent.Promoted promoted && promoted.champion().version() == 4);
      Assert.assertEquals(
          Jobs.result(run.job("v00004-champion"), DuelResult.class).verdict(), Verdict.BETTER);
      Assert.assertEquals(
          Jobs.result(run.job("v00010-champion"), DuelResult.class).verdict(), Verdict.WORSE);
      // 8 つ前のラダー: v9 は v1 と打つ（同じルールベースの打ち手なので、全ての牌山で 0）
      DuelResult ladder = Jobs.result(run.job("v00009-ladder8"), DuelResult.class);
      Assert.assertEquals(ladder.rank().mean(), 0.0);
      Assert.assertEquals(
          RecordFile.read(EvaluationPlan.class, jobs.read("v00009-ladder8").plan()).value().duel().opponent().version(), 1);

      // 報告
      List<String> baseline = Files.readAllLines(run.report(EvaluationReports.BASELINE), StandardCharsets.UTF_8);
      Assert.assertEquals(
          baseline.stream().skip(1).map(line -> line.split("\t")[0] + line.split("\t")[1]).toList(),
          List.of("4reference", "5reference", "10reference"),
          baseline.toString());
      String ratings = Files.readString(run.report(EvaluationReports.RATINGS), StandardCharsets.UTF_8);
      Assert.assertTrue(ratings.startsWith("# Ratings against v00002"), ratings);
      String alarms = Files.readString(run.report(EvaluationReports.ALARMS), StandardCharsets.UTF_8);
      for (Signal signal : Signal.values()) {
        Assert.assertTrue(alarms.lines().anyMatch(line -> line.startsWith("# " + signal + "\t")), alarms);
      }
      List<String> raised =
          alarms.lines().filter(line -> !line.startsWith("#") && !line.startsWith("job\t")).toList();
      List<String> signals = new ArrayList<>();
      for (String line : raised) {
        String[] fields = line.split("\t");
        signals.add(fields[0] + ":" + fields[2] + "@" + fields[7]);
      }
      Assert.assertEquals(
          new TreeSet<>(signals),
          new TreeSet<>(
              List.of(
                  "v00010-champion:CHAMPION_WORSE@-",
                  "v00010-rx2:ANCHOR_DROP@4",
                  "v00010-reference:BASELINE_DROP@4")),
          alarms);
    } finally {
      DurableFiles.delete(root);
    }
  }
}
