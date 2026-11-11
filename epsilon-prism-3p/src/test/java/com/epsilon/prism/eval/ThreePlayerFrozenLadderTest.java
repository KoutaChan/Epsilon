package com.epsilon.prism.eval;

import com.epsilon.agent.eval.Alarm;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.Signal;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.eval.WallValues;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.JobRunner;
import com.epsilon.agent.run.JobWork;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.run.PrismCarry;
import com.epsilon.prism.run.PrismState;
import com.epsilon.prism.run.Regression;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 三人打ちの系譜の凍結したラダー（{@link EvaluationJobs} と {@link com.epsilon.agent.eval.EvaluationSettings#scheduled}）を確かめる。
 * 合成の決闘（{@link SyntheticDuels}）で、次を確かめる。
 *
 * <ul>
 *   <li>王者との決闘で昇格したバージョンには、onPromotion の基準の卓のジョブが 1 回だけ足され、同じ実行の中で打たれる。卓の相手より新しい
 *       バージョンなら、卓の最初のバージョンより前でも足され、後で基準の卓を打つバージョンと対になる。定期的に測るバージョン・相手より
 *       新しくないバージョン・onPromotion が false の卓・昇格しなかった決闘では足されない。結果を反映する前と後のどちらで落ちても、続きから
 *       回せば同じジョブになる。
 *   <li>ラダーの相手が基準の卓の相手と同じバージョンになる回は、席の決め方・ツモの順・判定に使う量も基準の卓と同じなら、ラダーのジョブを
 *       作らず、基準の卓の結果でラダーの低下を判定する。どれかが違えばラダーを打ち、ラダーの結果で判定する。
 *   <li>基準の低下は、王者の同じ卓と同じ牌山で対にした差が、幅と z のしきい値の両方を越えたときだけ出る。出るのは卓ごと・同じ王者で 1 回で、
 *       王者が替わればまた出る。同値な打牌の切り方が違えば対にしない。
 * </ul>
 */
public class ThreePlayerFrozenLadderTest {

  private static final String TEST =
      "{kind = \"TEST\", evenMargin = 0.01, alpha = 0.01, spending = 3.0, checkEvery = 512, promote = true}";

  private Path root;
  private RunDirectory run;

  @BeforeMethod
  public void createRun() throws Exception {
    root = Files.createTempDirectory("prism-3p-frozen-ladder");
    run = new RunDirectory(root);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws Exception {
    DurableFiles.delete(root);
  }

  /** 1v2 の卓の設定のキーを全て書いた上書き。 @param onPromotion 書かないなら null */
  static List<String> table(
      String name, String opponent, String wallSet, int walls, String stop, int every, int from, Boolean onPromotion) {
    String prefix = "evaluation.tables." + name + ".";
    List<String> all =
        new ArrayList<>(
            List.of(
                prefix + "opponent=" + opponent,
                prefix + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v2\"]}",
                prefix + "draws=SEAT_STREAMS",
                prefix + "wallSet=" + wallSet,
                prefix + "walls=" + walls,
                prefix + "stop=" + stop,
                prefix + "score=RANK",
                prefix + "every=" + every,
                prefix + "from=" + from));
    if (onPromotion != null) {
      all.add(prefix + "onPromotion=" + onPromotion);
    }
    return all;
  }

  /** 王者の卓（判定で止め、昇格させる）。 */
  static List<String> championTable() {
    String prefix = "evaluation.tables.champion.";
    return List.of(
        prefix + "opponent=champion",
        prefix + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v2\", \"2v1\"]}",
        prefix + "draws=SEAT_STREAMS",
        prefix + "wallSet=FRESH",
        prefix + "walls=4096",
        prefix + "stop=" + TEST,
        prefix + "score=RANK",
        prefix + "every=1",
        prefix + "from=2");
  }

  @SafeVarargs
  static String[] concat(List<String>... parts) {
    List<String> all = new ArrayList<>();
    for (List<String> part : parts) {
      all.addAll(part);
    }
    return all.toArray(String[]::new);
  }

  /** 系譜の run に設定の最初の行を書き、バージョン 1〜{@code versions} を公開し、state を置く。 */
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

  /** 卓のジョブを作る（作った順に {@code order} の番号を振る）。 */
  private void create(PrismSettings settings, int version, List<String> tables) throws Exception {
    Jobs jobs = jobs();
    int order = 0;
    for (String table : tables) {
      TableSpec spec = settings.evaluation().tables().get(table);
      Assert.assertNotNull(spec, table);
      jobs.create(JobFile.pending(version, table, order++, 1, RecordFile.tree(spec)));
    }
  }

  /** バージョンを予定どおりの卓で測るジョブを作る（{@code PrismRun.jobs} と同じ卓）。 */
  private List<String> schedule(PrismSettings settings, int version) throws Exception {
    List<String> tables = new ArrayList<>(settings.evaluation().scheduled(version).keySet());
    create(settings, version, tables);
    return tables;
  }

  /** 実行できるジョブを全て実行し、全て済んだことを確かめる（評価のジョブは毎回作り直す）。 */
  private void runAll(
      PrismSettings settings, Function<JobFile, DuelResult> champion, SyntheticDuels.Places places) throws Exception {
    SyntheticDuels.Runner runner =
        new SyntheticDuels.Runner(new EvaluationJobs(run, settings, null, null), champion, places);
    Assert.assertTrue(jobs().runAll(runner, RunControl.detached(() -> false)).isEmpty());
    for (JobFile job : jobs().list().jobs()) {
      Assert.assertEquals(job.state(), JobFile.State.DONE, job.name() + " " + job.reason());
    }
  }

  /** 済んだジョブの結果をもう一度反映する。 */
  private void settleAgain(PrismSettings settings, String name, SyntheticDuels.Places places) throws Exception {
    SyntheticDuels.Runner runner =
        new SyntheticDuels.Runner(
            new EvaluationJobs(run, settings, null, null), SyntheticDuels::noChampion, places);
    Assert.assertTrue(jobs().run(jobs().read(name), runner, RunControl.detached(() -> false)).isEmpty());
  }

  private Jobs jobs() {
    return new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS));
  }

  /** ジョブの名前（名前の順）。 */
  private List<String> names() throws Exception {
    return jobs().list().jobs().stream().map(JobFile::name).sorted().toList();
  }

  private void crown(int version) throws Exception {
    Champion champion = Champion.of(EvaluationJobs.label(version), run.snapshot(version), version);
    Assert.assertTrue(ChampionStore.open(run.champion()).append(new StoreEvent.Initial(champion)));
  }

  private void promote(int version) throws Exception {
    Champion champion = Champion.of(EvaluationJobs.label(version), run.snapshot(version), version);
    String duel = RunDirectory.jobName(version, "champion");
    Assert.assertTrue(ChampionStore.open(run.champion()).append(new StoreEvent.Promoted(champion, duel)));
  }

  private List<Alarm> alarms() throws Exception {
    List<Alarm> lines = new ArrayList<>();
    for (String line : DurableFiles.lines(run.report(Alarm.FILE))) {
      lines.add(RecordFile.read(Alarm.class, Json.read(line, "alarm")).value());
    }
    return lines;
  }

  /** アラートを「ジョブ:シグナル@王者」（王者が無ければ「ジョブ:シグナル」）の形で、書いた順に返す。 */
  private List<String> raised() throws Exception {
    List<String> raised = new ArrayList<>();
    for (Alarm line : alarms()) {
      raised.add(line.alarm() + (line.champion() == null ? "" : "@" + line.champion()));
    }
    return raised;
  }

  private static Function<JobFile, DuelResult> verdicts(Map<Integer, Verdict> verdicts) {
    return job -> SyntheticDuels.decided(verdicts.get(job.version()), 512, 0.02);
  }

  // 昇格したバージョンの基準の卓

  @Test(timeOut = 600_000)
  public void aPromotedVersionIsMeasuredOnceOnTheOnPromotionTablesOnly() throws Exception {
    PrismSettings settings =
        lineage(
            8,
            concat(
                championTable(),
                table("reference", "version:1", "FIXED", 30, "FIXED", 4, 3, true),
                table("quiet", "version:1", "FIXED", 30, "FIXED", 4, 3, false),
                table("late", "version:4", "FIXED", 30, "FIXED", 4, 6, true)));
    Assert.assertEquals(settings.evaluation().tables().get("reference").onPromotion(), Boolean.TRUE);
    Assert.assertEquals(settings.evaluation().tables().get("quiet").onPromotion(), Boolean.FALSE);
    Function<JobFile, DuelResult> champion =
        verdicts(Map.of(3, Verdict.BETTER, 4, Verdict.BETTER, 5, Verdict.EVEN, 6, Verdict.BETTER, 7, Verdict.WORSE));
    SyntheticDuels.Places places = SyntheticDuels.drawn(job -> -0.1);

    // バージョン 2: 王者の保存先が空なので、打たずに最初の王者にする（昇格ではないので、基準のジョブは足さない）
    Assert.assertEquals(schedule(settings, 2), List.of("champion"));
    runAll(settings, champion, places);
    Assert.assertEquals(names(), List.of("v00002-champion"));

    // バージョン 3: 昇格。予定では測らない回なので reference だけを足し、同じ実行の中で打つ（late の相手のバージョン 4 より古い）
    Assert.assertEquals(schedule(settings, 3), List.of("champion"));
    runAll(settings, champion, places);
    Assert.assertEquals(ChampionStore.open(run.champion()).current().orElseThrow().version(), 3);
    Assert.assertEquals(
        names(),
        List.of("v00002-champion", "v00003-champion", "v00003-reference"),
        "quiet: false, late: not newer than version 4");
    JobFile added = jobs().read("v00003-reference");
    Assert.assertEquals(added.row(), 1);
    Assert.assertEquals(RecordFile.read(TableSpec.class, added.definition()).value(),
        settings.evaluation().tables().get("reference"));
    EvaluationPlan plan = RecordFile.read(EvaluationPlan.class, added.plan()).value();
    Assert.assertEquals(plan.duel().candidate().version(), 3);
    Assert.assertEquals(plan.duel().opponent().version(), 1);
    Assert.assertTrue(Files.exists(run.job("v00003-reference").resolve(Jobs.RESULT)));

    // 反映し直しても足さない（ジョブは 1 つのままで、作り直さない）
    settleAgain(settings, "v00003-champion", places);
    Assert.assertEquals(names(), List.of("v00002-champion", "v00003-champion", "v00003-reference"));
    Assert.assertEquals(jobs().read("v00003-reference").created(), added.created());

    // バージョン 4: 昇格したが、reference の予定のバージョン（4 の倍数）なので足さない。予定のジョブを作る前に王者との決闘だけを打っても同じ。
    // late は相手のバージョンそのものなので足さない
    create(settings, 4, List.of("champion"));
    runAll(settings, champion, places);
    Assert.assertEquals(ChampionStore.open(run.champion()).current().orElseThrow().version(), 4);
    Assert.assertFalse(names().contains("v00004-reference"), names().toString());
    Assert.assertFalse(names().contains("v00004-late"), names().toString());

    // バージョン 5: 互角なので足さない
    schedule(settings, 5);
    runAll(settings, champion, places);
    Assert.assertTrue(names().stream().noneMatch(name -> name.startsWith("v00005-") && !name.endsWith("champion")));

    // バージョン 6: 昇格。late の相手のバージョン 4 より新しいので、reference と late を足す
    schedule(settings, 6);
    runAll(settings, champion, places);
    Assert.assertTrue(names().containsAll(List.of("v00006-reference", "v00006-late")), names().toString());
    Assert.assertFalse(names().contains("v00006-quiet"), names().toString());

    // バージョン 7: 悪いという判定では足さない
    schedule(settings, 7);
    runAll(settings, champion, places);
    Assert.assertEquals(
        names().stream().filter(name -> name.startsWith("v00007-")).toList(), List.of("v00007-champion"));
  }

  // 基準の卓の最初のバージョンより前に昇格したバージョン

  /** 三人打ちの系譜のプリセットと同じ番号の基準の卓（相手はバージョン 32、バージョン 40 から 8 バージョンごと）と、王者の卓を持つ系譜。 */
  private PrismSettings beforeFrom() throws Exception {
    return lineage(40, concat(championTable(), table("reference", "version:32", "FIXED", 60, "FIXED", 8, 40, true)));
  }

  /** バージョンの王者との決闘のジョブを作り、実行できるジョブを全て実行する。 */
  private void duel(
      PrismSettings settings, int version, Function<JobFile, DuelResult> champion, SyntheticDuels.Places places)
      throws Exception {
    create(settings, version, List.of("champion"));
    runAll(settings, champion, places);
  }

  @Test(timeOut = 600_000)
  public void aVersionPromotedAfterTheBaselineOpponentIsMeasuredOnceBeforeFrom() throws Exception {
    PrismSettings settings = beforeFrom();
    crown(30);
    Function<JobFile, DuelResult> champion =
        verdicts(Map.of(31, Verdict.BETTER, 32, Verdict.BETTER, 35, Verdict.BETTER));
    SyntheticDuels.Places places = (job, wall, game) -> even(wall, game);

    // バージョン 31・32: 昇格したが、基準の卓の相手（バージョン 32）より新しくないので足さない
    for (int version : List.of(31, 32)) {
      duel(settings, version, champion, places);
      Assert.assertEquals(ChampionStore.open(run.champion()).current().orElseThrow().version(), version);
    }
    Assert.assertEquals(names(), List.of("v00031-champion", "v00032-champion"));

    // バージョン 35: 卓の最初のバージョン 40 より前だが、相手より新しいので足し、同じ実行の中で打つ
    duel(settings, 35, champion, places);
    Assert.assertEquals(ChampionStore.open(run.champion()).current().orElseThrow().version(), 35);
    List<String> measured = List.of("v00031-champion", "v00032-champion", "v00035-champion", "v00035-reference");
    Assert.assertEquals(names(), measured);
    JobFile added = jobs().read("v00035-reference");
    Assert.assertEquals(added.row(), 1);
    Assert.assertEquals(added.order(), 1);
    Assert.assertEquals(RecordFile.read(TableSpec.class, added.definition()).value(),
        settings.evaluation().tables().get("reference"));
    EvaluationPlan plan = RecordFile.read(EvaluationPlan.class, added.plan()).value();
    Assert.assertEquals(plan.duel().candidate().version(), 35);
    Assert.assertEquals(plan.duel().opponent().version(), 32);
    Assert.assertTrue(Files.exists(run.job("v00035-reference").resolve(Jobs.RESULT)));

    // 反映し直しても足さない（ジョブは 1 つのままで、作り直さない）
    settleAgain(settings, "v00035-champion", places);
    Assert.assertEquals(names(), measured);
    Assert.assertEquals(jobs().read("v00035-reference").created(), added.created());
  }

  @Test(timeOut = 600_000)
  public void theBaselineDropPairsWithAChampionMeasuredBeforeFrom() throws Exception {
    PrismSettings settings = beforeFrom();
    crown(34);
    Function<JobFile, DuelResult> champion = verdicts(Map.of(35, Verdict.BETTER, 40, Verdict.EVEN));
    // v40 は、番号が 6 の倍数の牌山（60 のうち 10）では 3 半荘とも 3 位で、ほかは王者 v35 と同じ
    SyntheticDuels.Places places =
        (job, wall, game) -> job.version() == 40 && wall % 6 == 0 ? 2 : even(wall, game);
    duel(settings, 35, champion, places);
    Assert.assertTrue(Files.exists(run.job("v00035-reference").resolve(Jobs.RESULT)));

    // バージョン 40: 予定の卓を打つ。王者は互角で替わらず、基準の卓は王者 v35 の基準の卓と対にして低下を出す
    Assert.assertEquals(schedule(settings, 40), List.of("champion", "reference"));
    runAll(settings, champion, places);
    Assert.assertEquals(ChampionStore.open(run.champion()).current().orElseThrow().version(), 35);
    Assert.assertEquals(raised(), List.of("v00040-reference:BASELINE_DROP@35"));
    double[] difference = dropDifference();
    Alarm line = alarms().getFirst();
    Assert.assertEquals(line.estimate().mean(), difference[0], 1e-12);
    Assert.assertEquals(line.estimate().error(), difference[1], 1e-12);
  }

  /** 決闘のジョブ 1 つの結果を反映する前か後で、JVM が落ちたときのように例外を投げるランナー。 */
  private record Falling(JobRunner runner, String name, boolean applied) implements JobRunner {

    @Override
    public Node.Table plan(JobFile job) throws Exception {
      return runner.plan(job);
    }

    @Override
    public JobWork open(JobFile job, Path directory) throws Exception {
      return runner.open(job, directory);
    }

    @Override
    public void settle(JobFile job, Node.Table result) throws Exception {
      if (job.name().equals(name) && !applied) {
        throw new IllegalStateException("The JVM fell before applying " + name);
      }
      runner.settle(job, result);
      if (job.name().equals(name)) {
        throw new IllegalStateException("The JVM fell after applying " + name);
      }
    }
  }

  /** 王者との決闘のジョブを作り、その結果を反映する前か後で落とす。 */
  private void fall(
      PrismSettings settings,
      int version,
      Function<JobFile, DuelResult> champion,
      SyntheticDuels.Places places,
      boolean applied)
      throws Exception {
    create(settings, version, List.of("champion"));
    String name = RunDirectory.jobName(version, "champion");
    JobRunner runner =
        new Falling(
            new SyntheticDuels.Runner(new EvaluationJobs(run, settings, null, null), champion, places), name, applied);
    Assert.expectThrows(IllegalStateException.class, () -> jobs().runAll(runner, RunControl.detached(() -> false)));
    Assert.assertEquals(jobs().read(name).state(), JobFile.State.RUNNING);
    Assert.assertTrue(Files.exists(run.job(name).resolve(Jobs.RESULT)));
  }

  @Test(timeOut = 600_000)
  public void resumingAPromotionGivesTheSameBaselineJobs() throws Exception {
    PrismSettings settings = beforeFrom();
    crown(34);
    Function<JobFile, DuelResult> champion = verdicts(Map.of(35, Verdict.BETTER, 36, Verdict.BETTER));
    SyntheticDuels.Places places = (job, wall, game) -> even(wall, game);

    // バージョン 35: 結果を書いた後、反映する前に落ちる。続きから回すと昇格させ、基準のジョブを足して打つ
    fall(settings, 35, champion, places, false);
    Assert.assertEquals(names(), List.of("v00035-champion"));
    Assert.assertEquals(ChampionStore.open(run.champion()).current().orElseThrow().version(), 34);
    runAll(settings, champion, places);
    Assert.assertEquals(names(), List.of("v00035-champion", "v00035-reference"));
    Assert.assertEquals(ChampionStore.open(run.champion()).current().orElseThrow().version(), 35);

    // バージョン 36: 反映した後、DONE にする前に落ちる。足したジョブは、続きから回しても作り直さずに 1 つのまま打つ
    fall(settings, 36, champion, places, true);
    List<String> measured = List.of("v00035-champion", "v00035-reference", "v00036-champion", "v00036-reference");
    Assert.assertEquals(names(), measured);
    JobFile added = jobs().read("v00036-reference");
    Assert.assertEquals(added.state(), JobFile.State.PENDING);
    runAll(settings, champion, places);
    Assert.assertEquals(names(), measured);
    Assert.assertEquals(jobs().read("v00036-reference").created(), added.created());
    Assert.assertEquals(ChampionStore.open(run.champion()).current().orElseThrow().version(), 36);
    Assert.assertEquals(
        ChampionStore.open(run.champion()).history().stream()
            .filter(entry -> entry.event() instanceof StoreEvent.Promoted)
            .count(),
        2L);
  }

  // ラダーの相手が、基準の卓の相手と同じになる回

  @Test(timeOut = 600_000)
  public void aLadderThatMeetsTheBaselineOpponentIsJudgedByTheBaseline() throws Exception {
    // 三人打ちの系譜で勧める卓を縮めたもの: 基準は version:4 を v8 から 4 バージョンごと、ラダーは 4 つ前（v8 から）と 8 つ前（v12 から）。v8 では 4 つ前が、v12 では 8 つ前が基準と重なる
    PrismSettings settings =
        lineage(
            16,
            concat(
                table("reference", "version:4", "FIXED", 300, "FIXED", 4, 8, false),
                table("ladder4", "version:-4", "FRESH", 300, "FIXED", 4, 8, null),
                table("ladder8", "version:-8", "FRESH", 300, "FIXED", 4, 12, null)));
    Map<String, Double> targets =
        Map.of(
            "v00008-reference", 0.4,
            "v00012-reference", 0.4,
            "v00012-ladder4", -0.3,
            "v00016-reference", 0.4,
            "v00016-ladder4", -0.3,
            "v00016-ladder8", 0.4);
    SyntheticDuels.Places places = SyntheticDuels.drawn(job -> targets.get(job.name()));

    Assert.assertEquals(schedule(settings, 8), List.of("reference"));
    Assert.assertEquals(settings.evaluation().coveredLadders("reference", 8), List.of("ladder4"));
    runAll(settings, SyntheticDuels::noChampion, places);
    Assert.assertEquals(raised(), List.of("v00008-reference:LADDER_WORSE"));

    Assert.assertEquals(schedule(settings, 12), List.of("ladder4", "reference"));
    Assert.assertEquals(settings.evaluation().coveredLadders("reference", 12), List.of("ladder8"));
    runAll(settings, SyntheticDuels::noChampion, places);
    Assert.assertEquals(raised(), List.of("v00008-reference:LADDER_WORSE", "v00012-reference:LADDER_WORSE"));

    // v16 ではどのラダーも基準と重ならない: 両方のラダーを打ち、基準の結果ではラダーの低下を出さない
    Assert.assertEquals(schedule(settings, 16), List.of("ladder4", "ladder8", "reference"));
    Assert.assertEquals(settings.evaluation().coveredLadders("reference", 16), List.of());
    runAll(settings, SyntheticDuels::noChampion, places);
    Assert.assertEquals(
        raised(),
        List.of(
            "v00008-reference:LADDER_WORSE", "v00012-reference:LADDER_WORSE", "v00016-ladder8:LADDER_WORSE"));
    Assert.assertEquals(
        names(),
        List.of(
            "v00008-reference", "v00012-ladder4", "v00012-reference", "v00016-ladder4", "v00016-ladder8",
            "v00016-reference"));

    // ラダーの低下の値は、基準の結果の区間の符号を反転したもの（候補が良いと正になる向き）。その上端は 0 より小さい
    Alarm first = alarms().getFirst();
    DuelResult baseline = Jobs.result(run.job("v00008-reference"), DuelResult.class);
    Assert.assertEquals(first.signal(), Signal.LADDER_WORSE);
    Assert.assertEquals(first.estimate(), baseline.rank().negate());
    Assert.assertTrue(first.estimate().high() < 0, first.estimate().toString());

    // 予算: ラダーの回数は、打ったラダー 3 回と、基準の結果で判定した 2 回
    String budget =
        String.format(Locale.ROOT, "# LADDER_WORSE\t%.4f\t3", 5 * EvaluationJobs.FIXED_ALPHA / 2);
    String table = Files.readString(run.report(EvaluationReports.ALARMS), StandardCharsets.UTF_8);
    Assert.assertTrue(table.lines().anyMatch(budget::equals), table);
  }

  @Test(timeOut = 600_000)
  public void aLadderOnAnotherPlanIsPlayedWhenItMeetsTheBaselineOpponent() throws Exception {
    // 前のテストと同じ卓の組で、4 つ前のラダーは効用で判定し、8 つ前のラダーは牌山の位置の順にツモる。v8 と v12 で相手は基準と重なるが、
    // 判定に使う量かツモの順が基準の卓と違うので、ラダーを打ち、ラダーの結果で判定する
    PrismSettings settings =
        lineage(
            12,
            concat(
                table("reference", "version:4", "FIXED", 300, "FIXED", 4, 8, false),
                table("ladder4", "version:-4", "FRESH", 300, "FIXED", 4, 8, null),
                List.of("evaluation.tables.ladder4.score=UTILITY"),
                table("ladder8", "version:-8", "FRESH", 300, "FIXED", 4, 12, null),
                List.of("evaluation.tables.ladder8.draws=WALL_ORDER")));
    // 基準の卓はどの回も悪く、ラダーは v8 の 4 つ前だけ悪い
    Map<String, Double> targets =
        Map.of(
            "v00008-reference", 0.4,
            "v00008-ladder4", 0.4,
            "v00012-reference", 0.4,
            "v00012-ladder4", -0.3,
            "v00012-ladder8", -0.3);
    SyntheticDuels.Places places = SyntheticDuels.drawn(job -> targets.get(job.name()));

    Assert.assertEquals(schedule(settings, 8), List.of("ladder4", "reference"));
    Assert.assertEquals(settings.evaluation().coveredLadders("reference", 8), List.of());
    runAll(settings, SyntheticDuels::noChampion, places);
    Assert.assertEquals(raised(), List.of("v00008-ladder4:LADDER_WORSE"));
    // ラダーの低下の値は、ラダーの結果の効用の区間
    DuelResult ladder = Jobs.result(run.job("v00008-ladder4"), DuelResult.class);
    Assert.assertEquals(alarms().getFirst().estimate(), ladder.utility());

    Assert.assertEquals(schedule(settings, 12), List.of("ladder4", "ladder8", "reference"));
    Assert.assertEquals(settings.evaluation().coveredLadders("reference", 12), List.of());
    runAll(settings, SyntheticDuels::noChampion, places);
    Assert.assertEquals(raised(), List.of("v00008-ladder4:LADDER_WORSE"));
    Assert.assertEquals(
        names(),
        List.of("v00008-ladder4", "v00008-reference", "v00012-ladder4", "v00012-ladder8", "v00012-reference"));

    // 予算: ラダーの回数は打ったラダーの 3 回だけで、基準の卓の結果では数えない
    String budget =
        String.format(Locale.ROOT, "# LADDER_WORSE\t%.4f\t1", 3 * EvaluationJobs.FIXED_ALPHA / 2);
    String table = Files.readString(run.report(EvaluationReports.ALARMS), StandardCharsets.UTF_8);
    Assert.assertTrue(table.lines().anyMatch(budget::equals), table);
  }

  // 基準の低下

  /** 全ての牌山で δ が 0 になる着順（3 半荘で 1〜3 位を 1 回ずつ）。 */
  private static int even(int wall, int game) {
    return (wall + game) % 3;
  }

  /**
   * 王者 2・候補 3 の基準の卓を合成の値で打ち、出たアラートを返す。候補は、番号（0 始まり）が 6 の倍数の牌山（60 のうち 10）では 3 半荘とも 3 位で、
   * ほかは王者と同じ。対にした差（候補が良いと正になる向き）は −1 が 10 個と 0 が 50 個で、平均は −1/6、標準誤差は √(Σ(x − m)² / 59 / 60)、
   * z は約 3.44。
   */
  private List<String> dropRun(String... evaluation) throws Exception {
    PrismSettings settings =
        lineage(3, concat(table("reference", "version:1", "FIXED", 60, "FIXED", 1, 2, false), List.of(evaluation)));
    crown(2);
    SyntheticDuels.Places places =
        (job, wall, game) -> job.version() == 3 && wall % 6 == 0 ? 2 : even(wall, game);
    schedule(settings, 2);
    schedule(settings, 3);
    runAll(settings, SyntheticDuels::noChampion, places);
    return raised();
  }

  /** {@code dropRun} で対にした差の、平均と標準誤差（手計算）。 */
  private static double[] dropDifference() {
    double mean = -10 / 60.0;
    double squares = 10 * Math.pow(-1 - mean, 2) + 50 * Math.pow(0 - mean, 2);
    return new double[] {mean, Math.sqrt(squares / 59 / 60)};
  }

  @Test(timeOut = 600_000)
  public void theBaselineDropNeedsTheMarginAndTheZLine() throws Exception {
    double[] difference = dropDifference();
    double z = -difference[0] / difference[1];
    Assert.assertTrue(z > 3.4 && z < 3.5, "z " + z);

    // 既定（幅 0.01、z 3）: 出る。値は対にした差の平均と標準誤差で、区間は z のしきい値で作る
    Assert.assertEquals(dropRun(), List.of("v00003-reference:BASELINE_DROP@2"));
    Alarm line = alarms().getFirst();
    Assert.assertEquals(line.version(), 3);
    Assert.assertEquals(line.estimate().mean(), difference[0], 1e-12);
    Assert.assertEquals(line.estimate().error(), difference[1], 1e-12);
    Assert.assertEquals(line.estimate().high(), difference[0] + 3 * difference[1], 1e-12);
    // 合成の値は、games.tsv から読んだ牌山ごとの値と同じ
    JobFile job = jobs().read("v00003-reference");
    WallValues values =
        WallValues.of(
            RecordFile.read(EvaluationPlan.class, job.plan()).value().duel(),
            DurableFiles.lines(run.job(job.name()).resolve(JobOutputs.GAMES)));
    for (int wall = 0; wall < 60; wall++) {
      Assert.assertEquals(values.rank()[wall], wall % 6 == 0 ? 1.0 : 0.0, 1e-12, "wall " + wall);
    }
  }

  @Test(timeOut = 600_000)
  public void insideTheMarginNoBaselineDrop() throws Exception {
    // |平均| 1/6 は幅 0.17 の内側
    Assert.assertEquals(dropRun("evaluation.dropMargin=0.17"), List.of());
  }

  @Test(timeOut = 600_000)
  public void justOutsideTheMarginABaselineDrop() throws Exception {
    Assert.assertEquals(dropRun("evaluation.dropMargin=0.16"), List.of("v00003-reference:BASELINE_DROP@2"));
  }

  @Test(timeOut = 600_000)
  public void belowTheZLineNoBaselineDrop() throws Exception {
    Assert.assertEquals(dropRun("evaluation.alarmZ=3.5"), List.of());
  }

  @Test(timeOut = 600_000)
  public void aboveTheZLineABaselineDrop() throws Exception {
    Assert.assertEquals(dropRun("evaluation.alarmZ=3.4"), List.of("v00003-reference:BASELINE_DROP@2"));
  }

  @Test(timeOut = 600_000)
  public void theBaselineDropIsRaisedOncePerChampionAndTable() throws Exception {
    PrismSettings settings =
        lineage(
            6,
            concat(
                table("reference", "version:1", "FIXED", 60, "FIXED", 1, 2, false),
                table("second", "version:1", "FIXED", 60, "FIXED", 1, 2, false)));
    crown(2);
    // v3・v4 は v2 より 10 牌山だけ悪い。v5 は v4 よりさらに 10 牌山だけ悪い（v4 が王者になった後に出る）。v6 は v4 と同じ
    SyntheticDuels.Places places =
        (job, wall, game) -> {
          boolean worse =
              switch (job.version()) {
                case 3, 4, 6 -> wall % 6 == 0;
                case 5 -> wall % 6 == 0 || wall % 6 == 1;
                default -> false;
              };
          return worse ? 2 : even(wall, game);
        };
    for (int version = 2; version <= 4; version++) {
      schedule(settings, version);
      runAll(settings, SyntheticDuels::noChampion, places);
    }
    List<String> once = List.of("v00003-reference:BASELINE_DROP@2", "v00003-second:BASELINE_DROP@2");
    Assert.assertEquals(raised(), once, "v4 is worse too, but the champion is the same");
    settleAgain(settings, "v00003-reference", places);
    Assert.assertEquals(raised(), once);

    promote(4);
    schedule(settings, 5);
    runAll(settings, SyntheticDuels::noChampion, places);
    List<String> again =
        List.of(
            "v00003-reference:BASELINE_DROP@2",
            "v00003-second:BASELINE_DROP@2",
            "v00005-reference:BASELINE_DROP@4",
            "v00005-second:BASELINE_DROP@4");
    Assert.assertEquals(raised(), again);
    double[] difference = dropDifference();
    Assert.assertEquals(alarms().get(2).estimate().mean(), difference[0], 1e-12);

    // 王者と同じ値のバージョンでは出ない
    schedule(settings, 6);
    runAll(settings, SyntheticDuels::noChampion, places);
    Assert.assertEquals(raised(), again);
  }

  @Test(timeOut = 600_000)
  public void jobsPlayedWithAnotherTwinDiscardAreNotPaired() throws Exception {
    List<String> tables = new ArrayList<>(table("reference", "version:1", "FIXED", 60, "FIXED", 1, 2, false));
    tables.addAll(table("second", "version:1", "FIXED", 60, "FIXED", 1, 2, false));
    PrismSettings settings = lineage(3, concat(tables));
    crown(2);
    SyntheticDuels.Places places =
        (job, wall, game) -> job.version() == 3 && wall % 6 == 0 ? 2 : even(wall, game);
    schedule(settings, 2);
    runAll(settings, SyntheticDuels::noChampion, places);
    // 設定の 2 行目で切り方を手出しに替える。候補の reference は 2 行目で、second は王者と同じ 1 行目で作る
    PrismSettings tedashi =
        ThreePlayerSettings.append(root, List.of(), concat(tables, List.of("behaviour.twinDiscard=TEDASHI")));
    Assert.assertNotEquals(tedashi.behaviour().twinDiscard(), settings.behaviour().twinDiscard());
    Jobs jobs = jobs();
    jobs.create(JobFile.pending(3, "reference", 0, 2, RecordFile.tree(tedashi.evaluation().tables().get("reference"))));
    jobs.create(JobFile.pending(3, "second", 1, 1, RecordFile.tree(settings.evaluation().tables().get("second"))));
    runAll(settings, SyntheticDuels::noChampion, places);
    Assert.assertEquals(raised(), List.of("v00003-second:BASELINE_DROP@2"));
  }
}
