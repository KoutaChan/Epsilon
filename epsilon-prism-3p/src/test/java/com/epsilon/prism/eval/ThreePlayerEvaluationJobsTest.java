package com.epsilon.prism.eval;

import com.epsilon.agent.eval.Alarm;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.Duel;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.PlayStats;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.Signal;
import com.epsilon.agent.eval.Stopped;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.eval.WallValues;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
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
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 三人打ちの run の評価のジョブ（{@link EvaluationJobs}）を確かめる。2×R の卓（相手はルールベースの打ち手 2 席で、全てのバージョンで同じ
 * 牌山を使う）を CPU で打つ。王者の同じ卓と同じ牌山で対にした低下のアラートは、わざと作った悪い差では 1 回だけ出て、同じ王者の間はもう出ない。
 * 差が無ければ出ず、王者が替われば新しい王者に対して出る。王者の保存先が空なら、王者の卓は打たずに候補を最初の王者にする。昇格の結果を
 * 2 回反映しても、1 回と同じになる。
 *
 * <p>ジョブの計画の作り方と結果の反映は {@link EvaluationJobs} のまま使う。打ち手だけを、バージョンごとに決めた打ち手（ルールベースの打ち手か、
 * 乱数で選ぶ打ち手）に置き換える（このテストのバージョンはネットワークを持たない）。
 */
public class ThreePlayerEvaluationJobsTest {

  private static final Variant THREE = Variant.TENHOU_3P;
  private static final String ANCHOR = "rx2", CHAMPION = "champion";

  /** 2×R の卓で打つ牌山の数。このテストでは、計画の牌山を全て打つ止め方にする。 */
  private static final int WALLS = 30;

  private Path root;
  private RunDirectory run;

  @BeforeMethod
  public void createRun() throws Exception {
    root = Files.createTempDirectory("prism-3p-evaluation-jobs");
    run = new RunDirectory(root);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws Exception {
    DurableFiles.delete(root);
  }

  /** バージョンごとの打ち手。 */
  enum Player {
    /** ルールベースの打ち手（卓の相手と同じ方策）。 */
    RULES,
    /** 乱数で選ぶ打ち手。 */
    RANDOM
  }

  /**
   * 乱数で選ぶ打ち手。判断の時点の観測から一様乱数を作り、確率 1/2 で最初の合法手を選び、残りは合法手から一様に選ぶ。同じ判断の時点では
   * 同じ値を引くので、割り当てや実行の順に依らない。
   */
  record RandomPlayer(Variant variant) implements InlinePolicy {

    @Override
    public String label() {
      return "random";
    }

    @Override
    public int act(Observation o) {
      long state = mix(SeedMixer.mix64(11), o.observer());
      state = mix(state, o.kyoku() * 64L + o.honba());
      state = mix(state, o.remainingWall());
      state = mix(state, o.events());
      state = mix(state, o.drawnTile());
      for (int seat = 0; seat < variant.seats(); seat++) {
        state = mix(state, o.score(seat));
      }
      for (int kind = 0; kind < 34; kind++) {
        state = mix(state, o.handCount(kind) * 64L + kind);
      }
      int n = o.legalActions();
      for (int a = 0; a < n; a++) {
        state = mix(state, o.action(a).code());
      }
      double u = (state >>> 11) * 0x1.0p-53;
      return u < 0.5 ? 0 : Math.min(n - 1, (int) ((u - 0.5) * 2 * n));
    }

    private static long mix(long state, long value) {
      return SeedMixer.mix64(state ^ SeedMixer.mix64(value));
    }

    @Override
    public Evaluation evaluate(Observation o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  /**
   * 計画の作り方と結果の反映は評価のジョブのまま使い、候補はバージョンごとに決めた打ち手で打つ。打たずに候補を最初の
   * 王者にするジョブは、評価のジョブのまま開く。王者の卓で王者と打つジョブは、打たずに、渡した判定になる結果を返す。
   *
   * @param players バージョンから打ち手への対応
   * @param verdict 王者の卓の判定
   */
  record Runner(EvaluationJobs jobs, TableLoop table, Map<Integer, Player> players, Verdict verdict)
      implements JobRunner {

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
      if (job.table().equals(CHAMPION)) {
        DuelResult result = measured(verdict);
        return new JobWork() {
          @Override
          public Chunk next(long walls, JobOutputs outputs) {
            return Chunk.decided(result.walls(), result);
          }

          @Override
          public void close() {}
        };
      }
      DuelSetup duel = plan.duel();
      Policy candidate =
          switch (players.get(job.version())) {
            case RULES -> rules(duel);
            case RANDOM -> new RandomPlayer(THREE);
          };
      return Duel.open(job.name(), table, duel, candidate, rules(duel), directory);
    }

    @Override
    public void settle(JobFile job, Node.Table result) throws Exception {
      jobs.settle(job, result);
    }

    /** 計画にある相手の指定（heuristic）で開く、ルールベースの打ち手。 */
    private static Policy rules(DuelSetup duel) throws Exception {
      Assert.assertEquals(duel.opponent().spec(), EvaluationTable.HEURISTIC);
      return EvaluationTable.open(
          EvaluationTable.HEURISTIC, THREE, null, TwinDiscard.TSUMOGIRI, PolicyOptions.defaults(), null);
    }
  }

  /** 2×R の卓だけを持つ三人打ちの run を作る。バージョン 1〜{@code versions} を公開しておく。 */
  private PrismSettings anchorRun(int versions) throws Exception {
    String table = "evaluation.tables." + ANCHOR + ".";
    PrismSettings settings =
        ThreePlayerSettings.create(
            root,
            List.of(),
            table + "opponent=heuristic",
            table + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v2\"]}",
            table + "draws=SEAT_STREAMS",
            table + "wallSet=FIXED",
            table + "walls=" + WALLS,
            table + "stop=FIXED",
            table + "score=RANK",
            table + "every=1",
            table + "from=1");
    Assert.assertEquals(settings.evaluation().alarmZ(), 3.0);
    publish(versions);
    return settings;
  }

  private void publish(int versions) throws Exception {
    for (int version = 1; version <= versions; version++) {
      Files.createDirectories(run.snapshot(version));
      Files.writeString(run.snapshot(version).resolve("weights.bin"), "version " + version);
    }
  }

  /**
   * 卓のジョブを作り、済むまで実行する。評価のジョブは毎回作り直す（アラートを出した王者を、アラートの行から読み直せることも確かめる）。
   *
   * @return 済んだジョブ
   */
  private JobFile play(PrismSettings settings, String table, int version, Map<Integer, Player> players, Verdict verdict)
      throws Exception {
    Jobs jobs = new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS));
    TableSpec spec = settings.evaluation().tables().get(table);
    JobFile job = JobFile.pending(version, table, 0, 1, RecordFile.tree(spec));
    Assert.assertTrue(jobs.create(job), job.name());
    try (TableLoop loop = new TableLoop(new TableSettings(16, 4, 1000L, 4))) {
      Runner runner = new Runner(new EvaluationJobs(run, settings, null, null), loop, players, verdict);
      Assert.assertTrue(jobs.run(job, runner, RunControl.detached(() -> false)).isEmpty(), job.name());
    }
    JobFile done = jobs.read(job.name());
    Assert.assertEquals(done.state(), JobFile.State.DONE, done.name() + " " + done.reason());
    return done;
  }

  /** 済んだジョブの結果をもう一度反映する（落ちた後に反映し直すのと同じ）。 */
  private void settleAgain(PrismSettings settings, JobFile done) throws Exception {
    Jobs jobs = new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS));
    try (TableLoop loop = new TableLoop(new TableSettings(16, 4, 1000L, 4))) {
      Runner runner = new Runner(new EvaluationJobs(run, settings, null, null), loop, Map.of(), Verdict.EVEN);
      Assert.assertTrue(jobs.run(done, runner, RunControl.detached(() -> false)).isEmpty(), done.name());
    }
  }

  @Test(timeOut = 900_000)
  public void theAnchorDropIsRaisedOncePerChampionAndOnlyForARealDrop() throws Exception {
    PrismSettings settings = anchorRun(6);
    Map<Integer, Player> players =
        Map.of(1, Player.RULES, 2, Player.RULES, 3, Player.RANDOM, 4, Player.RANDOM, 5, Player.RULES, 6, Player.RANDOM);
    crown(1);

    // 王者のバージョン自身の 2×R は、対にしない
    play(settings, ANCHOR, 1, players, null);
    Assert.assertEquals(anchorAlarms(), List.of());
    // 王者と同じ方策のバージョン: 全ての牌山で差が 0 なので、アラートは出ない
    JobFile even = play(settings, ANCHOR, 2, players, null);
    Assert.assertEquals(anchorAlarms(), List.of());
    Assert.assertEquals(result(1).rank().mean(), 0.0, "the rule bot against itself");
    DuelSetup duel = RecordFile.read(EvaluationPlan.class, even.plan()).value().duel();
    WallValues values = WallValues.of(duel, DurableFiles.lines(run.job(even.name()).resolve(JobOutputs.GAMES)));
    Assert.assertEquals(values.walls(), WALLS);
    Assert.assertTrue(Arrays.equals(values.rank(), new double[WALLS]), Arrays.toString(values.rank()));
    Assert.assertTrue(Arrays.equals(values.utility(), new double[WALLS]), Arrays.toString(values.utility()));

    // わざと作った悪い差（乱数で選ぶ打ち手）: 王者 1 と対にして 1 回出る。反映し直しても、2 回目は書かない
    JobFile worse = play(settings, ANCHOR, 3, players, null);
    List<String> once = List.of("v00003-rx2:ANCHOR_DROP@1");
    Assert.assertEquals(anchorAlarms(), once);
    settleAgain(settings, worse);
    Assert.assertEquals(anchorAlarms(), once);
    Alarm line = lines().getFirst();
    Assert.assertEquals(line.version(), 3);
    Assert.assertEquals(line.handling(), Alarm.Handling.REPORT);
    // 値は同じ牌山で対にした差（今のバージョン − 王者を、候補が良いと正になる向きで）。王者の 2×R の δ は 0
    double expected = result(1).rank().mean() - result(3).rank().mean();
    Assert.assertTrue(expected < 0, "the random player is not worse: " + expected);
    Assert.assertEquals(line.estimate().mean(), expected, 1e-12);
    Assert.assertTrue(line.estimate().high() < 0, line.estimate().toString());

    // 同じ王者の間は、もっと悪いバージョンでも出さない
    play(settings, ANCHOR, 4, players, null);
    Assert.assertEquals(anchorAlarms(), once);
    // 差の無いバージョンを王者にすると、その王者に対しては改めて出る
    play(settings, ANCHOR, 5, players, null);
    Assert.assertEquals(anchorAlarms(), once);
    promote(5);
    play(settings, ANCHOR, 6, players, null);
    Assert.assertEquals(anchorAlarms(), List.of("v00003-rx2:ANCHOR_DROP@1", "v00006-rx2:ANCHOR_DROP@5"));
    Assert.assertEquals(lines().getLast().estimate().mean(), result(5).rank().mean() - result(6).rank().mean(), 1e-12);
  }

  /** 王者が同じ卓を打っていなければ、対にせず、アラートを何も書かない。 */
  @Test(timeOut = 900_000)
  public void noAlarmWithoutTheChampionsTable() throws Exception {
    PrismSettings settings = anchorRun(2);
    crown(1);
    play(settings, ANCHOR, 2, Map.of(2, Player.RANDOM), null);
    Assert.assertTrue(result(2).rank().mean() > 0, "the random player is not worse");
    Assert.assertEquals(anchorAlarms(), List.of());
  }

  @Test(timeOut = 600_000)
  public void anEmptyStoreCrownsTheFirstVersionAndAPromotionSettlesOnce() throws Exception {
    String table = "evaluation.tables." + CHAMPION + ".";
    PrismSettings settings =
        ThreePlayerSettings.create(
            root,
            List.of(),
            table + "opponent=champion",
            table + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v2\", \"2v1\"]}",
            table + "draws=SEAT_STREAMS",
            table + "wallSet=FRESH",
            table + "walls=4096",
            table + "stop={kind = \"TEST\", evenMargin = 0.01, alpha = 0.01, spending = 3.0, checkEvery = 512,"
                + " promote = true}",
            table + "score=RANK",
            table + "every=1",
            table + "from=2");
    publish(3);
    DurableFiles.write(run.state(), new PrismState(3, 0, PrismCarry.NONE, 3, Regression.NONE));

    // 王者の保存先が空: 打たずにバージョン 2 を最初の王者にする。反映し直しても 1 回だけ
    JobFile crowning = play(settings, CHAMPION, 2, Map.of(), Verdict.BETTER);
    Assert.assertEquals(Jobs.result(run.job(crowning.name()), EvaluationJobs.Crowned.class).champion(), "v00002");
    settleAgain(settings, crowning);
    ChampionStore store = ChampionStore.open(run.champion());
    Assert.assertEquals(store.history().size(), 1);
    Assert.assertTrue(store.history().getFirst().event() instanceof StoreEvent.Initial, store.history().toString());
    Assert.assertEquals(store.current().orElseThrow().version(), 2);

    // 王者より良い: バージョン 3 を昇格させる。反映し直しても、保存先・state・アラートは変わらない
    JobFile promoted = play(settings, CHAMPION, 3, Map.of(), Verdict.BETTER);
    EvaluationPlan plan = RecordFile.read(EvaluationPlan.class, promoted.plan()).value();
    Assert.assertEquals(plan.duel().opponent().version(), 2);
    Assert.assertEquals(plan.duel().variant(), THREE);
    Assert.assertEquals(plan.duel().splits(), List.of("1v2", "2v1"));
    String state = Files.readString(run.state(), StandardCharsets.UTF_8);
    settleAgain(settings, promoted);
    Assert.assertEquals(Files.readString(run.state(), StandardCharsets.UTF_8), state);
    store = ChampionStore.open(run.champion());
    Assert.assertEquals(store.history().size(), 2);
    Assert.assertTrue(store.history().getLast().event() instanceof StoreEvent.Promoted, store.history().toString());
    Assert.assertEquals(store.current().orElseThrow().version(), 3);
    Assert.assertEquals(
        RecordFile.read(PrismState.class, run.state()).value().regression(), new Regression(0, "v00003-champion", List.of()));
    Assert.assertFalse(Files.exists(run.report(Alarm.FILE)));
  }

  private void crown(int version) throws Exception {
    Champion champion = Champion.of(label(version), run.snapshot(version), version);
    Assert.assertTrue(ChampionStore.open(run.champion()).append(new StoreEvent.Initial(champion)));
  }

  private void promote(int version) throws Exception {
    Champion champion = Champion.of(label(version), run.snapshot(version), version);
    String duel = RunDirectory.jobName(version, CHAMPION);
    Assert.assertTrue(ChampionStore.open(run.champion()).append(new StoreEvent.Promoted(champion, duel)));
  }

  private DuelResult result(int version) throws Exception {
    return Jobs.result(run.job(RunDirectory.jobName(version, ANCHOR)), DuelResult.class);
  }

  private List<Alarm> lines() throws Exception {
    List<Alarm> lines = new ArrayList<>();
    for (String line : DurableFiles.lines(run.report(Alarm.FILE))) {
      lines.add(RecordFile.read(Alarm.class, Json.read(line, "alarm")).value());
    }
    return lines;
  }

  /** 低下のアラートの「名前@王者」を、書いた順に返す。 */
  private List<String> anchorAlarms() throws Exception {
    List<String> alarms = new ArrayList<>();
    for (Alarm line : lines()) {
      Assert.assertEquals(line.signal(), Signal.ANCHOR_DROP, line.toString());
      alarms.add(line.alarm() + "@" + line.champion());
    }
    return alarms;
  }

  /** 渡した判定になる決闘の結果。順位の区間は、候補が良いと正になる向きの区間の符号を反転したもの。 */
  private static DuelResult measured(Verdict verdict) {
    Estimate better =
        switch (verdict) {
          case BETTER -> new Estimate(0.2, 0.05, 0.1, 0.3);
          case WORSE -> new Estimate(-0.2, 0.05, -0.3, -0.1);
          default -> new Estimate(0.0, 0.01, -0.02, 0.02);
        };
    List<Double> shares = List.of(1 / 3.0, 1 / 3.0, 1 / 3.0);
    return new DuelResult(
        512,
        3072,
        verdict,
        Stopped.DECIDED,
        Score.RANK,
        3.0,
        better.negate(),
        new Estimate(0, 0, 0, 0),
        List.of(),
        List.of(
            new DuelResult.SlotValue("candidate", 2 - better.mean(), better.error(), 0, 0, shares),
            new DuelResult.SlotValue("v00002", 2, 0, 0, 0, shares)),
        new PlayStats.Rates(0, 0, 0, 0, 0),
        0,
        0,
        1.0);
  }

  private static String label(int version) {
    return String.format("v%05d", version);
  }
}
