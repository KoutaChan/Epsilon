package com.epsilon.classic.campaign;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.eval.Alarm;
import com.epsilon.agent.eval.Alarms;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.ChampionStore.Entry;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.OnWorseSettings;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.SeatingSettings;
import com.epsilon.agent.eval.Signal;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.eval.WallSet;
import com.epsilon.agent.eval.Watch;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.run.ControlRequest;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.classic.campaign.Decision.Action;
import com.epsilon.classic.rulebot.RuleBot;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import org.testng.annotations.Test;

/**
 * 決闘の段と昇格の段を、ルールベースの打ち手と乱択の打ち手の小さな決闘で CPU で回す。学習器の今のネットワークは、打ち手の種類を書いた
 * ディレクトリで表す。王者を替える卓は 10 バージョンごとに、牌山 16 ごとに 4 回判定し、確かめの卓はルールベースの打ち手 3 席を相手に、同じ
 * 牌山 16 個を打つ。
 */
public class DuelDecideTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final long SEED = 20261110L;
  private static final String JAR = "classes";
  private static final String CHAMPION = "champion", OUTSIDE = "outside";
  private static final double MARGIN = 0.004;

  /** 印のディレクトリの、打ち手の種類を書くファイル。 */
  private static final String PLAYER = "player";

  /** 打ち手の種類。 */
  private static final String RULES = "heuristic", RANDOM = "random", BROKEN = "broken";

  private static EvaluationSettings evaluation(OnWorseSettings onWorse) {
    TableSpec champion =
        new TableSpec(
            "champion",
            new SeatingSettings.EveryPlacement(List.of("1v3", "3v1")),
            DrawOrder.SEAT_STREAMS,
            WallSet.FRESH,
            64,
            new StopSettings.Test(MARGIN, 0.01, 3, 16, true),
            Score.UTILITY,
            null,
            OUTSIDE,
            null,
            10,
            10);
    TableSpec outside =
        new TableSpec(
            RULES,
            new SeatingSettings.EveryPlacement(List.of("1v3")),
            DrawOrder.SEAT_STREAMS,
            WallSet.FIXED,
            16,
            new StopSettings.Fixed(),
            Score.UTILITY,
            null,
            null,
            MARGIN,
            0,
            0);
    return new EvaluationSettings(null, 3.0, 0.01, onWorse, Map.of(CHAMPION, champion, OUTSIDE, outside));
  }

  @Test(timeOut = 600_000)
  public void theFirstChampionIsCrownedAndABetterCandidateIsConfirmedAndPromoted() throws Exception {
    Path root = Files.createTempDirectory("classic-duel-promote");
    try (TableLoop table = new TableLoop(new TableSettings(64, 8, 1000L, 4))) {
      EvaluationSettings evaluation = evaluation(new OnWorseSettings.Pause(1));
      // 予定でない macro では、どちらの段も行わない
      Macro between = new Macro(root, 15, evaluation, RANDOM, table);
      assertFalse(new DuelStep<Macro>().enabled(between));
      assertFalse(new DecideStep<Macro>().enabled(between));

      // macro 10: 王者の保存先が空なので打たずに、乱択の打ち手を初めの王者にし、確かめの卓で測っておく
      Macro first = new Macro(root, 10, evaluation, RANDOM, table);
      Decision crowned = steps(first);
      assertTrue(DuelStep.recorded(first).crowns());
      assertEquals(crowned.action(), Action.CROWN);
      ChampionStore store = ChampionStore.open(first.run().champion());
      Champion initial = store.current().orElseThrow();
      assertEquals(initial.label(), "v00010");
      assertEquals(initial.version(), 10);
      Path measured = confirms(store,initial).resolve(crowned.check());
      assertEquals(crowned.check(), "v00010-outside");
      assertTrue(Files.exists(measured.resolve(Jobs.RESULT)), measured.toString());

      // macro 20: ルールベースの打ち手が王者に良く、確かめの卓でも王者より良いので王者になる
      Macro second = new Macro(root, 20, evaluation, RULES, table);
      Decision promoted = steps(second);
      DuelReport duel = DuelStep.recorded(second);
      assertEquals(duel.verdict(), Verdict.BETTER, duel.toString());
      assertEquals(duel.champion(), initial);
      assertEquals(promoted.action(), Action.PROMOTE, promoted.toString());
      assertEquals(promoted.check(), "v00020-outside");
      assertTrue(promoted.confirm().low() > -MARGIN, promoted.toString());
      assertEquals(promoted.alarms(), List.of());
      Champion candidate = duel.candidate();
      assertEquals(store.current().orElseThrow(), candidate);
      assertEquals(store.past(), List.of(initial));
      assertEquals(
          events(store),
          List.of(new StoreEvent.Initial(initial), new StoreEvent.Promoted(candidate, "v00020-champion")));
      // 確かめの卓で測った候補の値は、新しい王者の値として残る
      assertTrue(Files.exists(confirms(store,candidate).resolve("v00020-outside").resolve(Jobs.RESULT)));

      // 昇格の段を 2 度当てても、1 度と同じ。完了マーカーの前に落ちた形でも同じ
      byte[] history = Files.readAllBytes(first.run().champion().resolve(ChampionStore.HISTORY));
      byte[] current = Files.readAllBytes(first.run().champion().resolve(ChampionStore.CURRENT));
      assertEquals(new DecideStep<Macro>().run(second).metrics(), promoted);
      Files.delete(second.work().resolve(DecideStep.FILE));
      assertEquals(new DecideStep<Macro>().run(second).metrics(), promoted);
      assertEquals(Files.readAllBytes(first.run().champion().resolve(ChampionStore.HISTORY)), history);
      assertEquals(Files.readAllBytes(first.run().champion().resolve(ChampionStore.CURRENT)), current);
      assertEquals(new ControlRequests(second.run()).pending(), List.of());
    } finally {
      DurableFiles.delete(root);
    }
  }

  @Test(timeOut = 600_000)
  public void aWorseDuelHoldsTheChampionAndPausesTheRunOnce() throws Exception {
    Path root = Files.createTempDirectory("classic-duel-hold");
    try (TableLoop table = new TableLoop(new TableSettings(64, 8, 1000L, 4))) {
      Macro macro = new Macro(root, 20, evaluation(new OnWorseSettings.Pause(1)), RANDOM, table);
      Champion champion = crown(macro, 10, RULES);
      Decision held = steps(macro);
      DuelReport duel = DuelStep.recorded(macro);
      assertEquals(duel.verdict(), Verdict.WORSE, duel.toString());
      assertEquals(held.action(), Action.HOLD, held.toString());
      assertEquals(
          held.alarms().stream().map(Alarm::signal).toList(), List.of(Signal.CHAMPION_WORSE, Signal.WORSE_STREAK));
      assertEquals(held.alarms().get(1).handling(), Alarm.Handling.PAUSE);
      Alarms alarms = new Alarms(Watch.of(macro.evaluation()), macro.run());
      assertEquals(alarms.read(), held.alarms());
      ChampionStore store = ChampionStore.open(macro.run().champion());
      assertEquals(store.current().orElseThrow(), champion);
      assertEquals(events(store), List.of(new StoreEvent.Initial(champion), new StoreEvent.Paused("v00020-champion")));
      assertEquals(pauses(macro), 1);

      // 当て直しても、一時停止の要求とアラートの行は 1 つずつのまま
      Files.delete(macro.work().resolve(DecideStep.FILE));
      assertEquals(new DecideStep<Macro>().run(macro).metrics(), held);
      assertEquals(pauses(macro), 1);
      assertEquals(alarms.read(), held.alarms());
      assertEquals(store.history().size(), 2);
    } finally {
      DurableFiles.delete(root);
    }
  }

  @Test(timeOut = 600_000)
  public void aBetterCandidateThatTheConfirmTableRejectsLeavesTheChampion() throws Exception {
    Path root = Files.createTempDirectory("classic-duel-confirm");
    try (TableLoop table = new TableLoop(new TableSettings(64, 8, 1000L, 4))) {
      Macro macro = new Macro(root, 20, evaluation(new OnWorseSettings.Pause(1)), RANDOM, table);
      Champion champion = crown(macro, 10, RULES);
      // 決闘では良いと出た形（乱択の打ち手を候補に置いた決闘の段の報告）
      ChampionStore store = ChampionStore.open(macro.run().champion());
      Path checkpoint = store.candidate("v00020");
      macro.writeLearner(checkpoint);
      Champion candidate = Champion.of("v00020", checkpoint, 20);
      DuelReport better =
          new DuelReport("v00020-champion", candidate, champion, Verdict.BETTER, false, new Estimate(0.1, 0.02, 0.05, 0.15));
      new StepReport(StepReport.Status.RAN, 0, 1, JAR, better).write(macro.work().resolve(DuelStep.FILE));

      StepReport report = new DecideStep<Macro>().run(macro);
      Decision decision = (Decision) report.metrics();
      assertEquals(decision.action(), Action.NOT_CONFIRMED, decision.toString());
      assertEquals(decision.check(), "v00020-outside");
      assertTrue(decision.confirm().low() <= -MARGIN, decision.toString());
      assertTrue(decision.confirm().mean() < 0, decision.toString());
      assertEquals(store.current().orElseThrow(), champion);
      assertEquals(store.past(), List.of());
      assertEquals(
          events(store),
          List.of(
              new StoreEvent.Initial(champion),
              new StoreEvent.NotConfirmed(candidate, "v00020-champion", "v00020-outside")));
      // 王者を王者になった後の初めての良いで測り、次からはその値を使う
      assertTrue(Files.exists(confirms(store,champion).resolve("v00020-outside").resolve(Jobs.RESULT)));
      assertEquals(pauses(macro), 0);
    } finally {
      DurableFiles.delete(root);
    }
  }

  @Test(timeOut = 600_000)
  public void aFailedDuelIsUndecidedAndKeepsTheChampion() throws Exception {
    Path root = Files.createTempDirectory("classic-duel-failed");
    try (TableLoop table = new TableLoop(new TableSettings(64, 8, 1000L, 4))) {
      Macro macro = new Macro(root, 20, evaluation(new OnWorseSettings.Pause(1)), BROKEN, table);
      Champion champion = crown(macro, 10, RULES);
      Decision kept = steps(macro);
      DuelReport duel = DuelStep.recorded(macro);
      assertTrue(duel.failed(), duel.toString());
      assertEquals(duel.verdict(), Verdict.UNDECIDED);
      assertNull(duel.better());
      Path duels = ChampionStore.open(macro.run().champion()).duels();
      JobFile job = DuelJobs.jobs(duels).read("v00020-champion");
      assertEquals(job.state(), JobFile.State.FAILED);
      assertEquals(job.attempts(), Jobs.MAX_FAILURES);
      assertEquals(kept.action(), Action.KEEP, kept.toString());
      assertEquals(kept.alarms(), List.of());
      ChampionStore store = ChampionStore.open(macro.run().champion());
      assertEquals(store.current().orElseThrow(), champion);
      assertEquals(events(store), List.of(new StoreEvent.Initial(champion)));
      assertEquals(pauses(macro), 0);
    } finally {
      DurableFiles.delete(root);
    }
  }

  @Test(timeOut = 600_000)
  public void aFailedChampionMeasurementIsPlayedAgainAtTheNextBetterDuel() throws Exception {
    Path root = Files.createTempDirectory("classic-duel-measure-again");
    try (TableLoop table = new TableLoop(new TableSettings(64, 8, 1000L, 4))) {
      EvaluationSettings evaluation = evaluation(new OnWorseSettings.Pause(1));
      // macro 20: 決闘は良いと出たが、打ち手を開けず、王者を確かめの卓で測るジョブが FAILED になる
      Macro outage = new Macro(root, 20, evaluation, RULES, table).unavailable();
      Champion champion = crown(outage, 10, RANDOM);
      ChampionStore store = ChampionStore.open(outage.run().champion());
      Champion rejected = better(outage, champion);
      Decision failed = (Decision) new DecideStep<Macro>().run(outage).metrics();
      assertEquals(failed.action(), Action.NOT_CONFIRMED, failed.toString());
      assertNull(failed.confirm());
      Jobs measured = DuelJobs.jobs(confirms(store,champion));
      assertEquals(measured.read("v00020-outside").state(), JobFile.State.FAILED);
      assertEquals(store.current().orElseThrow(), champion);

      // macro 30: 打ち手を開けるようになったので、王者を測り直して候補と比べる
      Macro recovered = new Macro(root, 30, evaluation, RULES, table);
      Champion candidate = better(recovered, champion);
      Decision promoted = (Decision) new DecideStep<Macro>().run(recovered).metrics();
      assertEquals(promoted.action(), Action.PROMOTE, promoted.toString());
      assertEquals(promoted.check(), "v00030-outside");
      assertTrue(promoted.confirm().low() > -MARGIN, promoted.toString());
      assertEquals(measured.read("v00020-outside").state(), JobFile.State.DONE);
      assertEquals(store.current().orElseThrow(), candidate);
      assertEquals(
          events(store),
          List.of(
              new StoreEvent.Initial(champion),
              new StoreEvent.NotConfirmed(rejected, "v00020-champion", "v00020-outside"),
              new StoreEvent.Promoted(candidate, "v00030-champion")));
    } finally {
      DurableFiles.delete(root);
    }
  }

  @Test(timeOut = 600_000)
  public void theChampionIsMeasuredAgainWhenThePrecisionOfThePlayersChanges() throws Exception {
    Path root = Files.createTempDirectory("classic-duel-precision");
    try (TableLoop table = new TableLoop(new TableSettings(64, 8, 1000L, 4))) {
      EvaluationSettings evaluation = evaluation(new OnWorseSettings.Pause(1));
      // macro 10: 初めの王者を、FLOAT32 の打ち手で確かめの卓で測る
      Macro first = new Macro(root, 10, evaluation, RANDOM, table);
      assertEquals(steps(first).action(), Action.CROWN);
      Champion champion = ChampionStore.open(first.run().champion()).current().orElseThrow();
      assertEquals(first.opened(), List.of("v00010 FLOAT32"));

      // macro 20: 計算の型を BFLOAT16 に替えた後の良い。FLOAT32 で測った王者の値は使わず、王者も BFLOAT16 で測って候補と対にする
      Macro second = new Macro(root, 20, evaluation, RULES, table).computing(Precision.BFLOAT16);
      better(second, champion);
      Decision promoted = (Decision) new DecideStep<Macro>().run(second).metrics();
      assertEquals(promoted.action(), Action.PROMOTE, promoted.toString());
      assertEquals(second.opened(), List.of("v00010 BFLOAT16", "v00020 BFLOAT16"));

      // macro 30: 計算の型が同じなら、新しい王者（macro 20 の候補）を測った値をそのまま使う
      Macro third = new Macro(root, 30, evaluation, RULES, table).computing(Precision.BFLOAT16);
      better(third, ChampionStore.open(third.run().champion()).current().orElseThrow());
      new DecideStep<Macro>().run(third);
      assertEquals(third.opened(), List.of("v00030 BFLOAT16"));
    } finally {
      DurableFiles.delete(root);
    }
  }

  /**
   * 持ち主が置いた初めの王者（run の外のチェックポイント、バージョン 0）は、決闘の計画に置いたときの SHA-256 を持ち、開くときに照合する。
   * 置いた後に同じディレクトリへ別の打ち手を書き出し直すと、決闘はその打ち手を開かずに落ちる。
   */
  @Test(timeOut = 600_000)
  public void aPlacedChampionRewrittenAfterItWasPlacedIsNotOpened() throws Exception {
    Path root = Files.createTempDirectory("classic-duel-placed");
    try (TableLoop table = new TableLoop(new TableSettings(64, 8, 1000L, 4))) {
      EvaluationSettings evaluation = evaluation(new OnWorseSettings.Pause(1));
      Macro first = new Macro(root, 20, evaluation, RANDOM, table);
      Path outside = root.resolve("nano-stage8");
      marker(outside, RULES);
      Champion placed = Champion.placed("nano-stage8", outside, 0);
      ChampionStore store = ChampionStore.open(first.run().champion());
      store.append(new StoreEvent.Initial(placed));

      // 置いたままの王者とは打つ
      new DuelStep<Macro>().run(first);
      assertFalse(DuelStep.recorded(first).failed());
      assertEquals(first.opened(), List.of("v00020 FLOAT32", "nano-stage8 FLOAT32"));

      // 持ち主が同じディレクトリに別の打ち手を書き出し直した後の決闘は、王者を開かずに落ちる
      Files.writeString(outside.resolve(PLAYER), RANDOM, StandardCharsets.UTF_8);
      Macro second = new Macro(root, 30, evaluation, RANDOM, table);
      new DuelStep<Macro>().run(second);
      DuelReport duel = DuelStep.recorded(second);
      assertTrue(duel.failed(), duel.toString());
      JobFile refused = DuelJobs.jobs(store.duels()).read("v00030-champion");
      assertEquals(refused.state(), JobFile.State.FAILED);
      assertTrue(refused.reason().contains("but the duel was planned with " + placed.sha256()), refused.reason());
      assertEquals(second.opened(), List.of());

      // 計画の相手は系譜の外のチェックポイントで、置いたときの SHA-256 を持つ
      JobFile played = DuelJobs.jobs(store.duels()).read("v00020-champion");
      assertEquals(
          RecordFile.read(DuelSetup.class, played.plan()).value().opponent(),
          new DuelSetup.Entrant("nano:" + placed.checkpoint(), "nano-stage8", null, placed.sha256()));
    } finally {
      DurableFiles.delete(root);
    }
  }

  /** 決闘の段と昇格の段を回す。 @return 昇格の段の判定 */
  private static Decision steps(Macro macro) throws Exception {
    new DuelStep<Macro>().run(macro);
    assertTrue(new DuelStep<Macro>().done(macro).isPresent());
    Decision decision = (Decision) new DecideStep<Macro>().run(macro).metrics();
    assertEquals(new DecideStep<Macro>().done(macro).orElseThrow().metrics(), decision);
    return decision;
  }

  /** 王者の保存先が空のときに、打ち手を初めの王者に置く。 */
  private static Champion crown(Macro macro, int version, String player) throws IOException {
    ChampionStore store = ChampionStore.open(macro.run().champion());
    String label = DuelJobs.label(version);
    Path checkpoint = store.candidate(label);
    marker(checkpoint, player);
    Champion champion = Champion.of(label, checkpoint, version);
    store.append(new StoreEvent.Initial(champion));
    return champion;
  }

  /**
   * macro の学習器を候補に置き、決闘では王者に良いと出た形にする（決闘の段の報告を書く）。
   *
   * @return 候補
   */
  private static Champion better(Macro macro, Champion champion) throws IOException {
    String label = DuelJobs.label(macro.published());
    Path checkpoint = ChampionStore.open(macro.run().champion()).candidate(label);
    macro.writeLearner(checkpoint);
    Champion candidate = Champion.of(label, checkpoint, macro.published());
    DuelReport better =
        new DuelReport(
            label + "-champion", candidate, champion, Verdict.BETTER, false, new Estimate(0.1, 0.02, 0.05, 0.15));
    new StepReport(StepReport.Status.RAN, 0, 1, JAR, better).write(macro.work().resolve(DuelStep.FILE));
    return candidate;
  }

  /** @return FLOAT32 の打ち手で測った、確かめの卓のジョブのディレクトリ */
  private static Path confirms(ChampionStore store, Champion player) {
    return store.confirms(Precision.FLOAT32, player);
  }

  private static List<StoreEvent> events(ChampionStore store) throws IOException {
    return store.history().stream().map(Entry::event).toList();
  }

  /** @return 待っている一時停止の要求の数 */
  private static long pauses(Macro macro) throws IOException {
    List<ControlRequest> pending = new ControlRequests(macro.run()).pending();
    return pending.stream().filter(request -> request.request() instanceof Request.Pause).count();
  }

  /** 打ち手の種類と名前を書いた印のディレクトリを、全てを書いてから置く。名前を書くので、どの印も中身が違う。 */
  private static void marker(Path directory, String player) throws IOException {
    Path temporary = directory.resolveSibling(directory.getFileName() + ".tmp");
    Files.createDirectories(temporary);
    Files.writeString(temporary.resolve(PLAYER), player, StandardCharsets.UTF_8);
    Files.writeString(temporary.resolve("label"), directory.getFileName().toString(), StandardCharsets.UTF_8);
    Files.move(temporary, directory, StandardCopyOption.ATOMIC_MOVE);
  }

  /**
   * テストの macro。macro t はバージョン t + 1 を公開する。学習器の今のネットワークは、打ち手の種類を書いた印のディレクトリ。
   */
  private static final class Macro implements DuelMacro {

    private final RunDirectory run;
    private final int published;
    private final EvaluationSettings evaluation;
    private final String learner;
    private final TableLoop table;
    private final RunControl control = RunControl.detached(() -> false);
    private boolean unavailable;
    private Precision precision = Precision.FLOAT32;
    private final List<String> opened = new CopyOnWriteArrayList<>();

    /** @param published macro が公開したバージョン */
    Macro(Path root, int published, EvaluationSettings evaluation, String learner, TableLoop table) {
      this.run = new RunDirectory(root.resolve("run"));
      this.published = published;
      this.evaluation = evaluation;
      this.learner = learner;
      this.table = table;
    }

    /** どの打ち手も開けない macro にする（落ちた試みの代わり）。 */
    Macro unavailable() {
      unavailable = true;
      return this;
    }

    /** 打ち手の計算の型を替えた macro にする（反復に固定した行の inference.precision の代わり）。 */
    Macro computing(Precision other) {
      precision = other;
      return this;
    }

    /** @return この macro が開いた系譜の打ち手（開いた順に、重ねずに）。書き方は、チェックポイントの名前と計算の型 */
    List<String> opened() {
      return opened.stream().distinct().toList();
    }

    @Override
    public int number() {
      return published - 1;
    }

    @Override
    public int published() {
      return published;
    }

    @Override
    public Path work() {
      return run.work(number());
    }

    @Override
    public int settingsRow() {
      return 1;
    }

    @Override
    public String jar() {
      return JAR;
    }

    @Override
    public int launchRow() {
      return 1;
    }

    @Override
    public RunDirectory run() {
      return run;
    }

    @Override
    public EvaluationSettings evaluation() {
      return evaluation;
    }

    @Override
    public Variant variant() {
      return VARIANT;
    }

    @Override
    public RankUtility utility() {
      return RankUtility.TENHOU;
    }

    @Override
    public long seed() {
      return SEED;
    }

    @Override
    public Series series() {
      return Series.NANO;
    }

    @Override
    public TableLoop table() {
      return table;
    }

    @Override
    public RunControl control() {
      return control;
    }

    @Override
    public PolicyOptions options() {
      PolicyOptions defaults = PolicyOptions.defaults();
      return new PolicyOptions(defaults.device(), precision, defaults.maxBatch(), defaults.kernels());
    }

    @Override
    public NetworkRuntime runtime() {
      return null;
    }

    @Override
    public void writeLearner(Path directory) throws IOException {
      marker(directory, learner);
    }

    /** 系譜の打ち手は印を読んで開き、ほかは系列の実装で開く（確かめの卓のルールベースの打ち手）。 */
    @Override
    public Policy open(String spec) throws IOException {
      if (unavailable) {
        throw new IOException("The test players do not open now");
      }
      Series.Spec parsed = Series.parse(spec);
      if (parsed.series() != Series.NANO) {
        return DuelMacro.super.open(spec);
      }
      opened.add(parsed.checkpoint().getFileName() + " " + precision);
      String player = Files.readString(parsed.checkpoint().resolve(PLAYER), StandardCharsets.UTF_8);
      return switch (player) {
        case RULES -> new RuleBot(VARIANT);
        case RANDOM -> new RandomPlayer(VARIANT);
        default -> throw new IOException("The test player " + player + " does not open");
      };
    }
  }

  /**
   * 乱択の打ち手。判断の記録から一様値を作り、半分は最初の合法手（和了・槓・鳴き・立直が先に並ぶ）、残りは一様に選ぶ。同じ判断の時点では
   * 同じ値を引くので、割り当てや実行の順に依らない。
   */
  private record RandomPlayer(Variant variant) implements InlinePolicy {

    @Override
    public String label() {
      return RANDOM;
    }

    @Override
    public int act(Observation o) {
      long state = mix(SeedMixer.mix64(7), o.observer());
      state = mix(state, o.kyoku() * 64L + o.honba());
      state = mix(state, o.remainingWall());
      state = mix(state, o.events());
      state = mix(state, o.drawnTile());
      for (int seat = 0; seat < variant.seats(); seat++) {
        state = mix(state, o.score(seat));
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
}
