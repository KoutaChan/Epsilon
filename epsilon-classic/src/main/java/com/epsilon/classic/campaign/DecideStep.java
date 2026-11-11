package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.Alarm;
import com.epsilon.agent.eval.Alarms;
import com.epsilon.agent.eval.Arena;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.OnWorseSettings;
import com.epsilon.agent.eval.PairedWalls;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.eval.WallMean;
import com.epsilon.agent.eval.Watch;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.classic.campaign.Decision.Action;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 昇格の段。決闘の段の結果から判定を決め、判定を work の {@value #DECISION} に書いてから、run のアラートの行・王者の保存先・一時停止の
 * 要求に当てる。当て直すときは書いた判定を読み、アラートは行の名前で、王者の保存先には決闘の名前で 1 度だけ書き、待っている一時停止の要求が
 * あれば新しく置かないので、何度当てても 1 度と同じになる。
 *
 * <ul>
 *   <li>王者の保存先が空だった: 候補を初めの王者にする。確かめの卓があれば、その卓で王者を測っておく。
 *   <li>良い: 確かめの卓が無ければ、候補を王者にする。あれば候補を確かめの卓で測り、王者が王者になったときに同じ牌山で測った値と、牌山ごとに
 *       対にする。候補 − 王者の対の差の固定の 99% の区間の下端が −confirmMargin を上回れば候補を王者にし（前の王者は過去の王者になる）、
 *       外れたら王者を据え置いて記録する。どちらかを測るジョブが FAILED なら外れとする。前の macro で FAILED になった王者の測りは、
 *       次の良いのときに落ちた回数を戻して打ち直すので、一度の FAILED が後の候補を止めたままにしない。
 *   <li>悪い: これまでの王者の決闘の判定から悪いが続いた回数を数え、{@link Alarms} がアラートを決める。悪いが続いた回数が evaluation.onWorse
 *       の回数に達して、扱いが一時停止なら、王者を据え置いて一時停止の要求を置く（次の安全点で止まり、持ち主の再開を待つ）。
 *   <li>互角・未決・FAILED の決闘: 王者を据え置いて続ける。
 * </ul>
 *
 * <p>確かめの卓のジョブは、王者の保存先の打ち手の計算の型・打ち手のバージョンごとのディレクトリ（{@link ChampionStore#confirms}）に
 * 置く。打ち手の計算の型が替われば、王者を同じ牌山で測り直す。段の報告は work の {@value #FILE}（完了マーカー）。
 *
 * @param <I> campaign の反復
 */
public final class DecideStep<I extends DuelMacro> implements Step<I> {

  /** 段の報告（work の中）。 */
  static final String FILE = "DECIDE.json";

  /** 当てる前に書く判定（work の中）。 */
  static final String DECISION = "DECISION.json";

  private static final Logger LOG = LoggerFactory.getLogger(DecideStep.class);

  @Override
  public boolean enabled(I macro) {
    return DuelStep.table(macro).isPresent();
  }

  @Override
  public Optional<StepReport> done(I macro) throws IOException {
    Path file = macro.work().resolve(FILE);
    return Files.exists(file) ? Optional.of(StepReport.read(file, Decision.class)) : Optional.empty();
  }

  @Override
  public StepReport run(I macro) throws Exception {
    long started = System.nanoTime();
    Path file = macro.work().resolve(DECISION);
    Decision decision;
    if (Files.exists(file)) {
      decision = RecordFile.read(Decision.class, file).value();
    } else {
      decision = decide(macro, DuelStep.recorded(macro));
      DurableFiles.write(file, decision);
    }
    apply(macro, decision);
    StepReport report = StepReport.ran(macro, started, decision);
    report.write(macro.work().resolve(FILE));
    return report;
  }

  private static Decision decide(DuelMacro macro, DuelReport duel) throws Exception {
    TableSpec table = macro.evaluation().tables().get(DuelStep.table(macro).orElseThrow());
    if (duel.crowns()) {
      String check = table.confirm() == null ? null : RunDirectory.jobName(macro.published(), table.confirm());
      return new Decision(Action.CROWN, duel.job(), duel.candidate(), check, null, List.of());
    }
    List<Alarm> alarms = alarms(macro, duel);
    return switch (duel.verdict()) {
      case BETTER -> confirm(macro, table, duel, alarms);
      case WORSE -> {
        boolean hold = alarms.stream().anyMatch(alarm -> alarm.handling() == Alarm.Handling.PAUSE);
        yield new Decision(hold ? Action.HOLD : Action.KEEP, duel.job(), duel.candidate(), null, null, alarms);
      }
      case EVEN, UNDECIDED -> new Decision(Action.KEEP, duel.job(), duel.candidate(), null, null, alarms);
    };
  }

  /** 良いが出た候補を、確かめの卓で王者と比べる（確かめの卓が無ければ、そのまま王者にする）。 */
  private static Decision confirm(DuelMacro macro, TableSpec table, DuelReport duel, List<Alarm> alarms)
      throws Exception {
    if (table.confirm() == null) {
      return new Decision(Action.PROMOTE, duel.job(), duel.candidate(), null, null, alarms);
    }
    String name = table.confirm();
    TableSpec check = macro.evaluation().tables().get(name);
    String job = RunDirectory.jobName(macro.published(), name);
    Optional<DuelJobs.Played> champion = measure(macro, name, check, duel.champion());
    Optional<DuelJobs.Played> candidate =
        champion.isPresent() ? measure(macro, name, check, duel.candidate()) : Optional.empty();
    if (candidate.isEmpty()) {
      LOG.warn(
          "{}: the confirm table {} failed for {}, so {} stays the champion{}",
          duel.job(),
          name,
          champion.isEmpty() ? duel.champion().label() : duel.candidate().label(),
          duel.champion().label(),
          champion.isEmpty() ? "; the next better duel measures the champion again" : "");
      return new Decision(Action.NOT_CONFIRMED, duel.job(), duel.candidate(), job, null, alarms);
    }
    WallMean pairs = PairedWalls.difference(candidate.get().values(), champion.get().values());
    Estimate difference = Estimate.of(pairs.mean(), pairs.error(), candidate.get().result().critical());
    boolean confirmed = difference.low() > -check.confirmMargin();
    LOG.info(
        "{}: on the confirm table {}, {} - {} = {} [{}, {}] over {} walls against the margin {}: {}",
        duel.job(),
        name,
        duel.candidate().label(),
        duel.champion().label(),
        String.format(Locale.ROOT, "%+.4f", difference.mean()),
        String.format(Locale.ROOT, "%+.4f", difference.low()),
        String.format(Locale.ROOT, "%+.4f", difference.high()),
        pairs.walls(),
        check.confirmMargin(),
        confirmed ? "confirmed" : "not confirmed");
    return new Decision(
        confirmed ? Action.PROMOTE : Action.NOT_CONFIRMED, duel.job(), duel.candidate(), job, difference, alarms);
  }

  /**
   * 確かめの卓で打ち手を測ったジョブ。まだ測っていなければ、この macro のジョブとして測る。
   *
   * @param name 確かめの卓の名前
   * @param check 確かめの卓
   * @return 済んだジョブ。FAILED なら空
   */
  private static Optional<DuelJobs.Played> measure(DuelMacro macro, String name, TableSpec check, Champion player)
      throws Exception {
    DuelSetup setup =
        DuelJobs.setup(
            macro, check, DuelJobs.lineage(macro, player), DuelJobs.outside(check.opponent()), Arena.EVALUATION_SEED);
    Path directory =
        ChampionStore.open(macro.run().champion()).confirms(macro.options().precision(), player);
    return DuelJobs.play(macro, directory, measurement(macro, directory, name, check), setup);
  }

  /**
   * @param directory 打ち手の確かめの卓のジョブのディレクトリ
   * @return 打ち手を確かめの卓で測ったジョブ。前の macro で FAILED になっていれば、落ちた回数を戻して続きを待つジョブ。まだ無ければ、
   *     この macro に作るジョブ
   */
  private static JobFile measurement(DuelMacro macro, Path directory, String name, TableSpec check)
      throws IOException {
    Jobs jobs = DuelJobs.jobs(directory);
    for (JobFile job : jobs.list().jobs()) {
      if (!job.table().equals(name)) {
        continue;
      }
      if (job.state() == JobFile.State.FAILED && job.version() < macro.published()) {
        LOG.info("{} in {} FAILED at an earlier macro ({}); measuring again", job.name(), directory, job.reason());
        return jobs.retry(job.name());
      }
      return job;
    }
    return JobFile.pending(macro.published(), name, 0, macro.settingsRow(), RecordFile.tree(check));
  }

  /**
   * これまでの王者の決闘（バージョンの順）とこの決闘の判定から悪いが続いた回数を数え、この決闘のアラートを決める。FAILED の決闘は未決として
   * 数える。
   *
   * @return この決闘のアラート
   * @throws IllegalStateException 悪いが続いたときの扱いが王者へ戻す（classic の campaign は戻さない）とき
   */
  private static List<Alarm> alarms(DuelMacro macro, DuelReport duel) throws IOException {
    EvaluationSettings evaluation = macro.evaluation();
    if (evaluation.onWorse() instanceof OnWorseSettings.Rollback) {
      throw new IllegalStateException(
          "A classic campaign reports or pauses when the champion duels stay worse; it does not roll back");
    }
    String table = DuelStep.table(macro).orElseThrow();
    Path duels = ChampionStore.open(macro.run().champion()).duels();
    int streak = 0;
    for (JobFile job : DuelJobs.jobs(duels).list().jobs()) {
      if (job.table().equals(table) && job.version() < macro.published()) {
        Verdict verdict =
            job.state() == JobFile.State.DONE
                ? Jobs.result(duels.resolve(job.name()), DuelResult.class).verdict()
                : Verdict.UNDECIDED;
        streak = Alarms.streak(streak, verdict);
      }
    }
    return new Alarms(Watch.of(evaluation), macro.run())
        .championDuel(
            duel.job(),
            macro.published(),
            duel.verdict(),
            duel.better(),
            Alarms.streak(streak, duel.verdict()),
            List.of());
  }

  /** 判定を run のアラートの行・王者の保存先・一時停止の要求に当てる。何度当てても 1 度と同じ。 */
  private static void apply(DuelMacro macro, Decision decision) throws Exception {
    new Alarms(Watch.of(macro.evaluation()), macro.run()).write(decision.alarms());
    ChampionStore store = ChampionStore.open(macro.run().champion());
    Champion candidate = decision.candidate();
    switch (decision.action()) {
      case CROWN -> {
        if (store.append(new StoreEvent.Initial(candidate))) {
          LOG.info("{} made {} the first champion", decision.duel(), candidate.label());
        }
        TableSpec table = macro.evaluation().tables().get(DuelStep.table(macro).orElseThrow());
        if (table.confirm() != null) {
          String name = table.confirm();
          if (measure(macro, name, macro.evaluation().tables().get(name), candidate).isEmpty()) {
            LOG.warn(
                "{}: the confirm table {} failed for the first champion {}; the next better duel measures it again",
                decision.duel(),
                name,
                candidate.label());
          }
        }
      }
      case PROMOTE -> {
        if (store.append(new StoreEvent.Promoted(candidate, decision.duel()))) {
          LOG.info("{} promoted {} to champion", decision.duel(), candidate.label());
        }
      }
      case NOT_CONFIRMED -> {
        if (store.append(new StoreEvent.NotConfirmed(candidate, decision.duel(), decision.check()))) {
          LOG.info(
              "{} keeps the champion: {} was not confirmed by {}",
              decision.duel(),
              candidate.label(),
              decision.check());
        }
      }
      case HOLD -> {
        boolean recorded = store.append(new StoreEvent.Paused(decision.duel()));
        ControlRequests requests = new ControlRequests(macro.run());
        boolean waiting =
            requests.pending().stream()
                .anyMatch(request -> request.request() instanceof Request.Pause pause && pause.before() == null);
        if (!waiting) {
          requests.submit(new Request.Pause(null));
        }
        if (recorded || !waiting) {
          LOG.warn(
              "{}: the champion duels stayed worse; the run pauses at the next safe point until it is resumed",
              decision.duel());
        }
      }
      case KEEP -> {}
    }
  }
}
