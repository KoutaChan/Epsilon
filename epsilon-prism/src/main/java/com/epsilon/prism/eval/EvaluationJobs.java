package com.epsilon.prism.eval;

import com.epsilon.agent.eval.Alarm;
import com.epsilon.agent.eval.Alarms;
import com.epsilon.agent.eval.Arena;
import com.epsilon.agent.eval.Baseline;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.Duel;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.Sequential;
import com.epsilon.agent.eval.Split;
import com.epsilon.agent.eval.StopRule;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.eval.WallSet;
import com.epsilon.agent.eval.WallValues;
import com.epsilon.agent.eval.Watch;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.Jars;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.JobRunner;
import com.epsilon.agent.run.JobWork;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.run.PrismState;
import com.epsilon.prism.run.Regression;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 系譜の評価と決闘のジョブ。evaluate も同じジョブを回す。
 *
 * <p>ジョブを始めるときに、卓の相手を具体的なチェックポイントに決める。王者は保存先にいる今の王者、version:N はそのバージョン、version:-d は
 * d バージョン前のバージョンになる。次に決闘の計画（{@link DuelSetup}）を書き、{@link Duel} で打つ。計画には、打ち手（系譜のバージョンは
 * バージョン、系譜の外のチェックポイントは SHA-256 も）、席の割り当て方と分け方、ツモの順、牌山の種と最初の番号、止め方、チャンク、判定の
 * 時点と臨界値を書く。牌山の種は、全てのバージョンで同じ牌山を打つ卓なら
 * {@link Arena#EVALUATION_SEED}、ジョブごとに牌山を変える卓なら run の種とジョブの名前から作る。
 *
 * <p>打つときは、系譜の JVM の推論のリソースを使う。推論のデバイス・バッチ・推論のバックエンドは起動したときの設定の行から、計算の型は
 * ジョブを作った設定の行から取る。同値な打牌のどちらを切るかは、ジョブを作った設定の行から計画に書いた方に従う。classic の相手は、
 * デバイスの指定が無ければ四人打ち・三人打ちごとの既定のデバイスで打つ。
 *
 * <p>王者との決闘（相手が王者で、止め方が判定の卓）の結果は、王者の保存先、一時停止の要求、アラート、state の順に反映する。同じ決闘を 2 回
 * 反映しても、1 回と同じ結果になる。王者の保存先が空のとき、王者の卓のジョブは打たずに候補を初めの王者にする。リグレッションのときの扱いは、
 * 設定の行の evaluation.onWorse で決める。evaluate のジョブは state を持たないので、王者の保存先とアラートにだけ反映する。
 *
 * <p>系譜の run で昇格したバージョンは、昇格したバージョンも測る基準の卓（相手が version:N の卓）でも測る。測るのは N より新しいバージョン
 * だけで、卓の最初のバージョンより前でも測る。ただし、そのバージョンを予定どおり測る卓には、ジョブを足さない。
 *
 * <p>ルールベースの打ち手を相手に全てのバージョンが同じ牌山で打つ卓と、基準の卓の結果は、王者のバージョンの同じ卓の結果と牌山ごとに対にする。
 * 悪い向きの z が evaluation.alarmZ を超えればアラートを書く（基準の卓では、王者の値より evaluation.dropMargin 以上悪いときだけ）。対にするのは、
 * 同値な打牌を同じ方で切ったジョブどうしだけ。同じ王者の間はアラートを 1 回だけ書き、どの王者で書いたかはアラートの行から読み直す。系譜の
 * run で、予定のラダーを打たずに基準の卓の結果で代える回（{@link EvaluationSettings#coveredLadders}）は、ラダーの悪化も基準の卓の区間で
 * 判定する。
 *
 * <p>アラートは、どの卓も {@link Alarms} で決めて書く。
 *
 * <p>結果を反映するたびに、報告（{@link EvaluationReports}）を書き直す。
 */
public final class EvaluationJobs implements JobRunner {

  /** 計画の牌山を全て打つ卓の区間の、両側の有意水準。 */
  public static final double FIXED_ALPHA = 0.01;

  private static final Logger LOG = LoggerFactory.getLogger(EvaluationJobs.class);

  private final RunDirectory run;
  private final RunDirectory versions;
  private final String model;
  private final boolean lineage;
  private final Rows rows;
  private final InferenceSettings launched;
  private final String device;
  private final NetworkRuntime runtime;
  private final TableLoop table;
  private final EvaluationReports reports;
  private final Map<Integer, PrismSettings> settings = new HashMap<>();

  /** この JVM の jar の ID（初めて計画を決めるときに読む）。 */
  private String jar;

  /** ジョブを作った設定の行から、設定を読む。 */
  @FunctionalInterface
  private interface Rows {
    PrismSettings row(int row) throws IOException;
  }

  private EvaluationJobs(
      RunDirectory run,
      RunDirectory versions,
      String model,
      boolean lineage,
      Rows rows,
      InferenceSettings launched,
      String device,
      NetworkRuntime runtime,
      TableLoop table) {
    this.run = run;
    this.versions = versions;
    this.model = model;
    this.lineage = lineage;
    this.rows = rows;
    this.launched = launched;
    this.device = device;
    this.runtime = runtime;
    this.table = table;
    this.reports = new EvaluationReports(run, lineage);
  }

  /**
   * 系譜の run のジョブを回す。候補と相手は run のバージョンで、設定はジョブを作った設定の行から読む。
   *
   * @param machine 起動したときの設定の行（推論のデバイス・バッチ・推論のバックエンドを使う）
   * @param runtime 打ち手を載せる実行のリソース
   * @param table 卓のコア（ジョブの間は、ほかでは使わない）
   */
  public EvaluationJobs(RunDirectory run, PrismSettings machine, NetworkRuntime runtime, TableLoop table) {
    this(
        run,
        run,
        null,
        true,
        row -> new SettingsRows(run).row(row).bind(PrismSettings.class).value(),
        machine.inference(),
        null,
        runtime,
        table);
  }

  /**
   * evaluate のジョブを回す。ジョブ・王者の保存先・報告は out に書き、全てのジョブを同じ設定で打つ。
   *
   * @param out ジョブ・王者の保存先・報告のディレクトリ
   * @param versions 候補と相手のバージョンを持つ run（{@code --snapshots}。{@code --model} なら null）
   * @param model 候補の打ち手の指定（{@code --model}。バージョンを測るなら null）
   * @param settings 全てのジョブで使う設定
   * @param device classic の打ち手のデバイス（cpu か gpuN。null なら四人打ち・三人打ちごとの既定）
   * @param runtime 打ち手を載せる実行のリソース
   * @param table 卓のコア（ジョブの間は、ほかでは使わない）
   */
  public static EvaluationJobs measuring(
      RunDirectory out,
      RunDirectory versions,
      String model,
      PrismSettings settings,
      String device,
      NetworkRuntime runtime,
      TableLoop table) {
    if ((versions == null) == (model == null)) {
      throw new IllegalArgumentException("An evaluation measures the versions of a run or one model");
    }
    return new EvaluationJobs(
        out, versions, model, false, row -> settings, settings.inference(), device, runtime, table);
  }

  @Override
  public Node.Table plan(JobFile job) throws Exception {
    PrismSettings settings = settings(job);
    TableSpec table = table(job);
    DuelSetup.Entrant candidate = candidate(job.version());
    DuelSetup.Entrant opponent = opponent(table, job.version());
    if (opponent == null) {
      LOG.info(
          "{} plays no walls: the champion store is empty, so {} becomes the first champion",
          job.name(),
          candidate.label());
      return RecordFile.tree(new EvaluationPlan(candidate, null, jar(), settings.behaviour().twinDiscard()));
    }
    long wallSeed =
        table.wallSet() == WallSet.FIXED ? Arena.EVALUATION_SEED : Arena.freshSeed(settings.run().seed(), job.name());
    DuelSetup duel =
        DuelSetup.of(
            settings.run().variant(),
            candidate,
            opponent,
            table.seating().seating(),
            table.seating().splits(settings.run().variant()),
            table.draws(),
            table.wallSet(),
            wallSeed,
            Arena.WALL_BASE,
            table.walls(),
            rule(table.stop()),
            table.score(),
            settings.utility());
    LOG.info(
        "{} plans {} walls against {} (wall seed {}, {} looks)",
        job.name(),
        table.walls(),
        opponent.label(),
        wallSeed,
        duel.looks().size());
    return RecordFile.tree(new EvaluationPlan(candidate, duel, jar(), settings.behaviour().twinDiscard()));
  }

  @Override
  public JobWork open(JobFile job, Path directory) throws Exception {
    EvaluationPlan plan = plan(job.plan());
    if (plan.crowns()) {
      return new Crowning(plan.candidate().label());
    }
    DuelSetup duel = plan.duel();
    PrismSettings settings = settings(job);
    InferenceSettings inference =
        new InferenceSettings(
            launched.devices(),
            settings.inference().precision(),
            launched.kernels(),
            launched.maxBatch(),
            launched.encoders(),
            launched.slots(),
            launched.readyBatches());
    PolicyOptions options =
        new PolicyOptions(
            device != null ? device : settings.module().opponents().defaultDevice(),
            inference.precision(),
            inference.maxBatch(),
            Kernels.EAGER);
    EvaluationTable.verify(duel.candidate());
    EvaluationTable.verify(duel.opponent());
    TwinDiscard twinDiscard = plan.twinDiscard();
    Policy candidate =
        EvaluationTable.open(duel.candidate().spec(), duel.variant(), inference, twinDiscard, options, runtime);
    Policy opponent;
    try {
      opponent = EvaluationTable.open(duel.opponent().spec(), duel.variant(), inference, twinDiscard, options, runtime);
    } catch (IOException | RuntimeException | Error failure) {
      candidate.close();
      throw failure;
    }
    return Duel.open(job.name(), table, duel, candidate, opponent, directory);
  }

  /** 打ち手を載せる実行の資源が、失敗で GPU のリソースを隔離していれば、IllegalStateException を投げる。 */
  @Override
  public void requireUsable() {
    if (runtime != null) {
      runtime.requireUsable();
    }
  }

  /**
   * 結果を王者の保存先・state・アラートに反映し、報告を書き直す。何度呼んでも 1 回と同じ結果になる。2 回目を書かないように、
   * 王者の保存先は決闘の名前で、state は最後に反映した決闘の名前で、アラートは行の名前で見分ける。足すジョブは、もうあれば足さない。
   * 報告は、ジョブを作った設定の行の卓で書き直す。
   */
  @Override
  public void settle(JobFile job, Node.Table result) throws Exception {
    apply(job, result);
    reports.write(settings(job).evaluation(), TableSpec.kind(table(job).opponent()));
  }

  /** 結果を王者の保存先・state・アラートに反映する。 */
  private void apply(JobFile job, Node.Table result) throws IOException {
    EvaluationPlan plan = plan(job.plan());
    if (plan.crowns()) {
      Champion candidate = champion(plan.candidate());
      if (ChampionStore.open(run.champion()).append(new StoreEvent.Initial(candidate))) {
        LOG.info("{} made {} the first champion", job.name(), candidate.label());
      }
      return;
    }
    DuelSetup duel = plan.duel();
    DuelResult outcome = Jobs.result(result, DuelResult.class);
    Estimate better = duel.score() == Score.RANK ? outcome.rank().negate() : outcome.utility();
    TableSpec table = table(job);
    Alarms alarms = new Alarms(Watch.of(settings(job).evaluation()), run);
    TableSpec.Opponent kind = TableSpec.kind(table.opponent());
    if (kind == TableSpec.Opponent.CHAMPION && table.stop() instanceof StopSettings.Test test) {
      settleDuel(job, duel, test.promote(), outcome.verdict(), better, alarms);
    } else if (kind == TableSpec.Opponent.DISTANCE && !(table.stop() instanceof StopSettings.Test)) {
      alarms.write(alarms.ladder(job.name(), job.version(), better));
    } else if (kind == TableSpec.Opponent.VERSION) {
      settleBaseline(job, plan, outcome, better, alarms);
    } else if (kind == TableSpec.Opponent.HEURISTIC && table.wallSet() == WallSet.FIXED) {
      settleAnchor(job, plan, alarms);
    }
  }

  /** ルールベースの打ち手の卓の結果を、王者の同じ卓と牌山ごとに対にし、低下のアラートを書く。 */
  private void settleAnchor(JobFile job, EvaluationPlan plan, Alarms alarms) throws IOException {
    Optional<Paired> paired = paired(job, plan);
    if (paired.isPresent()) {
      Paired pair = paired.get();
      alarms.write(alarms.anchor(job.name(), job.version(), pair.champion(), pair.current(), pair.championValues()));
    }
  }

  /**
   * 基準の卓の結果を反映する。系譜の run で予定のラダーを打たなかった回は、この結果の区間の上端が 0 より小さければラダーのアラートを書く。
   * 王者の同じ卓と牌山ごとに対にした差が evaluation.dropMargin 以上悪く、悪い向きの z がしきい値を超えれば、低下のアラートを書く。候補が
   * 1 席だけの分け方の δ が、2 つ目の基準の卓を足すしきい値（{@link Baseline#SECOND_LINE}）を下回り、もっと新しいバージョンを相手にする基準の卓が
   * 設定に無ければ、ログで知らせる。卓を足すには設定を変える。
   */
  private void settleBaseline(JobFile job, EvaluationPlan plan, DuelResult outcome, Estimate better, Alarms alarms)
      throws IOException {
    EvaluationSettings evaluation = settings(job).evaluation();
    if (lineage && !evaluation.coveredLadders(job.table(), job.version()).isEmpty()) {
      alarms.write(alarms.ladder(job.name(), job.version(), better));
    }
    Optional<Paired> paired = paired(job, plan);
    if (paired.isPresent()) {
      Paired pair = paired.get();
      alarms.write(alarms.baseline(job.name(), job.version(), pair.champion(), pair.current(), pair.championValues()));
    }
    int base = table(job).opponentVersion(job.version());
    boolean later =
        evaluation.tables().values().stream()
            .anyMatch(
                other ->
                    TableSpec.kind(other.opponent()) == TableSpec.Opponent.VERSION
                        && other.opponentVersion(job.version()) > base);
    String oneVsRest = Split.oneAgainstRest(plan.duel().variant()).toString();
    for (DuelResult.SplitValue split : outcome.splits()) {
      if (!later
          && split.split().equals(oneVsRest)
          && Baseline.needsSecond(split.rank().mean(), Baseline.SECOND_LINE)) {
        LOG.warn(
            "{}: the {} rank {} is below {}; add a second baseline table against the current champion to keep the main"
                + " number sensitive",
            job.name(),
            oneVsRest,
            String.format(Locale.ROOT, "%.4f", split.rank().mean()),
            Baseline.SECOND_LINE);
      }
    }
  }

  /**
   * 王者のバージョンの同じ卓と、牌山ごとに対にした値。候補が良いほど大きくなる向きにそろえる。
   *
   * @param champion 王者のバージョン
   * @param current このジョブの値
   * @param championValues 王者の同じ卓のジョブの値
   */
  private record Paired(int champion, double[] current, double[] championValues) {}

  /**
   * ジョブの結果を、王者のバージョンの同じ卓のジョブの結果と牌山ごとに対にする。王者は、結果を反映する時点で保存先にいる王者。
   *
   * @return 対にした値。王者がいない、王者がこのバージョンより前でない、王者の同じ卓のジョブが済んでいない、そのジョブと相手・牌山・
   *     席の割り当て・ツモの順・効用・同値な打牌の切り方のどれかが違う、のどれかなら空
   */
  private Optional<Paired> paired(JobFile job, EvaluationPlan plan) throws IOException {
    Optional<Champion> stored = ChampionStore.open(run.champion()).current();
    if (stored.isEmpty() || stored.get().version() >= job.version()) {
      return Optional.empty();
    }
    Champion champion = stored.get();
    String name = RunDirectory.jobName(champion.version(), job.table());
    Path home = run.job(name);
    if (!Files.exists(home.resolve(Jobs.RESULT))) {
      LOG.info("{} is not paired: champion {} has no finished {} job", job.name(), champion.label(), job.table());
      return Optional.empty();
    }
    EvaluationPlan other = plan(RecordFile.read(JobFile.class, home.resolve(Jobs.JOB)).value().plan());
    if (other.crowns() || !other.duel().pairsWith(plan.duel())) {
      LOG.info(
          "{} is not paired: {} of champion {} played another opponent or other walls",
          job.name(),
          name,
          champion.label());
      return Optional.empty();
    }
    if (other.twinDiscard() != plan.twinDiscard()) {
      LOG.info(
          "{} is not paired: the {} job of champion {} played twin discards as {}, this job as {}",
          job.name(),
          job.table(),
          champion.label(),
          other.twinDiscard(),
          plan.twinDiscard());
      return Optional.empty();
    }
    Score score = plan.duel().score();
    return Optional.of(
        new Paired(
            champion.version(),
            values(job.name(), plan.duel()).better(score),
            values(name, other.duel()).better(score)));
  }

  /** 済んだジョブの games.tsv から、牌山ごとの値を読む。 */
  private WallValues values(String job, DuelSetup duel) throws IOException {
    return JobGames.read(run.job(job), duel);
  }

  /**
   * 王者との決闘の結果を反映する。昇格させ、悪い判定が続いた回数を数え、リグレッションの扱いを決める。evaluate では、昇格と、悪い判定の
   * アラートだけを扱う。系譜の run では、もっと後のバージョンの決闘をもう反映していれば、何も反映しない（決闘の結果はバージョンの順に
   * 反映する）。
   */
  private void settleDuel(
      JobFile job, DuelSetup duel, boolean promote, Verdict verdict, Estimate better, Alarms alarms)
      throws IOException {
    PrismState state = lineage ? RecordFile.read(PrismState.class, run.state()).value() : null;
    if (state != null && job.name().equals(state.regression().lastDuel())) {
      return;
    }
    if (state != null && settledLater(state.regression(), job.version())) {
      LOG.warn(
          "{} is not applied to the champion or the worse streak: the duel {} of a later version is already applied",
          job.name(),
          state.regression().lastDuel());
      return;
    }
    ChampionStore store = ChampionStore.open(run.champion());
    if (verdict == Verdict.BETTER
        && promote
        && store.append(new StoreEvent.Promoted(champion(duel.candidate()), job.name()))) {
      LOG.info("{} promoted version {} to champion", job.name(), job.version());
    }
    if (verdict == Verdict.BETTER && promote && lineage) {
      measurePromoted(job);
    }
    if (state == null) {
      alarms.write(alarms.championDuel(job.name(), job.version(), verdict, better));
      return;
    }
    Regression regression = state.regression();
    int streak = Alarms.streak(regression.streak(), verdict);
    Regression next = regression.after(job.name(), streak);
    int start = state.start();
    List<Alarm> raised =
        alarms.championDuel(job.name(), job.version(), verdict, better, streak, regression.rollbacks());
    for (Alarm alarm : raised) {
      switch (alarm.handling()) {
        case PAUSE -> {
          store.append(new StoreEvent.Paused(job.name()));
          new ControlRequests(run).submit(new Request.Pause(null));
          LOG.warn("{}: the champion duels were worse {} times in a row; the run pauses", job.name(), streak);
        }
        case ROLLBACK -> {
          Champion champion = store.current()
              .orElseThrow(() -> new IllegalStateException(job.name() + " rolls back, but the store has no champion"));
          store.append(new StoreEvent.RolledBack(champion, job.name()));
          next = regression.rolledBack(job.name(), streak, job.version());
          start = champion.version();
          LOG.warn(
              "{}: the champion duels were worse {} times in a row; the next iteration starts from version {}",
              job.name(),
              streak,
              start);
        }
        case REPORT -> {}
      }
    }
    alarms.write(raised);
    DurableFiles.write(run.state(), state.regressed(next, start));
  }

  /** 最後に反映した王者との決闘が、バージョン {@code version} より後のバージョンの決闘なら true。 */
  private static boolean settledLater(Regression regression, int version) {
    String last = regression.lastDuel();
    return last != null
        && RunDirectory.parseJobName(last)
                .orElseThrow(() -> new IllegalStateException("Not the name of a table job: " + last))
                .version()
            > version;
  }

  /**
   * 昇格したバージョンを測るジョブを、昇格したバージョンも測る基準の卓に足す。足すのは卓の相手より新しいバージョンだけで、卓の最初の
   * バージョンより前でも足す（後で基準の卓を打つバージョンを、この王者の値と対にできるように）。そのバージョンを予定どおり測る卓には
   * 足さない。ジョブを回している途中に足したジョブも、そのまま回る。
   */
  private void measurePromoted(JobFile job) throws IOException {
    Jobs jobs = new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS));
    int order = 0;
    for (Map.Entry<String, TableSpec> entry : settings(job).evaluation().tables().entrySet()) {
      TableSpec table = entry.getValue();
      if (Boolean.TRUE.equals(table.onPromotion())
          && job.version() > table.opponentVersion(job.version())
          && !table.scheduled(job.version())) {
        JobFile added = JobFile.pending(job.version(), entry.getKey(), order, job.row(), RecordFile.tree(table));
        if (jobs.create(added)) {
          LOG.info("{} adds the job {} to measure the promoted version", job.name(), added.name());
        }
      }
      order++;
    }
  }

  /**
   * ジョブの候補の打ち手を返す。evaluate で {@code --model} を指定したなら、そのモデルになる。
   *
   * @param version 測るバージョン
   * @throws IOException バージョンがまだ公開されていないか、チェックポイントが読めないとき
   */
  public DuelSetup.Entrant candidate(int version) throws IOException {
    return model != null ? EvaluationTable.entrant(model) : version(version);
  }

  /** 卓の相手を具体的な打ち手に決める。王者の卓で王者の保存先が空なら、null を返す（候補を初めの王者にする）。 */
  private DuelSetup.Entrant opponent(TableSpec table, int version) throws IOException {
    String written = table.opponent();
    return switch (TableSpec.kind(written)) {
      case HEURISTIC, CHECKPOINT -> EvaluationTable.entrant(written);
      case VERSION, DISTANCE -> version(table.opponentVersion(version));
      case CHAMPION -> {
        Optional<Champion> champion = ChampionStore.open(run.champion()).current();
        if (champion.isEmpty()) {
          yield null;
        }
        Champion current = champion.get();
        yield new DuelSetup.Entrant("prism:" + current.checkpoint(), current.label(), current.version(), null);
      }
    };
  }

  /** 系譜のバージョンの打ち手を作る。 */
  private DuelSetup.Entrant version(int version) throws IOException {
    return new DuelSetup.Entrant("prism:" + published(version), label(version), version, null);
  }

  /**
   * 公開したバージョンのディレクトリを返す。
   *
   * @throws IOException バージョンを持つ run が無いか、まだ公開していないとき
   */
  private Path published(int version) throws IOException {
    if (versions == null) {
      throw new IOException("Version " + version + " needs the run of its lineage, but this evaluation measures one model");
    }
    Path snapshot = versions.snapshot(version);
    if (!Files.isDirectory(snapshot)) {
      throw new IOException("Version " + version + " is not published in " + versions.root());
    }
    return snapshot;
  }

  /**
   * 打ち手を王者として保存先に書くための記録を作る。
   *
   * @throws IllegalStateException 打ち手が系譜のバージョンでないとき
   */
  private static Champion champion(DuelSetup.Entrant entrant) {
    if (entrant.version() == null) {
      throw new IllegalStateException(entrant.label() + " is not a version of the lineage, so it cannot be a champion");
    }
    return Champion.of(entrant.label(), EvaluationTable.spec(entrant.spec()).checkpoint(), entrant.version());
  }

  /** 卓の止め方を、牌山ごとの値の止め方に直す。 */
  static StopRule rule(StopSettings stop) {
    return switch (stop) {
      case StopSettings.Fixed fixed -> new StopRule.Fixed(FIXED_ALPHA);
      case StopSettings.Precision precision ->
          new StopRule.Precision(precision.error(), precision.alpha(), precision.checkEvery());
      case StopSettings.Test test ->
          new StopRule.Test(
              new Sequential(test.alpha(), test.spending(), test.checkEvery()), test.evenMargin(), test.evenMargin());
    };
  }

  /** バージョンの名前（{@code v00112}）。 */
  static String label(int version) {
    return String.format("v%05d", version);
  }

  /** この JVM の jar の ID（{@link Jars#running}）。 */
  private String jar() throws IOException {
    if (jar == null) {
      jar = Jars.running();
    }
    return jar;
  }

  private PrismSettings settings(JobFile job) throws IOException {
    PrismSettings bound = settings.get(job.row());
    if (bound == null) {
      bound = rows.row(job.row());
      settings.put(job.row(), bound);
    }
    return bound;
  }

  private static TableSpec table(JobFile job) {
    return RecordFile.read(TableSpec.class, job.definition()).value();
  }

  private static EvaluationPlan plan(Node.Table plan) {
    return RecordFile.read(EvaluationPlan.class, plan).value();
  }

  /**
   * 王者の保存先が空だったので、相手と打たずに候補を初めの王者にしたジョブの結果（result.json と報告の行に書く）。
   *
   * @param champion 初めの王者にした候補の名前
   */
  public record Crowned(String champion) {}

  /** 王者の保存先が空のときの、王者の卓のジョブ。打たずに、候補を初めの王者にした結果を返す。 */
  private record Crowning(String candidate) implements JobWork {

    @Override
    public Chunk next(long walls, JobOutputs outputs) {
      return Chunk.decided(walls, new Crowned(candidate));
    }

    @Override
    public void close() {}
  }
}
