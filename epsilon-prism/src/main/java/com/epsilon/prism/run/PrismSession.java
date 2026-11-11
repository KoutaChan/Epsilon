package com.epsilon.prism.run;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.agent.run.Closing;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.IterationFile;
import com.epsilon.agent.run.IterationReport;
import com.epsilon.agent.run.JobRunner;
import com.epsilon.agent.run.Launch;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.RunSession;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.eval.EvaluationJobs;
import com.epsilon.prism.grp.GrpBackground;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiSetup;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.luck.LedgerReport;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 系譜の JVM が持つリソース（推論と学習器のストリーム、卓のコア、Φ、GRP のバックグラウンドの作業、評価のジョブ）と、段を順に回す
 * 仕組みが呼ぶ操作。リソースは、起動したときの行の速さに関わるキーで 1 回だけ開き、段やジョブでは開き直さない。閉じるときは、
 * バックグラウンドの作業を止めてからリソースを閉じる。
 */
final class PrismSession implements RunSession<PrismSettings, PrismState, PrismIteration> {

  /** バージョン 1 の初期化したままのネットワークを作るときに、run の乱数の種に足す値（run の種 + 100）。 */
  private static final int BLANK_SEED = 100;

  private static final Logger LOG = LoggerFactory.getLogger(PrismSession.class);

  private final Launch<PrismSettings> launch;
  private final TrainingResources resources;
  private final GrpBackground background;
  private final EvaluationJobs jobs;

  /** 最後に作った反復の設定（まだ無ければ起動したときの行の設定）。古いコホートを消すときに、残す数をここから読む。 */
  private PrismSettings latest;

  private PrismSession(
      Launch<PrismSettings> launch, TrainingResources resources, GrpBackground background, EvaluationJobs jobs) {
    this.launch = launch;
    this.resources = resources;
    this.background = background;
    this.jobs = jobs;
    this.latest = launch.settings();
  }

  /**
   * 起動したときの行の設定でリソースを開き、GRP のバックグラウンドの作業を始めて 1 回要求する。確定したコホートに半荘の記録か数の
   * ファイルが残っていれば、先に grp の下へ移す。
   *
   * @throws IOException Φ の GRP か表が読めないとき
   */
  static PrismSession open(Launch<PrismSettings> launch) throws IOException {
    PrismSettings settings = launch.settings();
    settings.runtime().apply();
    settings.diagnostics().apply();
    RunDirectory run = launch.run();
    for (int sealed : sealedCohorts(run)) {
      CollectStep.finish(run, sealed);
    }
    TrainingResources resources = TrainingResources.open(settings, run.root());
    try {
      RunControl control = launch.control();
      GrpBackground background =
          GrpBackground.start(
              new GrpFiles(run.root()),
              new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed()),
              control::awaitRunning);
      background.request();
      return new PrismSession(
          launch, resources, background, new EvaluationJobs(run, settings, resources.runtime(), resources.table()));
    } catch (RuntimeException | Error failure) {
      resources.close();
      throw failure;
    }
  }

  RunDirectory run() {
    return launch.run();
  }

  int launchRow() {
    return launch.row();
  }

  PrismSettings machine() {
    return launch.settings();
  }

  RunControl control() {
    return launch.control();
  }

  TrainingResources resources() {
    return resources;
  }

  GrpBackground background() {
    return background;
  }

  /** バージョン 1 を置く。初期化したままのネットワークで、s ≡ 0 なので argmax p0 で打つ。もう置いてあれば何もしない。 */
  @Override
  public PrismState begin(PrismSettings settings) throws IOException {
    Path first = run().snapshot(1);
    if (!Files.isDirectory(first)) {
      LOG.warn("Starting a PRISM lineage from a blank network: s = 0 everywhere, so version 1 plays argmax p0");
      Path part = PrismStep.part(first);
      DurableFiles.delete(part);
      try (Model blank =
          Fit.blank(
              settings.module().input(), settings.network().shape(), Device.cpu(), settings.run().seed() + BLANK_SEED)) {
        Snapshot.fill(part, (PrismNetwork) blank.getBlock(), PolicyParams.of(1));
      }
      DurableFiles.seal(part, first);
    }
    // バージョン 1 では、学習中のネットワークと打つネットワークが同じ
    return PrismState.initial();
  }

  @Override
  public PrismIteration iteration(IterationFile fixed, PrismSettings settings, PrismState state) {
    latest = settings;
    return new PrismIteration(fixed, settings, state, this);
  }

  /**
   * 段の報告から、次の state（持ち越し・行の総数）と、クラスごとの詳しい表の行を作る。帳簿を当てはめた反復では、
   * 学習の段の出力にある帳簿の報告の全体も、reports の {@value LearnReport#LEDGER_LINES} に 1 行残す。反復の作業ディレクトリは
   * 反復を閉じると消えるので、その前に残しておく。
   */
  @Override
  public Closing<PrismState> close(PrismIteration iteration, PrismState state, Map<String, StepReport> steps)
      throws IOException {
    int t = iteration.number();
    PhiReport phi = (PhiReport) steps.get(PrismStep.PHI.name()).metrics();
    CollectReport collect = (CollectReport) steps.get(PrismStep.COLLECT.name()).metrics();
    LearnReport learn = (LearnReport) steps.get(PrismStep.LEARN.name()).metrics();
    PrismCarry carry = new PrismCarry(learn.appliedR2(), CarryKey.of(iteration.settings(), phi.value().entry()));
    // コホートを確定した後に王者へ戻すと決まった反復では、戻しを次の反復に送る
    int start = state.start() == collect.start() ? t + 1 : state.start();
    PrismState next = new PrismState(t + 1, state.totalRows() + collect.rows(), carry, start, state.regression());
    CollectReport.Classes collected =
        RecordFile.read(CollectReport.Classes.class, run().rows(t).resolve(CollectReport.CLASSES)).value();
    Path learned = iteration.work().resolve(LearnStep.LEARN);
    LedgerReport.Classes ledger = RecordFile.read(LedgerReport.Classes.class, learned.resolve(LearnReport.CLASSES)).value();
    LedgerReport.Summary summary = learn.ledger();
    Map<String, Record> reports = new LinkedHashMap<>();
    reports.put(
        NoiseLine.FILE,
        new NoiseLine(
            collected.classes(),
            ledger.classes(),
            summary == null ? null : summary.maxAbsZ(),
            summary == null ? List.of() : summary.biased()));
    Path full = learned.resolve(LearnReport.LEDGER);
    if (Files.exists(full)) {
      reports.put(LearnReport.LEDGER_LINES, RecordFile.read(LedgerReport.Full.class, full).value());
    }
    return new Closing<>(next, carry, reports);
  }

  @Override
  public JobRunner jobRunner() {
    return jobs;
  }

  /**
   * 残す数（{@code fit.keepCohorts}）より古いコホートを消す。公開したバージョンのスナップショットは消さない。王者の保存先と、評価の卓の
   * 相手（version:N、version:-d）が、バージョンをそのディレクトリで参照するため。
   */
  @Override
  public void clean(PrismState state) throws IOException {
    int newest = state.next() - 1 - latest.fit().keepCohorts();
    for (int cohort : sealedCohorts(run())) {
      if (cohort <= newest) {
        DurableFiles.delete(run().rows(cohort));
        LOG.info("Removed cohort {} (keeping {})", cohort, latest.fit().keepCohorts());
      }
    }
  }

  @Override
  public String summary(IterationReport report) {
    CollectReport collect = (CollectReport) report.steps().get(PrismStep.COLLECT.name()).metrics();
    LearnReport learn = (LearnReport) report.steps().get(PrismStep.LEARN.name()).metrics();
    return String.format(
        Locale.ROOT,
        "games=%d rows=%d steps=%d loss=%s gain=%.4f r2=%.4f",
        collect.games(),
        collect.rows(),
        learn.fit().steps(),
        learn.fit().loss() == null ? "-" : String.format(Locale.ROOT, "%.4e", learn.fit().loss()),
        learn.tempo().gain(),
        learn.appliedR2());
  }

  /** GRP のバックグラウンドの作業を止め、リソースを閉じる。 */
  @Override
  public void close() {
    try (resources) {
      background.close();
    }
  }

  /** 確定したコホートの番号（小さい順）。 */
  private static List<Integer> sealedCohorts(RunDirectory run) throws IOException {
    Path rows = run.rows(1).getParent();
    if (!Files.isDirectory(rows)) {
      return List.of();
    }
    try (Stream<Path> listed = Files.list(rows)) {
      return listed
          .filter(Files::isDirectory)
          .map(path -> path.getFileName().toString())
          .filter(name -> name.matches("\\d+"))
          .map(Integer::parseInt)
          .sorted()
          .toList();
    }
  }
}
