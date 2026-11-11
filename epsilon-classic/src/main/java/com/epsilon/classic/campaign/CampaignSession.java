package com.epsilon.classic.campaign;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.pytorch.engine.PtEngine;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.run.Closing;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.IterationFile;
import com.epsilon.agent.run.IterationReport;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobRunner;
import com.epsilon.agent.run.JobWork;
import com.epsilon.agent.run.Launch;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.RunSession;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.learn.SelfPlayLearner;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * campaign の JVM のリソース（学習器とそのストリーム、自己対局と評価の推論のランタイムと卓のコア、自己対局）と、段の並びの仕組みが呼ぶ
 * 操作。リソースは起動の行の設定で 1 回だけ開き、段は開き直さない。
 *
 * <p>学習器も JVM の間 1 つ持つ。macro t の学習の段は、この JVM が前の macro で学んだ学習器が同じ学び方でバージョン t を持っていれば、そのまま
 * 続けて学ぶ。持っていなければ開き直す: バージョン 1 は AdamW の状態 0 と学習率の制御のウォームアップから、ほかは learner/&lt;t&gt; の途中の
 * 状態から（{@link SelfPlayLearner#resume}）。どちらでも、同じ重みと同じ途中の状態から学ぶ。
 *
 * <p>campaign はジョブを作らない。評価の卓は決闘の段が打つ。
 */
final class CampaignSession implements RunSession<ClassicSettings, CampaignState, CampaignIteration> {

  private static final Logger LOG = LoggerFactory.getLogger(CampaignSession.class);

  private final Launch<ClassicSettings> launch;
  private final NetworkRuntime learning;
  private final NetworkRuntime runtime;
  private final TableLoop table;
  private final Collector collector;

  /** 学習器のモデル（学習器が無ければ null）。 */
  private Model learnerModel;

  private SelfPlayLearner learner;

  /** 学習器の重みのバージョン。学んでいる間と、学んだ重みを確定する前は 0。 */
  private int learnerVersion;

  private CampaignSession(
      Launch<ClassicSettings> launch,
      NetworkRuntime learning,
      NetworkRuntime runtime,
      TableLoop table,
      Collector collector) {
    this.launch = launch;
    this.learning = learning;
    this.runtime = runtime;
    this.table = table;
    this.collector = collector;
  }

  /**
   * 起動の行の設定でリソースを開く（途中で失敗したら、開いた分を閉じる）。学習器のストリームは、推論より先に取る。
   *
   * @param collectors 自己対局の開き方
   */
  static CampaignSession open(Launch<ClassicSettings> launch, Collector.Factory collectors) throws IOException {
    ClassicSettings machine = launch.settings();
    NetworkRuntime learning = NetworkRuntime.forLearning();
    NetworkRuntime runtime = null;
    TableLoop table = null;
    try {
      for (Device device : machine.learner().networkDevices()) {
        if (device.isGpu()) {
          learning.streams(device);
        }
      }
      InferenceSettings inference = machine.inference();
      runtime = new NetworkRuntime(inference.encoders(), inference.slots(), inference.readyBatches());
      table = new TableLoop(machine.table());
      Collector collector = collectors.open(machine, runtime, table);
      return new CampaignSession(launch, learning, runtime, table, collector);
    } catch (IOException | RuntimeException | Error failure) {
      try (learning;
          NetworkRuntime openedRuntime = runtime;
          TableLoop openedTable = table) {
        // 開いた分を逆の順に閉じる
      } catch (RuntimeException | Error closing) {
        failure.addSuppressed(closing);
      }
      throw failure;
    }
  }

  RunDirectory run() {
    return launch.run();
  }

  int launchRow() {
    return launch.row();
  }

  ClassicSettings machine() {
    return launch.settings();
  }

  Collector collector() {
    return collector;
  }

  /** 自己対局と評価の推論のランタイム。 */
  NetworkRuntime runtime() {
    return runtime;
  }

  /** 自己対局と評価の卓のコア。 */
  TableLoop table() {
    return table;
  }

  /** 段の中で安全点を確かめる仕組み（決闘のジョブはチャンクの後に呼ぶ）。 */
  RunControl control() {
    return launch.control();
  }

  /**
   * チェックポイントのファイルを、まだ無いディレクトリに写す。{@code <to>.part} に全てを写し、ルールと今の入力の形に合うことを確かめてから
   * rename するので、ディレクトリがあれば中身はそろっている。
   *
   * @throws IllegalStateException 写す元がファイルのほかを持つか、ルールか今の入力の形と合わないとき
   */
  static void copyNetwork(Path from, Path to, Variant variant) throws IOException {
    Path part = CampaignStep.part(to);
    DurableFiles.delete(part);
    DurableFiles.createDirectories(part);
    try (Stream<Path> files = Files.list(from)) {
      for (Path file : files.sorted().toList()) {
        if (!Files.isRegularFile(file)) {
          throw new IllegalStateException("The network " + from + " holds more than files: " + file);
        }
        Files.copy(file, part.resolve(file.getFileName()));
      }
    }
    ClassicCheckpoint.verify(part, variant);
    DurableFiles.seal(part, to);
  }

  /**
   * 学習器の始めのネットワークをバージョン 1 として写し（もう写してあれば何もしない）、王者の保存先が空ならバージョン 1 を初めの王者に
   * する。途中で落ちてもう一度呼ばれても、同じ結果になる。
   *
   * @throws SettingsException 始めのネットワーク（run.start）が設定に無い
   * @throws IllegalStateException 始めのネットワークが run のルールか今の入力の形と合わない
   */
  @Override
  public CampaignState begin(ClassicSettings settings) throws IOException {
    Path start = settings.run().start();
    if (start == null) {
      throw new SettingsException(
          List.of(
              "The campaign " + run().root() + " has no start network: create it again with --set run.start=DIR"));
    }
    Path first = run().snapshot(1);
    if (!Files.isDirectory(first)) {
      copyNetwork(start, first, settings.run().variant());
      LOG.info("Version 1 of the campaign is the start network {}", start);
    }
    ChampionStore store = ChampionStore.open(run().champion());
    if (store.current().isEmpty()) {
      Champion champion = Champion.of(label(1), first, 1);
      store.append(new StoreEvent.Initial(champion));
      LOG.info("Version 1 is the first champion");
    }
    return CampaignState.initial();
  }

  @Override
  public CampaignIteration iteration(IterationFile fixed, ClassicSettings settings, CampaignState state) {
    return new CampaignIteration(fixed, settings, state, this);
  }

  @Override
  public Closing<CampaignState> close(
      CampaignIteration iteration, CampaignState state, Map<String, StepReport> steps) {
    CollectReport collect = (CollectReport) steps.get(CampaignStep.COLLECT.name()).metrics();
    LearnReport learn = (LearnReport) steps.get(CampaignStep.LEARN.name()).metrics();
    CampaignState next =
        new CampaignState(
            iteration.number() + 1, state.games() + collect.games(), state.decisions() + collect.decisions());
    return new Closing<>(next, new CampaignCarry(learn.rate().next()), Map.of());
  }

  @Override
  public JobRunner jobRunner() {
    return NoJobs.RUNNER;
  }

  /**
   * 次の macro が要らない出力を消す: 最後に閉じた macro より前のコホート（最後のコホートは次の収集が迷いの分布を読む）と、次のバージョンより
   * 前の学習器の途中の状態。公開したバージョンは消さない（王者と過去の王者が参照する）。
   */
  @Override
  public void clean(CampaignState state) throws IOException {
    int closed = state.next() - 1;
    for (int cohort : numbered(run().rows(1).getParent())) {
      if (cohort < closed) {
        DurableFiles.delete(run().rows(cohort));
        LOG.info("Removed cohort {}", cohort);
      }
    }
    for (int version : numbered(LearnStep.learner(run(), 1).getParent())) {
      if (version < state.next()) {
        DurableFiles.delete(LearnStep.learner(run(), version));
        LOG.info("Removed the learner state of version {}", version);
      }
    }
  }

  @Override
  public String summary(IterationReport report) {
    CollectReport collect = (CollectReport) report.steps().get(CampaignStep.COLLECT.name()).metrics();
    LearnReport learn = (LearnReport) report.steps().get(CampaignStep.LEARN.name()).metrics();
    return String.format(
        Locale.ROOT,
        "games=%d decisions=%d steps=%d/%d kl=%.3e rate=%.3e",
        collect.games(),
        collect.decisions(),
        learn.actorSteps(),
        learn.steps(),
        learn.kl(),
        learn.rate().next());
  }

  /**
   * macro の学習器で、コホートを 1 回の macro だけ学ぶ。学んでいる間と、学んだ重みを確定する（{@link #learned}）までは、学習器はどの
   * バージョンも持たないものとする（学習の段が落ちれば、次の学習の段は開き直す）。
   *
   * @param seed 並べ替えの乱数の種
   */
  SelfPlayLearner.Macro learn(CampaignIteration iteration, TrajectoryBatch batch, long seed) throws IOException {
    SelfPlayLearner opened = learner(iteration);
    learnerVersion = 0;
    return opened.learn(batch, seed);
  }

  /**
   * 学んだ学習器の途中の状態と重みを、新しいディレクトリに書く。
   *
   * @param optimizer 途中の状態を書くディレクトリ
   * @param network 重みのチェックポイントを書くディレクトリ
   * @param step チェックポイントの学習のステップ数
   */
  void saveLearner(Path optimizer, Path network, long step) throws IOException {
    learner.save(optimizer);
    ClassicCheckpoint.write(network, (ClassicNetwork) learnerModel.getBlock(), step);
  }

  /** 学習器の重みを、確定したバージョンとして覚える。 */
  void learned(int version) {
    learnerVersion = version;
  }

  /** 推論のランタイム・卓のコア・自己対局を閉じ、学習器と学習器のストリームを閉じる。 */
  @Override
  public void close() {
    try (learning;
        runtime;
        table;
        collector) {
      closeLearner();
    }
  }

  /** macro のバージョンの重みと途中の状態を持つ学習器。持っていなければ開き直す。 */
  private SelfPlayLearner learner(CampaignIteration iteration) throws IOException {
    int version = iteration.version();
    SelfPlayLearner.Settings wanted = iteration.learning();
    if (learner != null && learnerVersion == version && learner.settings().equals(wanted)) {
      return learner;
    }
    closeLearner();
    NetworkDevices devices = machine().learner().networkDevices();
    Model model = Model.newInstance("classic-learner", devices.primary(), PtEngine.ENGINE_NAME);
    try {
      model.setBlock(
          ClassicCheckpoint.read(run().snapshot(version), iteration.settings().run().variant(), model.getNDManager()));
      SelfPlayLearner opened =
          version == 1
              ? new SelfPlayLearner(model, devices, learning, wanted)
              : SelfPlayLearner.resume(
                  model, devices, learning, wanted, LearnStep.learner(run(), version).resolve(LearnStep.OPTIMIZER));
      learnerModel = model;
      learner = opened;
      learnerVersion = version;
      LOG.info("Opened the learner at version {} on {}", version, devices.devices());
      return opened;
    } catch (IOException | RuntimeException | Error failure) {
      model.close();
      throw failure;
    }
  }

  private void closeLearner() {
    try (Model model = learnerModel) {
      if (learner != null) {
        learner.close();
      }
    } finally {
      learner = null;
      learnerModel = null;
      learnerVersion = 0;
    }
  }

  /** バージョンの名前（王者の保存先に書く）。 */
  static String label(int version) {
    return String.format(Locale.ROOT, "v%05d", version);
  }

  /** ディレクトリの中の、数だけの名前のディレクトリの番号（小さい順）。ディレクトリが無ければ空。 */
  private static List<Integer> numbered(Path directory) throws IOException {
    if (!Files.isDirectory(directory)) {
      return List.of();
    }
    try (Stream<Path> listed = Files.list(directory)) {
      return listed
          .filter(Files::isDirectory)
          .map(path -> path.getFileName().toString())
          .filter(name -> name.matches("\\d+"))
          .map(Integer::parseInt)
          .sorted()
          .toList();
    }
  }

  /** ジョブを持たない run のジョブの中身。ジョブがあれば拒む。 */
  private enum NoJobs implements JobRunner {
    RUNNER;

    @Override
    public Node.Table plan(JobFile job) {
      throw new IllegalStateException("The classic campaign has no jobs, but found " + job.name());
    }

    @Override
    public JobWork open(JobFile job, Path directory) {
      throw new IllegalStateException("The classic campaign has no jobs, but found " + job.name());
    }
  }
}
