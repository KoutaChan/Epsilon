package com.epsilon.classic.campaign;

import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepReport;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.learn.SelfPlayLearner;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.mahjong.util.SeedMixer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 学習の段。macro t のコホートで、バージョン t の学習器を 1 回の macro だけ学ばせる（{@link SelfPlayLearner#learn}）。学んだ重み
 * （network/）・学習器の途中の状態（optimizer/）・段の報告を learner/&lt;t+1&gt;.part に書いてから learner/&lt;t+1&gt; に rename して確定する
 * （完了マーカー）。
 *
 * <p>確定の後の仕上げは、学んだ重みを snapshots/&lt;t+1&gt; に rename して、バージョン t + 1 として公開すること。仕上げは完了マーカーを
 * 見るたびに済ませる（何度でも同じ）。learner/&lt;v&gt; に残る途中の状態は、バージョン v から学ぶ macro が学習器を開き直すときに読む。
 *
 * <p>重みのチェックポイントの学習のステップ数は、バージョン t のステップ数にこの macro のステップの数を足したもの。
 */
final class LearnStep implements Step<CampaignIteration> {

  /** 学習器の途中の状態を、バージョンごとに並べるディレクトリ。 */
  static final String LEARNER = "learner";

  /** 学んだ重み（公開するまでの保存先）。 */
  static final String NETWORK = "network";

  /** 学習器の途中の状態（{@link SelfPlayLearner#save}）。 */
  static final String OPTIMIZER = "optimizer";

  /** 並べ替えの乱数の種のソルト。 */
  private static final long SEED_SALT = 0x4C45_4152_4E21_2121L;

  private static final Logger LOG = LoggerFactory.getLogger(LearnStep.class);

  @Override
  public Optional<StepReport> done(CampaignIteration iteration) throws IOException {
    int version = iteration.published();
    Path done = learner(iteration.run(), version);
    if (!Files.isDirectory(done)) {
      return Optional.empty();
    }
    publish(iteration.run(), version);
    return Optional.of(StepReport.read(done.resolve(CampaignStep.REPORT), LearnReport.class));
  }

  @Override
  public StepReport run(CampaignIteration iteration) throws Exception {
    long started = System.nanoTime();
    int t = iteration.number();
    RunDirectory run = iteration.run();
    Path done = learner(run, iteration.published());
    Path part = CampaignStep.part(done);
    DurableFiles.delete(part);
    ClassicSettings settings = iteration.settings();
    TrajectoryBatch batch = Cohort.trajectories(run.rows(t), settings.run().variant());
    CampaignSession session = iteration.session();
    long step = ClassicCheckpoint.header(run.snapshot(iteration.version())).step();
    SelfPlayLearner.Macro macro = session.learn(iteration, batch, SeedMixer.derive(settings.run().seed(), SEED_SALT, t));
    DurableFiles.createDirectories(part);
    session.saveLearner(part.resolve(OPTIMIZER), part.resolve(NETWORK), step + macro.steps().size());
    StepReport report = StepReport.ran(iteration, started, LearnReport.of(iteration.version(), macro));
    report.write(part.resolve(CampaignStep.REPORT));
    DurableFiles.seal(part, done);
    session.learned(iteration.published());
    publish(run, iteration.published());
    LOG.info(
        "t={} learned version {} from {} decisions in {} steps ({} policy updates), kl={} rate {} -> {}",
        t,
        iteration.published(),
        macro.decisions(),
        macro.steps().size(),
        macro.actorSteps(),
        String.format(Locale.ROOT, "%.3e", macro.kl()),
        macro.rate().action(),
        String.format(Locale.ROOT, "%.3e", macro.rate().next()));
    return report;
  }

  /** バージョンの学習器の途中の状態と、学習の段の報告のディレクトリ（learner/&lt;5 桁のバージョン&gt;）。 */
  static Path learner(RunDirectory run, int version) {
    return run.root().resolve(LEARNER).resolve(String.format(Locale.ROOT, "%05d", version));
  }

  /**
   * 確定した学習の段の仕上げ: 学んだ重みを snapshots/&lt;version&gt; に rename する。もう移していれば何もしない。
   *
   * @throws IllegalStateException 重みが残っているのに、バージョンがもう公開されている
   */
  static void publish(RunDirectory run, int version) throws IOException {
    Path state = learner(run, version);
    Path network = state.resolve(NETWORK);
    if (!Files.exists(network)) {
      return;
    }
    Path snapshot = run.snapshot(version);
    if (Files.exists(snapshot)) {
      throw new IllegalStateException(
          "Version " + version + " is already published at " + snapshot + " while " + network + " waits");
    }
    DurableFiles.createDirectories(run.snapshots());
    Files.move(network, snapshot, StandardCopyOption.ATOMIC_MOVE);
    DurableFiles.syncDirectory(run.snapshots());
    DurableFiles.syncDirectory(state);
  }
}
