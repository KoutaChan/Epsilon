package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepReport;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.learn.UncertaintyCdf;
import com.epsilon.mahjong.util.SeedMixer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 収集の段。学習器のバージョン t で macro t の自己対局を打ち（{@link Collector}）、学習器の席の軌跡と迷いの分布と段の報告を
 * rows/&lt;t&gt;.part に書いてから rows/&lt;t&gt; に rename して確定する（完了マーカー）。相手の候補は王者の保存先の今の王者と、新しいものから
 * {@code selfPlay.pool} 人までの過去の王者で、自己対局がその中から macro の相手を 1 人選ぶ。探索は macro t − 1 のコホートの迷いの分布を
 * 引く（macro 1 は空の分布）。
 */
final class CollectStep implements Step<CampaignIteration> {

  /** macro の乱数の種のソルト。 */
  private static final long SEED_SALT = 0x434F_4C4C_4543_5421L;

  private static final Logger LOG = LoggerFactory.getLogger(CollectStep.class);

  @Override
  public Optional<StepReport> done(CampaignIteration iteration) throws IOException {
    Path sealed = iteration.run().rows(iteration.number());
    if (!Files.isDirectory(sealed)) {
      return Optional.empty();
    }
    return Optional.of(StepReport.read(sealed.resolve(CampaignStep.REPORT), CollectReport.class));
  }

  @Override
  public StepReport run(CampaignIteration iteration) throws Exception {
    long started = System.nanoTime();
    int t = iteration.number();
    RunDirectory run = iteration.run();
    Path sealed = run.rows(t);
    Path part = CampaignStep.part(sealed);
    DurableFiles.delete(part);
    ClassicSettings settings = iteration.settings();
    ChampionStore store = ChampionStore.open(run.champion());
    Champion champion =
        store.current().orElseThrow(() -> new IllegalStateException("The campaign " + run.root() + " has no champion"));
    List<Champion> pool = store.past(settings.selfPlay().pool());
    UncertaintyCdf calibration = t == 1 ? UncertaintyCdf.empty() : Cohort.calibration(run.rows(t - 1));
    Collector.Macro macro =
        new Collector.Macro(
            t,
            SeedMixer.derive(settings.run().seed(), SEED_SALT, t),
            settings,
            run.snapshot(iteration.version()),
            champion,
            pool,
            calibration);
    Collector.Collected collected = iteration.session().collector().collect(macro);
    TrajectoryBatch trajectories = collected.trajectories();
    if (trajectories.variant() != settings.run().variant()) {
      throw new IllegalStateException(
          "Macro " + t + " collected " + trajectories.variant() + " games, not " + settings.run().variant());
    }
    if (trajectories.games().size() != settings.schedule().games()) {
      throw new IllegalStateException(
          "Macro " + t + " collected " + trajectories.games().size() + " games, not " + settings.schedule().games());
    }
    Champion opponent = collected.opponent();
    if (!opponent.equals(champion) && !pool.contains(opponent)) {
      throw new IllegalStateException(
          "Macro " + t + " played against " + opponent.label() + ", who is neither the champion nor in the pool");
    }
    long bytes = Cohort.write(part, collected);
    CollectReport metrics =
        new CollectReport(
            iteration.version(),
            trajectories.games().size(),
            trajectories.size(),
            champion.label(),
            pool.size(),
            opponent.label(),
            bytes);
    StepReport report = StepReport.ran(iteration, started, metrics);
    report.write(part.resolve(CampaignStep.REPORT));
    DurableFiles.seal(part, sealed);
    LOG.info(
        "t={} collected {} games and {} decisions of version {} against {} (champion {}, {} past champions, {} bytes)",
        t,
        metrics.games(),
        metrics.decisions(),
        metrics.version(),
        metrics.opponent(),
        metrics.champion(),
        metrics.pool(),
        bytes);
    return report;
  }
}
