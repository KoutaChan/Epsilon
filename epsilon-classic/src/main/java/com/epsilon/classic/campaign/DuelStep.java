package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.Arena;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.Duel;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.settings.RecordFile;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 決闘の段。王者を替える卓（{@link EvaluationSettings#promoting}）をこの macro で打つときだけ行い、学習器の今のネットワークを候補として王者の
 * 保存先に書いてから、王者と決闘する（{@link Duel}）。牌山は run の種とジョブの名前から作る、決闘ごとの牌山。決闘は王者の保存先のジョブとして
 * 打つので、チャンクの後の安全点で止めても続きから打てる。
 *
 * <p>王者の保存先が空なら打たずに、候補を初めの王者にすると報告に書く（王者にするのは昇格の段）。ジョブが落ちた回数の上限で FAILED になれば、
 * 判定を未決として報告に書く。段の報告は work の {@value #FILE}（完了マーカー）。
 *
 * @param <I> campaign の反復
 */
public final class DuelStep<I extends DuelMacro> implements Step<I> {

  /** 段の報告（work の中）。 */
  static final String FILE = "DUEL.json";

  private static final Logger LOG = LoggerFactory.getLogger(DuelStep.class);

  @Override
  public boolean enabled(I macro) {
    return table(macro).isPresent();
  }

  @Override
  public Optional<StepReport> done(I macro) throws IOException {
    Path file = macro.work().resolve(FILE);
    return Files.exists(file) ? Optional.of(StepReport.read(file, DuelReport.class)) : Optional.empty();
  }

  @Override
  public StepReport run(I macro) throws Exception {
    long started = System.nanoTime();
    String table = table(macro).orElseThrow();
    TableSpec spec = macro.evaluation().tables().get(table);
    ChampionStore store = ChampionStore.open(macro.run().champion());
    Champion candidate = candidate(macro, store);
    String name = RunDirectory.jobName(macro.published(), table);
    Optional<Champion> champion = store.current();
    DuelReport report;
    if (champion.isEmpty()) {
      LOG.info(
          "{} plays no walls: the champion store is empty, so {} becomes the first champion",
          name,
          candidate.label());
      report = DuelReport.uncontested(name, candidate);
    } else {
      DuelSetup setup =
          DuelJobs.setup(
              macro,
              spec,
              DuelJobs.lineage(macro, candidate),
              DuelJobs.lineage(macro, champion.get()),
              Arena.freshSeed(macro.seed(), name));
      JobFile job = JobFile.pending(macro.published(), table, 0, macro.settingsRow(), RecordFile.tree(spec));
      Optional<DuelJobs.Played> played = DuelJobs.play(macro, store.duels(), job, setup);
      report =
          played.isPresent()
              ? DuelReport.played(name, candidate, champion.get(), played.get())
              : DuelReport.fell(name, candidate, champion.get());
    }
    StepReport step = StepReport.ran(macro, started, report);
    step.write(macro.work().resolve(FILE));
    return step;
  }

  /**
   * @return 王者を替える卓をこの macro で打つなら、その名前
   */
  static Optional<String> table(DuelMacro macro) {
    EvaluationSettings evaluation = macro.evaluation();
    return evaluation.promoting().filter(name -> evaluation.tables().get(name).scheduled(macro.published()));
  }

  /**
   * 段の報告に書いた決闘の結果。
   *
   * @throws IOException 決闘の段が済んでいないか、報告が読めないとき
   */
  static DuelReport recorded(DuelMacro macro) throws IOException {
    Path file = macro.work().resolve(FILE);
    if (!Files.exists(file)) {
      throw new IOException("Macro " + macro.number() + " has no " + FILE + ": its duel step is not done");
    }
    return (DuelReport) StepReport.read(file, DuelReport.class).metrics();
  }

  /** 候補。チェックポイントがまだ無ければ、学習器の今のネットワークを書く。 */
  private static Champion candidate(DuelMacro macro, ChampionStore store) throws IOException {
    String label = DuelJobs.label(macro.published());
    Path checkpoint = store.candidate(label);
    if (!Files.isDirectory(checkpoint)) {
      macro.writeLearner(checkpoint);
    }
    return Champion.of(label, checkpoint, macro.published());
  }
}
