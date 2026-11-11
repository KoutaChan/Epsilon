package com.epsilon.prism.run;

import com.epsilon.agent.run.SafePoint;
import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepReport;
import com.epsilon.prism.grp.ActivePhi;
import com.epsilon.prism.grp.GrpBackground;
import com.epsilon.prism.grp.GrpFiles;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Φ の段。自分の GRP を学ぶ run では、GRP のバックグラウンドの作業が反復 t − 1 の区切り（監視と関門）を終えるのを待ってから、
 * 反復 t で使う項目を phi.jsonl から決める。項目のモードは mode.jsonl から決め、求めたモードと実際に使ったモードを報告に書く（点数だけの
 * Φ のモードはいつも GRP）。待つ間は {@link com.epsilon.agent.run.RunControl#CHECK_INTERVAL} ごとにセーフポイントの確かめを行い、
 * 止めるなら何も書かずに抜ける。完了マーカーは、決めた値を書いた work の {@value #FILE} の rename。
 */
final class PhiStep implements Step<PrismIteration> {

  /** 完了マーカーになるファイル（work の中）。 */
  static final String FILE = GrpFiles.PHI_STEP;

  private static final Logger LOG = LoggerFactory.getLogger(PhiStep.class);

  @Override
  public Optional<StepReport> done(PrismIteration iteration) throws IOException {
    Path file = iteration.work().resolve(FILE);
    return Files.exists(file) ? Optional.of(StepReport.read(file, PhiReport.class)) : Optional.empty();
  }

  @Override
  public StepReport run(PrismIteration iteration) throws Exception {
    long started = System.nanoTime();
    int t = iteration.number();
    GrpBackground background = iteration.session().background();
    iteration
        .session()
        .control()
        .await(SafePoint.waiting(PrismStep.PHI.name()), timeout -> background.awaitBoundary(t - 1, timeout));
    double waited = (System.nanoTime() - started) / 1e9;
    if (waited >= 1) {
      LOG.info("t={} waited {}s for the GRP boundary of c{}", t, String.format("%.1f", waited), t - 1);
    }
    // 固定の GRP は、この反復に決めた行のファイルで開く（起動したときの行と違えば開き直す）
    ActivePhi phi = iteration.session().resources().phi(t, iteration.settings().phi());
    StepReport report = StepReport.ran(iteration, started, new PhiReport(phi.value(), waited));
    report.write(iteration.work().resolve(FILE));
    iteration.phi(phi);
    return report;
  }

  /**
   * 完了マーカーに書いた Φ の報告を読む。
   *
   * @throws IOException 完了マーカーが無いか、読めないとき
   */
  static PhiReport recorded(PrismIteration iteration) throws IOException {
    Path file = iteration.work().resolve(FILE);
    if (!Files.exists(file)) {
      throw new IOException("Iteration " + iteration.number() + " has no " + FILE + ": its Φ step is not done");
    }
    return (PhiReport) StepReport.read(file, PhiReport.class).metrics();
  }
}
