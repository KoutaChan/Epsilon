package com.epsilon.prism.run;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.collect.CollectionMetrics;
import com.epsilon.prism.collect.Collector;
import com.epsilon.prism.collect.RowSink;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.grp.ActivePhi;
import com.epsilon.prism.grp.GameLogObserver;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiSetup;
import com.epsilon.prism.luck.LedgerFit;
import com.epsilon.prism.luck.LuckSettlement;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 収集の段。全席が開始バージョンのネットワーク（どれを使うかは {@code collectWith} の設定で選ぶ）で打つ。打ち方は貪欲な手に一様分布を
 * 混ぜた方策 b。ラベルの行・ロールイン・根の記録・帳簿は rows/&lt;t&gt;.part に書く。同じディレクトリに段の報告
 * （{@value PrismStep#REPORT}）とクラスごとの数（{@value CollectReport#CLASSES}）も書き、自分の GRP を学ぶ run なら主経路の半荘の記録
 * （{@link GrpFiles#unmovedRecords}）と数のファイル（{@link GrpFiles#unmovedCounts}）も入れる。最後に rows/&lt;t&gt; へ rename して
 * 確定する。この rename が完了マーカーになる。
 *
 * <p>確定の後の仕上げでは、半荘の記録を grp/records/c&lt;t&gt;.bin へ、数のファイルを grp/counts/c&lt;t&gt;.bin へ rename する。どちらかが
 * コホートの中に残っている間は、GRP のバックグラウンドの作業はそのコホートを確定していないとみなす（{@link GrpFiles#sealed}）。仕上げは
 * 完了マーカーを見るたびに行い、その後でバックグラウンドの作業を要求する。記録や数のファイルが書けなくても系譜は止めない。
 */
final class CollectStep implements Step<PrismIteration> {

  private static final Logger LOG = LoggerFactory.getLogger(CollectStep.class);

  @Override
  public Optional<StepReport> done(PrismIteration iteration) throws IOException {
    Path sealed = iteration.run().rows(iteration.number());
    if (!Files.isDirectory(sealed)) {
      return Optional.empty();
    }
    finish(iteration.run(), iteration.number());
    iteration.session().background().request();
    StepReport report = StepReport.read(sealed.resolve(PrismStep.REPORT), CollectReport.class);
    iteration.collected(((CollectReport) report.metrics()).start());
    return Optional.of(report);
  }

  @Override
  public StepReport run(PrismIteration iteration) throws Exception {
    long started = System.nanoTime();
    int t = iteration.number();
    RunDirectory run = iteration.run();
    Path sealed = run.rows(t);
    Path part = PrismStep.part(sealed);
    DurableFiles.delete(part);
    PrismSettings settings = iteration.settings();
    PrismSettings machine = iteration.machine();
    Variant variant = settings.run().variant();
    ActivePhi phi = iteration.phi();
    TrainingResources resources = iteration.session().resources();
    LuckSettlement luck = iteration.luck();
    LedgerFit.WriterTap ledger = luck.tap(machine.table().writers(), settings.fit().rows());
    GameLogObserver games = gameLog(t, phi, settings);
    InferenceSettings inference = machine.inference();
    Path snapshot = run.snapshot(iteration.start());
    Collector.Result collected;
    try (NetworkPolicy policy =
            Snapshot.open(
                snapshot,
                Snapshot.net(snapshot, settings.memory().collectWith()),
                inference.networkDevices().devices(),
                resources.runtime(),
                inference.maxBatch(),
                settings.inference().precision(),
                InferencePath.of(inference.kernels()));
        CohortWriter writer =
            new CohortWriter(
                part,
                variant,
                t,
                luck.width(),
                machine.table().writers(),
                ledger == null ? CohortWriter.Tap.NONE : ledger)) {
      // 推論の計測をこの収集から数え直す（同じ JVM で前に行った評価の数を持ち越さない）
      NetworkRuntime runtime = resources.runtime();
      runtime.resetInferenceMetrics();
      long seed = settings.run().seed();
      collected =
          Collector.collect(
              resources.table(),
              new Collector.Cohort(
                  variant,
                  seed,
                  CrnSeeds.derive(seed, CrnSeeds.Domain.HAND_KEY, t),
                  settings.schedule().gameIds(t),
                  settings.noise().crn().draws(),
                  Sampling.of(settings),
                  luck.recording(),
                  luck.ura(),
                  new Behaviour(
                      settings.behaviour().uniformFloor(),
                      settings.behaviour().exploreTemperature(),
                      settings.noise().crn().actions().scheme()),
                  settings.behaviour().twinDiscard(),
                  games),
              policy,
              phi.phi(),
              new RowSink(writer));
      // 推論が全て返るのを待ち、方策を閉じる前に、受付とデバイスごとの待ち・合成・空きをログに書く
      runtime.awaitInferenceIdle();
      runtime.logInference(collected.admission());
      writeGameLog(t, part, games, collected.counts());
      writer.seal();
    }
    CohortWriter.Manifest written = CohortWriter.readManifest(part);
    CollectionMetrics.Snapshot counts = collected.counts();
    if (written.rows() != counts.rowsWritten()) {
      throw new IllegalStateException(
          "Cohort " + t + " sealed " + written.rows() + " rows but " + counts.rowsWritten() + " were sent");
    }
    collected = collected.withRows(new Collector.RowStats(written.rows(), written.bytes()));
    LOG.info("t={} {}", t, collected.summary());
    DurableFiles.write(part.resolve(CollectReport.CLASSES), CollectReport.Classes.of(counts));
    StepReport report =
        StepReport.ran(iteration, started, CollectReport.of(iteration.start(), collected, written.rows(), written.bytes()));
    report.write(part.resolve(PrismStep.REPORT));
    DurableFiles.seal(part, sealed);
    finish(run, t);
    iteration.session().background().request();
    iteration.ledger(ledger);
    return report;
  }

  /**
   * 確定の後の仕上げ。確定したコホートの半荘の記録を grp/records へ、数のファイルを grp/counts へ rename する。ファイルが無いか、
   * もう移してあれば何もしない。
   *
   * @param t 確定したコホートの番号
   */
  static void finish(RunDirectory run, int t) throws IOException {
    GrpFiles grp = new GrpFiles(run.root());
    boolean records = move(grp.unmovedRecords(t), grp.records(t));
    boolean counts = move(grp.unmovedCounts(t), grp.counts(t));
    if (records || counts) {
      DurableFiles.syncDirectory(run.rows(t));
    }
  }

  /** コホートの中のファイルを rename で移す。無ければ何もせず false を返す。 */
  private static boolean move(Path unmoved, Path target) throws IOException {
    if (!Files.exists(unmoved)) {
      return false;
    }
    DurableFiles.createDirectories(target.getParent());
    Files.move(unmoved, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    DurableFiles.syncDirectory(target.getParent());
    return true;
  }

  /** コホート {@code t} で主経路の半荘を記録するオブザーバーを開く。記録しない run では null。開けなければ警告を 1 回ログに書いて null を返す。 */
  private static GameLogObserver gameLog(int t, ActivePhi phi, PrismSettings settings) {
    if (!new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed()).records()) {
      return null;
    }
    try {
      return GameLogObserver.open(t, phi.value(), settings.behaviour().exploreTemperature());
    } catch (RuntimeException failure) {
      LOG.warn("t={} the game log is off for this cohort: {}", t, failure.toString());
      return null;
    }
  }

  /**
   * 半荘の記録と数のファイルを、確定する前のコホートのディレクトリに書く。記録しない run では何もしない。どちらかが書けなくても、警告を
   * 1 回ログに書いて続ける。
   *
   * @param part 確定する前のコホートのディレクトリ
   */
  private static void writeGameLog(int t, Path part, GameLogObserver games, CollectionMetrics.Snapshot counts) {
    if (games == null) {
      return;
    }
    try {
      GameRecordCodec.write(part.resolve(GrpFiles.UNMOVED_RECORDS), games.seal(counts));
    } catch (IOException | RuntimeException failure) {
      LOG.warn("t={} the game log was not written: {}", t, failure.toString());
    }
    try {
      CountFile.write(part.resolve(GrpFiles.UNMOVED_COUNTS), games.countedHands(counts));
    } catch (IOException | RuntimeException failure) {
      LOG.warn("t={} the hand counts were not written: {}", t, failure.toString());
    }
  }
}
