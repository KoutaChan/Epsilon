package com.epsilon.prism.run;

import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.learn.AnchorScores;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.learn.Teacher;
import com.epsilon.prism.luck.LedgerOutcome;
import com.epsilon.prism.luck.LuckSettlement;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.noise.TempoSettings;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCounts;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 学習の段。運の精算の帳簿を当てはめる反復では、帳簿の係数を半荘の偶奇で分けて交差適合で当てはめる（{@link LuckSettlement}）。
 * そのうえで、コホート t の生徒の行（どの行を使うかは設定で決める）で、バージョン start のネットワーク θ から生徒を学習する
 * （{@link Fit}）。θ に平均と学習中のどちらのネットワークを使うかは {@code learnFrom} の設定で選ぶ。
 *
 * <p>教師は (1 − K′_t)·s_θ + K′_t·Ỹ。Ỹ は帳簿で補正した Y で、補正しなければ生の Y。アンカーは凍結した θ。補助のヘッドの項も
 * K′_t 倍し、学習率には {@code lrMode} が K′_t から決める倍率を掛ける。K′_t は、前の反復で教師に掛けた補正のホールドアウトでの R² の
 * 分だけ K_t を上げたゲイン。持ち越しのキーが違えば R² を捨てて K_t を使う。
 *
 * <p>帳簿の当てはめは CPU だけを使う。その間に別のスレッドで、全ての生徒の行のアンカーの s を出し（アンカーを混ぜる設定のときだけ）、
 * 生徒を読み込んで学習器を準備する。生徒・段の報告・クラスごとの帳簿の当てはめ・帳簿の当てはめの報告の全体
 * （{@value LearnReport#LEDGER}。当てはめた反復だけ）は work/&lt;t&gt;/learn.part に書き、learn に rename する。この rename が完了
 * マーカーになる。途中で落ちたら初めから学習し直す。
 */
final class LearnStep implements Step<PrismIteration> {

  /** 学習の段の完了マーカーになるディレクトリ（work の中）。 */
  static final String LEARN = "learn";

  /** 学習の段の出力に入る、生徒のネットワーク。 */
  static final String STUDENT = "student";

  /** 学習で行を並べ替える乱数の種のソルト（derive(seed, LEARN, t, 4)）。 */
  private static final int SINGLE_SALT = 4;

  private static final Logger LOG = LoggerFactory.getLogger(LearnStep.class);

  @Override
  public Optional<StepReport> done(PrismIteration iteration) throws IOException {
    Path learned = iteration.work().resolve(LEARN);
    return Files.isDirectory(learned)
        ? Optional.of(StepReport.read(learned.resolve(PrismStep.REPORT), LearnReport.class))
        : Optional.empty();
  }

  @Override
  public StepReport run(PrismIteration iteration) throws Exception {
    long started = System.nanoTime();
    int t = iteration.number();
    RunDirectory run = iteration.run();
    Path learned = iteration.work().resolve(LEARN);
    Path part = PrismStep.part(learned);
    DurableFiles.delete(part);
    DurableFiles.createDirectories(part);
    PrismSettings settings = iteration.settings();
    TempoSettings tempo = settings.noise().tempo();
    CarryKey key = CarryKey.of(settings, iteration.phiEntry());
    PrismCarry carry = iteration.state().carry();
    String reset = key.resetReason(carry.key(), tempo.resetsOnPhiKind());
    double priorR2 = reset == null ? carry.ledgerR2() : 0;
    if (reset != null && carry.ledgerR2() > 0) {
      LOG.info("t={} drops the carried R² {}: {}", t, carry.ledgerR2(), reset);
    }
    Learned learnt;
    try (CohortStore store = new CohortStore()) {
      store.addCohort(run.rows(t));
      RowCounts counts = RowCounts.of(store.select(t, t, index -> true), store::index);
      int[] rows = store.select(t, t, settings.fit().rows());
      LOG.info("t={} student rows {} (holdout rows {})", t, rows.length, counts.holdout());
      learnt = learn(iteration, store, rows, counts.holdout(), priorR2, reset, part.resolve(STUDENT));
    }
    DurableFiles.write(part.resolve(LearnReport.CLASSES), learnt.ledger().classes());
    if (learnt.ledger().report() != null) {
      DurableFiles.write(part.resolve(LearnReport.LEDGER), learnt.ledger().report().full());
    }
    StepReport step = StepReport.ran(iteration, started, learnt.report());
    step.write(part.resolve(PrismStep.REPORT));
    DurableFiles.seal(part, learned);
    return step;
  }

  /**
   * 学習の結果。
   *
   * @param ledger 運の精算の結果
   */
  private record Learned(LearnReport report, LedgerOutcome ledger) {}

  /**
   * 帳簿を当てはめ、生徒を学習して保存する。生徒を学習するときは、帳簿の当てはめの間に別のスレッドで、アンカーの s の計算（アンカーを
   * 混ぜる設定のときだけ）と学習器の準備を進める。帳簿は CPU だけを使うので、その間も学習のデバイスは空いている。
   *
   * @param holdout コホートのホールドアウトの行の数
   * @param priorR2 K′ を決める R²
   * @param reset 持ち越しを捨てた理由（捨てなければ null）
   * @param student 生徒を保存するディレクトリ
   */
  private static Learned learn(
      PrismIteration iteration, CohortStore store, int[] rows, long holdout, double priorR2, String reset, Path student)
      throws IOException {
    int t = iteration.number();
    PrismSettings settings = iteration.settings();
    PrismSettings machine = iteration.machine();
    NetworkDevices devices = machine.learner().networkDevices();
    int threads = machine.learner().decodeThreads();
    NetworkRuntime learning = iteration.session().resources().learning();
    Path start = Snapshot.net(iteration.run().snapshot(iteration.start()), settings.memory().learnFrom());
    TempoSettings tempo = settings.noise().tempo();
    int steps = settings.fit().steps(rows.length);
    boolean fits = steps > 0;
    double gain = tempo.gain(t, priorR2);
    LedgerOutcome ledger;
    AnchorScores anchor = null;
    Fit.Prepared prepared;
    try (ExecutorService ahead = Executors.newFixedThreadPool(2)) {
      Future<AnchorScores> scores =
          fits && tempo.anchored()
              ? ahead.submit(
                  () ->
                      AnchorScores.compute(
                          store, rows, start, iteration.start(), InferencePath.of(machine.inference().kernels()),
                          devices, threads))
              : null;
      Future<Fit.Prepared> warmed =
          fits
              ? ahead.submit(
                  () ->
                      Fit.prepare(
                          store,
                          rows,
                          start,
                          plan(settings, t, steps, priorR2, start),
                          (float) gain,
                          devices,
                          threads,
                          learning))
              : null;
      try {
        ledger = settle(iteration, store, rows, threads);
        if (scores != null) {
          anchor = await(scores);
        }
      } catch (IOException | RuntimeException | Error failure) {
        if (warmed != null) {
          closeAfter(warmed, failure);
        }
        throw failure;
      }
      prepared = warmed == null ? null : await(warmed);
    }
    double anchorWeight = tempo.anchored() ? 1 - gain : 0;
    LearnReport.Tempo filter =
        new LearnReport.Tempo(
            gain,
            anchorWeight,
            tempo.learningRateScale(t, priorR2),
            tempo instanceof TempoSettings.On on ? on.scheduledGain(t) : gain,
            priorR2);
    FitResult result;
    try {
      if (prepared == null) {
        Snapshot.copyModel(start, student);
        result = FitResult.empty(rows.length);
      } else {
        if (tempo.anchored()) {
          anchor.requireVersion(iteration.start());
        }
        Teacher teacher = new Teacher(ledger.correction(), tempo.anchored() ? anchor : null, anchorWeight, (float) gain);
        result = prepared.fit(teacher);
        prepared.network().save(student);
      }
    } finally {
      if (prepared != null) {
        prepared.close();
      }
    }
    LOG.info(result.summary(t));
    LearnReport report =
        new LearnReport(
            iteration.start(),
            rows.length,
            holdout,
            reset,
            filter,
            ledger.summary(),
            ledger.appliedR2(),
            LearnReport.Fit.of(result));
    return new Learned(report, ledger);
  }

  /**
   * 反復 t で生徒を学習する計画を作る。学習率には {@code lrMode} の倍率を掛け、ネットワークの形は始めのネットワークに合わせる。行を
   * 並べ替える乱数の種は derive(seed, LEARN, t, {@value #SINGLE_SALT})。
   *
   * @param steps ステップ数（1 以上）
   * @param priorR2 K′ を決める R²
   * @param start 始めのネットワーク
   */
  private static FitPlan plan(PrismSettings settings, int t, int steps, double priorR2, Path start) throws IOException {
    return new FitPlan(
        settings.fit().fit(steps, settings.noise().tempo().learningRateScale(t, priorR2)),
        PrismNetwork.readShape(start),
        settings.fit().auxiliary(),
        FitPlan.TOKEN_BUDGET,
        CrnSeeds.derive(settings.run().seed(), CrnSeeds.Domain.LEARN, t, SINGLE_SALT));
  }

  /**
   * 反復の運の精算のうち、学習の段で行う部分（{@link LuckSettlement#learn}）。和は、収集のライターが同じ JVM で作ったものがあれば
   * それを使い、なければ行のストアから行を読み直して作る。どちらも同じ行を同じ順に足すので、係数はビット単位で同じになる。
   */
  private static LedgerOutcome settle(PrismIteration iteration, CohortStore store, int[] rows, int threads) {
    LuckSettlement luck = iteration.luck();
    try (ExecutorService readers = Executors.newFixedThreadPool(threads)) {
      Iterable<Row> stored = store.iterate(rows, iteration.settings().fit().batch(), readers, threads);
      return luck.learn(iteration.number(), iteration.ledger(), stored, rows.length);
    }
  }

  /** 失敗の後で、先に始めた学習器の準備が終わるのを待って閉じる。準備と閉じるときの例外は、元の失敗に添える。 */
  private static void closeAfter(Future<Fit.Prepared> warmed, Throwable failure) {
    try {
      await(warmed).close();
    } catch (IOException | RuntimeException | Error problem) {
      failure.addSuppressed(problem);
    }
  }

  /** 先に始めたジョブの完了を待つ。ジョブが投げた例外は、包みを外して投げ直す。 */
  private static <T> T await(Future<T> future) throws IOException {
    try {
      return future.get();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while waiting for the learner preparation", interrupted);
    } catch (ExecutionException failure) {
      Throwable cause = failure.getCause();
      if (cause instanceof IOException unreadable) {
        throw unreadable;
      }
      if (cause instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw new IOException("The learner preparation failed", cause);
    }
  }
}
