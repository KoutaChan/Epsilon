package com.epsilon.prism.collect;

import com.epsilon.agent.learn.StageTimer;
import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TablePlan;
import com.epsilon.agent.play.TableSeat;
import com.epsilon.agent.play.TableStage;
import com.epsilon.agent.runtime.InferenceAdmission;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.luck.LedgerRecording;
import com.epsilon.prism.luck.UraAudit;
import com.epsilon.prism.luck.UraValues;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.variant.PrismVariant;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * PRISM のコホートを 1 つ集める。卓のコア（{@link TableLoop}）に PRISM の処理（{@link CollectContext}・{@link CollectionPath}）を載せる。
 * 全席が同じバージョンの方策を行動方策 b で打ち、ラベル行と根の記録を行の書き出し先に書く。局末の値は、局末を作った進行のスレッドがその場で
 * Φ から求め、ドライバがその局の受け取り先（ラベル行・根の記録）に渡す。行の記録に失敗したら収集を止める。
 */
public final class Collector {

  /**
   * 集めるコホート。半荘 ID は run 全体で一意で、卓ではそのまま牌山の番号になる（割り当ては 0）。
   *
   * @param iterationSeed コホートの種（局のキーと乱数のキーの元）
   * @param draws 局のツモの順
   * @param sampling 収集の率（null なら {@link Sampling#OFF}）
   * @param ledger 運の精算の設定（計算する組と、値を切る幅。帳簿を付けない収集は null）
   * @param ura 立直の和了で終わった局の、局末の値の作り方（{@link UraValues}）
   * @param behaviour 全席の行動方策 b（null なら方策の σ）
   * @param twinDiscard 同値な打牌のうち、全ての経路で切る方
   * @param observer 主経路の対局を見る観測者（見なければ null。観測してもラベル行は変わらない）
   */
  public record Cohort(
      Variant variant,
      long runSeed,
      long iterationSeed,
      List<Long> gameIds,
      DrawOrder draws,
      Sampling sampling,
      LedgerRecording ledger,
      UraSettings ura,
      Behaviour behaviour,
      TwinDiscard twinDiscard,
      GameObserver observer) {
    public Cohort {
      Objects.requireNonNull(draws, "draws");
      Objects.requireNonNull(ura, "ura");
      Objects.requireNonNull(twinDiscard, "twinDiscard");
      if (sampling == null) sampling = Sampling.OFF;
    }
  }

  /**
   * 書いた行。
   *
   * @param rows 行の数
   * @param bytes 圧縮後のバイト数
   */
  public record RowStats(long rows, long bytes) {}

  /**
   * コホート 1 つを集めた結果。
   *
   * @param rows 書き出し先に渡した行の数（バイト数は 0。確定した後のマニフェストの値に {@link #withRows} で差し替える）
   * @param ura 裏ドラの期待値の監査
   */
  public record Result(
      CollectionMetrics.Snapshot counts,
      Map<CollectStage, StageTimer.Entry> stages,
      InferenceAdmission.Metrics admission,
      Phi.Stats phi,
      RowStats rows,
      long batches,
      int largestBatch,
      double seconds,
      UraAudit.Report ura) {
    /** 1 秒あたりの NN 推論の判断の数。 */
    public double modelDecisionsPerSecond() {
      return counts.modelDecisions() / seconds;
    }

    /** 書いた行だけを、確定した後のマニフェストの値に差し替えた結果。 */
    public Result withRows(RowStats written) {
      return new Result(counts, stages, admission, phi, written, batches, largestBatch, seconds, ura);
    }

    /**
     * 収集のログの 1 行。時間と速さ、写しと分岐の判断の数、半荘と局の数、書き出し先へ渡した行の数、ラベル行（MAIN・ROLLIN）、根の数（うち
     * 写しを作った数と d = played の数）、一様分布から出た手のために作らなかった根、分岐、|C ∪ {played}| の平均、ε_C で 1 手を足した判断、
     * 打った手が候補集合の外だった判断、主経路で b が貪欲でない手を打った判断、主経路で副露している席のラベル行を並べる。
     */
    public String summary() {
      long labels = 0, copyLabels = 0;
      for (int c = 0; c < counts.labelRows().length; c++) {
        labels += counts.labelRows()[c];
        copyLabels += counts.copyLabelRows()[c];
      }
      long rowsWithValues = labels + copyLabels;
      return String.format(
          Locale.ROOT,
          "collect %.1fs %.0f model decisions/s (copy %d, fork %d) games=%d hands=%d rows=%d"
              + " labels=%d rollin=%d roots=%d (copies %d, same %d, floor skipped %d) forks=%d"
              + " |C|=%.2f explored=%d outside=%d mainNonGreedy=%d openLabels=%d",
          seconds,
          modelDecisionsPerSecond(),
          counts.copyDecisions(),
          counts.forkDecisions(),
          counts.games(),
          counts.hands(),
          counts.rowsWritten(),
          labels,
          copyLabels,
          counts.rootCount(),
          counts.copies(),
          counts.sameDeviations(),
          counts.floorRoots(),
          counts.forks(),
          rowsWithValues == 0 ? 0.0 : counts.measured() / (double) rowsWithValues,
          counts.explored(),
          counts.playedOutside(),
          counts.mainNonGreedy(),
          counts.breakdown().openLabels());
    }
  }

  private Collector() {}

  /**
   * コホートを集め、ラベル行と根の記録を書き出し先に書く。
   *
   * @param policy 全席の方策
   * @param rows 行の書き出し先
   * @return 集めた結果。行の数はライターに渡した数で、バイト数は確定したマニフェストの値に後で差し替える
   * @throws IllegalStateException コホートのルール（四人打ち・三人打ち）の PRISM のモジュールが classpath に無いとき、または Φ の
   *     ルールがコホートと違うとき
   */
  public static Result collect(TableLoop table, Cohort cohort, PolicyService policy, Phi phi, RowSink rows) {
    Objects.requireNonNull(rows, "rows");
    if (!PrismVariant.present(cohort.variant()) || phi.variant() != cohort.variant())
      throw new IllegalStateException(
          "Variant mismatch: the cohort holds " + cohort.variant() + " games and Φ reads " + phi.variant()
              + (PrismVariant.present(cohort.variant()) ? "" : ", and no PRISM module reads " + cohort.variant()));
    Phi.Stats before = phi.stats();
    var context =
        new CollectContext(
            cohort.variant(),
            cohort.runSeed(),
            cohort.iterationSeed(),
            phi,
            cohort.sampling(),
            cohort.ledger(),
            cohort.ura(),
            cohort.behaviour(),
            cohort.twinDiscard(),
            new CollectionMetrics(),
            new StageTimer<>(CollectStage.class),
            rows);
    List<TableSeat<PolicyRequest>> seats = Collections.nCopies(cohort.variant().seats(), policy);
    TableLoop.Result played =
        table.play(
            new TablePlan<PolicyRequest>(
                cohort.variant(),
                cohort.runSeed(),
                cohort.iterationSeed(),
                GameKey.onePerWall(cohort.gameIds()),
                cohort.draws(),
                cohort.behaviour(),
                game -> seats,
                cohort.observer()),
            context);
    CollectionMetrics.Snapshot counts = context.metrics.snapshot(played);
    // 予約した行（ラベル行と根）は、全て値がそろって書き出し先へ渡っているはず。欠けたらコードの誤りなので、黙って捨てない
    if (counts.labels() + counts.rootCount() != counts.rowsWritten())
      throw new IllegalStateException(
          "Collection lost rows: labels="
              + counts.labels()
              + " roots="
              + counts.rootCount()
              + " written="
              + counts.rowsWritten());
    return new Result(
        counts,
        stages(context.timer, played.stages()),
        played.admission(),
        phi.stats().since(before),
        new RowStats(rows.rows(), 0),
        played.batches(),
        played.largestBatch(),
        played.seconds(),
        context.ura.report());
  }

  /** 収集の段と卓のコアの段を、収集の段の名前で 1 つの表にまとめる（{@link CollectStage} の順）。 */
  private static Map<CollectStage, StageTimer.Entry> stages(
      StageTimer<CollectStage> timer, Map<TableStage, StageTimer.Entry> table) {
    Map<CollectStage, StageTimer.Entry> stages = new EnumMap<>(CollectStage.class);
    stages.putAll(timer.snapshot());
    table.forEach((stage, entry) -> stages.put(CollectStage.of(stage), entry));
    return stages;
  }
}
