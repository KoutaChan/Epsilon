package com.epsilon.prism.luck;

import com.epsilon.agent.settings.Case;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.noise.LedgerCalibration;
import com.epsilon.prism.noise.LedgerCalibrationSource;
import com.epsilon.prism.noise.LedgerCells;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.noise.LedgerSettings;
import com.epsilon.prism.noise.NoiseSettings;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Predicate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 1 反復ぶんの運の精算を、記録 → 当てはめ → 監査 → 補正の順に受け持つ。反復ごとに、設定のノイズの項目とルールから作る。帳簿を扱う処理は
 * 全てこのクラスを通す。
 *
 * <p>帳簿の段ごとの動き:
 *
 * <ul>
 *   <li>OFF: 帳簿を計算せず、行の帳簿は 0 列。当てはめも補正もせず、持ち越す R² は 0。
 *   <li>RECORD: 計算して行に書くが、当てはめない。教師は Y で、持ち越す R² は 0。
 *   <li>AUDIT: 当てはめて監査を報告するが、教師は Y のまま。持ち越す R² は 0。
 *   <li>APPLY: 当てはめて監査を報告し、教師を Ỹ = Y − β·L にする。持ち越す R² は、掛けた補正のホールドアウトの R²。
 * </ul>
 *
 * <p>帳簿の補正率（{@link LedgerCalibration}。当てはめる段だけ）の動き:
 *
 * <ul>
 *   <li>OFF: 補正率を求めない。上の段の動きのまま。
 *   <li>SHADOW: 当てはめに入れた行を貯め、係数を当てはめた後でもう一度読んで、前の反復から持ち越した補正率 h を掛けた補正 Y − h(o)·β·L
 *       を測り、今のコホートの行で次の反復の補正率を当てはめる（{@link LedgerCalibrator}）。教師と持ち越す R² は上の段の動きのまま。
 *       もう一度読むのは教師を決めた後（{@link Settled#calibrate}）なので、生徒の学習と並べて回せる。
 *   <li>APPLY: SHADOW に加えて、教師を Ỹ = Y − h(o)·β·L にし、持ち越す R² を、その補正のホールドアウトのスロットの R² にする。
 * </ul>
 *
 * <p>持ち越した補正率が、補正率の設定の学習元の選び方（{@link LedgerCalibrationSource}）のコホートのものでないか、形・ルール・帳簿の
 * 列・補正に使う組・区分が違うか、当てはめた反復の補正 β·L の意味（{@link LedgerMeaning#scaleFree}。帳簿の切り幅なども含む）が今と
 * 違うときは、補正率を 1（帳簿の補正のまま）にして、理由を報告に書く。格子と罰の強さは、持ち越した補正率のものをそのまま使い、今の設定の値は次の反復の補正率を当てはめるときに使う。
 * 報告の掛けた腕（{@link LedgerReport.Arm}）は実際の教師から決め、帳簿の補正が無ければ RAW とする。
 */
public final class LuckSettlement {

  private static final Logger LOG = LoggerFactory.getLogger(LuckSettlement.class);

  private final LedgerCatalog catalog;

  /** 収集に渡す値（帳簿を書かない段では null）。 */
  private final LedgerRecording recording;

  /** 係数の当てはめ（当てはめない段では null）。 */
  private final LedgerFit fit;

  /** 偏りの警告のしきい値（z の絶対値。当てはめない段では使わない）。 */
  private final double biasZ;

  private final boolean corrects;
  private final UraSettings ura;
  private final LedgerMeaning meaning;

  /** 帳簿の補正率の設定（補正率を求めない段では null）。 */
  private final LedgerCalibration calibration;

  private final Variant variant;

  /** 補正に使う組と、係数を当てはめる区分（当てはめない段では空と null）。 */
  private final Set<LedgerGroup> apply;

  private final LedgerCells cells;

  private LuckSettlement(
      LedgerCatalog catalog,
      LedgerRecording recording,
      LedgerFit fit,
      double biasZ,
      boolean corrects,
      UraSettings ura,
      LedgerMeaning meaning,
      LedgerCalibration calibration,
      Variant variant,
      Set<LedgerGroup> apply,
      LedgerCells cells) {
    this.catalog = catalog;
    this.recording = recording;
    this.fit = fit;
    this.biasZ = biasZ;
    this.corrects = corrects;
    this.ura = ura;
    this.meaning = meaning;
    this.calibration = calibration;
    this.variant = variant;
    this.apply = apply;
    this.cells = cells;
  }

  /** 反復の設定のうちノイズの項目（帳簿・当てはめの重み・裏ドラ）と、ルールから作る。 */
  public static LuckSettlement of(NoiseSettings noise, Variant variant) {
    LedgerSettings ledger = noise.ledger();
    LedgerCatalog catalog = LedgerCatalog.of(variant);
    LedgerSettings.Fitting fitting = ledger instanceof LedgerSettings.Fitting f ? f : null;
    return new LuckSettlement(
        catalog,
        LedgerRecording.of(ledger),
        fitting == null
            ? null
            : new LedgerFit(
                fitting.apply(),
                fitting.ridge(),
                fitting.minRowsPerColumn(),
                fitting.cells(),
                variant),
        fitting == null ? Double.NaN : fitting.biasZ(),
        ledger.corrects(),
        noise.ura(),
        LedgerMeaning.of(ledger, catalog, noise.lossWeight(), noise.ura()),
        fitting != null && fitting.calibration().scales() ? fitting.calibration() : null,
        variant,
        fitting == null ? Set.of() : fitting.apply(),
        fitting == null ? null : fitting.cells());
  }

  /** 収集に渡す帳簿の値。帳簿を書かない段では null。 */
  public LedgerRecording recording() {
    return recording;
  }

  /** 立直の和了で終わった局で、局末の値をどう作るか。収集に渡し、収集はこれで {@link UraValues} を作る。 */
  public UraSettings ura() {
    return ura;
  }

  /** 行に書く帳簿の列の数（測った手 1 つあたり）。コホートのマニフェストに書く。帳簿を書かない段では 0。 */
  public int width() {
    return recording == null ? 0 : catalog.width();
  }

  /**
   * コホートのライターから行を受け取り、当てはめに使う和を収集の間に作る受け取り先を返す。補正率を求める段では、行も貯める。
   *
   * @param writers コホートのライターの数
   * @param rows 和に入れる行（学習の段が選ぶ行と同じ選び方）
   * @return 和を作る受け取り先。当てはめない段では null
   */
  public LedgerFit.WriterTap tap(int writers, Predicate<RowCodec.Index> rows) {
    if (fit == null) {
      return null;
    }
    return calibration != null ? fit.tapRetaining(writers, rows) : fit.tap(writers, rows);
  }

  /** {@link #learn(int, LedgerFit.WriterTap, Iterable, int, LedgerCalibrator, LedgerMeaning)} で、補正率を持ち越していないとき。 */
  public LedgerOutcome learn(
      int cohort, LedgerFit.WriterTap streamed, Iterable<Row> stored, int selected) {
    return learn(cohort, streamed, stored, selected, null, meaning);
  }

  /**
   * {@link #learn(int, LedgerFit.WriterTap, Iterable, int, LedgerCalibrator, LedgerMeaning)} で、持ち越した補正率を今と同じ帳簿の設定で
   * 当てはめたとき。
   */
  public LedgerOutcome learn(
      int cohort,
      LedgerFit.WriterTap streamed,
      Iterable<Row> stored,
      int selected,
      LedgerCalibrator carried) {
    return learn(cohort, streamed, stored, selected, carried, meaning);
  }

  /**
   * 学習の段で帳簿を扱う（{@link #settle} の後に {@link Settled#calibrate} を続けて回す）。当てはめる段では係数を当てはめて報告し、偏りの z
   * の絶対値がしきい値を越えた列があれば警告を 1 行出す。補正する段では、教師の補正と、掛けた補正の R² も返す。補正率を求める段では、
   * 持ち越した補正率を掛けた補正を測り、次の反復の補正率を当てはめる。
   *
   * @param cohort コホートの番号（ログとエラーメッセージに書く）
   * @param streamed 同じ JVM の収集が作った和（無ければ null。あれば行を読まない）
   * @param stored 学習の段が選んだ行（和が無いときだけ読む）
   * @param selected 学習の段が選んだ行の数
   * @param carried 前の反復から持ち越した補正率（無ければ null）
   * @param fitUnder 持ち越した補正率を当てはめた反復の帳簿の意味（{@link #meaning}。前の反復の持ち越しのキーにある）
   * @return 帳簿の結果。当てはめない段では {@link LedgerOutcome#NONE}
   * @throws IllegalStateException 和に足した行の数が、選んだ行の数と違うとき
   */
  public LedgerOutcome learn(
      int cohort,
      LedgerFit.WriterTap streamed,
      Iterable<Row> stored,
      int selected,
      LedgerCalibrator carried,
      LedgerMeaning fitUnder) {
    return settle(cohort, streamed, stored, selected, carried, fitUnder).calibrate();
  }

  /**
   * 学習の段の帳簿の扱いのうち、教師を決めるまで。係数を当てはめて報告し、教師に掛ける補正を決める（補正率を教師に使う段では、持ち越した
   * 補正率を掛けた補正）。補正率を求める段の 2 回目の走査（補正率の報告・次の反復の補正率・掛けた補正の R²）は教師に要らないので、
   * {@link Settled#calibrate} に残し、生徒の学習と並べて回せるようにする。引数は {@link #learn} と同じ。
   *
   * @throws IllegalStateException 和に足した行の数が、選んだ行の数と違うとき
   */
  public Settled settle(
      int cohort,
      LedgerFit.WriterTap streamed,
      Iterable<Row> stored,
      int selected,
      LedgerCalibrator carried) {
    return settle(cohort, streamed, stored, selected, carried, meaning);
  }

  /**
   * {@link #settle(int, LedgerFit.WriterTap, Iterable, int, LedgerCalibrator)} で、持ち越した補正率を当てはめた反復の帳簿の意味を
   * 渡すとき（引数は {@link #learn(int, LedgerFit.WriterTap, Iterable, int, LedgerCalibrator, LedgerMeaning)} と同じ）。
   *
   * @throws IllegalStateException 和に足した行の数が、選んだ行の数と違うとき
   */
  public Settled settle(
      int cohort,
      LedgerFit.WriterTap streamed,
      Iterable<Row> stored,
      int selected,
      LedgerCalibrator carried,
      LedgerMeaning fitUnder) {
    if (fit == null) {
      return new Settled(LedgerOutcome.NONE, null, null);
    }
    LedgerFit.Result result;
    Iterable<FitRow> retained = null;
    if (streamed != null) {
      result = streamed.fit();
      long seen = result.report().rows() + result.report().skippedRows();
      if (seen != selected) {
        throw new IllegalStateException(
            "Ledger of cohort "
                + cohort
                + " summed "
                + seen
                + " rows but "
                + selected
                + " are selected");
      }
      if (calibration != null) {
        retained = streamed.retained();
      }
    } else if (calibration != null) {
      LedgerFit.Retained kept = fit.fitRetaining(stored);
      result = kept.result();
      retained = kept.rows();
    } else {
      result = fit.fit(stored);
    }
    LedgerReport report = result.report();
    LedgerReport.Summary summary = LedgerReport.Summary.of(report, biasZ, streamed != null);
    LOG.info(
        String.format(
            Locale.ROOT,
            "t=%d ledger %s rows=%d skipped=%d r2=%.4f forks=%.3f groups=%s %.1fs",
            cohort,
            streamed != null ? "streamed" : "re-read",
            report.rows(),
            report.skippedRows(),
            report.all().r2(),
            report.all().equivalentForks(),
            report.all().r2ByGroup(),
            report.seconds()));
    if (!summary.biased().isEmpty()) {
      LOG.warn(
          "t={} the ledger columns {} have |z| above {} (max |z| {})",
          cohort,
          summary.biased(),
          biasZ,
          summary.maxAbsZ());
    }
    LedgerCorrection correction = corrects && report.rows() > 0 ? result.correction() : null;
    if (corrects && correction == null) {
      LOG.warn("t={} no rows carry a ledger: the student learns the raw Y", cohort);
    }
    double r2 = report.all().r2();
    double applied = correction == null || !Double.isFinite(r2) ? 0 : Math.max(0, r2);
    LedgerOutcome fitted = new LedgerOutcome(report, summary, correction, applied);
    if (calibration == null) {
      return new Settled(fitted, null, correction);
    }
    String fallback = mismatch(carried, fitUnder, cohort);
    LedgerCalibrator scale = fallback == null ? carried : null;
    LedgerReport.Arm arm =
        correction == null
            ? LedgerReport.Arm.RAW
            : calibration.applies() && scale != null
                ? LedgerReport.Arm.COMMON_SCALE
                : LedgerReport.Arm.CLASS;
    return new Settled(
        fitted,
        new Pending(cohort, retained, result.correction(), scale, fallback, arm),
        arm == LedgerReport.Arm.COMMON_SCALE ? correction.scaled(scale) : correction);
  }

  /**
   * 補正率の 2 回目の走査に渡す値。
   *
   * @param rows 当てはめに入れた行（走査の後に放す）
   * @param current 帳簿の補正（補正率を掛けない β·L）
   * @param scale 今のコホートに掛ける補正率（出どころが無ければ null）
   * @param fallback 補正率を 1 にした理由（補正率を掛けるなら null）
   * @param applied 教師に実際に掛けた腕
   */
  private record Pending(
      int cohort,
      Iterable<FitRow> rows,
      LedgerCorrection current,
      LedgerCalibrator scale,
      String fallback,
      LedgerReport.Arm applied) {}

  /**
   * 教師を決めた後の帳簿の扱い（{@link #settle} が返す）。{@link #correction} は教師に掛ける補正で、{@link #calibrate} が補正率の 2 回目の
   * 走査を回して運の精算の結果を返す。補正率を求めない段では、走査は無く、結果をすぐに返す。{@link #calibrate} は別のスレッドから
   * 呼んでよい。
   */
  public final class Settled {
    private final LedgerOutcome fitted;
    private final LedgerCorrection teacher;
    private Pending pending;
    private LedgerOutcome outcome;

    private Settled(LedgerOutcome fitted, Pending pending, LedgerCorrection teacher) {
      this.fitted = fitted;
      this.pending = pending;
      this.teacher = teacher;
      outcome = pending == null ? fitted : null;
    }

    /** 教師に掛ける補正（補正しない段では null）。補正率を教師に使う段では、持ち越した補正率を掛けた補正。 */
    public LedgerCorrection correction() {
      return teacher;
    }

    /**
     * 補正率の 2 回目の走査を回し、運の精算の結果を返す。補正率を教師に使う段では、掛けた補正の R² を、補正率を掛けた補正のスロットの R²
     * にする。走査した行はここで放す。2 度目からは同じ結果を返す。
     */
    public synchronized LedgerOutcome calibrate() {
      if (outcome == null) {
        outcome = calibrated(fitted, pending, teacher);
        pending = null;
      }
      return outcome;
    }
  }

  private LedgerOutcome calibrated(
      LedgerOutcome fitted, Pending pending, LedgerCorrection teacher) {
    LedgerCalibration.Scaling scaling = (LedgerCalibration.Scaling) calibration;
    int cohort = pending.cohort();
    LedgerCalibrator scale = pending.scale();
    var pass = new LedgerCalibrator.Pass(pending.current(), scale, scaling, variant);
    for (FitRow row : pending.rows()) pass.add(row);
    LedgerCalibrator next =
        LedgerCalibrator.fit(
            pass.sums(), scaling, variant, cohort, catalog.columns(), apply, cells);
    double applied =
        switch (pending.applied()) {
          case RAW -> 0;
          case ROOT -> throw new IllegalStateException("Common scale cannot teach with a root model");
          case CLASS -> fitted.appliedR2();
          case COMMON_SCALE -> {
            double shadowR2 = pass.shadowR2();
            yield Double.isFinite(shadowR2) ? Math.max(0, shadowR2) : 0;
          }
        };
    LedgerReport.Calibration calibrated =
        pass.report(
            caseName(calibration), scaling.kind(), pending.applied(), pending.fallback(), next);
    LedgerReport.Arms j = calibrated.j();
    LOG.info(
        String.format(
            Locale.ROOT,
            "t=%d ledger calibration %s applied=%s source=%s rows=%d J raw=%s current=%s"
                + " shadow=%s slot r2 current=%s shadow=%s mean h=%s next=%s",
            cohort,
            calibrated.mode(),
            calibrated.applied(),
            scale == null ? "none" : String.valueOf(scale.cohort()),
            calibrated.rows(),
            j.raw(),
            j.current(),
            j.shadow(),
            calibrated.slotR2().current(),
            calibrated.slotR2().shadow(),
            calibrated.scale().mean(),
            next == null ? "none" : String.valueOf(next.cohort())));
    if (pending.fallback() != null) {
      LOG.info("t={} ledger calibration scales by 1: {}", cohort, pending.fallback());
    }
    return new LedgerOutcome(
        fitted.report(),
        fitted.summary().with(calibrated.summary()),
        teacher,
        applied,
        new LedgerOutcome.Calibration(calibrated, pending.current(), scale, next));
  }

  /**
   * 補正率を求めた結果に、和了の判断の教師の差を足す（{@link LedgerCalibrator} の和了の判断の差）。補正率を求めない結果は、そのまま返す。
   *
   * @param outcome {@link #learn} の結果
   * @param rows 学習の段が選んだ行のうち、主経路の FINAL_END の行
   * @param scores 行ごとのアンカーの s（{@code rows} と同じ順。アンカーが無ければ null で、ツモの行は数えない）
   * @return 和了の判断の教師の差を報告に入れた結果
   * @throws IllegalArgumentException 行と s の数が違うとき
   */
  public LedgerOutcome gates(LedgerOutcome outcome, List<Row> rows, List<float[]> scores) {
    LedgerOutcome.Calibration calibrated = outcome.calibration();
    if (calibrated == null) {
      return outcome;
    }
    if (scores != null && scores.size() != rows.size()) {
      throw new IllegalArgumentException(
          "Gate rows " + rows.size() + " have " + scores.size() + " anchor scores");
    }
    var gates = new LedgerCalibrator.Gates(calibrated.current(), calibrated.scale(), variant);
    for (int i = 0; i < rows.size(); i++)
      gates.add(rows.get(i), scores == null ? null : scores.get(i));
    LedgerReport.Calibration report = calibrated.report().withGates(gates.gates());
    return new LedgerOutcome(
        outcome.report(),
        outcome.summary(),
        outcome.correction(),
        outcome.appliedR2(),
        new LedgerOutcome.Calibration(
            report, calibrated.current(), calibrated.scale(), calibrated.next()));
  }

  /**
   * 持ち越した補正率を今のコホートに掛けられるかを確かめる。
   *
   * @param fitUnder 持ち越した補正率を当てはめた反復の帳簿の意味
   * @return 掛けられない理由（英語の 1 文）。掛けられるなら null
   */
  private String mismatch(LedgerCalibrator carried, LedgerMeaning fitUnder, int cohort) {
    LedgerCalibration.Scaling scaling = (LedgerCalibration.Scaling) calibration;
    int expected =
        switch (scaling.source()) {
          case PREVIOUS_COHORT -> cohort - 1;
        };
    if (carried == null) {
      return "no calibrator was carried from cohort " + expected;
    }
    if (carried.cohort() != expected) {
      return "the carried calibrator was fit on cohort " + carried.cohort() + ", not " + expected;
    }
    if (carried.kind() != scaling.kind()) {
      return "the carried calibrator is " + carried.kind() + ", not " + scaling.kind();
    }
    if (carried.variant() != variant) {
      return "the carried calibrator was fit for " + carried.variant() + ", not " + variant;
    }
    if (!carried.columns().equals(catalog.columns())) {
      return "the carried calibrator scales the ledger columns " + carried.columns();
    }
    if (!carried.apply().equals(apply)) {
      return "the carried calibrator scales the groups " + carried.apply() + ", not " + apply;
    }
    if (carried.cells() != cells) {
      return "the carried calibrator scales coefficients fit by " + carried.cells();
    }
    if (fitUnder == null) {
      return "the ledger settings that fit the carried calibrator are unknown";
    }
    String changed = meaning.scaleFree().change(fitUnder.scaleFree());
    if (changed != null) {
      return "the carried calibrator was fit on another ledger correction: " + changed;
    }
    return null;
  }

  /** 設定の選択肢を、設定に書く名前で返す。 */
  private static String caseName(Object choice) {
    return choice.getClass().getAnnotation(Case.class).value();
  }

  /** 持ち越すキーのうち、運の精算に関わる部分。 */
  public LedgerMeaning meaning() {
    return meaning;
  }
}
