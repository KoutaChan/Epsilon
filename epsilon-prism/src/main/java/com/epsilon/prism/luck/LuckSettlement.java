package com.epsilon.prism.luck;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.noise.LedgerSettings;
import com.epsilon.prism.noise.NoiseSettings;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import java.util.Locale;
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

  private LuckSettlement(
      LedgerCatalog catalog,
      LedgerRecording recording,
      LedgerFit fit,
      double biasZ,
      boolean corrects,
      UraSettings ura,
      LedgerMeaning meaning) {
    this.catalog = catalog;
    this.recording = recording;
    this.fit = fit;
    this.biasZ = biasZ;
    this.corrects = corrects;
    this.ura = ura;
    this.meaning = meaning;
  }

  /** 反復の設定のうちノイズの項目（帳簿・当てはめの重み・裏ドラ）と、ルールから作る。 */
  public static LuckSettlement of(NoiseSettings noise, Variant variant) {
    LedgerSettings ledger = noise.ledger();
    LedgerCatalog catalog = LedgerCatalog.of(variant);
    LedgerSettings.Fitting fitting = ledger instanceof LedgerSettings.Fitting f ? f : null;
    return new LuckSettlement(
        catalog,
        LedgerRecording.of(ledger),
        fitting == null ? null : new LedgerFit(fitting.apply(), fitting.ridge(), fitting.minRowsPerColumn()),
        fitting == null ? Double.NaN : fitting.biasZ(),
        ledger.corrects(),
        noise.ura(),
        LedgerMeaning.of(ledger, catalog, noise.lossWeight(), noise.ura()));
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
   * コホートのライターから行を受け取り、当てはめに使う和を収集の間に作る受け取り先を返す。
   *
   * @param writers コホートのライターの数
   * @param rows 和に入れる行（学習の段が選ぶ行と同じ選び方）
   * @return 和を作る受け取り先。当てはめない段では null
   */
  public LedgerFit.WriterTap tap(int writers, Predicate<RowCodec.Index> rows) {
    return fit == null ? null : fit.tap(writers, rows);
  }

  /**
   * 学習の段で帳簿を扱う。当てはめる段では係数を当てはめて報告し、偏りの z の絶対値がしきい値を越えた列があれば警告を 1 行出す。補正する
   * 段では、教師の補正と、掛けた補正の R² も返す。
   *
   * @param cohort コホートの番号（ログとエラーメッセージに書く）
   * @param streamed 同じ JVM の収集が作った和（無ければ null。あれば行を読まない）
   * @param stored 学習の段が選んだ行（和が無いときだけ読む）
   * @param selected 学習の段が選んだ行の数
   * @return 帳簿の結果。当てはめない段では {@link LedgerOutcome#NONE}
   * @throws IllegalStateException 和に足した行の数が、選んだ行の数と違うとき
   */
  public LedgerOutcome learn(int cohort, LedgerFit.WriterTap streamed, Iterable<Row> stored, int selected) {
    if (fit == null) {
      return LedgerOutcome.NONE;
    }
    LedgerFit.Result result;
    if (streamed != null) {
      result = streamed.fit();
      long seen = result.report().rows() + result.report().skippedRows();
      if (seen != selected) {
        throw new IllegalStateException(
            "Ledger of cohort " + cohort + " summed " + seen + " rows but " + selected + " are selected");
      }
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
    return new LedgerOutcome(report, summary, correction, applied);
  }

  /** 持ち越すキーのうち、運の精算に関わる部分。 */
  public LedgerMeaning meaning() {
    return meaning;
  }
}
