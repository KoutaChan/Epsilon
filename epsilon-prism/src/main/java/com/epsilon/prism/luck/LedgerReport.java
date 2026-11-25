package com.epsilon.prism.luck;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.noise.LedgerCalibrationKind;
import com.epsilon.prism.noise.LedgerCalibrationSource;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.rows.Row;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 運の精算の係数を当てはめた結果の報告。列ごとの偏りの確かめも含む。学習の段は、報告の全部を {@link Full} の形で書き、要約を段の報告
 * （{@link Summary}）とクラスごとの表（{@link Classes}）に書く。帳簿の補正率を求める反復では、その報告（{@link Calibration}）も
 * 両方に入れる。
 *
 * @param rows 当てはめた行数
 * @param slots 当てはめた行の、測った手の数
 * @param skippedRows 飛ばした行の数（帳簿のない行、測った手が 2 つ未満の行、測った手の帳簿が欠けた行）
 * @param all 全てのクラスをまとめた交差適合の結果
 * @param byClass クラスごとの結果（行のないクラスは載せない）
 * @param byCell 局・順位別の偶奇ごとの係数。両側ともクラス係数を使う区分は省く。クラス別の当てはめでは null
 * @param bias 偏りの確かめ
 * @param seconds 当てはめにかかった時間（秒）。{@link LedgerFit#fit} では行を読む時間を含む。{@link LedgerFit.WriterTap#fit} では
 *     貯めた行を足して解く時間
 */
public record LedgerReport(
    long rows,
    long slots,
    long skippedRows,
    Line all,
    Map<String, Line> byClass,
    Map<String, List<Fold>> byCell,
    Bias bias,
    double seconds) {

  /**
   * 1 つの区分で当てはめた結果。
   *
   * @param rows 行数
   * @param slots 測った手の数
   * @param variance 行の中で中心化したラベルの平方の、手 1 つあたりの平均
   * @param residual 補正した後の平方の、手 1 つあたりの平均（ホールドアウト）
   * @param r2 ホールドアウトの決定係数（1 − residual / variance）
   * @param r2ByFold 偶奇ごとの R²
   * @param equivalentForks 同等の分岐の本数（variance / residual）
   * @param r2ByGroup 組ごとに、その組の列だけで当てはめたときの R²（組の順）
   * @param beta 両方の偶奇の行で当てはめた係数
   * @param betaByFold 偶奇 g の行に掛ける係数（偶奇 1 − g の行で当てはめたもの）
   * @param fallback 偶奇 g ごとに、係数が代わりの値か。クラスごとの区分では、クラスをまとめた当てはめ（それも足りなければ 0）を使ったとき
   *     true。全体の区分では、行が足りずに 0 にしたとき true
   */
  public record Line(
      long rows,
      long slots,
      double variance,
      double residual,
      double r2,
      double[] r2ByFold,
      double equivalentForks,
      Map<LedgerGroup, Double> r2ByGroup,
      double[] beta,
      double[][] betaByFold,
      boolean[] fallback) {}

  /**
   * 1 つの列の偏りの確かめ。有限でない値は null にして書かない。
   *
   * @param name 列の名前
   * @param mean 手 1 つあたりの、列の生の値の平均
   * @param sd 手ごとの列の値の標準偏差
   * @param nonzero 列の値が 0 でない手の割合
   * @param z 半荘で束ねた平均の z（実装が正しければ標準正規分布に従う）
   */
  public record FeatureBias(
      String name,
      @Key(doc = "手 1 つあたりの、列の生の値の平均", presence = Presence.ABSENT) Double mean,
      @Key(doc = "手ごとの列の値の標準偏差", presence = Presence.ABSENT) Double sd,
      @Key(doc = "列の値が 0 でない手の割合", presence = Presence.ABSENT) Double nonzero,
      @Key(doc = "半荘で束ねた平均の z", presence = Presence.ABSENT) Double z) {}

  /**
   * 偏りの確かめ。
   *
   * @param all 全ての行で求めた、列ごとの結果（列の順。z は半荘で束ねたもの）
   */
  public record Bias(List<FeatureBias> all) {

    /** z の絶対値の最大を返す。z を持つ列が無ければ null。 */
    public Double maxAbsZ() {
      Double max = null;
      for (FeatureBias feature : all)
        if (feature.z() != null && (max == null || Math.abs(feature.z()) > max))
          max = Math.abs(feature.z());
      return max;
    }

    /** z の絶対値が {@code threshold} を越えた列の名前を、列の順に返す。 */
    public List<String> above(double threshold) {
      List<String> names = new ArrayList<>();
      for (FeatureBias feature : all)
        if (feature.z() != null && Math.abs(feature.z()) > threshold) names.add(feature.name());
      return List.copyOf(names);
    }
  }

  /** 報告の全部を、有限でない値を書かない書き出し用の形にして返す。 */
  public Full full(LedgerRecording recording) {
    return full(recording, null);
  }

  /**
   * 報告の全部と補正率の校正の報告を、有限でない値を書かない書き出し用の形にして返す。
   *
   * @param calibration 補正率の校正の報告（補正率を求めなければ null）
   */
  public Full full(LedgerRecording recording, Calibration calibration) {
    return full(recording, calibration, null, null);
  }

  Full full(
      LedgerRecording recording,
      Calibration calibration,
      RootAudit.Report roots,
      LedgerCalibrator appliedScale) {
    Map<String, Segment> classes = new LinkedHashMap<>();
    byClass.forEach((name, line) -> classes.put(name, Segment.of(line)));
    return new Full(
        rows,
        slots,
        skippedRows,
        Segment.of(all),
        classes,
        bias,
        seconds,
        recording,
        byCell,
        calibration,
        roots,
        appliedScale);
  }

  /**
   * 報告の全部を書き出す形（学習の段が出力する ledger.json）。有限でない値は書かない。
   *
   * @param rows 当てはめた行数
   * @param slots 当てはめた行の、測った手の数
   * @param skippedRows 飛ばした行の数（帳簿のない行、測った手が 2 つ未満の行、測った手の帳簿が欠けた行）
   * @param all 全てのクラスをまとめた結果
   * @param byClass クラスごとの結果
   * @param bias 偏りの確かめ
   * @param seconds 当てはめにかかった時間（秒）
   * @param byCell 局・順位の区分名ごとの偶奇の係数（クラス別の報告では省略）
   * @param recording 行の収集時に使った列の組と切り幅。記録がなければ null
   * @param calibration 帳簿の補正率の校正（補正率を求めない反復では null）
   */
  public record Full(
      long rows,
      long slots,
      long skippedRows,
      Segment all,
      Map<String, Segment> byClass,
      Bias bias,
      double seconds,
      @Key(doc = "係数を当てはめた行の帳簿の収集条件", presence = Presence.ABSENT) LedgerRecording recording,
      @Key(doc = "判断クラス・局・自席順位別の係数（両側ともクラス係数を使う区分は省略）", presence = Presence.ABSENT)
          Map<String, List<Fold>> byCell,
      @Key(doc = "帳簿の補正率の校正（補正率を求めない反復では無い）", presence = Presence.ABSENT) Calibration calibration,
      @Key(doc = "根の係数による教師の監査", presence = Presence.ABSENT) RootAudit.Report roots,
      @Key(doc = "教師に掛けた共通の補正率", presence = Presence.ABSENT) LedgerCalibrator appliedScale) {

    public Full(
        long rows,
        long slots,
        long skippedRows,
        Segment all,
        Map<String, Segment> byClass,
        Bias bias,
        double seconds,
        LedgerRecording recording) {
      this(
          rows, slots, skippedRows, all, byClass, bias, seconds, recording, null, null, null, null);
    }

    public Full(
        long rows,
        long slots,
        long skippedRows,
        Segment all,
        Map<String, Segment> byClass,
        Bias bias,
        double seconds,
        LedgerRecording recording,
        Map<String, List<Fold>> byCell) {
      this(
          rows,
          slots,
          skippedRows,
          all,
          byClass,
          bias,
          seconds,
          recording,
          byCell,
          null,
          null,
          null);
    }

    /** 凍結したアンカーで採点済みの和了判断から、報告に各補正の教師差を加える。 */
    public Full withWins(
        Variant variant, LedgerCalibrator previous, List<Row> decisions, List<float[]> scores)
        throws IOException {
      if (calibration == null && (roots == null || roots.model() == null)) return this;
      var baseline = LedgerCorrection.restoreClasses(this, recording, variant, previous);
      var measured = calibration;
      var candidate = roots;
      if (calibration != null) {
        LedgerCalibrator carried = calibration.origin() == null ? null : previous;
        if (calibration.origin() != null
            && (carried == null || !calibration.origin().equals(Origin.of(carried))))
          throw new IOException("Missing calibrator for saved win audit");
        measured =
            calibration.withGates(
                new LedgerCalibrator.Gates(baseline.scaled(null), baseline.scaled(carried), variant)
                    .measure(decisions, scores));
      }
      if (roots != null && roots.model() != null)
        candidate =
            roots.withGates(
                new LedgerCalibrator.Gates(
                        baseline.scaled(
                            calibration != null && calibration.origin() != null ? previous : null),
                        new LedgerCorrection.ByRoot(roots.model()),
                        variant)
                    .measure(decisions, scores));
      return new Full(
          rows,
          slots,
          skippedRows,
          all,
          byClass,
          bias,
          seconds,
          recording,
          byCell,
          measured,
          candidate,
          appliedScale);
    }

    public Full {
      byClass = Collections.unmodifiableMap(new LinkedHashMap<>(byClass));
      if (byCell != null) byCell = Collections.unmodifiableMap(new LinkedHashMap<>(byCell));
    }
  }

  /**
   * 1 つの区分の結果（{@link Line}）を書き出す形。係数は列の名前をキーにする。
   *
   * @param folds 偶奇ごとの結果（偶数、奇数の順）
   */
  public record Segment(
      long rows,
      long slots,
      @Key(doc = "行の中で中心化したラベルの平方の、手 1 つあたりの平均", presence = Presence.ABSENT) Double variance,
      @Key(doc = "補正した後の平方の、手 1 つあたりの平均（ホールドアウト）", presence = Presence.ABSENT) Double residual,
      @Key(doc = "ホールドアウトの決定係数", presence = Presence.ABSENT) Double r2,
      @Key(doc = "同等の分岐の本数", presence = Presence.ABSENT) Double equivalentForks,
      Map<LedgerGroup, Double> r2ByGroup,
      Map<String, Double> beta,
      List<Fold> folds) {

    public Segment {
      r2ByGroup = groups(r2ByGroup);
      beta = Collections.unmodifiableMap(new LinkedHashMap<>(beta));
      folds = List.copyOf(folds);
    }

    static Segment of(Line line) {
      List<Fold> folds = new ArrayList<>(2);
      for (int fold = 0; fold < 2; fold++)
        folds.add(
            new Fold(
                finite(line.r2ByFold()[fold]),
                columns(line.betaByFold()[fold]),
                line.fallback()[fold]));
      return new Segment(
          line.rows(),
          line.slots(),
          finite(line.variance()),
          finite(line.residual()),
          finite(line.r2()),
          finite(line.equivalentForks()),
          line.r2ByGroup(),
          columns(line.beta()),
          folds);
    }
  }

  /**
   * 偶奇の片方の結果を書き出す形。
   *
   * @param r2 その偶奇の行のホールドアウトの決定係数
   * @param beta その偶奇の行に掛ける係数（列の名前ごと）
   * @param fallback 係数が代わりの値か
   */
  public record Fold(
      @Key(doc = "その偶奇の行のホールドアウトの決定係数", presence = Presence.ABSENT) Double r2,
      Map<String, Double> beta,
      boolean fallback) {

    public Fold {
      beta = Collections.unmodifiableMap(new LinkedHashMap<>(beta));
    }
  }

  /**
   * 段の報告に書く要約。有限でない値は書かない。
   *
   * @param rows 当てはめた行の数
   * @param slots 当てはめた行の、測った手の数
   * @param skippedRows 飛ばした行の数（帳簿のない行など）
   * @param r2 全体のホールドアウトの決定係数
   * @param equivalentForks 同等の分岐の本数
   * @param r2ByGroup 組ごとの R²（組の順）
   * @param maxAbsZ 列の z の絶対値の最大
   * @param biased z の絶対値が偏りの警告のしきい値を越えた列の名前（列の順）
   * @param seconds 当てはめにかかった時間（秒）
   * @param streamed 収集のライターが作った和を使ったか（false なら行を読み直した）
   * @param calibration 帳簿の補正率の校正の要約（補正率を求めない反復では null）
   */
  public record Summary(
      long rows,
      long slots,
      long skippedRows,
      @Key(doc = "全体のホールドアウトの決定係数", presence = Presence.ABSENT) Double r2,
      @Key(doc = "同等の分岐の本数", presence = Presence.ABSENT) Double equivalentForks,
      Map<LedgerGroup, Double> r2ByGroup,
      @Key(doc = "列の z の絶対値の最大", presence = Presence.ABSENT) Double maxAbsZ,
      List<String> biased,
      double seconds,
      boolean streamed,
      @Key(doc = "帳簿の補正率の校正の要約（補正率を求めない反復では無い）", presence = Presence.ABSENT)
          CalibrationSummary calibration) {

    public Summary {
      r2ByGroup = groups(r2ByGroup);
      biased = List.copyOf(biased);
    }

    public Summary(
        long rows,
        long slots,
        long skippedRows,
        Double r2,
        Double equivalentForks,
        Map<LedgerGroup, Double> r2ByGroup,
        Double maxAbsZ,
        List<String> biased,
        double seconds,
        boolean streamed) {
      this(
          rows,
          slots,
          skippedRows,
          r2,
          equivalentForks,
          r2ByGroup,
          maxAbsZ,
          biased,
          seconds,
          streamed,
          null);
    }

    /** 補正率の校正の要約を付けた要約を返す。 */
    public Summary with(CalibrationSummary calibration) {
      return new Summary(
          rows,
          slots,
          skippedRows,
          r2,
          equivalentForks,
          r2ByGroup,
          maxAbsZ,
          biased,
          seconds,
          streamed,
          calibration);
    }

    /**
     * 報告から作る。
     *
     * @param biasZ 偏りの警告のしきい値（z の絶対値）
     * @param streamed 収集のライターが作った和を使ったか
     */
    public static Summary of(LedgerReport report, double biasZ, boolean streamed) {
      return new Summary(
          report.rows(),
          report.slots(),
          report.skippedRows(),
          finite(report.all().r2()),
          finite(report.all().equivalentForks()),
          report.all().r2ByGroup(),
          report.bias().maxAbsZ(),
          report.bias().above(biasZ),
          report.seconds(),
          streamed);
    }
  }

  /**
   * 1 つの判断のクラスの要約。
   *
   * @param rows 当てはめた行の数
   * @param r2 ホールドアウトの決定係数
   * @param equivalentForks 同等の分岐の本数
   */
  public record ClassSummary(
      long rows,
      @Key(doc = "ホールドアウトの決定係数", presence = Presence.ABSENT) Double r2,
      @Key(doc = "同等の分岐の本数", presence = Presence.ABSENT) Double equivalentForks) {}

  /**
   * クラスごとの要約（学習の段が出力する classes.json）。
   *
   * @param classes クラスの名前ごとの要約（行の無いクラスは載せない）
   */
  public record Classes(Map<String, ClassSummary> classes) {

    /** 帳簿の係数を当てはめなかった反復の要約。 */
    public static final Classes NONE = new Classes(Map.of());

    public Classes {
      classes = Collections.unmodifiableMap(new LinkedHashMap<>(classes));
    }

    /** 報告から作る。 */
    public static Classes of(LedgerReport report) {
      Map<String, ClassSummary> classes = new LinkedHashMap<>();
      report
          .byClass()
          .forEach(
              (name, line) ->
                  classes.put(
                      name,
                      new ClassSummary(
                          line.rows(), finite(line.r2()), finite(line.equivalentForks()))));
      return new Classes(classes);
    }
  }

  /**
   * 帳簿の補正率の校正の報告（ledger.json の calibration）。腕は 3 つで、生の Y（raw）、帳簿の補正 Y − β·L（current）、補正率を掛けた
   * 補正 Y − h(o)·β·L（shadow）。β はどの腕でも偶奇が逆の行で当てはめたもの、h は前のコホートの行で当てはめたもの
   * （{@link LedgerCalibrator}）。J は、行ごとに測った手の中で中心化した残差の二乗を測った手の数で割り、行を等しい重みで平均した値。
   * スロットの R² は、手ごとの残差の二乗の和で求めた 1 − 残差 / 生の値で、{@link Line#r2} と同じ数え方。有限でない値は書かない。
   *
   * @param mode 補正率のモード（設定に書く名前）
   * @param kind 補正率の形
   * @param applied 教師に実際に掛けた腕（{@link Arm#reportName}）。帳簿の補正が無ければ（帳簿を当てはめるだけのモードや、帳簿のある
   *     行が無いとき）raw
   * @param origin 今のコホートに掛けた補正率の出どころ（無ければ null で、補正率は 1）
   * @param fallback 補正率を 1 にした理由（出どころがあれば null）
   * @param rows 測った行の数
   * @param slots 測った手の数
   * @param j 腕ごとの J
   * @param slotR2 腕ごとのスロットの R²（生は 0）。教師に掛けた腕の値（負なら 0）が、次の K′ を決める R² になる
   * @param scale 行ごとの補正率の分布
   * @param bands 局面の帯ごとの J。全て（all）、南場の最後の局より前（early）と後（last）、後の局の順位 × 最も近い他家との点差の帯
   *     （rank{順位}_{下の境}_{上の境}）。行の無い帯も載せる
   * @param audits 補正の向きの監査（{@link Audit}）。事前に決めた群の順に、行の無い群も載せる（無ければ null）
   * @param gates 和了の判断の教師の差（{@link LuckSettlement#gates} が足す。足す前は空）
   * @param next 今のコホートの行で当てはめた、次の反復の補正率（当てはめられなければ null）
   */
  public record Calibration(
      String mode,
      LedgerCalibrationKind kind,
      String applied,
      @Key(doc = "今のコホートに掛けた補正率の出どころ", presence = Presence.ABSENT) Origin origin,
      @Key(doc = "補正率を 1 にした理由", presence = Presence.ABSENT) String fallback,
      long rows,
      long slots,
      Arms j,
      Arms slotR2,
      Scale scale,
      List<Band> bands,
      @Key(doc = "補正の向きの監査", presence = Presence.ABSENT) List<Audit> audits,
      List<Gate> gates,
      @Key(doc = "次の反復の補正率の当てはめ", presence = Presence.ABSENT) Next next) {

    public Calibration {
      Objects.requireNonNull(mode, "mode");
      Objects.requireNonNull(kind, "kind");
      Arm.of(applied);
      bands = List.copyOf(bands);
      audits = audits == null ? null : List.copyOf(audits);
      gates = List.copyOf(gates);
    }

    Calibration withApplied(Arm arm) {
      return new Calibration(
          mode,
          kind,
          arm.reportName(),
          origin,
          fallback,
          rows,
          slots,
          j,
          slotR2,
          scale,
          bands,
          audits,
          gates,
          next);
    }

    /** 和了の判断の教師の差を入れた報告を返す。 */
    public Calibration withGates(List<Gate> gates) {
      return new Calibration(
          mode, kind, applied, origin, fallback, rows, slots, j, slotR2, scale, bands, audits,
          gates, next);
    }

    /** 段の報告に書く要約。 */
    public CalibrationSummary summary() {
      return new CalibrationSummary(
          mode,
          applied,
          origin == null ? null : origin.cohort(),
          fallback,
          j,
          slotR2,
          scale.mean(),
          next == null ? null : next.cohort());
    }
  }

  /** 教師に掛けた補正の腕。報告には {@link #reportName} で書く。 */
  public enum Arm {
    /** 補正しない Y。 */
    RAW("raw"),
    /** 帳簿の補正 Y − β·L。 */
    CLASS("current"),
    /** 補正率を掛けた補正 Y − h(o)·β·L。 */
    COMMON_SCALE("shadow"),
    ROOT("rootCoefficients");

    private final String reportName;

    Arm(String reportName) {
      this.reportName = reportName;
    }

    /** 報告に書く名前。 */
    public String reportName() {
      return reportName;
    }

    /**
     * 報告に書いた名前の腕。
     *
     * @throws IllegalArgumentException 腕の名前でないとき
     */
    public static Arm of(String reportName) {
      for (Arm arm : values()) if (arm.reportName.equals(reportName)) return arm;
      throw new IllegalArgumentException("Unknown ledger arm: " + reportName);
    }
  }

  /**
   * 腕ごとの値。有限でない値は null。
   *
   * @param raw 生の Y
   * @param current 帳簿の補正 Y − β·L
   * @param shadow 補正率を掛けた補正 Y − h(o)·β·L
   */
  public record Arms(
      @Key(doc = "生の Y", presence = Presence.ABSENT) Double raw,
      @Key(doc = "帳簿の補正", presence = Presence.ABSENT) Double current,
      @Key(doc = "補正率を掛けた補正", presence = Presence.ABSENT) Double shadow) {

    static Arms of(double raw, double current, double shadow) {
      return new Arms(finite(raw), finite(current), finite(shadow));
    }
  }

  /**
   * 今のコホートに掛けた補正率の出どころ。
   *
   * @param source 当てはめる行を取ったコホートの選び方
   * @param cohort 補正率を当てはめた行のコホート
   * @param rows 当てはめた行の数
   * @param gapGrid 点差の軸の格子の点の数
   * @param roundGrid 局の番号の軸の格子の点の数
   * @param regularization 係数の二乗の罰の係数
   * @param shrink 補正率を 0 へ縮める罰の強さ
   */
  public record Origin(
      LedgerCalibrationSource source,
      int cohort,
      long rows,
      int gapGrid,
      int roundGrid,
      double regularization,
      double shrink) {

    /** 補正率の出どころ。 */
    public static Origin of(LedgerCalibrator calibrator) {
      return new Origin(
          calibrator.source(),
          calibrator.cohort(),
          calibrator.rows(),
          calibrator.gapGrid(),
          calibrator.roundGrid(),
          calibrator.regularization(),
          calibrator.shrink());
    }
  }

  /**
   * 行ごとの補正率の分布。
   *
   * @param mean 平均
   * @param min 最小
   * @param max 最大
   * @param histogram [0, 1] を等分した柱ごとの行の数（1 は最後の柱に入れる）
   */
  public record Scale(
      @Key(doc = "平均", presence = Presence.ABSENT) Double mean,
      @Key(doc = "最小", presence = Presence.ABSENT) Double min,
      @Key(doc = "最大", presence = Presence.ABSENT) Double max,
      List<Long> histogram) {

    public Scale {
      histogram = List.copyOf(histogram);
    }
  }

  /**
   * 局面の帯 1 つの J。差の標準誤差は半荘で束ねて求める（{@link LedgerCalibrator} の帯の和）。
   *
   * @param name 帯の名前
   * @param rows 行の数
   * @param games 半荘の数
   * @param j 腕ごとの J
   * @param currentMinusRaw J_今 − J_生
   * @param seCurrentMinusRaw その標準誤差
   * @param shadowMinusRaw J_影 − J_生
   * @param seShadowMinusRaw その標準誤差
   * @param shadowVsCurrent 帳簿の補正に対する J の減り 1 − J_影 / J_今
   * @param seShadowVsCurrent その標準誤差（比を線形化したもの）
   */
  public record Band(
      String name,
      long rows,
      long games,
      Arms j,
      @Key(doc = "J_今 − J_生", presence = Presence.ABSENT) Double currentMinusRaw,
      @Key(doc = "J_今 − J_生 の標準誤差", presence = Presence.ABSENT) Double seCurrentMinusRaw,
      @Key(doc = "J_影 − J_生", presence = Presence.ABSENT) Double shadowMinusRaw,
      @Key(doc = "J_影 − J_生 の標準誤差", presence = Presence.ABSENT) Double seShadowMinusRaw,
      @Key(doc = "1 − J_影 / J_今", presence = Presence.ABSENT) Double shadowVsCurrent,
      @Key(doc = "1 − J_影 / J_今 の標準誤差", presence = Presence.ABSENT) Double seShadowVsCurrent,
      @Key(doc = "独立したコホートを合わせるための、半荘単位の対応差の集計", presence = Presence.ABSENT) Paired paired,
      @Key(doc = "根の補正の J から比較の腕の J を引いた対応差", presence = Presence.ABSENT) Double shadowMinusCurrent,
      @Key(doc = "比較の腕との対応差の半荘単位の標準誤差", presence = Presence.ABSENT) Double seShadowMinusCurrent) {}

  /**
   * 生からの二乗誤差の差を、半荘を単位にして合算するための値。半荘ごとの行数を n、差の和から n × 全行の平均差を引いた値を r とする。
   *
   * @param rows 行数
   * @param games 半荘数
   * @param mean 1 行あたりの平均差
   * @param countSquares n の二乗の和
   * @param countResidual n と r の積の和
   * @param residualSquares r の二乗の和
   */
  public record Paired(
      long rows,
      long games,
      double mean,
      double countSquares,
      double countResidual,
      double residualSquares) {
    /** 異なる半荘からなる 2 つの標本を、合算後の平均差のまわりに中心化する。 */
    public Paired plus(Paired other) {
      long total = Math.addExact(rows, other.rows);
      double pooled = mean + (other.mean - mean) * ((double) other.rows / total);
      double shift = mean - pooled, otherShift = other.mean - pooled;
      return new Paired(
          total,
          Math.addExact(games, other.games),
          pooled,
          countSquares + other.countSquares,
          countResidual
              + shift * countSquares
              + other.countResidual
              + otherShift * other.countSquares,
          squaresAt(shift) + other.squaresAt(otherShift));
    }

    private double squaresAt(double shift) {
      return residualSquares + shift * (2 * countResidual + shift * countSquares);
    }

    /** 対応差の平均の標準誤差。半荘が 1 件のときは推定できないため null を返す。 */
    public Double se() {
      return games < 2
          ? null
          : Math.sqrt((double) games / (games - 1) * Math.max(0, residualSquares)) / rows;
    }
  }

  /**
   * 補正の向きの監査の、事前に決めた群 1 つ（{@link LedgerCalibrator} の監査の群）。補正は行の全ての手に同じ量を足しても J を変えないので、
   * 中心化しない補正の平均と、打った手の補正の偏りを別に数える。帳簿の条件付き平均が 0 なら、どちらの期待値も 0。
   *
   * @param group 群の名前
   * @param rows 行の数
   * @param games 半荘の数
   * @param played 打った手を測った行の数
   * @param rowMean 行の測った手の補正の平均（中心化しない）の、行の平均
   * @param playedMinusOthers 打った手の補正 − ほかの測った手の補正の平均の、行の平均（打った手を測った行だけ）
   */
  public record Audit(
      String group, long rows, long games, long played, Signed rowMean, Signed playedMinusOthers) {}

  /**
   * 補正の符号付きの平均と、その半荘で束ねた標準誤差。今は帳簿の補正 β·L、影は補正率を掛けた補正 h(o)·β·L。有限でない値は null。
   *
   * @param current 今の平均
   * @param seCurrent その標準誤差
   * @param shadow 影の平均
   * @param seShadow その標準誤差
   */
  public record Signed(
      @Key(doc = "今の平均", presence = Presence.ABSENT) Double current,
      @Key(doc = "今の平均の標準誤差", presence = Presence.ABSENT) Double seCurrent,
      @Key(doc = "影の平均", presence = Presence.ABSENT) Double shadow,
      @Key(doc = "影の平均の標準誤差", presence = Presence.ABSENT) Double seShadow) {}

  /**
   * 和了の判断の教師の差の、1 つの帯と和了の種類（{@link LedgerCalibrator} の和了の判断の差）。
   *
   * @param group 帯（FINAL_END か first_{下の境}_{上の境}）
   * @param kind 和了の種類（tsumo か ron）
   * @param rows 行の数
   * @param games 半荘の数
   * @param mean 腕ごとの差の平均
   * @param sd 腕ごとの差の標準偏差（n − 1 で割る）
   * @param correction 和了と比べる手の補正の差（β·L_和了 − β·L_比べる手。影はそれに h(o) を掛けた値）の符号付きの平均（無ければ
   *     null）
   * @param currentMinusRawSd SD_今 − SD_生
   * @param seCurrentMinusRawSd その標準誤差
   * @param shadowMinusRawSd SD_影 − SD_生
   * @param seShadowMinusRawSd その標準誤差
   */
  public record Gate(
      String group,
      String kind,
      long rows,
      long games,
      Arms mean,
      Arms sd,
      @Key(doc = "和了と比べる手の補正の差の平均", presence = Presence.ABSENT) Signed correction,
      @Key(doc = "SD_今 − SD_生", presence = Presence.ABSENT) Double currentMinusRawSd,
      @Key(doc = "SD_今 − SD_生 の標準誤差", presence = Presence.ABSENT) Double seCurrentMinusRawSd,
      @Key(doc = "SD_影 − SD_生", presence = Presence.ABSENT) Double shadowMinusRawSd,
      @Key(doc = "SD_影 − SD_生 の標準誤差", presence = Presence.ABSENT) Double seShadowMinusRawSd) {}

  /**
   * 今のコホートの行で当てはめた、次の反復の補正率の要約。J は当てはめた行での値（ホールドアウトではない）。
   *
   * @param cohort 当てはめた行のコホート
   * @param rows 当てはめた行の数
   * @param j 生・帳簿の補正・当てはめた補正率での J
   * @param meanCoefficient 係数の平均
   * @param zeros 0 の係数の数
   * @param ones 1 の係数の数
   * @param coefficients 係数の数
   * @param calibrator 当てはめた補正率（係数まで。次のコホートに掛ける。無ければ null）
   */
  public record Next(
      int cohort,
      long rows,
      Arms j,
      @Key(doc = "係数の平均", presence = Presence.ABSENT) Double meanCoefficient,
      int zeros,
      int ones,
      int coefficients,
      @Key(doc = "当てはめた補正率", presence = Presence.ABSENT) LedgerCalibrator calibrator) {

    static Next of(LedgerCalibrator next, LedgerCalibrator.Sums sums) {
      double[] theta = next.theta();
      double[] all = new double[theta.length];
      Arrays.fill(all, 1);
      int zeros = 0, ones = 0;
      double total = 0;
      for (double value : theta) {
        total += value;
        if (value == 0) zeros++;
        if (value == 1) ones++;
      }
      double n = sums.rows;
      return new Next(
          next.cohort(),
          next.rows(),
          Arms.of(sums.c / n, sums.residual(all) / n, sums.residual(theta) / n),
          finite(total / theta.length),
          zeros,
          ones,
          theta.length,
          next);
    }
  }

  /**
   * 段の報告に書く、補正率の校正の要約。
   *
   * @param mode 補正率のモード（設定に書く名前）
   * @param applied 教師に実際に掛けた腕（{@link Arm#reportName}）
   * @param origin 今のコホートに掛けた補正率を当てはめたコホート（無ければ null）
   * @param fallback 補正率を 1 にした理由（出どころがあれば null）
   * @param j 腕ごとの J
   * @param slotR2 腕ごとのスロットの R²
   * @param meanScale 補正率の平均
   * @param next 次の反復の補正率を当てはめたコホート（当てはめられなければ null）
   */
  public record CalibrationSummary(
      String mode,
      String applied,
      @Key(doc = "今のコホートに掛けた補正率を当てはめたコホート", presence = Presence.ABSENT) Integer origin,
      @Key(doc = "補正率を 1 にした理由", presence = Presence.ABSENT) String fallback,
      Arms j,
      Arms slotR2,
      @Key(doc = "補正率の平均", presence = Presence.ABSENT) Double meanScale,
      @Key(doc = "次の反復の補正率を当てはめたコホート", presence = Presence.ABSENT) Integer next) {}

  /** 有限なら値を、そうでなければ null を返す。 */
  static Double finite(double value) {
    return Double.isFinite(value) ? value : null;
  }

  /** 組ごとの値のうち有限なものだけを、組の順に並べた変更できない表にして返す。 */
  private static Map<LedgerGroup, Double> groups(Map<LedgerGroup, Double> values) {
    Map<LedgerGroup, Double> finite = new EnumMap<>(LedgerGroup.class);
    values.forEach(
        (group, value) -> {
          if (value != null && Double.isFinite(value)) finite.put(group, value);
        });
    return Collections.unmodifiableMap(finite);
  }

  /** 列ごとの係数のうち有限なものだけを、列の名前をキーにして列の順に並べた表にして返す。 */
  static Map<String, Double> columns(double[] beta) {
    Map<String, Double> named = new LinkedHashMap<>();
    LedgerColumn[] columns = LedgerColumn.values();
    for (int column = 0; column < beta.length; column++)
      if (Double.isFinite(beta[column])) named.put(columns[column].reportName(), beta[column]);
    return named;
  }
}
