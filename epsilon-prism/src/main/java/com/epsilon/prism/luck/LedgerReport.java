package com.epsilon.prism.luck;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.noise.LedgerGroup;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 運の精算の係数を当てはめた結果の報告。列ごとの偏りの確かめも含む。学習の段は、報告の全部を {@link Full} の形で書き、要約を段の報告
 * （{@link Summary}）とクラスごとの表（{@link Classes}）に書く。
 *
 * @param rows 当てはめた行数
 * @param slots 当てはめた行の、測った手の数
 * @param skippedRows 飛ばした行の数（帳簿のない行、測った手が 2 つ未満の行、測った手の帳簿が欠けた行）
 * @param all 全てのクラスをまとめた交差適合の結果
 * @param byClass クラスごとの結果（行のないクラスは載せない）
 * @param bias 偏りの確かめ
 * @param seconds 当てはめにかかった時間（秒）。{@link LedgerFit#fit} では行を読む時間を含む。{@link LedgerFit.WriterTap#fit} では
 *     貯めた行を足して解く時間
 */
public record LedgerReport(
    long rows, long slots, long skippedRows, Line all, Map<String, Line> byClass, Bias bias, double seconds) {

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
        if (feature.z() != null && (max == null || Math.abs(feature.z()) > max)) max = Math.abs(feature.z());
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
  public Full full() {
    Map<String, Segment> classes = new LinkedHashMap<>();
    byClass.forEach((name, line) -> classes.put(name, Segment.of(line)));
    return new Full(rows, slots, skippedRows, Segment.of(all), classes, bias, seconds);
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
   */
  public record Full(
      long rows, long slots, long skippedRows, Segment all, Map<String, Segment> byClass, Bias bias, double seconds) {

    public Full {
      byClass = Collections.unmodifiableMap(new LinkedHashMap<>(byClass));
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
        folds.add(new Fold(finite(line.r2ByFold()[fold]), columns(line.betaByFold()[fold]), line.fallback()[fold]));
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
      boolean streamed) {

    public Summary {
      r2ByGroup = groups(r2ByGroup);
      biased = List.copyOf(biased);
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
      report.byClass()
          .forEach(
              (name, line) ->
                  classes.put(name, new ClassSummary(line.rows(), finite(line.r2()), finite(line.equivalentForks()))));
      return new Classes(classes);
    }
  }

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
  private static Map<String, Double> columns(double[] beta) {
    Map<String, Double> named = new LinkedHashMap<>();
    LedgerColumn[] columns = LedgerColumn.values();
    for (int column = 0; column < beta.length; column++)
      if (Double.isFinite(beta[column])) named.put(columns[column].reportName(), beta[column]);
    return named;
  }
}
