package com.epsilon.prism.run;

import com.epsilon.grp.Phi;
import com.epsilon.prism.collect.CollectionMetrics;
import com.epsilon.prism.collect.Collector;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.luck.UraAudit;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

/**
 * 収集の段の報告。行の数とバイト数は、確定したマニフェストの値。
 *
 * @param start 全ての席が打ったバージョン
 * @param games 主経路の半荘の数
 * @param hands 主経路の局の数
 * @param modelDecisions ネットワークで決めた判断の数（主経路・写し・分岐の全て）
 * @param forcedDecisions 合法手が 1 つの判断の数
 * @param decisionsPerSecond 1 秒あたりにネットワークで決めた判断の数
 * @param rows 確定した行の数
 * @param bytes 確定した行のバイト数
 * @param labels 主経路のラベル行の数
 * @param copyLabels 写しの中のラベル行の数
 * @param roots 根の記録の数
 * @param copies 作った写しの数
 * @param forks 候補の値を測るために作った分岐の数
 * @param explored 候補集合に一様に選んだ 1 手を足した判断の数
 * @param playedOutside 打った手が候補集合の外だったラベル行の数
 * @param mainNonGreedy 主経路で貪欲でない手を打った判断の数
 * @param breakdown 判断する席の手の形と、写しの根のクラスごとの内訳
 * @param ura 裏ドラの期待値を監査した集計（値を受け取り先に渡した局末を数える）
 * @param phi 収集の間に Φ を呼んだ回数の集計（メモに当たった数と求めた時間、終盤の数え上げの分）
 */
public record CollectReport(
    int start,
    long games,
    long hands,
    long modelDecisions,
    long forcedDecisions,
    double decisionsPerSecond,
    long rows,
    long bytes,
    long labels,
    long copyLabels,
    long roots,
    long copies,
    long forks,
    long explored,
    long playedOutside,
    long mainNonGreedy,
    CollectionMetrics.Breakdown breakdown,
    UraAudit.Summary ura,
    Phi.Stats phi) {

  /** 確定したコホートに置く、クラスごとの数のファイル。 */
  public static final String CLASSES = "classes.json";

  /**
   * 収集の結果から作る。
   *
   * @param start 打ったバージョン
   * @param result 収集の結果
   * @param rows 確定した行の数
   * @param bytes 確定した行のバイト数
   */
  public static CollectReport of(int start, Collector.Result result, long rows, long bytes) {
    CollectionMetrics.Snapshot counts = result.counts();
    long labels = 0;
    long copyLabels = 0;
    for (int c = 0; c < counts.labelRows().length; c++) {
      labels += counts.labelRows()[c];
      copyLabels += counts.copyLabelRows()[c];
    }
    double rate = result.seconds() > 0 ? counts.modelDecisions() / result.seconds() : 0;
    return new CollectReport(
        start,
        counts.games(),
        counts.hands(),
        counts.modelDecisions(),
        counts.forcedDecisions(),
        rate,
        rows,
        bytes,
        labels,
        copyLabels,
        counts.rootCount(),
        counts.copies(),
        counts.forks(),
        counts.explored(),
        counts.playedOutside(),
        counts.mainNonGreedy(),
        counts.breakdown(),
        result.ura().summary(),
        result.phi());
  }

  /**
   * 判断のクラス 1 つ分の収集の数。
   *
   * @param mainDecisions 主経路でネットワークが決めた判断の数
   * @param labels 主経路のラベル行の数
   * @param copyLabels 写しの中のラベル行の数
   * @param roots 根の記録の数
   */
  public record ClassCounts(long mainDecisions, long labels, long copyLabels, long roots) {}

  /**
   * クラスごとの収集の数。確定したコホートの {@value #CLASSES} に書く。
   *
   * @param classes クラスごとの数（数が全て 0 のクラスは載せない）
   */
  public record Classes(Map<DecisionClass, ClassCounts> classes) {

    public Classes {
      Map<DecisionClass, ClassCounts> ordered = new EnumMap<>(DecisionClass.class);
      ordered.putAll(classes);
      classes = Collections.unmodifiableMap(ordered);
    }

    /** 収集の数から作る。 */
    public static Classes of(CollectionMetrics.Snapshot counts) {
      Map<DecisionClass, ClassCounts> classes = new EnumMap<>(DecisionClass.class);
      for (DecisionClass c : DecisionClass.values()) {
        int i = c.ordinal();
        ClassCounts row =
            new ClassCounts(counts.mainDecisions()[i], counts.labelRows()[i], counts.copyLabelRows()[i], counts.roots()[i]);
        if (row.mainDecisions() + row.labels() + row.copyLabels() + row.roots() > 0) {
          classes.put(c, row);
        }
      }
      return new Classes(classes);
    }
  }
}
