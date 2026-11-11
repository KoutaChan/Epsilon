package com.epsilon.prism.rows;

import com.epsilon.prism.core.DecisionClass;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * コホートの行を数えた結果。行の種類ごとに、判断のクラスの名前から行の数を引く表を持つ。表には 0 でないクラスだけを、クラスの順に並べる。
 * 行のインデックス（{@link RowCodec.Index}）から数え、反復の記録の {@code labels}・{@code copyLabels}・{@code roots} に書く。
 *
 * @param labels クラスごとの、主経路のラベル行（MAIN）の数
 * @param copyLabels クラスごとの、写しの中のラベル行（ROLLIN）の数
 * @param roots クラスごとの、根の記録（ROOT）の数
 * @param holdout ホールドアウト（HOLDOUT）の半荘の行の数
 */
public record RowCounts(
    Map<String, Long> labels, Map<String, Long> copyLabels, Map<String, Long> roots, long holdout) {

  /**
   * 行を数える。
   *
   * @param rows 数える行の番号
   * @param index 行の番号からインデックスを引く関数
   * @return 数えた結果
   */
  public static RowCounts of(int[] rows, IntFunction<RowCodec.Index> index) {
    long[][] counts = new long[Row.Source.values().length][DecisionClass.COUNT];
    long holdout = 0;
    for (int row : rows) {
      RowCodec.Index entry = index.apply(row);
      counts[entry.source().ordinal()][entry.decisionClass().ordinal()]++;
      if (entry.purpose() == Row.Purpose.HOLDOUT) holdout++;
    }
    return new RowCounts(
        byClass(counts[Row.Source.MAIN.ordinal()]),
        byClass(counts[Row.Source.ROLLIN.ordinal()]),
        byClass(counts[Row.Source.ROOT.ordinal()]),
        holdout);
  }

  /** 反復の記録に、種類ごとの表を {@code labels}・{@code copyLabels}・{@code roots} のキーで書く。ホールドアウトの数は書かない。 */
  public void report(Map<String, Object> report) {
    report.put("labels", labels);
    report.put("copyLabels", copyLabels);
    report.put("roots", roots);
  }

  /** クラスごとの数を、0 でないクラスだけクラスの順に並べた表にする。 */
  private static Map<String, Long> byClass(long[] counts) {
    Map<String, Long> result = new LinkedHashMap<>();
    for (DecisionClass c : DecisionClass.values())
      if (counts[c.ordinal()] > 0) result.put(c.name(), counts[c.ordinal()]);
    return result;
  }
}
