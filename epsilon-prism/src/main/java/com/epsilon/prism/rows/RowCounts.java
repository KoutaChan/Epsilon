package com.epsilon.prism.rows;

import com.epsilon.prism.core.DecisionClass;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * コホートの行の数え（反復の記録の labels・copyLabels・roots。行の索引 {@link RowCodec.Index} から数える）: 行の種類ごとに、類の名前 → 行の数
 * （0 でない類だけを類の順に）。
 *
 * @param labels 主経路のラベル行（MAIN）
 * @param copyLabels 写しの中のラベル行（ROLLIN）
 * @param roots 根の記録（ROOT）
 * @param holdout 保留（HOLDOUT）の半荘の行の数
 */
public record RowCounts(
    Map<String, Long> labels, Map<String, Long> copyLabels, Map<String, Long> roots, long holdout) {

  /**
   * 行を数える。
   *
   * @param rows 行
   * @param index 行の索引を引く口
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

  /** 反復の記録に labels・copyLabels・roots を書く。 */
  public void report(Map<String, Object> report) {
    report.put("labels", labels);
    report.put("copyLabels", copyLabels);
    report.put("roots", roots);
  }

  /** 類ごとの数を、0 でない類だけ類の順に並べた表にする。 */
  private static Map<String, Long> byClass(long[] counts) {
    Map<String, Long> result = new LinkedHashMap<>();
    for (DecisionClass c : DecisionClass.values())
      if (counts[c.ordinal()] > 0) result.put(c.name(), counts[c.ordinal()]);
    return result;
  }
}
