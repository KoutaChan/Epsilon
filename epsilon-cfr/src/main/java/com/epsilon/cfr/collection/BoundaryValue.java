package com.epsilon.cfr.collection;

import java.util.List;

/** 凍結した境界評価器。公開情報であるGRP境界系列から、席番号順の終局期待効用を求める。 */
@FunctionalInterface
public interface BoundaryValue {
  double[] evaluate(float[] publicBoundaryPrefix);

  /** 入力順を保って一括評価する。軽量な境界関数は、1件ずつ評価する実装だけでも利用できる。 */
  default List<double[]> evaluateBatch(List<float[]> publicBoundaryPrefixes) {
    return publicBoundaryPrefixes.stream().map(this::evaluate).toList();
  }
}
