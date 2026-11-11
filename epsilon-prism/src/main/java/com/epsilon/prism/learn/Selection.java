package com.epsilon.prism.learn;

import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import java.util.function.Predicate;

/**
 * 学習に使う行の選び方（行のインデックスに対する述語）。どれも TRAIN の行のうち、測った手が 2 つ以上ある行だけを選ぶ。
 * 差分整合の分母に入るのはこの行だけだからである。
 */
public enum Selection implements Predicate<RowCodec.Index> {
  /** ラベル行（MAIN・ROLLIN）。 */
  LABELS,
  /**
   * ラベル行と根の記録（ROOT）。系譜の生徒と refit に使う。根の記録は、実際に打った手と逸脱した手 d の 2 手の値を持つ。d が実際の手と
   * 同じ根は 1 手しかないので入らない。根の 2 手は同じ牌山・同じ乱数のキーでの局末の値の対なので、ラベル行と同じく Q の差を偏りなく表す。
   */
  LABELS_AND_ROOTS;

  @Override
  public boolean test(RowCodec.Index index) {
    Row.Source source = index.source();
    return index.purpose() == Row.Purpose.TRAIN
        && (source.label() || (this == LABELS_AND_ROOTS && source == Row.Source.ROOT))
        && index.measured() >= 2;
  }
}
