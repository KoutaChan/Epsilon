package com.epsilon.prism.learn;

import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import java.util.function.Predicate;

/**
 * 学習に使う行の選び（索引の述語）。どれも TRAIN の行で、測った手が 2 つ以上（差分整合の分母に入る行）。
 */
public enum Selection implements Predicate<RowCodec.Index> {
  /** ラベル行（MAIN・ROLLIN）。 */
  LABELS,
  /**
   * ラベル行と根の記録（ROOT。played と逸脱 d の 2 手の値。d = played の根は 1 手なので入らない）。系譜の生徒と refit の行。根の 2 手は同じ山・
   * 同じ鍵の局末の値の対で、ラベル行と同じ不偏な Q の差を持つ。
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
