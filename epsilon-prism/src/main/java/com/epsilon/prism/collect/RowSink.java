package com.epsilon.prism.collect;

import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;

/**
 * 収集の行の受け口。行はドライバのスレッドから {@link CohortWriter} へ渡し、渡した行を数える。行は学習の教師そのものなので、記録のどこか
 * （捕捉・神託・到達・局末・行・書き手）の失敗はその場で投げて収集を止める。
 */
public final class RowSink {

  private final CohortWriter writer;
  private final LongAdder rows = new LongAdder();

  /**
   * 受け口を作る。
   *
   * @param writer コホートの行の書き手（封印と閉じるのは呼び出し側）
   */
  public RowSink(CohortWriter writer) {
    this.writer = Objects.requireNonNull(writer, "writer");
  }

  /**
   * 書き手に渡した行の数を返す。
   *
   * @return 行の数
   */
  public long rows() {
    return rows.sum();
  }

  /** 行を書き手に渡す。ドライバのスレッドで呼ぶ。 */
  void accept(Row row) {
    writer.accept(row);
    rows.increment();
  }

  /** 書き終えていない行の数。 */
  long backlog() {
    return writer.backlog();
  }
}
