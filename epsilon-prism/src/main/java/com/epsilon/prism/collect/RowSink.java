package com.epsilon.prism.collect;

import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;

/**
 * 収集した行の書き出し先。行をドライバのスレッドから {@link CohortWriter} へ渡し、渡した行を数える。行は学習の教師そのものなので、
 * 記録のどこか（O の取り込み・Oracle・聴牌への到達・局末・行・ライター）で失敗したら、その場で投げて収集を止める。
 */
public final class RowSink {

  private final CohortWriter writer;
  private final LongAdder rows = new LongAdder();

  /**
   * 書き出し先を作る。
   *
   * @param writer コホートの行のライター（確定して閉じるのは呼び出し元）
   */
  public RowSink(CohortWriter writer) {
    this.writer = Objects.requireNonNull(writer, "writer");
  }

  /**
   * ライターに渡した行の数を返す。
   *
   * @return 行の数
   */
  public long rows() {
    return rows.sum();
  }

  /** 行をライターに渡す。ドライバのスレッドで呼ぶ。 */
  void accept(Row row) {
    writer.accept(row);
    rows.increment();
  }

  /** 書き終えていない行の数。 */
  long backlog() {
    return writer.backlog();
  }
}
