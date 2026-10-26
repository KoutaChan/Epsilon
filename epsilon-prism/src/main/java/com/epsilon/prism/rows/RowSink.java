package com.epsilon.prism.rows;

import java.util.concurrent.atomic.LongAdder;

/**
 * 行の受け口。どのスレッドからでも呼べ、行を捨てない。背圧は {@link #backlog()} を見て、ドライバが新しい半荘の開始を止めることで掛ける。
 */
public interface RowSink extends AutoCloseable {
  void accept(PrismRow row);

  /** まだ書き終えていない行数。 */
  long backlog();

  /** 書き終えた行数と圧縮後のバイト数。 */
  RowStats stats();

  /**
   * 全行を受け取り終えたことを確定する（保存する受け口なら、残りを書き終えて同期し、封印する）。収集が最後まで終わったときだけ呼ぶ。
   */
  default void seal() throws java.io.IOException {}

  /** 書き手を止める。{@link #seal} の前に閉じたら（収集の失敗）、保存する受け口は封印しない。 */
  @Override
  void close() throws java.io.IOException;

  record RowStats(long rows, long bytes) {}

  /** 行を捨てて数だけ数える。速度の計測で保存を外すときに使う。 */
  static RowSink counting() {
    return new RowSink() {
      private final LongAdder rows = new LongAdder();

      @Override
      public void accept(PrismRow row) {
        rows.increment();
      }

      @Override
      public long backlog() {
        return 0;
      }

      @Override
      public RowStats stats() {
        return new RowStats(rows.sum(), 0);
      }

      @Override
      public void close() {}
    };
  }
}
