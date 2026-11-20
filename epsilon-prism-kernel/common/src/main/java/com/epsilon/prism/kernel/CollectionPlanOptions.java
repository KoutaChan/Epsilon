package com.epsilon.prism.kernel;

import java.util.List;

/** 収集の計画を作るときに決める、層 0 の因果マスクの有無と、pack の中の ΣC・区分の重みの位置。 */
public record CollectionPlanOptions(boolean causal, Count count, Segments segments) {
  public static final CollectionPlanOptions BASE = new CollectionPlanOptions(false, null, null);

  public record Count(long offset, int firstCode, int width) {}

  public record Segments(long offset, SegmentLayout layout) {}

  /** 入力の 1 列と、区分の表の中でその列が読む行の範囲。 */
  public record SegmentSpan(int column, int firstRow, int rows) {}

  /** 推論の計画と学習の起動に渡す、区分の表の記述。 */
  public record SegmentLayout(int codeWidth, int tableRows, List<SegmentSpan> spans) {
    // native の記述が持てる範囲の数の上限。
    private static final int MAX_SPANS = CollectionCapabilities.limit("MaxSegmentSpans");
    static final int INTS = 3 + MAX_SPANS * 3;

    public SegmentLayout {
      spans = List.copyOf(spans);
      if (spans.isEmpty() || spans.size() > MAX_SPANS) {
        throw new IllegalArgumentException("invalid number of candidate table spans");
      }
    }

    void encode(int[] target, int at) {
      target[at++] = codeWidth;
      target[at++] = tableRows;
      target[at++] = spans.size();
      for (SegmentSpan span : spans) {
        target[at++] = span.column();
        target[at++] = span.firstRow();
        target[at++] = span.rows();
      }
    }
  }
}
