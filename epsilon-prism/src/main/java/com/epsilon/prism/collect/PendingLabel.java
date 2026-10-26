package com.epsilon.prism.collect;

import java.util.Arrays;

/**
 * 局末の値を待つラベル行（MAIN・ROLLIN）。測る手 C ∪ {played}（{@link CandidateSet#measured}）の値がそろったら、Y を入れた行を受け口へ渡す。
 *
 * <p>枠: 打った手 played はその経路（主経路か写し）の局末の値、ほかの手は分岐の局末の値（写しの d が測る手に入っていれば、その写しの値を d の枠に
 * 使ってよい。写しも分岐も「o で d、以後は全席 b」の同じ過程）。値は判断の席の Φ。全ての枠はドライバのスレッドで受け取る。
 */
final class PendingLabel implements Pending {
  private final CollectContext context;
  private final RowDraft draft;

  /** 合法手ごとの値（測っていない手は NaN）。 */
  private final float[] values;

  private int remaining;

  /**
   * @param expected 待つ値の数（|C ∪ {played}|）
   */
  PendingLabel(CollectContext context, RowDraft draft, int expected) {
    this.context = context;
    this.draft = draft;
    values = new float[draft.actions()];
    Arrays.fill(values, Float.NaN);
    remaining = expected;
  }

  @Override
  public void accept(int action, double[] seatValues) {
    if (remaining <= 0 || !Float.isNaN(values[action]))
      throw new IllegalStateException(
          "Label slot delivered twice: game="
              + draft.gameId()
              + " decision="
              + draft.decisionId()
              + " action="
              + action);
    values[action] = (float) seatValues[draft.seat()];
    if (--remaining > 0) return;
    context.sink.accept(draft.label(values));
    context.metrics.rowsWritten.increment();
  }
}
