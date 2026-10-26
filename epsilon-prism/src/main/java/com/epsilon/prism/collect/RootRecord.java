package com.epsilon.prism.collect;

import java.util.Arrays;

/**
 * 写しの根の記録（ROOT、論文 3.5 と 命題 10）。主経路の判断 o で、打った手 played の枠（主経路の局末の値 y_main）と逸脱 d の枠（写しの局末の値
 * y_br）を待つ。d = played なら写しを作らず played の 1 枠だけ（y_br = y_main）。
 *
 * <p>そろったら {@link RowDraft#root} で行にして受け口へ渡す: values は played と d の枠だけ、deviation = d、deviationProbability = q(d|o)、
 * propensity = β_c(o)·q(d|o)。全ての枠はドライバのスレッドで受け取る。
 */
final class RootRecord implements Pending {
  private final CollectContext context;
  private final RowDraft draft;
  private final RollIn.Deviation deviation;

  /** β_c(o)。 */
  private final double rollinRate;

  private final float[] values;
  private int remaining;

  RootRecord(CollectContext context, RowDraft draft, RollIn.Deviation deviation, double rollinRate) {
    this.context = context;
    this.draft = draft;
    this.deviation = deviation;
    this.rollinRate = rollinRate;
    values = new float[draft.actions()];
    Arrays.fill(values, Float.NaN);
    remaining = deviation.action() == draft.played() ? 1 : 2;
  }

  /** d = played（写しを作らない）か。 */
  boolean sameAsPlayed() {
    return deviation.action() == draft.played();
  }

  @Override
  public void accept(int action, double[] seatValues) {
    if (remaining <= 0
        || !Float.isNaN(values[action])
        || (action != draft.played() && action != deviation.action()))
      throw new IllegalStateException(
          "Unexpected root slot: game="
              + draft.gameId()
              + " decision="
              + draft.decisionId()
              + " action="
              + action);
    values[action] = (float) seatValues[draft.seat()];
    if (--remaining > 0) return;
    context.sink.accept(
        draft.root(
            values,
            deviation.action(),
            deviation.probability(),
            (float) (rollinRate * deviation.probability())));
    context.metrics.rowsWritten.increment();
  }
}
