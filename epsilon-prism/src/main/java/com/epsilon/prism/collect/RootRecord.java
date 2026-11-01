package com.epsilon.prism.collect;

import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import java.util.Arrays;

/**
 * 写しの根の記録（ROOT）。主経路の判断 o で、打った手 played の枠（主経路の局末の値 y_main）と逸脱 d の枠（写しの局末の値
 * y_br）を待つ。d = played なら写しを作らず played の 1 枠だけ（y_br = y_main）。
 *
 * <p>そろったら行にして受け口へ渡す: values は played と d の枠だけ、deviation = d、deviationProbability = q(d|o)、propensity = β_c(o)·q(d|o)
 * （どれも下書き {@link RowDraft} が持つ）。全ての枠はドライバのスレッドで受け取る。帳簿を付ける収集では、played と d の運の精算の帳簿
 * （{@link EventLedger}）もラベル行と同じ丸めで書く。played と d の局末の 3 語も集める（{@link CollectContext#write}）。
 */
final class RootRecord implements Pending {
  private final CollectContext context;
  private final RowDraft draft;

  private final float[] values;

  /** 手ごとの運の精算の帳簿（{@link PendingLabel} と同じ形。帳簿を付けない収集は null）。 */
  private final float[] ledger;

  /** 手ごとの局末の 3 語（{@link PendingLabel} と同じ形）。 */
  private final long[] outcomes;

  private int remaining;

  RootRecord(CollectContext context, RowDraft draft) {
    this.context = context;
    this.draft = draft;
    values = new float[draft.actions()];
    Arrays.fill(values, Float.NaN);
    ledger = context.eventLedger ? new float[draft.actions() * EventLedger.FEATURES] : null;
    if (ledger != null) Arrays.fill(ledger, Float.NaN);
    outcomes = new long[draft.actions() * OutcomeCodec.WORDS];
    remaining = sameAsPlayed() ? 1 : 2;
  }

  /** d = played（写しを作らない）か。 */
  boolean sameAsPlayed() {
    return draft.deviation() == draft.played();
  }

  @Override
  public int seat() {
    return draft.seat();
  }

  @Override
  public void acceptLedger(int action, double[] terms) {
    PendingLabel.write(context, draft, ledger, action, terms);
  }

  @Override
  public void acceptOutcome(int action, HandEndFacts end, OutcomeCodec codec) {
    draft.record().outcome(codec, end, draft.seat(), action, outcomes);
  }

  @Override
  public void accept(int action, double[] seatValues) {
    if (remaining <= 0
        || !Float.isNaN(values[action])
        || (action != draft.played() && action != draft.deviation()))
      throw new IllegalStateException(
          "Unexpected root slot: game="
              + draft.gameId()
              + " decision="
              + draft.decisionId()
              + " action="
              + action);
    values[action] = (float) seatValues[draft.seat()];
    if (--remaining > 0) return;
    context.write(draft, values, ledger, outcomes);
  }
}
