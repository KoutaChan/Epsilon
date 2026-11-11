package com.epsilon.prism.collect;

import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerTracker;
import java.util.Arrays;

/**
 * 写しの根の記録（ROOT）。主経路の判断 o で、打った手 played の値（主経路の局末の値 y_main）と、逸脱 d の値（写しの局末の値 y_br）を
 * 待つ。d = played なら写しを作らず、played の値 1 つだけを待つ（y_br = y_main）。
 *
 * <p>値がそろったら行にして書き出し先へ渡す。values には played と d の値だけが入り、deviation = d、deviationProbability = q(d|o)、
 * propensity = β_c(o)·q(d|o) になる（どれも下書き {@link RowDraft} が持つ）。全ての値はドライバのスレッドで受け取る。帳簿を付ける収集
 * では、played と d の運の精算の帳簿（{@link LedgerTracker}）も、ラベル行と同じ丸め方で書く。played と d の局末の 3 ワードも集める
 * （{@link CollectContext#write}）。
 */
final class RootRecord implements Pending {
  private final CollectContext context;
  private final RowDraft draft;

  private final float[] values;

  /** 手ごとの運の精算の帳簿。形は {@link PendingLabel} と同じ。帳簿を付けない収集では null。 */
  private final float[] ledger;

  /** 行の帳簿を丸める乱数のキー（{@link PendingLabel#roundingKey}）。 */
  private final long roundingKey;

  /** 手ごとの局末の 3 ワード。形は {@link PendingLabel} と同じ。 */
  private final long[] outcomes;

  private int remaining;

  RootRecord(CollectContext context, RowDraft draft) {
    this.context = context;
    this.draft = draft;
    values = new float[draft.actions()];
    Arrays.fill(values, Float.NaN);
    ledger = context.ledger == null ? null : new float[draft.actions() * LedgerColumn.COUNT];
    if (ledger != null) Arrays.fill(ledger, Float.NaN);
    roundingKey = PendingLabel.roundingKey(context, draft, ledger);
    outcomes = new long[draft.actions() * OutcomeCodec.WORDS];
    remaining = sameAsPlayed() ? 1 : 2;
  }

  /** d = played か。そのときは写しを作らない。 */
  boolean sameAsPlayed() {
    return draft.deviation() == draft.played();
  }

  @Override
  public int seat() {
    return draft.seat();
  }

  @Override
  public DecisionClass decisionClass() {
    return draft.decisionClass();
  }

  @Override
  public void acceptLedger(int action, double[] terms) {
    PendingLabel.write(ledger, roundingKey, action, terms);
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
