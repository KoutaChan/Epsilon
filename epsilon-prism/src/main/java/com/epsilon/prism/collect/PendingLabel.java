package com.epsilon.prism.collect;

import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import java.util.Arrays;

/**
 * 局末の値を待つラベル行（MAIN・ROLLIN）。測る手 C ∪ {played}（{@link CandidateSet#measured}）の値がそろったら、Y を入れた行を受け口へ渡す。
 *
 * <p>枠: 打った手 played はその経路（主経路か写し）の局末の値、ほかの手は分岐の局末の値（写しの d が測る手に入っていれば、その写しの値を d の枠に
 * 使ってよい。写しも分岐も「o で d、以後は全席 b」の同じ過程）。値は判断の席の Φ。全ての枠はドライバのスレッドで受け取る。
 *
 * <p>帳簿を付ける収集（{@link CollectContext#eventLedger}）では、手ごとの運の精算の帳簿（{@link EventLedger}）を確率的丸め（{@link
 * EventLedger#half}）で fp16 で表せる値にして行に書く。同じ枠の局末の 3 語も集める（{@link CollectContext#write}）。
 */
final class PendingLabel implements Pending {
  private final CollectContext context;
  private final RowDraft draft;

  /** 合法手ごとの値（測っていない手は NaN）。 */
  private final float[] values;

  /** 手ごとの帳簿（長さ 合法手 × {@value EventLedger#FEATURES}、測っていない手は NaN。帳簿を付けない収集は null）。 */
  private final float[] ledger;

  /** 手ごとの局末の 3 語（長さ 合法手 × {@link OutcomeCodec#WORDS}、記録しない手は 0）。 */
  private final long[] outcomes;

  private int remaining;

  /**
   * @param expected 待つ値の数（|C ∪ {played}|）
   */
  PendingLabel(CollectContext context, RowDraft draft, int expected) {
    this.context = context;
    this.draft = draft;
    values = new float[draft.actions()];
    Arrays.fill(values, Float.NaN);
    ledger = context.eventLedger ? new float[draft.actions() * EventLedger.FEATURES] : null;
    if (ledger != null) Arrays.fill(ledger, Float.NaN);
    outcomes = new long[draft.actions() * OutcomeCodec.WORDS];
    remaining = expected;
  }

  @Override
  public int seat() {
    return draft.seat();
  }

  @Override
  public void acceptLedger(int action, double[] terms) {
    write(context, draft, ledger, action, terms);
  }

  @Override
  public void acceptOutcome(int action, HandEndFacts end, OutcomeCodec codec) {
    draft.record().outcome(codec, end, draft.seat(), action, outcomes);
  }

  /** 手 action の帳簿 terms を丸めて ledger に書く（{@link RootRecord} も使う）。 */
  static void write(CollectContext context, RowDraft draft, float[] ledger, int action, double[] terms) {
    if (ledger == null) throw new IllegalStateException("Ledger delivered to a collection without ledgers");
    for (int feature = 0; feature < EventLedger.FEATURES; feature++)
      ledger[action * EventLedger.FEATURES + feature] =
          EventLedger.half(
              terms[feature],
              EventLedger.coin(context.iterationSeed, draft.gameId(), draft.decisionId(), action, feature));
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
    if (ledger != null && EventLedger.dumping()) EventLedger.dump(draft, values, ledger);
    context.write(draft, values, ledger, outcomes);
  }
}
