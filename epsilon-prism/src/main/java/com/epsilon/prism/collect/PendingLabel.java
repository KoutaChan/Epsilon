package com.epsilon.prism.collect;

import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerRounding;
import com.epsilon.prism.luck.LedgerTracker;
import java.util.Arrays;

/**
 * 局末の値を待つラベル行（MAIN・ROLLIN）。測る手 C ∪ {played}（{@link CandidateSet#measured}）の値がそろったら、Y を入れた行を
 * 書き出し先へ渡す。
 *
 * <p>打った手 played には、その経路（主経路か写し）の局末の値が入る。ほかの手には分岐の局末の値が入る。写しも分岐も「o で d を打ち、
 * 以後は全席が b で打つ」同じ過程なので、写しの d が測る手に入っていれば、d にはその写しの値を使ってよい。値は判断の席の Φ。全ての値は
 * ドライバのスレッドで受け取る。
 *
 * <p>帳簿を付ける収集（{@link CollectContext#ledger}）では、手ごとの運の精算の帳簿（{@link LedgerTracker}）を確率的に丸め
 * （{@link LedgerRounding}）、fp16 で表せる値にして行に書く。同じ受け取り先の局末の 3 ワードも集める（{@link CollectContext#write}）。
 */
final class PendingLabel implements Pending {
  private final CollectContext context;
  private final RowDraft draft;

  /** 合法手ごとの値（測っていない手は NaN）。 */
  private final float[] values;

  /** 手ごとの帳簿。長さは合法手の数 × {@link LedgerColumn#COUNT} で、測っていない手は NaN。帳簿を付けない収集では null。 */
  private final float[] ledger;

  /** 行の帳簿を丸める乱数のキー（{@link LedgerRounding#key}）。帳簿を付けない収集では 0。 */
  private final long roundingKey;

  /** 手ごとの局末の 3 ワード。長さは合法手の数 × {@link OutcomeCodec#WORDS} で、記録しない手は 0。 */
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
    ledger = context.ledger == null ? null : new float[draft.actions() * LedgerColumn.COUNT];
    if (ledger != null) Arrays.fill(ledger, Float.NaN);
    roundingKey = roundingKey(context, draft, ledger);
    outcomes = new long[draft.actions() * OutcomeCodec.WORDS];
    remaining = expected;
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
    write(ledger, roundingKey, action, terms);
  }

  @Override
  public void acceptOutcome(int action, HandEndFacts end, OutcomeCodec codec) {
    draft.record().outcome(codec, end, draft.seat(), action, outcomes);
  }

  /** 行の帳簿を丸める乱数のキー。{@link RootRecord} も使う。帳簿を付けない収集では 0。 */
  static long roundingKey(CollectContext context, RowDraft draft, float[] ledger) {
    return ledger == null ? 0 : LedgerRounding.key(context.iterationSeed, draft.gameId(), draft.decisionId());
  }

  /** 手を打った経路の帳簿を丸めて、行の帳簿に書く。{@link RootRecord} も使う。 */
  static void write(float[] ledger, long roundingKey, int action, double[] terms) {
    if (ledger == null) throw new IllegalStateException("Ledger delivered to a collection without ledgers");
    LedgerRounding.write(roundingKey, action, terms, ledger);
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
    context.write(draft, values, ledger, outcomes);
  }
}
