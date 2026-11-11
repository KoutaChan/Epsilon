package com.epsilon.prism.collect;

import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerTracker;

/**
 * 局末の値を待つ記録（{@link PendingLabel}・{@link RootRecord}）。進行ワーカーが判断の時点で作り、値はドライバのスレッドだけが渡す
 * （{@link CollectContext#accept}）。作った進行の HandEnd がドライバのイベントの列を通るので、作ったことは受け取るより前に見える。
 *
 * <p>受け取り先ごとに、{@link #acceptLedger} → {@link #acceptOutcome} → {@link #accept} の順に渡す。
 */
interface Pending {
  /**
   * 手 {@code action} を打った経路の、局末の全席の値を受け取る。ドライバのスレッドで、受け取り先ごとに一度だけ呼ぶ。全ての受け取り先の
   * 値がそろったら、行を書き出し先へ渡す。
   *
   * @param action 手の位置
   * @param seatValues 全席の Φ（半荘の終わりでは最終の効用）。判断の席の値だけを使う
   */
  void accept(int action, double[] seatValues);

  /** 判断の席。運の精算の帳簿を足す向きと、聴牌への到達を追う席になる。席を持たない記録は -1。 */
  default int seat() {
    return -1;
  }

  /** 判断のクラス。裏ドラの監査はこのクラスで分ける。 */
  DecisionClass decisionClass();

  /**
   * 手 {@code action} を打った経路の、運の精算の帳簿（{@link LedgerTracker}。長さは {@link LedgerColumn#COUNT}）を受け取る。
   * 帳簿を付ける収集で、{@link #accept} の直前にドライバのスレッドで呼ぶ。
   */
  default void acceptLedger(int action, double[] ledger) {}

  /**
   * 手 {@code action} を打った経路の、局末の事実を受け取る。行を書く収集だけで、帳簿の後、{@link #accept} の前にドライバのスレッドで呼ぶ。
   *
   * @param end 経路の局末の事実
   * @param codec 局末の 3 ワードのコーデック（ドライバに 1 つ）
   */
  default void acceptOutcome(int action, HandEndFacts end, OutcomeCodec codec) {}
}
