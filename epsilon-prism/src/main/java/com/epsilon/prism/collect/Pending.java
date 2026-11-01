package com.epsilon.prism.collect;

import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OutcomeCodec;

/**
 * 局末の値を待つ記録（{@link PendingLabel}・{@link RootRecord}）。進行ワーカーが判断の時点で作り、値はドライバのスレッドだけが渡す（{@link
 * CollectionDriver} の settle。作った進行の HandEnd がドライバのイベントキューを通るので、作成は受け取りより前に見える）。
 *
 * <p>枠ごとの受け渡しの順は {@link #acceptLedger} → {@link #acceptOutcome} → {@link #accept}。
 */
interface Pending {
  /**
   * 手 action を打った経路の局末の 4 席の値。ドライバのスレッドで、枠ごとに一度だけ呼ぶ。全ての枠がそろったら行を受け口へ渡す。
   *
   * @param seatValues 4 席の Φ（半荘末は最終効用）。判断の席の値だけを使う
   */
  void accept(int action, double[] seatValues);

  /** 判断の席（運の精算の帳簿を足す向きと、聴牌への到達を追う席。席を持たない記録は -1）。 */
  default int seat() {
    return -1;
  }

  /**
   * 手 action の経路の運の精算の帳簿（{@link EventLedger}、長さ {@value EventLedger#FEATURES}）。帳簿を付ける収集で、{@link #accept} の直前に
   * ドライバのスレッドで呼ぶ。
   */
  default void acceptLedger(int action, double[] ledger) {}

  /**
   * 手 action の経路の局末の事実（行を書く収集だけ）。帳簿の後・{@link #accept} の前に、ドライバのスレッドで呼ぶ。
   *
   * @param end 経路の局末の事実
   * @param codec 局末の 3 語の書き手（ドライバに 1 つ）
   */
  default void acceptOutcome(int action, HandEndFacts end, OutcomeCodec codec) {}
}
