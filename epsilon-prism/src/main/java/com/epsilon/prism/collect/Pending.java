package com.epsilon.prism.collect;

/**
 * 局末の値を待つ記録（{@link PendingLabel}・{@link RootRecord}）。進行ワーカーが判断の時点で作り、値はドライバのスレッドだけが渡す（{@link
 * CollectionDriver} の settle。作った進行の HandEnd がドライバのイベントキューを通るので、作成は受け取りより前に見える）。
 */
interface Pending {
  /**
   * 手 action を打った経路の局末の 4 席の値。ドライバのスレッドで、枠ごとに一度だけ呼ぶ。全ての枠がそろったら行を受け口へ渡す。
   *
   * @param seatValues 4 席の Φ（半荘末は最終効用）。判断の席の値だけを使う
   */
  void accept(int action, double[] seatValues);
}
