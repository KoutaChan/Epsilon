package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDScope;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * 順伝播の一部で作る配列のスコープ。スコープを開いている間にこのスレッドが作った配列は、残すと決めたものを除いて、スコープを閉じるときに
 * 閉じる。残した配列は外のスコープに移り、外のスコープを閉じるときに閉じる。外にスコープが無ければ、配列を持つマネージャーを閉じるまで残る。
 *
 * <p>DJL は、マネージャーを閉じるまで配列を持ち続ける。スコープを使わないと、順伝播の途中の配列が全てバッチの終わりまでデバイスに残る。
 * スコープを使えば、次の部分が読む出力だけを残し、途中の配列はその場で解放できる。
 *
 * <p>スコープの中で作る配列は、順伝播を流すストリームで確保し、同じストリームでだけ読む。そのため、閉じた配列のメモリを後の確保が使い回しても、
 * 先に積んだ演算が読み終える前にメモリが書き換わることはない。ほかのストリームが読む配列（転送のストリームで確保した入力と、ホストへコピーする
 * 確率）はスコープの外で作り、スコープでは閉じない。学習の順伝播でも閉じてよい。逆伝播に要る値は、自動微分のグラフが配列とは別に持っている。
 *
 * <p>残してよいのはスコープの中で作った配列だけ。スコープの外で作った配列を残すと、外のスコープがそれを閉じてしまう。
 */
final class ArrayScope implements AutoCloseable {

  private final NDScope arrays = new NDScope();
  private final List<NDArray> kept = new ArrayList<>();

  /**
   * 順伝播の一部をスコープの中で計算し、結果だけを残す。
   *
   * @param part スコープの中で新しい配列を作って返す計算
   * @return 計算の結果
   */
  static NDArray of(Supplier<NDArray> part) {
    try (ArrayScope scope = new ArrayScope()) {
      return scope.keep(part.get());
    }
  }

  /**
   * 配列を残す。残した配列は、このスコープを閉じても閉じない。
   *
   * @param array スコープの中で作った配列（null なら何もしない）
   * @return 同じ配列
   */
  NDArray keep(NDArray array) {
    if (array != null) {
      NDScope.unregister(array);
      kept.add(array);
    }
    return array;
  }

  /** 残さない配列を閉じ、残した配列を外のスコープへ渡す。 */
  @Override
  public void close() {
    arrays.close();
    for (NDArray array : kept) {
      NDScope.register(array);
    }
  }
}
