package com.epsilon.mahjong.client;

import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.SeatView;

/** ライブの対局で、自分の席の手を選ぶ。クライアントは、自分に判断が回ってくるたびに 1 回呼ぶ。 */
@FunctionalInterface
public interface Decider {

  /**
   * 合法手から 1 つ選ぶ。合法手は、エンジンの合法手のうちサーバが出したものに絞ってある。
   *
   * @param view 判断する席から見た卓。この呼び出しの間だけ有効
   * @param decisions 席ごとの判断。この呼び出しの間だけ有効
   * @param slot 判断する席が、席ごとの判断の何番目か
   * @return 選んだ手（合法手の何番目か）
   */
  int decide(SeatView view, Decisions decisions, int slot);
}
