package com.epsilon.agent.play;

import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.TableState;

/**
 * 卓の主経路を観測するインターフェース。進行のスレッドから並行に呼ばれる（同じ半荘は 1 度に 1 つのスレッドから）。渡すビューと局末は
 * 呼び出しの間だけ有効で、観測する側は局面を書き換えない。
 */
public interface GameObserver {

  /**
   * 半荘が終わったときに呼ぶ。
   *
   * @param last 終局の卓
   */
  void gameEnded(GameKey game, TableState last);

  /**
   * 判断の手が決まったとき、エンジンを進める前に呼ぶ。
   *
   * @param choice {@code slot} 番目の判断で選んだ合法手の位置
   */
  default void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {}

  /** エンジンが止まるたびに呼ぶ。止まるのは、局の最初の判断の時点と、判断を進めた後の次の判断の時点か局末。牌譜を書くのに使う。 */
  default void advanced(GameKey game, FullView full) {}

  /** 局が終わったとき、次局を配る前に呼ぶ。半荘が終わる局でも、{@link #gameEnded} の前に呼ぶ。 */
  default void roundEnded(GameKey game, FullView full, RoundEnd end) {}
}
