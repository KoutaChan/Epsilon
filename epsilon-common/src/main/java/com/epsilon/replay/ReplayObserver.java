package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import java.util.List;

/**
 * 再生の判断と、適用した出来事を同期で受け取る。局面と合法手は呼び出しの間だけ借り、保持も変更もしない。局面は全席の伏せた牌と、分かっていれば牌山と裏ドラを含むので、方策の入力には判断する席の
 * {@link GameState#publicObservation} を使う。
 */
@FunctionalInterface
public interface ReplayObserver {
  /**
   * 手を適用する前の判断を受け取る。
   *
   * @param decision 値なので保持してよい
   * @param legal 読みだけの合法手。手番は生成器の順で、牌譜が宣言した持ち点不足のリーチの打牌は末尾。応答は窓を開いた時に作った手
   */
  void onDecision(ReplayDecision decision, GameState state, List<Action> legal);

  /**
   * 出来事を適用した後の局面を受け取る。どの出来事も 1 回だけ、それが解く判断より後に、位置の順に届く。応答の並びのロンは、並びの全ての判断の後に精算して届く。位置の順の例外は、同じ並びで
   * ロンの後に鳴きがある壊れた牌譜だけで、その鳴きは並びの最初のロンより先に届く。
   *
   * @param revision 局面の改訂。増えるだけで、同じなら局面も同じ（逆は成り立たない）
   * @param event 牌譜の出来事。和了は精算した形（牌譜に無い採点の内訳を補うことがある）で、このとき局面の手番の出来事はまだ和了の元（ツモ・打牌・槍槓の加槓）
   */
  default void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {}
}
