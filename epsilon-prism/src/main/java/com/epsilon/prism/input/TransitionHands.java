package com.epsilon.prism.input;

import com.epsilon.core.Action;
import com.epsilon.prism.Limits;
import com.epsilon.prism.observe.Observation;

/**
 * 判断の遷移の後の手を作る。Tokenizer と前線（{@link FrontierFeatures}）が同じ決まりで遷移をたどり、同じ後の手と形の枠を使うための部品。
 * 割り当てをしない。
 *
 * <p>遷移は代表の手（{@link Observation#representative}）だけが {@link Observation#transitionsOf} 個持つ。後の手は次のとおり。
 *
 * <ul>
 *   <li>打牌・立直: 今の手に打牌を当てた手。形の枠は捨てる牌。
 *   <li>チー・ポン: 鳴いた後の打牌の順に、鳴いた手（合法手ごとに 1 回作る）に打牌を当てた手。形の枠は {@link ShapeAnalysis#callSlot}。
 *   <li>見送り: 今の手。形の枠は {@link ShapeAnalysis#CURRENT}。
 *   <li>ほか（槓・和了・九種九牌）: 今の手に手を当てた手（和了・九種九牌は今の手のまま）。形の枠は {@link ShapeAnalysis#UNSHARED}。
 * </ul>
 */
final class TransitionHands {

  private final AfterHand current = new AfterHand();
  private final AfterHand called = new AfterHand();
  private final AfterHand after = new AfterHand();
  private final int[] firstCallDiscard = new int[Limits.MAX_LEGAL_ACTIONS];
  private Observation o;
  private Action discard;
  private int slot;

  /**
   * 判断の時点の手を作り、チー・ポンの後の打牌の位置を数える。
   *
   * @param o 判断の記録
   * @param tiles 牌ごとの枚数
   * @param events O の出来事
   */
  void load(Observation o, TileCounts tiles, EventTable events) {
    this.o = o;
    current.loadCurrent(o, tiles, events);
    int callDiscards = 0;
    for (int action = 0, actions = o.legalActions(); action < actions; action++) {
      firstCallDiscard[action] = callDiscards;
      callDiscards += o.callDiscardsOf(action);
    }
  }

  /** 判断の時点の手。 */
  AfterHand current() {
    return current;
  }

  /**
   * 遷移の後の手を作る。同じ合法手の遷移は番号の順に続けて作る（チー・ポンの鳴いた手は番号 0 で作る）。
   *
   * @param action 代表の合法手の番号
   * @param index その手の遷移の番号
   * @return 後の手（次に作るまで有効）
   */
  AfterHand move(int action, int index) {
    Action chosen = o.action(action);
    discard = null;
    switch (chosen.type()) {
      case DAHAI, RIICHI_DAHAI -> {
        discard = chosen;
        slot = chosen.tileType();
        after.copyFrom(current);
        after.discard(chosen);
      }
      case CHI, PON -> {
        discard = o.callDiscard(firstCallDiscard[action] + index);
        slot = ShapeAnalysis.callSlot(chosen, o.lastEventTile(), discard.tileType());
        if (index == 0) {
          called.copyFrom(current);
          called.apply(chosen, o);
        }
        after.copyFrom(called);
        after.discard(discard);
      }
      case PASS -> {
        slot = ShapeAnalysis.CURRENT;
        after.copyFrom(current);
      }
      case DAIMINKAN, ANKAN, KAKAN, TSUMO_AGARI, RON_AGARI, KYUSHU_KYUHAI -> {
        slot = ShapeAnalysis.UNSHARED;
        after.copyFrom(current);
        after.apply(chosen, o);
      }
    }
    return after;
  }

  /** 遷移で捨てる牌の手（打牌・立直はその手、チー・ポンは鳴いた後の打牌。ほかは null）。 */
  Action discard() {
    return discard;
  }

  /** 遷移の形の枠（{@link ShapeAnalysis#analyze} に渡す）。 */
  int slot() {
    return slot;
  }

  /**
   * 遷移が局面を終える手（和了・九種九牌）なら true を返す。形と待ちを解析しない。
   *
   * @param type 手の種類
   * @return 和了・九種九牌なら true
   */
  static boolean terminal(Action.Type type) {
    return type == Action.Type.TSUMO_AGARI
        || type == Action.Type.RON_AGARI
        || type == Action.Type.KYUSHU_KYUHAI;
  }
}
