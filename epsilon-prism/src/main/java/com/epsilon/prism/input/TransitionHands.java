package com.epsilon.prism.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.prism.Limits;

/**
 * 判断の遷移ごとに、遷移の後の手を作る。入力のライターとフロンティアの計算が、同じ決まりで遷移をたどり、同じ後の手と形のスロットを
 * 使うための部品。メモリを割り当てない。
 *
 * <p>遷移を持つのは代表の手（{@link Observation#representative}）だけで、その数は {@link Observation#transitionsOf} 個。遷移の後の手は
 * 次のとおり。
 *
 * <ul>
 *   <li>打牌・立直: 今の手からその牌を打牌した手。形のスロットは捨てる牌の種類。
 *   <li>チー・ポン: 鳴いた手（合法手ごとに 1 回作る）から、鳴いた後の打牌の 1 つを打牌した手。遷移は鳴いた後の打牌の順に並ぶ。形のスロットは {@link ShapeAnalysis#callSlot}。
 *   <li>見送り: 今の手。形のスロットは {@link ShapeAnalysis#CURRENT}。
 *   <li>ほか（槓・北抜き・和了・九種九牌）: 今の手でその合法手を行った後の手（和了・九種九牌は今の手のまま）。形のスロットは {@link ShapeAnalysis#UNSHARED}。
 * </ul>
 */
public final class TransitionHands {

  private final AfterHand current = new AfterHand();
  private final AfterHand called = new AfterHand();
  private final AfterHand after = new AfterHand();
  private final int[] firstCallDiscard = new int[Limits.MAX_LEGAL_ACTIONS];
  private Observation o;
  private Action discard;
  private int slot;

  /**
   * 判断の時点の手を作る。合法手ごとに、チー・ポンの後の打牌が、判断の記録にある鳴いた後の打牌の列で何番目から始まるかも数えておく。
   *
   * @param o 判断の記録
   * @param tiles 種類ごとの枚数
   * @param seats 席ごとの公開情報
   */
  public void load(Observation o, TileCounts tiles, SeatFacts seats) {
    this.o = o;
    current.loadCurrent(o, tiles, seats);
    int callDiscards = 0;
    for (int action = 0, actions = o.legalActions(); action < actions; action++) {
      firstCallDiscard[action] = callDiscards;
      callDiscards += o.callDiscardsOf(action);
    }
  }

  /** 判断の時点の手。 */
  public AfterHand current() {
    return current;
  }

  /**
   * 遷移の後の手を作る。同じ合法手の遷移は、番号の順に続けて呼ぶこと（チー・ポンの鳴いた手は番号 0 のときに作る）。
   *
   * @param action 代表の合法手の番号
   * @param index その手の遷移の番号
   * @return 後の手（次に作るまで有効）
   */
  public AfterHand move(int action, int index) {
    Action chosen = o.action(action);
    discard = null;
    switch (chosen.type()) {
      case DAHAI, RIICHI_DAHAI -> {
        discard = chosen;
        slot = chosen.tile();
        after.copyFrom(current);
        after.discard(chosen);
      }
      case CHI, PON -> {
        discard = o.callDiscard(firstCallDiscard[action] + index);
        slot = ShapeAnalysis.callSlot(chosen, o.lastEventTile(), discard.tile());
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
      case DAIMINKAN, ANKAN, KAKAN, NUKI, TSUMO_AGARI, RON_AGARI, KYUSHU_KYUHAI -> {
        slot = ShapeAnalysis.UNSHARED;
        after.copyFrom(current);
        after.apply(chosen, o);
      }
    }
    return after;
  }

  /** 遷移で牌を捨てる手。打牌・立直はその手、チー・ポンは鳴いた後の打牌、ほかは null。 */
  public Action discard() {
    return discard;
  }

  /** 遷移の形のスロット（{@link ShapeAnalysis#analyze} に渡す）。 */
  public int slot() {
    return slot;
  }

  /** 局を終える手（和了・九種九牌）なら true。このときは形と待ちを解析しない。 */
  public static boolean terminal(Action.Type type) {
    return type == Action.Type.TSUMO_AGARI
        || type == Action.Type.RON_AGARI
        || type == Action.Type.KYUSHU_KYUHAI;
  }
}
