package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionGenerator;
import com.epsilon.engine.EngineActionBuffer;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.PostCallDiscards;
import com.epsilon.engine.WinLegality.RonStatus;
import java.util.List;

/** 再生の合法手。返す一覧はここで使い回す読みだけの見方で、手番の一覧は次の手番の生成まで、席の応答の一覧は次に窓を開くまで有効。 */
final class LegalActions {
  /** リーチの宣言に要る山の残り（通常の手番の生成と同じ条件）。 */
  private static final int RIICHI_WALL_TILES = 4;

  private final GameState state;
  private final ActionGenerator generator = new ActionGenerator();
  private final EngineActionBuffer turnActions = new EngineActionBuffer();
  private final EngineActionBuffer riichiDiscards = new EngineActionBuffer();
  private final EngineActionBuffer[] responseActions =
      new EngineActionBuffer[GameState.NUM_PLAYERS];

  LegalActions(GameState state) {
    this.state = state;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
      responseActions[seat] = new EngineActionBuffer();
  }

  List<Action> turn(int seat, TurnEvent.Draw draw) {
    generator.generateTurnActionsInto(turnActions, state, seat, draw);
    return turnActions.borrowedView();
  }

  /**
   * ツモの後の打牌の判断の合法手。牌譜がリーチを宣言した席で、持ち点だけが足りない（リーチしていない・門前・山が 4 枚以上）なら、聴牌のリーチの打牌を末尾に足す。点は変えない（成立の時に
   * 通常どおり払い、負になってよい）。
   */
  List<Action> discard(int seat, TurnEvent.Draw draw, boolean riichiDeclared) {
    generator.generateTurnActionsInto(turnActions, state, seat, draw);
    if (riichiDeclared
        && state.getScore(seat) < HanchanProgression.RIICHI_COST
        && !state.isRiichi(seat)
        && state.hand(seat).isMenzen()
        && state.remainingWallTiles() >= RIICHI_WALL_TILES) {
      generator.generateRiichiDahaiActionsInto(riichiDiscards, state.hand(seat), draw);
      for (int index = 0; index < riichiDiscards.size(); index++)
        turnActions.add(riichiDiscards.get(index));
    }
    return turnActions.borrowedView();
  }

  /** チー・ポンの直後の打牌（喰い替えを除く）。 */
  List<Action> afterCall(int seat, Meld call) {
    PostCallDiscards.generateInto(state.hand(seat), call, turnActions);
    return turnActions.borrowedView();
  }

  /** seat の応答の合法手を作る（{@link #responses(int)} で読む）。PASS はほかの手がある時だけ最後に入る。 */
  RonStatus respond(int seat, TurnEvent.ResponseSource origin) {
    return generator.generateResponseActionsInto(responseActions[seat], state, seat, origin);
  }

  List<Action> responses(int seat) {
    return responseActions[seat].borrowedView();
  }
}
