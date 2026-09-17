package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.HandView;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.WinLegality.RonStatus;
import java.util.List;

/** 麻雀の各決定点に対する合法アクション生成器。ワーカーごとに所有し、決定点間で再利用する。 */
public final class ActionGenerator {
  private final HandScoreEvaluator winEvaluation = new HandScoreEvaluator();
  private final HandShapeAnalyzer handAnalysis = new HandShapeAnalyzer();

  public ActionGenerator() {}

  /**
   * 出力一覧を消去し、ツモ後の全合法行動を書き込む。
   *
   * @param actions 再利用する出力一覧
   * @param state 現在の局状態
   * @param player 手番の席番号（0-3）
   * @param draw 現在のツモイベント
   */
  public void generateTurnActionsInto(
      List<Action> actions, GameState state, int player, TurnEvent.Draw draw) {
    TurnActionGenerator.generateInto(
        actions, state, player, draw, null, winEvaluation, handAnalysis);
  }

  /**
   * サーバーがリーチ宣言を許可した手牌について、宣言牌だけを列挙する。
   *
   * <p>点数・門前・残り山などの宣言条件は再判定せず、通常手番と同じ聴牌形・赤牌・ツモ切りの候補を返す。
   *
   * @param actions 再利用する出力一覧
   * @param hand 自摸後の自家手牌
   * @param draw 現在のツモイベント
   */
  public void generateRiichiDahaiActionsInto(List<Action> actions, Hand hand, TurnEvent.Draw draw) {
    actions.clear();
    TurnActionGenerator.addRiichiDeclarationActions(actions, hand, draw, handAnalysis);
  }

  void generateTurnActionsInto(
      EngineDecisionPoint decision, GameState state, int player, TurnEvent.Draw draw) {
    TurnActionGenerator.generateInto(
        decision.mutableLegalActions(), state, player, draw, decision, winEvaluation, handAnalysis);
  }

  /**
   * 出力一覧を消去して応答行動を書き込み、RON可否の詳細も返す。
   *
   * @param actions 再利用する出力一覧
   * @param state 現在の局状態
   * @param player 応答する席番号（0-3）
   * @param source 応答対象の打牌または槓宣言
   * @return RON不可理由と見逃し時のフリテン効果
   */
  public RonStatus generateResponseActionsInto(
      List<Action> actions, GameState state, int player, TurnEvent.ResponseSource source) {
    return ResponseActionGenerator.generateInto(
        actions, state, player, source, null, winEvaluation, handAnalysis);
  }

  RonStatus generateResponseActionsInto(
      EngineDecisionPoint decision, GameState state, int player, TurnEvent.ResponseSource source) {
    return ResponseActionGenerator.generateInto(
        decision.mutableLegalActions(),
        state,
        player,
        source,
        decision,
        winEvaluation,
        handAnalysis);
  }

  /** 行動一覧が不要な境界でRON可否だけを返す。 */
  public RonStatus ronStatus(GameState state, int player, TurnEvent.ResponseSource source) {
    return WinLegality.ronStatus(state, player, source, winEvaluation, handAnalysis);
  }

  /**
   * 出力一覧を消去し、鳴き直後の合法打牌行動を書き込む。
   *
   * @param actions 再利用する出力一覧
   * @param hand 鳴きを既に適用した手牌
   * @param restriction 鳴いた牌形から導出した喰い替え制約
   */
  public static void generateCallDahaiActionsInto(
      List<Action> actions, HandView hand, PostCallDahaiRestriction restriction) {
    actions.clear();
    long discardableTileTypeMask = hand.concealedTileTypeMask() & ~restriction.forbiddenMask();
    for (long remaining = discardableTileTypeMask; remaining != 0L; remaining &= remaining - 1) {
      int tileType = Long.numberOfTrailingZeros(remaining);
      if (!hand.hasAkaTile(tileType)) {
        actions.add(Action.dahai(tileType));
        continue;
      }
      actions.add(Action.dahai(tileType, true));
      if (hand.hasNonAkaTile(tileType)) {
        actions.add(Action.dahai(tileType, false));
      }
    }
  }
}
