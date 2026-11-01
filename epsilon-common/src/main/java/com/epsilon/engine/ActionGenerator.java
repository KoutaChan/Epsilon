package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.WinLegality.RonStatus;
import java.util.List;

/**
 * 麻雀の各決定点に対する合法アクション生成器。ワーカーごとに所有し、決定点間で再利用する。
 *
 * <p>生成はいつも判断点へ書く（和了が合法なら点も判断点の箱へ書く）。一覧へ書く public な口（牌譜の再生・対局の client）は、この生成器の
 * 作業用の判断点へ生成してから一覧へ写す。作業用の判断点はその口を初めて使うときに作る（{@link GameEngine} は判断点の口だけを使うので、
 * 分岐ごとに作る生成器は作業用の判断点を持たない）。
 */
public final class ActionGenerator {
  private final HandScoreEvaluator winEvaluation = new HandScoreEvaluator();
  private final HandShapeAnalyzer handAnalysis = new HandShapeAnalyzer();

  /**
   * 一覧へ書く public な口の作業用の判断点。この生成器だけが持ち、1 回の呼び出しの間だけ使う（ID・種類・席は束ねない）。初めて使うまで
   * null（{@link #listScratch}）。
   */
  private EngineDecisionPoint scratch;

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
    EngineDecisionPoint decision = listScratch();
    TurnActionGenerator.generateInto(decision, state, player, draw, winEvaluation, handAnalysis);
    decision.mutableLegalActions().copyInto(actions);
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
    TurnActionGenerator.generateInto(decision, state, player, draw, winEvaluation, handAnalysis);
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
    EngineDecisionPoint decision = listScratch();
    RonStatus ronStatus =
        ResponseActionGenerator.generateInto(
            decision, state, player, source, winEvaluation, handAnalysis);
    decision.mutableLegalActions().copyInto(actions);
    return ronStatus;
  }

  RonStatus generateResponseActionsInto(
      EngineDecisionPoint decision, GameState state, int player, TurnEvent.ResponseSource source) {
    return ResponseActionGenerator.generateInto(
        decision, state, player, source, winEvaluation, handAnalysis);
  }

  /** 一覧へ書く口の作業用の判断点。初めての呼び出しで作り、以後は同じものを返す。 */
  private EngineDecisionPoint listScratch() {
    if (scratch == null) scratch = new EngineDecisionPoint();
    return scratch;
  }

  /** 行動一覧が不要な境界でRON可否だけを返す（点は求めない probe の判定）。 */
  public RonStatus ronStatus(GameState state, int player, TurnEvent.ResponseSource source) {
    return WinLegality.ronStatus(state, player, source, winEvaluation, handAnalysis);
  }
}
