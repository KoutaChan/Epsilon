package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import java.util.List;

/** {@link Player} を同期的に呼び出して {@link GameEngine} を最後まで進行する。 */
public final class SynchronousGameRunner {

  private final GameEngine engine;
  private final Player[] players;
  private final EngineSelectionBuffer selectionBuffer = new EngineSelectionBuffer();

  /**
   * 無作為な牌山シードを使う新しいエンジンと4席のプレイヤーを結び付ける。
   *
   * @param players 席順どおりに並んだ4要素のプレイヤー配列
   */
  public SynchronousGameRunner(Player[] players) {
    this(new GameEngine(), players);
  }

  /**
   * 指定した牌山シードを使う新しいエンジンと4席のプレイヤーを結び付ける。
   *
   * @param players 席順どおりに並んだ4要素のプレイヤー配列
   * @param wallSeed 牌山生成に使うシード
   */
  public SynchronousGameRunner(Player[] players, long wallSeed) {
    this(new GameEngine(wallSeed), players);
  }

  /**
   * 無作為な牌山シードと指定記録処理を使う新しいエンジンを生成する。
   *
   * @param players 席順どおりに並んだ4要素のプレイヤー配列
   * @param recorder 局のイベントを受け取る記録処理
   */
  public SynchronousGameRunner(Player[] players, GameRecorder recorder) {
    this(new GameEngine(recorder), players);
  }

  /**
   * 指定した牌山シードと記録処理を使う新しいエンジンを生成する。
   *
   * @param players 席順どおりに並んだ4要素のプレイヤー配列
   * @param wallSeed 牌山生成に使うシード
   * @param recorder 局のイベントを受け取る記録処理
   */
  public SynchronousGameRunner(Player[] players, long wallSeed, GameRecorder recorder) {
    this(new GameEngine(wallSeed, recorder), players);
  }

  /**
   * 呼び出し元が構成したエンジンと4席のプレイヤーを結び付ける。
   *
   * @param engine 進行対象の対局エンジン
   * @param players 席順どおりに並んだ4要素のプレイヤー配列
   * @throws IllegalArgumentException プレイヤー数が4でない場合
   */
  public SynchronousGameRunner(GameEngine engine, Player[] players) {
    if (players.length != GameState.NUM_PLAYERS) {
      throw new IllegalArgumentException("Exactly 4 players required");
    }
    this.engine = engine;
    // 呼び出し元が所有する固定席配列を実行処理寿命中借用する。実行中の差し替えは禁止。
    this.players = players;
  }

  /**
   * 半荘を最後まで進行し、最終点を返す。
   *
   * @return 席順どおりに並ぶ終局時の持ち点
   */
  public int[] playHanchan() {
    GameStepResult step = engine.stepHanchan();
    while (true) {
      switch (step) {
        case GameStepResult.AwaitingDecisions awaiting ->
            step = engine.commitDecisions(select(awaiting.decisions()));
        case GameStepResult.RoundSettled settled -> {
          notifyRoundSettled(settled.settlement());
          step = engine.stepHanchan();
        }
        case GameStepResult.HanchanEnded ended -> {
          notifyRoundSettled(ended.settlement());
          return ended.snapshotFinalScores();
        }
        case GameStepResult.RoundEnded ignored ->
            throw new IllegalStateException("Standalone round ended during a hanchan");
      }
    }
  }

  /**
   * 呼び出し元が初期化した局を最後まで進行する。
   *
   * @return 和了または流局による局結果
   */
  public RoundResult playRound() {
    GameStepResult step = engine.stepRound();
    while (true) {
      switch (step) {
        case GameStepResult.AwaitingDecisions awaiting ->
            step = engine.commitDecisions(select(awaiting.decisions()));
        case GameStepResult.RoundEnded ended -> {
          return ended.result();
        }
        case GameStepResult.RoundSettled ignored ->
            throw new IllegalStateException("Standalone round was settled as a hanchan round");
        case GameStepResult.HanchanEnded ignored ->
            throw new IllegalStateException("Round stepping ended a hanchan");
      }
    }
  }

  /**
   * 現在の局を最後まで進め、点棒精算まで適用する。
   *
   * @return 局結果、精算後持ち点および次局遷移
   */
  public RoundSettlement playOutRound() {
    return engine.settleRound(playRound());
  }

  private List<EngineDecisionSelection> select(List<EngineDecisionPoint> decisions) {
    selectionBuffer.clear();
    for (int decisionIndex = 0; decisionIndex < decisions.size(); decisionIndex++) {
      EngineDecisionPoint decision = decisions.get(decisionIndex);
      Action action =
          players[decision.player()].selectAction(
              engine.getState(), decision.player(), decision.legalActions());
      selectionBuffer.add(decision.id(), action);
    }
    return selectionBuffer;
  }

  private void notifyRoundSettled(RoundSettlement settlement) {
    for (Player player : players) {
      player.onRoundSettled(settlement);
    }
  }
}
