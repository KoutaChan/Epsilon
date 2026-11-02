package com.epsilon.prism.luck;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameRecorder;
import com.epsilon.engine.GameStepResult;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * 運の網の試験の道具: 種を決めた半荘をエンジンで打ち、判断の境界ごとに前の境界からの偶然の出来事を {@code EventLedger.Tracker.observe}
 * と同じ規則（表示は古い順、境界の直前のツモはその後。局の最初の境界では表示を数えない）で復元して {@link LuckSink} に渡す。
 *
 * <p>打ち方は和了・槓・立直を必ず選び、鳴きを半分の確率で選び、打牌は向聴が最小の手（同点は乱数）にする。槓が多いので、嶺上のツモ・
 * 暗槓の表示・打牌の後の遅れた表示・1 つの境界の 2 つの表示が出る。席別のツモ列は本番と同じく使う。
 */
final class LuckGames {

  /** 局と境界を受け取る。 */
  interface Visitor {

    /** 出来事の受け口。 */
    LuckSink sink();

    /** 局の最初の境界の前に呼ぶ。 */
    default void roundStarted(int game, int round) {}

    /** 境界の出来事を全て渡した後に呼ぶ。 */
    default void restored() {}

    /** 局の終わり（最後の境界の後に起きた出来事は復元しない）。 */
    default void roundEnded(int game, int round) {}

    /** 局の終わり（精算の後の状態つき）。既定は {@link #roundEnded(int, int)} に委ねる。 */
    default void roundEnded(int game, int round, GameState state) {
      roundEnded(game, round);
    }
  }

  private LuckGames() {}

  /**
   * 半荘を打つ。半荘 {@code g} の牌山と打ち方の乱数の種は {@code seed + g}。
   *
   * @param seed 最初の半荘の種
   * @param games 半荘の数
   * @param recorder エンジンの記録処理（なければ null）
   * @param visitor 受け取る先
   */
  static void play(long seed, int games, GameRecorder recorder, Visitor visitor) {
    HandShapeAnalyzer shapes = new HandShapeAnalyzer();
    for (int game = 0; game < games; game++) {
      GameEngine engine = new GameEngine(seed + game, recorder);
      engine.getState().enableSeatStreamDraws();
      SplittableRandom random = new SplittableRandom(~(seed + game));
      int round = 0;
      int last = -1;
      GameStepResult step = engine.stepHanchan();
      while (!(step instanceof GameStepResult.HanchanEnded)) {
        if (step instanceof GameStepResult.AwaitingDecisions awaiting) {
          List<EngineDecisionPoint> points = awaiting.decisions();
          GameState state = engine.getState();
          if (last < 0) {
            visitor.roundStarted(game, round);
          }
          last = restore(state, points, last, visitor.sink());
          visitor.restored();
          step = engine.commitDecisions(select(state, points, random, shapes));
        } else {
          visitor.roundEnded(game, round++, engine.getState());
          last = -1;
          step = engine.stepHanchan();
        }
      }
      visitor.roundEnded(game, round, engine.getState());
    }
  }

  /** 境界の出来事を復元して渡し、今の表示の数を返す（{@code EventLedger.Tracker.observe} と同じ規則）。 */
  private static int restore(
      GameState state, List<EngineDecisionPoint> points, int last, LuckSink sink) {
    int count = state.doraState().indicatorCount();
    int fresh = last < 0 ? 0 : count - last;
    if (fresh < 0) {
      throw new IllegalStateException("dora indicators went back to " + count);
    }
    TurnEvent.Draw draw = freshDraw(state, points);
    for (int indicator = count - fresh; indicator < count; indicator++) {
      sink.reveal(state, indicator, draw);
    }
    if (draw != null) {
      sink.draw(state, draw);
    }
    return count;
  }

  /** TURN の境界でツモの直後なら、そのツモ。ほかは null。 */
  private static TurnEvent.Draw freshDraw(GameState state, List<EngineDecisionPoint> points) {
    if (points.size() != 1 || points.getFirst().kind() != EngineDecisionKind.TURN) {
      return null;
    }
    return state.turnEvent() instanceof TurnEvent.Draw draw
            && draw.player() == points.getFirst().player()
            && draw.tileType() >= 0
        ? draw
        : null;
  }

  private static EngineSelectionBuffer select(
      GameState state,
      List<EngineDecisionPoint> points,
      SplittableRandom random,
      HandShapeAnalyzer shapes) {
    EngineSelectionBuffer selections = new EngineSelectionBuffer();
    for (EngineDecisionPoint point : points) {
      selections.add(point.id(), choose(state, point, random, shapes));
    }
    return selections;
  }

  private static Action choose(
      GameState state,
      EngineDecisionPoint point,
      SplittableRandom random,
      HandShapeAnalyzer shapes) {
    List<Action> legal = point.legalActions();
    if (legal.size() == 1) {
      return legal.getFirst();
    }
    Action preferred = null;
    int priority = 0;
    for (Action action : legal) {
      int rank =
          switch (action.type()) {
            case TSUMO_AGARI, RON_AGARI -> 3;
            case ANKAN, KAKAN, DAIMINKAN -> 2;
            case RIICHI_DAHAI -> 1;
            default -> 0;
          };
      if (rank > priority) {
        preferred = action;
        priority = rank;
      }
    }
    if (preferred != null) {
      return preferred;
    }
    Action pass = null;
    List<Action> calls = new ArrayList<>();
    for (Action action : legal) {
      switch (action.type()) {
        case PASS -> pass = action;
        case CHI, PON -> calls.add(action);
        default -> {}
      }
    }
    if (pass != null) {
      return !calls.isEmpty() && random.nextBoolean()
          ? calls.get(random.nextInt(calls.size()))
          : pass;
    }
    int best = Integer.MAX_VALUE;
    List<Action> ties = new ArrayList<>();
    for (Action action : legal) {
      if (action.type() != Action.Type.DAHAI) {
        continue;
      }
      int shanten =
          shapes.calculateMinimumAfterRemoving(state.hand(point.player()), action.tileType());
      if (shanten < best) {
        best = shanten;
        ties.clear();
      }
      if (shanten == best) {
        ties.add(action);
      }
    }
    if (!ties.isEmpty()) {
      return ties.get(random.nextInt(ties.size()));
    }
    for (Action action : legal) {
      if (action.type() != Action.Type.KYUSHU_KYUHAI) {
        return action;
      }
    }
    return legal.getFirst();
  }
}
