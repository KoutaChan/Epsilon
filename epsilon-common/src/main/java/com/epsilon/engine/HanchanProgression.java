package com.epsilon.engine;

import com.epsilon.core.GameState;

/** 局結果を点棒へ反映し、次局または半荘終了を決める。 */
public final class HanchanProgression {

  public static final int RIICHI_COST = 1000;

  public static final int SOUTH_END_KYOKU = 7;
  public static final int WEST_END_KYOKU = 11;
  public static final int TOP_THRESHOLD = 30000;

  private HanchanProgression() {}

  /** 単独和了の供託授与前点棒から、現在局で半荘が終了するかを返す。 */
  public static boolean shouldFinishAfterSingleWin(
      int kyoku, int oya, int winner, int score0, int score1, int score2, int score3) {
    return shouldFinish(kyoku, oya, winner == oya, true, score0, score1, score2, score3);
  }

  static RoundSettlement settle(GameState state, RoundResult result) {
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      state.addScore(player, result.pointDelta().get(player));
    }

    int availableSticks = state.getKyotakuCount();
    int nextKyotakuCount = result instanceof RoundResult.Winning ? 0 : availableSticks;
    RoundTransition transition = nextTransition(state, result, nextKyotakuCount);

    if (result instanceof RoundResult.Winning winning) {
      state.addScore(winning.riichiStickReceiver(), availableSticks * RIICHI_COST);
    }
    state.setKyotakuCount(nextKyotakuCount);

    if (transition instanceof RoundTransition.HanchanFinished && nextKyotakuCount > 0) {
      state.addScore(topPlayerIndex(state), nextKyotakuCount * RIICHI_COST);
      state.setKyotakuCount(0);
    }

    return new RoundSettlement(
        result,
        state.getScore(0),
        state.getScore(1),
        state.getScore(2),
        state.getScore(3),
        transition);
  }

  private static RoundTransition nextTransition(
      GameState state, RoundResult result, int nextKyotakuCount) {
    int kyoku = state.getKyokuIndex();
    int oya = state.getOya();
    boolean dealerContinues = oyaStays(result, oya);

    if (shouldFinishHanchan(state, result, dealerContinues)) {
      return RoundTransition.HanchanFinished.INSTANCE;
    }

    int nextKyoku = dealerContinues ? kyoku : kyoku + 1;
    int nextOya = dealerContinues ? oya : (oya + 1) % GameState.NUM_PLAYERS;
    return new RoundTransition.NextRound(
        nextKyoku, nextOya, nextHonba(result, oya, state.getHonba()), nextKyotakuCount);
  }

  private static boolean shouldFinishHanchan(
      GameState state, RoundResult result, boolean dealerContinues) {
    boolean dealerCanEnd = !(result instanceof RoundResult.AbortiveDraw);
    return shouldFinish(
        state.getKyokuIndex(),
        state.getOya(),
        dealerContinues,
        dealerCanEnd,
        state.getScore(0),
        state.getScore(1),
        state.getScore(2),
        state.getScore(3));
  }

  private static boolean shouldFinish(
      int kyoku,
      int oya,
      boolean dealerContinues,
      boolean dealerCanEnd,
      int score0,
      int score1,
      int score2,
      int score3) {
    if (score0 < 0 || score1 < 0 || score2 < 0 || score3 < 0) return true;
    if (kyoku >= WEST_END_KYOKU) return true;
    if (kyoku < SOUTH_END_KYOKU) return false;
    int topScore = Math.max(Math.max(score0, score1), Math.max(score2, score3));
    if (!dealerContinues) return topScore >= TOP_THRESHOLD;
    return dealerCanEnd
        && score(oya, score0, score1, score2, score3) >= TOP_THRESHOLD
        && topPlayerIndex(score0, score1, score2, score3) == oya;
  }

  private static int score(int player, int score0, int score1, int score2, int score3) {
    return switch (player) {
      case 0 -> score0;
      case 1 -> score1;
      case 2 -> score2;
      case 3 -> score3;
      default -> throw new IndexOutOfBoundsException(player);
    };
  }

  private static boolean oyaStays(RoundResult result, int oya) {
    return switch (result) {
      case RoundResult.AbortiveDraw ignored -> true;
      case RoundResult.ExhaustiveDraw draw -> draw.isTenpai(oya);
      case RoundResult.TsumoAgari tsumo -> tsumo.winner() == oya;
      case RoundResult.RonAgari ron -> ron.hasWinner(oya);
    };
  }

  private static int nextHonba(RoundResult result, int oya, int currentHonba) {
    return switch (result) {
      case RoundResult.AbortiveDraw ignored -> currentHonba + 1;
      case RoundResult.ExhaustiveDraw ignored -> currentHonba + 1;
      case RoundResult.TsumoAgari tsumo -> tsumo.winner() == oya ? currentHonba + 1 : 0;
      case RoundResult.RonAgari ron -> ron.hasWinner(oya) ? currentHonba + 1 : 0;
    };
  }

  private static int topPlayerIndex(GameState state) {
    return topPlayerIndex(
        state.getScore(0), state.getScore(1), state.getScore(2), state.getScore(3));
  }

  private static int topPlayerIndex(int score0, int score1, int score2, int score3) {
    int topPlayer = 0;
    int topScore = score0;
    if (score1 > topScore) {
      topPlayer = 1;
      topScore = score1;
    }
    if (score2 > topScore) {
      topPlayer = 2;
      topScore = score2;
    }
    if (score3 > topScore) {
      topPlayer = 3;
    }
    return topPlayer;
  }
}
