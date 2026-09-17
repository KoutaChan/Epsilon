package com.epsilon.engine;

import com.epsilon.core.GameState;

/** 局結果を点棒へ反映し、次局または半荘終了を決める。 */
final class HanchanProgression {

  static final int RIICHI_COST = 1000;

  private static final int SOUTH_END_KYOKU = 7;
  private static final int WEST_END_KYOKU = 11;
  private static final int TOP_THRESHOLD = 30000;

  private HanchanProgression() {}

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
    int kyoku = state.getKyokuIndex();
    int oya = state.getOya();

    if (hasBustedPlayer(state)) {
      return true;
    }

    if (kyoku >= WEST_END_KYOKU) {
      return true;
    }

    if (kyoku < SOUTH_END_KYOKU) {
      return false;
    }

    if (!dealerContinues) {
      return topScore(state) >= TOP_THRESHOLD;
    }

    boolean dealerCanEnd = !(result instanceof RoundResult.AbortiveDraw);
    return dealerCanEnd && state.getScore(oya) >= TOP_THRESHOLD && topPlayerIndex(state) == oya;
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

  private static boolean hasBustedPlayer(GameState state) {
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      if (state.getScore(player) < 0) {
        return true;
      }
    }
    return false;
  }

  private static int topScore(GameState state) {
    int top = Integer.MIN_VALUE;
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      top = Math.max(top, state.getScore(player));
    }
    return top;
  }

  private static int topPlayerIndex(GameState state) {
    int topPlayer = 0;
    int topScore = state.getScore(0);
    for (int player = 1; player < GameState.NUM_PLAYERS; player++) {
      if (state.getScore(player) > topScore) {
        topPlayer = player;
        topScore = state.getScore(player);
      }
    }
    return topPlayer;
  }
}
