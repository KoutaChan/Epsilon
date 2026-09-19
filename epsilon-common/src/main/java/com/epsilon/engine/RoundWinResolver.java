package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.TurnEvent;

/** 判断境界で確定済みの和了結果を消費し、点棒移動だけを解決する。 */
final class RoundWinResolver {

  private final GameState state;

  RoundWinResolver(GameState state) {
    this.state = state;
  }

  RoundResult resolveTsumo(int player, HandScoreBuffer agari) {
    Hand hand = state.hand(player);
    WinClaim.Tsumo claim =
        WinSettlementCalculator.createTsumoClaim(
            player, state.getOya(), state.getHonba(), agari, hand, state.isRiichi(player));
    return new RoundResult.TsumoAgari(claim);
  }

  RoundResult resolveRons(
      int[] winnerBuffer,
      int winnerCount,
      TurnEvent.ResponseSource response,
      EngineDecisionBatch decisions) {
    if (winnerCount < 1 || winnerCount > 3) {
      throw new IllegalArgumentException("RON winner count out of range: " + winnerCount);
    }
    WinClaim.Ron first =
        resolveRon(
            winnerBuffer[0], response, state.getHonba(), winResult(decisions, winnerBuffer[0]));
    return switch (winnerCount) {
      case 1 -> new RoundResult.RonAgari(first);
      case 2 ->
          new RoundResult.RonAgari(
              first,
              resolveRon(winnerBuffer[1], response, 0, winResult(decisions, winnerBuffer[1])));
      case 3 ->
          new RoundResult.RonAgari(
              first,
              resolveRon(winnerBuffer[1], response, 0, winResult(decisions, winnerBuffer[1])),
              resolveRon(winnerBuffer[2], response, 0, winResult(decisions, winnerBuffer[2])));
      default -> throw new AssertionError(winnerCount);
    };
  }

  private WinClaim.Ron resolveRon(
      int winner, TurnEvent.ResponseSource response, int honba, HandScoreBuffer agari) {
    Hand hand = state.hand(winner);
    return WinSettlementCalculator.createRonClaim(
        winner, response.player(), state.getOya(), honba, agari, hand, state.isRiichi(winner));
  }

  private static HandScoreBuffer winResult(EngineDecisionBatch decisions, int player) {
    for (int index = 0; index < decisions.size(); index++) {
      EngineDecisionPoint decision = decisions.get(index);
      if (decision.player() == player) return decision.requireImmediateWinResult();
    }
    throw new IllegalStateException("winning decision not found for player " + player);
  }
}
