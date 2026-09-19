package com.epsilon.engine;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.River;
import com.epsilon.core.Tile;

/** 1局内の流局判定と点棒移動を解決する。 */
final class RoundDrawResolver {
  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();

  private static final int TENPAI_PENALTY_TOTAL = 3000;

  private final GameState state;
  private final int[] scoreDelta = new int[GameState.NUM_PLAYERS];

  RoundDrawResolver(GameState state) {
    this.state = state;
  }

  RoundResult resolveExhaustiveDraw() {
    scoreDelta[0] = 0;
    scoreDelta[1] = 0;
    scoreDelta[2] = 0;
    scoreDelta[3] = 0;
    int tenpaiMask = tenpaiMask();
    int nagashiWinnerMask = applyNagashiMangan(scoreDelta);
    if (nagashiWinnerMask == 0) {
      applyTenpaiPayments(scoreDelta, tenpaiMask);
    }
    return new RoundResult.ExhaustiveDraw(
        tenpaiMask,
        nagashiWinnerMask,
        new PointDelta(scoreDelta[0], scoreDelta[1], scoreDelta[2], scoreDelta[3]));
  }

  RoundResult resolveAbortiveDraw(RoundResult.AbortiveDrawReason reason) {
    return new RoundResult.AbortiveDraw(reason);
  }

  RoundResult afterDiscard(boolean riichiDahai) {
    if (isSuufonRenda()) {
      return resolveAbortiveDraw(RoundResult.AbortiveDrawReason.FOUR_WINDS);
    }
    if (riichiDahai && countRiichiPlayers() == GameState.NUM_PLAYERS) {
      return resolveAbortiveDraw(RoundResult.AbortiveDrawReason.FOUR_RIICHI);
    }
    return null;
  }

  RoundResult afterKan(int lastKanPlayer) {
    if (state.publicState().totalKanCount() < 4
        || state.publicState().kanCount(lastKanPlayer) == 4) {
      return null;
    }
    return resolveAbortiveDraw(RoundResult.AbortiveDrawReason.FOUR_KANS);
  }

  private int applyNagashiMangan(int[] scoreDelta) {
    int winnerMask = 0;
    int oya = state.getOya();
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      if (!isNagashiMangan(player)) {
        continue;
      }
      winnerMask |= 1 << player;
      WinPayment.Tsumo payment =
          WinSettlementCalculator.calculateTsumoPayment(
              2000, player == oya ? WinnerRole.DEALER : WinnerRole.CHILD, 0);
      WinSettlementCalculator.addTsumoPayments(scoreDelta, player, oya, payment);
    }
    return winnerMask;
  }

  private int tenpaiMask() {
    int mask = 0;
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      Hand hand = state.hand(player);
      int shanten = shapes.calculateMinimum(hand);
      if (shanten <= 0) {
        mask |= 1 << player;
      }
    }
    return mask;
  }

  private static void applyTenpaiPayments(int[] scoreDelta, int tenpaiMask) {
    int tenpaiCount = Integer.bitCount(tenpaiMask);
    if (tenpaiCount == 0 || tenpaiCount == GameState.NUM_PLAYERS) {
      return;
    }
    int share = TENPAI_PENALTY_TOTAL / tenpaiCount;
    int penalty = TENPAI_PENALTY_TOTAL / (GameState.NUM_PLAYERS - tenpaiCount);
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      scoreDelta[player] = (tenpaiMask & (1 << player)) != 0 ? share : -penalty;
    }
  }

  private boolean isSuufonRenda() {
    if (state.isFirstTurnCallOccurred()) {
      return false;
    }
    int firstTile = -1;
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      River river = state.river(player);
      if (river.size() == 0) {
        return false;
      }
      int tile = river.discard(0).tileType();
      if (firstTile < 0) {
        firstTile = tile;
      } else if (firstTile != tile) {
        return false;
      }
    }
    return Tile.isWind(firstTile);
  }

  private boolean isNagashiMangan(int player) {
    River river = state.river(player);
    if (river.size() == 0) {
      return false;
    }
    for (int discardIndex = 0; discardIndex < river.size(); discardIndex++) {
      River.Discard discard = river.discard(discardIndex);
      if (discard.called() || !Tile.isTerminalOrHonor(discard.tileType())) {
        return false;
      }
    }
    return true;
  }

  private int countRiichiPlayers() {
    int count = 0;
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      if (state.isRiichi(player)) {
        count++;
      }
    }
    return count;
  }
}
