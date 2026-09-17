package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.WinLegality.RonStatus;
import java.util.List;

/** 他家の打牌または槓宣言への応答行動を生成する。 */
final class ResponseActionGenerator {

  private ResponseActionGenerator() {}

  static RonStatus generateInto(
      List<Action> actions,
      GameState state,
      int player,
      TurnEvent.ResponseSource source,
      EngineDecisionPoint decision,
      HandScoreEvaluator scoreEvaluator,
      HandShapeAnalyzer shapeAnalyzer) {
    actions.clear();
    RonStatus ronStatus =
        decision == null
            ? WinLegality.ronStatus(state, player, source, scoreEvaluator, shapeAnalyzer)
            : WinLegality.evaluateRonInto(
                state,
                player,
                source,
                decision.mutableImmediateWinResult(),
                scoreEvaluator,
                shapeAnalyzer);
    if (decision != null) decision.setImmediateWinAvailable(ronStatus.canRon());
    if (ronStatus.canRon()) {
      actions.add(Action.ronAgari());
    }

    if (source instanceof TurnEvent.KanAttempt) {
      addPassWhenChoiceExists(actions);
      return ronStatus;
    }

    TurnEvent.Discard discard = (TurnEvent.Discard) source;
    if (state.isRiichi(player) || state.remainingWallTiles() <= 0) {
      addPassWhenChoiceExists(actions);
      return ronStatus;
    }

    Hand hand = state.hand(player);
    int discardedTileType = discard.tileType();
    int concealedTileCount = hand.concealedTileCount();
    int matchingTileCount = hand.count(discardedTileType);

    if (matchingTileCount >= 2
        && concealedTileCount >= 3
        && hasDiscardAfterPon(hand, discardedTileType)) {
      addPonActions(actions, hand, discardedTileType);
    }
    if (matchingTileCount >= 3 && state.canKan()) {
      actions.add(Action.daiminkan(discardedTileType));
    }
    if (state.getRelativePosition(player, discard.player()) == 3
        && Tile.isNumberTile(discardedTileType)
        && concealedTileCount >= 3) {
      addChiPatterns(actions, hand, discardedTileType);
    }

    addPassWhenChoiceExists(actions);
    return ronStatus;
  }

  private static void addPassWhenChoiceExists(List<Action> actions) {
    if (!actions.isEmpty()) {
      actions.add(Action.pass());
    }
  }

  private static void addPonActions(List<Action> actions, Hand hand, int tileType) {
    if (!hand.hasAkaTile(tileType)) {
      actions.add(Action.pon(tileType));
      return;
    }
    actions.add(Action.pon(tileType, true));
    if (hand.hasMultipleNonAkaTiles(tileType)) {
      actions.add(Action.pon(tileType, false));
    }
  }

  private static void addChiPatterns(List<Action> actions, Hand hand, int discardedTileType) {
    int discardedTileNumber = Tile.numberOf(discardedTileType);
    int suitBaseTileType = Tile.suitOf(discardedTileType) * 9;

    if (discardedTileNumber >= 2
        && hand.count(suitBaseTileType + discardedTileNumber - 2) > 0
        && hand.count(suitBaseTileType + discardedTileNumber - 1) > 0) {
      addChiPattern(actions, hand, suitBaseTileType + discardedTileNumber - 2, discardedTileType);
    }
    if (discardedTileNumber >= 1
        && discardedTileNumber <= 7
        && hand.count(suitBaseTileType + discardedTileNumber - 1) > 0
        && hand.count(suitBaseTileType + discardedTileNumber + 1) > 0) {
      addChiPattern(actions, hand, suitBaseTileType + discardedTileNumber - 1, discardedTileType);
    }
    if (discardedTileNumber <= 6
        && hand.count(suitBaseTileType + discardedTileNumber + 1) > 0
        && hand.count(suitBaseTileType + discardedTileNumber + 2) > 0) {
      addChiPattern(actions, hand, discardedTileType, discardedTileType);
    }
  }

  private static void addChiPattern(
      List<Action> actions, Hand hand, int sequenceBaseTileType, int calledTileType) {
    PostCallDahaiRestriction restriction =
        PostCallDahaiRestriction.afterChi(sequenceBaseTileType, calledTileType);
    if (hasDiscardAfterChi(hand, sequenceBaseTileType, calledTileType, restriction)) {
      addChiActions(actions, hand, sequenceBaseTileType, calledTileType);
    }
  }

  private static boolean hasDiscardAfterChi(
      Hand hand,
      int sequenceBaseTileType,
      int calledTileType,
      PostCallDahaiRestriction restriction) {
    long remainingTileTypeMask = hand.concealedTileTypeMask();
    for (int offset = 0; offset < 3; offset++) {
      int consumedTileType = sequenceBaseTileType + offset;
      if (consumedTileType != calledTileType && hand.count(consumedTileType) == 1) {
        remainingTileTypeMask &= ~(1L << consumedTileType);
      }
    }
    return (remainingTileTypeMask & ~restriction.forbiddenMask()) != 0L;
  }

  private static boolean hasDiscardAfterPon(Hand hand, int ponTileType) {
    return (hand.concealedTileTypeMask() & ~(1L << ponTileType)) != 0L;
  }

  private static void addChiActions(
      List<Action> actions, Hand hand, int sequenceBaseTileType, int calledTileType) {
    boolean canConsumeAkaFromHand = false;
    boolean mustConsumeAkaFromHand = false;
    for (int offset = 0; offset < 3; offset++) {
      int tileType = sequenceBaseTileType + offset;
      if (tileType != calledTileType && hand.hasAkaTile(tileType)) {
        canConsumeAkaFromHand = true;
        if (!hand.hasNonAkaTile(tileType)) {
          mustConsumeAkaFromHand = true;
        }
      }
    }
    if (!canConsumeAkaFromHand) {
      actions.add(Action.chiSequence(sequenceBaseTileType, calledTileType));
      return;
    }
    actions.add(Action.chiSequence(sequenceBaseTileType, calledTileType, true));
    if (!mustConsumeAkaFromHand) {
      actions.add(Action.chiSequence(sequenceBaseTileType, calledTileType, false));
    }
  }
}
