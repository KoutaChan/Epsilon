package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import java.util.List;

/** ツモ後の手番アクションを生成する。 */
final class TurnActionGenerator {

  private TurnActionGenerator() {}

  static void generateInto(
      List<Action> actions,
      GameState state,
      int player,
      TurnEvent.Draw draw,
      EngineDecisionPoint decision,
      HandScoreEvaluator scoreEvaluator,
      HandShapeAnalyzer shapeAnalyzer) {
    actions.clear();
    Hand hand = state.hand(player);

    boolean canTsumo =
        decision == null
            ? WinLegality.canTsumo(state, player, draw, scoreEvaluator, shapeAnalyzer)
            : WinLegality.evaluateTsumoInto(
                state,
                player,
                draw,
                decision.mutableImmediateWinResult(),
                scoreEvaluator,
                shapeAnalyzer);
    if (decision != null) decision.setImmediateWinAvailable(canTsumo);
    if (canTsumo) {
      actions.add(Action.tsumoAgari());
    }

    if (state.isRiichi(player)) {
      addRiichiActions(actions, state, hand, draw, shapeAnalyzer);
      return;
    }

    if (state.canKan()) {
      addKanActions(actions, hand);
    }
    if (hand.isMenzen() && state.getScore(player) >= 1000 && state.remainingWallTiles() >= 4) {
      addRiichiDeclarationActions(actions, hand, draw, shapeAnalyzer);
    }
    addDahaiActions(actions, hand, draw);
    addKyushuKyuhai(actions, state, player, hand);
  }

  private static void addKanActions(List<Action> actions, Hand hand) {
    for (long remaining = hand.concealedTileTypeMask();
        remaining != 0L;
        remaining &= remaining - 1) {
      int tileType = Long.numberOfTrailingZeros(remaining);
      if (hand.count(tileType) == Tile.TILES_PER_TYPE) {
        actions.add(Action.ankan(tileType));
      }
    }
    for (int meldIndex = 0; meldIndex < hand.meldCount(); meldIndex++) {
      Meld meld = hand.meld(meldIndex);
      if (meld.type() == Meld.Type.PON && hand.count(meld.baseTileType()) > 0) {
        actions.add(Action.kakan(meld.baseTileType()));
      }
    }
  }

  private static void addKyushuKyuhai(
      List<Action> actions, GameState state, int player, Hand hand) {
    if (!state.isFirstDraw(player) || hand.meldCount() != 0 || state.isFirstTurnCallOccurred()) {
      return;
    }
    int terminalOrHonorTileTypeCount =
        Long.bitCount(hand.concealedTileTypeMask() & Tile.TERMINAL_OR_HONOR_TYPE_MASK);
    if (terminalOrHonorTileTypeCount >= 9) {
      actions.add(Action.kyushuKyuhai());
    }
  }

  private static void addRiichiActions(
      List<Action> actions,
      GameState state,
      Hand hand,
      TurnEvent.Draw draw,
      HandShapeAnalyzer shapeAnalyzer) {
    int drawnTileType = draw.tileType();
    if (drawnTileType >= 0
        && hand.count(drawnTileType) == Tile.TILES_PER_TYPE
        && state.canKan()
        && canRiichiAnkanPreserveWait(hand, drawnTileType, shapeAnalyzer)) {
      actions.add(Action.ankan(drawnTileType));
    }
    if (drawnTileType >= 0) {
      actions.add(Action.dahai(drawnTileType, draw.isAkaTile(), true));
    }
  }

  private static boolean canRiichiAnkanPreserveWait(
      Hand hand, int kanTileType, HandShapeAnalyzer shapeAnalyzer) {
    int meldCount = hand.meldCount();
    long waitMaskBeforeKan =
        shapeAnalyzer.agariTileTypeMaskAfterRemoving(hand, kanTileType, 1, meldCount);
    long waitMaskAfterKan =
        shapeAnalyzer.agariTileTypeMaskAfterRemoving(
            hand, kanTileType, Tile.TILES_PER_TYPE, meldCount + 1);
    return waitMaskBeforeKan == waitMaskAfterKan;
  }

  static void addRiichiDeclarationActions(
      List<Action> actions, Hand hand, TurnEvent.Draw draw, HandShapeAnalyzer shapeAnalyzer) {
    int drawnTileType = draw.tileType();
    boolean drawnTileIsAka = draw.isAkaTile();

    for (long remaining = hand.concealedTileTypeMask();
        remaining != 0L;
        remaining &= remaining - 1) {
      int discardTileType = Long.numberOfTrailingZeros(remaining);
      if (shapeAnalyzer.calculateMinimumAfterRemoving(hand, discardTileType) == 0) {
        addDahaiVariants(actions, hand, discardTileType, drawnTileType, drawnTileIsAka, true);
      }
    }
  }

  private static void addDahaiActions(List<Action> actions, Hand hand, TurnEvent.Draw draw) {
    int drawnTileType = draw.tileType();
    boolean drawnTileIsAka = draw.isAkaTile();
    for (long remaining = hand.concealedTileTypeMask();
        remaining != 0L;
        remaining &= remaining - 1) {
      int discardTileType = Long.numberOfTrailingZeros(remaining);
      addDahaiVariants(actions, hand, discardTileType, drawnTileType, drawnTileIsAka, false);
    }
  }

  private static void addDahaiVariants(
      List<Action> actions,
      Hand hand,
      int discardTileType,
      int drawnTileType,
      boolean drawnTileIsAka,
      boolean riichi) {
    boolean isDrawnTileType = discardTileType == drawnTileType;
    boolean canDiscardAkaFromHand =
        hand.hasAkaTile(discardTileType) && (!isDrawnTileType || !drawnTileIsAka);
    boolean canDiscardNonAkaFromHand =
        hand.hasNonAkaTile(discardTileType)
            && (!isDrawnTileType || drawnTileIsAka || hand.hasMultipleNonAkaTiles(discardTileType));

    if (canDiscardAkaFromHand) {
      actions.add(createDiscardAction(discardTileType, true, false, riichi));
    }
    if (canDiscardNonAkaFromHand) {
      actions.add(createDiscardAction(discardTileType, false, false, riichi));
    }
    if (isDrawnTileType) {
      actions.add(createDiscardAction(discardTileType, drawnTileIsAka, true, riichi));
    }
  }

  private static Action createDiscardAction(
      int tileType, boolean selectAkaTile, boolean tsumogiri, boolean riichi) {
    return riichi
        ? Action.riichiDahai(tileType, selectAkaTile, tsumogiri)
        : Action.dahai(tileType, selectAkaTile, tsumogiri);
  }
}
