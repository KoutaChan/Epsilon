package com.epsilon.pico.ai.decision.input;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.Action;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.Tile;
import com.epsilon.engine.ActionEffect;
import com.epsilon.engine.DecisionHandAnalysisBuffer;
import com.epsilon.engine.EngineDecisionBuffer;

/** 行動の適用結果、特徴量の計算、連続バッファへの書き込みを順に実行する。 */
final class DecisionFeatureEncoder {

  static final class Scratch {
    private final short[] tileCategories = new short[Tile.NUM_TILE_TYPES];
    private final DecisionActionRouteEncoder.Scratch actionRoutes =
        new DecisionActionRouteEncoder.Scratch();
  }

  private DecisionFeatureEncoder() {}

  static void encode(EngineDecisionBuffer decision, DecisionInputWriter writer, Scratch scratch) {
    PublicObservation state = decision.state();
    int player = decision.playerIndex();
    DecisionActionRouteEncoder.encode(decision.legalActions(), writer, scratch.actionRoutes);
    short[] tileCategories = scratch.tileCategories;
    boolean ronIsLegal = false;
    for (int actionSlot = 0; actionSlot < decision.actionCount(); actionSlot++) {
      ronIsLegal |= decision.action(actionSlot).type() == Action.Type.RON_AGARI;
    }

    for (int actionSlot = 0; actionSlot < decision.actionCount(); actionSlot++) {
      Action action = decision.action(actionSlot);
      RiichiState riichiStatus = resultingRiichiStatus(state, player, action);
      DecisionActionEncoder.encode(decision, actionSlot, riichiStatus, writer);
      DecisionInputSchema.RonFuritenKind ronFuriten =
          resultingRonFuritenKind(state, player, action, ronIsLegal);

      for (int transitionSlot = 0;
          transitionSlot < decision.transitionCount(actionSlot);
          transitionSlot++) {
        Action discard = decision.discardAction(actionSlot, transitionSlot);
        int discardedTileType = discard == null ? -1 : discard.tileType();
        long resultingRiverMask = decision.ownRiverTileTypeMask();
        if (Tile.isValidType(discardedTileType)) {
          resultingRiverMask |= 1L << discardedTileType;
        }
        DecisionInputSchema.DiscardContext discardContext =
            discardContext(state, player, action, discard);
        DecisionHandAnalysisBuffer analysis =
            decision.analyzeTransition(
                actionSlot, transitionSlot, riichiStatus, resultingRiverMask);
        DecisionTransitionEncoder.encode(
            decision,
            actionSlot,
            transitionSlot,
            transitionKind(decision.continuation(actionSlot), discard),
            discardContext,
            ronFuriten,
            calledIntoMeldTileMask(state, player, action, discardContext),
            analysis,
            tileCategories,
            writer);
      }
    }
  }

  private static DecisionInputSchema.ActionTransitionKind transitionKind(
      ActionEffect.NextStep continuation, Action discard) {
    if (discard != null) {
      return DecisionInputSchema.ActionTransitionKind.DISCARD;
    }
    return switch (continuation) {
      case DECISION_COMPLETE -> DecisionInputSchema.ActionTransitionKind.IDENTITY;
      case RINSHAN_DRAW -> DecisionInputSchema.ActionTransitionKind.RINSHAN_PENDING;
      case ROUND_COMPLETE -> DecisionInputSchema.ActionTransitionKind.TERMINAL;
      case IMMEDIATE_DISCARD -> throw new AssertionError("discard transition missing");
    };
  }

  private static DecisionInputSchema.DiscardContext discardContext(
      PublicObservation state, int player, Action rootAction, Action discard) {
    if (discard == null) {
      return DecisionInputSchema.DiscardContext.NONE;
    }
    return switch (rootAction.type()) {
      case CHI -> DecisionInputSchema.DiscardContext.AFTER_CHI;
      case PON -> DecisionInputSchema.DiscardContext.AFTER_PON;
      case DAHAI, RIICHI_DAHAI -> currentDiscardContext(state, player);
      default -> throw new AssertionError("unexpected discard root: " + rootAction.type());
    };
  }

  private static DecisionInputSchema.RonFuritenKind resultingRonFuritenKind(
      PublicObservation state, int player, Action action, boolean ronIsLegal) {
    if (action.type() == Action.Type.RON_AGARI) {
      return DecisionInputSchema.RonFuritenKind.NONE;
    }
    if (ronIsLegal || state.isTemporaryFuriten(player)) {
      return state.isRiichi(player)
          ? DecisionInputSchema.RonFuritenKind.RIICHI_PERSISTENT
          : DecisionInputSchema.RonFuritenKind.TEMPORARY;
    }
    return DecisionInputSchema.RonFuritenKind.NONE;
  }

  private static long calledIntoMeldTileMask(
      PublicObservation state,
      int player,
      Action action,
      DecisionInputSchema.DiscardContext discardContext) {
    if (action.type() == Action.Type.CHI) {
      int sequenceBaseTileType = action.chiSequenceBaseTileType();
      return 0b111L << sequenceBaseTileType;
    }
    if (action.type() == Action.Type.PON
        || action.type() == Action.Type.DAIMINKAN
        || action.type() == Action.Type.ANKAN
        || action.type() == Action.Type.KAKAN) {
      return 1L << action.tileType();
    }
    if (discardContext == DecisionInputSchema.DiscardContext.AFTER_CHI
        || discardContext == DecisionInputSchema.DiscardContext.AFTER_PON) {
      return pendingCallMeld(state, player).tileTypeMask();
    }
    return 0L;
  }

  private static DecisionInputSchema.DiscardContext currentDiscardContext(
      PublicObservation state, int player) {
    Meld pendingCall = pendingCallMeld(state, player);
    if (pendingCall == null) {
      return DecisionInputSchema.DiscardContext.TURN;
    }
    return pendingCall.type() == Meld.Type.CHI
        ? DecisionInputSchema.DiscardContext.AFTER_CHI
        : DecisionInputSchema.DiscardContext.AFTER_PON;
  }

  private static Meld pendingCallMeld(PublicObservation state, int player) {
    HandView hand = state.hand(player);
    int meldIndex = hand.meldCount() - 1;
    if (meldIndex < 0 || hand.meldCallAfterRiverIndex(meldIndex) != state.river(player).size()) {
      return null;
    }
    Meld meld = hand.meld(meldIndex);
    return meld.type() == Meld.Type.CHI || meld.type() == Meld.Type.PON ? meld : null;
  }

  private static RiichiState resultingRiichiStatus(
      PublicObservation state, int player, Action action) {
    if (action.type() == Action.Type.RIICHI_DAHAI) {
      return state.isBeforeFirstDiscard(player) && !state.firstTurnCallOccurred()
          ? RiichiState.DOUBLE_RIICHI
          : RiichiState.RIICHI;
    }
    if (state.isDoubleRiichi(player)) {
      return RiichiState.DOUBLE_RIICHI;
    }
    return state.isRiichi(player) ? RiichiState.RIICHI : RiichiState.NONE;
  }
}
