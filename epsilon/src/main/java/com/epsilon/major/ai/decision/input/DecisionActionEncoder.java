package com.epsilon.major.ai.decision.input;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionEffect;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.WinSettlementProjection;

/** 解析済みの行動候補を、行動特徴量の連続バッファへ書き込む。 */
final class DecisionActionEncoder {

  private DecisionActionEncoder() {}

  static void encode(
      EngineDecisionBuffer decision,
      int actionSlot,
      RiichiState resultingRiichiStatus,
      DecisionInputWriter writer) {
    PublicObservation state = decision.state();
    int player = decision.playerIndex();
    Action action = decision.action(actionSlot);
    ActionEffect.NextStep continuation = decision.continuation(actionSlot);
    WinSettlementProjection immediateWin = decision.immediateWin(actionSlot);
    writer.action(
        actionSlot, DecisionInputSchema.ActionInt.ID, DecisionFeatureCodec.actionId(action));
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.TYPE,
        DecisionFeatureCodec.actionType(action.type()));
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.GROUP,
        DecisionFeatureCodec.actionGroup(action.type().group()));
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.PRIMARY_TILE,
        Tile.isValidType(action.tileType()) ? action.tileType() + 1 : 0);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.TILE_SELECTION,
        action.tileSelection().ordinal() + 1);
    if (action.type() == Action.Type.CHI) {
      int sequenceBaseTileType = action.chiSequenceBaseTileType();
      writer.action(actionSlot, DecisionInputSchema.ActionInt.CHI_BASE, sequenceBaseTileType + 1);
      writer.action(
          actionSlot,
          DecisionInputSchema.ActionInt.CHI_CALLED_POSITION,
          action.tileType() - sequenceBaseTileType + 1);
    }
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.DISCARD_IDENTITY,
        action.discardIdentityIndex() + 1);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.RESULTING_MELD_AKA_SOURCE,
        action.type().createsMeld()
            ? decision.meldAkaSource(actionSlot).ordinal() + 1
            : DecisionInputSchema.PAD_ID);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.RESULTING_RIICHI_STATUS,
        categoryId(resultingRiichiStatus));
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.WIN_CONTEXT,
        categoryId(winContext(state, player, action)));
    if (immediateWin != null) {
      writer.action(
          actionSlot,
          DecisionInputSchema.ActionInt.URA_ELIGIBLE,
          immediateWin.uraEligible() ? 1 : 0);
      DecisionWinPointFactsEncoder.encodeAction(
          immediateWin.visibleScore(), immediateWin.paoApplies(), writer, actionSlot);
    }
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.STARTS_IPPATSU,
        action.type() == Action.Type.RIICHI_DAHAI ? 1 : 0);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.BREAKS_IPPATSU,
        action.type().interruptsIppatsu() ? 1 : 0);
    writer.action(
        actionSlot, DecisionInputSchema.ActionInt.FOLLOW_UP_KIND, continuation.ordinal() + 1);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionInt.TERMINAL,
        continuation == ActionEffect.NextStep.ROUND_COMPLETE ? 1 : 0);
  }

  private static DecisionInputSchema.WinConditions winContext(
      PublicObservation state, int player, Action action) {
    if (action.type() == Action.Type.TSUMO_AGARI) {
      if (state.isBeforeFirstDiscard(player) && !state.firstTurnCallOccurred()) {
        return state.seatWindTileType(player) == Tile.TON
            ? DecisionInputSchema.WinConditions.TENHOU
            : DecisionInputSchema.WinConditions.CHIIHOU;
      }
      if (state.turnEvent() instanceof TurnEvent.Draw draw && draw.isRinshanDraw()) {
        return DecisionInputSchema.WinConditions.RINSHAN;
      }
      return state.isWallExhausted()
          ? DecisionInputSchema.WinConditions.HAITEI
          : DecisionInputSchema.WinConditions.TSUMO;
    }
    if (action.type() == Action.Type.RON_AGARI) {
      if (state.turnEvent() instanceof TurnEvent.KanAttempt kanAttempt
          && kanAttempt.kanKind() == TurnEvent.KanKind.KAKAN) {
        return DecisionInputSchema.WinConditions.CHANKAN;
      }
      return state.isWallExhausted()
          ? DecisionInputSchema.WinConditions.HOUTEI
          : DecisionInputSchema.WinConditions.RON;
    }
    return DecisionInputSchema.WinConditions.NONE;
  }

  private static int categoryId(Enum<?> value) {
    return value.ordinal() + 1;
  }
}
