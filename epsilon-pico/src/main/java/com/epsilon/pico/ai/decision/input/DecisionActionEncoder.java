package com.epsilon.pico.ai.decision.input;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionEffect;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.VisibleHandScoreBuffer;
import com.epsilon.engine.WinSettlementProjection;

/** 候補となる行動の適用結果を、行動入力用の連続バッファへ書き込む。 */
final class DecisionActionEncoder {

  private static final int RIICHI_DECLARATION_POINTS = 1000;
  private static final float SCORE_NORMALIZER = 100000.0f;
  private static final float HAN_NORMALIZER = 13.0f;
  private static final float FU_NORMALIZER = 110.0f;
  private static final float BASE_POINTS_NORMALIZER = 32000.0f;
  private static final float DORA_INDICATOR_COUNT_NORMALIZER = 5.0f;
  private static final DecisionInputSchema.ActionFloat[] ACTION_FLOATS =
      DecisionInputSchema.ActionFloat.values();

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
      writer.action(
          actionSlot,
          DecisionInputSchema.ActionInt.SETTLEMENT_ASSUMPTION,
          categoryId(immediateWin.settlementAssumption()));
      writer.action(
          actionSlot,
          DecisionInputSchema.ActionInt.SOLE_WIN_PROJECTED_RANK,
          immediateWin.projectedRank(player) + 1);
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

    int declarationCost = action.type() == Action.Type.RIICHI_DAHAI ? RIICHI_DECLARATION_POINTS : 0;
    int scoreAfterDeclaration = state.score(player) - declarationCost;
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionFloat.RIICHI_DECLARATION_COST,
        declarationCost / SCORE_NORMALIZER);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionFloat.SELF_SCORE_AFTER_DECLARATION,
        scoreAfterDeclaration / SCORE_NORMALIZER);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionFloat.SCORE_TO_FIRST_AFTER_DECLARATION,
        (scoreAfterDeclaration
                - extremeScoreAfterDeclaration(state, player, scoreAfterDeclaration, true))
            / SCORE_NORMALIZER);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionFloat.SCORE_TO_FOURTH_AFTER_DECLARATION,
        (scoreAfterDeclaration
                - extremeScoreAfterDeclaration(state, player, scoreAfterDeclaration, false))
            / SCORE_NORMALIZER);
    if (immediateWin != null) {
      encodeImmediateWin(immediateWin, player, writer, actionSlot);
    }
  }

  private static void encodeImmediateWin(
      WinSettlementProjection immediateWin,
      int player,
      DecisionInputWriter writer,
      int actionSlot) {
    VisibleHandScoreBuffer agari = immediateWin.visibleScore();
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionFloat.NORMALIZED_VISIBLE_HAN_WITHOUT_URA,
        agari.visibleHan() / HAN_NORMALIZER);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionFloat.NORMALIZED_VISIBLE_FU,
        agari.fu() / FU_NORMALIZER);
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionFloat.NORMALIZED_VISIBLE_BASE_POINTS,
        agari.basePoints() / BASE_POINTS_NORMALIZER);
    for (int relativeSeat = 0; relativeSeat < GameState.NUM_PLAYERS; relativeSeat++) {
      int absoluteSeat = (player + relativeSeat) % GameState.NUM_PLAYERS;
      writer.action(
          actionSlot,
          ACTION_FLOATS[
              DecisionInputSchema.ActionFloat.NORMALIZED_PAYMENT_FLOOR_SELF.ordinal()
                  + relativeSeat],
          immediateWin.paymentFloor(absoluteSeat) / SCORE_NORMALIZER);
    }
    writer.action(
        actionSlot,
        DecisionInputSchema.ActionFloat.NORMALIZED_URA_INDICATOR_COUNT,
        immediateWin.uraIndicatorCount() / DORA_INDICATOR_COUNT_NORMALIZER);
    for (int relativeSeat = 0; relativeSeat < GameState.NUM_PLAYERS; relativeSeat++) {
      int absoluteSeat = (player + relativeSeat) % GameState.NUM_PLAYERS;
      writer.action(
          actionSlot,
          ACTION_FLOATS[
              DecisionInputSchema.ActionFloat.NORMALIZED_SOLE_WIN_SCORE_GAP_SELF.ordinal()
                  + relativeSeat],
          immediateWin.scoreGap(absoluteSeat) / SCORE_NORMALIZER);
    }
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

  private static int extremeScoreAfterDeclaration(
      PublicObservation state, int player, int selfScoreAfterDeclaration, boolean maximum) {
    int extreme = maximum ? Integer.MIN_VALUE : Integer.MAX_VALUE;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      int score = seat == player ? selfScoreAfterDeclaration : state.score(seat);
      extreme = maximum ? Math.max(extreme, score) : Math.min(extreme, score);
    }
    return extreme;
  }

  private static int categoryId(Enum<?> value) {
    return value.ordinal() + 1;
  }
}
