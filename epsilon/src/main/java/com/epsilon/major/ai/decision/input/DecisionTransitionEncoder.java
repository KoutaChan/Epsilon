package com.epsilon.major.ai.decision.input;

import com.epsilon.core.Action;
import com.epsilon.core.HandView;
import com.epsilon.core.Tile;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.HandAnalysisBuffer;

/** 解析済みの遷移を、遷移特徴量の連続バッファへ書き込む。 */
final class DecisionTransitionEncoder {

  private static final float SHANTEN_NORMALIZER = 9.0f;
  private static final float UKEIRE_COUNT_NORMALIZER =
      Tile.NUM_TILE_TYPES * (float) Tile.TILES_PER_TYPE;
  private static final float TILE_TYPE_COUNT_NORMALIZER = Tile.NUM_TILE_TYPES;
  private static final float DORA_COUNT_NORMALIZER = 20.0f;
  private static final float AKA_TILE_COUNT_NORMALIZER = 3.0f;

  private DecisionTransitionEncoder() {}

  static void encode(
      EngineDecisionBuffer decision,
      int actionSlot,
      int transitionSlot,
      DecisionInputSchema.ActionTransitionKind kind,
      DecisionInputSchema.DiscardContext discardContext,
      DecisionInputSchema.RonFuritenKind resultingRonFuriten,
      long calledIntoMeldTileMask,
      HandAnalysisBuffer hand,
      short[] tileCategories,
      DecisionInputWriter writer) {
    Action discard = decision.discardAction(actionSlot, transitionSlot);
    HandView resultingHand = decision.handAfterTransition(actionSlot, transitionSlot);
    int discardedTileType = discard == null ? -1 : discard.tileType();
    writer.transition(
        actionSlot, transitionSlot, DecisionInputSchema.ActionTransitionInt.KIND, categoryId(kind));
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.DISCARD_CONTEXT,
        categoryId(discardContext));
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.DISCARD_ACTION_ID,
        discard == null ? 0 : DecisionFeatureCodec.actionId(canonicalDiscard(discard)));
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.DISCARD_TILE,
        discard == null ? 0 : discard.tileType() + 1);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.TILE_SELECTION,
        discard == null ? 0 : discard.tileSelection().ordinal() + 1);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.RESULTING_DISCARD_FURITEN,
        hand.isDiscardFuriten(decision.ownRiverTileTypeMask(), discardedTileType) ? 1 : 0);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.RESULTING_RON_FURITEN_KIND,
        categoryId(resultingRonFuriten));
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.SPECIAL_HANDS_AVAILABLE,
        hand.specialHandShantenAvailable() ? 1 : 0);
    writer.transition(
        actionSlot, transitionSlot, DecisionInputSchema.ActionTransitionInt.PRESENT, 1);

    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_MIN_SHANTEN,
        normalizeShanten(hand.minimumShanten()));
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_NORMAL_SHANTEN,
        normalizeShanten(hand.standardShanten()));
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_CHIITOI_SHANTEN,
        hand.specialHandShantenAvailable() ? normalizeShanten(hand.chiitoitsuShanten()) : 0.0f);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_KOKUSHI_SHANTEN,
        hand.specialHandShantenAvailable() ? normalizeShanten(hand.kokushiShanten()) : 0.0f);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_UKEIRE_COUNT,
        hand.liveImprovingCopies() / UKEIRE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_UKEIRE_KINDS,
        hand.liveImprovingTileTypes() / TILE_TYPE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_SHAPE_WAIT_KINDS,
        Long.bitCount(hand.shapeWaitTileTypeMask()) / TILE_TYPE_COUNT_NORMALIZER);
    encodeDrawHorizon(hand, writer, actionSlot, transitionSlot);
    encodeYakuTenpaiFrontier(hand, writer, actionSlot, transitionSlot);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_DORA_COUNT,
        hand.doraCount() / DORA_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_AKA_TILE_COUNT,
        Integer.bitCount(hand.ownedAkaMask()) / AKA_TILE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_PRIMARY_TILE_UNSEEN_COPIES,
        unseenTileCopies(discardedTileType, decision) / (float) Tile.TILES_PER_TYPE);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.MENZEN,
        hand.menzen() ? 1.0f : 0.0f);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_MELD_COUNT,
        hand.meldCount() / (float) DecisionInputSchema.MAX_MELDS_PER_PLAYER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_CONCEALED_TILE_COUNT,
        resultingHand.concealedTileCount() / 14.0f);

    for (int tileType = 0; tileType < Tile.NUM_TILE_TYPES; tileType++) {
      int ukeireRemaining =
          hasTile(hand.improvingTileTypeMask(), tileType) ? decision.unseenCopies(tileType) : 0;
      tileCategories[tileType] =
          (short)
              DecisionFeatureCodec.transitionTile(
                  resultingHand.count(tileType),
                  ukeireRemaining,
                  hasTile(hand.shapeWaitTileTypeMask(), tileType),
                  hasTile(hand.ronWaitTileTypeMask(), tileType),
                  hasTile(hand.tsumoWaitTileTypeMask(), tileType),
                  tileType == discardedTileType,
                  hasTile(calledIntoMeldTileMask, tileType),
                  hasTile(hand.akaWinningTileAvailableMask(), tileType));
    }
    writer.transitionTiles(actionSlot, transitionSlot, tileCategories);
    for (int waitSlot = 0; waitSlot < hand.waitCount(); waitSlot++) {
      writer.waitTile(actionSlot, transitionSlot, waitSlot, hand.waitTileType(waitSlot) + 1);
      DecisionWinPointFactsEncoder.encodeWait(
          hand.ronYakuBits(waitSlot),
          hand.ronHanWithoutUra(waitSlot),
          hand.ronFu(waitSlot),
          hand.ronPaoApplies(waitSlot),
          DecisionInputSchema.WaitWinType.RON,
          writer,
          actionSlot,
          transitionSlot,
          waitSlot);
      DecisionWinPointFactsEncoder.encodeWait(
          hand.tsumoYakuBits(waitSlot),
          hand.tsumoHanWithoutUra(waitSlot),
          hand.tsumoFu(waitSlot),
          hand.tsumoPaoApplies(waitSlot),
          DecisionInputSchema.WaitWinType.TSUMO,
          writer,
          actionSlot,
          transitionSlot,
          waitSlot);
    }
  }

  private static void encodeDrawHorizon(
      HandAnalysisBuffer analysis,
      DecisionInputWriter writer,
      int actionSlot,
      int transitionSlot) {
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.CURRENT_UKEIRE_HIT_WITHIN_1_SELF_DRAW,
        analysis.improvingHitWithinOneDraw());
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.CURRENT_UKEIRE_HIT_WITHIN_2_SELF_DRAWS,
        analysis.improvingHitWithinTwoDraws());
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.CURRENT_UKEIRE_HIT_WITHIN_3_SELF_DRAWS,
        analysis.improvingHitWithinThreeDraws());
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_LIVE_SHAPE_WAIT_COPIES,
        analysis.liveShapeWaitCopies() / UKEIRE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_INTRINSIC_RON_YAKU_WAIT_KINDS,
        analysis.ronWaitTileTypesWithoutRiichi() / TILE_TYPE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_INTRINSIC_RON_YAKU_WAIT_COPIES,
        analysis.ronWaitCopiesWithoutRiichi() / UKEIRE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_KINDS,
        analysis.tsumoWaitTileTypesWithoutRiichi() / TILE_TYPE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_COPIES,
        analysis.tsumoWaitCopiesWithoutRiichi() / UKEIRE_COUNT_NORMALIZER);
  }

  private static void encodeYakuTenpaiFrontier(
      HandAnalysisBuffer analysis,
      DecisionInputWriter writer,
      int actionSlot,
      int transitionSlot) {
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat
            .NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_KINDS,
        Long.bitCount(analysis.furitenFreeRonFrontierTileTypeMask()) / TILE_TYPE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat
            .NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_COPIES,
        analysis.furitenFreeRonFrontierCopies() / UKEIRE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat
            .NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_KINDS,
        Long.bitCount(analysis.furitenOnlyRonFrontierTileTypeMask()) / TILE_TYPE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat
            .NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_COPIES,
        analysis.furitenOnlyRonFrontierCopies() / UKEIRE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_KINDS,
        Long.bitCount(analysis.tsumoFrontierTileTypeMask()) / TILE_TYPE_COUNT_NORMALIZER);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionFloat.NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_COPIES,
        analysis.tsumoFrontierCopies() / UKEIRE_COUNT_NORMALIZER);
  }

  private static Action canonicalDiscard(Action discard) {
    return discard.type() == Action.Type.RIICHI_DAHAI
        ? Action.dahai(discard.tileType(), discard.tileSelection())
        : discard;
  }

  private static int unseenTileCopies(int tileType, EngineDecisionBuffer decision) {
    return Tile.isValidType(tileType) ? decision.unseenCopies(tileType) : 0;
  }

  private static boolean hasTile(long mask, int tileType) {
    return (mask & (1L << tileType)) != 0L;
  }

  private static float normalizeShanten(int shanten) {
    return (shanten + 1.0f) / SHANTEN_NORMALIZER;
  }

  private static int categoryId(Enum<?> value) {
    return value.ordinal() + 1;
  }
}
