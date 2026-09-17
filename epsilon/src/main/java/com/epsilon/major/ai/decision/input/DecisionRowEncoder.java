package com.epsilon.major.ai.decision.input;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.GameState;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.River;
import com.epsilon.core.RoundPublicStateIndex;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.DecisionHandAnalysisBuffer;
import com.epsilon.engine.EngineDecisionBuffer;

/**
 * 判断するプレイヤーが観測できる局・プレイヤー・牌・河・面子の情報を、一行の入力へ符号化する。
 *
 * <p>生の履歴トークンに加えて、ネットワークが複数構成要素を暗黙に突き合わせなくても済むよう、初巡の継続、立直宣言位置、永久フリテン、相手別河枚数、立直者自身の河と立直後に全家が通した現物を決定論的に導出する。裏ドラ、牌山順、他家の副露に含まれない手牌は公開観測ではないため入力しない。
 */
final class DecisionRowEncoder {

  private DecisionRowEncoder() {}

  static void encodeDecisionRow(
      EngineDecisionBuffer decision,
      DecisionBoundaryContext boundaryContext,
      DecisionHostBatch batch,
      int row,
      DecisionFeatureEncoder.Scratch scratch) {
    batch.inputs().prepareTransitions(row, decision);
    DecisionInputWriter writer = batch.inputs().writer(row);
    PublicObservation state = decision.state();
    int player = decision.playerIndex();
    DecisionHandAnalysisBuffer currentHand = decision.analyzeCurrentHand();
    encodeStateFeatures(state, player, decision, currentHand, writer);
    writer.boundaryContext(boundaryContext);
    DecisionFeatureEncoder.encode(decision, writer, scratch);
  }

  static void encodeStateOnlyRow(EngineDecisionBuffer decision, DecisionHostBatch batch, int row) {
    batch.inputs().prepareTransitions(row, decision);
    DecisionInputWriter writer = batch.inputs().writer(row);
    encodeStateFeatures(
        decision.state(), decision.playerIndex(), decision, decision.analyzeCurrentHand(), writer);
    writer.boundaryContext(DecisionBoundaryContext.uniform());
  }

  private static void encodeStateFeatures(
      PublicObservation state,
      int player,
      EngineDecisionBuffer context,
      DecisionHandAnalysisBuffer currentHand,
      DecisionInputWriter writer) {
    RoundPublicStateIndex publicState = state.publicState();
    TurnEvent event = state.turnEvent();
    int currentPlayerRelativeSeat = relativeSeatOrSelf(state, player, state.currentPlayer());
    int sourcePlayerRelativeSeat = relativeSeatOfEventPlayer(state, player, event);

    writer.round(DecisionInputSchema.RoundInt.PLAYER_SEAT, player + 1);
    writer.round(
        DecisionInputSchema.RoundInt.CURRENT_PLAYER_RELATIVE_SEAT, currentPlayerRelativeSeat + 1);
    writer.round(
        DecisionInputSchema.RoundInt.SOURCE_PLAYER_RELATIVE_SEAT, sourcePlayerRelativeSeat + 1);
    writer.round(DecisionInputSchema.RoundInt.KYOKU_INDEX, state.roundIndex() + 1);
    writer.round(
        DecisionInputSchema.RoundInt.DEALER_RELATIVE_SEAT,
        state.relativePosition(player, state.dealer()) + 1);
    writer.round(DecisionInputSchema.RoundInt.BAKAZE, state.roundWindTileType() + 1);
    writer.round(DecisionInputSchema.RoundInt.JIKAZE, state.seatWindTileType(player) + 1);
    writer.round(DecisionInputSchema.RoundInt.HONBA, state.honba());
    writer.round(DecisionInputSchema.RoundInt.KYOTAKU, state.riichiSticks());
    writer.round(DecisionInputSchema.RoundInt.WALL_REMAINING, state.remainingWallTiles());
    writer.round(DecisionInputSchema.RoundInt.TURN_NUMBER, state.turnNumber());
    writer.round(DecisionInputSchema.RoundInt.TOTAL_KAN, publicState.totalKanCount());
    writer.round(
        DecisionInputSchema.RoundInt.DORA_INDICATOR_COUNT, state.doraState().indicatorCount());
    writer.round(
        DecisionInputSchema.RoundInt.ALL_LAST,
        state.roundIndex() >= GameState.HANCHAN_KYOKU_COUNT - 1 ? 1 : 0);
    writer.round(DecisionInputSchema.RoundInt.INITIAL_DISCARD_CYCLE, state.isFirstTurn() ? 1 : 0);
    writer.round(
        DecisionInputSchema.RoundInt.SELF_BEFORE_FIRST_DISCARD,
        state.isBeforeFirstDiscard(player) ? 1 : 0);
    writer.round(
        DecisionInputSchema.RoundInt.UNINTERRUPTED_FIRST_DRAW,
        state.isBeforeFirstDiscard(player) && !state.firstTurnCallOccurred() ? 1 : 0);
    writer.round(
        DecisionInputSchema.RoundInt.FIRST_TURN_CALL_OCCURRED,
        state.firstTurnCallOccurred() ? 1 : 0);
    writer.round(DecisionInputSchema.RoundInt.LAST_LIVE_TILE, state.isWallExhausted() ? 1 : 0);
    encodeEvent(state, player, event, writer);
    int selfScore = state.score(player);
    int highestScore = Integer.MIN_VALUE;
    int lowestScore = Integer.MAX_VALUE;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      highestScore = Math.max(highestScore, state.score(seat));
      lowestScore = Math.min(lowestScore, state.score(seat));
    }
    writer.round(DecisionInputSchema.RoundFloat.HONBA, state.honba() / 10.0f);
    writer.round(DecisionInputSchema.RoundFloat.KYOTAKU, state.riichiSticks() / 10.0f);
    writer.round(DecisionInputSchema.RoundFloat.WALL_REMAINING, state.remainingWallTiles() / 70.0f);
    writer.round(DecisionInputSchema.RoundFloat.TURN, state.turnNumber() / 18.0f);
    writer.round(DecisionInputSchema.RoundFloat.SELF_SCORE, selfScore / 100000.0f);
    writer.round(
        DecisionInputSchema.RoundFloat.SCORE_LEAD,
        (selfScore - highestOpponentScore(state, player)) / 100000.0f);
    writer.round(
        DecisionInputSchema.RoundFloat.SCORE_TO_FIRST, (selfScore - highestScore) / 100000.0f);
    writer.round(
        DecisionInputSchema.RoundFloat.SCORE_TO_FOURTH, (selfScore - lowestScore) / 100000.0f);

    encodePlayers(state, player, context, currentHand, writer);
    encodeTiles(state, player, context, currentHand, writer);
    encodeRivers(state, player, writer);
    encodeMelds(state, player, writer);
    DecisionPublicHistory.encode(state, player, writer);
  }

  private static void encodePlayers(
      PublicObservation state,
      int player,
      EngineDecisionBuffer context,
      DecisionHandAnalysisBuffer currentHand,
      DecisionInputWriter writer) {
    RoundPublicStateIndex publicState = state.publicState();
    int selfScore = state.score(player);
    long selfWaitTileTypeMask = currentHand.shapeWaitTileTypeMask();
    long selfRiverTileTypeMask = context.ownRiverTileTypeMask();
    for (int relativeSeat = 0; relativeSeat < GameState.NUM_PLAYERS; relativeSeat++) {
      int seat = (player + relativeSeat) % GameState.NUM_PLAYERS;
      HandView hand = state.hand(seat);
      River river = state.river(seat);
      int openMeldCount = openMeldCount(hand);
      writer.player(relativeSeat, DecisionInputSchema.PlayerInt.RELATIVE_SEAT, relativeSeat + 1);
      writer.player(relativeSeat, DecisionInputSchema.PlayerInt.ABSOLUTE_SEAT, seat + 1);
      writer.player(
          relativeSeat, DecisionInputSchema.PlayerInt.JIKAZE, state.seatWindTileType(seat) + 1);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerInt.RIICHI_STATUS,
          enumCategoryId(riichiStatus(state, seat)));
      writer.player(
          relativeSeat, DecisionInputSchema.PlayerInt.IPPATSU, state.isIppatsu(seat) ? 1 : 0);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerInt.BEFORE_FIRST_DISCARD,
          state.isBeforeFirstDiscard(seat) ? 1 : 0);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerInt.SELF_TEMPORARY_FURITEN,
          relativeSeat == 0 && state.isTemporaryFuriten(seat) ? 1 : 0);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerInt.SELF_PERMANENT_FURITEN,
          relativeSeat == 0 && (selfWaitTileTypeMask & selfRiverTileTypeMask) != 0L ? 1 : 0);
      writer.player(relativeSeat, DecisionInputSchema.PlayerInt.MENZEN, hand.isMenzen() ? 1 : 0);
      writer.player(relativeSeat, DecisionInputSchema.PlayerInt.RANK, rank(state, seat) + 1);
      writer.player(relativeSeat, DecisionInputSchema.PlayerInt.MELD_COUNT, hand.meldCount());
      writer.player(relativeSeat, DecisionInputSchema.PlayerInt.OPEN_MELD_COUNT, openMeldCount);
      writer.player(
          relativeSeat, DecisionInputSchema.PlayerInt.KAN_COUNT, publicState.kanCount(seat));
      writer.player(relativeSeat, DecisionInputSchema.PlayerInt.RIVER_COUNT, river.size());
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerInt.RIICHI_DECLARATION_INDEX,
          context.riichiDeclarationIndex(relativeSeat) + 1);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerInt.DISCARDS_AFTER_RIICHI,
          context.discardsAfterRiichi(relativeSeat));
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerInt.POST_RIICHI_TSUMOGIRI_COUNT,
          context.postRiichiTsumogiriCount(relativeSeat));

      writer.player(
          relativeSeat, DecisionInputSchema.PlayerFloat.SCORE, state.score(seat) / 100000.0f);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerFloat.SCORE_FROM_SELF,
          (state.score(seat) - selfScore) / 100000.0f);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerFloat.RIVER_PROGRESS,
          river.size() / (float) DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerFloat.OPEN_MELD_FRACTION,
          openMeldCount / (float) DecisionInputSchema.MAX_MELDS_PER_PLAYER);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerFloat.RIICHI_DECLARATION_PROGRESS,
          context.riichiDeclarationIndex(relativeSeat) < 0
              ? 0.0f
              : (context.riichiDeclarationIndex(relativeSeat) + 1.0f)
                  / DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER);
      writer.player(
          relativeSeat,
          DecisionInputSchema.PlayerFloat.DEALER,
          seat == state.dealer() ? 1.0f : 0.0f);
    }
  }

  private static void encodeTiles(
      PublicObservation state,
      int player,
      EngineDecisionBuffer context,
      DecisionHandAnalysisBuffer currentHand,
      DecisionInputWriter writer) {
    HandView selfHand = state.hand(player);
    long currentUkeireTileTypeMask = currentHand.improvingTileTypeMask();
    long currentWaitTileTypeMask = currentHand.shapeWaitTileTypeMask();
    long shimochaRiichiGenbutsuMask = context.riichiGenbutsuTileTypeMask(1);
    long toimenRiichiGenbutsuMask = context.riichiGenbutsuTileTypeMask(2);
    long kamichaRiichiGenbutsuMask = context.riichiGenbutsuTileTypeMask(3);
    for (int tileType = 0; tileType < Tile.NUM_TILE_TYPES; tileType++) {
      long tileTypeBit = 1L << tileType;
      int handCount = selfHand.count(tileType);
      int visibleCount = context.visibleTileCount(tileType);
      int doraMultiplicity = context.doraMultiplicity(tileType);
      int selfRiverCount = context.riverDiscardCount(0, tileType);
      int shimochaRiverCount = context.riverDiscardCount(1, tileType);
      int toimenRiverCount = context.riverDiscardCount(2, tileType);
      int kamichaRiverCount = context.riverDiscardCount(3, tileType);
      writer.tile(tileType, DecisionInputSchema.TileInt.TILE_TYPE, tileType + 1);
      writer.tile(tileType, DecisionInputSchema.TileInt.SELF_HAND_COUNT, handCount);
      writer.tile(
          tileType,
          DecisionInputSchema.TileInt.SELF_HAS_AKA,
          selfHand.hasAkaTile(tileType) ? 1 : 0);
      writer.tile(tileType, DecisionInputSchema.TileInt.VISIBLE_COUNT, visibleCount);
      writer.tile(
          tileType,
          DecisionInputSchema.TileInt.DORA_INDICATOR_MULTIPLICITY,
          context.doraIndicatorMultiplicity(tileType));
      writer.tile(tileType, DecisionInputSchema.TileInt.DORA_MULTIPLICITY, doraMultiplicity);
      writer.tile(tileType, DecisionInputSchema.TileInt.SELF_RIVER_COUNT, selfRiverCount);
      writer.tile(tileType, DecisionInputSchema.TileInt.SHIMOCHA_RIVER_COUNT, shimochaRiverCount);
      writer.tile(tileType, DecisionInputSchema.TileInt.TOIMEN_RIVER_COUNT, toimenRiverCount);
      writer.tile(tileType, DecisionInputSchema.TileInt.KAMICHA_RIVER_COUNT, kamichaRiverCount);
      writer.tile(
          tileType,
          DecisionInputSchema.TileInt.SELF_UKEIRE,
          (currentUkeireTileTypeMask & tileTypeBit) != 0L ? 1 : 0);
      writer.tile(
          tileType,
          DecisionInputSchema.TileInt.SELF_WAIT,
          (currentWaitTileTypeMask & tileTypeBit) != 0L ? 1 : 0);
      writer.tile(
          tileType,
          DecisionInputSchema.TileInt.SELF_DISCARDED_WAIT,
          (currentWaitTileTypeMask & tileTypeBit) != 0L && selfRiverCount > 0 ? 1 : 0);
      writer.tile(
          tileType,
          DecisionInputSchema.TileInt.SHIMOCHA_RIICHI_GENBUTSU,
          (shimochaRiichiGenbutsuMask & tileTypeBit) != 0L ? 1 : 0);
      writer.tile(
          tileType,
          DecisionInputSchema.TileInt.TOIMEN_RIICHI_GENBUTSU,
          (toimenRiichiGenbutsuMask & tileTypeBit) != 0L ? 1 : 0);
      writer.tile(
          tileType,
          DecisionInputSchema.TileInt.KAMICHA_RIICHI_GENBUTSU,
          (kamichaRiichiGenbutsuMask & tileTypeBit) != 0L ? 1 : 0);

      writer.tile(
          tileType,
          DecisionInputSchema.TileFloat.SELF_HAND_FRACTION,
          handCount / (float) Tile.TILES_PER_TYPE);
      writer.tile(
          tileType,
          DecisionInputSchema.TileFloat.VISIBLE_FRACTION,
          visibleCount / (float) Tile.TILES_PER_TYPE);
      writer.tile(
          tileType,
          DecisionInputSchema.TileFloat.UNSEEN_FRACTION,
          context.unseenCopies(tileType) / (float) Tile.TILES_PER_TYPE);
      writer.tile(tileType, DecisionInputSchema.TileFloat.DORA_MULTIPLICITY, doraMultiplicity);
    }
  }

  private static void encodeRivers(
      PublicObservation state, int player, DecisionInputWriter writer) {
    for (int relativeSeat = 0; relativeSeat < GameState.NUM_PLAYERS; relativeSeat++) {
      River river = state.river((player + relativeSeat) % GameState.NUM_PLAYERS);
      for (int riverIndex = 0; riverIndex < river.size(); riverIndex++) {
        River.Discard discard = river.discard(riverIndex);
        writer.river(
            relativeSeat,
            riverIndex,
            DecisionInputSchema.RiverInt.RELATIVE_PLAYER,
            relativeSeat + 1);
        writer.river(relativeSeat, riverIndex, DecisionInputSchema.RiverInt.INDEX, riverIndex + 1);
        writer.river(
            relativeSeat, riverIndex, DecisionInputSchema.RiverInt.TILE, discard.tileType() + 1);
        writer.river(
            relativeSeat, riverIndex, DecisionInputSchema.RiverInt.IS_AKA, discard.aka() ? 1 : 0);
        writer.river(
            relativeSeat,
            riverIndex,
            DecisionInputSchema.RiverInt.TSUMOGIRI,
            discard.tsumogiri() ? 1 : 0);
        writer.river(
            relativeSeat,
            riverIndex,
            DecisionInputSchema.RiverInt.RIICHI,
            discard.riichiDeclaration() ? 1 : 0);
        writer.river(
            relativeSeat,
            riverIndex,
            DecisionInputSchema.RiverInt.CALLED,
            discard.called() ? 1 : 0);
        writer.river(
            relativeSeat, riverIndex, DecisionInputSchema.RiverInt.TURN, discard.turnNumber() + 1);
        writer.river(
            relativeSeat,
            riverIndex,
            DecisionInputSchema.RiverInt.GLOBAL_SEQUENCE,
            discard.sequence() + 1);
        writer.river(relativeSeat, riverIndex, DecisionInputSchema.RiverInt.PRESENT, 1);
        writer.river(
            relativeSeat,
            riverIndex,
            DecisionInputSchema.RiverFloat.PLAYER_PROGRESS,
            riverIndex / (float) DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER);
        writer.river(
            relativeSeat,
            riverIndex,
            DecisionInputSchema.RiverFloat.GLOBAL_PROGRESS,
            Math.max(0, discard.sequence()) / 70.0f);
      }
    }
  }

  private static void encodeMelds(PublicObservation state, int player, DecisionInputWriter writer) {
    for (int relativeSeat = 0; relativeSeat < GameState.NUM_PLAYERS; relativeSeat++) {
      HandView hand = state.hand((player + relativeSeat) % GameState.NUM_PLAYERS);
      for (int meldIndex = 0; meldIndex < hand.meldCount(); meldIndex++) {
        Meld meld = hand.meld(meldIndex);
        writer.meld(
            relativeSeat, meldIndex, DecisionInputSchema.MeldInt.RELATIVE_PLAYER, relativeSeat + 1);
        writer.meld(relativeSeat, meldIndex, DecisionInputSchema.MeldInt.INDEX, meldIndex + 1);
        writer.meld(
            relativeSeat, meldIndex, DecisionInputSchema.MeldInt.TYPE, meld.type().ordinal() + 1);
        writer.meld(
            relativeSeat,
            meldIndex,
            DecisionInputSchema.MeldInt.BASE_TILE,
            meld.baseTileType() + 1);
        writer.meld(
            relativeSeat,
            meldIndex,
            DecisionInputSchema.MeldInt.CALLED_TILE,
            meld.calledTileType() + 1);
        writer.meld(
            relativeSeat,
            meldIndex,
            DecisionInputSchema.MeldInt.SOURCE,
            meld.relativeSource().ordinal() + 1);
        writer.meld(
            relativeSeat,
            meldIndex,
            DecisionInputSchema.MeldInt.AKA_SOURCE,
            meld.akaSource().ordinal() + 1);
        writer.meld(relativeSeat, meldIndex, DecisionInputSchema.MeldInt.SIZE, meld.size());
        writer.meld(
            relativeSeat,
            meldIndex,
            DecisionInputSchema.MeldInt.CALL_AFTER_RIVER,
            hand.meldCallAfterRiverIndex(meldIndex) + 1);
        writer.meld(
            relativeSeat,
            meldIndex,
            DecisionInputSchema.MeldInt.KAN_AFTER_RIVER,
            hand.meldKanAfterRiverIndex(meldIndex) + 1);
        writer.meld(relativeSeat, meldIndex, DecisionInputSchema.MeldInt.PRESENT, 1);
        writer.meld(
            relativeSeat,
            meldIndex,
            DecisionInputSchema.MeldFloat.PLAYER_PROGRESS,
            Math.max(0, hand.meldCallAfterRiverIndex(meldIndex))
                / (float) DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER);
        writer.meld(
            relativeSeat,
            meldIndex,
            DecisionInputSchema.MeldFloat.KAN_PROGRESS,
            Math.max(0, hand.meldKanAfterRiverIndex(meldIndex))
                / (float) DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER);
      }
    }
  }

  private static void encodeEvent(
      PublicObservation state, int player, TurnEvent event, DecisionInputWriter writer) {
    int eventTypeId = enumCategoryId(DecisionInputSchema.EventType.NONE);
    int eventKanKindId = 0;
    int eventPlayerRelativeSeatId = 0;
    int eventTileTypeId = 0;
    int eventAkaFlag = 0;
    int eventDrawSourceId = 0;
    int eventRinshanDrawFlag = 0;
    int eventDoraRevealPendingFlag = 0;
    if (event instanceof TurnEvent.Draw draw) {
      eventTypeId = enumCategoryId(DecisionInputSchema.EventType.DRAW);
      eventPlayerRelativeSeatId = state.relativePosition(player, draw.player()) + 1;
      eventTileTypeId = draw.tileType() + 1;
      eventAkaFlag = draw.isAkaTile() ? 1 : 0;
      eventDrawSourceId = enumCategoryId(draw.drawSource());
      eventRinshanDrawFlag = draw.isRinshanDraw() ? 1 : 0;
      eventDoraRevealPendingFlag = draw.doraRevealPending() ? 1 : 0;
    } else if (event instanceof TurnEvent.Discard discard) {
      eventTypeId = enumCategoryId(DecisionInputSchema.EventType.DISCARD);
      eventPlayerRelativeSeatId = state.relativePosition(player, discard.player()) + 1;
      eventTileTypeId = discard.tileType() + 1;
      eventAkaFlag = discard.isAkaTile() ? 1 : 0;
    } else if (event instanceof TurnEvent.KanAttempt kan) {
      eventTypeId = enumCategoryId(DecisionInputSchema.EventType.KAN_ATTEMPT);
      eventKanKindId = enumCategoryId(kan.kanKind());
      eventPlayerRelativeSeatId = state.relativePosition(player, kan.player()) + 1;
      eventTileTypeId = kan.tileType() + 1;
      eventAkaFlag = kan.isAkaTile() ? 1 : 0;
      eventDoraRevealPendingFlag = kan.doraRevealPending() ? 1 : 0;
    }
    writer.round(DecisionInputSchema.RoundInt.EVENT_TYPE, eventTypeId);
    writer.round(DecisionInputSchema.RoundInt.EVENT_KAN_KIND, eventKanKindId);
    writer.round(
        DecisionInputSchema.RoundInt.EVENT_PLAYER_RELATIVE_SEAT, eventPlayerRelativeSeatId);
    writer.round(DecisionInputSchema.RoundInt.EVENT_TILE, eventTileTypeId);
    writer.round(DecisionInputSchema.RoundInt.EVENT_IS_AKA, eventAkaFlag);
    writer.round(DecisionInputSchema.RoundInt.EVENT_DRAW_SOURCE, eventDrawSourceId);
    writer.round(DecisionInputSchema.RoundInt.EVENT_RINSHAN, eventRinshanDrawFlag);
    writer.round(DecisionInputSchema.RoundInt.EVENT_DEFERRED_DORA, eventDoraRevealPendingFlag);
  }

  private static int relativeSeatOfEventPlayer(
      PublicObservation state, int player, TurnEvent event) {
    return switch (event) {
      case TurnEvent.Draw draw -> state.relativePosition(player, draw.player());
      case TurnEvent.ResponseSource source -> state.relativePosition(player, source.player());
      case TurnEvent.None ignored -> 0;
    };
  }

  private static int relativeSeatOrSelf(PublicObservation state, int observerSeat, int targetSeat) {
    return targetSeat >= 0 ? state.relativePosition(observerSeat, targetSeat) : 0;
  }

  private static RiichiState riichiStatus(PublicObservation state, int player) {
    if (state.isDoubleRiichi(player)) {
      return RiichiState.DOUBLE_RIICHI;
    }
    return state.isRiichi(player) ? RiichiState.RIICHI : RiichiState.NONE;
  }

  private static int openMeldCount(HandView hand) {
    int count = 0;
    for (int meldIndex = 0; meldIndex < hand.meldCount(); meldIndex++) {
      Meld meld = hand.meld(meldIndex);
      if (!meld.preservesMenzen()) {
        count++;
      }
    }
    return count;
  }

  private static int enumCategoryId(Enum<?> value) {
    return value.ordinal() + 1;
  }

  private static int highestOpponentScore(PublicObservation state, int player) {
    int highestOpponentScore = Integer.MIN_VALUE;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      if (seat != player) {
        highestOpponentScore = Math.max(highestOpponentScore, state.score(seat));
      }
    }
    return highestOpponentScore;
  }

  private static int rank(PublicObservation state, int player) {
    int rank = 0;
    for (int other = 0; other < GameState.NUM_PLAYERS; other++) {
      rank +=
          state.score(other) > state.score(player)
                  || state.score(other) == state.score(player) && other < player
              ? 1
              : 0;
    }
    return rank;
  }
}
