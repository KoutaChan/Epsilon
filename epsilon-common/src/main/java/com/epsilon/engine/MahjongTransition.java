package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Meld;
import com.epsilon.core.TurnEvent;

/** 実対局・シミュレーション・牌譜再生で共有する、局面の更新処理。 */
public final class MahjongTransition {

  private final GameState state;
  private final GameRecorder recorder;
  private final ActionEffect.GameHandMutationTarget handMutationTarget =
      new ActionEffect.GameHandMutationTarget();

  public MahjongTransition(GameState state) {
    this(state, null);
  }

  public MahjongTransition(GameState state, GameRecorder recorder) {
    this.state = state;
    this.recorder = recorder;
  }

  public TurnEvent.Draw drawFromWall(int player) {
    int physicalTileId = state.drawLiveWallTile();
    state.hand(player).addPhysicalTile(physicalTileId);
    TurnEvent.Draw draw =
        state.recordDraw(player, physicalTileId, TurnEvent.DrawSource.WALL, false);
    if (recorder != null) {
      recorder.recordDraw(player, draw.tileType(), draw.isAkaTile());
    }
    return draw;
  }

  public TurnEvent.Draw drawFromDeadWall(int player, boolean delayDoraRevealUntilDiscard) {
    revealPendingDora();
    if (!delayDoraRevealUntilDiscard) {
      revealDora();
    }
    int physicalTileId = state.drawRinshanTile();
    state.hand(player).addPhysicalTile(physicalTileId);
    TurnEvent.Draw draw =
        state.recordDraw(
            player, physicalTileId, TurnEvent.DrawSource.RINSHAN, delayDoraRevealUntilDiscard);
    if (recorder != null) {
      recorder.recordRinshanDraw(player, draw.tileType(), draw.isAkaTile());
    }
    return draw;
  }

  /** 牌譜の既知ツモを基準の牌山カーソルと手牌へ同時反映する。 */
  public TurnEvent.Draw reconstructDraw(int player, int physicalTileId, boolean rinshan) {
    if (rinshan) {
      if (!(state.getTurnEvent() instanceof TurnEvent.Draw draw && draw.isRinshanDraw())) {
        state.consumeReconstructionRinshanDraw();
      }
    } else {
      state.consumeReconstructionLiveWallDraw();
    }
    if (physicalTileId >= 0) {
      state.hand(player).addPhysicalTile(physicalTileId);
    }
    TurnEvent.Draw draw =
        state.recordDraw(
            player,
            physicalTileId,
            rinshan ? TurnEvent.DrawSource.RINSHAN : TurnEvent.DrawSource.WALL,
            false);
    state.setCurrentPlayer(player);
    return draw;
  }

  public TurnEvent.Discard discard(
      int player,
      int tileType,
      boolean riichi,
      Action.TileSelection tileSelection,
      boolean drewThisTurn) {
    Hand hand = state.hand(player);
    boolean discardedRedTile =
        ActionEffect.removeDiscardedTile(hand, tileType, tileSelection.usesAkaTileFromHand());
    boolean revealPendingDoraAfterDiscard = shouldRevealDeferredDora(state.getTurnEvent());
    TurnEvent.Discard discard =
        state.commitDiscard(
            player,
            tileType,
            state.getTurnNumber(),
            riichi,
            discardedRedTile,
            tileSelection.isTsumogiri(),
            state.totalRiverDahaiCount());
    if (recorder != null) {
      recorder.recordDahai(player, tileType, discardedRedTile, tileSelection.isTsumogiri(), riichi);
    }
    if (drewThisTurn) {
      state.recordNormalDahai();
    }
    if (revealPendingDoraAfterDiscard) {
      revealDora();
    }
    return discard;
  }

  /** 牌譜上の打牌を通常遷移と同じ手牌・河更新へ通す。ドラ公開イベントは牌譜側で別途適用する。 */
  public TurnEvent.Discard reconstructDiscard(int player, Action action, boolean drewThisTurn) {
    return reconstructDiscard(player, action, drewThisTurn, true);
  }

  /** 他家の非公開手牌を含まない牌譜の再生では、河だけを更新し、不明な手牌を推測で補わない。 */
  public TurnEvent.Discard reconstructDiscard(
      int player, Action action, boolean drewThisTurn, boolean concealedHandKnown) {
    boolean riichi = action.type() == Action.Type.RIICHI_DAHAI;
    boolean doubleRiichi = riichi && state.isFirstDraw(player) && !state.isFirstTurnCallOccurred();
    boolean discardedRedTile = action.tileSelection().usesAkaTileFromHand();
    if (concealedHandKnown) {
      discardedRedTile =
          ActionEffect.removeDiscardedTile(
              state.hand(player), action.tileType(), action.tileSelection().usesAkaTileFromHand());
    }
    TurnEvent.Discard discard =
        state.commitDiscard(
            player,
            action.tileType(),
            state.getTurnNumber(),
            riichi,
            discardedRedTile,
            action.tileSelection().isTsumogiri(),
            state.totalRiverDahaiCount());
    if (riichi) {
      state.setRiichi(player, true);
      state.setDoubleRiichi(player, doubleRiichi);
      state.setIppatsu(player, true);
    } else {
      state.setIppatsu(player, false);
    }
    if (drewThisTurn) {
      state.recordNormalDahai();
    }
    state.setCurrentPlayer(player);
    return discard;
  }

  /** 復号済みの牌譜の副露を、既知手牌（副露・暗槓を除く）の実牌消費と公開インデックスへ一括反映する。 */
  public void applyReconstructedMeld(
      int player, Meld meld, int[] consumedPhysicalTileIds, boolean concealedHandKnown) {
    Hand hand = state.hand(player);
    if (concealedHandKnown) {
      for (int physicalTileId : consumedPhysicalTileIds) {
        hand.removePhysicalTile(physicalTileId);
      }
    }
    if (meld.type() == Meld.Type.KAKAN) {
      state.replacePonWithKakan(
          player, hand.requirePon(meld.baseTileType()), meld, state.river(player).size());
    } else {
      state.addMeld(player, meld, state.river(player).size());
    }
    switch (meld.type()) {
      case CHI, PON, DAIMINKAN -> {
        int source = (player + meld.relativeSource().playerOffset()) % GameState.NUM_PLAYERS;
        state.markLastDiscardCalled(source);
        state.clearTurnEvent();
        state.clearAllIppatsu();
      }
      case ANKAN, KAKAN -> {}
    }
    state.setCurrentPlayer(player);
  }

  public TurnEvent.KanAttempt recordKanAttempt(
      int player, int tileType, boolean aka, TurnEvent.KanKind kind) {
    return state.recordKanAttempt(player, tileType, aka, kind);
  }

  public void declareRiichi(int player, Action action) {
    boolean doubleRiichi = state.isFirstDraw(player) && !state.isFirstTurnCallOccurred();
    discard(player, action.tileType(), true, action.tileSelection(), true);
    state.setRiichi(player, true);
    state.setDoubleRiichi(player, doubleRiichi);
    state.setIppatsu(player, true);
  }

  public void commitRiichiPayment(int player) {
    state.addScore(player, -HanchanProgression.RIICHI_COST);
    state.setKyotakuCount(state.getKyotakuCount() + 1);
    if (recorder != null) {
      recorder.recordRiichiAccepted(player);
    }
  }

  public void cancelRiichiDeclaration(int player) {
    state.setRiichi(player, false);
    state.setDoubleRiichi(player, false);
    state.setIppatsu(player, false);
  }

  public Meld completeAnkan(int player, int tileType) {
    Meld ankan = ActionEffect.applyAnkan(state, player, tileType, handMutationTarget);
    if (recorder != null) {
      recorder.recordAnkan(player, tileType, ankan.consumedHandTileIsAka());
    }
    return ankan;
  }

  public Meld declareKakan(int player, int tileType) {
    Meld kakan = ActionEffect.applyKakan(state, player, tileType, handMutationTarget);
    if (recorder != null) {
      recorder.recordKakan(player, tileType, kakan.addedTileIsAka());
    }
    return kakan;
  }

  public Meld completeDaiminkan(int player, TurnEvent.Discard discard) {
    Meld daiminkan = ActionEffect.applyDaiminkan(state, player, discard, handMutationTarget);
    int sourcePlayerOffset = state.getRelativePosition(player, discard.player());
    markCalled(discard, player);
    if (recorder != null) {
      recorder.recordDaiminkan(
          player,
          discard.tileType(),
          sourcePlayerOffset,
          discard.isAkaTile(),
          daiminkan.consumedHandTileIsAka());
    }
    return daiminkan;
  }

  public Meld applyPon(int player, TurnEvent.Discard discard, boolean consumeAkaFromHand) {
    Meld pon =
        ActionEffect.applyPon(state, player, discard, consumeAkaFromHand, handMutationTarget);
    int sourcePlayerOffset = state.getRelativePosition(player, discard.player());
    markCalled(discard, player);
    if (recorder != null) {
      recorder.recordPon(
          player,
          discard.tileType(),
          sourcePlayerOffset,
          discard.isAkaTile(),
          pon.consumedHandTileIsAka());
    }
    return pon;
  }

  public Meld applyChi(int player, TurnEvent.Discard discard, Action chiAction) {
    Meld chi = ActionEffect.applyChi(state, player, discard, chiAction, handMutationTarget);
    markCalled(discard, player);
    if (recorder != null) {
      recorder.recordChi(
          player,
          chiAction.chiTileTypes(),
          discard.tileType(),
          discard.isAkaTile(),
          chi.consumedHandTileIsAka());
    }
    return chi;
  }

  /** 牌譜再生で成立済み槓の未知嶺上ツモ境界へ進める。 */
  public TurnEvent.Draw beginReconstructionRinshan(int player) {
    state.consumeReconstructionRinshanDraw();
    state.clearAllIppatsu();
    state.setCurrentPlayer(player);
    return state.recordDraw(player, -1, TurnEvent.DrawSource.RINSHAN, false);
  }

  public void revealReconstructedDora(int indicatorTileType) {
    state.revealDoraIndicator(indicatorTileType);
  }

  private void markCalled(TurnEvent.Discard discard, int callingPlayer) {
    state.markLastDiscardCalled(discard.player());
    state.setCurrentPlayer(callingPlayer);
    state.clearTurnEvent();
    state.clearAllIppatsu();
  }

  private void revealPendingDora() {
    if (shouldRevealDeferredDora(state.getTurnEvent())) {
      revealDora();
    }
  }

  private void revealDora() {
    state.revealDoraIndicator();
    if (recorder != null) {
      int index = state.doraState().indicatorCount() - 1;
      recorder.recordNewDora(state.doraIndicatorPhysicalTileId(index));
    }
  }

  private static boolean shouldRevealDeferredDora(TurnEvent event) {
    return switch (event) {
      case TurnEvent.Draw draw -> draw.doraRevealPending();
      case TurnEvent.KanAttempt kan -> kan.doraRevealPending();
      default -> false;
    };
  }
}
