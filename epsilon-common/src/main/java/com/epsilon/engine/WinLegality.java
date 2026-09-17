package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.TurnEvent;

/** 天鳳4人打ちの和了可否判定。 */
public final class WinLegality {

  /** RON 不可の理由と、見逃し時のフリテン効果を区別した判定。 */
  public enum RonStatus {
    /** 和了形でない、またはルール上RON対象にならない。 */
    UNAVAILABLE,

    /** 現在すでに同巡・リーチ後・捨て牌フリテンである。 */
    FURITEN,

    /** 和了形だが役がなく、見逃し扱いにより同巡フリテンを生じる。 */
    NO_YAKU,

    /** RONが合法。 */
    AVAILABLE;

    /**
     * RON 行動を生成できる状態かを返す。
     *
     * @return 合法RONなら {@code true}
     */
    public boolean canRon() {
      return this == AVAILABLE;
    }

    /**
     * 応答せず進めた直後に同巡フリテンを設定すべき状態かを返す。
     *
     * @return 役なし和了形なら {@code true}
     */
    public boolean createsImmediateFuriten() {
      return this == NO_YAKU;
    }

    /**
     * 合法RONを見逃した場合にフリテンを設定すべきかを返す。
     *
     * @return RON可能状態なら {@code true}
     */
    public boolean createsFuritenWhenDeclined() {
      return this == AVAILABLE;
    }
  }

  private WinLegality() {}

  /** ワーカーが所有する評価器を使い、現在のツモに役があるか判定する。 */
  static boolean canTsumo(
      GameState state,
      int player,
      TurnEvent.Draw draw,
      HandScoreEvaluator scores,
      HandShapeAnalyzer shapeAnalyzer) {
    Hand hand = state.hand(player);
    if (shapeAnalyzer.calculateMinimum(hand) != -1) {
      return false;
    }
    return (scores.probeCompleted(
            hand,
            draw.tileType(),
            HandScoreEvaluator.TSUMO,
            WinConditions.fromGameState(state, player, draw))
        != 0);
  }

  static boolean evaluateTsumoInto(
      GameState state,
      int player,
      TurnEvent.Draw draw,
      HandScoreBuffer destination,
      HandScoreEvaluator scores,
      HandShapeAnalyzer shapeAnalyzer) {
    destination.clear();
    Hand hand = state.hand(player);
    if (shapeAnalyzer.calculateMinimum(hand) != -1) return false;
    return scores.scoreCompleted(
        hand,
        draw.tileType(),
        WinMethod.TSUMO,
        WinConditions.fromGameState(state, player, draw),
        state.doraState(),
        state.isRiichi(player),
        hand.ownedAkaMask(),
        destination);
  }

  /** ワーカーが所有する評価器を使い、RON可否と見逃し時の振聴効果を判定する。 */
  static RonStatus ronStatus(
      GameState state,
      int player,
      TurnEvent.ResponseSource source,
      HandScoreEvaluator scores,
      HandShapeAnalyzer shapeAnalyzer) {
    RonStatus precondition = ronPrecondition(state, player, source, shapeAnalyzer);
    if (precondition != null) return precondition;
    Hand hand = state.hand(player);
    boolean hasYaku =
        (scores.probeAfterAdding(
                hand,
                source.tileType(),
                HandScoreEvaluator.RON,
                WinConditions.fromGameState(state, player, source))
            != 0);
    return hasYaku ? RonStatus.AVAILABLE : RonStatus.NO_YAKU;
  }

  static RonStatus evaluateRonInto(
      GameState state,
      int player,
      TurnEvent.ResponseSource source,
      HandScoreBuffer destination,
      HandScoreEvaluator scores,
      HandShapeAnalyzer shapeAnalyzer) {
    destination.clear();
    RonStatus precondition = ronPrecondition(state, player, source, shapeAnalyzer);
    if (precondition != null) return precondition;
    Hand hand = state.hand(player);
    boolean hasYaku =
        scores.scoreAfterAdding(
            hand,
            source.tileType(),
            WinMethod.RON,
            WinConditions.fromGameState(state, player, source),
            state.doraState(),
            state.isRiichi(player),
            AkaTileMask.includeTileIfAka(
                hand.ownedAkaMask(), source.tileType(), source.isAkaTile()),
            destination);
    return hasYaku ? RonStatus.AVAILABLE : RonStatus.NO_YAKU;
  }

  private static RonStatus ronPrecondition(
      GameState state,
      int player,
      TurnEvent.ResponseSource source,
      HandShapeAnalyzer shapeAnalyzer) {
    if (source instanceof TurnEvent.KanAttempt kan && kan.kanKind() == TurnEvent.KanKind.ANKAN) {
      return RonStatus.UNAVAILABLE;
    }
    if (state.isTemporaryFuriten(player)) return RonStatus.FURITEN;
    Hand hand = state.hand(player);
    int winningTileType = source.tileType();
    if ((PossibleDrawsLookup.possibleDraws(hand, hand) & (1L << winningTileType)) == 0L
        || shapeAnalyzer.calculateAfterAdding(hand, winningTileType) != -1) {
      return RonStatus.UNAVAILABLE;
    }
    return hasDiscardedWinningWait(state, player, shapeAnalyzer) ? RonStatus.FURITEN : null;
  }

  private static boolean hasDiscardedWinningWait(
      GameState state, int player, HandShapeAnalyzer shapeAnalyzer) {
    Hand hand = state.hand(player);
    return (shapeAnalyzer.agariTileTypeMask(hand)
            & PossibleDrawsLookup.possibleDraws(hand, hand)
            & state.publicState().riverTileTypeMask(player))
        != 0L;
  }
}
