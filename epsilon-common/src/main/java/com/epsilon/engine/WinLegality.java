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

/**
 * 天鳳4人打ちの和了可否判定。
 *
 * <p>RON の前提（{@link #ronPrecondition}）は 1 本で、終わりの段が 2 つある。probe（{@link #ronStatus}）は役の有無だけを返し、score
 * （{@link #evaluateRonInto}）は点を判断点の箱へ書く。合法手の生成器はいつも score を通る。
 */
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

  /**
   * ツモ和了を採点して destination へ書く。和了の形でなければ役を見ずに偽を返す（destination は空）。
   *
   * @return 役のある和了なら {@code true}
   */
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

  /**
   * player が source でRONできるか（役・振聴・五枚目を含む {@link #ronStatus} が AVAILABLE）。診断用で、状態は変えない。評価器は呼び出し側の
   * ワーカーが所有する。
   */
  public static boolean canRon(
      GameState state,
      int player,
      TurnEvent.ResponseSource source,
      HandScoreEvaluator scores,
      HandShapeAnalyzer shapeAnalyzer) {
    return ronStatus(state, player, source, scores, shapeAnalyzer) == RonStatus.AVAILABLE;
  }

  /** RON の判定の probe の終わり: 前提（{@link #ronPrecondition}）の後、役の有無だけを調べる。点は求めない。 */
  static RonStatus ronStatus(
      GameState state,
      int player,
      TurnEvent.ResponseSource source,
      HandScoreEvaluator scores,
      HandShapeAnalyzer shapeAnalyzer) {
    RonStatus precondition = ronPrecondition(state, player, source, shapeAnalyzer);
    if (precondition != RonStatus.AVAILABLE) return precondition;
    boolean hasYaku =
        scores.probeAfterAdding(
                state.hand(player),
                source.tileType(),
                HandScoreEvaluator.RON,
                WinConditions.fromGameState(state, player, source))
            != 0;
    return hasYaku ? RonStatus.AVAILABLE : RonStatus.NO_YAKU;
  }

  /**
   * RON の判定の score の終わり: 前提（{@link #ronPrecondition}）の後、和了を採点して destination へ書く。前提で止まったときと役がないとき
   * destination は空。
   */
  static RonStatus evaluateRonInto(
      GameState state,
      int player,
      TurnEvent.ResponseSource source,
      HandScoreBuffer destination,
      HandScoreEvaluator scores,
      HandShapeAnalyzer shapeAnalyzer) {
    destination.clear();
    RonStatus precondition = ronPrecondition(state, player, source, shapeAnalyzer);
    if (precondition != RonStatus.AVAILABLE) return precondition;
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

  /**
   * RON の前提を、役を見ずに次の順で検査する（順は {@link RonStatus} の理由を決めるので変えない）。
   *
   * <ol>
   *   <li>暗槓への応答は UNAVAILABLE（国士の槍槓もこの規則では認めない）。
   *   <li>同巡・立直後の振聴なら FURITEN。
   *   <li>和了牌が自分で 4 枚持つ種類か、足しても和了の形にならないなら UNAVAILABLE。
   *   <li>和了の形の待ちのどれかを自分が捨てていれば FURITEN（河の牌種の mask を読む）。
   * </ol>
   *
   * <p>全てを通れば {@link RonStatus#AVAILABLE} を返す。これは「役の検査へ進む」の意味で、終わりの段（probe の {@link #ronStatus}、score の
   * {@link #evaluateRonInto}）が役を見て AVAILABLE か NO_YAKU に決める。
   *
   * <p>引ける牌種（{@link PossibleDrawsLookup#possibleDraws}。自分の手牌と副露で 4 枚そろった種類を除く）は 1 回の検査で 1 回だけ求め、
   * 3 と 4 で使い回す。3 は和了牌がその集合にないときに calculateAfterAdding の前で止まるので、calculateAfterAdding が手に 4 枚ある種類
   * （5 枚目）を受けることはない（受ければ例外を投げる）。
   */
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
    long possibleDraws = PossibleDrawsLookup.possibleDraws(hand, hand);
    if ((possibleDraws & (1L << winningTileType)) == 0L
        || shapeAnalyzer.calculateAfterAdding(hand, winningTileType) != -1) {
      return RonStatus.UNAVAILABLE;
    }
    long discardedWaits =
        shapeAnalyzer.agariTileTypeMask(hand)
            & possibleDraws
            & state.publicState().riverTileTypeMask(player);
    return discardedWaits != 0L ? RonStatus.FURITEN : RonStatus.AVAILABLE;
  }
}
