package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.core.Action;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.ScoreRanking;
import com.epsilon.core.TurnEvent;

/**
 * 現在合法なロン・ツモについて、裏ドラを含まない和了点と、所定の精算条件で計算した和了後の順位を保持する。
 *
 * <p>得点値は裏ドラを含まない。TSUMO は現在局面で精算が一意に決まり、RON
 * の順位は単独和了仮定で計算する。複数ロンでも本場を除いて計算できる和了点と、仮定を含む順位の予測を混同しないよう別フィールドで保持する。
 */
public final class WinSettlementProjection {

  /** 和了後順位を計算する際に置いた精算仮定。 */
  public enum SettlementAssumption {
    /** 現在の本場・供託を含むTSUMO精算が一意に確定する。 */
    EXACT_TSUMO,
    /** RON権利者が一人だけで、本場・供託をすべて受け取ると仮定する。 */
    SOLE_RON
  }

  private final VisibleHandScoreBuffer visibleScore = new VisibleHandScoreBuffer();
  private final PointDeltaBuffer paymentFloor = new PointDeltaBuffer();
  private final PointDeltaBuffer projectedDelta = new PointDeltaBuffer();
  private final HandScoreBuffer scoreBuffer = new HandScoreBuffer();
  private final int[] projectedScores = new int[GameState.NUM_PLAYERS];
  private final int[] projectedRanks = new int[GameState.NUM_PLAYERS];
  private final int[] scoreGaps = new int[GameState.NUM_PLAYERS];
  private boolean uraEligible;
  private int uraIndicatorCount;
  private boolean paoApplies;
  private SettlementAssumption settlementAssumption = SettlementAssumption.EXACT_TSUMO;

  WinSettlementProjection() {}

  /**
   * 公開情報から確定する役・翻・符・基本点を返す。
   *
   * @return 裏ドラを除く公開和了価値
   */
  public VisibleHandScoreBuffer visibleScore() {
    return visibleScore;
  }

  /**
   * 指定席の裏ドラを除いた和了点による点棒増減を返す。
   *
   * @param absolutePlayer 絶対席
   * @return 受取は正、支払は負の点数増減
   */
  public int paymentFloor(int absolutePlayer) {
    return paymentFloor.get(absolutePlayer);
  }

  /**
   * 明示された精算仮定を適用した後の0始まり順位を返す。
   *
   * @param absolutePlayer 対象の絶対席
   * @return 指定した精算条件で計算した0始まりの順位
   */
  public int projectedRank(int absolutePlayer) {
    return projectedRanks[absolutePlayer];
  }

  /**
   * 和了者の精算後得点から指定席の精算後得点を引いた差を返す。
   *
   * @param absolutePlayer 比較対象の絶対席
   * @return 和了者基準の得点差
   */
  public int scoreGap(int absolutePlayer) {
    return scoreGaps[absolutePlayer];
  }

  /**
   * 立直済みで裏ドラ対象かを返す。裏ドラ翻数は公開値へ含めない。
   *
   * @return 裏ドラを開示し得るなら {@code true}
   */
  public boolean uraEligible() {
    return uraEligible;
  }

  /**
   * 和了時に開示され得る裏ドラ表示牌枚数を返す。
   *
   * @return 裏ドラ表示牌数
   */
  public int uraIndicatorCount() {
    return uraIndicatorCount;
  }

  /**
   * 和了後の順位計算に使用した精算仮定を返す。
   *
   * @return 単独ロンなど、計算に用いた精算条件
   */
  public SettlementAssumption settlementAssumption() {
    return settlementAssumption;
  }

  /** 包が成立する和了なら {@code true}。 */
  public boolean paoApplies() {
    return paoApplies;
  }

  /**
   * RONまたはTSUMOの公開得点下限と和了後順位を、実対局と同じ和了・精算規則で計算する。
   *
   * <p>ロンの点棒移動は、複数ロン時にも各和了者について計算できるよう、本場を0として求める。順位だけは現在本場と供託を一人で受け取る単独RON仮定を使う。
   *
   * @param state 行動が合法と判定された現在局面
   * @param player 和了する絶対席
   * @param action RONまたはTSUMO
   * @return 公開得点、支払下限、仮定付き和了後順位
   */
  WinSettlementProjection load(
      GameState state, int player, Action action, HandScoreEvaluator scoreEvaluator) {
    Hand hand = state.hand(player);
    HandScoreBuffer result;
    int doraCount = state.doraState().indicatorCount();
    if (action.type() == Action.Type.TSUMO_AGARI) {
      TurnEvent.Draw draw = (TurnEvent.Draw) state.getTurnEvent();
      result =
          requireScore(
              scoreEvaluator.scoreCompleted(
                  hand,
                  draw.tileType(),
                  WinMethod.TSUMO,
                  WinConditions.fromGameState(state, player, draw),
                  state.doraState(),
                  false,
                  hand.ownedAkaMask(),
                  scoreBuffer),
              scoreBuffer);
      WinSettlementCalculator.settleTsumoInto(
          player, state.getOya(), state.getHonba(), result, hand, paymentFloor);
      projectedDelta.bind(
          paymentFloor.player0(),
          paymentFloor.player1(),
          paymentFloor.player2(),
          paymentFloor.player3());
      settlementAssumption = SettlementAssumption.EXACT_TSUMO;
    } else if (action.type() == Action.Type.RON_AGARI) {
      TurnEvent.ResponseSource source = (TurnEvent.ResponseSource) state.getTurnEvent();
      result =
          requireScore(
              scoreEvaluator.scoreAfterAdding(
                  hand,
                  source.tileType(),
                  WinMethod.RON,
                  WinConditions.fromGameState(state, player, source),
                  state.doraState(),
                  false,
                  AkaTileMask.includeTileIfAka(
                      hand.ownedAkaMask(), source.tileType(), source.isAkaTile()),
                  scoreBuffer),
              scoreBuffer);
      WinSettlementCalculator.settleRonInto(
          player, source.player(), state.getOya(), 0, result, hand, paymentFloor);
      WinSettlementCalculator.settleRonInto(
          player, source.player(), state.getOya(), state.getHonba(), result, hand, projectedDelta);
      settlementAssumption = SettlementAssumption.SOLE_RON;
    } else {
      throw new IllegalArgumentException("immediate win projection requires RON or TSUMO");
    }

    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      projectedScores[seat] = state.getScore(seat) + projectedDelta.get(seat);
    }
    projectedScores[player] += state.getKyotakuCount() * HanchanProgression.RIICHI_COST;
    ScoreRanking.byScoreThenSeat(projectedScores, projectedRanks);
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      scoreGaps[seat] = projectedScores[player] - projectedScores[seat];
    }
    VisibleHandScoreBuffer.copyInto(result, visibleScore);
    paoApplies = PaoRules.applies(hand, result);
    uraEligible = state.isRiichi(player);
    uraIndicatorCount = doraCount;
    return this;
  }

  private static HandScoreBuffer requireScore(boolean available, HandScoreBuffer result) {
    if (!available) {
      throw new IllegalStateException("legal win action has no public yaku");
    }
    return result;
  }
}
