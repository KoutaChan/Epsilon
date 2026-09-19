package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.core.GameState;
import com.epsilon.core.HandView;

/** 和了時の点棒移動を計算する。 */
final class WinSettlementCalculator {

  private WinSettlementCalculator() {}

  /** ツモ和了の精算結果を、和了明細や支払いオブジェクトを生成せずに書き込む。 */
  static void settleTsumoInto(
      int winnerSeat,
      int dealerSeat,
      int honba,
      HandScoreBuffer score,
      HandView winnerHand,
      PointDeltaBuffer destination) {
    int bonus = honba * 100;
    boolean dealerWin = winnerSeat == dealerSeat;
    int dealerPayment = dealerTsumoPayment(score.basePoints(), dealerWin, bonus);
    int childPayment = childTsumoPayment(score.basePoints(), dealerWin, bonus);
    int liableSeat = PaoRules.liableSeat(winnerSeat, winnerHand, score);
    destination.bind(
        tsumoDelta(0, winnerSeat, dealerSeat, dealerPayment, childPayment, liableSeat),
        tsumoDelta(1, winnerSeat, dealerSeat, dealerPayment, childPayment, liableSeat),
        tsumoDelta(2, winnerSeat, dealerSeat, dealerPayment, childPayment, liableSeat),
        tsumoDelta(3, winnerSeat, dealerSeat, dealerPayment, childPayment, liableSeat));
  }

  /** ロン和了の精算結果を、和了明細や支払いオブジェクトを生成せずに書き込む。 */
  static void settleRonInto(
      int winnerSeat,
      int discarderSeat,
      int dealerSeat,
      int honba,
      HandScoreBuffer score,
      HandView winnerHand,
      PointDeltaBuffer destination) {
    int basePayment = ronBasePayment(score.basePoints(), winnerSeat == dealerSeat);
    int honbaPayment = honba * 300;
    int liableSeat = PaoRules.liableSeat(winnerSeat, winnerHand, score);
    destination.bind(
        ronDelta(0, winnerSeat, discarderSeat, basePayment, honbaPayment, liableSeat),
        ronDelta(1, winnerSeat, discarderSeat, basePayment, honbaPayment, liableSeat),
        ronDelta(2, winnerSeat, discarderSeat, basePayment, honbaPayment, liableSeat),
        ronDelta(3, winnerSeat, discarderSeat, basePayment, honbaPayment, liableSeat));
  }

  static WinClaim.Tsumo createTsumoClaim(
      int winnerSeat,
      int dealerSeat,
      int honba,
      HandScoreBuffer score,
      HandView winnerHand,
      boolean riichiDeclared) {
    WinPayment.Tsumo payment =
        calculateTsumoPayment(
            score.basePoints(),
            winnerSeat == dealerSeat ? WinnerRole.DEALER : WinnerRole.CHILD,
            honba);
    int dealerPayment = dealerPayment(payment);
    int childPayment = childPayment(payment);
    int liableSeat = PaoRules.liableSeat(winnerSeat, winnerHand, score);
    return new WinClaim.Tsumo(
        winnerSeat,
        score.snapshot(),
        payment,
        honba,
        new PointDelta(
            tsumoDelta(0, winnerSeat, dealerSeat, dealerPayment, childPayment, liableSeat),
            tsumoDelta(1, winnerSeat, dealerSeat, dealerPayment, childPayment, liableSeat),
            tsumoDelta(2, winnerSeat, dealerSeat, dealerPayment, childPayment, liableSeat),
            tsumoDelta(3, winnerSeat, dealerSeat, dealerPayment, childPayment, liableSeat)),
        riichiDeclared);
  }

  static WinPayment.Tsumo calculateTsumoPayment(
      int basePoints, WinnerRole winnerRole, int honba) {
    int bonus = honba * 100;
    boolean dealerWin = winnerRole == WinnerRole.DEALER;
    int childPayment = childTsumoPayment(basePoints, dealerWin, bonus);
    return dealerWin
        ? new WinPayment.DealerTsumo(childPayment)
        : new WinPayment.ChildTsumo(
            dealerTsumoPayment(basePoints, false, bonus), childPayment);
  }

  /** 複数人の流し満貫を重ねられるよう、通常ツモの精算額を既存配列へ加算する。 */
  static void addTsumoPayments(
      int[] scoreDelta, int winnerSeat, int dealerSeat, WinPayment.Tsumo payment) {
    int dealerPayment = dealerPayment(payment);
    int childPayment = childPayment(payment);
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      scoreDelta[seat] +=
          tsumoDelta(seat, winnerSeat, dealerSeat, dealerPayment, childPayment, -1);
    }
  }

  static WinClaim.Ron createRonClaim(
      int winnerSeat,
      int discarderSeat,
      int dealerSeat,
      int honba,
      HandScoreBuffer score,
      HandView winnerHand,
      boolean riichiDeclared) {
    int honbaPayment = honba * 300;
    int basePayment = ronBasePayment(score.basePoints(), winnerSeat == dealerSeat);
    WinPayment.Ron payment = new WinPayment.Ron(basePayment + honbaPayment);
    int liableSeat = PaoRules.liableSeat(winnerSeat, winnerHand, score);
    return new WinClaim.Ron(
        winnerSeat,
        discarderSeat,
        score.snapshot(),
        payment,
        honba,
        new PointDelta(
            ronDelta(0, winnerSeat, discarderSeat, basePayment, honbaPayment, liableSeat),
            ronDelta(1, winnerSeat, discarderSeat, basePayment, honbaPayment, liableSeat),
            ronDelta(2, winnerSeat, discarderSeat, basePayment, honbaPayment, liableSeat),
            ronDelta(3, winnerSeat, discarderSeat, basePayment, honbaPayment, liableSeat)),
        riichiDeclared);
  }

  private static int dealerTsumoPayment(int basePoints, boolean dealerWin, int bonus) {
    return dealerWin ? 0 : ScorePayments.tsumoFromDealer(basePoints) + bonus;
  }

  private static int childTsumoPayment(int basePoints, boolean dealerWin, int bonus) {
    return ScorePayments.tsumoFromChild(basePoints, dealerWin) + bonus;
  }

  private static int ronBasePayment(int basePoints, boolean dealerWin) {
    return ScorePayments.ronPoints(basePoints, dealerWin);
  }

  private static int dealerPayment(WinPayment.Tsumo payment) {
    return payment instanceof WinPayment.ChildTsumo childTsumo ? childTsumo.fromDealer() : 0;
  }

  private static int childPayment(WinPayment.Tsumo payment) {
    return switch (payment) {
      case WinPayment.DealerTsumo dealerTsumo -> dealerTsumo.each();
      case WinPayment.ChildTsumo childTsumo -> childTsumo.fromChild();
    };
  }

  private static int tsumoDelta(
      int seat,
      int winnerSeat,
      int dealerSeat,
      int dealerPayment,
      int childPayment,
      int liableSeat) {
    int winnerGain =
        winnerSeat == dealerSeat ? childPayment * 3 : dealerPayment + childPayment * 2;
    if (seat == winnerSeat) {
      return winnerGain;
    }
    if (liableSeat >= 0) {
      return seat == liableSeat ? -winnerGain : 0;
    }
    return -(winnerSeat == dealerSeat
        ? childPayment
        : seat == dealerSeat ? dealerPayment : childPayment);
  }

  private static int ronDelta(
      int seat,
      int winnerSeat,
      int discarderSeat,
      int basePayment,
      int honbaPayment,
      int liableSeat) {
    int winnerGain = basePayment + honbaPayment;
    if (seat == winnerSeat) {
      return winnerGain;
    }
    if (liableSeat < 0 || liableSeat == discarderSeat) {
      return seat == discarderSeat ? -winnerGain : 0;
    }
    int halfPayment = basePayment / 2;
    if (seat == liableSeat) {
      return -halfPayment - honbaPayment;
    }
    return seat == discarderSeat ? -halfPayment : 0;
  }
}
