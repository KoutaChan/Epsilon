package com.epsilon.engine;

/** 基本点から親子別の支払額を求める。切り上げ満貫は採用しない。 */
public final class ScorePayments {
  private ScorePayments() {}

  public static int ronPoints(int basePoints, boolean dealer) {
    return roundUp100(basePoints * (dealer ? 6 : 4));
  }

  public static int tsumoFromDealer(int basePoints) {
    return roundUp100(basePoints * 2);
  }

  public static int tsumoFromChild(int basePoints, boolean winnerIsDealer) {
    return roundUp100(basePoints * (winnerIsDealer ? 2 : 1));
  }

  public static int tsumoTotal(int basePoints, boolean dealer) {
    return dealer
        ? tsumoFromChild(basePoints, true) * 3
        : tsumoFromDealer(basePoints) + tsumoFromChild(basePoints, false) * 2;
  }

  static WinPayment.Ron ron(int basePoints, WinnerRole role) {
    return new WinPayment.Ron(ronPoints(basePoints, role == WinnerRole.DEALER));
  }

  static WinPayment.Tsumo tsumo(int basePoints, WinnerRole role) {
    return role == WinnerRole.DEALER
        ? new WinPayment.DealerTsumo(tsumoFromChild(basePoints, true))
        : new WinPayment.ChildTsumo(tsumoFromDealer(basePoints), tsumoFromChild(basePoints, false));
  }

  private static int roundUp100(int points) {
    return ((points + 99) / 100) * 100;
  }
}
