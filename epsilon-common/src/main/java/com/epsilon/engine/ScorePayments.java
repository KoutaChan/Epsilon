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

  private static int roundUp100(int points) {
    return ((points + 99) / 100) * 100;
  }
}
