package com.epsilon.prism.core;

import com.epsilon.mahjong.rules.Variant;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * {@link DecisionClass} のテスト。オーラス（南 4 局と西入の局）の和了だけが {@link DecisionClass#FINAL_END} に分かれ、九種九牌、ほかの種別、
 * オーラスより前の局のクラスは変わらないことを確かめる。クラスの番号は行の形式に入るので、既存のクラスの番号が変わらないことも確かめる。
 */
public class DecisionClassTest {

  private static final Variant FOUR = Variant.TENHOU_4P;
  private static final int SOUTH_FOUR = FOUR.lastRound();

  @Test
  public void winsInTheFinalRoundsAreTheirOwnClass() {
    Assert.assertEquals(DecisionClass.of(FOUR, Kind.WIN, Stage.EARLY, 0, 0), DecisionClass.END);
    Assert.assertEquals(DecisionClass.of(FOUR, Kind.WIN, Stage.LATE, 2, SOUTH_FOUR - 1), DecisionClass.END);
    Assert.assertEquals(
        DecisionClass.of(FOUR, Kind.WIN, Stage.LATE, 2, SOUTH_FOUR), DecisionClass.FINAL_END);
    Assert.assertEquals(
        DecisionClass.of(FOUR, Kind.WIN, Stage.MID, 0, FOUR.finalRound()),
        DecisionClass.FINAL_END);
    // 九種九牌は局をやり直すだけで半荘を終えないので、オーラスでも END のまま
    Assert.assertEquals(DecisionClass.of(FOUR, Kind.KYUSHU, Stage.EARLY, 0, 0), DecisionClass.END);
    Assert.assertEquals(
        DecisionClass.of(FOUR, Kind.KYUSHU, Stage.EARLY, 0, SOUTH_FOUR), DecisionClass.END);
    Assert.assertEquals(
        DecisionClass.of(FOUR, Kind.RIICHI, Stage.LATE, 1, SOUTH_FOUR), DecisionClass.RIICHI);
    Assert.assertEquals(DecisionClass.of(FOUR, Kind.KAN, Stage.MID, 0, SOUTH_FOUR), DecisionClass.KAN);
    Assert.assertEquals(DecisionClass.of(FOUR, Kind.CALL, Stage.MID, 0, SOUTH_FOUR), DecisionClass.CALL);
    Assert.assertEquals(
        DecisionClass.of(FOUR, Kind.DISCARD, Stage.LATE, 1, SOUTH_FOUR), DecisionClass.THREAT);
    Assert.assertEquals(
        DecisionClass.of(FOUR, Kind.POSTCALL, Stage.EARLY, 0, SOUTH_FOUR), DecisionClass.DISCARD_EARLY);
    Assert.assertTrue(DecisionClass.FINAL_END.commitment());
  }

  @Test
  public void threePlayerFinalRoundsFollowTheVariant() {
    Variant three = Variant.TENHOU_3P;
    Assert.assertEquals(
        DecisionClass.of(three, Kind.WIN, Stage.LATE, 0, three.lastRound() - 1), DecisionClass.END);
    Assert.assertEquals(
        DecisionClass.of(three, Kind.WIN, Stage.LATE, 0, three.lastRound()), DecisionClass.FINAL_END);
    Assert.assertEquals(
        DecisionClass.of(three, Kind.WIN, Stage.LATE, 0, three.finalRound()), DecisionClass.FINAL_END);
  }

  /** 北抜きと、抜いた北への応答は、三人打ちにだけあるクラス。オーラスの応答は和了と同じ FINAL_END に入る。 */
  @Test
  public void northDecisionsHaveTheirOwnClasses() {
    Variant three = Variant.TENHOU_3P;
    Assert.assertEquals(DecisionClass.of(three, Kind.NUKI, Stage.EARLY, 0, 0), DecisionClass.NUKI);
    Assert.assertEquals(DecisionClass.of(three, Kind.NUKI, Stage.LATE, 1, three.finalRound()), DecisionClass.NUKI);
    Assert.assertEquals(DecisionClass.of(three, Kind.NUKI_RESPONSE, Stage.MID, 0, 0), DecisionClass.NUKI_RESPONSE);
    Assert.assertEquals(
        DecisionClass.of(three, Kind.NUKI_RESPONSE, Stage.MID, 1, three.lastRound() - 1), DecisionClass.NUKI_RESPONSE);
    Assert.assertEquals(
        DecisionClass.of(three, Kind.NUKI_RESPONSE, Stage.LATE, 0, three.lastRound()), DecisionClass.FINAL_END);
    Assert.assertEquals(
        DecisionClass.of(three, Kind.NUKI_RESPONSE, Stage.LATE, 0, three.finalRound()), DecisionClass.FINAL_END);
    Assert.assertTrue(DecisionClass.NUKI.commitment());
    Assert.assertTrue(DecisionClass.NUKI_RESPONSE.commitment());
  }

  /** 局の段階の区切りは山の残り枚数で決まり、その枚数は三人打ちが 40 枚と 20 枚、四人打ちが 50 枚と 25 枚。 */
  @Test
  public void stageBoundariesFollowTheVariant() {
    Variant three = Variant.TENHOU_3P;
    Assert.assertEquals(Stage.of(three, 40), Stage.EARLY);
    Assert.assertEquals(Stage.of(three, 39), Stage.MID);
    Assert.assertEquals(Stage.of(three, 20), Stage.MID);
    Assert.assertEquals(Stage.of(three, 19), Stage.LATE);
    Assert.assertEquals(Stage.of(FOUR, 50), Stage.EARLY);
    Assert.assertEquals(Stage.of(FOUR, 49), Stage.MID);
    Assert.assertEquals(Stage.of(FOUR, 25), Stage.MID);
    Assert.assertEquals(Stage.of(FOUR, 24), Stage.LATE);
  }

  @Test
  public void existingClassesKeepTheirRowNumbers() {
    String[] names = {
      "DISCARD_EARLY", "DISCARD_MID", "DISCARD_LATE", "THREAT", "RIICHI", "CALL", "KAN", "END",
      "FINAL_END", "NUKI", "NUKI_RESPONSE"
    };
    Assert.assertEquals(DecisionClass.COUNT, names.length);
    for (int ordinal = 0; ordinal < names.length; ordinal++) {
      Assert.assertEquals(DecisionClass.of(ordinal).name(), names[ordinal]);
    }
  }
}
