package com.epsilon.prism.core;

import com.epsilon.engine.HanchanProgression;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * {@link DecisionClass}: オーラス（南 4 局と西入の局）の和了だけが {@link DecisionClass#FINAL_END} に分かれ、九種九牌・ほかの種別・オーラスより前の局は
 * 変わらないこと。番号は行の形式に入るので、既存の類の番号が変わらないこと。
 */
public class DecisionClassTest {

  private static final int SOUTH_FOUR = HanchanProgression.SOUTH_END_KYOKU;

  @Test
  public void winsInTheFinalRoundsAreTheirOwnClass() {
    Assert.assertEquals(DecisionClass.of(Kind.WIN, Stage.EARLY, 0, 0), DecisionClass.END);
    Assert.assertEquals(DecisionClass.of(Kind.WIN, Stage.LATE, 2, SOUTH_FOUR - 1), DecisionClass.END);
    Assert.assertEquals(
        DecisionClass.of(Kind.WIN, Stage.LATE, 2, SOUTH_FOUR), DecisionClass.FINAL_END);
    Assert.assertEquals(
        DecisionClass.of(Kind.WIN, Stage.MID, 0, HanchanProgression.WEST_END_KYOKU),
        DecisionClass.FINAL_END);
    // 九種九牌は局をやり直すだけで半荘を終えないので、オーラスでも END
    Assert.assertEquals(DecisionClass.of(Kind.KYUSHU, Stage.EARLY, 0, 0), DecisionClass.END);
    Assert.assertEquals(
        DecisionClass.of(Kind.KYUSHU, Stage.EARLY, 0, SOUTH_FOUR), DecisionClass.END);
    Assert.assertEquals(
        DecisionClass.of(Kind.RIICHI, Stage.LATE, 1, SOUTH_FOUR), DecisionClass.RIICHI);
    Assert.assertEquals(DecisionClass.of(Kind.KAN, Stage.MID, 0, SOUTH_FOUR), DecisionClass.KAN);
    Assert.assertEquals(DecisionClass.of(Kind.CALL, Stage.MID, 0, SOUTH_FOUR), DecisionClass.CALL);
    Assert.assertEquals(
        DecisionClass.of(Kind.DISCARD, Stage.LATE, 1, SOUTH_FOUR), DecisionClass.THREAT);
    Assert.assertEquals(
        DecisionClass.of(Kind.POSTCALL, Stage.EARLY, 0, SOUTH_FOUR), DecisionClass.DISCARD_EARLY);
    Assert.assertTrue(DecisionClass.FINAL_END.commitment());
  }

  @Test
  public void existingClassesKeepTheirRowNumbers() {
    String[] names = {
      "DISCARD_EARLY", "DISCARD_MID", "DISCARD_LATE", "THREAT", "RIICHI", "CALL", "KAN", "END",
      "FINAL_END"
    };
    Assert.assertEquals(DecisionClass.COUNT, names.length);
    for (int ordinal = 0; ordinal < names.length; ordinal++) {
      Assert.assertEquals(DecisionClass.of(ordinal).name(), names[ordinal]);
    }
  }
}
