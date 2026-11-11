package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.EnumSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 入力のフィールドの幅と位置、相対席での書き方、段階、重みのテンソルの大きさを確かめる。 */
public class GrpLayoutTest {

  @Test
  public void widthsMatchTheDesign() {
    Assert.assertEquals(GrpLayout.of(Variant.TENHOU_4P, GrpFixtures.plainFields()).width(), 40);
    Assert.assertEquals(GrpLayout.of(Variant.TENHOU_3P, GrpFixtures.plainFields()).width(), 26);
  }

  @Test
  public void offsetsFollowTheDeclarationOrder() {
    GrpLayout layout = GrpLayout.of(Variant.TENHOU_4P, EnumSet.of(GrpField.GAP, GrpField.ROUND, GrpField.SCORE));
    Assert.assertEquals(layout.offset(GrpField.ROUND), 0);
    Assert.assertEquals(layout.offset(GrpField.SCORE), 12);
    Assert.assertEquals(layout.offset(GrpField.GAP), 16);
    Assert.assertEquals(layout.width(), 22);
    Assert.assertFalse(layout.has(GrpField.RANK));
    Assert.assertThrows(IllegalArgumentException.class, () -> layout.offset(GrpField.RANK));
    Assert.assertEquals(layout.fields().toString(), "[ROUND, SCORE, GAP]");
  }

  @Test
  public void aLayoutNeedsAField() {
    Assert.assertThrows(
        IllegalArgumentException.class, () -> GrpLayout.of(Variant.TENHOU_4P, EnumSet.noneOf(GrpField.class)));
  }

  @Test
  public void encodesTheTableInRelativeSeats() {
    // 南 3（局 6）で親は絶対席 2。相対席 0〜3 は絶対席 2・3・0・1。絶対席 0 と 2 が同点のトップで、0 が上。
    GrpState state = new GrpState(Variant.TENHOU_4P, 6, 2, 1, new int[] {32_000, 18_000, 32_000, 17_000});
    GrpLayout layout = GrpLayout.of(Variant.TENHOU_4P, GrpFixtures.plainFields());
    float[] x = new float[layout.width()];
    layout.encode(state, x);
    float[] expected = new float[40];
    expected[6] = 1;
    expected[12] = 2 * 300 / 10_000f;
    expected[13] = 1_000 / 10_000f;
    float[] scores = {7_000 / 10_000f, -8_000 / 10_000f, 7_000 / 10_000f, -7_000 / 10_000f};
    System.arraycopy(scores, 0, expected, 14, 4);
    int[] ranks = {1, 3, 0, 2};
    for (int seat = 0; seat < 4; seat++) {
      expected[18 + seat * 4 + ranks[seat]] = 1;
    }
    float[] gaps = {0, 14_000 / 10_000f, 15_000 / 10_000f, 14_000 / 10_000f, 15_000 / 10_000f, 1_000 / 10_000f};
    System.arraycopy(gaps, 0, expected, 34, 6);
    Assert.assertEquals(x, expected);
  }

  @Test
  public void phasesFollowTheNextRound() {
    GrpPhase[] four = {
      GrpPhase.EAST, GrpPhase.EAST, GrpPhase.EAST, GrpPhase.EAST, GrpPhase.SOUTH, GrpPhase.SOUTH, GrpPhase.PENULT,
      GrpPhase.FINAL, GrpPhase.FINAL, GrpPhase.FINAL, GrpPhase.FINAL, GrpPhase.FINAL
    };
    for (int round = 0; round < four.length; round++) {
      Assert.assertEquals(GrpPhase.of(Variant.TENHOU_4P, round), four[round], "round " + round);
    }
    GrpPhase[] three = {
      GrpPhase.EAST, GrpPhase.EAST, GrpPhase.EAST, GrpPhase.SOUTH, GrpPhase.PENULT, GrpPhase.FINAL, GrpPhase.FINAL,
      GrpPhase.FINAL, GrpPhase.FINAL
    };
    for (int round = 0; round < three.length; round++) {
      Assert.assertEquals(GrpPhase.of(Variant.TENHOU_3P, round), three[round], "round " + round);
    }
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpPhase.of(Variant.TENHOU_4P, 12));
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpPhase.of(Variant.TENHOU_3P, 9));
  }

  @Test
  public void tensorSizesFollowTheShape() {
    GrpShape plain = GrpFixtures.shape(Variant.TENHOU_4P, 128, false);
    int[] starts = GrpTensor.starts(plain);
    Assert.assertEquals(starts, new int[] {0, 5_120, 5_248, 21_632, 21_760, 24_832, 24_856});
    GrpShape heads = new GrpShape(Variant.TENHOU_4P, GrpFixtures.plainFields(), 128, true);
    Assert.assertEquals(heads.heads(), 4);
    Assert.assertEquals(GrpTensor.W3.size(heads), 4 * 128 * 24);
    Assert.assertEquals(GrpTensor.B3.size(heads), 4 * 24);
    Assert.assertEquals(GrpTensor.W1.size(heads), 40 * 128);
    Assert.assertEquals(GrpFixtures.shape(Variant.TENHOU_3P, 8, false).outputs(), 6);
  }
}
