package com.epsilon.grp;

import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 卓のキーを確かめる。キーの往復、キーに収まらない卓、GrpState の作り方と、渡した値の検査。 */
public class GrpKeyTest {

  @Test
  public void keyWidthsFollowTheSeats() {
    Assert.assertEquals(GrpKey.bits(Variant.TENHOU_4P), 60);
    Assert.assertEquals(GrpKey.bits(Variant.TENHOU_3P), 49);
  }

  @Test
  public void randomStatesRoundTrip() {
    for (Variant variant : Variant.values()) {
      Random random = new Random(3 + variant.ordinal());
      for (int trial = 0; trial < 20_000; trial++) {
        GrpState state = GrpFixtures.state(variant, random);
        long key = state.key();
        Assert.assertNotEquals(key, GrpKey.NONE, state.toString());
        Assert.assertEquals(key >>> GrpKey.bits(variant), 0L, state.toString());
        Assert.assertEquals(GrpState.ofKey(variant, key), state);
      }
    }
  }

  @Test
  public void widestValuesRoundTrip() {
    GrpState four = new GrpState(Variant.TENHOU_4P, 11, 63, 63, new int[] {204_700, 0, 100, 204_700});
    Assert.assertEquals(GrpState.ofKey(Variant.TENHOU_4P, four.key()), four);
    GrpState three = new GrpState(Variant.TENHOU_3P, 8, 63, 0, new int[] {0, 204_700, 105_000});
    Assert.assertEquals(GrpState.ofKey(Variant.TENHOU_3P, three.key()), three);
  }

  @Test
  public void statesOutsideTheFieldsHaveNoKey() {
    int[] even = {25_000, 25_000, 25_000, 25_000};
    assertNoKey(new GrpState(Variant.TENHOU_4P, 0, 64, 0, even));
    assertNoKey(new GrpState(Variant.TENHOU_4P, 0, 0, 64, even));
    assertNoKey(new GrpState(Variant.TENHOU_4P, 0, 0, 0, new int[] {204_800, 0, 0, 0}));
    assertNoKey(new GrpState(Variant.TENHOU_4P, 0, 0, 0, new int[] {-100, 25_100, 25_000, 25_000}));
    assertNoKey(new GrpState(Variant.TENHOU_4P, 0, 0, 0, new int[] {25_050, 24_950, 25_000, 25_000}));
    assertNoKey(new GrpState(Variant.TENHOU_3P, 0, 0, 0, new int[] {35_000, 35_010, 34_990}));
  }

  @Test
  public void foreignKeysAreRejected() {
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpState.ofKey(Variant.TENHOU_4P, GrpKey.NONE));
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpState.ofKey(Variant.TENHOU_3P, 1L << 49));
    int[] even = {25_000, 25_000, 25_000, 25_000};
    long westFour = new GrpState(Variant.TENHOU_4P, 11, 0, 0, even).key();
    long threePlayerBits = westFour & (1L << GrpKey.bits(Variant.TENHOU_3P)) - 1;
    Assert.assertTrue(GrpKey.valid(Variant.TENHOU_4P, westFour));
    Assert.assertFalse(GrpKey.valid(Variant.TENHOU_3P, threePlayerBits));
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpState.ofKey(Variant.TENHOU_3P, threePlayerBits));
  }

  @Test
  public void tablesAndStatesShareTheKey() {
    TableState table = new TableState(Variant.TENHOU_4P);
    table.set(new RoundStart(Variant.TENHOU_4P, 5, 2, 1, new int[] {31_000, 18_000, 26_500, 23_500}));
    GrpState state = GrpState.of(table);
    Assert.assertEquals(state, new GrpState(Variant.TENHOU_4P, 5, 2, 1, new int[] {31_000, 18_000, 26_500, 23_500}));
    Assert.assertEquals(GrpKey.of(table), state.key());
    Assert.assertEquals(state.dealer(), 1);
  }

  @Test
  public void statesCheckTheirShapeAndKeepTheirScores() {
    int[] scores = {25_000, 25_000, 25_000, 25_000};
    Assert.assertThrows(IllegalArgumentException.class, () -> new GrpState(Variant.TENHOU_4P, 12, 0, 0, scores));
    Assert.assertThrows(IllegalArgumentException.class, () -> new GrpState(Variant.TENHOU_4P, -1, 0, 0, scores));
    Assert.assertThrows(IllegalArgumentException.class, () -> new GrpState(Variant.TENHOU_4P, 0, -1, 0, scores));
    Assert.assertThrows(IllegalArgumentException.class, () -> new GrpState(Variant.TENHOU_4P, 0, 0, -1, scores));
    Assert.assertThrows(IllegalArgumentException.class, () -> new GrpState(Variant.TENHOU_3P, 0, 0, 0, scores));
    GrpState state = new GrpState(Variant.TENHOU_4P, 0, 0, 0, scores);
    scores[0] = 0;
    state.scores()[1] = 0;
    Assert.assertEquals(state.score(0), 25_000);
    Assert.assertEquals(state.score(1), 25_000);
  }

  private static void assertNoKey(GrpState state) {
    Assert.assertEquals(state.key(), GrpKey.NONE, state.toString());
  }
}
