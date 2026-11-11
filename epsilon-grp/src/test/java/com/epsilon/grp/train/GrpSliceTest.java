package com.epsilon.grp.train;

import java.util.EnumMap;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 半荘 ID のハッシュで分けたときの割合を確かめる。関門のコホートでも、関門用に回らない半荘はふだんと同じに分かれる。 */
public class GrpSliceTest {

  private static final int GAMES = 200_000;

  @Test
  public void slicesTakeTheirShares() {
    Map<GrpSlice, Integer> plain = new EnumMap<>(GrpSlice.class);
    Map<GrpSlice, Integer> guarded = new EnumMap<>(GrpSlice.class);
    for (long gameId = 0; gameId < GAMES; gameId++) {
      GrpSlice slice = GrpSlice.of(gameId);
      plain.merge(slice, 1, Integer::sum);
      GrpSlice gated = GrpSlice.guarded(gameId);
      guarded.merge(gated, 1, Integer::sum);
      if (gated != GrpSlice.GUARD) {
        Assert.assertEquals(gated, slice);
      }
      Assert.assertEquals(GrpSlice.of(gameId), slice);
    }
    Assert.assertFalse(plain.containsKey(GrpSlice.GUARD));
    Assert.assertEquals(plain.get(GrpSlice.STOP) / (double) GAMES, 1.0 / 16, 0.002);
    Assert.assertEquals(guarded.get(GrpSlice.GUARD) / (double) GAMES, 1.0 / 4, 0.004);
    Assert.assertEquals(guarded.get(GrpSlice.STOP) / (double) GAMES, 3.0 / 4 / 16, 0.002);
  }
}
