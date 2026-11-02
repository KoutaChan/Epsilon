package com.epsilon.core;

import java.util.HashSet;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 席別ツモ列（WASO）の不変条件。 */
public class WallSeatStreamTest {
  private static Wall dealt(long seed, int dealer) {
    Wall wall = new Wall(seed);
    for (int i = 0; i < 52; i++) wall.draw();
    wall.enableSeatStreams(dealer);
    return wall;
  }

  @Test
  public void roundRobinDrawsMatchPositionalOrder() {
    for (int dealer = 0; dealer < 4; dealer++) {
      Wall positional = new Wall(77L);
      for (int i = 0; i < 52; i++) positional.draw();
      Wall streamed = dealt(77L, dealer);
      for (int k = 0; !positional.isExhausted(); k++)
        Assert.assertEquals(streamed.draw((dealer + k) % 4), positional.draw());
    }
  }

  @Test
  public void anySeatOrderDrawsEachTileOnce() {
    var random = new SplittableRandom(5);
    for (int trial = 0; trial < 200; trial++) {
      Wall wall = dealt(trial, trial % 4);
      var seen = new HashSet<Integer>();
      while (!wall.isExhausted()) Assert.assertTrue(seen.add(wall.draw(random.nextInt(4))));
      Assert.assertEquals(seen.size(), Wall.LIVE_WALL_SIZE - 52);
    }
  }

  @Test
  public void seatGetsItsOwnStreamDespiteSkippedTurns() {
    Wall reference = dealt(9L, 0);
    int[] seatOne = new int[4];
    // 席0→1→2→3 の順なら、席1は位置 53, 57, 61, 65 の牌を引く。
    Wall positional = new Wall(9L);
    for (int i = 0; i < 52; i++) positional.draw();
    int[] positions = new int[20];
    for (int i = 0; i < 20; i++) positions[i] = positional.draw();
    for (int i = 0; i < 4; i++) seatOne[i] = positions[1 + 4 * i];
    // 席1の手番を2回飛ばしても（席0が2回続けて引く）、席1の次のツモは自分の列の先頭。
    Wall skipped = new Wall(reference);
    skipped.draw(0);
    skipped.draw(0);
    Assert.assertEquals(skipped.draw(1), seatOne[0]);
    Assert.assertEquals(skipped.draw(1), seatOne[1]);
  }

  @Test
  public void copyIsIndependent() {
    Wall wall = dealt(3L, 2);
    wall.draw(2);
    Wall copy = new Wall(wall);
    int a = wall.draw(3), b = copy.draw(3);
    Assert.assertEquals(a, b);
    int c = wall.draw(0);
    int d = copy.draw(1);
    Assert.assertNotEquals(c, d);
    Assert.assertEquals(wall.remaining(), copy.remaining());
  }

  @Test
  public void reshuffleClearsStreams() {
    Wall wall = dealt(1L, 0);
    Assert.assertTrue(wall.seatStreamsEnabled());
    wall.init();
    Assert.assertFalse(wall.seatStreamsEnabled());
  }
}
