package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.shape.HandShape.Block;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌を出し入れするたびに差分で更新するコードとマスクが、一から数え直した値と一致することを確かめる。 */
public class HandShapeTest {

  private static final int STEPS = 50_000;

  @Test
  public void codesAndMasksFollowRandomChanges() {
    Random random = new Random(5);
    HandShape hand = new HandShape();
    HandShape copy = new HandShape();
    int[] counts = new int[Tiles.KINDS];
    for (int step = 0; step < STEPS; step++) {
      int kind = random.nextInt(Tiles.KINDS);
      if (random.nextBoolean()) {
        if (counts[kind] < Tiles.COPIES && hand.size() < HandShape.MAX_SIZE) {
          hand.add(kind);
          counts[kind]++;
        }
      } else if (counts[kind] > 0) {
        hand.remove(kind);
        counts[kind]--;
      }
      assertMatches(hand, counts);
      if (step % 1_000 == 0) {
        copy.copyFrom(hand);
        assertMatches(copy, counts);
      }
    }
  }

  @Test
  public void limitsThrowAndKeepTheHand() {
    HandShape hand = ShapeFixtures.shape("1111m", 0);
    Assert.assertThrows(IllegalStateException.class, () -> hand.add(Tiles.M1));
    Assert.assertThrows(IllegalStateException.class, () -> hand.remove(Tiles.M2));
    Assert.assertEquals(hand.count(Tiles.M1), 4);
    HandShape full = ShapeFixtures.shape("123m456p789s11z", 1);
    Assert.assertEquals(full.size(), HandShape.MAX_SIZE);
    Assert.assertThrows(IllegalStateException.class, () -> full.add(Tiles.CHUN));
    Assert.assertThrows(IllegalStateException.class, full::addMeld);
    HandShape melds = ShapeFixtures.shape("", 4);
    Assert.assertEquals(melds.size(), 12);
    Assert.assertThrows(IllegalStateException.class, melds::addMeld);
  }

  @Test
  public void clearEmptiesTheHand() {
    HandShape hand = ShapeFixtures.shape("123m55z", 2);
    hand.clear();
    assertMatches(hand, new int[Tiles.KINDS]);
    Assert.assertEquals(hand.melds(), 0);
    Assert.assertEquals(hand.toString(), "HandShape[ melds=0]");
  }

  private static void assertMatches(HandShape hand, int[] counts) {
    int tiles = 0;
    long kinds = 0;
    long pairs = 0;
    long full = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Assert.assertEquals(hand.count(kind), counts[kind]);
      tiles += counts[kind];
      kinds |= counts[kind] >= 1 ? Tiles.bit(kind) : 0;
      pairs |= counts[kind] >= 2 ? Tiles.bit(kind) : 0;
      full |= counts[kind] == Tiles.COPIES ? Tiles.bit(kind) : 0;
    }
    Assert.assertEquals(hand.tiles(), tiles);
    Assert.assertEquals(hand.kinds(), kinds);
    Assert.assertEquals(hand.pairs(), pairs);
    Assert.assertEquals(hand.full(), full);
    for (Block block : Block.values()) {
      int code = 0;
      for (int kind = block.last(); kind >= block.first(); kind--) {
        code = code * Block.RADIX + counts[kind];
      }
      Assert.assertEquals(hand.code(block), code, block.name());
    }
  }
}
