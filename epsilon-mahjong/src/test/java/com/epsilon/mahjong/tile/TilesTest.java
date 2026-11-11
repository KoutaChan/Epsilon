package com.epsilon.mahjong.tile;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌のグループ（色・風牌・三元牌）、物理 id、グループの中での番号、種類のビットマスク。 */
public class TilesTest {

  @Test
  public void groupsCoverEveryKindOnceInDeclarationOrder() {
    int next = 0;
    for (Tiles.Group group : Tiles.Group.values()) {
      Assert.assertEquals(group.first(), next, group.name());
      for (int kind = group.first(); kind <= group.last(); kind++) {
        Assert.assertSame(Tiles.Group.of(kind), group);
      }
      next = group.last() + 1;
    }
    Assert.assertEquals(next, Tiles.KINDS);
  }

  @Test
  public void suitsAreTheNineKindGroups() {
    int suits = 0;
    for (Tiles.Group group : Tiles.Group.values()) {
      if (group.suited()) {
        suits++;
        Assert.assertEquals(group.size(), Tiles.SUIT_SIZE);
      }
    }
    Assert.assertEquals(suits, Tiles.NUMBER_SUITS);
    Assert.assertEquals(Long.bitCount(Tiles.FIVES), Tiles.NUMBER_SUITS);
    Assert.assertEquals(Tiles.Group.WIND.size(), 4);
    Assert.assertEquals(Tiles.Group.DRAGON.size(), 3);
  }

  @Test
  public void physicalIdsSplitIntoKindAndCopy() {
    for (int id = 0; id < Tiles.PHYSICAL_IDS; id++) {
      int kind = Tiles.kind(id);
      int copy = Tiles.copy(id);
      Assert.assertTrue(kind < Tiles.KINDS && copy < Tiles.COPIES);
      Assert.assertEquals(Tiles.physicalId(kind, copy), id);
    }
    Assert.assertEquals(Tiles.PHYSICAL_IDS, 136);
    Assert.assertEquals(Tiles.physicalId(Tiles.P5, Tiles.RED_COPY), 52);
    Assert.assertEquals(Tiles.physicalId(Tiles.S5, Tiles.RED_COPY), 88);
  }

  @Test
  public void numbersCountWithinTheGroup() {
    Assert.assertEquals(Tiles.number(Tiles.M1), 0);
    Assert.assertEquals(Tiles.number(Tiles.P5), Tiles.FIVE);
    Assert.assertEquals(Tiles.number(Tiles.S9), 8);
    Assert.assertEquals(Tiles.number(Tiles.TON), 0);
    Assert.assertEquals(Tiles.number(Tiles.PEI), 3);
    Assert.assertEquals(Tiles.number(Tiles.CHUN), 2);
  }

  @Test
  public void masksHoldTheirKinds() {
    Assert.assertEquals(Long.bitCount(Tiles.ALL_KINDS), Tiles.KINDS);
    Assert.assertEquals(Long.bitCount(Tiles.TERMINALS_AND_HONORS), 13);
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Tiles.Group group = Tiles.Group.of(kind);
      boolean terminal = group.suited() && (kind == group.first() || kind == group.last());
      Assert.assertEquals(Tiles.contains(Tiles.TERMINALS, kind), terminal, "kind " + kind);
      Assert.assertEquals(Tiles.contains(Tiles.HONORS, kind), !group.suited(), "kind " + kind);
      Assert.assertEquals(
          Tiles.contains(Tiles.FIVES, kind), group.suited() && Tiles.number(kind) == Tiles.FIVE, "kind " + kind);
    }
    Assert.assertEquals(Tiles.range(Tiles.M2, Tiles.M8), 0b1111_1110L);
    Assert.assertEquals(Tiles.range(Tiles.M1, Tiles.CHUN), Tiles.ALL_KINDS);
  }
}
