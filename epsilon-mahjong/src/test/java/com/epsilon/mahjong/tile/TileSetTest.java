package com.epsilon.mahjong.tile;

import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌の集合の枚数・並び順・赤 5・ドラの巡り方。 */
public class TileSetTest {

  @Test
  public void fullUsesEveryTileInIdOrder() {
    TileSet set = TileSet.FULL;
    Assert.assertEquals(set.size(), 136);
    for (int index = 0; index < set.size(); index++) {
      Assert.assertEquals(set.physicalId(index), index);
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Assert.assertEquals(set.copies(kind), Tiles.COPIES);
    }
    Assert.assertEquals(set.kinds(), Tiles.ALL_KINDS);
  }

  @Test
  public void sanmaHas108TilesWithoutTwoToEightMan() {
    TileSet set = TileSet.SANMA;
    Assert.assertEquals(set.size(), 108);
    for (int kind = Tiles.M2; kind <= Tiles.M8; kind++) {
      Assert.assertEquals(set.copies(kind), 0);
    }
    int previous = -1;
    for (int index = 0; index < set.size(); index++) {
      int id = set.physicalId(index);
      Assert.assertTrue(id > previous, "ascending at " + index);
      Assert.assertTrue(id < 4 || id > 31, "2-8 man present: " + id);
      Assert.assertTrue(set.contains(id));
      previous = id;
    }
    for (int id = 4; id <= 31; id++) {
      Assert.assertFalse(set.contains(id));
    }
    Assert.assertFalse(set.contains(-1));
    Assert.assertFalse(set.contains(Tiles.PHYSICAL_IDS));
  }

  @Test
  public void redFivesAreCopyZero() {
    Assert.assertEquals(redIds(TileSet.FULL), List.of(16, 52, 88));
    Assert.assertEquals(redIds(TileSet.SANMA), List.of(52, 88));
    Assert.assertEquals(TileSet.FULL.redKinds(), Tiles.bit(Tiles.M5) | Tiles.bit(Tiles.P5) | Tiles.bit(Tiles.S5));
    Assert.assertEquals(TileSet.SANMA.redKinds(), Tiles.bit(Tiles.P5) | Tiles.bit(Tiles.S5));
  }

  @Test
  public void doraWrapsInsideEachGroup() {
    TileSet full = TileSet.FULL;
    Assert.assertEquals(full.doraOf(Tiles.M1), Tiles.M2);
    Assert.assertEquals(full.doraOf(Tiles.M9), Tiles.M1);
    Assert.assertEquals(full.doraOf(Tiles.P9), Tiles.P1);
    Assert.assertEquals(full.doraOf(Tiles.S9), Tiles.S1);
    Assert.assertEquals(full.doraOf(Tiles.PEI), Tiles.TON);
    Assert.assertEquals(full.doraOf(Tiles.HAKU), Tiles.HATSU);
    Assert.assertEquals(full.doraOf(Tiles.CHUN), Tiles.HAKU);
    TileSet sanma = TileSet.SANMA;
    Assert.assertEquals(sanma.doraOf(Tiles.M1), Tiles.M9);
    Assert.assertEquals(sanma.doraOf(Tiles.M9), Tiles.M1);
    Assert.assertEquals(sanma.doraOf(Tiles.P9), Tiles.P1);
    Assert.assertEquals(sanma.doraOf(Tiles.PEI), Tiles.TON);
    Assert.assertThrows(IllegalArgumentException.class, () -> sanma.doraOf(Tiles.M5));
  }

  /** どの集合でも、ドラはグループ（色・風牌・三元牌）ごとに、使う種類を昇順に巡って先頭に戻る。 */
  @Test
  public void doraCyclesThroughTheUsedKindsOfEachGroup() {
    for (TileSet set : TileSet.values()) {
      for (Tiles.Group group : Tiles.Group.values()) {
        List<Integer> used = new ArrayList<>();
        for (int kind = group.first(); kind <= group.last(); kind++) {
          if (set.copies(kind) > 0) {
            used.add(kind);
          }
        }
        Assert.assertFalse(used.isEmpty(), set + " " + group);
        for (int index = 0; index < used.size(); index++) {
          int next = used.get((index + 1) % used.size());
          Assert.assertEquals(set.doraOf(used.get(index)), next, set + " " + group);
        }
      }
    }
  }

  private static List<Integer> redIds(TileSet set) {
    List<Integer> ids = new ArrayList<>();
    for (int id = 0; id < Tiles.PHYSICAL_IDS; id++) {
      if (set.red(id)) {
        ids.add(id);
      }
    }
    return ids;
  }
}
