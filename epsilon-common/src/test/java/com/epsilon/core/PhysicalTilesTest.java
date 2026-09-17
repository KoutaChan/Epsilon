package com.epsilon.core;

import java.util.stream.IntStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌山からの物理牌の取得順序と重複防止、赤牌を含むコピーの独立性、王牌の表示牌の順序を検証する。 */
public class PhysicalTilesTest {
  @Test
  public void consumesLiveWallWithoutRepeatingPhysicalTiles() {
    Wall wall = new Wall();
    wall.initWithFullWall(IntStream.range(0, 136).toArray());
    boolean[] seen = new boolean[136];
    // 完全な牌山からの復元は配牌52枚を消費した位置から始まる。
    for (int remaining = 70; remaining > 0; remaining--) {
      Assert.assertEquals(wall.remaining(), remaining);
      int physical = wall.draw();
      Assert.assertFalse(seen[physical]);
      seen[physical] = true;
    }
    Assert.assertTrue(wall.isExhausted());
    Assert.expectThrows(IllegalStateException.class, wall::draw);
  }

  @Test
  public void independentSnapshotPreservesTheRedTileIdentity() {
    Hand hand = new Hand();
    hand.addPhysicalTile(Tile.AKA_M5_ID);
    hand.addPhysicalTile(Tile.AKA_M5_ID + 1);
    Hand snapshot = hand.snapshotCopy();
    hand.removePhysicalTile(Tile.AKA_M5_ID);
    Assert.assertFalse(hand.hasAkaTile(Tile.M5));
    Assert.assertTrue(snapshot.hasAkaTile(Tile.M5));
    Assert.assertEquals(snapshot.count(Tile.M5), 2);
    snapshot.removePhysicalTile(Tile.AKA_M5_ID + 1);
    Assert.assertTrue(snapshot.containsPhysicalTile(Tile.AKA_M5_ID));
    Assert.assertTrue(hand.containsPhysicalTile(Tile.AKA_M5_ID + 1));
  }

  @Test
  public void deadWallIndicatorsKeepTenhouPhysicalOrder() {
    Wall wall = new Wall();
    wall.initWithFullWall(IntStream.range(0, 136).toArray());
    for (int slot = 0; slot < 5; slot++) {
      Assert.assertEquals(wall.doraIndicatorPhysicalTileIdAt(slot), 130 - slot * 2);
      Assert.assertEquals(wall.uraDoraIndicatorPhysicalTileIdAt(slot), 131 - slot * 2);
    }
    int previous = wall.remaining();
    wall.drawFromDeadWall();
    Assert.assertEquals(wall.remaining(), previous - 1);
  }
}
