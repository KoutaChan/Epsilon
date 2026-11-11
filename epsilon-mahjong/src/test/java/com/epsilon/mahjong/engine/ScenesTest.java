package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Scenes で作る牌山を確かめる。置いた牌は位置の式で決まる所に入り、残りの位置は乱数の種で決まる並びで埋まる。エンジンは、置いた
 * 配牌とツモのとおりに局を始める。読めない表記と、置けない牌は拒む。
 */
public class ScenesTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  @Test
  public void placedTilesSitAtTheirWallPositions() {
    Scenes scenes =
        Scenes.at(VARIANT, 1, 0, 0, 25_000, 25_000, 25_000, 25_000)
            .hand(1, "123m456p789s1122z")
            .draws("0m")
            .drawAt(5, "3z")
            .rinshan("4z")
            .indicator("5z")
            .indicator(1, "6z")
            .ura(0, "7z");
    int[] wall = scenes.wall();
    Face[] hand = Scenes.faces("123m456p789s1122z");
    for (int index = 0; index < Variant.HAND; index++) {
      int tile = wall[Wall.dealtPosition(VARIANT, 0, index)];
      Assert.assertEquals(Face.of(Tiles.kind(tile), VARIANT.tiles().red(tile)), hand[index], "dealt " + index);
    }
    int first = wall[Wall.livePosition(VARIANT, 0)];
    Assert.assertEquals(Tiles.kind(first), Tiles.M5);
    Assert.assertTrue(VARIANT.tiles().red(first));
    Assert.assertEquals(Tiles.kind(wall[Wall.livePosition(VARIANT, 5)]), Tiles.SHA);
    Assert.assertEquals(Tiles.kind(wall[Wall.rinshanPosition(VARIANT, 1)]), Tiles.PEI);
    Assert.assertEquals(Tiles.kind(wall[Wall.indicatorPosition(VARIANT, 0)]), Tiles.HAKU);
    Assert.assertEquals(Tiles.kind(wall[Wall.indicatorPosition(VARIANT, 1)]), Tiles.HATSU);
    Assert.assertEquals(Tiles.kind(wall[Wall.uraPosition(VARIANT, 0)]), Tiles.CHUN);

    TileSet set = VARIANT.tiles();
    int[] every = new int[set.size()];
    for (int index = 0; index < every.length; index++) {
      every[index] = set.physicalId(index);
    }
    int[] sorted = wall.clone();
    Arrays.sort(sorted);
    Assert.assertEquals(sorted, every, "the wall holds every tile once");
    Assert.assertEquals(scenes.wall(), wall, "the default fill is the same every time");
    int[] other = scenes.wall(Scenes.FILL_SEED + 1);
    Assert.assertFalse(Arrays.equals(other, wall), "another seed fills another order");
    Assert.assertEquals(other[Wall.uraPosition(VARIANT, 0)], wall[Wall.uraPosition(VARIANT, 0)]);

    Engine engine = new Engine(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    Assert.assertEquals(scenes.start(engine, ws), Boundary.DECISIONS);
    Assert.assertEquals(engine.decisions().seat(0), 1, "the dealer of round 1 draws first");
    SeatView view = engine.view(1);
    Assert.assertEquals(view.count(Tiles.M5), 1);
    Assert.assertEquals(view.count(Tiles.TON), 2);
    Assert.assertEquals(view.indicator(0), wall[Wall.indicatorPosition(VARIANT, 0)]);
  }

  @Test
  public void badNotationsAndTilesThatCannotBePlacedAreRejected() {
    Assert.assertEquals(Scenes.faces("05m7z"), new Face[] {Face.M5_RED, Face.M5, Face.CHUN});
    Assert.assertThrows(IllegalArgumentException.class, () -> Scenes.faces("12x"));
    Assert.assertThrows(IllegalArgumentException.class, () -> Scenes.faces("123m45"));
    Assert.assertThrows(IllegalArgumentException.class, () -> Scenes.faces("0z"));
    Assert.assertThrows(IllegalArgumentException.class, () -> Scenes.first(VARIANT).hand(0, "123m456p789s112z"));
    Assert.assertThrows(IllegalArgumentException.class, () -> Scenes.first(VARIANT).indicator("1z").indicator("2z"));
    Assert.assertThrows(IllegalArgumentException.class, () -> Scenes.first(VARIANT).draws("1z1z1z1z1z"));
    Assert.assertThrows(IllegalArgumentException.class, () -> Scenes.first(VARIANT).indicator("12z"));
    Assert.assertThrows(IllegalArgumentException.class, () -> Scenes.first(Variant.TENHOU_3P).draws("2m"));
  }
}
