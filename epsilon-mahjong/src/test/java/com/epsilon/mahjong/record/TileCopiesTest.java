package com.epsilon.mahjong.record;

import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 表記ごとの牌の枚数と、割り当てを外した物理牌を割り当て直す動きを確かめる。 */
public class TileCopiesTest {

  /** 四人打ちは赤の 5 が 1 枚、赤でない 5 が 3 枚、ほかが 4 枚。三人打ちは 2〜8 萬が 0 枚で、赤は 5p・5s だけ。枚数の合計は牌の総数になる。 */
  @Test
  public void copiesFollowTheTileSet() {
    Variant four = Variant.TENHOU_4P;
    Assert.assertEquals(TileCopies.copies(four, Face.M5_RED), 1);
    Assert.assertEquals(TileCopies.copies(four, Face.M5), 3);
    Assert.assertEquals(TileCopies.copies(four, Face.S5), 3);
    Assert.assertEquals(TileCopies.copies(four, Face.M1), 4);
    Assert.assertEquals(TileCopies.copies(four, Face.CHUN), 4);
    Variant three = Variant.TENHOU_3P;
    Assert.assertEquals(TileCopies.copies(three, Face.M5_RED), 0);
    Assert.assertEquals(TileCopies.copies(three, Face.M5), 0);
    Assert.assertEquals(TileCopies.copies(three, Face.M2), 0);
    Assert.assertEquals(TileCopies.copies(three, Face.M9), 4);
    Assert.assertEquals(TileCopies.copies(three, Face.P5_RED), 1);
    Assert.assertEquals(TileCopies.copies(three, Face.P5), 3);
    Assert.assertEquals(TileCopies.copies(three, Face.PEI), 4);
    for (Variant variant : Variant.values()) {
      int total = 0;
      for (Face face : Face.values()) {
        total += TileCopies.copies(variant, face);
      }
      Assert.assertEquals(total, variant.tiles().size(), variant.toString());
    }
  }

  /** 割り当てを外した物理牌は手とツモから消え、その後は番号の小さい順に割り当て直される。 */
  @Test
  public void aReleasedCopyIsAssignedAgain() {
    TileCopies copies = new TileCopies(Variant.TENHOU_4P);
    int dealt = copies.deal(0, Face.M1);
    int drawn = copies.draw(0, Face.M1);
    Assert.assertEquals(dealt, Tiles.physicalId(Tiles.M1, 0));
    Assert.assertEquals(drawn, Tiles.physicalId(Tiles.M1, 1));
    copies.release(dealt);
    Assert.assertEquals(copies.count(0, Tiles.M1), 1);
    Assert.assertEquals(copies.drawn(0), drawn);
    Assert.assertEquals(copies.reveal(Face.M1), dealt);
    copies.release(drawn);
    Assert.assertEquals(copies.count(0, Tiles.M1), 0);
    Assert.assertEquals(copies.drawn(0), TurnEvent.NONE);
    Assert.assertEquals(copies.deal(1, Face.M1), drawn);

    int red = copies.reveal(Face.P5_RED);
    Assert.assertEquals(red, Tiles.physicalId(Tiles.P5, Tiles.RED_COPY));
    Assert.assertThrows(IllegalArgumentException.class, () -> copies.reveal(Face.P5_RED));
    copies.release(red);
    Assert.assertEquals(copies.reveal(Face.P5_RED), red);
  }

  /** まだ割り当てていない物理牌（一度外した牌も含む）、そのルールで使わない牌、範囲の外の番号は外せない。 */
  @Test
  public void onlyAnAssignedTileIsReleased() {
    TileCopies copies = new TileCopies(Variant.TENHOU_3P);
    Assert.assertThrows(IllegalArgumentException.class, () -> copies.release(Tiles.physicalId(Tiles.M1, 0)));
    Assert.assertThrows(IllegalArgumentException.class, () -> copies.release(Tiles.physicalId(Tiles.M5, 1)));
    Assert.assertThrows(IllegalArgumentException.class, () -> copies.release(Tiles.PHYSICAL_IDS));
    int north = copies.deal(2, Face.PEI);
    copies.release(north);
    Assert.assertThrows(IllegalArgumentException.class, () -> copies.release(north));
  }
}
