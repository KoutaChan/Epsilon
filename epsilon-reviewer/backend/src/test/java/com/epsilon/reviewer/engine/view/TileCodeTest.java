package com.epsilon.reviewer.engine.view;

import com.epsilon.core.Tile;
import com.epsilon.replay.TileNotation;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 37 の牌の記号、読み・確かめ、手牌の並びの順を確かめる。 */
public class TileCodeTest {
  @Test
  public void thirtySevenCodesMatchTheCommonNotation() {
    Set<String> codes = new HashSet<>();
    for (int id = 0; id < Tile.NUM_TILES; id++) {
      String code = TileCode.of(Tile.typeOf(id), Tile.isAka(id));
      Assert.assertEquals(code, TileNotation.formatPhysicalTile(id));
      Assert.assertTrue(TileCode.isValid(code), code);
      Assert.assertEquals(TileCode.type(code), Tile.typeOf(id), code);
      Assert.assertEquals(TileCode.red(code), Tile.isAka(id), code);
      codes.add(code);
    }
    Assert.assertEquals(codes.size(), 37);
    Assert.assertSame(TileCode.of(Tile.P5, true), TileCode.of(Tile.P5, true));
  }

  @Test
  public void onlyTheThirtySevenCodesAreValid() {
    for (String code : List.of("0p", "5z", "6sr", "1z", "5", "E5", "5mR", "", "10m", "e"))
      Assert.assertFalse(TileCode.isValid(code), code);
    Assert.assertFalse(TileCode.isValid(null));
    Assert.expectThrows(IllegalArgumentException.class, () -> TileCode.type("0p"));
    Assert.expectThrows(IllegalArgumentException.class, () -> TileCode.of(Tile.M4, true));
  }

  @Test
  public void orderSortsBySuitNumberAndPutsTheRedFiveFirst() {
    List<String> sorted = new ArrayList<>(List.of("C", "1p", "5m", "E", "5mr", "9s", "4m", "P"));
    sorted.sort((a, b) -> Integer.compare(TileCode.order(a), TileCode.order(b)));
    Assert.assertEquals(sorted, List.of("4m", "5mr", "5m", "1p", "9s", "E", "P", "C"));
  }
}
