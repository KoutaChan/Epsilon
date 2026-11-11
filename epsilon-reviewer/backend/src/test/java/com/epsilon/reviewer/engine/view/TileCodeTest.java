package com.epsilon.reviewer.engine.view;

import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 37 種の牌の記号、記号の読み取りと検査、手牌を並べる順、物理牌の記号を確かめる。 */
public class TileCodeTest {
  @Test
  public void thirtySevenCodesAreTheMjaiFaces() {
    Set<String> codes = new HashSet<>();
    for (Face face : Face.values()) {
      String code = TileCode.of(face.kind(), face.red());
      Assert.assertEquals(code, face.mjai());
      Assert.assertTrue(TileCode.isValid(code), code);
      Assert.assertSame(TileCode.face(code), face, code);
      Assert.assertEquals(TileCode.kind(code), face.kind(), code);
      Assert.assertEquals(TileCode.red(code), face.red(), code);
      codes.add(code);
    }
    Assert.assertEquals(codes.size(), 37);
    Assert.assertEquals(TileCode.of(Tiles.P5, true), "5pr");
    Assert.assertEquals(TileCode.of(Tiles.TON, false), "E");
    Assert.assertSame(TileCode.of(Tiles.P5, true), TileCode.of(Tiles.P5, true));
  }

  @Test
  public void onlyTheThirtySevenCodesAreValid() {
    for (String code :
        List.of("0p", "0m", "5z", "1z", "7z", "6sr", "5", "E5", "5mR", "", "10m", "e", " 5m"))
      Assert.assertFalse(TileCode.isValid(code), code);
    Assert.assertFalse(TileCode.isValid(null));
    for (String code : List.of("0p", "5z", "1z"))
      Assert.expectThrows(IllegalArgumentException.class, () -> TileCode.face(code));
    Assert.expectThrows(IllegalArgumentException.class, () -> TileCode.kind("0p"));
    Assert.expectThrows(IllegalArgumentException.class, () -> TileCode.red("1z"));
    Assert.expectThrows(IllegalArgumentException.class, () -> TileCode.order("0s"));
    Assert.expectThrows(IllegalArgumentException.class, () -> TileCode.face(null));
    Assert.expectThrows(IllegalArgumentException.class, () -> TileCode.of(Tiles.M4, true));
  }

  @Test
  public void orderSortsBySuitNumberAndPutsTheRedFiveFirst() {
    List<String> sorted = new ArrayList<>(List.of("C", "1p", "5m", "E", "5mr", "9s", "4m", "P"));
    sorted.sort((a, b) -> Integer.compare(TileCode.order(a), TileCode.order(b)));
    Assert.assertEquals(sorted, List.of("4m", "5mr", "5m", "1p", "9s", "E", "P", "C"));
    List<String> all = new ArrayList<>();
    for (Face face : Face.values()) all.add(face.mjai());
    all.sort((a, b) -> Integer.compare(TileCode.order(a), TileCode.order(b)));
    for (int i = 1; i < all.size(); i++)
      Assert.assertTrue(TileCode.order(all.get(i - 1)) < TileCode.order(all.get(i)), all.get(i));
  }

  /** 物理牌の記号は、牌のセットごとの赤の決まりに従う（三人打ちは萬子に赤が無い）。 */
  @Test
  public void physicalTilesFollowTheirTileSet() {
    int redMan = Tiles.physicalId(Tiles.M5, Tiles.RED_COPY);
    int redPin = Tiles.physicalId(Tiles.P5, Tiles.RED_COPY);
    Assert.assertEquals(TileCode.of(TileSet.FULL, redMan), "5mr");
    Assert.assertEquals(TileCode.of(TileSet.FULL, redPin), "5pr");
    Assert.assertEquals(TileCode.of(TileSet.FULL, Tiles.physicalId(Tiles.P5, 1)), "5p");
    Assert.assertEquals(TileCode.of(TileSet.SANMA, redPin), "5pr");
    Assert.assertEquals(TileCode.of(TileSet.FULL, Tiles.physicalId(Tiles.CHUN, 3)), "C");
    int reds = 0;
    for (int index = 0; index < TileSet.FULL.size(); index++) {
      int id = TileSet.FULL.physicalId(index);
      String code = TileCode.of(TileSet.FULL, id);
      Assert.assertEquals(TileCode.kind(code), Tiles.kind(id), code);
      if (TileCode.red(code)) reds++;
    }
    Assert.assertEquals(reds, 3);
  }
}
