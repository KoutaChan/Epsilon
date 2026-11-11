package com.epsilon.mahjong.tile;

import java.util.EnumSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌の 37 通りの表記（赤 5 を含む）と、mjai・雀魂の表記との間の変換。 */
public class FaceTest {

  /** 赤でない表記の種類は、列挙の宣言の順だけで決まる。だから Tiles の同じ名前の定数と一致するかを確かめる。 */
  @Test
  public void plainFacesCarryTheKindOfTheirTilesConstant() throws ReflectiveOperationException {
    int plain = 0;
    for (Face face : Face.values()) {
      if (!face.red()) {
        Assert.assertEquals(face.kind(), Tiles.class.getField(face.name()).getInt(null), face.name());
        Assert.assertSame(Face.of(face.kind(), false), face);
        plain++;
      }
    }
    Assert.assertEquals(plain, Tiles.KINDS);
    Assert.assertEquals(Face.values().length, Tiles.KINDS + Tiles.NUMBER_SUITS);
  }

  @Test
  public void redFacesAreTheFives() {
    Assert.assertSame(Face.of(Tiles.M5, true), Face.M5_RED);
    Assert.assertSame(Face.of(Tiles.P5, true), Face.P5_RED);
    Assert.assertSame(Face.of(Tiles.S5, true), Face.S5_RED);
    for (Face face : new Face[] {Face.M5_RED, Face.P5_RED, Face.S5_RED}) {
      Assert.assertTrue(face.red());
      Assert.assertTrue(Tiles.contains(Tiles.FIVES, face.kind()));
    }
    Assert.assertThrows(IllegalArgumentException.class, () -> Face.of(Tiles.M4, true));
    Assert.assertThrows(IllegalArgumentException.class, () -> Face.of(Tiles.CHUN, true));
    Assert.assertThrows(IllegalArgumentException.class, () -> Face.of(Tiles.KINDS, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> Face.of(-1, false));
  }

  @Test
  public void notationsOfBothFormats() {
    assertNotation(Face.M1, "1m", "1m");
    assertNotation(Face.P9, "9p", "9p");
    assertNotation(Face.S5, "5s", "5s");
    assertNotation(Face.M5_RED, "5mr", "0m");
    assertNotation(Face.P5_RED, "5pr", "0p");
    assertNotation(Face.S5_RED, "5sr", "0s");
    assertNotation(Face.TON, "E", "1z");
    assertNotation(Face.NAN, "S", "2z");
    assertNotation(Face.SHA, "W", "3z");
    assertNotation(Face.PEI, "N", "4z");
    assertNotation(Face.HAKU, "P", "5z");
    assertNotation(Face.HATSU, "F", "6z");
    assertNotation(Face.CHUN, "C", "7z");
  }

  @Test
  public void everyNotationParsesBack() {
    for (Face face : Face.values()) {
      Assert.assertSame(Face.parse(face.mjai()), face);
    }
    EnumSet<Face> majsoul = EnumSet.noneOf(Face.class);
    for (String suit : new String[] {"m", "p", "s"}) {
      for (int number = 0; number <= 9; number++) {
        Assert.assertTrue(majsoul.add(Face.parse(number + suit)), number + suit);
      }
    }
    for (int honor = 1; honor <= 7; honor++) {
      Assert.assertTrue(majsoul.add(Face.parse(honor + "z")), honor + "z");
    }
    Assert.assertEquals(majsoul, EnumSet.allOf(Face.class));
  }

  @Test
  public void unknownNotationsAreRejected() {
    for (String text : new String[] {"", "0z", "8z", "5zr", "4mr", "10m", "e", "5m ", "?", null}) {
      Assert.assertThrows(IllegalArgumentException.class, () -> Face.parse(text));
    }
  }

  private static void assertNotation(Face face, String mjai, String majsoul) {
    Assert.assertEquals(face.mjai(), mjai, face.name());
    Assert.assertSame(Face.parse(majsoul), face, face.name());
  }
}
