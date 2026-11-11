package com.epsilon.agent.facts;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotSame;

import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 牌の関係の定数とその順、関係の表はネットワークとの契約なので、名前の並びと 34 × 34 の表をテストの定数で固定する。ネットワークは、
 * 重みの表と関係ごとのバイアスを関係の ordinal で引く。表の行はクエリの牌の種類、列はキーの牌の種類で、各文字は関係の ordinal を
 * 36 進の 1 桁で書いたもの。
 */
public class MahjongTileRelationTest {

  private static final List<String> NAMES =
      List.of(
          "SELF",
          "SAME_SUIT_DELTA_MINUS_8",
          "SAME_SUIT_DELTA_MINUS_7",
          "SAME_SUIT_DELTA_MINUS_6",
          "SAME_SUIT_DELTA_MINUS_5",
          "SAME_SUIT_DELTA_MINUS_4",
          "SAME_SUIT_DELTA_MINUS_3",
          "SAME_SUIT_DELTA_MINUS_2",
          "SAME_SUIT_DELTA_MINUS_1",
          "SAME_SUIT_DELTA_PLUS_1",
          "SAME_SUIT_DELTA_PLUS_2",
          "SAME_SUIT_DELTA_PLUS_3",
          "SAME_SUIT_DELTA_PLUS_4",
          "SAME_SUIT_DELTA_PLUS_5",
          "SAME_SUIT_DELTA_PLUS_6",
          "SAME_SUIT_DELTA_PLUS_7",
          "SAME_SUIT_DELTA_PLUS_8",
          "CROSS_SUIT_SAME_RANK",
          "CROSS_SUIT_OTHER",
          "WIND_PAIR",
          "DRAGON_PAIR",
          "WIND_DRAGON",
          "NUMBER_HONOR");

  private static final List<String> TABLE =
      List.of(
          "09ABCDEFGHIIIIIIIIHIIIIIIIIMMMMMMM",
          "809ABCDEFIHIIIIIIIIHIIIIIIIMMMMMMM",
          "7809ABCDEIIHIIIIIIIIHIIIIIIMMMMMMM",
          "67809ABCDIIIHIIIIIIIIHIIIIIMMMMMMM",
          "567809ABCIIIIHIIIIIIIIHIIIIMMMMMMM",
          "4567809ABIIIIIHIIIIIIIIHIIIMMMMMMM",
          "34567809AIIIIIIHIIIIIIIIHIIMMMMMMM",
          "234567809IIIIIIIHIIIIIIIIHIMMMMMMM",
          "123456780IIIIIIIIHIIIIIIIIHMMMMMMM",
          "HIIIIIIII09ABCDEFGHIIIIIIIIMMMMMMM",
          "IHIIIIIII809ABCDEFIHIIIIIIIMMMMMMM",
          "IIHIIIIII7809ABCDEIIHIIIIIIMMMMMMM",
          "IIIHIIIII67809ABCDIIIHIIIIIMMMMMMM",
          "IIIIHIIII567809ABCIIIIHIIIIMMMMMMM",
          "IIIIIHIII4567809ABIIIIIHIIIMMMMMMM",
          "IIIIIIHII34567809AIIIIIIHIIMMMMMMM",
          "IIIIIIIHI234567809IIIIIIIHIMMMMMMM",
          "IIIIIIIIH123456780IIIIIIIIHMMMMMMM",
          "HIIIIIIIIHIIIIIIII09ABCDEFGMMMMMMM",
          "IHIIIIIIIIHIIIIIII809ABCDEFMMMMMMM",
          "IIHIIIIIIIIHIIIIII7809ABCDEMMMMMMM",
          "IIIHIIIIIIIIHIIIII67809ABCDMMMMMMM",
          "IIIIHIIIIIIIIHIIII567809ABCMMMMMMM",
          "IIIIIHIIIIIIIIHIII4567809ABMMMMMMM",
          "IIIIIIHIIIIIIIIHII34567809AMMMMMMM",
          "IIIIIIIHIIIIIIIIHI234567809MMMMMMM",
          "IIIIIIIIHIIIIIIIIH123456780MMMMMMM",
          "MMMMMMMMMMMMMMMMMMMMMMMMMMM0JJJLLL",
          "MMMMMMMMMMMMMMMMMMMMMMMMMMMJ0JJLLL",
          "MMMMMMMMMMMMMMMMMMMMMMMMMMMJJ0JLLL",
          "MMMMMMMMMMMMMMMMMMMMMMMMMMMJJJ0LLL",
          "MMMMMMMMMMMMMMMMMMMMMMMMMMMLLLL0KK",
          "MMMMMMMMMMMMMMMMMMMMMMMMMMMLLLLK0K",
          "MMMMMMMMMMMMMMMMMMMMMMMMMMMLLLLKK0");

  @Test
  public void constantsKeepTheirNamesAndOrder() {
    assertEquals(Arrays.stream(MahjongTileRelation.values()).map(Enum::name).toList(), NAMES);
  }

  @Test
  public void relationIdsAreThePinnedTable() {
    long[] ids = MahjongTileRelation.relationIds();
    assertEquals(ids.length, Tiles.KINDS * Tiles.KINDS);
    for (int query = 0; query < Tiles.KINDS; query++) {
      for (int key = 0; key < Tiles.KINDS; key++) {
        long pinned = Character.digit(TABLE.get(query).charAt(key), Character.MAX_RADIX);
        assertEquals(ids[query * Tiles.KINDS + key], pinned, query + " to " + key);
        assertEquals(MahjongTileRelation.between(query, key).ordinal(), pinned, query + " to " + key);
      }
    }
  }

  @Test
  public void sameSuitDeltaIsKeyMinusQuery() {
    assertEquals(MahjongTileRelation.between(Tiles.P2, Tiles.P5), MahjongTileRelation.SAME_SUIT_DELTA_PLUS_3);
    assertEquals(MahjongTileRelation.between(Tiles.S9, Tiles.S1), MahjongTileRelation.SAME_SUIT_DELTA_MINUS_8);
    assertEquals(MahjongTileRelation.between(Tiles.M5, Tiles.S5), MahjongTileRelation.CROSS_SUIT_SAME_RANK);
    assertEquals(MahjongTileRelation.between(Tiles.TON, Tiles.CHUN), MahjongTileRelation.WIND_DRAGON);
    assertEquals(MahjongTileRelation.between(Tiles.HAKU, Tiles.M1), MahjongTileRelation.NUMBER_HONOR);
  }

  @Test
  public void relationIdsAreACopy() {
    long[] first = MahjongTileRelation.relationIds();
    first[0] = -1;
    long[] second = MahjongTileRelation.relationIds();
    assertNotSame(first, second);
    assertEquals(second[0], MahjongTileRelation.SELF.ordinal());
  }
}
