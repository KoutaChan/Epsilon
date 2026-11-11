package com.epsilon.mahjong.rules;

import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.Test;

/** ルールのフィールドと、席の数から求める値を確かめる。 */
public class VariantTest {

  @Test
  public void fourPlayers() {
    Variant variant = Variant.TENHOU_4P;
    Assert.assertEquals(variant.seats(), 4);
    Assert.assertEquals(variant.seatMask(), 0b1111);
    Assert.assertSame(variant.tiles(), TileSet.FULL);
    Assert.assertEquals(variant.startPoints(), 25_000);
    Assert.assertEquals(variant.returnPoints(), 30_000);
    Assert.assertEquals(variant.liveWall(), 70);
    Assert.assertEquals(variant.rinshanTiles(), 4);
    Assert.assertEquals(variant.ronHonba(), 300);
    Assert.assertEquals(variant.notenTotal(), 3_000);
    Assert.assertEquals(variant.riichiMinWall(), 4);
    Assert.assertEquals(variant.lastRound(), 7);
    Assert.assertEquals(variant.finalRound(), 11);
    Assert.assertEquals(variant.maxDiscards(), 86);
    Assert.assertEquals(variant.maxDrawsPerSeat(), 18);
    Assert.assertEquals(variant.maxPublicEvents(), 106);
    Assert.assertTrue(variant.has(RuleFlag.CHI));
    Assert.assertFalse(variant.has(RuleFlag.NUKI));
    Assert.assertFalse(variant.has(RuleFlag.NORTH_CALLS));
    Assert.assertEquals(variant.meldableKinds(), Tiles.ALL_KINDS);
    for (AbortiveDraw draw : AbortiveDraw.values()) {
      Assert.assertTrue(variant.aborts(draw), draw.name());
    }
  }

  @Test
  public void threePlayers() {
    Variant variant = Variant.TENHOU_3P;
    Assert.assertEquals(variant.seats(), 3);
    Assert.assertEquals(variant.seatMask(), 0b111);
    Assert.assertSame(variant.tiles(), TileSet.SANMA);
    Assert.assertEquals(variant.startPoints(), 35_000);
    Assert.assertEquals(variant.returnPoints(), 40_000);
    Assert.assertEquals(variant.liveWall(), 55);
    Assert.assertEquals(variant.rinshanTiles(), 8);
    Assert.assertEquals(variant.ronHonba(), 200);
    Assert.assertEquals(variant.notenTotal(), 2_000);
    Assert.assertEquals(variant.riichiMinWall(), 3);
    Assert.assertEquals(variant.lastRound(), 5);
    Assert.assertEquals(variant.finalRound(), 8);
    Assert.assertEquals(variant.maxDiscards(), 67);
    Assert.assertEquals(variant.maxDrawsPerSeat(), 19);
    Assert.assertEquals(variant.maxPublicEvents(), 87);
    Assert.assertFalse(variant.has(RuleFlag.CHI));
    Assert.assertTrue(variant.has(RuleFlag.NUKI));
    Assert.assertTrue(variant.has(RuleFlag.NORTH_CALLS));
    Assert.assertEquals(variant.meldableKinds(), TileSet.SANMA.kinds());
    Assert.assertTrue(variant.aborts(AbortiveDraw.NINE_TERMINALS));
    Assert.assertTrue(variant.aborts(AbortiveDraw.FOUR_KANS));
    Assert.assertFalse(variant.aborts(AbortiveDraw.FOUR_WINDS));
    Assert.assertFalse(variant.aborts(AbortiveDraw.FOUR_RIICHI));
    Assert.assertFalse(variant.aborts(AbortiveDraw.TRIPLE_RON));
  }

  @Test
  public void wallAndEventsFitTheVariant() {
    for (Variant variant : Variant.values()) {
      Assert.assertEquals(
          variant.liveWall() + Variant.DEAD_WALL + Variant.HAND * variant.seats(), variant.tiles().size());
      Assert.assertTrue(variant.maxPublicEvents() <= Byte.MAX_VALUE, variant.name());
      Assert.assertEquals(variant.finalRound() + 1, 3 * variant.seats());
    }
  }

  /** 打牌の回数の上限は、山の枚数に各席が 4 回鳴いた後の打牌を足した数。1 席のツモの上限は、山の枚数を席の数で割って切り上げた数。 */
  @Test
  public void discardAndDrawLimitsComeFromTheLiveWall() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      int nuki = variant.has(RuleFlag.NUKI) ? Tiles.COPIES : 0;
      Assert.assertEquals(variant.maxDiscards(), variant.liveWall() + Variant.MAX_MELDS * seats, variant.name());
      Assert.assertEquals(
          variant.maxPublicEvents(),
          variant.maxDiscards() + Variant.MAX_MELDS * seats + Variant.MAX_KANS + nuki,
          variant.name());
      Assert.assertTrue(variant.maxDrawsPerSeat() * seats >= variant.liveWall(), variant.name());
      Assert.assertTrue((variant.maxDrawsPerSeat() - 1) * seats < variant.liveWall(), variant.name());
    }
  }

  @Test
  public void roundNumbersGiveDealerAndWinds() {
    Variant three = Variant.TENHOU_3P;
    Assert.assertEquals(three.dealer(0), 0);
    Assert.assertEquals(three.roundWind(0), Tiles.TON);
    Assert.assertEquals(three.dealer(3), 0);
    Assert.assertEquals(three.roundWind(3), Tiles.NAN);
    Assert.assertEquals(three.dealer(5), 2);
    Assert.assertEquals(three.dealer(8), 2);
    Assert.assertEquals(three.roundWind(8), Tiles.SHA);
    Assert.assertEquals(three.seatWind(4, 1), Tiles.TON);
    Assert.assertEquals(three.seatWind(4, 2), Tiles.NAN);
    Assert.assertEquals(three.seatWind(4, 0), Tiles.SHA);
    Variant four = Variant.TENHOU_4P;
    Assert.assertEquals(four.dealer(7), 3);
    Assert.assertEquals(four.roundWind(7), Tiles.NAN);
    Assert.assertEquals(four.roundWind(11), Tiles.SHA);
    Assert.assertEquals(four.seatWind(1, 0), Tiles.PEI);
    Assert.assertEquals(four.seatWind(1, 1), Tiles.TON);
  }
}
