package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Factory;
import org.testng.annotations.Test;

/** 物理牌ごとの未公開状態と、種類ごとの見えていない枚数を確かめる。 */
public class HiddenTileTest {

  private static final int STEPS = 30_000;

  private final Variant variant;
  private final TileSet set;
  private final int seats;

  @Factory(dataProvider = "variants")
  public HiddenTileTest(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
    seats = variant.seats();
  }

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test
  public void hiddenTilesAreTheTilesNoOneHasSeen() {
    RandomGames games = RandomGames.start(variant, 505, new GreedyChooser(variant));
    int uraEnds = 0;
    for (int step = 0; step < STEPS; step++) {
      uraEnds += assertHiddenMatchesWhatWasSeen(games.engine) ? 1 : 0;
      games.step();
    }
    Assert.assertTrue(uraEnds > 0, "no round end revealed ura indicators");
  }

  @Test
  public void hiddenRejectsTilesOutsideTheSet() {
    FullView full = RandomGames.start(variant, 707).engine.full();
    Assert.assertThrows(IllegalArgumentException.class, () -> full.hidden(Tiles.PHYSICAL_IDS));
    if (variant == Variant.TENHOU_3P)
      Assert.assertThrows(
          IllegalArgumentException.class, () -> full.hidden(Tiles.physicalId(Tiles.M5, 1)));
  }

  /**
   * 見えていない牌が、配牌・めくったドラ表示牌・偶然のイベントのどれにも出ていない牌と一致することを確かめる。
   *
   * @return 偶然のイベントに局末の裏ドラ表示牌があれば true
   */
  private boolean assertHiddenMatchesWhatWasSeen(Engine engine) {
    FullView full = engine.full();
    boolean[] seen = new boolean[Tiles.PHYSICAL_IDS];
    int[] tiles = (int[]) Fields.read(Fields.read(Fields.read(engine, "round"), "wall"), "tiles");
    for (int offset = 0; offset < seats; offset++) {
      for (int index = 0; index < Variant.HAND; index++) {
        seen[tiles[Wall.dealtPosition(variant, offset, index)]] = true;
      }
    }
    SeatView table = full.seat(0);
    for (int index = 0; index < table.indicatorCount(); index++) {
      seen[table.indicator(index)] = true;
    }
    ChanceLog chance = full.chance();
    boolean ura = false;
    for (int index = 0; index < chance.size(); index++) {
      seen[chance.tile(index)] = true;
      ura |= chance.kind(index) == ChanceKind.URA;
    }
    int[] hiddenOfKind = new int[Tiles.KINDS];
    for (int index = 0; index < set.size(); index++) {
      int tile = set.physicalId(index);
      Assert.assertEquals(full.hidden(tile), !seen[tile], "tile " + tile);
      hiddenOfKind[Tiles.kind(tile)] += seen[tile] ? 0 : 1;
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Assert.assertEquals(hiddenOfKind[kind], full.unseen(kind), "kind " + kind);
    }
    return ura;
  }
}
