package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Factory;
import org.testng.annotations.Test;

/**
 * 誰にも見えていない物理牌と、物理牌を指定して求めるツモの点数を確かめる（四人打ちと三人打ち）。
 *
 * <ul>
 *   <li>見えていない牌は、配牌にも、めくったドラ表示牌にも、偶然のイベント（ツモ・嶺上ツモ・ドラ表示牌・局末の裏ドラ表示牌）にも
 *       出ていない牌と一致する。種類ごとの数は、その種類の見えていない枚数と一致する。
 *   <li>赤でない牌なら、ツモの点数は種類で求めた点数と同じ。赤の牌なら、役と符は同じで、翻だけが赤 1 枚分多い。実際に引いた牌
 *       なら、和了の合法手の点数から裏ドラを除いたものと同じ。
 *   <li>見えている牌（実際に引いた牌を除く）、ルールに無い牌、ツモの直後でない席、ほかのルールの作業領域は拒む。
 * </ul>
 */
public class HiddenTileTest {

  private static final int STEPS = 30_000;

  private final Variant variant;
  private final TileSet set;
  private final int seats;
  private final RuleWorkspace ws;
  private final WinScore byKind = new WinScore();
  private final WinScore byTile = new WinScore();

  @Factory(dataProvider = "variants")
  public HiddenTileTest(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
    seats = variant.seats();
    ws = new RuleWorkspace(variant);
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
  public void tileScoresCountRedFivesAndNoUra() {
    RandomGames games = RandomGames.start(variant, 606, new GreedyChooser(variant));
    int redWins = 0;
    int drawnWins = 0;
    int riichiDrawnWins = 0;
    for (int step = 0; step < STEPS; step++) {
      Engine engine = games.engine;
      if (games.boundary() == Boundary.DECISIONS && afterDraw(engine)) {
        Decisions decisions = engine.decisions();
        int seat = decisions.seat(0);
        FullView full = engine.full();
        for (int index = 0; index < set.size(); index++) {
          int tile = set.physicalId(index);
          if (full.hidden(tile)) {
            redWins += assertLikeTheKind(full, seat, tile) ? 1 : 0;
          }
        }
        int drawn = full.chance().tile(full.chance().size() - 1);
        boolean won = full.tsumoScoreTile(seat, drawn, ws, byTile);
        Assert.assertEquals(won, decisions.hasWin(0));
        if (won) {
          drawnWins++;
          WinScore legal = decisions.win(0);
          riichiDrawnWins += legal.ura() > 0 ? 1 : 0;
          Assert.assertEquals(byTile.ura(), 0, byTile.toString());
          Assert.assertEquals(byTile.yaku(), legal.yaku(), byTile + " against " + legal);
          Assert.assertEquals(byTile.fu(), legal.fu(), byTile + " against " + legal);
          Assert.assertEquals(byTile.dora(), legal.dora(), byTile + " against " + legal);
          Assert.assertEquals(byTile.aka(), legal.aka(), byTile + " against " + legal);
          Assert.assertEquals(byTile.nuki(), legal.nuki(), byTile + " against " + legal);
          Assert.assertEquals(byTile.yakuman(), legal.yakuman(), byTile + " against " + legal);
          if (legal.yakuman() == 0) {
            Assert.assertEquals(byTile.han(), legal.han() - legal.ura(), byTile + " against " + legal);
          }
        }
      }
      games.step();
    }
    Assert.assertTrue(redWins > 0, "no hidden red five completed a hand");
    Assert.assertTrue(drawnWins > 20, drawnWins + " tsumo wins");
    Assert.assertTrue(riichiDrawnWins > 0, "no tsumo win had ura");
  }

  @Test
  public void seenTilesAndOtherSeatsAreRejected() {
    RandomGames games = RandomGames.start(variant, 707, new GreedyChooser(variant));
    while (games.boundary() != Boundary.DECISIONS || !afterDraw(games.engine)) {
      games.step();
    }
    Engine engine = games.engine;
    FullView full = engine.full();
    int seat = engine.decisions().seat(0);
    int seen = -1;
    for (int index = 0; index < set.size() && seen < 0; index++) {
      int tile = set.physicalId(index);
      if (full.seat(seat).holds(tile) && tile != full.chance().tile(full.chance().size() - 1)) {
        seen = tile;
      }
    }
    int held = seen;
    Assert.assertThrows(IllegalArgumentException.class, () -> full.tsumoScoreTile(seat, held, ws, byTile));
    Assert.assertThrows(IllegalArgumentException.class, () -> full.tsumoScoreTile(seat, -1, ws, byTile));
    Assert.assertThrows(IllegalArgumentException.class, () -> full.hidden(Tiles.PHYSICAL_IDS));
    if (variant == Variant.TENHOU_3P) {
      int manzu = Tiles.physicalId(Tiles.M5, 1);
      Assert.assertThrows(IllegalArgumentException.class, () -> full.hidden(manzu));
      Assert.assertThrows(IllegalArgumentException.class, () -> full.tsumoScoreTile(seat, manzu, ws, byTile));
    }
    int hidden = firstHidden(full);
    Assert.assertThrows(
        IllegalStateException.class, () -> full.tsumoScoreTile((seat + 1) % seats, hidden, ws, byTile));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> full.tsumoScoreTile(seat, hidden, new RuleWorkspace(other(variant)), byTile));
  }

  /**
   * 物理牌を指定して求めた点数を、種類で求めた点数と比べる。
   *
   * @return 赤の牌で和了の形になったら true
   */
  private boolean assertLikeTheKind(FullView full, int seat, int tile) {
    boolean kindWins = full.tsumoScore(seat, Tiles.kind(tile), ws, byKind);
    boolean tileWins = full.tsumoScoreTile(seat, tile, ws, byTile);
    Assert.assertEquals(tileWins, kindWins, "tile " + tile);
    if (!kindWins) {
      return false;
    }
    Assert.assertEquals(byTile.ura(), 0, byTile.toString());
    if (!set.red(tile)) {
      Assert.assertEquals(byTile.toString(), byKind.toString(), "tile " + tile);
      return false;
    }
    Assert.assertEquals(byTile.aka(), byKind.aka() + 1, byTile + " against " + byKind);
    Assert.assertEquals(byTile.yaku(), byKind.yaku(), byTile + " against " + byKind);
    Assert.assertEquals(byTile.fu(), byKind.fu(), byTile + " against " + byKind);
    Assert.assertEquals(byTile.dora(), byKind.dora(), byTile + " against " + byKind);
    if (byKind.yakuman() == 0) {
      Assert.assertEquals(byTile.han(), byKind.han() + 1, byTile + " against " + byKind);
    }
    return true;
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

  /** 直前のイベントが、判断する席のツモか。 */
  private static boolean afterDraw(Engine engine) {
    Decisions decisions = engine.decisions();
    TurnEvent last = engine.view(decisions.seat(0)).last();
    return decisions.kind(0) == EngineDecisionKind.TURN
        && last.kind() == TurnEvent.Kind.DRAW
        && last.seat() == decisions.seat(0);
  }

  private int firstHidden(FullView full) {
    for (int index = 0; index < set.size(); index++) {
      if (full.hidden(set.physicalId(index))) {
        return set.physicalId(index);
      }
    }
    throw new AssertionError("No hidden tile");
  }

  private static Variant other(Variant variant) {
    return variant == Variant.TENHOU_4P ? Variant.TENHOU_3P : Variant.TENHOU_4P;
  }
}
