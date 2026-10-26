package com.epsilon.core;

import java.util.Arrays;
import java.util.HashSet;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 見えない牌の並べ替えの不変条件。 */
public class WallReshuffleTest {
  private static final int DEALT = 52;

  private static Wall dealt(long seed) {
    Wall wall = new Wall(seed);
    for (int i = 0; i < DEALT; i++) wall.draw();
    return wall;
  }

  private static int[] positions(Wall wall) {
    int[] tiles = new int[136];
    for (int position = 0; position < tiles.length; position++)
      tiles[position] = wall.physicalTileIdAt(position);
    return tiles;
  }

  /** 槓1回・表示牌2枚の公開で見えている位置（配牌と10枚のツモ、使った嶺上牌、公開済みの表示牌 130・128）。裏ドラは見えない。 */
  private static boolean seenAfterOneKan(int position) {
    return position < DEALT + 10 || position == 135 || position == 130 || position == 128;
  }

  @Test
  public void keepsSeenTilesAndPermutesTheRest() {
    Wall wall = dealt(11L);
    for (int i = 0; i < 10; i++) wall.draw();
    wall.drawFromDeadWall();
    int[] before = positions(wall);
    wall.reshuffleUnseen(99L, 2);
    int[] after = positions(wall);
    int moved = 0;
    for (int position = 0; position < before.length; position++) {
      if (seenAfterOneKan(position)) Assert.assertEquals(after[position], before[position]);
      else if (after[position] != before[position]) moved++;
    }
    Assert.assertTrue(moved > 40, "moved=" + moved);
    int[] sortedBefore = before.clone(), sortedAfter = after.clone();
    Arrays.sort(sortedBefore);
    Arrays.sort(sortedAfter);
    Assert.assertEquals(sortedAfter, sortedBefore);
    Assert.assertEquals(wall.remaining(), Wall.LIVE_WALL_SIZE - 1 - DEALT - 10);
  }

  @Test
  public void isDeterministicInTheSeed() {
    Wall first = dealt(5L), second = new Wall(first), third = new Wall(first);
    first.reshuffleUnseen(1L, 1);
    second.reshuffleUnseen(1L, 1);
    third.reshuffleUnseen(2L, 1);
    Assert.assertEquals(positions(first), positions(second));
    Assert.assertNotEquals(positions(first), positions(third));
  }

  @Test
  public void copyIsUnaffected() {
    Wall wall = dealt(8L);
    Wall copy = new Wall(wall);
    int[] before = positions(copy);
    wall.reshuffleUnseen(3L, 1);
    Assert.assertEquals(positions(copy), before);
  }

  /** 次のツモは見えない牌（王牌の残りと裏ドラを含む 83枚）のどれもが同じ確率になる。 */
  @Test
  public void nextDrawIsUniformOverUnseenTiles() {
    Wall base = dealt(21L);
    var unseen = new HashSet<Integer>();
    for (int position = DEALT; position < 136; position++)
      if (position != 130) unseen.add(base.physicalTileIdAt(position));
    assertUniform(unseen, seed -> {
      Wall wall = new Wall(base);
      wall.reshuffleUnseen(seed, 1);
      return wall.draw();
    });
  }

  /** 席別ツモ列でも、鳴きで手番が飛んだ後の次のツモが見えない牌の上で一様になる。 */
  @Test
  public void seatStreamDrawIsUniformAfterSkippedTurn() {
    Wall base = dealt(34L);
    base.enableSeatStreams(0);
    var seen = new HashSet<Integer>();
    seen.add(base.draw(0));
    seen.add(base.draw(1));
    var unseen = new HashSet<Integer>();
    for (int position = DEALT; position < 136; position++) {
      int tile = base.physicalTileIdAt(position);
      if (position != 130 && !seen.contains(tile)) unseen.add(tile);
    }
    assertUniform(unseen, seed -> {
      Wall wall = new Wall(base);
      wall.reshuffleUnseen(seed, 1);
      return wall.draw(3);
    });
  }

  @Test
  public void seatStreamsDrawEachTileOnceAfterReshuffle() {
    var random = new SplittableRandom(7);
    for (int trial = 0; trial < 100; trial++) {
      Wall wall = dealt(trial);
      wall.enableSeatStreams(trial % 4);
      var drawn = new HashSet<Integer>();
      for (int i = 0; i < 9; i++) Assert.assertTrue(drawn.add(wall.draw(random.nextInt(4))));
      wall.reshuffleUnseen(trial * 31L, 1);
      while (!wall.isExhausted()) Assert.assertTrue(drawn.add(wall.draw(random.nextInt(4))));
      Assert.assertEquals(drawn.size(), Wall.LIVE_WALL_SIZE - DEALT);
    }
  }

  @Test(expectedExceptions = IllegalStateException.class)
  public void rejectsReconstructedWall() {
    Wall wall = new Wall(1L);
    wall.initForReconstruction(DEALT);
    wall.reshuffleUnseen(1L, 0);
  }

  @Test
  public void gameStateKeepsHandsAndDora() {
    GameState state = new GameState(13L);
    state.startRound(0, 0, 0, 0);
    state.dealInitialHands();
    GameState copy = state.snapshotCopy();
    copy.reshuffleUnseen(4L);
    for (int seat = 0; seat < 4; seat++)
      Assert.assertEquals(
          copy.hand(seat).copyConcealedTileCounts(), state.hand(seat).copyConcealedTileCounts());
    Assert.assertEquals(copy.doraState().indicatorCount(), state.doraState().indicatorCount());
    Assert.assertEquals(
        copy.doraState().indicatorTileType(0), state.doraState().indicatorTileType(0));
    Assert.assertEquals(copy.remainingWallTiles(), state.remainingWallTiles());
    // 公開済みの表示牌の裏ドラも並べ替わり、点数計算が読む DoraState の裏ドラはその牌に写し直される。
    Assert.assertEquals(
        DoraState.packedTile(copy.doraState().uraDoraTileTypesPacked(), 0),
        Tile.doraFrom(Tile.typeOf(copy.uraDoraIndicatorPhysicalTileId(0))));
    int uraChanged = 0;
    for (long seed = 0; seed < 50; seed++) {
      GameState other = state.snapshotCopy();
      other.reshuffleUnseen(seed);
      Assert.assertEquals(
          DoraState.packedTile(other.doraState().uraDoraTileTypesPacked(), 0),
          Tile.doraFrom(Tile.typeOf(other.uraDoraIndicatorPhysicalTileId(0))));
      if (other.uraDoraIndicatorPhysicalTileId(0) != state.uraDoraIndicatorPhysicalTileId(0))
        uraChanged++;
    }
    Assert.assertTrue(uraChanged > 40, "uraChanged=" + uraChanged);
    int differences = 0;
    for (int i = 0; i < 20; i++)
      if (copy.drawLiveWallTile() != state.drawLiveWallTile()) differences++;
    Assert.assertTrue(differences > 10, "differences=" + differences);
  }

  private interface Draw {
    int apply(long seed);
  }

  private static void assertUniform(HashSet<Integer> support, Draw draw) {
    int trials = support.size() * 100;
    var counts = new java.util.HashMap<Integer, Integer>();
    for (int seed = 0; seed < trials; seed++) {
      int tile = draw.apply(seed * 0x9E3779B97F4A7C15L);
      Assert.assertTrue(support.contains(tile), "tile " + tile + " is not unseen");
      counts.merge(tile, 1, Integer::sum);
    }
    double expected = 100, chi = 0;
    for (int tile : support) {
      double observed = counts.getOrDefault(tile, 0);
      chi += (observed - expected) * (observed - expected) / expected;
    }
    int df = support.size() - 1;
    Assert.assertTrue(chi < df + 5 * Math.sqrt(2.0 * df), "chi2=" + chi + " df=" + df);
  }
}
