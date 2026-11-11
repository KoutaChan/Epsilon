package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌山の作り方、位置の式、席別ツモ列、牌譜の牌山のテスト。 */
public class WallTest {

  @Test
  public void everyPhysicalTileIsInTheWallExactlyOnce() {
    for (Variant variant : Variant.values()) {
      TileSet set = variant.tiles();
      Wall wall = new Wall(variant);
      wall.start(0x5EEDL, 1, DrawOrder.WALL_ORDER);
      for (int round = 0; round < 20; round++) {
        wall.shuffle(round % variant.seats());
        Set<Integer> seen = new HashSet<>();
        for (int position = 0; position < set.size(); position++) {
          int tile = wall.tile(position);
          Assert.assertTrue(set.contains(tile), variant + " position " + position + " holds " + tile);
          Assert.assertTrue(seen.add(tile), variant + " repeats " + tile);
        }
        Assert.assertEquals(seen.size(), set.size());
      }
    }
  }

  @Test
  public void theWallIsFisherYatesFromTheBackOverTheTileSetOrder() {
    for (Variant variant : Variant.values()) {
      TileSet set = variant.tiles();
      for (long seed : new long[] {0, 42, -7, 0x0123_4567_89AB_CDEFL}) {
        for (int epoch = 1; epoch <= 3; epoch++) {
          Wall wall = new Wall(variant);
          wall.start(seed, epoch, DrawOrder.WALL_ORDER);
          wall.shuffle(0);
          int[] expected = new int[set.size()];
          for (int index = 0; index < expected.length; index++) {
            expected[index] = set.physicalId(index);
          }
          Random random = new Random(SeedMixer.indexed(seed, Wall.SALT, epoch));
          for (int index = expected.length - 1; index > 0; index--) {
            int other = random.nextInt(index + 1);
            int tile = expected[index];
            expected[index] = expected[other];
            expected[other] = tile;
          }
          Assert.assertEquals(positions(wall, set.size()), expected, variant + " seed " + seed + " epoch " + epoch);
          Assert.assertEquals(wall.epoch(), epoch + 1);
        }
      }
    }
  }

  @Test
  public void sameSeedAndEpochGiveTheSameWallAndOthersDiffer() {
    for (Variant variant : Variant.values()) {
      int size = variant.tiles().size();
      for (long seed = 1; seed <= 20; seed++) {
        int[] first = shuffled(variant, seed, 5);
        Assert.assertEquals(shuffled(variant, seed, 5), first);
        Assert.assertFalse(Arrays.equals(shuffled(variant, seed, 6), first), variant + " seed " + seed);
        Assert.assertFalse(Arrays.equals(shuffled(variant, seed + 1000, 5), first), variant + " seed " + seed);
        Assert.assertEquals(first.length, size);
      }
    }
  }

  @Test
  public void eachShuffleUsesTheNextEpoch() {
    Wall wall = new Wall(Variant.TENHOU_4P);
    wall.start(77, 1, DrawOrder.WALL_ORDER);
    for (int epoch = 1; epoch <= 4; epoch++) {
      wall.shuffle(epoch % 4);
      Assert.assertEquals(positions(wall, 136), shuffled(Variant.TENHOU_4P, 77, epoch));
    }
  }

  @Test
  public void theDealTakesPacketsOfFourFromTheDealerThenOneEach() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      Wall wall = new Wall(variant);
      wall.start(3, 1, DrawOrder.WALL_ORDER);
      wall.shuffle(0);
      boolean[] dealt = new boolean[Variant.HAND * seats];
      for (int offset = 0; offset < seats; offset++) {
        for (int index = 0; index < Variant.HAND; index++) {
          int position = index < 12 ? index / 4 * 4 * seats + offset * 4 + index % 4 : 12 * seats + offset;
          Assert.assertEquals(wall.dealt(offset, index), wall.tile(position), variant + " " + offset + "/" + index);
          Assert.assertFalse(dealt[position]);
          dealt[position] = true;
        }
      }
    }
  }

  @Test
  public void indicatorsUraAndReplacementsFollowTheirPositions() {
    for (Variant variant : Variant.values()) {
      int size = variant.tiles().size();
      Wall wall = new Wall(variant);
      wall.start(11, 1, DrawOrder.WALL_ORDER);
      wall.shuffle(0);
      for (int index = 0; index < WinContext.MAX_INDICATORS; index++) {
        Assert.assertEquals(wall.indicator(index), wall.tile(size - 6 - 2 * index));
        Assert.assertEquals(wall.ura(index), wall.tile(size - 5 - 2 * index));
      }
      Assert.assertThrows(IndexOutOfBoundsException.class, () -> wall.indicator(WinContext.MAX_INDICATORS));
      int live = variant.liveWall();
      for (int k = 1; k <= variant.rinshanTiles(); k++) {
        int position = k <= 4 ? size - k : size - 10 - k;
        Assert.assertEquals(wall.drawRinshan(k % variant.seats()), wall.tile(position), variant + " replacement " + k);
        Assert.assertEquals(wall.liveLeft(), live - k);
      }
      Assert.assertThrows(IllegalStateException.class, () -> wall.drawRinshan(0));
    }
  }

  /** 席別ツモ列を使わない局では、嶺上牌を引いた枚数だけ山のツモが早く尽き、嶺上牌・山のツモ・表示牌・裏ドラ表示牌に同じ牌は出ない。 */
  @Test
  public void liveDrawsStopBeforeThePositionsTheReplacementsTook() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      Wall wall = new Wall(variant);
      wall.start(19, 1, DrawOrder.WALL_ORDER);
      wall.shuffle(1);
      Set<Integer> drawn = new HashSet<>();
      for (int k = 0; k < variant.rinshanTiles(); k++) {
        Assert.assertTrue(drawn.add(wall.drawRinshan(k % seats)));
      }
      int draws = 0;
      while (wall.liveLeft() > 0) {
        int position = Variant.HAND * seats + draws;
        Assert.assertEquals(wall.draw(draws % seats), wall.tile(position));
        Assert.assertTrue(drawn.add(wall.tile(position)));
        draws++;
      }
      Assert.assertEquals(draws, variant.liveWall() - variant.rinshanTiles());
      Assert.assertThrows(IllegalStateException.class, () -> wall.draw(0));
      Assert.assertThrows(IllegalStateException.class, () -> wall.drawRinshan(0));
      for (int index = 0; index < WinContext.MAX_INDICATORS; index++) {
        Assert.assertFalse(drawn.contains(wall.indicator(index)));
        Assert.assertFalse(drawn.contains(wall.ura(index)));
      }
    }
  }

  @Test
  public void withoutSeatStreamsDrawsReadThePositionsInOrder() {
    Wall wall = new Wall(Variant.TENHOU_4P);
    wall.start(5, 1, DrawOrder.WALL_ORDER);
    wall.shuffle(2);
    int[] positions = positions(wall, 136);
    for (int draw = 0; draw < Variant.TENHOU_4P.liveWall(); draw++) {
      Assert.assertEquals(wall.draw((draw * 3) % 4), positions[52 + draw]);
    }
    Assert.assertEquals(wall.liveLeft(), 0);
  }

  /**
   * どの順に席が引いても、嶺上・補充が混じっても、列が残る間は各席が自分の列を順に引く。嶺上・補充も山を 1 枚減らし、王牌の牌は変わらない。
   */
  @Test
  public void seatStreamsGiveEachSeatItsOwnTilesInOrder() {
    Random random = new Random(8);
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      int start = Variant.HAND * seats;
      int size = variant.tiles().size();
      for (int trial = 0; trial < 50; trial++) {
        int dealer = random.nextInt(seats);
        Wall wall = new Wall(variant);
        wall.start(random.nextLong(), 1 + trial, DrawOrder.SEAT_STREAMS);
        wall.shuffle(dealer);
        int[] positions = positions(wall, size);
        int[][] streams = new int[seats][variant.liveWall()];
        int[] streamSizes = new int[seats];
        for (int k = 0; k < variant.liveWall(); k++) {
          int seat = (dealer + k) % seats;
          streams[seat][streamSizes[seat]++] = positions[start + k];
        }
        int[] taken = new int[seats];
        int shortest = Arrays.stream(streamSizes).min().orElseThrow();
        int replacements = 0;
        for (int draw = 0; draw < shortest * seats; draw++) {
          int seat = random.nextInt(seats);
          while (taken[seat] == shortest) {
            seat = (seat + 1) % seats;
          }
          boolean replace = replacements < variant.rinshanTiles() && random.nextInt(5) == 0;
          int left = wall.liveLeft();
          int tile = replace ? wall.drawRinshan(seat) : wall.draw(seat);
          Assert.assertEquals(tile, streams[seat][taken[seat]], variant + " seat " + seat);
          Assert.assertEquals(wall.liveLeft(), left - 1);
          taken[seat]++;
          replacements += replace ? 1 : 0;
        }
        for (int position = size - Variant.DEAD_WALL; position < size; position++) {
          Assert.assertEquals(wall.tile(position), positions[position], variant + " dead wall " + position);
        }
      }
    }
  }

  /** 席別ツモ列でも、嶺上が混じっても、山の牌は 1 回ずつしか出てこない。 */
  @Test
  public void seatStreamsWithReplacementsNeverRepeatATile() {
    Random random = new Random(21);
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      for (int trial = 0; trial < 200; trial++) {
        Wall wall = new Wall(variant);
        wall.start(random.nextLong(), 1, DrawOrder.SEAT_STREAMS);
        wall.shuffle(random.nextInt(seats));
        Set<Integer> seen = new HashSet<>();
        for (int offset = 0; offset < seats; offset++) {
          for (int index = 0; index < Variant.HAND; index++) {
            Assert.assertTrue(seen.add(wall.dealt(offset, index)));
          }
        }
        int replacements = 0;
        while (wall.liveLeft() > 0) {
          boolean replace = replacements < variant.rinshanTiles() && random.nextInt(6) == 0;
          int seat = random.nextInt(seats);
          int tile = replace ? wall.drawRinshan(seat) : wall.draw(seat);
          replacements += replace ? 1 : 0;
          Assert.assertTrue(seen.add(tile), variant + " trial " + trial + " repeats " + tile);
        }
        for (int index = 0; index < WinContext.MAX_INDICATORS; index++) {
          Assert.assertFalse(seen.contains(wall.indicator(index)));
          Assert.assertFalse(seen.contains(wall.ura(index)));
        }
        Assert.assertEquals(
            seen.size(), Variant.HAND * seats + variant.liveWall(), variant + " trial " + trial);
        Assert.assertEquals(sortedPositions(wall, variant), sortedSet(variant.tiles()));
      }
    }
  }

  @Test
  public void aCopyDrawsTheSameTilesAndStaysIndependent() {
    Random random = new Random(4);
    for (Variant variant : Variant.values()) {
      Wall wall = new Wall(variant);
      wall.start(123, 9, DrawOrder.SEAT_STREAMS);
      wall.shuffle(1);
      for (int draw = 0; draw < 10; draw++) {
        wall.draw(random.nextInt(variant.seats()));
      }
      wall.drawRinshan(random.nextInt(variant.seats()));
      Wall copy = new Wall(variant);
      copy.copyFrom(wall);
      Fields.assertSameFields(wall, copy);
      Wall snapshot = new Wall(variant);
      snapshot.copyFrom(wall);
      long order = random.nextLong();
      Random first = new Random(order);
      Random second = new Random(order);
      while (wall.liveLeft() > 0) {
        Assert.assertEquals(copy.draw(second.nextInt(variant.seats())), wall.draw(first.nextInt(variant.seats())));
      }
      Fields.assertSameFields(wall, copy);
      copy.shuffle(0);
      Assert.assertEquals(copy.epoch(), 11);
      Assert.assertEquals(wall.epoch(), 10);
      Assert.assertEquals(snapshot.liveLeft(), variant.liveWall() - 11);
      Assert.assertThrows(IllegalArgumentException.class, () -> copy.copyFrom(new Wall(other(variant))));
    }
  }

  @Test
  public void aRecordedWallReadsKnownPositionsAndRejectsUnknownOnes() {
    Variant variant = Variant.TENHOU_4P;
    int[] recorded = new int[136];
    Arrays.fill(recorded, Wall.UNKNOWN);
    for (int position = 0; position < 52; position++) {
      recorded[position] = position + 20;
    }
    recorded[52] = 3;
    recorded[136 - 6] = 0;
    Wall wall = new Wall(variant);
    wall.load(recorded);
    Assert.assertEquals(wall.dealt(1, 0), 24);
    Assert.assertEquals(wall.indicator(0), 0);
    Assert.assertEquals(wall.liveLeft(), 70);
    Assert.assertEquals(wall.draw(0), 3);
    Assert.assertThrows(IllegalStateException.class, () -> wall.draw(1));
    Assert.assertThrows(IllegalStateException.class, () -> wall.ura(0));
    Assert.assertThrows(IllegalStateException.class, () -> wall.drawRinshan(0));
    Assert.assertEquals(wall.liveLeft(), 69);

    Assert.assertThrows(IllegalArgumentException.class, () -> wall.load(new int[108]));
    int[] repeated = recorded.clone();
    repeated[60] = 3;
    Assert.assertThrows(IllegalArgumentException.class, () -> wall.load(repeated));
    Wall three = new Wall(Variant.TENHOU_3P);
    int[] outside = new int[108];
    Arrays.fill(outside, Wall.UNKNOWN);
    outside[0] = Tiles.physicalId(Tiles.M2, 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> three.load(outside));
  }

  private static int[] shuffled(Variant variant, long seed, int epoch) {
    Wall wall = new Wall(variant);
    wall.start(seed, epoch, DrawOrder.WALL_ORDER);
    wall.shuffle(0);
    return positions(wall, variant.tiles().size());
  }

  private static int[] positions(Wall wall, int size) {
    int[] tiles = new int[size];
    for (int position = 0; position < size; position++) {
      tiles[position] = wall.tile(position);
    }
    return tiles;
  }

  private static int[] sortedPositions(Wall wall, Variant variant) {
    int[] tiles = positions(wall, variant.tiles().size());
    Arrays.sort(tiles);
    return tiles;
  }

  private static int[] sortedSet(TileSet set) {
    int[] tiles = new int[set.size()];
    for (int index = 0; index < tiles.length; index++) {
      tiles[index] = set.physicalId(index);
    }
    return tiles;
  }

  private static Variant other(Variant variant) {
    return variant == Variant.TENHOU_4P ? Variant.TENHOU_3P : Variant.TENHOU_4P;
  }
}
