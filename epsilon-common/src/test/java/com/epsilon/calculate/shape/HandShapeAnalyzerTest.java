package com.epsilon.calculate.shape;

import com.epsilon.core.Tile;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 牌を1枚ずつ追加する全列挙と照合し、受け入れ牌と待ち牌の判定が数牌・字牌・特殊形で一致することを検証する。 */
public class HandShapeAnalyzerTest {
  @DataProvider
  public Object[][] meldCounts() {
    return new Object[][] {{0}, {1}, {2}, {3}, {4}};
  }

  @Test(dataProvider = "meldCounts")
  public void randomShapesMatchCompleteDrawEnumeration(int melds) {
    SplittableRandom random = new SplittableRandom(0x41a9cL + melds);
    HandShapeAnalyzer actual = new HandShapeAnalyzer();
    HandShapeAnalyzer reference = new HandShapeAnalyzer();
    HandShapeAnalysisBuffer analysis = new HandShapeAnalysisBuffer();
    HandShapeState hand = new HandShapeState();
    for (int sample = 0; sample < 1000; sample++) {
      hand.clear();
      hand.setMeldCount(melds);
      int count = random.nextInt(15 - 3 * melds);
      while (hand.concealedTileCount() < count) {
        int tile = random.nextInt(Tile.NUM_TILE_TYPES);
        if (hand.count(tile) < Tile.TILES_PER_TYPE) hand.add(tile);
      }
      assertMasks(hand, actual, reference, analysis);
    }
  }

  @Test(dataProvider = "meldCounts")
  public void everyDiscardFromGeneratedCompleteHandsMatchesCompleteDrawEnumeration(int melds) {
    SplittableRandom random = new SplittableRandom(0x51b9eL + melds);
    HandShapeAnalyzer actual = new HandShapeAnalyzer();
    HandShapeAnalyzer reference = new HandShapeAnalyzer();
    HandShapeAnalysisBuffer analysis = new HandShapeAnalysisBuffer();
    HandShapeState hand = new HandShapeState();
    int[] counts = new int[Tile.NUM_TILE_TYPES];
    for (int sample = 0; sample < 500; sample++) {
      completeHand(random, melds, counts);
      hand.load(counts, melds);
      for (long tiles = hand.concealedTileTypeMask(); tiles != 0L; tiles &= tiles - 1) {
        int discard = Long.numberOfTrailingZeros(tiles);
        hand.remove(discard);
        assertMasks(hand, actual, reference, analysis);
        hand.add(discard);
      }
    }
  }

  @DataProvider
  public Object[][] boundaryHands() {
    return new Object[][] {
      {3, new int[] {Tile.M2, Tile.M3, Tile.TON, Tile.TON}, mask(Tile.M1, Tile.M4)},
      {3, new int[] {Tile.P7, Tile.P9, Tile.TON, Tile.TON}, mask(Tile.P8)},
      {3, new int[] {Tile.S8, Tile.S9, Tile.TON, Tile.TON}, mask(Tile.S7)},
      {3, new int[] {Tile.M9, Tile.P1, Tile.TON, Tile.TON}, 0L},
      {3, new int[] {Tile.P9, Tile.S1, Tile.TON, Tile.TON}, 0L},
      {3, new int[] {Tile.S9, Tile.TON, Tile.NAN, Tile.NAN}, 0L},
      {3, new int[] {Tile.TON, Tile.NAN, Tile.SHA, Tile.SHA}, 0L},
      {3, new int[] {Tile.TON, Tile.TON, Tile.NAN, Tile.NAN}, mask(Tile.TON, Tile.NAN)},
      {3, new int[] {Tile.TON, Tile.TON, Tile.TON, Tile.TON}, 0L},
      {4, new int[] {Tile.CHUN}, mask(Tile.CHUN)},
      {
        0,
        new int[] {
          Tile.M1, Tile.M1, Tile.M3, Tile.M3, Tile.M5, Tile.M5, Tile.M7, Tile.M7, Tile.P1, Tile.P1,
          Tile.P3, Tile.P3, Tile.S9
        },
        mask(Tile.S9)
      },
      {
        0,
        new int[] {
          Tile.M1,
          Tile.M9,
          Tile.P1,
          Tile.P9,
          Tile.S1,
          Tile.S9,
          Tile.TON,
          Tile.NAN,
          Tile.SHA,
          Tile.PEI,
          Tile.HAKU,
          Tile.HATSU,
          Tile.CHUN
        },
        Tile.TERMINAL_OR_HONOR_TYPE_MASK
      },
      {
        0,
        new int[] {
          Tile.M1,
          Tile.M1,
          Tile.M9,
          Tile.P1,
          Tile.P9,
          Tile.S1,
          Tile.S9,
          Tile.TON,
          Tile.NAN,
          Tile.SHA,
          Tile.PEI,
          Tile.HAKU,
          Tile.HATSU
        },
        mask(Tile.CHUN)
      }
    };
  }

  @Test(dataProvider = "boundaryHands")
  public void waitsRespectSuitHonorAndSpecialHandBoundaries(int melds, int[] tiles, long expected) {
    HandShapeState hand = new HandShapeState();
    hand.setMeldCount(melds);
    for (int tile : tiles) hand.add(tile);
    HandShapeAnalyzer actual = new HandShapeAnalyzer();
    Assert.assertEquals(actual.agariTileTypeMask(hand), expected);
    assertMasks(hand, actual, new HandShapeAnalyzer(), new HandShapeAnalysisBuffer());
  }

  private static void assertMasks(
      HandShapeState hand,
      HandShapeAnalyzer actual,
      HandShapeAnalyzer reference,
      HandShapeAnalysisBuffer analysis) {
    long low = hand.packedConcealedTileCountsLow();
    long high = hand.packedConcealedTileCountsHigh();
    int minimum = reference.calculateMinimum(hand);
    long expected = enumerateDraws(hand, minimum, reference);
    actual.analyzeInto(hand, analysis);
    Assert.assertEquals(analysis.minimumShanten(), minimum);
    Assert.assertEquals(analysis.ukeireTileTypeMask(), expected);
    Assert.assertEquals(actual.improvingTileTypeMask(hand, minimum), expected);
    Assert.assertEquals(actual.improvingTileTypeMask(hand, 0), enumerateDraws(hand, 0, reference));
    Assert.assertEquals(
        actual.improvingTileTypeMask(hand, minimum + 1),
        enumerateDraws(hand, minimum + 1, reference));
    long waits = hand.effectiveTileCount() == 13 ? enumerateDraws(hand, 0, reference) : 0L;
    Assert.assertEquals(analysis.agariTileTypeMask(), waits);
    if (hand.effectiveTileCount() == 13) {
      Assert.assertEquals(actual.agariTileTypeMask(hand), waits);
    }
    Assert.assertEquals(hand.packedConcealedTileCountsLow(), low);
    Assert.assertEquals(hand.packedConcealedTileCountsHigh(), high);
  }

  private static long enumerateDraws(
      HandShapeState hand, int threshold, HandShapeAnalyzer reference) {
    if (hand.effectiveTileCount() == 14) return 0L;
    long mask = 0L;
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      if (hand.count(tile) < Tile.TILES_PER_TYPE
          && reference.calculateAfterAdding(hand, tile) < threshold) {
        mask |= 1L << tile;
      }
    }
    return mask;
  }

  private static void completeHand(SplittableRandom random, int melds, int[] counts) {
    boolean valid;
    do {
      Arrays.fill(counts, 0);
      counts[random.nextInt(Tile.NUM_TILE_TYPES)] = 2;
      for (int group = melds; group < 4; group++) {
        if (random.nextBoolean()) {
          counts[random.nextInt(Tile.NUM_TILE_TYPES)] += 3;
        } else {
          int start = random.nextInt(3) * 9 + random.nextInt(7);
          counts[start]++;
          counts[start + 1]++;
          counts[start + 2]++;
        }
      }
      valid = true;
      for (int count : counts) valid &= count <= Tile.TILES_PER_TYPE;
    } while (!valid);
  }

  private static long mask(int... tiles) {
    long mask = 0L;
    for (int tile : tiles) mask |= 1L << tile;
    return mask;
  }
}
