package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 表を引く解析器を総当たり（{@code BruteShape}）と照合し、よく知られた手の待ち、三人打ちの萬子、受け入れも確かめる。 */
public class ShapeAnalyzerTest {

  private static final int SHANTEN_SAMPLES = 2_000;
  private static final int IMPROVING_SAMPLES = 400;

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants")
  public void shantenMatchesBruteForceOnRandomHands(Variant variant) {
    ShapeAnalyzer analyzer = new ShapeAnalyzer(variant);
    BruteShape brute = new BruteShape(variant.tiles());
    Random random = new Random(31 + variant.ordinal());
    for (int sample = 0; sample < SHANTEN_SAMPLES; sample++) {
      int melds = random.nextInt(Variant.MAX_MELDS + 1);
      int[] counts = randomHand(random, variant.tiles(), melds, sample, HandShape.MAX_SIZE);
      HandShape hand = ShapeFixtures.shape(counts, melds);
      Assert.assertEquals(analyzer.shanten(hand), brute.distance(counts, melds) - 1, hand.toString());
      Assert.assertEquals(analyzer.standardShanten(hand), brute.standard(counts, melds) - 1, hand.toString());
      if (melds == 0) {
        Assert.assertEquals(ShapeAnalyzer.chiitoitsuShanten(hand), brute.chiitoitsu(counts, melds) - 1, hand.toString());
        Assert.assertEquals(ShapeAnalyzer.kokushiShanten(hand), brute.kokushi(counts, melds) - 1, hand.toString());
      } else {
        Assert.assertThrows(IllegalArgumentException.class, () -> ShapeAnalyzer.chiitoitsuShanten(hand));
        Assert.assertThrows(IllegalArgumentException.class, () -> ShapeAnalyzer.kokushiShanten(hand));
      }
    }
  }

  @Test(dataProvider = "variants")
  public void improvingAndWaitsMatchBruteForce(Variant variant) {
    ShapeAnalyzer analyzer = new ShapeAnalyzer(variant);
    ShapeAnalyzer.Analysis analysis = new ShapeAnalyzer.Analysis();
    BruteShape brute = new BruteShape(variant.tiles());
    Random random = new Random(47 + variant.ordinal());
    for (int sample = 0; sample < IMPROVING_SAMPLES; sample++) {
      int melds = random.nextInt(Variant.MAX_MELDS + 1);
      int[] counts = randomHand(random, variant.tiles(), melds, sample, HandShape.MAX_SIZE - 1);
      HandShape hand = ShapeFixtures.shape(counts, melds);
      long improving = analyzer.improving(hand);
      Assert.assertEquals(improving, brute.improving(counts, melds), hand.toString());
      Assert.assertEquals(improving & ~variant.tiles().kinds(), 0L, hand.toString());
      analyzer.analyze(hand, analysis);
      Assert.assertEquals(analysis.improving(), improving, hand.toString());
      Assert.assertEquals(analysis.shanten(), analyzer.shanten(hand), hand.toString());
      Assert.assertEquals(analysis.standardShanten(), analyzer.standardShanten(hand), hand.toString());
      if (melds == 0) {
        Assert.assertEquals(analysis.chiitoitsuShanten(), ShapeAnalyzer.chiitoitsuShanten(hand), hand.toString());
        Assert.assertEquals(analysis.kokushiShanten(), ShapeAnalyzer.kokushiShanten(hand), hand.toString());
      } else {
        Assert.assertThrows(IllegalStateException.class, analysis::chiitoitsuShanten);
        Assert.assertThrows(IllegalStateException.class, analysis::kokushiShanten);
      }
      if (hand.size() == HandShape.MAX_SIZE - 1) {
        Assert.assertEquals(analyzer.waits(hand), brute.waits(counts, melds), hand.toString());
      }
    }
  }

  @Test
  public void pureNineGatesWaitsOnEveryMan() {
    ShapeAnalyzer analyzer = new ShapeAnalyzer(Variant.TENHOU_4P);
    HandShape hand = ShapeFixtures.shape("1112345678999m", 0);
    Assert.assertEquals(analyzer.shanten(hand), 0);
    Assert.assertEquals(analyzer.waits(hand), Tiles.range(Tiles.M1, Tiles.M9));
    Assert.assertEquals(analyzer.acceptance(hand, new byte[Tiles.KINDS]), 23);
  }

  @Test(dataProvider = "variants")
  public void chiitoitsuWaitsOnItsSingle(Variant variant) {
    ShapeAnalyzer analyzer = new ShapeAnalyzer(variant);
    HandShape hand = ShapeFixtures.shape("1199m1133p1155s5z", 0);
    Assert.assertEquals(analyzer.shanten(hand), 0);
    Assert.assertEquals(analyzer.standardShanten(hand), 3);
    Assert.assertEquals(analyzer.waits(hand), Tiles.bit(Tiles.HAKU));
  }

  @Test(dataProvider = "variants")
  public void kokushiWaitsOnTheMissingOrphanOrAllThirteen(Variant variant) {
    ShapeAnalyzer analyzer = new ShapeAnalyzer(variant);
    Assert.assertEquals(analyzer.waits(ShapeFixtures.shape("19m19p19s1234567z", 0)), Tiles.TERMINALS_AND_HONORS);
    Assert.assertEquals(analyzer.waits(ShapeFixtures.shape("119m19p19s123456z", 0)), Tiles.bit(Tiles.CHUN));
  }

  @Test(dataProvider = "variants")
  public void normalHandsWaitOnTheirShape(Variant variant) {
    ShapeAnalyzer analyzer = new ShapeAnalyzer(variant);
    Assert.assertEquals(
        analyzer.waits(ShapeFixtures.shape("111m456p789s11z23s", 0)), ShapeFixtures.kinds("14s"));
    Assert.assertEquals(
        analyzer.waits(ShapeFixtures.shape("11m99m123p456p789s", 0)), ShapeFixtures.kinds("19m"));
    Assert.assertEquals(analyzer.waits(ShapeFixtures.shape("11p23s", 3)), ShapeFixtures.kinds("14s"));
    Assert.assertEquals(analyzer.waits(ShapeFixtures.shape("5z", 4)), Tiles.bit(Tiles.HAKU));
    Assert.assertEquals(analyzer.waits(ShapeFixtures.shape("111m456p789s1z357s", 0)), 0L);
  }

  /** 1m 4 枚・9m 4 枚・5p 2 枚と面子 1 つ。四人打ちは 2m・3m・7m・8m で面子を作れるので 1 向聴、三人打ちは 1m・9m が孤立して 2 向聴。 */
  @Test
  public void sanmaManIsIsolatedTerminals() {
    HandShape hand = ShapeFixtures.shape("1111m9999m55p", 1);
    ShapeAnalyzer four = new ShapeAnalyzer(Variant.TENHOU_4P);
    ShapeAnalyzer three = new ShapeAnalyzer(Variant.TENHOU_3P);
    Assert.assertEquals(four.shanten(hand), 1);
    Assert.assertEquals(three.shanten(hand), 2);
    long middleMan = Tiles.range(Tiles.M2, Tiles.M8);
    Assert.assertEquals(four.improving(hand) & middleMan, ShapeFixtures.kinds("2378m"));
    Assert.assertEquals(three.improving(hand) & middleMan, 0L);
    Assert.assertTrue(Tiles.contains(three.improving(hand), Tiles.P5));
    Assert.assertFalse(Tiles.contains(four.improving(hand), Tiles.P5));
  }

  @Test
  public void remainingCountsCopiesLessVisibleLessOwnHand() {
    ShapeAnalyzer analyzer = new ShapeAnalyzer(Variant.TENHOU_4P);
    HandShape hand = ShapeFixtures.shape("123m456p789s11z23s", 0);
    byte[] visible = new byte[Tiles.KINDS];
    visible[Tiles.S1] = 2;
    visible[Tiles.S4] = 1;
    visible[Tiles.S2] = 3;
    Assert.assertEquals(analyzer.improving(hand), ShapeFixtures.kinds("14s"));
    Assert.assertEquals(analyzer.acceptance(hand, visible), (4 - 2) + (4 - 1));
    Assert.assertEquals(analyzer.remaining(hand, visible, Tiles.bit(Tiles.S2)), 4 - 3 - 1);
    Assert.assertEquals(analyzer.remaining(hand, visible, Tiles.bit(Tiles.TON)), 4 - 2);
  }

  @Test
  public void sanmaCountsNothingOfTwoToEightMan() {
    ShapeAnalyzer analyzer = new ShapeAnalyzer(Variant.TENHOU_3P);
    HandShape hand = ShapeFixtures.shape("19m19p19s1234567z", 0);
    long all = Tiles.ALL_KINDS;
    Assert.assertEquals(analyzer.remaining(hand, new byte[Tiles.KINDS], Tiles.range(Tiles.M2, Tiles.M8)), 0);
    Assert.assertEquals(
        analyzer.remaining(hand, new byte[Tiles.KINDS], all), TileSet.SANMA.size() - hand.tiles());
    Assert.assertEquals(analyzer.acceptance(hand, new byte[Tiles.KINDS]), 13 * 4 - hand.tiles());
  }

  @Test
  public void drawsAndWaitsNeedRoomForOneMoreTile() {
    ShapeAnalyzer analyzer = new ShapeAnalyzer(Variant.TENHOU_4P);
    HandShape complete = ShapeFixtures.shape("123m456p789s11z234s", 0);
    Assert.assertEquals(analyzer.shanten(complete), -1);
    Assert.assertThrows(IllegalArgumentException.class, () -> analyzer.improving(complete));
    Assert.assertThrows(IllegalArgumentException.class, () -> analyzer.waits(complete));
    HandShape short12 = ShapeFixtures.shape("123m456p789s11z2s", 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> analyzer.waits(short12));
  }

  /**
   * テストの手（種類ごとの枚数）を返す。サンプルの番号を 4 で割った余りで、和了の形（14 枚相当でなければ和了の形に近い手）、
   * 和了の形に近い手、ランダムな牌、么九牌だけの手を作り分ける。
   *
   * @param melds 副露の数
   * @param size 副露を 3 枚ずつと数えた手の枚数（ランダムな牌と么九牌だけの手は、ときどきこれより少ない）
   */
  private static int[] randomHand(Random random, TileSet set, int melds, int sample, int size) {
    int tiles = size - HandShape.MELD_SIZE * melds;
    return switch (sample % 4) {
      case 0 -> size == HandShape.MAX_SIZE
          ? ShapeFixtures.randomComplete(random, set, Variant.MAX_MELDS - melds)
          : ShapeFixtures.randomNearComplete(random, set, Variant.MAX_MELDS - melds);
      case 1 -> ShapeFixtures.randomNearComplete(random, set, Variant.MAX_MELDS - melds);
      case 2 -> ShapeFixtures.randomTiles(random, set, random.nextInt(4) == 0 ? random.nextInt(tiles + 1) : tiles);
      default -> orphans(random, melds == 0 ? tiles : random.nextInt(tiles + 1));
    };
  }

  private static int[] orphans(Random random, int tiles) {
    int[] counts = new int[Tiles.KINDS];
    for (int placed = 0; placed < tiles; ) {
      int kind = random.nextInt(Tiles.KINDS);
      if (Tiles.contains(Tiles.TERMINALS_AND_HONORS, kind) && counts[kind] < Tiles.COPIES) {
        counts[kind]++;
        placed++;
      }
    }
    return counts;
  }
}
