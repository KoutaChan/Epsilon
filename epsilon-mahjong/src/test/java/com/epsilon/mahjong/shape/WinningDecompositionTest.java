package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.WinningDecomposition.Wait;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 和了形の分解を、答えが分かっている手と、総当たり（手の牌から作れる面子を全て組み合わせる）で確かめる。 */
public class WinningDecompositionTest {

  private static final int SAMPLES = 400;

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test
  public void tripletsAndRunsOfTheSameTilesAreTwoDecompositions() {
    Assert.assertEquals(
        decompositions(ShapeFixtures.shape("111222333m456p77s", 0)),
        List.of("7s7s 1m1m1m 2m2m2m 3m3m3m 4p5p6p", "7s7s 1m2m3m 1m2m3m 1m2m3m 4p5p6p"));
  }

  @Test
  public void decompositionExposesPairMeldsAndMasks() {
    WinningDecomposition decomposition = new WinningDecomposition();
    decomposition.bind(ShapeFixtures.shape("111m999m123p456s77s", 0));
    Assert.assertTrue(decomposition.next());
    Assert.assertEquals(decomposition.pair(), Tiles.S7);
    Assert.assertEquals(decomposition.meldCount(), 4);
    Assert.assertEquals(decomposition.triplets(), ShapeFixtures.kinds("19m"));
    Assert.assertEquals(decomposition.runs(), Tiles.bit(Tiles.P1) | Tiles.bit(Tiles.S4));
    Assert.assertEquals(decomposition.meld(0), Tiles.M1);
    Assert.assertFalse(decomposition.run(0));
    Assert.assertEquals(decomposition.meld(2), Tiles.P1);
    Assert.assertTrue(decomposition.run(2));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> decomposition.meld(4));
    Assert.assertFalse(decomposition.next());
  }

  @Test
  public void identicalRunsStayApartInTheMeldsButShareTheRunMask() {
    WinningDecomposition decomposition = new WinningDecomposition();
    decomposition.bind(ShapeFixtures.shape("112233m445566p77s", 0));
    Assert.assertTrue(decomposition.next());
    Assert.assertEquals(decomposition.meldCount(), 4);
    Assert.assertEquals(decomposition.runs(), Tiles.bit(Tiles.M1) | Tiles.bit(Tiles.P4));
    Assert.assertEquals(decomposition.triplets(), 0L);
    Assert.assertEquals(decomposition.meld(1), Tiles.M1);
    Assert.assertEquals(decomposition.meld(3), Tiles.P4);
    Assert.assertFalse(decomposition.next());
  }

  @Test
  public void sevenPairsAndThirteenOrphansHaveNoNormalDecomposition() {
    Assert.assertEquals(decompositions(ShapeFixtures.shape("1199m1133p1155s55z", 0)), List.of());
    Assert.assertEquals(decompositions(ShapeFixtures.shape("119m19p19s1234567z", 0)), List.of());
  }

  @Test
  public void calledMeldsLeaveFewerConcealedMelds() {
    Assert.assertEquals(decompositions(ShapeFixtures.shape("123p456s11z", 2)), List.of("EE 1p2p3p 4s5s6s"));
    Assert.assertEquals(decompositions(ShapeFixtures.shape("99m", 4)), List.of("9m9m"));
  }

  @Test
  public void waitShapesFollowTheWinningTilePosition() {
    assertWaits("123m456m789m11p23s", "1s", Wait.RYANMEN);
    assertWaits("123m456m789m11p23s", "4s", Wait.RYANMEN);
    assertWaits("123m456m789m11p12s", "3s", Wait.PENCHAN);
    assertWaits("123m456m789m11p89s", "7s", Wait.PENCHAN);
    assertWaits("123m456m789m11p13s", "2s", Wait.KANCHAN);
    assertWaits("123m456m789m11p99s", "9s", Wait.SHANPON);
    assertWaits("123m456m789m111p5z", "5z", Wait.TANKI);
    assertWaits("123p456p789p2344m", "4m", Wait.TANKI, Wait.RYANMEN);
    assertWaits("123p456p789p2344m", "1m", Wait.RYANMEN);
  }

  @Test
  public void bindChecksTheHandSize() {
    WinningDecomposition decomposition = new WinningDecomposition();
    HandShape thirteen = ShapeFixtures.shape("123m456m789m11p23s", 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> decomposition.bind(thirteen));
    HandShape four = ShapeFixtures.shape("1111m456m789m11p2s", 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> decomposition.bind(four, Tiles.M1));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> decomposition.bind(ShapeFixtures.shape("123m456m789m11p234s", 0), Tiles.S1));
  }

  /** ランダムな和了形の分解が総当たりと一致する。1 枚を抜いて和了牌として渡しても同じ分解が出る。どの分解にも、和了牌の待ちの形が 1 つ以上ある。 */
  @Test(dataProvider = "variants")
  public void decompositionsMatchEnumerationOnRandomCompleteHands(Variant variant) {
    TileSet set = variant.tiles();
    Random random = new Random(59 + variant.ordinal());
    WinningDecomposition decomposition = new WinningDecomposition();
    for (int sample = 0; sample < SAMPLES; sample++) {
      int handMelds = random.nextInt(Variant.MAX_MELDS + 1);
      int[] counts = ShapeFixtures.randomComplete(random, set, Variant.MAX_MELDS - handMelds);
      HandShape hand = ShapeFixtures.shape(counts, handMelds);
      List<String> expected = enumerate(counts, Variant.MAX_MELDS - handMelds);
      Assert.assertEquals(decompositions(hand), expected, hand.toString());
      int winning = ShapeFixtures.randomHeld(random, counts);
      hand.remove(winning);
      decomposition.bind(hand, winning);
      List<String> found = new ArrayList<>();
      while (decomposition.next()) {
        found.add(describe(decomposition));
        Assert.assertNotEquals(decomposition.waits(winning), 0, hand + " + " + winning);
      }
      Collections.sort(found);
      Assert.assertEquals(found, expected, hand + " + " + winning);
    }
  }

  private static void assertWaits(String concealed, String winning, Wait... expected) {
    HandShape hand = ShapeFixtures.shape(concealed, 0);
    int kind = Long.numberOfTrailingZeros(ShapeFixtures.kinds(winning));
    WinningDecomposition decomposition = new WinningDecomposition();
    decomposition.bind(hand, kind);
    Assert.assertTrue(decomposition.next(), concealed + " + " + winning);
    int mask = 0;
    for (Wait wait : expected) {
      mask |= wait.bit();
    }
    Assert.assertEquals(decomposition.waits(kind), mask, concealed + " + " + winning);
    Assert.assertEquals(decomposition.waits(kind == Tiles.CHUN ? Tiles.HAKU : Tiles.CHUN), 0);
    Assert.assertFalse(decomposition.next(), concealed + " + " + winning);
  }

  private static List<String> decompositions(HandShape hand) {
    WinningDecomposition decomposition = new WinningDecomposition();
    decomposition.bind(hand);
    List<String> found = new ArrayList<>();
    while (decomposition.next()) {
      found.add(describe(decomposition));
    }
    Collections.sort(found);
    return found;
  }

  private static String describe(WinningDecomposition decomposition) {
    StringBuilder text = new StringBuilder(face(decomposition.pair())).append(face(decomposition.pair()));
    for (int i = 0; i < decomposition.meldCount(); i++) {
      text.append(' ').append(meld(decomposition.meld(i), decomposition.run(i)));
    }
    return text.toString();
  }

  /** 雀頭の種類ごとに、残りの牌をちょうど使い切る面子の組を全て求め、昇順に並べて返す。面子の候補は手の牌から作れるもの（種類の昇順、同じ種類では刻子が先）で、同じ面子を何度選んでもよい。 */
  private static List<String> enumerate(int[] counts, int melds) {
    List<Integer> kinds = new ArrayList<>();
    List<Boolean> runs = new ArrayList<>();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      if (counts[kind] >= HandShape.MELD_SIZE) {
        kinds.add(kind);
        runs.add(false);
      }
      if (Tiles.Group.of(kind).suited()
          && Tiles.number(kind) + 2 < Tiles.SUIT_SIZE
          && counts[kind] > 0
          && counts[kind + 1] > 0
          && counts[kind + 2] > 0) {
        kinds.add(kind);
        runs.add(true);
      }
    }
    List<String> found = new ArrayList<>();
    for (int pair = 0; pair < Tiles.KINDS; pair++) {
      if (counts[pair] < 2) {
        continue;
      }
      int[] rest = counts.clone();
      rest[pair] -= 2;
      cover(rest, kinds, runs, 0, melds, new ArrayList<>(), face(pair) + face(pair), found);
    }
    Collections.sort(found);
    return found;
  }

  private static void cover(
      int[] rest, List<Integer> kinds, List<Boolean> runs, int from, int left, List<String> chosen, String pair,
      List<String> found) {
    if (left == 0) {
      for (int count : rest) {
        if (count != 0) {
          return;
        }
      }
      found.add(pair + (chosen.isEmpty() ? "" : " " + String.join(" ", chosen)));
      return;
    }
    for (int i = from; i < kinds.size(); i++) {
      int kind = kinds.get(i);
      boolean run = runs.get(i);
      if (!take(rest, kind, run, -1)) {
        continue;
      }
      chosen.add(meld(kind, run));
      cover(rest, kinds, runs, i, left - 1, chosen, pair, found);
      chosen.remove(chosen.size() - 1);
      take(rest, kind, run, 1);
    }
  }

  /**
   * 面子の 3 枚を、残りの枚数に足すか、残りの枚数から引く。どれかの枚数が負になるときは何も変えない。
   *
   * @param rest 種類ごとの残りの枚数
   * @param kind 刻子の種類、または順子の最初の種類
   * @param run 順子なら true
   * @param sign 1 なら足し、−1 なら引く
   * @return 足し引きしたか（負になって何も変えなかったときは false）
   */
  private static boolean take(int[] rest, int kind, boolean run, int sign) {
    for (int i = 0; i < HandShape.MELD_SIZE; i++) {
      int tile = run ? kind + i : kind;
      rest[tile] += sign;
      if (rest[tile] < 0) {
        for (int j = 0; j <= i; j++) {
          rest[run ? kind + j : kind] -= sign;
        }
        return false;
      }
    }
    return true;
  }

  private static String meld(int kind, boolean run) {
    StringBuilder text = new StringBuilder();
    for (int i = 0; i < HandShape.MELD_SIZE; i++) {
      text.append(face(run ? kind + i : kind));
    }
    return text.toString();
  }

  private static String face(int kind) {
    return Face.of(kind, false).mjai();
  }
}
