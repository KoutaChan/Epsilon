package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.shape.ShapeFixtures;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 手牌の状態を確かめる。見るのは、枚数、形のコード、物理牌の番号、面子、抜き、待ち、振聴。 */
public class HandTest {

  private static final int GAMES = 40;
  private static final int CYCLES = 75;

  /** 和了の形から始めて打牌とツモを繰り返す（途中で聴牌の 13 枚を何度も通る）。どの打牌・ツモの後も、枚数・形のコード・待ちが数え直した値と合う。 */
  @Test
  public void countsShapeCodesCopiesAndWaitsFollowDiscardsAndDraws() {
    Random random = new Random(12);
    for (Variant variant : Variant.values()) {
      TileSet set = variant.tiles();
      ShapeAnalyzer analyzer = new ShapeAnalyzer(variant);
      for (int game = 0; game < GAMES; game++) {
        Hand hand = new Hand(variant);
        boolean[] held = new boolean[Tiles.PHYSICAL_IDS];
        int[] counts = ShapeFixtures.randomComplete(random, set, Variant.MAX_MELDS);
        for (int kind = 0; kind < Tiles.KINDS; kind++) {
          for (int copy = 0; copy < counts[kind]; copy++) {
            hand.add(Tiles.physicalId(kind, copy));
            held[Tiles.physicalId(kind, copy)] = true;
          }
        }
        List<Integer> pool = new ArrayList<>();
        for (int index = 0; index < set.size(); index++) {
          if (!held[set.physicalId(index)]) {
            pool.add(set.physicalId(index));
          }
        }
        assertHand(hand, held, analyzer);
        for (int cycle = 0; cycle < CYCLES; cycle++) {
          int discard = randomHeld(random, held);
          hand.discard(discard);
          held[discard] = false;
          pool.add(discard);
          assertHand(hand, held, analyzer);
          int draw = pool.remove(random.nextInt(pool.size()));
          hand.add(draw);
          held[draw] = true;
          assertHand(hand, held, analyzer);
        }
      }
    }
  }

  @Test
  public void waitsAreComputedOnceTheHandHasThirteenTiles() {
    Hand hand = new Hand(Variant.TENHOU_4P);
    for (int kind : ShapeFixtures.tiles("123456789m123p")) {
      hand.add(Tiles.physicalId(kind, 1));
      Assert.assertEquals(hand.waits(), 0L);
    }
    long waits = Tiles.bit(Tiles.P1) | Tiles.bit(Tiles.P4);
    hand.add(Tiles.physicalId(Tiles.P4, 1));
    Assert.assertEquals(hand.waits(), waits);
    hand.add(Tiles.physicalId(Tiles.CHUN, 1));
    Assert.assertEquals(hand.waits(), 0L);
    hand.discard(Tiles.physicalId(Tiles.CHUN, 1));
    Assert.assertEquals(hand.waits(), waits);
  }

  @Test
  public void meldsUpgradesAndNukiKeepTheShapeAndTheWaits() {
    Hand hand = new Hand(Variant.TENHOU_4P);
    for (int kind : ShapeFixtures.tiles("123m456p789s1122z")) {
      hand.add(firstFree(hand, kind));
    }
    Assert.assertEquals(hand.waits(), Tiles.bit(Tiles.TON) | Tiles.bit(Tiles.NAN));
    hand.take(Tiles.physicalId(Tiles.TON, 3));
    hand.take(Tiles.physicalId(Tiles.TON, 2));
    int pon = Meld.pon(Tiles.TON, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE).id();
    hand.addMeld(pon);
    Assert.assertEquals(hand.meldCount(), 1);
    Assert.assertEquals(hand.meld(0), pon);
    Assert.assertEquals(hand.shape().melds(), 1);
    Assert.assertEquals(hand.shape().size(), HandShape.MAX_SIZE);
    Assert.assertEquals(hand.waits(), 0L);
    hand.discard(Tiles.physicalId(Tiles.NAN, 3));
    Assert.assertEquals(hand.waits(), Tiles.bit(Tiles.NAN));
    hand.add(Tiles.physicalId(Tiles.TON, 1));
    Assert.assertEquals(hand.waits(), 0L);
    hand.take(Tiles.physicalId(Tiles.TON, 1));
    int kakan = Meld.kakan(Meld.of(pon), false).id();
    Assert.assertEquals(hand.upgrade(Tiles.TON, false), kakan);
    Assert.assertEquals(hand.meld(0), kakan);
    Assert.assertEquals(hand.meldCount(), 1);
    Assert.assertEquals(hand.waits(), Tiles.bit(Tiles.NAN));
    Assert.assertThrows(IllegalStateException.class, () -> hand.upgrade(Tiles.TON, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> hand.addMeld(kakan));
    Assert.assertThrows(IllegalStateException.class, hand::addNuki);

    Hand three = new Hand(Variant.TENHOU_3P);
    for (int kind : ShapeFixtures.tiles("19m123p456p789s114z")) {
      three.add(firstFree(three, kind));
    }
    Assert.assertEquals(three.shape().size(), HandShape.MAX_SIZE);
    three.take(Tiles.physicalId(Tiles.PEI, 3));
    three.addNuki();
    Assert.assertEquals(three.nuki(), 1);
    Assert.assertEquals(three.count(Tiles.PEI), 0);
    Assert.assertEquals(three.shape().size(), HandShape.MAX_SIZE - 1);
    Assert.assertEquals(three.waits(), 0L);
    for (int more = 1; more < Tiles.COPIES; more++) {
      three.addNuki();
    }
    Assert.assertThrows(IllegalStateException.class, three::addNuki);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> three.addMeld(Meld.chi(Tiles.P1, Tiles.P1, Meld.AkaSource.NONE).id()));
  }

  @Test
  public void anAnkanTakesFourTilesAndCountsAsOneMeld() {
    Hand hand = new Hand(Variant.TENHOU_4P);
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      hand.add(Tiles.physicalId(Tiles.P5, copy));
    }
    for (int kind : ShapeFixtures.tiles("123m456s78s77z")) {
      hand.add(firstFree(hand, kind));
    }
    Assert.assertEquals(hand.shape().size(), HandShape.MAX_SIZE);
    Assert.assertEquals(hand.redFives(), Tiles.bit(Tiles.P5));
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      hand.take(Tiles.physicalId(Tiles.P5, copy));
    }
    hand.addMeld(Meld.ankan(Tiles.P5, true).id());
    Assert.assertEquals(hand.redFives(), 0L);
    Assert.assertEquals(hand.shape().size(), HandShape.MAX_SIZE - 1);
    Assert.assertEquals(hand.waits(), Tiles.bit(Tiles.S3) | Tiles.bit(Tiles.S6) | Tiles.bit(Tiles.S9));
  }

  @Test
  public void pickChoosesTheLowestCopyOfTheAskedRedness() {
    Hand hand = new Hand(Variant.TENHOU_4P);
    hand.add(Tiles.physicalId(Tiles.S5, 3));
    hand.add(Tiles.physicalId(Tiles.S5, 0));
    hand.add(Tiles.physicalId(Tiles.S5, 2));
    hand.add(Tiles.physicalId(Tiles.S1, 1));
    Assert.assertEquals(hand.pick(Tiles.S5, true), Tiles.physicalId(Tiles.S5, 0));
    Assert.assertEquals(hand.pick(Tiles.S5, false), Tiles.physicalId(Tiles.S5, 2));
    Assert.assertEquals(hand.pick(Tiles.S5, false, Tiles.physicalId(Tiles.S5, 2)), Tiles.physicalId(Tiles.S5, 3));
    Assert.assertEquals(hand.pick(Tiles.S1, false), Tiles.physicalId(Tiles.S1, 1));
    Assert.assertThrows(IllegalStateException.class, () -> hand.pick(Tiles.S1, true));
    Assert.assertThrows(IllegalStateException.class, () -> hand.pick(Tiles.S1, false, Tiles.physicalId(Tiles.S1, 1)));
    Assert.assertThrows(IllegalStateException.class, () -> hand.pick(Tiles.S2, false));
    Assert.assertEquals(hand.redFives(), Tiles.bit(Tiles.S5));
    Assert.assertTrue(hand.holds(Tiles.physicalId(Tiles.S5, 0)));
    Assert.assertFalse(hand.holds(Tiles.physicalId(Tiles.S5, 1)));
    Assert.assertFalse(hand.holds(Tiles.PHYSICAL_IDS));
  }

  @Test
  public void invalidChangesThrowAndKeepTheHand() {
    Hand hand = new Hand(Variant.TENHOU_3P);
    hand.add(Tiles.physicalId(Tiles.M1, 0));
    Assert.assertThrows(IllegalStateException.class, () -> hand.add(Tiles.physicalId(Tiles.M1, 0)));
    Assert.assertThrows(IllegalArgumentException.class, () -> hand.add(Tiles.physicalId(Tiles.M2, 0)));
    Assert.assertThrows(IllegalStateException.class, () -> hand.discard(Tiles.physicalId(Tiles.M1, 1)));
    Assert.assertThrows(IllegalStateException.class, () -> hand.take(Tiles.physicalId(Tiles.M9, 0)));
    Assert.assertEquals(hand.count(Tiles.M1), 1);
    Assert.assertTrue(hand.holds(Tiles.physicalId(Tiles.M1, 0)));
    Assert.assertEquals(hand.shape().tiles(), 1);
  }

  @Test
  public void theMissedRonMarkAndClearing() {
    Hand hand = new Hand(Variant.TENHOU_4P);
    Assert.assertFalse(hand.missedRon());
    hand.setMissedRon(true);
    Assert.assertTrue(hand.missedRon());
    hand.setMissedRon(false);
    Assert.assertFalse(hand.missedRon());
    hand.setMissedRon(true);
    hand.add(Tiles.physicalId(Tiles.M1, 0));
    hand.clear();
    Assert.assertFalse(hand.missedRon());
    Fields.assertSameFields(new Hand(Variant.TENHOU_4P), hand);
  }

  @Test
  public void aCopyEqualsTheOriginalAndStaysIndependent() {
    Hand hand = new Hand(Variant.TENHOU_4P);
    for (int kind : ShapeFixtures.tiles("123m456p789s1122z")) {
      hand.add(firstFree(hand, kind));
    }
    hand.setMissedRon(true);
    Hand copy = new Hand(Variant.TENHOU_4P);
    copy.copyFrom(hand);
    Fields.assertSameFields(hand, copy);
    copy.add(Tiles.physicalId(Tiles.CHUN, 0));
    copy.setMissedRon(false);
    Assert.assertTrue(hand.missedRon());
    Assert.assertFalse(hand.holds(Tiles.physicalId(Tiles.CHUN, 0)));
    Assert.assertEquals(hand.waits(), Tiles.bit(Tiles.TON) | Tiles.bit(Tiles.NAN));
    Assert.assertEquals(copy.waits(), 0L);
    Assert.assertThrows(IllegalArgumentException.class, () -> copy.copyFrom(new Hand(Variant.TENHOU_3P)));
  }

  private static void assertHand(Hand hand, boolean[] held, ShapeAnalyzer analyzer) {
    int[] counts = new int[Tiles.KINDS];
    for (int physicalId = 0; physicalId < Tiles.PHYSICAL_IDS; physicalId++) {
      Assert.assertEquals(hand.holds(physicalId), held[physicalId], "tile " + physicalId);
      counts[Tiles.kind(physicalId)] += held[physicalId] ? 1 : 0;
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Assert.assertEquals(hand.count(kind), counts[kind]);
    }
    HandShape recount = ShapeFixtures.shape(counts, 0);
    Fields.assertSameFields(recount, hand.shape());
    Assert.assertEquals(hand.waits(), recount.size() == Variant.HAND ? analyzer.waits(recount) : 0L);
  }

  private static int randomHeld(Random random, boolean[] held) {
    while (true) {
      int physicalId = random.nextInt(Tiles.PHYSICAL_IDS);
      if (held[physicalId]) {
        return physicalId;
      }
    }
  }

  /** その種類の物理牌のうち、手牌に無く、番号がいちばん大きいもの。赤（番号 0）は最後に選ぶ。 */
  private static int firstFree(Hand hand, int kind) {
    for (int copy = Tiles.COPIES - 1; copy >= 0; copy--) {
      if (!hand.holds(Tiles.physicalId(kind, copy))) {
        return Tiles.physicalId(kind, copy);
      }
    }
    throw new IllegalStateException("All copies of kind " + kind + " are held");
  }
}
