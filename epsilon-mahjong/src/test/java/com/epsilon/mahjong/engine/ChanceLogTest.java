package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 偶然のイベント（ツモ・ドラ表示牌・嶺上ツモ・裏ドラ表示牌）の記録を確かめる。記録の順と席、誰にも見えていない牌の枚数を照合する。 */
public class ChanceLogTest {

  @Test
  public void appendKeepsTheOrderAndTheSeats() {
    ChanceLog log = new ChanceLog(Variant.TENHOU_4P);
    log.begin(new Hand[] {new Hand(Variant.TENHOU_4P)}, 0);
    log.appendDraw(1, 9);
    log.appendIndicator(17);
    log.appendRinshan(2, 135);
    log.appendUra(100);
    Assert.assertEquals(log.size(), 4);
    ChanceKind[] kinds = {ChanceKind.DRAW, ChanceKind.INDICATOR, ChanceKind.RINSHAN, ChanceKind.URA};
    int[] seats = {1, ChanceLog.NO_SEAT, 2, ChanceLog.NO_SEAT};
    int[] tiles = {9, 17, 135, 100};
    for (int index = 0; index < kinds.length; index++) {
      Assert.assertEquals(log.kind(index), kinds[index]);
      Assert.assertEquals(log.seat(index), seats[index]);
      Assert.assertEquals(log.tile(index), tiles[index]);
    }
    Assert.assertEquals(log.start(Tiles.M1), 3);
    Assert.assertEquals(log.unseen(Tiles.M1), 3);
    Assert.assertEquals(log.unseen(Tiles.CHUN), 3);
    Assert.assertEquals(log.start(Tiles.CHUN), 4);
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> log.kind(4));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> log.appendDraw(4, 1));
  }

  /** 局を最後まで進める。どのイベントの後も、誰にも見えていない枚数が、配牌・最初のドラ表示牌・イベントから数え直した値と合う。 */
  @Test
  public void unseenCountsMatchARecountThroughWholeRounds() {
    Random random = new Random(31);
    for (Variant variant : Variant.values()) {
      TileSet set = variant.tiles();
      RandomRound round = new RandomRound(variant);
      round.start(
          RoundStart.first(variant),
          random.nextLong(),
          1,
          random.nextBoolean() ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER);
      for (int game = 0; game < 20; game++) {
        int[] dealt = new int[Tiles.KINDS];
        for (Hand hand : round.hands) {
          for (int kind = 0; kind < Tiles.KINDS; kind++) {
            dealt[kind] += hand.count(kind);
          }
        }
        int indicator = Tiles.kind(round.wall.indicator(0));
        dealt[indicator]++;
        for (int kind = 0; kind < Tiles.KINDS; kind++) {
          Assert.assertEquals(round.chance.start(kind), set.copies(kind) - dealt[kind]);
        }
        do {
          assertRecount(round.chance, set, dealt);
        } while (round.turn(random));
        round.revealUra();
        assertRecount(round.chance, set, dealt);
        int total = 0;
        for (int kind = 0; kind < Tiles.KINDS; kind++) {
          total += round.chance.unseen(kind);
        }
        Assert.assertEquals(total + round.chance.size() + Variant.HAND * variant.seats() + 1, set.size());
        Assert.assertEquals(round.wall.liveLeft(), 0);
        round.deal(RoundStart.first(variant));
      }
    }
  }

  @Test
  public void aTileThatEveryoneHasSeenCannotComeAgain() {
    ChanceLog log = new ChanceLog(Variant.TENHOU_3P);
    Hand hand = new Hand(Variant.TENHOU_3P);
    for (int copy = 0; copy < 3; copy++) {
      hand.add(Tiles.physicalId(Tiles.HAKU, copy));
    }
    log.begin(new Hand[] {hand}, Tiles.physicalId(Tiles.HAKU, 3));
    Assert.assertEquals(log.unseen(Tiles.HAKU), 0);
    Assert.assertThrows(IllegalStateException.class, () -> log.appendDraw(0, Tiles.physicalId(Tiles.HAKU, 3)));
    Assert.assertThrows(IllegalArgumentException.class, () -> log.appendDraw(0, Tiles.physicalId(Tiles.M5, 1)));
    Assert.assertEquals(log.size(), 0);
    hand.add(Tiles.physicalId(Tiles.HAKU, 3));
    Assert.assertThrows(IllegalArgumentException.class, () -> log.begin(new Hand[] {hand}, Tiles.physicalId(Tiles.HAKU, 0)));
  }

  @Test
  public void theCapacityHoldsEveryDrawKanDoraAndUra() {
    for (Variant variant : Variant.values()) {
      ChanceLog log = new ChanceLog(variant);
      log.begin(new Hand[] {new Hand(variant)}, variant.tiles().physicalId(0));
      int capacity = variant.liveWall() + Variant.MAX_KANS + 1 + Variant.MAX_KANS;
      for (int index = 1; index <= capacity; index++) {
        log.appendIndicator(variant.tiles().physicalId(index));
      }
      Assert.assertThrows(IllegalStateException.class, () -> log.appendUra(variant.tiles().physicalId(capacity + 1)));
    }
  }

  private static void assertRecount(ChanceLog log, TileSet set, int[] dealt) {
    int[] seen = dealt.clone();
    for (int index = 0; index < log.size(); index++) {
      seen[Tiles.kind(log.tile(index))]++;
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Assert.assertEquals(log.unseen(kind), set.copies(kind) - seen[kind], "kind " + kind);
    }
  }
}
