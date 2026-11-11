package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.action.Meld.RelativeSource;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 公開イベントの記録を確かめる。見るのは、追記の順、種類ごとのフィールド、鳴かれた打牌の書き換え、河での番号、容量。 */
public class EventLogTest {

  @Test
  public void appendKeepsTheOrderAndTheFieldsOfEachKind() {
    EventLog log = new EventLog(Variant.TENHOU_4P);
    Assert.assertEquals(log.appendDiscard(0, Tiles.M5, true, false, false), 0);
    int pon = Meld.pon(Tiles.M5, RelativeSource.KAMICHA, AkaSource.CALLED_TILE).id();
    Assert.assertEquals(log.appendMeld(1, pon), 1);
    Assert.assertEquals(log.appendDiscard(1, Tiles.CHUN, false, false, true), 2);
    int ankan = Meld.ankan(Tiles.S9, false).id();
    Assert.assertEquals(log.appendMeld(2, ankan), 3);
    Assert.assertEquals(log.appendDiscard(2, Tiles.S1, false, true, false), 4);
    int kakan = Meld.kakan(Meld.of(pon), false).id();
    Assert.assertEquals(log.appendKakan(1, kakan), 5);
    Assert.assertEquals(log.size(), 6);

    assertEvent(log, 0, EventKind.DISCARD, 0, Tiles.M5, EventLog.NONE);
    Assert.assertTrue(log.red(0));
    Assert.assertEquals(log.calledBy(0), 1);
    assertEvent(log, 1, EventKind.MELD, 1, EventLog.NONE, pon);
    assertEvent(log, 2, EventKind.DISCARD, 1, Tiles.CHUN, EventLog.NONE);
    Assert.assertTrue(log.riichi(2));
    Assert.assertFalse(log.tsumogiri(2));
    Assert.assertEquals(log.calledBy(2), EventLog.NONE);
    assertEvent(log, 3, EventKind.MELD, 2, EventLog.NONE, ankan);
    assertEvent(log, 4, EventKind.DISCARD, 2, Tiles.S1, EventLog.NONE);
    Assert.assertTrue(log.tsumogiri(4));
    assertEvent(log, 5, EventKind.KAKAN, 1, EventLog.NONE, kakan);
    Assert.assertEquals(log.meld(1), pon, "the pon event keeps the pon after the kakan");
    Assert.assertEquals(log.dahai(), 3);
    Assert.assertEquals(log.interruptions(), 3);
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> log.kind(6));
  }

  @Test
  public void aCallWritesOnlyItsDiscardAndMustFollowIt() {
    EventLog log = new EventLog(Variant.TENHOU_4P);
    log.appendDiscard(3, Tiles.P5, false, false, false);
    int chi = Meld.chi(Tiles.P4, Tiles.P5, AkaSource.NONE).id();
    int redChi = Meld.chi(Tiles.P4, Tiles.P5, AkaSource.CALLED_TILE).id();
    int otherKind = Meld.pon(Tiles.P4, RelativeSource.KAMICHA, AkaSource.NONE).id();
    Assert.assertThrows(IllegalStateException.class, () -> log.appendMeld(1, chi));
    Assert.assertThrows(IllegalStateException.class, () -> log.appendMeld(0, redChi));
    Assert.assertThrows(IllegalStateException.class, () -> log.appendMeld(0, otherKind));
    Assert.assertEquals(log.size(), 1);
    Assert.assertEquals(log.calledBy(0), EventLog.NONE);
    log.appendMeld(0, chi);
    Assert.assertEquals(log.calledBy(0), 0);
    Assert.assertEquals(log.calledBy(1), EventLog.NONE);
    Assert.assertThrows(IllegalStateException.class, () -> log.appendMeld(0, chi));
    log.appendDiscard(0, Tiles.SHA, false, false, false);
    int daiminkan = Meld.daiminkan(Tiles.SHA, RelativeSource.KAMICHA, AkaSource.NONE).id();
    Assert.assertThrows(IllegalStateException.class, () -> log.appendMeld(2, daiminkan));
    log.appendMeld(1, daiminkan);
    Assert.assertEquals(log.calledBy(2), 1);
    Assert.assertThrows(IllegalStateException.class, () -> log.appendMeld(3, daiminkan));
  }

  @Test
  public void riversListEachSeatsDiscardsInOrderIncludingCalledOnes() {
    EventLog log = new EventLog(Variant.TENHOU_3P);
    log.appendDiscard(0, Tiles.P1, false, false, false);
    log.appendDiscard(1, Tiles.P9, false, false, false);
    log.appendMeld(2, Meld.pon(Tiles.P9, RelativeSource.KAMICHA, AkaSource.NONE).id());
    log.appendDiscard(2, Tiles.S9, false, false, false);
    log.appendNuki(0);
    log.appendDiscard(0, Tiles.M9, false, true, false);
    Assert.assertEquals(log.riverSize(0), 2);
    Assert.assertEquals(log.river(0, 0), 0);
    Assert.assertEquals(log.river(0, 1), 5);
    Assert.assertEquals(log.riverSize(1), 1);
    Assert.assertEquals(log.river(1, 0), 1);
    Assert.assertEquals(log.calledBy(log.river(1, 0)), 2);
    Assert.assertEquals(log.riverSize(2), 1);
    Assert.assertEquals(log.river(2, 0), 3);
    Assert.assertEquals(log.kind(4), EventKind.NUKI);
    Assert.assertEquals(log.tile(4), Tiles.PEI);
    Assert.assertEquals(log.dahai(), 4);
    Assert.assertEquals(log.interruptions(), 2);
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> log.river(1, 1));
    log.clear();
    Assert.assertEquals(log.size(), 0);
    Assert.assertEquals(log.riverSize(0), 0);
    Assert.assertEquals(log.dahai(), 0);
    Assert.assertEquals(log.interruptions(), 0);
  }

  @Test
  public void aKakanNeedsTheSeatsPon() {
    EventLog log = new EventLog(Variant.TENHOU_4P);
    log.appendDiscard(0, Tiles.HAKU, false, false, false);
    Meld pon = Meld.pon(Tiles.HAKU, RelativeSource.KAMICHA, AkaSource.NONE);
    log.appendMeld(1, pon.id());
    int kakan = Meld.kakan(pon, false).id();
    Assert.assertThrows(IllegalStateException.class, () -> log.appendKakan(2, kakan));
    Meld other = Meld.pon(Tiles.HAKU, RelativeSource.TOIMEN, AkaSource.NONE);
    Assert.assertThrows(IllegalArgumentException.class, () -> log.appendKakan(1, Meld.kakan(other, false).id()));
    Assert.assertThrows(IllegalArgumentException.class, () -> log.appendKakan(1, pon.id()));
    Assert.assertEquals(log.appendKakan(1, kakan), 2);
  }

  @Test
  public void variantsLimitTheKindsOfEvents() {
    EventLog four = new EventLog(Variant.TENHOU_4P);
    Assert.assertThrows(IllegalStateException.class, () -> four.appendNuki(0));
    EventLog three = new EventLog(Variant.TENHOU_3P);
    three.appendDiscard(0, Tiles.P2, false, false, false);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> three.appendMeld(1, Meld.chi(Tiles.P1, Tiles.P2, AkaSource.NONE).id()));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.appendDiscard(0, Tiles.M5, false, false, false));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> three.appendDiscard(3, Tiles.P2, false, false, false));
    Assert.assertEquals(three.size(), 1);
  }

  @Test
  public void theCapacityIsTheVariantLimit() {
    for (Variant variant : Variant.values()) {
      EventLog log = new EventLog(variant);
      int seats = variant.seats();
      for (int event = 0; event < variant.maxPublicEvents(); event++) {
        log.appendDiscard(event % seats, Tiles.P1, false, false, false);
      }
      Assert.assertEquals(log.size(), variant.maxPublicEvents());
      Assert.assertThrows(IllegalStateException.class, () -> log.appendDiscard(0, Tiles.P1, false, false, false));
      Assert.assertEquals(log.size(), variant.maxPublicEvents());
    }
  }

  @Test
  public void aCopyEqualsTheOriginalAndStaysIndependent() {
    EventLog log = new EventLog(Variant.TENHOU_4P);
    log.appendDiscard(2, Tiles.S5, false, false, true);
    log.appendMeld(3, Meld.pon(Tiles.S5, RelativeSource.KAMICHA, AkaSource.CONSUMED_HAND_TILE).id());
    EventLog copy = new EventLog(Variant.TENHOU_4P);
    copy.copyFrom(log);
    Fields.assertSameFields(log, copy);
    copy.appendDiscard(3, Tiles.NAN, false, false, false);
    Assert.assertEquals(log.size(), 2);
    Assert.assertEquals(log.riverSize(3), 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> copy.copyFrom(new EventLog(Variant.TENHOU_3P)));
  }

  private static void assertEvent(EventLog log, int event, EventKind kind, int seat, int tile, int meld) {
    Assert.assertEquals(log.kind(event), kind, "event " + event);
    Assert.assertEquals(log.seat(event), seat, "event " + event);
    Assert.assertEquals(log.tile(event), tile, "event " + event);
    Assert.assertEquals(log.meld(event), meld, "event " + event);
  }
}
