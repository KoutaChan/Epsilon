package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 1 席のビューは状態をコピーせずに読み、他家の伏せた牌を見せない。 */
public class SeatReaderTest {

  @Test
  public void aSeatReadsThePublicStateAndOnlyItsOwnConcealedTiles() {
    Random random = new Random(55);
    for (Variant variant : Variant.values()) {
      RandomRound round = new RandomRound(variant);
      round.start(RoundStart.first(variant), random.nextLong(), 1, DrawOrder.SEAT_STREAMS);
      for (int turn = 0; turn < 12 && round.turn(random); turn++) {
        for (int seat = 0; seat < variant.seats(); seat++) {
          assertView(round, new SeatReader(round.state, round.hands, round.decisions, seat));
        }
      }
    }
  }

  @Test
  public void anotherSeatsDrawIsShownWithoutItsTile() {
    RandomRound round = new RandomRound(Variant.TENHOU_4P);
    round.start(RoundStart.first(Variant.TENHOU_4P), 9, 1, DrawOrder.WALL_ORDER);
    int drawn = round.wall.draw(0);
    round.hands[0].add(drawn);
    round.state.draw(0, drawn, false);
    SeatReader own = new SeatReader(round.state, round.hands, round.decisions, 0);
    SeatReader other = new SeatReader(round.state, round.hands, round.decisions, 1);
    Assert.assertSame(own.last(), round.state.last());
    Assert.assertEquals(own.last().tile(), Tiles.kind(drawn));
    TurnEvent hidden = other.last();
    Assert.assertNotSame(hidden, round.state.last());
    Assert.assertEquals(hidden.kind(), TurnEvent.Kind.DRAW);
    Assert.assertEquals(hidden.seat(), 0);
    Assert.assertEquals(hidden.tile(), TurnEvent.NONE);
    Assert.assertFalse(hidden.red());
    Assert.assertFalse(hidden.rinshan());
    round.state.discard(0, drawn, true, false);
    round.hands[0].discard(drawn);
    Assert.assertSame(other.last(), round.state.last());
    Assert.assertEquals(other.last().tile(), Tiles.kind(drawn));
  }

  @Test
  public void theDecisionKindComesFromTheSeatsSlot() {
    RandomRound round = new RandomRound(Variant.TENHOU_3P);
    round.start(RoundStart.first(Variant.TENHOU_3P), 2, 1, DrawOrder.WALL_ORDER);
    round.decisions.open(2, EngineDecisionKind.NUKI_RESPONSE);
    round.decisions.add(Action.code(Action.Type.RON_AGARI));
    round.decisions.close(1);
    Assert.assertEquals(
        new SeatReader(round.state, round.hands, round.decisions, 2).kind(), EngineDecisionKind.NUKI_RESPONSE);
    SeatReader without = new SeatReader(round.state, round.hands, round.decisions, 1);
    Assert.assertThrows(IllegalStateException.class, without::kind);
    Assert.assertThrows(
        IndexOutOfBoundsException.class, () -> new SeatReader(round.state, round.hands, round.decisions, 3));
  }

  private static void assertView(RandomRound round, SeatView view) {
    PublicState state = round.state;
    Hand own = round.hands[view.seat()];
    Assert.assertSame(view.variant(), round.variant);
    Assert.assertEquals(view.round(), state.round());
    Assert.assertEquals(view.dealer(), state.dealer());
    Assert.assertEquals(view.current(), state.current());
    Assert.assertEquals(view.honba(), state.honba());
    Assert.assertEquals(view.kyotaku(), state.kyotaku());
    Assert.assertEquals(view.liveLeft(), round.wall.liveLeft());
    Assert.assertSame(view.events(), state.events());
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Assert.assertEquals(view.count(kind), own.count(kind));
      Assert.assertEquals(view.visible(kind), state.visible(kind));
      Assert.assertEquals(view.doraWeight(kind), state.dora().weight(kind));
    }
    for (int physicalId = 0; physicalId < Tiles.PHYSICAL_IDS; physicalId++) {
      Assert.assertEquals(view.holds(physicalId), own.holds(physicalId));
    }
    for (int seat = 0; seat < round.variant.seats(); seat++) {
      Hand hand = round.hands[seat];
      Assert.assertEquals(view.score(seat), state.score(seat));
      Assert.assertEquals(view.meldCount(seat), hand.meldCount());
      for (int index = 0; index < hand.meldCount(); index++) {
        Assert.assertEquals(view.meld(seat, index), hand.meld(index));
      }
      Assert.assertEquals(view.nuki(seat), hand.nuki());
      Assert.assertEquals(view.riichi(seat), state.riichi(seat));
      Assert.assertEquals(view.ippatsu(seat), state.ippatsu(seat));
      Assert.assertEquals(view.riverKinds(seat), state.riverKinds(seat));
      Assert.assertEquals(view.riichiSafe(seat), state.riichiSafe(seat));
    }
    Assert.assertEquals(view.indicatorCount(), state.dora().count());
    for (int index = 0; index < view.indicatorCount(); index++) {
      Assert.assertEquals(view.indicator(index), state.dora().indicator(index));
      Assert.assertEquals(view.indicatorEvent(index), state.dora().revealedAt(index));
    }
    Assert.assertEquals(view.furiten(), own.missedRon());
    Assert.assertEquals(view.waits(), own.waits());
    TurnEvent last = view.last();
    Assert.assertEquals(last.kind(), state.last().kind());
    Assert.assertEquals(last.seat(), state.last().seat());
    boolean hidden = last.kind() == TurnEvent.Kind.DRAW && last.seat() != view.seat();
    Assert.assertEquals(last.tile(), hidden ? TurnEvent.NONE : state.last().tile());
    int slot = round.decisions.slotOf(view.seat());
    if (slot >= 0) {
      Assert.assertEquals(view.kind(), round.decisions.kind(slot));
    }
  }
}
