package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.action.Meld.RelativeSource;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 公開の状態を書くメソッドを確かめる。見えている枚数・河にある牌の種類・立直者の現物・ドラは、数え直した値と照合する。 */
public class PublicStateTest {

  @Test
  public void startReadsTheRoundStartAndEmptiesTheRound() {
    PublicState state = new PublicState(Variant.TENHOU_3P);
    state.reveal(Tiles.physicalId(Tiles.P1, 0));
    state.discard(0, Tiles.physicalId(Tiles.P2, 0), false, false);
    state.start(table(new RoundStart(Variant.TENHOU_3P, 4, 2, 1, new int[] {40_000, 30_000, 34_000})));
    Assert.assertEquals(state.round(), 4);
    Assert.assertEquals(state.dealer(), 1);
    Assert.assertEquals(state.current(), 1);
    Assert.assertEquals(state.honba(), 2);
    Assert.assertEquals(state.kyotaku(), 1);
    Assert.assertEquals(state.score(2), 34_000);
    Assert.assertEquals(state.liveLeft(), Variant.TENHOU_3P.liveWall());
    Assert.assertEquals(state.events().size(), 0);
    Assert.assertEquals(state.dora().count(), 0);
    Assert.assertEquals(state.visible(Tiles.P1), 0);
    Assert.assertEquals(state.riverKinds(0), 0L);
    Assert.assertEquals(state.last().kind(), TurnEvent.Kind.NONE);
    Assert.assertThrows(IllegalArgumentException.class, () -> state.start(table(RoundStart.first(Variant.TENHOU_4P))));
  }

  @Test
  public void discardsCallsAndKansCountTheVisibleTiles() {
    PublicState state = started(Variant.TENHOU_4P);
    state.reveal(Tiles.physicalId(Tiles.S4, 1));
    Assert.assertEquals(state.dora().weight(Tiles.S5), 1);
    state.draw(0, Tiles.physicalId(Tiles.S5, 0), false);
    Assert.assertEquals(state.last().kind(), TurnEvent.Kind.DRAW);
    Assert.assertEquals(state.last().tile(), Tiles.S5);
    Assert.assertTrue(state.last().red());
    state.discard(0, Tiles.physicalId(Tiles.S5, 0), true, false);
    Assert.assertEquals(state.last().kind(), TurnEvent.Kind.DISCARD);
    Assert.assertTrue(state.last().red());
    state.call(1, Meld.pon(Tiles.S5, RelativeSource.KAMICHA, AkaSource.CALLED_TILE).id());
    Assert.assertEquals(state.visible(Tiles.S5), 3);
    Assert.assertEquals(state.current(), 1);
    Assert.assertEquals(state.last().kind(), TurnEvent.Kind.NONE);
    Assert.assertEquals(state.riverKinds(0), Tiles.bit(Tiles.S5));
    state.discard(1, Tiles.physicalId(Tiles.HAKU, 0), false, false);
    state.draw(2, Tiles.physicalId(Tiles.HAKU, 1), false);
    state.ankan(2, Meld.ankan(Tiles.NAN, false).id());
    Assert.assertEquals(state.visible(Tiles.NAN), 4);
    Assert.assertEquals(state.last().kind(), TurnEvent.Kind.KAN);
    Assert.assertEquals(state.last().kanKind(), TurnEvent.KanKind.ANKAN);
    state.reveal(Tiles.physicalId(Tiles.S9, 0));
    Assert.assertEquals(state.dora().weight(Tiles.S1), 1);
    Assert.assertEquals(state.dora().revealedAt(1), 4);
    state.draw(2, Tiles.physicalId(Tiles.HAKU, 2), true);
    Assert.assertTrue(state.last().rinshan());
    state.discard(2, Tiles.physicalId(Tiles.HAKU, 2), true, false);
    state.draw(1, Tiles.physicalId(Tiles.S5, 1), false);
    state.kakan(1, Meld.kakan(Meld.pon(Tiles.S5, RelativeSource.KAMICHA, AkaSource.CALLED_TILE), false).id());
    Assert.assertEquals(state.visible(Tiles.S5), 4);
    Assert.assertEquals(state.last().kanKind(), TurnEvent.KanKind.KAKAN);
    Assert.assertFalse(state.last().red());
    Assert.assertEquals(state.kans(), 2);
    Assert.assertEquals(state.liveLeft(), Variant.TENHOU_4P.liveWall() - 4);
    Assert.assertEquals(state.rinshanDrawn(), 1);
    Assert.assertEquals(state.visible(Tiles.HAKU), 2);
    Assert.assertEquals(state.visible(Tiles.S9), 1);
    Assert.assertEquals(state.visible(Tiles.S4), 1);
  }

  @Test
  public void aKanDoraWaitsUntilTheNextReveal() {
    PublicState state = started(Variant.TENHOU_4P);
    state.reveal(Tiles.physicalId(Tiles.M1, 1));
    state.deferDora();
    Assert.assertTrue(state.dora().pending());
    state.draw(0, Tiles.physicalId(Tiles.M2, 1), true);
    Assert.assertTrue(state.last().doraPending());
    Assert.assertThrows(IllegalStateException.class, state::deferDora);
    state.reveal(Tiles.physicalId(Tiles.M3, 1));
    Assert.assertFalse(state.dora().pending());
    Assert.assertEquals(state.dora().count(), 2);
    state.draw(0, Tiles.physicalId(Tiles.M4, 1), false);
    Assert.assertFalse(state.last().doraPending());
  }

  @Test
  public void riichiPaysADepositAndCollectsLaterDiscardsAsSafe() {
    PublicState state = started(Variant.TENHOU_4P);
    state.discard(2, Tiles.physicalId(Tiles.P7, 1), false, true);
    Assert.assertEquals(state.riichi(2), RiichiState.NONE);
    state.riichi(2, false);
    Assert.assertEquals(state.riichi(2), RiichiState.RIICHI);
    Assert.assertTrue(state.ippatsu(2));
    Assert.assertEquals(state.kyotaku(), 1);
    Assert.assertEquals(state.score(2), 24_000);
    Assert.assertEquals(state.riichiSafe(2), 0L, "the declared tile is in the river, not after the riichi");
    state.discard(3, Tiles.physicalId(Tiles.P8, 1), false, false);
    state.discard(0, Tiles.physicalId(Tiles.P9, 1), false, true);
    state.riichi(0, true);
    Assert.assertEquals(state.riichi(0), RiichiState.DOUBLE_RIICHI);
    state.discard(1, Tiles.physicalId(Tiles.P1, 1), false, false);
    Assert.assertEquals(state.riichiSafe(2), Tiles.bit(Tiles.P8) | Tiles.bit(Tiles.P9) | Tiles.bit(Tiles.P1));
    Assert.assertEquals(state.riichiSafe(0), Tiles.bit(Tiles.P1));
    Assert.assertEquals(state.riichiMask(), 0b0101);
    state.clearIppatsu(2);
    Assert.assertFalse(state.ippatsu(2));
    Assert.assertTrue(state.ippatsu(0));
    state.clearIppatsu();
    Assert.assertFalse(state.ippatsu(0));
    Assert.assertThrows(IllegalStateException.class, () -> state.riichi(2, false));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> state.riichi(4));
  }

  @Test
  public void nukiShowsTheNorthAndKeepsItOutOfTheRiver() {
    PublicState state = started(Variant.TENHOU_3P);
    state.draw(0, Tiles.physicalId(Tiles.PEI, 2), false);
    state.nuki(0);
    Assert.assertEquals(state.visible(Tiles.PEI), 1);
    Assert.assertEquals(state.riverKinds(0), 0L);
    Assert.assertEquals(state.events().kind(0), EventKind.NUKI);
    Assert.assertEquals(state.last().kind(), TurnEvent.Kind.NUKI);
    Assert.assertEquals(state.last().tile(), Tiles.PEI);
    Assert.assertFalse(state.last().doraPending());
    state.deferDora();
    state.draw(0, Tiles.physicalId(Tiles.PEI, 3), true);
    state.nuki(0);
    Assert.assertEquals(state.visible(Tiles.PEI), 2);
    Assert.assertTrue(state.last().doraPending());
    Assert.assertThrows(IllegalStateException.class, () -> started(Variant.TENHOU_4P).nuki(0));
  }

  @Test
  public void anOpponentDrawWithoutATileKeepsOnlyThePublicFacts() {
    PublicState state = started(Variant.TENHOU_4P);
    state.draw(3, TurnEvent.NONE, true);
    Assert.assertEquals(state.last().tile(), TurnEvent.NONE);
    Assert.assertFalse(state.last().red());
    Assert.assertTrue(state.last().rinshan());
    Assert.assertEquals(state.current(), 3);
    Assert.assertThrows(IllegalArgumentException.class, () -> state.draw(3, Tiles.PHYSICAL_IDS, false));
  }

  /** 局を最後まで進める。どの手番の後も、見えている枚数・河にある牌の種類・ドラの重みが、イベントから数え直した値と合う。 */
  @Test
  public void visibleRiverAndDoraMatchARecountThroughWholeRounds() {
    Random random = new Random(41);
    for (Variant variant : Variant.values()) {
      RandomRound round = new RandomRound(variant);
      round.start(RoundStart.first(variant), random.nextLong(), 1, DrawOrder.WALL_ORDER);
      for (int game = 0; game < 20; game++) {
        do {
          assertRecount(round.state);
          Assert.assertEquals(round.state.liveLeft(), round.wall.liveLeft());
        } while (round.turn(random));
        round.deal(RoundStart.first(variant));
      }
    }
  }

  @Test
  public void aCopyEqualsTheOriginalAndStaysIndependent() {
    PublicState state = started(Variant.TENHOU_4P);
    state.reveal(Tiles.physicalId(Tiles.M1, 1));
    state.discard(0, Tiles.physicalId(Tiles.M2, 1), false, true);
    state.riichi(0, false);
    state.deferDora();
    PublicState copy = new PublicState(Variant.TENHOU_4P);
    copy.copyFrom(state);
    Fields.assertSameFields(state, copy);
    copy.discard(1, Tiles.physicalId(Tiles.M3, 1), false, false);
    copy.reveal(Tiles.physicalId(Tiles.M4, 1));
    Assert.assertEquals(state.events().size(), 1);
    Assert.assertEquals(state.riichiSafe(0), 0L);
    Assert.assertTrue(state.dora().pending());
    Assert.assertEquals(state.last().kind(), TurnEvent.Kind.DISCARD);
    Assert.assertThrows(IllegalArgumentException.class, () -> copy.copyFrom(new PublicState(Variant.TENHOU_3P)));
  }

  private static PublicState started(Variant variant) {
    PublicState state = new PublicState(variant);
    state.start(table(RoundStart.first(variant)));
    return state;
  }

  private static TableState table(RoundStart start) {
    TableState table = new TableState(start.variant());
    table.set(start);
    return table;
  }

  private static void assertRecount(PublicState state) {
    TileSet set = state.variant().tiles();
    EventLog events = state.events();
    int[] visible = new int[Tiles.KINDS];
    long[] rivers = new long[state.variant().seats()];
    for (int event = 0; event < events.size(); event++) {
      switch (events.kind(event)) {
        case DISCARD -> {
          visible[events.tile(event)]++;
          rivers[events.seat(event)] |= Tiles.bit(events.tile(event));
        }
        case MELD -> {
          Meld meld = Meld.of(events.meld(event));
          for (int index = 0; index < meld.size(); index++) {
            visible[meld.tile(index)]++;
          }
          if (meld.source() != RelativeSource.NONE) {
            visible[meld.called()]--;
          }
        }
        case KAKAN -> visible[Meld.of(events.meld(event)).base()]++;
        case NUKI -> visible[Tiles.PEI]++;
      }
    }
    int[] weights = new int[Tiles.KINDS];
    for (int index = 0; index < state.dora().count(); index++) {
      int indicator = Tiles.kind(state.dora().indicator(index));
      visible[indicator]++;
      weights[set.doraOf(indicator)]++;
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Assert.assertEquals(state.visible(kind), visible[kind], "visible " + kind);
      Assert.assertEquals(state.dora().weight(kind), weights[kind], "weight " + kind);
    }
    for (int seat = 0; seat < rivers.length; seat++) {
      Assert.assertEquals(state.riverKinds(seat), rivers[seat], "river " + seat);
    }
  }
}
