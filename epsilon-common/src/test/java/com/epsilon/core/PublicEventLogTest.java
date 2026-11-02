package com.epsilon.core;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import org.testng.annotations.Test;

/** 河と副露の時系列は公開の出来事の記録への添字で、鳴かれた印は副露を足す時に記録が付ける。 */
public class PublicEventLogTest {

  @Test
  public void riverReadsItsDiscardsFromTheLog() {
    GameState state = round();
    for (int i = 0; i < 30; i++) {
      state.commitDiscard(0, i, i, i == 3, false, i % 2 == 0, i);
    }
    state.addMeld(1, Meld.pon(29, Meld.RelativeSource.KAMICHA), 0);

    River river = state.river(0);
    assertEquals(river.size(), 30);
    assertEquals(river.tileType(29), 29);
    assertEquals(river.event(29), 29);
    assertTrue(river.called(29));
    assertFalse(river.called(28));
    assertEquals(river.riichiDeclarationIndex(), 3);
    assertEquals(river.postRiichiDiscardCount(), 26);
    assertEquals(river.postRiichiTsumogiriCount(), 13);
    assertEquals(state.hand(1).meldCreationEvent(0), 30);
    assertEquals(state.hand(1).meldCallAfterRiverIndex(0), 0);
    assertEquals(state.hand(1).meldKanAfterRiverIndex(0), -1);
    assertTrue(state.isFirstTurnCallOccurred());
  }

  @Test
  public void kakanIsItsOwnEventAndKeepsThePon() {
    GameState state = round();
    state.commitDiscard(0, Tile.HAKU, false);
    Meld pon = Meld.pon(Tile.HAKU, Meld.RelativeSource.KAMICHA);
    state.addMeld(1, pon, 0);
    state.commitDiscard(1, Tile.M1, false);
    state.replacePonWithKakan(1, pon, Meld.kakan(pon), 1);

    Hand hand = state.hand(1);
    assertEquals(hand.meld(0).type(), Meld.Type.KAKAN);
    assertEquals(hand.meldCreationEvent(0), 1);
    assertEquals(hand.meldKakanEvent(0), 3);
    assertEquals(hand.meldCallAfterRiverIndex(0), 0);
    assertEquals(hand.meldKanAfterRiverIndex(0), 1);
  }

  @Test
  public void forkCopiesTheLogIndependently() {
    GameState state = round();
    state.commitDiscard(0, Tile.M5, false);
    GameState fork = state.snapshotCopy();
    fork.addMeld(1, Meld.pon(Tile.M5, Meld.RelativeSource.KAMICHA), 0);
    fork.commitDiscard(1, Tile.P1, false);

    assertTrue(fork.river(0).called(0));
    assertFalse(state.river(0).called(0));
    assertEquals(state.river(1).size(), 0);
    assertFalse(state.isFirstTurnCallOccurred());
    assertEquals(fork.river(1).event(0), 2);
  }

  @Test
  public void callMustFollowItsDiscard() {
    GameState state = round();
    state.commitDiscard(0, Tile.M1, false);
    state.commitDiscard(1, Tile.M2, false);
    expectThrows(
        IllegalStateException.class,
        () -> state.addMeld(2, Meld.pon(Tile.M1, Meld.RelativeSource.TOIMEN), 0));
    expectThrows(
        IllegalStateException.class,
        () -> state.addMeld(2, Meld.pon(Tile.M1, Meld.RelativeSource.KAMICHA), 0));
    state.addMeld(2, Meld.pon(Tile.M2, Meld.RelativeSource.KAMICHA), 0);
    assertTrue(state.river(1).called(0));
  }

  private static GameState round() {
    GameState state = new GameState(1L);
    state.startRound(0, 0, 0, 0);
    return state;
  }
}
