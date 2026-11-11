package com.epsilon.prism4p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.input.EventTable;
import org.testng.annotations.Test;

/**
 * イベントを 1 回だけ読んで 3 つの表をまとめて作ると（{@link EventTable#load(Observation, TileCounts, SeatFacts)}）、表ごとに別に読んで
 * 作った結果と全ての値で一致する。表は判断ごとに使い回すので、前の判断の値が残っていれば見つかる。同じ手の代表が、定義どおり同じ手のうち
 * 番号の最も小さい手になることも照合する。
 */
public class SharedEventReadTest {

  private static final int GAMES = 12;

  @Test(timeOut = 600_000)
  public void sharedReadGivesTheSeparateTables() {
    Walk walk = new Walk();
    for (TableWalk.Policy policy : TableWalk.Policy.values()) {
      new TableWalk(TENHOU_4P, policy).run(9_100 + policy.ordinal(), GAMES, walk);
    }
    assertTrue(walk.decisions > 10_000, "decisions " + walk.decisions);
    assertTrue(walk.kans > 0 && walk.riichi > 0, "kans " + walk.kans + " riichi " + walk.riichi);
  }

  private static final class Walk implements TableWalk.Visitor {
    private final RowEncoder encoder = new RowEncoder();
    private final EventTable separateEvents = new EventTable(TENHOU_4P);
    private final TileCounts separateTiles = new TileCounts(TENHOU_4P);
    private final SeatFacts separateSeats = new SeatFacts(TENHOU_4P);
    private final EventTable sharedEvents = new EventTable(TENHOU_4P);
    private final TileCounts sharedTiles = new TileCounts(TENHOU_4P);
    private final SeatFacts sharedSeats = new SeatFacts(TENHOU_4P);
    private long decisions;
    private long kans;
    private long riichi;

    @Override
    public void decision(Engine engine, int slot) {
      Observation o = encoder.capture(engine, slot);
      separateEvents.load(o);
      separateTiles.load(o);
      separateSeats.load(o);
      sharedEvents.load(o, sharedTiles, sharedSeats);
      String where = "decision " + decisions;
      events(where);
      tiles(where);
      seats(where);
      representatives(o, where);
      decisions++;
      kans += sharedEvents.kans();
      for (int seat = 0; seat < TENHOU_4P.seats(); seat++) {
        riichi += sharedSeats.riichiDeclaration(seat) >= 0 ? 1 : 0;
      }
    }

    private void events(String where) {
      EventTable a = separateEvents;
      EventTable b = sharedEvents;
      assertEquals(b.events(), a.events(), where);
      assertEquals(b.publicEvents(), a.publicEvents(), where);
      assertEquals(b.kans(), a.kans(), where);
      assertEquals(b.indicators(), a.indicators(), where);
      for (int e = 0; e < a.events(); e++) {
        assertSame(b.kind(e), a.kind(e), where);
        assertEquals(b.author(e), a.author(e), where);
        assertEquals(b.tile(e), a.tile(e), where);
        assertEquals(b.aka(e), a.aka(e), where);
        assertEquals(b.tsumogiri(e), a.tsumogiri(e), where);
        assertEquals(b.riichi(e), a.riichi(e), where);
        assertEquals(b.calledBy(e), a.calledBy(e), where);
        assertEquals(b.discardIndex(e), a.discardIndex(e), where);
        assertSame(b.meld(e), a.meld(e), where);
        assertEquals(b.link(e), a.link(e), where);
        assertEquals(b.position(e), a.position(e), where);
      }
      for (int p = 0; p < a.publicEvents(); p++) {
        assertEquals(b.entity(p), a.entity(p), where);
      }
      for (int k = 0; k < a.kans(); k++) {
        assertEquals(b.kanEvent(k), a.kanEvent(k), where);
      }
      for (int seat = 0; seat < TENHOU_4P.seats(); seat++) {
        assertEquals(b.discards(seat), a.discards(seat), where);
        assertEquals(b.lastDiscard(seat), a.lastDiscard(seat), where);
        assertEquals(b.riichiDeclaration(seat), a.riichiDeclaration(seat), where);
        assertEquals(b.meldCount(seat), a.meldCount(seat), where);
      }
    }

    private void tiles(String where) {
      TileCounts a = separateTiles;
      TileCounts b = sharedTiles;
      long doraKinds = 0L;
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        assertEquals(b.hand(kind), a.hand(kind), where);
        assertEquals(b.visible(kind), a.visible(kind), where);
        assertEquals(b.unseen(kind), a.unseen(kind), where);
        assertEquals(b.dora(kind), a.dora(kind), where);
        assertEquals(b.indicators(kind), a.indicators(kind), where);
        if (a.dora(kind) > 0) {
          doraKinds |= Tiles.bit(kind);
        }
      }
      assertEquals(b.totalUnseen(), a.totalUnseen(), where);
      assertEquals(b.handRed(), a.handRed(), where);
      assertEquals(b.unseenRed(), a.unseenRed(), where);
      assertEquals(a.doraKinds(), doraKinds, where);
      assertEquals(b.doraKinds(), doraKinds, where);
    }

    private void seats(String where) {
      SeatFacts a = separateSeats;
      SeatFacts b = sharedSeats;
      for (int seat = 0; seat < TENHOU_4P.seats(); seat++) {
        assertEquals(b.meldCount(seat), a.meldCount(seat), where);
        for (int index = 0; index < a.meldCount(seat); index++) {
          assertSame(b.meld(seat, index), a.meld(seat, index), where);
        }
        assertEquals(b.openMelds(seat), a.openMelds(seat), where);
        assertEquals(b.kans(seat), a.kans(seat), where);
        assertEquals(b.riichiDeclaration(seat), a.riichiDeclaration(seat), where);
        assertEquals(b.riverKinds(seat), a.riverKinds(seat), where);
        assertEquals(b.riichiSafe(seat), a.riichiSafe(seat), where);
        assertEquals(b.rank(seat), a.rank(seat), where);
      }
    }

    private static void representatives(Observation o, String where) {
      for (int action = 0; action < o.legalActions(); action++) {
        int expected = action;
        for (int other = 0; other < action; other++) {
          if (Observation.twins(o.action(action), o.action(other))) {
            expected = other;
            break;
          }
        }
        assertEquals(o.representative(action), expected, where + " action " + action);
      }
    }
  }
}
