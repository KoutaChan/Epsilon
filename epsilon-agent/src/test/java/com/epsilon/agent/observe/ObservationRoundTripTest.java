package com.epsilon.agent.observe;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.agent.observe.ObservationSchema.LastEvent;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.Legal;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.engine.WinSettlementProjection;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Factory;
import org.testng.annotations.Test;

/**
 * 観測の記録（{@link Observation}）を書いて読み戻すテスト。ランダムに打つ自己対局で、判断の時点ごとに、判断する全ての席の観測を
 * 書く。読んだ値は、席から見た局面、席ごとの判断、エンジンがルールから求める値（鳴いた後の打牌、和了したときの精算の見込み）と
 * 一致する。観測の {@code long} の列を別のバッチの別の位置へ写して読んでも、同じ値になる。観測に書かずに求める値（場風・初巡・
 * 鳴きの有無・第一打の前か・ツモ牌・同値な打牌・遷移の数・槓ドラの表示の位置・抜いた北の数）も、局面と照らし合わせる。四人打ちと
 * 三人打ちの両方で回す。
 */
public class ObservationRoundTripTest {

  private static final int STEPS = 300_000;

  /** 写す先のバッチを空にするまでに足す観測の数。写す先での位置が元と違うように、毎回は空にせず続けて足す。 */
  private static final int COPIES_PER_BATCH = 64;

  private final Variant variant;
  private final int seats;
  private final ObservationWriter writer;
  private final ObservationBatch batch;
  private final ObservationBatch copies;
  private final Observation read;
  private final Observation reread;
  private final Observation.EventRecord record = new Observation.EventRecord();
  private final RuleWorkspace ws;
  private final WinSettlementProjection projection;
  private final short[] discards = new short[ObservationSchema.MAX_CALL_DISCARDS];
  private final int[] kinds = new int[EngineDecisionKind.values().length];
  private boolean kindsRejected;
  private long decisions;
  private long callDiscards;
  private long wins;
  private long kakans;
  private long kanDoras;
  private long nukis;

  @Factory(dataProvider = "variants")
  public ObservationRoundTripTest(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    writer = new ObservationWriter(variant);
    batch = new ObservationBatch(variant, 64);
    copies = new ObservationBatch(variant, 64);
    read = new Observation(variant);
    reread = new Observation(variant);
    ws = new RuleWorkspace(variant);
    projection = new WinSettlementProjection(variant);
  }

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(timeOut = 900_000)
  public void readsBackEveryDecisionOfRandomGames() {
    RandomPlay play = new RandomPlay(variant, 41);
    for (int step = 0; step < STEPS; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        visit(play);
      }
      play.step();
    }
    assertTrue(decisions > 200_000, "decisions " + decisions);
    // 抜いた北への応答には北で待つ聴牌が要るので、ランダムに打つ対局ではまず起きない。この応答は牌譜の ron-on-nuki で確かめる。
    for (EngineDecisionKind kind : EngineDecisionKind.values()) {
      if (kind.in(variant) && kind != EngineDecisionKind.NUKI_RESPONSE) {
        assertTrue(kinds[kind.ordinal()] > 0, "no " + kind + " decision");
      }
    }
    assertTrue(kindsRejected, "no decision with both discard and meld events");
    assertTrue(callDiscards > 0, "no call discard");
    assertTrue(wins > 0, "no immediate win");
    assertTrue(kakans > 0, "no kakan");
    assertTrue(kanDoras > 0, "no kan dora");
    assertEquals(nukis > 0, variant.has(RuleFlag.NUKI), nukis + " nuki reads");
  }

  private void visit(RandomPlay play) {
    Decisions boundary = play.engine.decisions();
    batch.clear();
    for (int slot = 0; slot < boundary.size(); slot++) {
      SeatView view = play.engine.view(boundary.seat(slot));
      int index = writer.capture(view, boundary, slot, batch);
      read.bind(batch, index);
      assertEquals(read.longs(), batch.longs(index));
      assertEquals(read.longs(), ObservationSchema.of(variant).longs(batch.words(), batch.start(index)));
      if (copies.size() == COPIES_PER_BATCH) {
        copies.clear();
      }
      reread.bind(copies, copies.add(read));
      matches(read, view, boundary, slot);
      matches(reread, view, boundary, slot);
      if (!kindsRejected) {
        otherKindRejected(read);
      }
      decisions++;
      kinds[boundary.kind(slot).ordinal()]++;
      for (int action = 0; action < read.legalActions(); action++) {
        callDiscards += read.callDiscardsOf(action);
      }
      wins += read.wins();
      kanDoras += read.kanDoraIndicators();
      for (int event = 0; event < read.events(); event++) {
        if (read.eventKind(event) == EventKind.MELD && read.meld(event).type() == Meld.Type.KAKAN) {
          kakans++;
        }
        if (read.eventKind(event) == EventKind.NUKI) {
          nukis++;
        }
      }
    }
  }

  /** 読んだ値が、席から見た局面・席ごとの判断・エンジンがルールから求める値と一致することを確かめる。 */
  private void matches(Observation r, SeatView view, Decisions boundary, int slot) {
    int observer = view.seat();
    assertEquals(r.decisionKind(), boundary.kind(slot));
    assertEquals(r.observer(), observer);
    assertEquals(r.absoluteSeat(r.dealer()), view.dealer());
    assertEquals(r.kyoku(), view.round());
    assertEquals(r.roundWindTile(), variant.roundWind(view.round()));
    assertEquals(r.honba(), view.honba());
    assertEquals(r.kyotaku(), view.kyotaku());
    assertEquals(r.remainingWall(), view.liveLeft());
    assertEquals(r.furitenTemporary(), view.furiten());
    lastEvent(r, view.last(), observer);
    hand(r, view);
    dora(r, view);
    seats(r, view);
    events(r, view.events(), observer);
    legal(r, view, boundary, slot);
  }

  private static void lastEvent(Observation r, TurnEvent last, int observer) {
    switch (last.kind()) {
      case NONE -> {
        assertNull(r.lastEvent());
        assertEquals(r.lastEventSeat(), -1);
        assertEquals(r.lastEventTile(), -1);
      }
      case DRAW, DISCARD, KAN, NUKI -> {
        LastEvent expected =
            switch (last.kind()) {
              case DRAW -> LastEvent.DRAW;
              case DISCARD -> LastEvent.DISCARD;
              case NUKI -> LastEvent.NUKI;
              default -> LastEvent.KAN_ATTEMPT;
            };
        assertEquals(r.lastEvent(), expected);
        assertEquals(r.absoluteSeat(r.lastEventSeat()), last.seat());
        assertEquals(r.lastEventTile(), last.tile());
      }
    }
    assertEquals(r.lastEventAka(), last.red());
    assertEquals(r.lastEventRinshan(), last.kind() == TurnEvent.Kind.DRAW && last.rinshan());
    boolean pendingKind =
        last.kind() == TurnEvent.Kind.DRAW || last.kind() == TurnEvent.Kind.KAN || last.kind() == TurnEvent.Kind.NUKI;
    assertEquals(r.deferredDora(), pendingKind && last.doraPending());
    assertEquals(r.lastKanKind(), last.kind() == TurnEvent.Kind.KAN ? last.kanKind() : null);
    boolean own = last.kind() == TurnEvent.Kind.DRAW && last.seat() == observer;
    assertEquals(r.drawnTile(), own ? last.tile() : -1);
  }

  private void hand(Observation r, SeatView view) {
    int[] counts = new int[Tiles.KINDS];
    r.handCounts(counts);
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      assertEquals(r.handCount(kind), view.count(kind));
      assertEquals(counts[kind], view.count(kind));
    }
    TileSet set = variant.tiles();
    int aka = 0;
    for (int kind : new int[] {Tiles.M5, Tiles.P5, Tiles.S5}) {
      int red = Tiles.physicalId(kind, Tiles.RED_COPY);
      if (set.red(red) && view.holds(red)) {
        aka |= 1 << Tiles.Group.of(kind).ordinal();
      }
    }
    assertEquals(r.handAkaMask(), aka);
  }

  /** ドラ表示と、槓ドラの表示の位置を確かめる。槓ドラの表示の直前のイベントの番号に 1 を足すと、めくった時点のイベントの数になる。 */
  private void dora(Observation r, SeatView view) {
    assertEquals(r.indicators(), view.indicatorCount());
    int aka = 0;
    for (int index = 0; index < view.indicatorCount(); index++) {
      int tile = view.indicator(index);
      assertEquals(r.indicatorTile(index), Tiles.kind(tile));
      if (variant.tiles().red(tile)) {
        aka |= 1 << Tiles.Group.of(Tiles.kind(tile)).ordinal();
      }
    }
    assertEquals(r.indicatorAkaMask(), aka);
    assertEquals(r.publicEvents(), r.events() + view.indicatorCount() - 1);
    for (int kan = 0; kan < r.kanDoraIndicators(); kan++) {
      assertEquals(r.kanDoraEvent(kan) + 1, view.indicatorEvent(kan + 1), "kan dora " + kan);
    }
  }

  private void seats(Observation r, SeatView view) {
    EventLog events = view.events();
    boolean firstCycle = false;
    for (int seat = 0; seat < seats; seat++) {
      int player = r.absoluteSeat(seat);
      assertEquals(r.nuki(seat), view.nuki(player));
      assertEquals(r.score(seat), view.score(player));
      assertEquals(r.riichi(seat), view.riichi(player));
      assertEquals(r.ippatsu(seat), view.ippatsu(player));
      assertEquals(r.beforeFirstDiscard(seat), events.riverSize(player) == 0);
      firstCycle |= events.riverSize(player) == 0;
    }
    assertEquals(r.firstCycle(), firstCycle);
  }

  /** 全てのイベントが、エンジンと同じ番号で読める。フィールドごとに読んでも、まとめて読んでも（{@code readEvent}）同じ値になる。 */
  private void events(Observation r, EventLog events, int observer) {
    assertEquals(r.events(), events.size());
    boolean called = false;
    for (int event = 0; event < events.size(); event++) {
      int author = Math.floorMod(events.seat(event) - observer, seats);
      r.readEvent(event, record);
      assertEquals(record.event(), event);
      assertEquals(record.author(), author);
      assertEquals(r.eventAuthor(event), author);
      switch (events.kind(event)) {
        case DISCARD -> {
          int calledBy = events.calledBy(event);
          int caller = calledBy == EventLog.NONE ? -1 : Math.floorMod(calledBy - observer, seats);
          assertEquals(r.eventKind(event), EventKind.DISCARD);
          assertEquals(record.kind(), EventKind.DISCARD);
          assertEquals(r.discardTile(event), events.tile(event));
          assertEquals(record.discardTile(), events.tile(event));
          assertEquals(r.discardAka(event), events.red(event));
          assertEquals(record.discardAka(), events.red(event));
          assertEquals(r.discardTsumogiri(event), events.tsumogiri(event));
          assertEquals(record.discardTsumogiri(), events.tsumogiri(event));
          assertEquals(r.discardRiichi(event), events.riichi(event));
          assertEquals(record.discardRiichi(), events.riichi(event));
          assertEquals(r.discardCalledBy(event), caller);
          assertEquals(record.discardCalledBy(), caller);
        }
        case MELD, KAKAN -> {
          called = true;
          assertEquals(r.eventKind(event), EventKind.MELD);
          assertEquals(record.kind(), EventKind.MELD);
          assertEquals(r.meld(event).id(), events.meld(event));
          assertEquals(record.meld().id(), events.meld(event));
        }
        case NUKI -> {
          assertEquals(r.eventKind(event), EventKind.NUKI);
          assertEquals(record.kind(), EventKind.NUKI);
          expectThrows(IllegalStateException.class, record::meld);
        }
      }
    }
    assertEquals(r.callMade(), called);
  }

  /** 種類ごとのフィールドは、その種類のイベントでなければ例外を投げる。打牌と副露の両方がある最初の判断で 1 回だけ確かめる。 */
  private void otherKindRejected(Observation r) {
    int discard = -1;
    int meld = -1;
    for (int event = 0; event < r.events(); event++) {
      if (r.eventKind(event) == EventKind.DISCARD) {
        discard = discard < 0 ? event : discard;
      } else if (r.eventKind(event) == EventKind.MELD) {
        meld = meld < 0 ? event : meld;
      }
    }
    if (discard < 0 || meld < 0) {
      return;
    }
    int discardEvent = discard;
    int meldEvent = meld;
    expectThrows(IllegalStateException.class, () -> r.meld(discardEvent));
    expectThrows(IllegalStateException.class, () -> r.discardTile(meldEvent));
    expectThrows(IllegalStateException.class, () -> r.discardCalledBy(meldEvent));
    expectThrows(IllegalStateException.class, () -> r.readEvent(meldEvent, record).discardTile());
    kindsRejected = true;
  }

  /** 合法手・鳴いた後の打牌・同値な打牌・遷移の数・選べる和了が、席ごとの判断やエンジンがルールから求める値と一致することを確かめる。 */
  private void legal(Observation r, SeatView view, Decisions boundary, int slot) {
    int count = boundary.legalCount(slot);
    assertEquals(r.legalActions(), count);
    int discardTotal = 0;
    int transitions = 0;
    int win = 0;
    for (int action = 0; action < count; action++) {
      int code = boundary.legal(slot, action);
      Action expected = Action.of(code);
      assertEquals(r.action(action), expected);
      int representative = action;
      for (int other = 0; other < action; other++) {
        if (twins(Action.of(boundary.legal(slot, other)), expected)) {
          representative = other;
          break;
        }
      }
      assertEquals(r.representative(action), representative);
      boolean call = expected.type() == Action.Type.CHI || expected.type() == Action.Type.PON;
      int callCount = call ? Legal.postCallDiscards(view, code, discards) : 0;
      assertEquals(r.callDiscardsOf(action), callCount);
      for (int discard = 0; discard < callCount; discard++) {
        assertEquals(r.callDiscard(discardTotal + discard).code(), discards[discard]);
      }
      discardTotal += callCount;
      int expectedTransitions = representative != action ? 0 : call ? callCount : 1;
      assertEquals(r.transitionsOf(action), expectedTransitions);
      transitions += expectedTransitions;
      if (expected.type() != Action.Type.TSUMO_AGARI && expected.type() != Action.Type.RON_AGARI) {
        assertEquals(r.winOf(action), -1);
        continue;
      }
      assertEquals(r.winOf(action), win);
      projection.load(view, code, ws);
      assertEquals(r.winHan(win), projection.score().han());
      assertEquals(r.winFuCode(win), ObservationSchema.encodeFu(projection.score().han(), projection.score().fu()));
      assertEquals(r.winYakuman(win), projection.score().yakuman());
      assertEquals(r.winPao(win), projection.pao() != RoundOutcome.NO_SEAT);
      assertEquals(r.winUraEligible(win), projection.uraEligible());
      assertEquals(r.winUraIndicators(win), projection.uraIndicators());
      assertEquals(r.winSettlement(win), projection.assumption());
      for (int seat = 0; seat < seats; seat++) {
        int player = r.absoluteSeat(seat);
        assertEquals(r.winRank(win, seat), projection.rank(player));
        assertEquals(r.winPayment(win, seat), projection.payment(player));
      }
      win++;
    }
    assertEquals(r.transitions(), transitions);
    assertEquals(r.countTransitions(), transitions);
    assertEquals(r.wins(), win);
  }

  /** 2 つの手が同値なら true。同値な打牌とは、種類（打牌か立直の打牌）・牌・赤の選び方が同じで、ツモ切りかどうかだけが違う打牌。 */
  private static boolean twins(Action first, Action second) {
    Action.Type type = first.type();
    return (type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI)
        && second.type() == type
        && second.tile() == first.tile()
        && second.selection().red() == first.selection().red();
  }
}
