package com.epsilon.prism.observe;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.River;
import com.epsilon.core.RoundLimits;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.WinSettlementProjection;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.Limits;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.observe.ObservationSchema.EventKind;
import com.epsilon.prism.observe.ObservationSchema.HeaderBits;
import com.epsilon.prism.observe.ObservationSchema.LastEvent;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 記録 O の捕捉と読み。
 *
 * <p>CALLS と GREEDY の固定の種の全ての判断で、捕捉 → 読む と、O の {@code long} の切り出しを別の置き場の別の位置に写して読む、の
 * 両方で、読んだ値が公開の観測・合法手・エンジンの判定（チー・ポンの後の打牌、即時の和了）と一致する。O に書かずに出す値（場風・初巡・
 * 鳴きの有無・第一打の前）も公開の観測と照らす。双子と遷移の数は Twins とエンジンの遷移の数から数え直して照らす。槓ドラの表示の位置の
 * 決まりは、判断を待つ全ての段（強制の判断だけの段を含む）で、表示が増えた最初の段の出来事の数と等号で照らす。公開の出来事（槓ドラの
 * 表示を含む）が 100 を超える局を含める。規則の上限を超える値は例外になる。
 */
public class ObservationRoundTripTest {
  private static final Logger log = LoggerFactory.getLogger(ObservationRoundTripTest.class);

  /** GREEDY の最初の半荘の種。 */
  private static final long SEED = 500;

  /**
   * CALLS の最初の半荘の種。4 万の半荘から、公開の出来事が 100 を超える局（種 8285・8441 で 103）を含むように
   * 選んだ。
   */
  private static final long CALLS_SEED = 8_200;

  @Test(timeOut = 1_200_000)
  public void readsBackEveryDecision() {
    log.info("OBSERVATION-T2 fingerprint {}", ObservationSchema.fingerprint());
    Check calls = walk(DecisionWalk.Policy.CALLS, CALLS_SEED, 300);
    Check greedy = walk(DecisionWalk.Policy.GREEDY, SEED, 60);
    for (Check check : List.of(calls, greedy)) {
      assertTrue(check.decisions > 10_000, "decisions " + check.decisions);
      assertTrue(check.kinds[EngineDecisionKind.RESPONSE.ordinal()] > 0, "no response decisions");
      assertTrue(check.kinds[EngineDecisionKind.POST_CALL_DAHAI.ordinal()] > 0, "no post-call");
      assertTrue(check.forcedSteps > 0, "no forced step");
      assertTrue(check.kindsRejected, "no decision with both discard and meld events");
    }
    assertTrue(calls.maxPublicEvents > 100, "max public events " + calls.maxPublicEvents);
    assertTrue(calls.kakans > 0, "no kakan");
    assertTrue(calls.kanDoraReveals > 0, "no kan dora reveal");
    assertTrue(greedy.wins > 0, "no immediate win");
    assertTrue(greedy.kinds[EngineDecisionKind.CHANKAN.ordinal()] > 0, "no chankan decision");
  }

  private static Check walk(DecisionWalk.Policy policy, long seed, int games) {
    Check check = new Check();
    new DecisionWalk(policy).run(seed, games, check);
    log.info(check.summary(policy));
    return check;
  }

  /** 110 符を超える手は 3 翻以上なら 110 符にまとめ（支払いは同じ）、3 翻未満なら例外にする。 */
  @Test
  public void foldsFuAboveTheTableAtThreeHan() {
    assertEquals(ObservationSchema.encodeFu(4, 120), ScoreMath.encodeFuCode(110));
    assertEquals(ObservationSchema.encodeFu(3, 110), ScoreMath.encodeFuCode(110));
    assertEquals(ObservationSchema.encodeFu(1, 30), ScoreMath.encodeFuCode(30));
    assertEquals(ObservationSchema.encodeFu(0, 0), ScoreMath.FU_CODE_NONE);
    expectThrows(IllegalArgumentException.class, () -> ObservationSchema.encodeFu(2, 120));
  }

  @Test
  public void rejectsValuesOutsideTheSchema() {
    GameEngine engine = new GameEngine(SEED);
    GameStepResult.AwaitingDecisions awaiting =
        (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    EngineDecisionPoint point = awaiting.decisions().get(0);
    PublicObservation o = engine.observation(point.player());
    List<Action> legal = point.legalActions();
    ObservationWriter writer = new ObservationWriter();
    ObservationArena arena = new ObservationArena(64);
    int base = writer.capture(o, point.kind(), legal, arena);
    int longs = ObservationSchema.longs(arena.words(), base);
    long[] truncated = Arrays.copyOfRange(arena.words(), base, base + longs - 1);
    expectThrows(IndexOutOfBoundsException.class, () -> new Observation().bind(truncated, 0));

    engine.getState().setScore(1, 25_050);
    expectThrows(IllegalArgumentException.class, () -> writer.capture(o, point.kind(), legal, arena));
    engine.getState().setScore(1, 25_000);
    engine.getState().setKyotakuCount(Limits.MAX_KYOTAKU + 1);
    expectThrows(IllegalArgumentException.class, () -> writer.capture(o, point.kind(), legal, arena));

    long[] header = new long[ObservationSchema.HEADER.longs(1)];
    ObservationSchema.HEADER.put(header, 0, 0, HeaderBits.WALL, RoundLimits.MAX_WALL);
    expectThrows(
        IllegalArgumentException.class,
        () ->
            ObservationSchema.HEADER.put(header, 0, 0, HeaderBits.WALL, RoundLimits.MAX_WALL + 1));
    expectThrows(
        IllegalArgumentException.class,
        () -> ObservationSchema.HEADER.put(header, 0, 0, HeaderBits.INDICATORS, 0));
    expectThrows(
        IllegalArgumentException.class,
        () -> ObservationSchema.HEADER.get(header, 0, 0, HeaderBits.INDICATOR));
  }

  /** 1 つの打ち方の全ての判断を照らし、数を数える。 */
  private static final class Check implements DecisionWalk.Visitor {
    private final ObservationWriter writer = new ObservationWriter();
    private final ObservationArena arena = new ObservationArena(256);
    private final ObservationArena copy = new ObservationArena(256);
    private final ObservationArena stepArena = new ObservationArena(256);
    private final Observation read = new Observation();
    private final Observation reread = new Observation();
    private final Observation step = new Observation();
    private final EngineDecisionBuffer judgment = new EngineDecisionBuffer();
    private final int[] kinds = new int[EngineDecisionKind.values().length];
    private int round = -1;
    private int lastIndicators;
    private boolean kindsRejected;
    private long decisions;
    private long forcedSteps;
    private long captureNanos;
    private long longs;
    private long callDiscards;
    private long wins;
    private long kakans;
    private long kanDoraReveals;
    private int maxEvents;
    private int maxPublicEvents;
    private long over100;
    private int maxRiver;

    @Override
    public void visit(GameEngine engine, EngineDecisionPoint point, int round) {
      PublicObservation o = engine.observation(point.player());
      List<Action> legal = point.legalActions();
      arena.clear();
      long started = System.nanoTime();
      int base = writer.capture(o, point.kind(), legal, arena);
      captureNanos += System.nanoTime() - started;
      read.bind(arena.words(), base);

      int length = ObservationSchema.longs(arena.words(), base);
      assertEquals(read.longs(), length);
      copy.clear();
      copy.allocate(3); // 写す先の位置を元と違う所にする
      int copied = copy.allocate(length);
      System.arraycopy(arena.words(), base, copy.words(), copied, length);
      reread.bind(copy.words(), copied);

      engine.analyzeDecision(point, judgment);
      matches(read, o, point);
      matches(reread, o, point);
      if (!kindsRejected) {
        otherKindRejected(read);
      }

      decisions++;
      kinds[point.kind().ordinal()]++;
      longs += read.longs();
      callDiscards += read.callDiscards();
      wins += read.wins();
      maxEvents = Math.max(maxEvents, read.events());
      maxPublicEvents = Math.max(maxPublicEvents, read.publicEvents());
      if (read.publicEvents() > 100) {
        over100++;
      }
      for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
        maxRiver = Math.max(maxRiver, o.river(player).size());
      }
      for (int event = 0; event < read.events(); event++) {
        if (read.eventKind(event) == EventKind.MELD
            && read.meld(event).type() == Meld.Type.KAKAN) {
          kakans++;
        }
      }
    }

    /**
     * 槓ドラの表示の位置: 判断を待つ全ての段で O を捕捉し、表示が増えた最初の段では、新しい表示の直前の出来事が今の最後の出来事に等しい。
     * ツモは出来事でないので、表示の後に判断を待つ最初の段の出来事の数は、表示の直前の出来事の番号 + 1 になる（加槓の槍槓の後も同じ）。
     */
    @Override
    public void awaiting(GameEngine engine, List<EngineDecisionPoint> points, int round) {
      EngineDecisionPoint point = points.get(0);
      boolean forced = true;
      for (EngineDecisionPoint each : points) {
        forced &= each.legalActions().size() == 1;
      }
      if (forced) {
        forcedSteps++;
      }
      stepArena.clear();
      int base =
          writer.capture(
              engine.observation(point.player()), point.kind(), point.legalActions(), stepArena);
      Observation r = step.bind(stepArena.words(), base);
      if (round != this.round) {
        this.round = round;
        lastIndicators = 1;
      }
      for (int kan = lastIndicators - 1; kan < r.kanDoraIndicators(); kan++) {
        assertEquals(r.kanDoraEvent(kan), r.events() - 1, "kan dora " + kan + " in round " + round);
        kanDoraReveals++;
      }
      lastIndicators = r.indicators();
    }

    /** 読んだ値が公開の観測・合法手・エンジンの判定と一致する。 */
    private void matches(Observation r, PublicObservation o, EngineDecisionPoint point) {
      int observer = point.player();
      assertEquals(r.decisionKind(), point.kind());
      assertEquals(r.observer(), observer);
      assertEquals(r.absoluteSeat(r.dealer()), o.dealer());
      assertEquals(r.kyoku(), o.roundIndex());
      assertEquals(r.roundWindTile(), o.roundWindTileType());
      assertEquals(r.honba(), o.honba());
      assertEquals(r.kyotaku(), o.riichiSticks());
      assertEquals(r.remainingWall(), o.remainingWallTiles());
      assertEquals(r.firstCycle(), o.isFirstTurn());
      assertEquals(r.callMade(), o.firstTurnCallOccurred());
      assertEquals(r.furitenTemporary(), o.isTemporaryFuriten(observer));
      lastEvent(r, o.turnEvent(), observer);

      PublicObservation.VisibleHand own = o.hand(observer);
      assertEquals(r.handAkaMask(), own.concealedAkaMask());
      for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
        assertEquals(r.handCount(tile), own.count(tile));
      }
      PublicObservation.VisibleDora dora = o.doraState();
      assertEquals(r.indicators(), dora.indicatorCount());
      for (int index = 0; index < dora.indicatorCount(); index++) {
        assertEquals(r.indicatorTile(index), dora.indicatorTileType(index));
      }
      assertEquals(r.indicatorAkaMask(), dora.visibleIndicatorAkaMask());
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        int player = r.absoluteSeat(seat);
        assertEquals(r.score(seat), o.score(player));
        RiichiState riichi =
            o.isDoubleRiichi(player)
                ? RiichiState.DOUBLE_RIICHI
                : o.isRiichi(player) ? RiichiState.RIICHI : RiichiState.NONE;
        assertEquals(r.riichi(seat), riichi);
        assertEquals(r.ippatsu(seat), o.isIppatsu(player));
        assertEquals(r.beforeFirstDiscard(seat), o.isBeforeFirstDiscard(player));
      }
      events(r, o, observer);
      legal(r, point.legalActions());
    }

    private static void lastEvent(Observation r, TurnEvent event, int observer) {
      switch (event) {
        case TurnEvent.Draw draw -> {
          tileEvent(r, LastEvent.DRAW, draw);
          assertEquals(r.lastEventRinshan(), draw.isRinshanDraw());
          assertEquals(r.deferredDora(), draw.doraRevealPending());
          assertNull(r.lastKanKind());
          boolean own = draw.player() == observer;
          assertEquals(r.drawnTile(), own ? draw.tileType() : -1);
          assertEquals(r.drawnAka(), own && draw.isAkaTile());
        }
        case TurnEvent.Discard discard -> {
          tileEvent(r, LastEvent.DISCARD, discard);
          assertNull(r.lastKanKind());
          assertEquals(r.drawnTile(), -1);
        }
        case TurnEvent.KanAttempt kan -> {
          tileEvent(r, LastEvent.KAN_ATTEMPT, kan);
          assertEquals(r.lastKanKind(), kan.kanKind());
          assertEquals(r.deferredDora(), kan.doraRevealPending());
        }
        case TurnEvent.None ignored -> {
          assertNull(r.lastEvent());
          assertEquals(r.lastEventSeat(), -1);
          assertEquals(r.lastEventTile(), -1);
          assertEquals(r.drawnTile(), -1);
        }
      }
    }

    private static void tileEvent(Observation r, LastEvent kind, TurnEvent.TileEvent event) {
      assertEquals(r.lastEvent(), kind);
      assertEquals(r.absoluteSeat(r.lastEventSeat()), event.player());
      assertEquals(r.lastEventTile(), event.tileType());
      assertEquals(r.lastEventAka(), event.isAkaTile());
    }

    /** 出来事: 河と副露の全てが、エンジンの番号の位置に読める。種類の違う欄を読むと例外になる。 */
    private void events(Observation r, PublicObservation o, int observer) {
      int total = 0;
      for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
        int author = Math.floorMod(player - observer, GameState.NUM_PLAYERS);
        River river = o.river(player);
        total += river.size();
        for (int index = 0; index < river.size(); index++) {
          int event = river.event(index);
          assertEquals(r.eventKind(event), EventKind.DISCARD);
          assertEquals(r.eventAuthor(event), author);
          assertEquals(r.discardTile(event), river.tileType(index));
          assertEquals(r.discardAka(event), river.aka(index));
          assertEquals(r.discardTsumogiri(event), river.tsumogiri(index));
          assertEquals(r.discardRiichi(event), river.riichiDeclaration(index));
          int caller = r.discardCalledBy(event);
          assertEquals(caller >= 0, river.called(index));
          if (caller >= 0) {
            assertEquals(r.eventKind(event + 1), EventKind.MELD);
            assertEquals(r.eventAuthor(event + 1), caller);
          }
        }
        PublicObservation.VisibleHand hand = o.hand(player);
        for (int index = 0; index < hand.meldCount(); index++) {
          Meld meld = hand.meld(index);
          int created = hand.meldCreationEvent(index);
          int kakan = hand.meldKakanEvent(index);
          total += kakan < 0 ? 1 : 2;
          assertEquals(r.eventKind(created), EventKind.MELD);
          assertEquals(r.eventAuthor(created), author);
          if (kakan < 0) {
            assertEquals(r.meld(created), meld);
          } else {
            assertEquals(r.meld(created).type(), Meld.Type.PON);
            assertEquals(Meld.kakan(r.meld(created), meld.addedTileIsAka()), meld);
            assertEquals(r.eventKind(kakan), EventKind.MELD);
            assertEquals(r.eventAuthor(kakan), author);
            assertEquals(r.meld(kakan), meld);
          }
        }
      }
      assertEquals(r.events(), total);
    }

    /** 種類ごとの欄は、その種類の出来事でなければ例外になる（打牌と副露の両方がある最初の判断で 1 回だけ）。 */
    private void otherKindRejected(Observation r) {
      int discard = -1;
      int meld = -1;
      for (int event = 0; event < r.events(); event++) {
        if (r.eventKind(event) == EventKind.DISCARD) {
          discard = discard < 0 ? event : discard;
        } else {
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
      kindsRejected = true;
    }

    /** 合法手・鳴いた後の打牌・双子・遷移の数・即時の和了がエンジンの判定と一致する。 */
    private void legal(Observation r, List<Action> legal) {
      assertEquals(r.legalActions(), legal.size());
      int discards = 0;
      int transitions = 0;
      int win = 0;
      for (int action = 0; action < legal.size(); action++) {
        Action expected = legal.get(action);
        assertEquals(r.action(action), expected);
        int representative = action;
        for (int other = 0; other < action; other++) {
          if (Twins.same(legal, action, other)) {
            representative = other;
            break;
          }
        }
        assertEquals(r.representative(action), representative);
        boolean call =
            expected.type() == Action.Type.CHI || expected.type() == Action.Type.PON;
        assertEquals(r.callDiscardsOf(action), call ? judgment.transitionCount(action) : 0);
        for (int discard = 0; discard < r.callDiscardsOf(action); discard++) {
          assertEquals(r.callDiscard(discards + discard), judgment.discardAction(action, discard));
        }
        discards += r.callDiscardsOf(action);
        int expectedTransitions = representative == action ? judgment.transitionCount(action) : 0;
        assertEquals(r.transitionsOf(action), expectedTransitions);
        transitions += expectedTransitions;
        WinSettlementProjection projection = judgment.immediateWin(action);
        if (projection == null) {
          assertEquals(r.winOf(action), -1);
          continue;
        }
        assertEquals(r.winOf(action), win);
        assertEquals(r.winHan(win), projection.visibleScore().visibleHan());
        assertEquals(
            r.winFuCode(win),
            ObservationSchema.encodeFu(
                projection.visibleScore().visibleHan(), projection.visibleScore().fu()));
        assertEquals(r.winYakuman(win), projection.visibleScore().yakumanMultiplier());
        assertEquals(r.winPao(win), projection.paoApplies());
        assertEquals(r.winUraEligible(win), projection.uraEligible());
        assertEquals(r.winUraIndicators(win), projection.uraIndicatorCount());
        assertEquals(r.winSettlement(win), projection.settlementAssumption());
        for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
          int player = r.absoluteSeat(seat);
          assertEquals(r.winRank(win, seat), projection.projectedRank(player));
          assertEquals(r.winPayment(win, seat), projection.paymentFloor(player));
        }
        win++;
      }
      assertEquals(r.callDiscards(), discards);
      assertEquals(r.transitions(), transitions);
      assertEquals(r.countTransitions(), transitions);
      assertEquals(r.wins(), win);
    }

    String summary(DecisionWalk.Policy policy) {
      return String.format(
          Locale.ROOT,
          "OBSERVATION-T2 %s: decisions %d kinds %s forcedSteps %d maxEvents %d"
              + " maxPublicEvents %d (>100: %d) maxRiver %d kakanReads %d kanDoraReveals %d"
              + " callDiscards %d wins %d bytes/decision %.1f capture %.0f ns/decision",
          policy,
          decisions,
          Arrays.toString(kinds),
          forcedSteps,
          maxEvents,
          maxPublicEvents,
          over100,
          maxRiver,
          kakans,
          kanDoraReveals,
          callDiscards,
          wins,
          8.0 * longs / decisions,
          (double) captureNanos / decisions);
    }
  }
}
