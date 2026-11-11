package com.epsilon.agent.facts;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;
import static org.testng.Assert.fail;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.observe.RandomPlay;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EventKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 観測からルールで求めた値を、エンジンの局面から数え直した値と照合する。ランダムな自己対局で、判断の時点ごとに、判断する全ての席を
 * 確かめる。牌の枚数・ドラ・見えない赤は、席から見える公開の枚数と物理牌から数え直す。河・立直の後に通った種類・面子・順位は、席から
 * 見た局面から数え直す。受け入れは、牌の種類ごとに手に 1 枚足したときの向聴と照合し、待ちはエンジンの待ちと照合する。聴牌の手では、
 * 観測から作った採点の条件を局面の条件と照合し、待ちの点数を総当たり（{@link DirectScoring}）の点数と照合する。
 */
public class ObservedFactsTest {

  private static final Logger log = LoggerFactory.getLogger(ObservedFactsTest.class);
  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final TileSet SET = VARIANT.tiles();
  private static final int SEATS = VARIANT.seats();
  private static final int STEPS = 150_000;

  private final ObservationWriter writer = new ObservationWriter(VARIANT);
  private final ObservationBatch batch = new ObservationBatch(VARIANT, 64);
  private final Observation o = new Observation(VARIANT);
  private final TileCounts tiles = new TileCounts(VARIANT);
  private final SeatFacts seats = new SeatFacts(VARIANT);
  private final ShapeFacts shape = new ShapeFacts(VARIANT);
  private final WaitScores scores = new WaitScores(VARIANT);
  private final ShapeAnalyzer analyzer = new ShapeAnalyzer(VARIANT);
  private final DirectScoring brute = new DirectScoring(VARIANT);
  private final HandShape hand = new HandShape();
  private final HandShape drawn = new HandShape();
  private final int[] absoluteScores = new int[SEATS];
  private long decisions;
  private long improvingHands;
  private long tenpaiHands;
  private long riichiSafeSeats;
  private long kakanMelds;
  private boolean meldedSpecialRejected;

  @Test(timeOut = 900_000)
  public void factsMatchTheEngineAtEveryDecisionOfRandomGames() {
    RandomPlay play = new RandomPlay(VARIANT, 59);
    for (int step = 0; step < STEPS; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        visit(play.engine);
      }
      play.step();
    }
    log.info(
        String.format(
            Locale.ROOT,
            "FACTS: decisions %d improvingHands %d tenpaiHands %d scoredWins %d redWins %d riichiSafeSeats %d"
                + " kakans %d",
            decisions,
            improvingHands,
            tenpaiHands,
            brute.scoredWins(),
            brute.redWins(),
            riichiSafeSeats,
            kakanMelds));
    assertTrue(decisions > 100_000, "decisions " + decisions);
    assertTrue(improvingHands > 10_000, "improving hands " + improvingHands);
    assertTrue(tenpaiHands > 100, "tenpai hands " + tenpaiHands);
    assertTrue(brute.scoredWins() > 100, "scored wins " + brute.scoredWins());
    assertTrue(brute.redWins() > 0, "no red winning tile was scored");
    assertTrue(riichiSafeSeats > 0, "no riichi safe tiles");
    assertTrue(kakanMelds > 0, "no kakan");
    assertTrue(meldedSpecialRejected, "no melded hand");
  }

  private void visit(Engine engine) {
    Decisions boundary = engine.decisions();
    batch.clear();
    for (int slot = 0; slot < boundary.size(); slot++) {
      SeatView view = engine.view(boundary.seat(slot));
      o.bind(batch, writer.capture(view, boundary, slot, batch));
      tiles.load(o);
      seats.load(o);
      tileCounts(view);
      seatFacts(view);
      ObservedHand.shape(tiles, seats, hand);
      shape.analyze(hand, tiles);
      shapeFacts(view);
      if (shape.waits() != 0) {
        waitScores(view);
      }
      decisions++;
    }
  }

  private void tileCounts(SeatView view) {
    int total = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      assertEquals(tiles.hand(kind), view.count(kind));
      assertEquals(tiles.visible(kind), view.visible(kind), "visible " + kind);
      int unseen = SET.copies(kind) - view.visible(kind) - view.count(kind);
      assertEquals(tiles.unseen(kind), unseen);
      total += unseen;
      assertEquals(tiles.dora(kind), view.doraWeight(kind), "dora " + kind);
      int shown = 0;
      for (int index = 0; index < view.indicatorCount(); index++) {
        shown += Tiles.kind(view.indicator(index)) == kind ? 1 : 0;
      }
      assertEquals(tiles.indicators(kind), shown);
    }
    assertEquals(tiles.totalUnseen(), total);
    assertEquals(tiles.handRed(), heldRed(view));
    long unseenRed = 0;
    for (long fives = Tiles.FIVES & SET.kinds(); fives != 0; fives &= fives - 1) {
      int kind = Long.numberOfTrailingZeros(fives);
      int red = Tiles.physicalId(kind, Tiles.RED_COPY);
      if (!view.holds(red) && !shown(view, red)) {
        unseenRed |= Tiles.bit(kind);
      }
    }
    assertEquals(tiles.unseenRed(), unseenRed);
  }

  /** 席から見た自分の純手牌にある、赤の 5 の種類。 */
  private static long heldRed(SeatView view) {
    long red = 0;
    for (long fives = Tiles.FIVES & SET.kinds(); fives != 0; fives &= fives - 1) {
      int kind = Long.numberOfTrailingZeros(fives);
      if (view.holds(Tiles.physicalId(kind, Tiles.RED_COPY))) {
        red |= Tiles.bit(kind);
      }
    }
    return red;
  }

  /** 赤の物理牌が、表示牌・河・面子のどこかに出ているか。 */
  private static boolean shown(SeatView view, int red) {
    for (int index = 0; index < view.indicatorCount(); index++) {
      if (view.indicator(index) == red) {
        return true;
      }
    }
    int kind = Tiles.kind(red);
    EventLog events = view.events();
    for (int event = 0; event < events.size(); event++) {
      if (events.kind(event) == EventKind.DISCARD && events.red(event) && events.tile(event) == kind) {
        return true;
      }
    }
    for (int seat = 0; seat < SEATS; seat++) {
      for (int index = 0; index < view.meldCount(seat); index++) {
        Meld meld = Meld.of(view.meld(seat, index));
        if (meld.red() && Tiles.contains(meld.kinds(), kind)) {
          return true;
        }
      }
    }
    return false;
  }

  private void seatFacts(SeatView view) {
    for (int seat = 0; seat < SEATS; seat++) {
      absoluteScores[seat] = view.score(seat);
    }
    EventLog events = view.events();
    for (int seat = 0; seat < SEATS; seat++) {
      int player = (view.seat() + seat) % SEATS;
      assertEquals(seats.riverKinds(seat), view.riverKinds(player));
      assertEquals(seats.riichiSafe(seat), view.riichiSafe(player));
      assertEquals(seats.safe(seat), view.riverKinds(player) | view.riichiSafe(player));
      assertEquals(seats.rank(seat), Placement.rank(absoluteScores, player));
      assertEquals(seats.riichiDeclaration(seat), declaration(events, player));
      riichiSafeSeats += seats.riichiSafe(seat) != 0 ? 1 : 0;
      assertEquals(seats.meldCount(seat), view.meldCount(player));
      int open = 0;
      int kans = 0;
      for (int index = 0; index < view.meldCount(player); index++) {
        Meld meld = Meld.of(view.meld(player, index));
        assertEquals(seats.meld(seat, index), meld);
        open += meld.type().concealed() ? 0 : 1;
        kans += meld.type().kan() ? 1 : 0;
        kakanMelds += meld.type() == Meld.Type.KAKAN ? 1 : 0;
      }
      assertEquals(seats.openMelds(seat), open);
      assertEquals(seats.kans(seat), kans);
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        int danger = seats.danger(seat, kind, tiles);
        boolean safe = Tiles.contains(seats.safe(seat), kind);
        if (danger < 0 || danger >= SeatFacts.DANGER_CLASSES || (danger == 0) != safe) {
          fail("Danger " + danger + " of kind " + kind + " against " + Long.toBinaryString(seats.safe(seat)));
        }
      }
    }
  }

  /** その席が立直の宣言牌を打ったイベントの番号。立直していなければ −1。 */
  private static int declaration(EventLog events, int player) {
    for (int event = 0; event < events.size(); event++) {
      if (events.kind(event) == EventKind.DISCARD && events.seat(event) == player && events.riichi(event)) {
        return event;
      }
    }
    return -1;
  }

  private void shapeFacts(SeatView view) {
    assertEquals(hand.melds(), view.meldCount(view.seat()));
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      assertEquals(hand.count(kind), view.count(kind));
    }
    if (shape.special()) {
      int least = Math.min(shape.standardShanten(), Math.min(shape.chiitoitsuShanten(), shape.kokushiShanten()));
      assertEquals(shape.shanten(), least);
    } else {
      assertEquals(shape.shanten(), shape.standardShanten());
      if (!meldedSpecialRejected) {
        expectThrows(IllegalStateException.class, shape::chiitoitsuShanten);
        meldedSpecialRejected = true;
      }
    }
    if (hand.size() != Variant.HAND) {
      assertEquals(shape.improving(), 0L);
      assertEquals(shape.waits(), 0L);
      assertEquals(shape.acceptance(), 0);
      return;
    }
    assertEquals(shape.waits(), view.waits(), "waits of " + hand);
    long improving = 0;
    int acceptance = 0;
    for (long kinds = SET.kinds(); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      if (hand.count(kind) == Tiles.COPIES) {
        continue;
      }
      drawn.copyFrom(hand);
      drawn.add(kind);
      if (analyzer.shanten(drawn) < shape.shanten()) {
        improving |= Tiles.bit(kind);
        acceptance += SET.copies(kind) - view.visible(kind) - view.count(kind);
      }
    }
    assertEquals(shape.improving(), improving, "improving of " + hand);
    assertEquals(shape.acceptance(), acceptance);
    improvingHands++;
  }

  /** 観測から作った採点の条件を局面の条件と照合し、待ちの点数を総当たりの点数と照合する。 */
  private void waitScores(SeatView view) {
    tenpaiHands++;
    ObservedHand.context(o, tiles, seats, scores.context());
    WinContext context = scores.context();
    int observer = view.seat();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      assertEquals(context.hand().count(kind), view.count(kind));
    }
    assertEquals(context.meldCount(), view.meldCount(observer));
    for (int index = 0; index < context.meldCount(); index++) {
      assertEquals(context.meld(index).id(), view.meld(observer, index));
    }
    assertEquals(context.seatWind(), VARIANT.seatWind(view.round(), observer));
    assertEquals(context.roundWind(), VARIANT.roundWind(view.round()));
    assertEquals(context.indicatorCount(), view.indicatorCount());
    for (int index = 0; index < view.indicatorCount(); index++) {
      assertEquals(context.indicator(index), Tiles.kind(view.indicator(index)));
    }
    assertEquals(context.riichi(), view.riichi(observer));
    assertEquals(context.redFives(), heldRed(view));
    long waits = shape.waits();
    scores.load(waits, tiles.unseenRed());
    assertEquals(context.redFives(), heldRed(view));
    brute.check(scores, waits, tiles.unseenRed());
  }
}
