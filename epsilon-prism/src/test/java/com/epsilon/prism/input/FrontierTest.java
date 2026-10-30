package com.epsilon.prism.input;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.HandAnalysisBuffer;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.Limits;
import com.epsilon.prism.codec.Bfloat16;
import com.epsilon.prism.input.FrontierField.Measure;
import com.epsilon.prism.input.FrontierField.Reach;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 前線の腕: 同じ判断の全ての遷移で、前線の牌のマスク（{@link TenpaiFrontier}）と {@link FrontierFeatures} が塊に書いた値（共有の枠で写した遷移を
 * 含む）が、エンジンの {@code analyzeTransition} の前線（自分の河に捨てる牌を足したもの）と一致する。
 */
public class FrontierTest {
  private static final Logger log = LoggerFactory.getLogger(FrontierTest.class);

  private static final long GREEDY_SEED = 500;
  private static final long CALLS_SEED = 8_200;
  private static final BatchPlan.Capacity ONE_ROW =
      new BatchPlan.Capacity(
          1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS);
  private static final Reach[] REACHES = Reach.values();

  @Test(timeOut = 1_200_000)
  public void frontierMatchesTheEngine() {
    Check greedy = new Check();
    new DecisionWalk(DecisionWalk.Policy.GREEDY).run(GREEDY_SEED, 40, greedy);
    Check calls = new Check();
    new DecisionWalk(DecisionWalk.Policy.CALLS).run(CALLS_SEED, 40, calls);
    log.info(greedy.summary("GREEDY"));
    log.info(calls.summary("CALLS"));
    assertTrue(greedy.searched > 10_000, "searched transitions " + greedy.searched);
    for (Reach reach : REACHES) {
      assertTrue(greedy.reached[reach.ordinal()] > 0, "no frontier of " + reach);
    }
    assertTrue(greedy.shared > 1_000, "shared slots " + greedy.shared);
    assertTrue(calls.callFrontiers > 0, "call transitions with a frontier " + calls.callFrontiers);
  }

  /** 1 つの打ち方の全ての判断を照らす。 */
  private static final class Check implements DecisionWalk.Visitor {
    private final ObservationWriter writer = new ObservationWriter();
    private final ObservationArena arena = new ObservationArena(256);
    private final Observation read = new Observation();
    private final HostBatch batch =
        new HostBatch(ONE_ROW, ByteBuffer.allocate(BatchPlan.maxBytes(ONE_ROW)));
    private final FrontierBatch frontierBatch =
        new FrontierBatch(batch, ByteBuffer.allocate(FrontierBatch.maxBytes(ONE_ROW)));
    private final FrontierFeatures features = new FrontierFeatures();
    private final EngineDecisionBuffer engineAnalysis = new EngineDecisionBuffer();
    private final EventTable events = new EventTable();
    private final TileCounts tiles = new TileCounts();
    private final TileSafety safety = new TileSafety();
    private final TransitionHands hands = new TransitionHands();
    private final ShapeAnalysis shapes = new ShapeAnalysis();
    private final TenpaiFrontier frontier = new TenpaiFrontier();
    private final boolean[] seen = new boolean[ShapeAnalysis.UNSHARED + 1];
    private final long[] reached = new long[REACHES.length];
    private long decisions;
    private long transitions;
    private long searched;
    private long frontiers;
    private long shared;
    private long callFrontiers;

    @Override
    public void visit(GameEngine engine, EngineDecisionPoint point, int round) {
      PublicObservation o = engine.observation(point.player());
      List<Action> legal = point.legalActions();
      arena.clear();
      read.bind(arena.words(), writer.capture(o, point.kind(), legal, arena));
      BatchPlan plan = batch.plan();
      plan.clear();
      assertEquals(plan.add(read), 0);
      plan.seal();
      features.write(read, frontierBatch, 0);
      assertEquals(frontierBatch.bytes(), read.transitions() * FrontierBatch.TRANSITION_BYTES);

      engine.analyzeDecision(point, engineAnalysis);
      events.load(read);
      tiles.load(read, events);
      safety.load(events);
      hands.load(read, tiles, events);
      shapes.reset();
      Arrays.fill(seen, false);
      WinConditions conditions =
          WinConditions.publicDecision(
              SeatFacts.seatWindTile(read, 0), read.roundWindTile(), RiichiState.NONE);
      int transition = 0;
      for (int action = 0; action < legal.size(); action++) {
        if (read.representative(action) != action) {
          continue;
        }
        Action.Type type = read.action(action).type();
        for (int index = 0; index < read.transitionsOf(action); index++, transition++) {
          transition(read, action, index, type, transition, conditions);
        }
      }
      assertEquals(transition, read.transitions());
      decisions++;
    }

    private void transition(
        Observation read,
        int action,
        int index,
        Action.Type type,
        int transition,
        WinConditions conditions) {
      String where = read.action(action) + " #" + index;
      AfterHand after = hands.move(action, index);
      Action discard = engineAnalysis.discardAction(action, index);
      Action mine = hands.discard();
      assertEquals(mine == null ? -1 : mine.tileType(), discard == null ? -1 : discard.tileType(), where);
      long river =
          engineAnalysis.ownRiverTileTypeMask() | (discard == null ? 0L : 1L << discard.tileType());
      assertEquals(safety.river(0) | (mine == null ? 0L : 1L << mine.tileType()), river, where);
      // 前線は立直の構えを見ない（エンジンも立直なしの条件で探す）ので、待ちの採点の構えは立直なしで足りる。
      HandAnalysisBuffer expected =
          engineAnalysis.analyzeTransition(action, index, RiichiState.NONE, river);
      long[] masks = {
        expected.furitenFreeRonFrontierTileTypeMask(),
        expected.furitenOnlyRonFrontierTileTypeMask(),
        expected.tsumoFrontierTileTypeMask()
      };
      int[] live = {
        expected.furitenFreeRonFrontierCopies(),
        expected.furitenOnlyRonFrontierCopies(),
        expected.tsumoFrontierCopies()
      };
      if (!TransitionHands.terminal(type)) {
        int slot = hands.slot();
        if (slot != ShapeAnalysis.UNSHARED && seen[slot]) {
          shared++;
        }
        seen[slot] = true;
        shapes.analyze(after, tiles, ShapeAnalysis.UNSHARED);
        if (after.effectiveTileCount() == 13 && shapes.minimum() == 1) {
          searched++;
        }
      }
      // 和了・九種九牌の後の手（14 枚相当か聴牌）でも、探す手の決まりで前線は空になる。
      frontier.analyze(after, river, tiles, conditions);
      boolean any = false;
      for (Reach reach : REACHES) {
        long mask = masks[reach.ordinal()];
        assertEquals(frontier.mask(reach), mask, reach + " at " + where);
        if (mask != 0L) {
          reached[reach.ordinal()]++;
          any = true;
        }
      }
      for (FrontierField field : FrontierField.values()) {
        int reach = field.reach().ordinal();
        int count = field.measure() == Measure.KINDS ? Long.bitCount(masks[reach]) : live[reach];
        short written =
            frontierBatch
                .blob()
                .getShort((transition * FrontierField.WIDTH + field.ordinal()) * Short.BYTES);
        assertEquals(
            written,
            Bfloat16.fromFloat(count / field.measure().scale()),
            field + " at " + where);
      }
      transitions++;
      if (any) {
        frontiers++;
        callFrontiers += type == Action.Type.CHI || type == Action.Type.PON ? 1 : 0;
      }
    }

    String summary(String policy) {
      return String.format(
          Locale.ROOT,
          "FRONTIER %s: decisions %d transitions %d searched %d withFrontier %d freeRon %d"
              + " furitenRon %d tsumo %d sharedSlots %d callFrontiers %d",
          policy,
          decisions,
          transitions,
          searched,
          frontiers,
          reached[0],
          reached[1],
          reached[2],
          shared,
          callFrontiers);
    }
  }
}
