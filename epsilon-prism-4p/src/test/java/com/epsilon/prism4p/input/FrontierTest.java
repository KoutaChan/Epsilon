package com.epsilon.prism4p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.codec.Bfloat16;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.HandScorer;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.input.AfterHand;
import com.epsilon.prism.input.ShapeAnalysis;
import com.epsilon.prism.input.TransitionHands;
import com.epsilon.prism4p.input.FrontierField.Measure;
import com.epsilon.prism4p.input.FrontierField.Reach;
import java.util.Arrays;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * フロンティアの計算のテスト。貪欲に打つ対局と鳴きを選ぶ対局の全ての遷移で、フロンティアの牌のマスク（{@link TenpaiFrontier}）と、
 * {@link FrontierFeatures} が求めた BF16 の値が、総当たりで求めたフロンティアと一致する。形の解析を他の遷移と共有した遷移も含む。
 * 総当たりでは、見えない牌を 1 枚引いて手の牌を 1 枚捨て、聴牌になれば、待ちのどれかに役があるかを直に採点する。河は席のビューから読む。
 */
public class FrontierTest {
  private static final Logger log = LoggerFactory.getLogger(FrontierTest.class);

  private static final Variant VARIANT = TENHOU_4P;
  private static final long GREEDY_SEED = 500;
  private static final long CALLS_SEED = 8_200;
  private static final Reach[] REACHES = Reach.values();
  private static final int THIRTEEN = HandShape.MAX_SIZE - 1;

  @Test(timeOut = 1_200_000)
  public void frontierMatchesBruteForce() {
    Check greedy = new Check();
    new TableWalk(VARIANT, TableWalk.Policy.GREEDY).run(GREEDY_SEED, 20, greedy);
    Check calls = new Check();
    new TableWalk(VARIANT, TableWalk.Policy.CALLS).run(CALLS_SEED, 20, calls);
    log.info(greedy.summary("GREEDY"));
    log.info(calls.summary("CALLS"));
    assertTrue(greedy.searched > 10_000, "searched transitions " + greedy.searched);
    for (Reach reach : REACHES) {
      assertTrue(greedy.reached[reach.ordinal()] > 0, "no frontier of " + reach);
    }
    assertTrue(greedy.shared > 1_000, "shared slots " + greedy.shared);
    assertTrue(calls.callFrontiers > 0, "call transitions with a frontier " + calls.callFrontiers);
  }

  /** 1 つの打ち方で、全ての判断を照合する。 */
  private static final class Check implements TableWalk.Visitor {
    private final RowEncoder encoder = new RowEncoder();
    private final FrontierFeatures features = new FrontierFeatures();
    private final TileCounts tiles = new TileCounts(VARIANT);
    private final SeatFacts seats = new SeatFacts(VARIANT);
    private final TransitionHands hands = new TransitionHands();
    private final TenpaiFrontier frontier = new TenpaiFrontier(VARIANT);
    private final ShapeAnalyzer analyzer = new ShapeAnalyzer(VARIANT);
    private final HandScorer scorer = new HandScorer();
    private final WinScore score = new WinScore();
    private final WinContext context = new WinContext(VARIANT);
    private final HandShape drawn = new HandShape();
    private final HandShape probe = new HandShape();
    private final boolean[] seen = new boolean[ShapeAnalysis.UNSHARED + 1];
    private final long[] expected = new long[REACHES.length];
    private final long[] reached = new long[REACHES.length];
    private long decisions;
    private long transitions;
    private long searched;
    private long shared;
    private long callFrontiers;
    private short[] values;

    @Override
    public void decision(Engine engine, int slot) {
      int seat = engine.decisions().seat(slot);
      Observation read = encoder.encode(engine, slot);
      values = features.values(read);
      tiles.load(read);
      seats.load(read);
      hands.load(read, tiles, seats);
      Arrays.fill(seen, false);
      long ownRiver = engine.view(seat).riverKinds(seat);
      int seatWind = VARIANT.seatWind(read.kyoku(), seat);
      int transition = 0;
      for (int action = 0; action < read.legalActions(); action++) {
        if (read.representative(action) != action) {
          continue;
        }
        Action.Type type = read.action(action).type();
        for (int index = 0; index < read.transitionsOf(action); index++, transition++) {
          transition(read, action, index, type, transition, ownRiver, seatWind);
        }
      }
      assertEquals(transition, read.transitions());
      decisions++;
    }

    private void transition(
        Observation read, int action, int index, Action.Type type, int transition, long ownRiver, int seatWind) {
      String where = read.action(action) + " #" + index;
      AfterHand after = hands.move(action, index);
      Action discard = hands.discard();
      long river = ownRiver | (discard == null ? 0L : Tiles.bit(discard.tile()));
      Arrays.fill(expected, 0L);
      if (!TransitionHands.terminal(type)) {
        int slot = hands.slot();
        if (slot != ShapeAnalysis.UNSHARED && seen[slot]) {
          shared++;
        }
        seen[slot] = true;
        bruteForce(after, river, read, seatWind);
      }
      // 和了・九種九牌の後の手（14 枚相当か聴牌）は総当たりで探さない。フロンティアも、探す手の条件から外れるので空になる。
      frontier.analyze(after, river, tiles, read, seatWind);
      boolean any = false;
      for (Reach reach : REACHES) {
        long mask = expected[reach.ordinal()];
        assertEquals(frontier.mask(reach), mask, reach + " at " + where);
        if (mask != 0L) {
          reached[reach.ordinal()]++;
          any = true;
        }
      }
      for (FrontierField field : FrontierField.values()) {
        long mask = expected[field.reach().ordinal()];
        int count = field.measure() == Measure.KINDS ? Long.bitCount(mask) : live(mask);
        short value = values[transition * FrontierField.WIDTH + field.ordinal()];
        assertEquals(value, Bfloat16.fromFloat(count / field.measure().scale()), field + " at " + where);
      }
      transitions++;
      if (any && (type == Action.Type.CHI || type == Action.Type.PON)) {
        callFrontiers++;
      }
    }

    /** 後の手が 13 枚相当で 1 向聴なら、フロンティアを総当たりで求めて {@link #expected} に書く。 */
    private void bruteForce(AfterHand after, long river, Observation read, int seatWind) {
      HandShape shape = after.shape();
      if (after.effectiveTileCount() != THIRTEEN || analyzer.shanten(shape) != 1) {
        return;
      }
      searched++;
      context.clear();
      for (int meld = 0; meld < after.meldCount(); meld++) {
        context.addMeld(after.meld(meld));
      }
      context.setWinds(seatWind, read.roundWindTile());
      for (int index = 0; index < read.indicators(); index++) {
        context.addIndicator(read.indicatorTile(index));
      }
      context.setRiichi(RiichiState.NONE, false);
      for (int draw = 0; draw < Tiles.KINDS; draw++) {
        if (held(shape, after, draw) >= Tiles.COPIES || tiles.unseen(draw) == 0) {
          continue;
        }
        drawn.copyFrom(shape);
        drawn.add(draw);
        if (analyzer.shanten(drawn) != 0) {
          continue;
        }
        boolean free = false;
        boolean furiten = false;
        boolean self = false;
        for (int discard = 0; discard < Tiles.KINDS; discard++) {
          if (discard == draw || shape.count(discard) == 0) {
            continue;
          }
          probe.copyFrom(drawn);
          probe.remove(discard);
          if (analyzer.shanten(probe) != 0) {
            continue;
          }
          long waits = 0L;
          boolean ron = false;
          boolean tsumo = false;
          for (int wait = 0; wait < Tiles.KINDS; wait++) {
            if (held(probe, after, wait) >= Tiles.COPIES) {
              continue;
            }
            context.hand().copyFrom(probe);
            context.hand().add(wait);
            boolean complete = analyzer.shanten(context.hand()) == -1;
            if (!complete) {
              continue;
            }
            waits |= Tiles.bit(wait);
            context.hand().copyFrom(probe);
            ron |= yaku(wait, WinContext.Source.RON);
            tsumo |= yaku(wait, WinContext.Source.TSUMO);
          }
          if (ron) {
            if ((waits & (river | Tiles.bit(discard))) != 0L) {
              furiten = true;
            } else {
              free = true;
            }
          }
          self |= tsumo;
        }
        long bit = Tiles.bit(draw);
        if (free) {
          expected[Reach.FREE_RON.ordinal()] |= bit;
        } else if (furiten) {
          expected[Reach.FURITEN_RON.ordinal()] |= bit;
        }
        if (self) {
          expected[Reach.TSUMO.ordinal()] |= bit;
        }
      }
    }

    private boolean yaku(int wait, WinContext.Source source) {
      context.setWin(wait, source);
      return scorer.score(context, score) && score.base() > 0;
    }

    private int live(long kinds) {
      int live = 0;
      for (long rest = kinds; rest != 0L; rest &= rest - 1) {
        live += tiles.unseen(Long.numberOfTrailingZeros(rest));
      }
      return live;
    }

    /** 手の形と面子にある、その種類の牌の枚数。 */
    private static int held(HandShape shape, AfterHand hand, int kind) {
      int held = shape.count(kind);
      for (int index = 0; index < hand.meldCount(); index++) {
        Meld meld = hand.meld(index);
        for (int tile = 0; tile < meld.size(); tile++) {
          held += meld.tile(tile) == kind ? 1 : 0;
        }
      }
      return held;
    }

    String summary(String policy) {
      return String.format(
          Locale.ROOT,
          "FRONTIER %s: decisions %d transitions %d searched %d freeRon %d furitenRon %d tsumo %d sharedSlots %d"
              + " callFrontiers %d",
          policy,
          decisions,
          transitions,
          searched,
          reached[0],
          reached[1],
          reached[2],
          shared,
          callFrontiers);
    }
  }
}
