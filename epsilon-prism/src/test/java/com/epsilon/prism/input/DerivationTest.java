package com.epsilon.prism.input;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.core.Action;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.DoraState;
import com.epsilon.core.GameState;
import com.epsilon.core.HandView;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.Tile;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.HandAnalysisBuffer;
import com.epsilon.engine.WinPointDeltas;
import com.epsilon.engine.WinSettlementProjection;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.Limits;
import com.epsilon.prism.core.SuitSymmetry;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.PassGrade;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.input.InputSchema.WinExact;
import com.epsilon.prism.input.WaitScoring.Stance;
import com.epsilon.prism.input.WaitScoring.WinningTile;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
import com.epsilon.prism.policy.ActionGroups;
import com.epsilon.prism.policy.R0Features;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * Tokenizer の派生と書いた行。
 *
 * <ul>
 *   <li>同じ判断で、見える・見えない枚数、後の手、形（向聴 4 種・受け入れ・形の待ち・生きた受け入れ）、聴牌の待ちの点（赤でない和了牌はエンジンの
 *       {@code analyzeTransitionWaits}、赤の 5 はエンジンの局面の表ドラでの単独の採点）、ドラの数、即時の和了の後の点がエンジンと一致する。
 *   <li>追い越しの格が、格ごとに {@link WinPointDeltas} で点を動かして数えた値と一致する。
 *   <li>p0 の特徴が、Nano の符号化の遷移特徴から読んだ参照（{@link R0Features}）とビットで一致する。
 *   <li>全ての行が書き手の確かめ（語彙・鍵の符号・手の索引）を通る。
 *   <li>牌山の色を入れ替えた半荘の対の判断で、入力がビットで同じ（手と遷移は色の写しで対にする）。色の鍵が物理の色まで同点の判断は除き、数を出す。
 * </ul>
 */
public class DerivationTest {
  private static final Logger log = LoggerFactory.getLogger(DerivationTest.class);

  private static final long GREEDY_SEED = 500;
  private static final long CALLS_SEED = 8_200;
  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final BatchPlan.Capacity ONE_ROW =
      new BatchPlan.Capacity(
          1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS);
  private static final Tensor[] ROW_TENSORS = {
    Tensor.ROUND_CATEGORY,
    Tensor.ROUND_SCALAR,
    Tensor.SEAT_CATEGORY,
    Tensor.SEAT_SCALAR,
    Tensor.TILE_CATEGORY,
    Tensor.EVENT_CATEGORY,
    Tensor.EVENT_SCALAR,
    Tensor.RELATION,
    Tensor.ROW_COLUMN
  };
  private static final Tensor[] TRANSITION_TENSORS = {
    Tensor.TRANSITION_CATEGORY,
    Tensor.TRANSITION_SCALAR,
    Tensor.TRANSITION_CODE,
    Tensor.DISCARD_EXACT
  };

  @Test(timeOut = 1_200_000)
  public void derivationsMatchTheEngine() {
    Derivations greedy = new Derivations();
    new DecisionWalk(DecisionWalk.Policy.GREEDY).run(GREEDY_SEED, 60, greedy);
    Derivations calls = new Derivations();
    new DecisionWalk(DecisionWalk.Policy.CALLS).run(CALLS_SEED, 100, calls);
    log.info(greedy.summary("GREEDY"));
    log.info(calls.summary("CALLS"));
    for (Derivations check : List.of(greedy, calls)) {
      assertTrue(check.decisions > 10_000, "decisions " + check.decisions);
      assertTrue(check.priors > 10_000, "p0 features " + check.priors);
      assertTrue(check.grades > 1_000, "scored pass grades " + check.grades);
    }
    assertTrue(greedy.tenpai > 1_000, "tenpai transitions " + greedy.tenpai);
    assertTrue(greedy.akaWaits > 0, "aka winning tiles " + greedy.akaWaits);
    assertTrue(greedy.riichi > 0, "riichi transitions " + greedy.riichi);
    assertTrue(greedy.winChecks > 0 && calls.winChecks > 0, "immediate wins");
    assertTrue(calls.calls > 1_000, "call transitions " + calls.calls);
    assertTrue(calls.kans > 0, "kan transitions " + calls.kans);
  }

  @Test(timeOut = 1_200_000)
  public void permutedSuitsEncodeTheSameInput() {
    Mirror mirror = new Mirror();
    for (int view = 1; view < SuitSymmetry.VIEWS; view++) {
      mirror.view = view;
      new DecisionWalk(DecisionWalk.Policy.GREEDY)
          .runMirrored(GREEDY_SEED + 100 * view, 6, view, mirror);
      new DecisionWalk(DecisionWalk.Policy.CALLS)
          .runMirrored(CALLS_SEED + 100 * view, 6, view, mirror);
    }
    log.info(mirror.summary());
    assertTrue(mirror.compared > 10_000, "compared decisions " + mirror.compared);
    assertTrue(
        mirror.physicallyDifferent * 2 > mirror.compared,
        "physically different hands " + mirror.physicallyDifferent);
    assertTrue(mirror.tenpai > 100, "compared tenpai transitions " + mirror.tenpai);
    assertTrue(
        mirror.physicalTies * 20 < mirror.decisions,
        "physical ties " + mirror.physicalTies + " of " + mirror.decisions);
  }

  /** 1 判断を捕捉し、1 行の塊に書く。 */
  private static final class Encoder {
    private final ObservationWriter writer = new ObservationWriter();
    private final ObservationArena arena = new ObservationArena(256);
    private final Observation read = new Observation();
    private final Tokenizer tokenizer = new Tokenizer();
    private final InputWriter input = new InputWriter();
    private final HostBatch batch =
        new HostBatch(ONE_ROW, ByteBuffer.allocate(BatchPlan.maxBytes(ONE_ROW)));

    Observation encode(PublicObservation o, EngineDecisionKind kind, List<Action> legal) {
      arena.clear();
      read.bind(arena.words(), writer.capture(o, kind, legal, arena));
      BatchPlan plan = batch.plan();
      plan.clear();
      assertEquals(plan.add(read), 0);
      plan.seal();
      tokenizer.write(read, input.bind(batch, 0));
      return read;
    }

    byte[] bytes(Tensor tensor, int from, int count) {
      int row = tensor.width() * tensor.element().bytes();
      byte[] out = new byte[count * row];
      batch.blob().get(batch.plan().offset(tensor) + from * row, out);
      return out;
    }

    int column(int action, ActionColumn column) {
      int at =
          batch.plan().offset(Tensor.ACTION_COLUMN)
              + (action * Tensor.ACTION_COLUMN.width() + column.ordinal()) * Integer.BYTES;
      return batch.blob().getInt(at);
    }
  }

  /** 派生をエンジンと照らす。 */
  private static final class Derivations implements DecisionWalk.Visitor {
    private final Encoder encoder = new Encoder();
    private final EngineDecisionBuffer engineAnalysis = new EngineDecisionBuffer();
    private final EventTable events = new EventTable();
    private final TileCounts tiles = new TileCounts();
    private final AfterHand current = new AfterHand();
    private final AfterHand called = new AfterHand();
    private final AfterHand after = new AfterHand();
    private final ShapeAnalysis shapes = new ShapeAnalysis();
    private final WaitScoring waits = new WaitScoring();
    private final ScoreLadder ladder = new ScoreLadder();
    private final WinFacts wins = new WinFacts();
    private final HandScoreEvaluator evaluator = new HandScoreEvaluator();
    private final HandScoreBuffer single = new HandScoreBuffer();
    private final DecisionBatchBuilder.EncodingWorkspacePool nano =
        new DecisionBatchBuilder.EncodingWorkspacePool(1);
    private final int[] deltas = new int[SEATS];
    private final int[] scores = new int[SEATS];
    private long decisions;
    private long transitions;
    private long tenpai;
    private long akaWaits;
    private long riichi;
    private long calls;
    private long kans;
    private long grades;
    private long winChecks;
    private long priors;

    @Override
    public void visit(GameEngine engine, EngineDecisionPoint point, int round) {
      int player = point.player();
      PublicObservation o = engine.observation(player);
      List<Action> legal = point.legalActions();
      Observation read = encoder.encode(o, point.kind(), legal);
      engine.analyzeDecision(point, engineAnalysis);
      events.load(read);
      tiles.load(read, events);
      for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
        assertEquals(tiles.visible(tile), engineAnalysis.visibleTileCount(tile), "visible " + tile);
        assertEquals(tiles.unseen(tile), engineAnalysis.unseenCopies(tile), "unseen " + tile);
      }
      current.loadCurrent(read, tiles, events);
      assertSameHand(current, o.hand(player), "current hand");
      shapes.reset();
      int callDiscards = 0;
      for (int action = 0; action < legal.size(); action++) {
        if (read.representative(action) == action) {
          for (int index = 0; index < read.transitionsOf(action); index++) {
            transition(engine, o, read, action, index, callDiscards);
          }
        }
        callDiscards += read.callDiscardsOf(action);
      }
      ladder(read);
      for (int action = 0; action < legal.size(); action++) {
        WinSettlementProjection projection = engineAnalysis.immediateWin(action);
        if (projection != null) {
          win(read, action, projection);
        }
      }
      priors(o, legal);
      decisions++;
    }

    /** 1 つの遷移の後の手・形・待ちの点・ドラをエンジンと照らす。 */
    private void transition(
        GameEngine engine,
        PublicObservation o,
        Observation read,
        int action,
        int index,
        int callDiscards) {
      Action chosen = read.action(action);
      Action.Type type = chosen.type();
      if (type == Action.Type.CHI || type == Action.Type.PON) {
        if (index == 0) {
          called.copyFrom(current);
          called.apply(chosen, read);
        }
        after.copyFrom(called);
        after.discard(read.callDiscard(callDiscards + index));
        calls++;
      } else {
        after.copyFrom(current);
        after.apply(chosen, read);
      }
      kans += chosen.type().isKan() ? 1 : 0;
      transitions++;
      String where = chosen + " #" + index;
      assertSameHand(after, engineAnalysis.handAfterTransition(action, index), where);
      if (type == Action.Type.TSUMO_AGARI
          || type == Action.Type.RON_AGARI
          || type == Action.Type.KYUSHU_KYUHAI) {
        return;
      }
      RiichiState riichiState = transitionRiichi(o, read.observer(), type);
      HandAnalysisBuffer expected = engineAnalysis.analyzeTransitionWaits(action, index, riichiState);
      shapes.analyze(after, tiles, ShapeAnalysis.UNSHARED);
      assertEquals(shapes.minimum(), expected.minimumShanten(), where);
      assertEquals(shapes.standard(), expected.standardShanten(), where);
      assertEquals(shapes.chiitoi(), expected.chiitoitsuShanten(), where);
      assertEquals(shapes.kokushi(), expected.kokushiShanten(), where);
      assertEquals(shapes.special(), expected.specialHandShantenAvailable(), where);
      assertEquals(shapes.ukeire(), expected.improvingTileTypeMask(), where);
      assertEquals(shapes.waits(), expected.shapeWaitTileTypeMask(), where);
      assertEquals(shapes.liveUkeire(), expected.liveImprovingCopies(), where);
      assertEquals(bits(shapes.hitWithin(1)), bits(expected.improvingHitWithinOneDraw()), where);
      assertEquals(bits(shapes.hitWithin(2)), bits(expected.improvingHitWithinTwoDraws()), where);
      assertEquals(bits(shapes.hitWithin(3)), bits(expected.improvingHitWithinThreeDraws()), where);
      assertEquals(after.dora(), expected.doraCount() + Integer.bitCount(after.ownedAkaMask()));
      long waitMask = shapes.waits();
      if (waitMask == 0L) {
        return;
      }
      tenpai++;
      riichi += riichiState == RiichiState.NONE ? 0 : 1;
      int seatWind = SeatFacts.seatWindTile(read, 0);
      waits.score(after, waitMask, tiles, seatWind, read.roundWindTile(), riichiState);
      Stance stance = riichiState == RiichiState.NONE ? Stance.DAMA : Stance.RIICHI;
      assertEquals(waits.waits(), expected.waitCount(), where);
      long ronDama = 0L;
      long tsumoDama = 0L;
      for (int wait = 0; wait < waits.waits(); wait++) {
        int tile = waits.waitTile(wait);
        assertEquals(tile, expected.waitTileType(wait), where);
        long bit = 1L << tile;
        boolean ron = waits.available(wait, stance, WinMethod.RON, WinningTile.PLAIN);
        boolean tsumo = waits.available(wait, stance, WinMethod.TSUMO, WinningTile.PLAIN);
        assertEquals(ron, (expected.ronWaitTileTypeMask() & bit) != 0L, where);
        assertEquals(tsumo, (expected.tsumoWaitTileTypeMask() & bit) != 0L, where);
        if (ron) {
          assertEquals(waits.han(wait, stance, WinMethod.RON, WinningTile.PLAIN), expected.ronHanWithoutUra(wait));
          assertEquals(waits.fu(wait, stance, WinMethod.RON, WinningTile.PLAIN), expected.ronFu(wait));
          assertEquals(
              waits.basePoints(wait, stance, WinMethod.RON, WinningTile.PLAIN),
              expected.ronBasePoints(wait));
        }
        if (tsumo) {
          assertEquals(waits.han(wait, stance, WinMethod.TSUMO, WinningTile.PLAIN), expected.tsumoHanWithoutUra(wait));
          assertEquals(waits.fu(wait, stance, WinMethod.TSUMO, WinningTile.PLAIN), expected.tsumoFu(wait));
          assertEquals(
              waits.basePoints(wait, stance, WinMethod.TSUMO, WinningTile.PLAIN),
              expected.tsumoBasePoints(wait));
        }
        ronDama |= waits.available(wait, Stance.DAMA, WinMethod.RON, WinningTile.PLAIN) ? bit : 0L;
        tsumoDama |=
            waits.available(wait, Stance.DAMA, WinMethod.TSUMO, WinningTile.PLAIN) ? bit : 0L;
        if (tiles.akaUnseen(tile)) {
          akaWait(engine, read, wait, tile, seatWind, riichiState, stance);
        }
      }
      assertEquals(tiles.liveCopies(ronDama), expected.ronWaitCopiesWithoutRiichi(), where);
      assertEquals(tiles.liveCopies(tsumoDama), expected.tsumoWaitCopiesWithoutRiichi(), where);
    }

    /** 赤の 5 の和了牌の点を、エンジンの局面の表ドラで単独に採点した値と照らす。 */
    private void akaWait(
        GameEngine engine,
        Observation read,
        int wait,
        int tile,
        int seatWind,
        RiichiState riichiState,
        Stance stance) {
      DoraState dora = engine.getState().doraState();
      int owned = AkaTileMask.includeTile(after.ownedAkaMask(), tile);
      Stance[] stances = stance == Stance.DAMA ? new Stance[] {Stance.DAMA} : Stance.values();
      for (Stance each : stances) {
        RiichiState state = each == Stance.DAMA ? RiichiState.NONE : riichiState;
        WinConditions conditions = WinConditions.publicDecision(seatWind, read.roundWindTile(), state);
        for (WinMethod method : WinMethod.values()) {
          boolean available =
              evaluator.scoreAfterAdding(after, tile, method, conditions, dora, false, owned, single);
          assertEquals(waits.available(wait, each, method, WinningTile.AKA), available, "aka " + tile);
          if (available) {
            assertEquals(waits.basePoints(wait, each, method, WinningTile.AKA), single.basePoints());
            assertEquals(waits.han(wait, each, method, WinningTile.AKA), single.han());
          }
        }
      }
      akaWaits++;
    }

    /** 追い越しの格を、格ごとに点を動かして数えた値と照らす。 */
    private void ladder(Observation read) {
      ladder.load(read);
      for (int seat = 0; seat < SEATS; seat++) {
        scores[seat] = read.score(seat);
      }
      for (int seat = 1; seat < SEATS; seat++) {
        for (ScoreLadder.Pass pass : ScoreLadder.Pass.values()) {
          PassGrade expected =
              switch (pass) {
                case BY_DIRECT_RON -> grade(read, 0, seat, seat);
                case BY_OTHER_RON -> grade(read, 0, seat, seat == 1 ? 2 : 1);
                case BY_TSUMO -> grade(read, 0, seat, 0);
                case PASSED_BY_RON -> grade(read, seat, 0, 0);
                case PASSED_BY_TSUMO -> grade(read, seat, 0, seat);
              };
          assertEquals(ladder.grade(seat, pass), expected, pass + " seat " + seat);
          grades += expected.scored() ? 1 : 0;
        }
      }
    }

    /** winner が from から（from == winner ならツモで）和了して loser の上に出る最小の格。 */
    private PassGrade grade(Observation read, int winner, int loser, int from) {
      int winnerSeat = read.absoluteSeat(winner);
      int loserSeat = read.absoluteSeat(loser);
      if (above(scores[winner], winnerSeat, scores[loser], loserSeat)) {
        return PassGrade.NOT_NEEDED;
      }
      for (PassGrade grade : PassGrade.values()) {
        if (!grade.scored()) {
          continue;
        }
        WinPointDeltas.settleInto(
            grade.basePoints(),
            winner,
            from,
            read.dealer(),
            read.honba(),
            read.kyotaku(),
            deltas);
        if (above(
            scores[winner] + deltas[winner],
            winnerSeat,
            scores[loser] + deltas[loser],
            loserSeat)) {
          return grade;
        }
      }
      return PassGrade.UNREACHABLE;
    }

    /** 即時の和了の後の点（本場・供託込み）と順位をエンジンの射影と照らす。 */
    private void win(Observation read, int action, WinSettlementProjection projection) {
      wins.load(read, read.winOf(action));
      int observer = read.observer();
      int kyotaku = read.kyotaku() * HanchanProgression.RIICHI_COST;
      for (int seat = 1; seat < SEATS; seat++) {
        assertEquals(
            wins.scoreAfter(0) + kyotaku - wins.scoreAfter(seat),
            projection.scoreGap(read.absoluteSeat(seat)),
            "score gap");
      }
      int rank = projection.projectedRank(observer);
      assertEquals(wins.value(WinExact.POINTS), (float) (wins.scoreAfter(0) + kyotaku - read.score(0)));
      assertEquals(wins.value(WinExact.RANK_1), rank == 0 ? 1f : 0f);
      assertEquals(wins.value(WinExact.RANK_4), rank == 3 ? 1f : 0f);
      winChecks++;
    }

    /** p0 の特徴を、Nano の符号化の遷移特徴から読んだ値（{@link R0Features}）とビットで照らす。 */
    private void priors(PublicObservation o, List<Action> legal) {
      DecisionBatchBuilder builder =
          DecisionBatchBuilder.inference(1, DecisionBatchBuilder.selectInferenceBucket(legal), nano);
      try (DecisionBatchBuilder.EncodingSession session = builder.openEncoding()) {
        session.encodeObservedInferenceRow(0, o, legal);
      }
      DecisionHostBatch host = builder.buildEncodedInferenceRows();
      for (int action = 0; action < legal.size(); action++) {
        if (ActionGroups.of(legal.get(action).type()) != ActionGroups.DISCARD) {
          continue;
        }
        assertEquals(
            Float.floatToIntBits(encoder.batch.priorShanten(action)),
            Float.floatToIntBits(R0Features.shanten(host, 0, action)),
            "p0 shanten " + legal.get(action));
        assertEquals(
            Float.floatToIntBits(encoder.batch.priorUkeire(action)),
            Float.floatToIntBits(R0Features.ukeire(host, 0, action)),
            "p0 ukeire " + legal.get(action));
        priors++;
      }
    }

    String summary(String policy) {
      return String.format(
          Locale.ROOT,
          "INPUT-T3 %s: decisions %d transitions %d calls %d kans %d tenpai %d riichi %d"
              + " akaWaits %d scoredGrades %d wins %d p0Features %d",
          policy,
          decisions,
          transitions,
          calls,
          kans,
          tenpai,
          riichi,
          akaWaits,
          grades,
          winChecks,
          priors);
    }
  }

  /** 色を入れ替えた半荘の対の判断で、書いた行をビットで照らす。 */
  private static final class Mirror implements DecisionWalk.MirrorVisitor {
    private final Encoder left = new Encoder();
    private final Encoder right = new Encoder();
    private final EventTable events = new EventTable();
    private final TileCounts tiles = new TileCounts();
    private final SuitCanonicalizer suits = new SuitCanonicalizer();
    private int view;
    private long decisions;
    private long compared;
    private long physicalTies;
    private long physicallyDifferent;
    private long tenpai;

    @Override
    public void visit(
        GameEngine engine,
        EngineDecisionPoint point,
        GameEngine mirror,
        EngineDecisionPoint mirrorPoint) {
      List<Action> legal = point.legalActions();
      List<Action> mirrorLegal = mirrorPoint.legalActions();
      Observation a = left.encode(engine.observation(point.player()), point.kind(), legal);
      Observation b =
          right.encode(mirror.observation(mirrorPoint.player()), mirrorPoint.kind(), mirrorLegal);
      decisions++;
      boolean tie = physicalTie(a);
      assertEquals(physicalTie(b), tie, "physical ties of the pair");
      if (tie) {
        physicalTies++;
        return;
      }
      // 照らしが空でないこと: 物理の牌で見れば、対の手はたいてい違う。
      for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
        if (a.handCount(tile) != b.handCount(tile)) {
          physicallyDifferent++;
          break;
        }
      }
      String where = "view " + view + " decision " + decisions + " " + point.kind();
      for (Tensor tensor : ROW_TENSORS) {
        int count = left.batch.plan().count(tensor.axis());
        assertEquals(count, right.batch.plan().count(tensor.axis()), where);
        assertTrue(
            Arrays.equals(left.bytes(tensor, 0, count), right.bytes(tensor, 0, count)),
            tensor + " differs at " + where);
      }
      int[] firstCall = new int[legal.size()];
      int[] mirrorFirstCall = new int[legal.size()];
      for (int action = 1; action < legal.size(); action++) {
        firstCall[action] = firstCall[action - 1] + a.callDiscardsOf(action - 1);
        mirrorFirstCall[action] = mirrorFirstCall[action - 1] + b.callDiscardsOf(action - 1);
      }
      for (int action = 0; action < legal.size(); action++) {
        int mapped = SuitSymmetry.find(view, mirrorLegal, legal.get(action));
        assertTrue(mapped >= 0, where);
        assertTrue(
            Arrays.equals(
                left.bytes(Tensor.ACTION_EXACT, action, 1),
                right.bytes(Tensor.ACTION_EXACT, mapped, 1)),
            "action exact differs at " + where + " " + legal.get(action));
        assertEquals(
            left.column(action, ActionColumn.ROLE), right.column(mapped, ActionColumn.ROLE));
        assertEquals(
            left.column(action, ActionColumn.TRANSITIONS),
            right.column(mapped, ActionColumn.TRANSITIONS));
        assertMapped(legal, mirrorLegal, left.column(action, ActionColumn.BASE), right.column(mapped, ActionColumn.BASE));
        assertMapped(
            legal,
            mirrorLegal,
            left.column(action, ActionColumn.REPRESENTATIVE),
            right.column(mapped, ActionColumn.REPRESENTATIVE));
        assertEquals(left.batch.priorShanten(action), right.batch.priorShanten(mapped), where);
        assertEquals(left.batch.priorUkeire(action), right.batch.priorUkeire(mapped), where);
        int transitions = left.column(action, ActionColumn.TRANSITIONS);
        int first = left.column(action, ActionColumn.FIRST_TRANSITION);
        int mirrorFirst = right.column(mapped, ActionColumn.FIRST_TRANSITION);
        for (int index = 0; index < transitions; index++) {
          int mirrorIndex = index;
          if (a.callDiscardsOf(action) > 0) {
            Action discard = a.callDiscard(firstCall[action] + index);
            mirrorIndex = -1;
            for (int other = 0; other < transitions; other++) {
              if (SuitSymmetry.mapped(view, discard, b.callDiscard(mirrorFirstCall[mapped] + other))) {
                mirrorIndex = other;
              }
            }
            assertTrue(mirrorIndex >= 0, where);
          }
          for (Tensor tensor : TRANSITION_TENSORS) {
            assertTrue(
                Arrays.equals(
                    left.bytes(tensor, first + index, 1),
                    right.bytes(tensor, mirrorFirst + mirrorIndex, 1)),
                tensor + " differs at " + where + " " + legal.get(action) + " #" + index);
          }
          byte[] exact = left.bytes(Tensor.DISCARD_EXACT, first + index, 1);
          tenpai += exact[0] != 0 || exact[1] != 0 || exact[2] != 0 || exact[3] != 0 ? 1 : 0;
        }
      }
      compared++;
    }

    /** 手の番号の列（なしは −1）が、色の写しで対の手を指す。 */
    private void assertMapped(List<Action> legal, List<Action> mirrorLegal, int index, int mirrorIndex) {
      if (index < 0) {
        assertEquals(mirrorIndex, -1);
        return;
      }
      assertTrue(SuitSymmetry.mapped(view, legal.get(index), mirrorLegal.get(mirrorIndex)));
    }

    private boolean physicalTie(Observation o) {
      events.load(o);
      tiles.load(o, events);
      suits.load(o, events, tiles);
      return suits.physicalTie();
    }

    String summary() {
      return String.format(
          Locale.ROOT,
          "INPUT-T3 mirrored: decisions %d compared %d physicalTies %d physicallyDifferentHands %d"
              + " tenpaiTransitions %d",
          decisions,
          compared,
          physicalTies,
          physicallyDifferent,
          tenpai);
    }
  }

  private static void assertSameHand(HandView mine, HandView engine, String where) {
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      assertEquals(mine.count(tile), engine.count(tile), where + " tile " + tile);
    }
    assertEquals(mine.meldCount(), engine.meldCount(), where);
    for (int meld = 0; meld < mine.meldCount(); meld++) {
      assertEquals(mine.meld(meld), engine.meld(meld), where);
    }
    assertEquals(mine.concealedAkaMask(), engine.concealedAkaMask(), where);
    assertEquals(mine.ownedAkaMask(), engine.ownedAkaMask(), where);
    assertEquals(mine.isMenzen(), engine.isMenzen(), where);
    assertEquals(mine.kanCount(), engine.kanCount(), where);
  }

  /** 遷移の立直の状態（Nano と同じ決め方を公開の観測から）。 */
  private static RiichiState transitionRiichi(PublicObservation o, int player, Action.Type type) {
    if (type == Action.Type.RIICHI_DAHAI) {
      return o.isBeforeFirstDiscard(player) && !o.firstTurnCallOccurred()
          ? RiichiState.DOUBLE_RIICHI
          : RiichiState.RIICHI;
    }
    if (o.isDoubleRiichi(player)) {
      return RiichiState.DOUBLE_RIICHI;
    }
    return o.isRiichi(player) ? RiichiState.RIICHI : RiichiState.NONE;
  }

  private static boolean above(int scoreA, int seatA, int scoreB, int seatB) {
    return scoreA > scoreB || scoreA == scoreB && seatA < seatB;
  }

  private static int bits(float value) {
    return Float.floatToIntBits(value);
  }
}
