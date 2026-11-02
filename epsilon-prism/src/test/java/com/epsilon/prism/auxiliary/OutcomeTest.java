package com.epsilon.prism.auxiliary;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.calculate.shape.HandShapeAnalysisBuffer;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.RoundResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.RoundTransition;
import com.epsilon.engine.ScorePayments;
import com.epsilon.engine.WinClaim;
import com.epsilon.engine.WinLegality;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 補助の頭の教師の記録（局末の 3 語と神託）。
 *
 * <p>固定の種の半荘を 3 つの打ち方で打つ（GREEDY は槍槓・複数ロン、CALLS は四槓散了・九種九牌、TERMINALS は流し満貫を含む）。主経路の全ての
 * 判断を枠とし、局末に {@link HandEndAnalyzer} と {@link OutcomeCodec} で書いた 3 語を解いて、精算から手で数えた値と照らす。
 *
 * <ul>
 *   <li>局の終わり方・和了した他家・役のビット・次局・局末の点。
 *   <li>打点は WinPayment.total から本場の 300 × 本場を引いた値、失点は放銃した和了ごとの同じ値の和（包の半額は点の移動のまま）。
 *   <li>局末の聴牌と役のある聴牌は、向聴の解析と採点（役の有無は採点の結果）の別の道で数え、荒牌流局ではエンジンの tenpaiMask と一致する。
 *   <li>聴牌への到達は、打牌ごとに別の道で数えた聴牌の出来事が判断の時点の出来事の数以上か、和了か局末の聴牌。
 * </ul>
 *
 * <p>神託は全ての判断で、相手 3 席の 34 牌の全てについて放銃が {@link WinLegality#canRon} と一致し、向聴・待ち・放銃の点（採点の基本点から
 * ScorePayments で数えたロンの点）が一致する。
 */
public class OutcomeTest {
  private static final Logger log = LoggerFactory.getLogger(OutcomeTest.class);

  @Test(timeOut = 600_000)
  public void outcomesAndOracleMatchHandCountedValues() {
    Check check = new Check();
    new DecisionWalk(DecisionWalk.Policy.GREEDY).run(500, 30, check);
    new DecisionWalk(DecisionWalk.Policy.CALLS).run(8_270, 30, check);
    new DecisionWalk(DecisionWalk.Policy.TERMINALS).run(20_000, 10, check);
    log.info(check.summary());

    assertTrue(check.decisionCount > 10_000, "decisions " + check.decisionCount);
    for (Outcome outcome : Outcome.values()) {
      assertTrue(check.outcomes[outcome.ordinal()] > 0, "no decision ends with " + outcome);
    }
    assertTrue(check.multiRonSelf > 0, "no own ron among several winners");
    assertTrue(check.multiRonDealIn > 0, "no deal-in to several winners");
    assertTrue(check.chankanDealIn > 0, "no deal-in by a robbed kan");
    assertTrue(check.nagashi > 0, "no nagashi mangan");
    assertTrue(check.aborts.size() >= 2, "abortive draws " + check.aborts);
    assertTrue(check.honbaDealIn > 0, "no deal-in with honba");
    assertTrue(check.reachedByDiscard > 0, "no tenpai reached only at a discard");
    assertTrue(check.tenpaiWithoutYaku > 0, "no tenpai reached without a ron yaku");
    assertTrue(check.dangers > 0, "no danger tile");
  }

  /** 判断の席と、判断の時点の出来事の数（記録 O の頭）。 */
  private record Decision(int seat, int events) {}

  /** 全ての判断と局末を照らし、数を数える。 */
  private static final class Check implements DecisionWalk.Visitor {
    private final ObservationWriter writer = new ObservationWriter();
    private final ObservationArena arena = new ObservationArena(256);
    private final Observation observation = new Observation();
    private final OracleCapture oracles = new OracleCapture();
    private final HandEndAnalyzer analyzer = new HandEndAnalyzer();
    private final OutcomeCodec codec = new OutcomeCodec();
    private final TenpaiReach reach = new TenpaiReach();
    private final HandCount count = new HandCount();
    private final long[] words = new long[OutcomeCodec.WORDS];
    private final List<Decision> decisions = new ArrayList<>();
    private final int[] riverSizes = new int[GameState.NUM_PLAYERS];
    private final List<List<Integer>> tenpaiEvents = seatLists();
    private final List<List<Integer>> yakuTenpaiEvents = seatLists();
    private final int[] outcomes = new int[Outcome.values().length];
    private final Set<RoundResult.AbortiveDrawReason> aborts =
        EnumSet.noneOf(RoundResult.AbortiveDrawReason.class);
    private boolean chankanStage;
    private long decisionCount;
    private int multiRonSelf;
    private int multiRonDealIn;
    private int chankanDealIn;
    private int nagashi;
    private int honbaDealIn;
    private int reachedByDiscard;
    private int tenpaiWithoutYaku;
    private long dangers;
    private int rounds;

    @Override
    public void awaiting(GameEngine engine, List<EngineDecisionPoint> points, int round) {
      discards(engine.getState());
      chankanStage = points.get(0).kind() == EngineDecisionKind.CHANKAN;
    }

    @Override
    public void visit(GameEngine engine, EngineDecisionPoint point, int round) {
      GameState state = engine.getState();
      int seat = point.player();
      arena.clear();
      int base = writer.capture(engine.observation(seat), point.kind(), point.legalActions(), arena);
      decisions.add(new Decision(seat, observation.bind(arena.words(), base).events()));
      checkOracle(state, seat, oracles.capture(state, seat));
      decisionCount++;
    }

    @Override
    public void settled(GameEngine engine, RoundSettlement settlement, int round) {
      GameState state = engine.getState();
      discards(state);
      HandEndFacts facts = analyzer.analyze(state, settlement, reach);
      RoundResult result = settlement.result();
      int winners = 0;
      if (result instanceof RoundResult.Winning winning) {
        for (WinClaim claim : winning.claims()) {
          winners |= 1 << claim.winner();
        }
      }
      int tenpai = winners;
      int yakuTenpai = winners;
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        if ((winners & 1 << seat) == 0 && count.tenpai(state.hand(seat))) {
          tenpai |= 1 << seat;
          yakuTenpai |= count.yakuTenpai(state, seat) ? 1 << seat : 0;
        }
      }
      assertEquals(facts.tenpaiMask(), tenpai, "end tenpai in round " + round);
      assertEquals(facts.yakuTenpaiMask(), yakuTenpai, "end yaku tenpai in round " + round);
      switch (result) {
        case RoundResult.ExhaustiveDraw draw -> {
          assertEquals(facts.tenpaiMask(), draw.tenpaiMask(), "engine tenpai in round " + round);
          nagashi += draw.hasNagashiMangan() && !decisions.isEmpty() ? 1 : 0;
        }
        case RoundResult.AbortiveDraw abort -> aborts.add(abort.reason());
        default -> {}
      }
      for (Decision decision : decisions) {
        checkOutcome(settlement, facts, decision, tenpai, yakuTenpai);
      }
      rounds++;
      decisions.clear();
      reach.reset();
      Arrays.fill(riverSizes, 0);
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        tenpaiEvents.get(seat).clear();
        yakuTenpaiEvents.get(seat).clear();
      }
    }

    /** 前に見てから増えた打牌ごとに、到達の追跡と別の道の数えを進める。 */
    private void discards(GameState state) {
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        while (riverSizes[seat] < state.river(seat).size()) {
          int event = state.river(seat).event(riverSizes[seat]++);
          reach.discarded(state, seat);
          if (count.tenpai(state.hand(seat))) {
            tenpaiEvents.get(seat).add(event);
            if (count.yakuTenpai(state, seat)) {
              yakuTenpaiEvents.get(seat).add(event);
            }
          }
        }
      }
    }

    private void checkOutcome(
        RoundSettlement settlement,
        HandEndFacts facts,
        Decision decision,
        int tenpai,
        int yakuTenpai) {
      int seat = decision.seat();
      codec.write(facts, seat, decision.events(), words, 0);
      RoundResult result = settlement.result();
      Outcome outcome = OutcomeCodec.outcome(words, 0);
      String where = "seat " + seat + " events " + decision.events() + " " + result;
      int winner = -1;
      int winPoints = 0;
      int dealInPoints = 0;
      long yaku = 0L;
      switch (result) {
        case RoundResult.TsumoAgari tsumo -> {
          WinClaim.Tsumo claim = tsumo.claim();
          yaku = claim.score().yakuBits();
          if (claim.winner() == seat) {
            assertEquals(outcome, Outcome.SELF_TSUMO, where);
            winPoints = claim.payment().total() - 300 * claim.honba();
          } else {
            assertEquals(outcome, Outcome.OTHER_TSUMO, where);
            winner = relative(seat, claim.winner());
          }
        }
        case RoundResult.RonAgari ron -> {
          List<WinClaim.Ron> claims = ron.claims();
          WinClaim.Ron own = null;
          for (WinClaim.Ron claim : claims) {
            own = claim.winner() == seat ? claim : own;
          }
          if (own != null) {
            assertEquals(outcome, Outcome.SELF_RON, where);
            yaku = own.score().yakuBits();
            winPoints = own.payment().total() - 300 * own.honba();
            multiRonSelf += claims.size() > 1 ? 1 : 0;
          } else {
            yaku = claims.get(0).score().yakuBits();
            winner = relative(seat, claims.get(0).winner());
            if (ron.loser() == seat) {
              assertEquals(outcome, Outcome.DEAL_IN, where);
              for (WinClaim.Ron claim : claims) {
                int paid = -claim.pointDelta().get(seat);
                dealInPoints +=
                    paid == claim.payment().total()
                        ? claim.payment().total() - 300 * claim.honba()
                        : paid;
                honbaDealIn += claim.honba() > 0 ? 1 : 0;
              }
              multiRonDealIn += claims.size() > 1 ? 1 : 0;
              chankanDealIn += chankanStage ? 1 : 0;
            } else {
              assertEquals(outcome, Outcome.OTHER_RON, where);
            }
          }
        }
        case RoundResult.ExhaustiveDraw draw ->
            assertEquals(
                outcome, draw.isTenpai(seat) ? Outcome.DRAW_TENPAI : Outcome.DRAW_NOTEN, where);
        case RoundResult.AbortiveDraw ignored -> assertEquals(outcome, Outcome.ABORTIVE, where);
      }
      outcomes[outcome.ordinal()]++;
      assertEquals(OutcomeCodec.winner(words, 0), winner, where);
      assertEquals(OutcomeCodec.winPoints(words, 0), winPoints, where);
      assertEquals(OutcomeCodec.dealInPoints(words, 0), dealInPoints, where);
      assertEquals(OutcomeCodec.yakuBits(words, 0), yaku, where);

      boolean endTenpai = (tenpai & 1 << seat) != 0;
      boolean reachedTenpai = endTenpai || reaches(tenpaiEvents.get(seat), decision.events());
      boolean reachedYaku =
          (yakuTenpai & 1 << seat) != 0 || reaches(yakuTenpaiEvents.get(seat), decision.events());
      assertEquals(OutcomeCodec.tenpai(words, 0), reachedTenpai, "tenpai reach " + where);
      assertEquals(OutcomeCodec.yakuTenpai(words, 0), reachedYaku, "yaku reach " + where);
      reachedByDiscard += reachedTenpai && !endTenpai ? 1 : 0;
      tenpaiWithoutYaku += reachedTenpai && !reachedYaku ? 1 : 0;

      for (int opponent = 1; opponent < GameState.NUM_PLAYERS; opponent++) {
        int player = (seat + opponent) % GameState.NUM_PLAYERS;
        assertEquals(
            OutcomeCodec.opponentTenpai(words, 0, opponent), (tenpai & 1 << player) != 0, where);
      }
      switch (settlement.transition()) {
        case RoundTransition.NextRound next -> {
          assertEquals(OutcomeCodec.nextKyoku(words, 0), next.kyokuIndex(), where);
          assertEquals(OutcomeCodec.nextHonba(words, 0), next.honba(), where);
          assertEquals(OutcomeCodec.nextKyotaku(words, 0), next.kyotakuCount(), where);
          assertTrue(!OutcomeCodec.hanchanEnded(words, 0), where);
        }
        case RoundTransition.HanchanFinished ignored -> {
          assertTrue(OutcomeCodec.hanchanEnded(words, 0), where);
          assertEquals(OutcomeCodec.nextHonba(words, 0), 0, where);
          assertEquals(OutcomeCodec.nextKyotaku(words, 0), 0, where);
        }
      }
      for (int relative = 0; relative < GameState.NUM_PLAYERS; relative++) {
        int player = (seat + relative) % GameState.NUM_PLAYERS;
        assertEquals(
            OutcomeCodec.finalScore(words, 0, relative), settlement.finalScore(player), where);
      }
    }

    /** 神託を、canRon・向聴の解析・採点と照らす。 */
    private void checkOracle(GameState state, int seat, long[] oracle) {
      assertEquals(OracleCodec.longs(oracle, 0), oracle.length);
      int index = 0;
      for (int opponent = 1; opponent < GameState.NUM_PLAYERS; opponent++) {
        int player = (seat + opponent) % GameState.NUM_PLAYERS;
        Hand hand = state.hand(player);
        HandShapeAnalysisBuffer shape = count.analyze(hand);
        assertEquals(OracleCodec.shanten(oracle, 0, opponent), shape.minimumShanten());
        long waits = shape.minimumShanten() == 0 ? shape.agariTileTypeMask() : 0L;
        assertEquals(OracleCodec.waits(oracle, 0, opponent), waits);
        long danger = OracleCodec.danger(oracle, 0, opponent);
        for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
          TurnEvent.Discard discard = new TurnEvent.Discard(seat, tile, false);
          boolean ron = WinLegality.canRon(state, player, discard, count.scores, count.shapes);
          assertEquals((danger >>> tile & 1L) != 0, ron, "danger " + Tile.name(tile));
          if (ron) {
            dangers++;
            int points = count.ronPoints(state, player, discard);
            assertEquals(OracleCodec.points(oracle, 0, index++), points, "danger points");
          }
        }
      }
    }

    private static boolean reaches(List<Integer> events, int decisionEvents) {
      for (int event : events) {
        if (event >= decisionEvents) {
          return true;
        }
      }
      return false;
    }

    private static int relative(int seat, int player) {
      return Math.floorMod(player - seat, GameState.NUM_PLAYERS);
    }

    private static List<List<Integer>> seatLists() {
      List<List<Integer>> lists = new ArrayList<>();
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        lists.add(new ArrayList<>());
      }
      return lists;
    }

    String summary() {
      StringBuilder out = new StringBuilder("OUTCOME-T8 fingerprints outcome=");
      out.append(OutcomeCodec.fingerprint()).append(" oracle=").append(OracleCodec.fingerprint());
      out.append(" decisions=").append(decisionCount);
      out.append(" rounds=").append(rounds);
      for (Outcome outcome : Outcome.values()) {
        out.append(' ').append(outcome).append('=').append(outcomes[outcome.ordinal()]);
      }
      out.append(" multiRonSelf=").append(multiRonSelf);
      out.append(" multiRonDealIn=").append(multiRonDealIn);
      out.append(" chankanDealIn=").append(chankanDealIn);
      out.append(" nagashi=").append(nagashi);
      out.append(" aborts=").append(aborts);
      out.append(" honbaDealIn=").append(honbaDealIn);
      out.append(" reachedByDiscard=").append(reachedByDiscard);
      out.append(" tenpaiWithoutYaku=").append(tenpaiWithoutYaku);
      out.append(" dangers=").append(dangers);
      return out.toString();
    }
  }

  /** 聴牌・役のある聴牌・ロンの点を、記録の道とは別の口（向聴の解析と採点）で数える。 */
  private static final class HandCount {
    private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();
    private final HandScoreEvaluator scores = new HandScoreEvaluator();
    private final HandShapeAnalysisBuffer analysis = new HandShapeAnalysisBuffer();
    private final HandScoreBuffer score = new HandScoreBuffer();

    HandShapeAnalysisBuffer analyze(Hand hand) {
      shapes.analyzeInto(hand, analysis);
      return analysis;
    }

    boolean tenpai(Hand hand) {
      return analyze(hand).minimumShanten() <= 0;
    }

    /** 13 枚はロンで採点できる待ちがある聴牌、14 枚は 1 枚捨ててそうなれる。立直の宣言は採点の条件に入れる。 */
    boolean yakuTenpai(GameState state, int seat) {
      Hand hand = state.hand(seat);
      boolean riichi = state.river(seat).riichiDeclarationIndex() >= 0;
      WinConditions conditions =
          WinConditions.publicDecision(
              state.getJikaze(seat),
              state.getBakaze(),
              riichi ? RiichiState.RIICHI : RiichiState.NONE);
      if (hand.effectiveTileCount() == 13) {
        return ronScorable(hand, conditions);
      }
      for (long tiles = hand.concealedTileTypeMask(); tiles != 0L; tiles &= tiles - 1) {
        Hand discarded = hand.snapshotCopy();
        discarded.remove(Long.numberOfTrailingZeros(tiles));
        if (ronScorable(discarded, conditions)) {
          return true;
        }
      }
      return false;
    }

    private boolean ronScorable(Hand hand, WinConditions conditions) {
      if (analyze(hand).minimumShanten() != 0) {
        return false;
      }
      for (long waits = analysis.agariTileTypeMask(); waits != 0L; waits &= waits - 1) {
        int wait = Long.numberOfTrailingZeros(waits);
        if (scores.scoreAfterAdding(
            hand, wait, WinMethod.RON, conditions, null, false, hand.ownedAkaMask(), score)) {
          return true;
        }
      }
      return false;
    }

    /** そのロンの和了の点（裏なし。基本点から ScorePayments で数える）。 */
    int ronPoints(GameState state, int player, TurnEvent.Discard discard) {
      Hand hand = state.hand(player);
      assertTrue(
          scores.scoreAfterAdding(
              hand,
              discard.tileType(),
              WinMethod.RON,
              WinConditions.fromGameState(state, player, discard),
              state.doraState(),
              false,
              hand.ownedAkaMask(),
              score));
      return ScorePayments.ronPoints(score.basePoints(), player == state.getOya());
    }
  }
}
