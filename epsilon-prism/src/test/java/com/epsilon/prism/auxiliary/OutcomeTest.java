package com.epsilon.prism.auxiliary;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EventKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.HandEndTracker;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 補助のヘッドに渡す教師の記録（局末の結果を詰めた 3 ワードと、Oracle の記録）のテスト。
 *
 * <p>乱数の種を固定した半荘を 3 通りの打ち方で打ち、全ての判断する席で確かめる。局末には、{@link HandEndAnalyzer} と
 * {@link OutcomeCodec} で書いた 3 ワードを読み戻し、エンジンの局末・{@link Settlement}・{@link ScoreMath} から別に求めた値と照合する。
 * 局末の聴牌は形の解析で数え、荒牌ではエンジンが出す聴牌の席と一致することも確かめる。Oracle の記録は全ての判断で、相手ごとに 34 種の牌を
 * {@link FullView#ronScore} と形の解析で照合する。
 *
 * <p>包・ダブロン・流し満貫・途中流局は、手で作った局末の事実で確かめる。立直の宣言牌でのロンと、14 枚で宣言した九種九牌は、手で作った
 * 牌山の局で確かめる。
 */
public class OutcomeTest {
  private static final Logger log = LoggerFactory.getLogger(OutcomeTest.class);

  private static final Variant VARIANT = TENHOU_4P;
  private static final int SEATS = VARIANT.seats();
  private static final int PLAIN_COPY = Tiles.RED_COPY + 1;
  private static final int MAX_POINTS = 576_000;

  @Test(timeOut = 600_000)
  public void outcomesAndOracleMatchIndependentCounts() {
    Check check = new Check();
    new TableWalk(VARIANT, TableWalk.Policy.GREEDY).run(500, 30, check);
    new TableWalk(VARIANT, TableWalk.Policy.CALLS).run(8_270, 30, check);
    new TableWalk(VARIANT, TableWalk.Policy.TERMINALS).run(20_000, 10, check);
    log.info(check.summary());

    assertTrue(check.decisionCount > 10_000, "decisions " + check.decisionCount);
    for (Outcome outcome : Outcome.values()) {
      assertTrue(check.outcomes[outcome.ordinal()] > 0, "no decision ends with " + outcome);
    }
    assertTrue(check.multiRonSelf > 0, "no own ron among several winners");
    assertTrue(check.multiRonDealIn > 0, "no deal-in to several winners");
    assertTrue(check.chankanDealIn > 0, "no deal-in by a robbed kan");
    assertTrue(check.aborts.size() >= 2, "abortive draws " + check.aborts);
    assertTrue(check.honbaDealIn > 0, "no deal-in with honba");
    assertTrue(check.reachedByDiscard > 0, "no tenpai reached only at a discard");
    assertTrue(check.tenpaiWithoutYaku > 0, "no tenpai reached without a ron yaku");
    assertTrue(check.dangers > 0, "no danger tile");
  }

  /** 包のあるロンでは、放銃した席が払う点は半額になり、包が無ければ全額になる。和了した席の打点は、どちらでも全額。 */
  @Test
  public void paoHalvesTheDealIn() {
    OutcomeCodec codec = OutcomeCodec.of(VARIANT);
    long[] words = new long[OutcomeCodec.WORDS];
    int base = ScoreMath.base(0, 0, 1);
    for (int pao : new int[] {RoundOutcome.NO_SEAT, 2}) {
      RoundOutcome outcome = new RoundOutcome(VARIANT);
      outcome.setRon(base, 1, 0, pao, 0);
      HandEndFacts facts = facts(outcome, new long[] {1L}, 3, 0b0010);
      int full = ScoreMath.ron(base, false);
      int[] delta = new int[SEATS];
      Settlement.ron(VARIANT, base, 1, 0, 3, 0, pao, delta);

      codec.write(facts, 0, 0, words, 0);
      assertEquals(codec.outcome(words, 0), Outcome.DEAL_IN);
      assertEquals(codec.dealInPoints(words, 0), pao == RoundOutcome.NO_SEAT ? full : full / 2);
      assertEquals(codec.dealInPoints(words, 0), -delta[0]);
      assertEquals(codec.winner(words, 0), 1);
      codec.write(facts, 1, 0, words, 0);
      assertEquals(codec.outcome(words, 0), Outcome.SELF_RON);
      assertEquals(codec.winPoints(words, 0), full);
      codec.write(facts, 2, 0, words, 0);
      assertEquals(codec.outcome(words, 0), Outcome.OTHER_RON);
      assertEquals(codec.dealInPoints(words, 0), 0);
    }
  }

  /** ダブロンでは、他家から見た和了者と役は最初の和了者のものになる。放銃した席の点は 2 件の合計で、和了した席にはそれぞれ自分の和了を書く。 */
  @Test
  public void doubleRonUsesTheFirstWinner() {
    OutcomeCodec codec = OutcomeCodec.of(VARIANT);
    long[] words = new long[OutcomeCodec.WORDS];
    int first = ScoreMath.base(3, 40, 0);
    int second = ScoreMath.base(1, 30, 0);
    RoundOutcome outcome = new RoundOutcome(VARIANT);
    outcome.setRon(first, 2, 1, RoundOutcome.NO_SEAT, 0);
    outcome.addRonWinner(second, 3, RoundOutcome.NO_SEAT);
    HandEndFacts facts = facts(outcome, new long[] {0b101L, 0b10L}, 3, 0b1100);

    codec.write(facts, 0, 0, words, 0);
    assertEquals(codec.outcome(words, 0), Outcome.OTHER_RON);
    assertEquals(codec.winner(words, 0), 2);
    assertEquals(codec.yakuBits(words, 0), 0b101L);
    codec.write(facts, 1, 0, words, 0);
    assertEquals(codec.outcome(words, 0), Outcome.DEAL_IN);
    assertEquals(codec.winner(words, 0), 1);
    assertEquals(codec.dealInPoints(words, 0), ScoreMath.ron(first, false) + ScoreMath.ron(second, true));
    codec.write(facts, 3, 0, words, 0);
    assertEquals(codec.outcome(words, 0), Outcome.SELF_RON);
    assertEquals(codec.winPoints(words, 0), ScoreMath.ron(second, true));
    assertEquals(codec.yakuBits(words, 0), 0b10L);
  }

  /** 流し満貫は荒牌として扱う。和了の打点は 0 で、点の動きは局末の持ち点にだけ表れる。途中流局には、和了者も点の動きも無い。 */
  @Test
  public void nagashiAndAbortiveDraws() {
    OutcomeCodec codec = OutcomeCodec.of(VARIANT);
    long[] words = new long[OutcomeCodec.WORDS];
    RoundOutcome nagashi = new RoundOutcome(VARIANT);
    nagashi.setExhaustive(0b0100, 0b0100, 0);
    HandEndFacts facts = facts(nagashi, new long[0], 0, 0b0100);
    codec.write(facts, 2, 0, words, 0);
    assertEquals(codec.outcome(words, 0), Outcome.DRAW_TENPAI);
    assertEquals(codec.winPoints(words, 0), 0);
    assertEquals(codec.finalScore(words, 0, 0), facts.next().score(2));
    codec.write(facts, 0, 0, words, 0);
    assertEquals(codec.outcome(words, 0), Outcome.DRAW_NOTEN);

    RoundOutcome abort = new RoundOutcome(VARIANT);
    abort.setAbortive(AbortiveDraw.FOUR_WINDS, 0);
    codec.write(facts(abort, new long[0], 0, 0), 1, 0, words, 0);
    assertEquals(codec.outcome(words, 0), Outcome.ABORTIVE);
    assertEquals(codec.winner(words, 0), -1);
    assertEquals(codec.yakuBits(words, 0), 0L);
    assertEquals(codec.dealInPoints(words, 0), 0);
  }

  /**
   * 親の立直の宣言牌を南家がロンする局。立直は成立しないが、宣言した席は役のある聴牌と数え、聴牌に届いたイベントは宣言牌の打牌とする。
   * 立直せずに同じ牌を捨てた局では、役の無い聴牌になる。
   */
  @Test
  public void seatRonnedOnItsRiichiTileHasYakuTenpai() {
    for (boolean riichi : new boolean[] {true, false}) {
      Scene scene =
          new Scene(
              new int[][] {
                {Tiles.M1, Tiles.M2, Tiles.M3, Tiles.P4, Tiles.P5, Tiles.P6, Tiles.S7, Tiles.S8, Tiles.S9,
                  Tiles.P1, Tiles.P3, Tiles.TON, Tiles.TON},
                {Tiles.M2, Tiles.M2, Tiles.M4, Tiles.M4, Tiles.M6, Tiles.M6, Tiles.P2, Tiles.P2, Tiles.P4,
                  Tiles.P4, Tiles.P6, Tiles.P6, Tiles.PEI},
                {Tiles.M1, Tiles.M4, Tiles.M7, Tiles.P1, Tiles.P4, Tiles.P7, Tiles.S1, Tiles.S4, Tiles.S7,
                  Tiles.TON, Tiles.NAN, Tiles.SHA, Tiles.HAKU},
                {Tiles.M2, Tiles.M5, Tiles.M8, Tiles.P2, Tiles.P5, Tiles.P8, Tiles.S2, Tiles.S5, Tiles.S8,
                  Tiles.NAN, Tiles.SHA, Tiles.HATSU, Tiles.CHUN}
              },
              new int[] {Tiles.PEI});
      scene.choose(0, riichi ? Action.Type.RIICHI_DAHAI : Action.Type.DAHAI, Tiles.PEI);
      scene.choose(1, Action.Type.RON_AGARI, -1);
      HandEndFacts facts = scene.end();
      assertEquals(facts.outcome().kind(), EndKind.RON);
      assertEquals(facts.outcome().loser(), 0);
      assertEquals(facts.outcome().riichiMask(), 0, "a riichi ronned on its tile does not stand");
      assertTrue(facts.tenpai(0));
      assertEquals(facts.yakuTenpai(0), riichi, "riichi " + riichi);
      assertEquals(facts.reach().lastTenpaiEvent(0), 0);
      assertEquals(facts.reach().lastYakuTenpaiEvent(0), riichi ? 0 : TenpaiReach.NONE);
      long[] words = new long[OutcomeCodec.WORDS];
      OutcomeCodec.of(VARIANT).write(facts, 0, 0, words, 0);
      assertEquals(OutcomeCodec.of(VARIANT).outcome(words, 0), Outcome.DEAL_IN);
      assertTrue(OutcomeCodec.of(VARIANT).tenpai(words, 0));
      assertEquals(OutcomeCodec.of(VARIANT).yakuTenpai(words, 0), riichi);
    }
  }

  /** 九種九牌を宣言した 14 枚の手は、1 枚捨てて聴牌になれるなら聴牌と数える（ここでは 2 萬を捨てて国士無双の 13 面待ち）。 */
  @Test
  public void nineTerminalsHandWithFourteenTilesCountsItsBestDiscard() {
    Scene scene =
        new Scene(
            new int[][] {
              {Tiles.M1, Tiles.M9, Tiles.P1, Tiles.P9, Tiles.S1, Tiles.S9, Tiles.TON, Tiles.NAN, Tiles.SHA,
                Tiles.PEI, Tiles.HAKU, Tiles.HATSU, Tiles.M2},
              {Tiles.M2, Tiles.M3, Tiles.M4, Tiles.M5, Tiles.M6, Tiles.M7, Tiles.P2, Tiles.P3, Tiles.P4,
                Tiles.P5, Tiles.P7, Tiles.S2, Tiles.S4},
              {Tiles.M3, Tiles.M6, Tiles.M8, Tiles.P3, Tiles.P6, Tiles.P8, Tiles.S3, Tiles.S6, Tiles.S8,
                Tiles.TON, Tiles.NAN, Tiles.SHA, Tiles.PEI},
              {Tiles.M4, Tiles.M7, Tiles.P4, Tiles.P7, Tiles.S4, Tiles.S7, Tiles.S2, Tiles.S5, Tiles.M9,
                Tiles.P9, Tiles.S9, Tiles.HAKU, Tiles.HATSU}
            },
            new int[] {Tiles.CHUN});
    scene.choose(0, Action.Type.KYUSHU_KYUHAI, -1);
    HandEndFacts facts = scene.end();
    assertEquals(facts.outcome().kind(), EndKind.ABORTIVE);
    assertEquals(facts.outcome().abortive(), AbortiveDraw.NINE_TERMINALS);
    assertTrue(facts.tenpai(0), "the declarer is tenpai after discarding 2m");
    assertTrue(facts.yakuTenpai(0), "thirteen orphans is a yaku");
    assertEquals(facts.tenpaiMask() & ~1, 0);
  }

  private static HandEndFacts facts(RoundOutcome outcome, long[] yaku, int dealer, int tenpai) {
    TableState next = new TableState(VARIANT);
    next.set(1, 0, 0, new int[] {31_000, 17_000, 24_000, 28_000});
    return new HandEndFacts(VARIANT, outcome, yaku, dealer, next, tenpai, tenpai, new TenpaiReach(VARIANT));
  }

  /** 手で作った牌山で打つ東 1 局（親は席 0）。配牌は席ごとに 13 枚の牌の種類で決め、ツモは並べた順に引く。牌山の残りは使っていない牌で埋める。 */
  private static final class Scene {
    private final Engine engine = new Engine(VARIANT);
    private final RuleWorkspace ws = new RuleWorkspace(VARIANT);
    private final HandEndTracker tracker = new HandEndTracker(VARIANT);
    private Boundary boundary;

    Scene(int[][] hands, int[] draws) {
      int[] wall = new int[VARIANT.tiles().size()];
      Arrays.fill(wall, -1);
      int[] used = new int[Tiles.KINDS];
      for (int seat = 0; seat < SEATS; seat++) {
        for (int index = 0; index < Variant.HAND; index++) {
          int position = index < 12 ? index / 4 * 4 * SEATS + seat * 4 + index % 4 : 12 * SEATS + seat;
          wall[position] = take(used, hands[seat][index]);
        }
      }
      for (int draw = 0; draw < draws.length; draw++) {
        wall[Variant.HAND * SEATS + draw] = take(used, draws[draw]);
      }
      int kind = 0;
      for (int position = 0; position < wall.length; position++) {
        while (wall[position] < 0) {
          if (used[kind] < Tiles.COPIES) {
            wall[position] = take(used, kind);
          } else {
            kind++;
          }
        }
      }
      engine.startRecordedRound(RoundStart.first(VARIANT), wall);
      boundary = engine.advance(ws);
    }

    private static int take(int[] used, int kind) {
      if (used[kind] == Tiles.COPIES) {
        throw new IllegalArgumentException("Five copies of " + kind);
      }
      int copy = (PLAIN_COPY + used[kind]++) % Tiles.COPIES;
      return Tiles.physicalId(kind, copy);
    }

    /** 指定した席では、行動の種類が合い、牌を指定したときは牌も合う手を選ぶ。ほかの席は見送り、見送れなければ最初の合法手を選ぶ。 */
    void choose(int seat, Action.Type type, int tile) {
      assertEquals(boundary, Boundary.DECISIONS);
      tracker.scan(engine.full());
      Decisions decisions = engine.decisions();
      int[] choices = new int[decisions.size()];
      for (int slot = 0; slot < decisions.size(); slot++) {
        boolean mine = decisions.seat(slot) == seat;
        choices[slot] = find(decisions, slot, mine ? type : Action.Type.PASS, mine ? tile : -1);
        if (choices[slot] < 0) {
          assertFalse(mine, "seat " + seat + " has no " + type);
          choices[slot] = 0;
        }
      }
      boundary = engine.commit(choices, ws);
    }

    HandEndFacts end() {
      assertEquals(boundary, Boundary.ROUND_END);
      return tracker.end(engine);
    }

    private static int find(Decisions decisions, int slot, Action.Type type, int tile) {
      for (int index = 0; index < decisions.legalCount(slot); index++) {
        Action action = Action.of(decisions.legal(slot, index));
        if (action.type() == type && (tile < 0 || action.tile() == tile)) {
          return index;
        }
      }
      return -1;
    }
  }

  /** 判断する席と、判断の時点までのイベントの数（観測 O のヘッダーにある数）。 */
  private record Decision(int seat, int events) {}

  /** 全ての判断と局末を照合し、場合ごとの件数を数える。 */
  private static final class Check implements TableWalk.Visitor {
    private final ObservationWriter writer = new ObservationWriter(VARIANT);
    private final ObservationBatch batch = new ObservationBatch(VARIANT, 1 << 12);
    private final Observation read = new Observation(VARIANT);
    private final HandEndTracker tracker = new HandEndTracker(VARIANT);
    private final OutcomeCodec codec = OutcomeCodec.of(VARIANT);
    private final OracleCodec oracles = OracleCodec.of(VARIANT);
    private final ShapeAnalyzer shapes = new ShapeAnalyzer(VARIANT);
    private final RuleWorkspace ws = new RuleWorkspace(VARIANT);
    private final WinScore score = new WinScore();
    private final HandShape hand = new HandShape();
    private final HandShape trial = new HandShape();
    private final long[] words = new long[OutcomeCodec.WORDS];
    private final int[] delta = new int[SEATS];
    private final List<Decision> decisions = new ArrayList<>();
    private final List<List<Integer>> tenpaiEvents = new ArrayList<>();
    private final int[] outcomes = new int[Outcome.values().length];
    private final Set<AbortiveDraw> aborts = EnumSet.noneOf(AbortiveDraw.class);
    private int seen;
    private int dealer = -1;
    private int honba;
    private long decisionCount;
    private int rounds;
    private int multiRonSelf;
    private int multiRonDealIn;
    private int chankanDealIn;
    private int nagashi;
    private int honbaDealIn;
    private int riichiRonned;
    private int nineTerminals;
    private int reachedByDiscard;
    private int tenpaiWithoutYaku;
    private long dangers;
    private long fifthTiles;

    Check() {
      for (int seat = 0; seat < SEATS; seat++) {
        tenpaiEvents.add(new ArrayList<>());
      }
    }

    @Override
    public void decision(Engine engine, int slot) {
      FullView full = engine.full();
      tracker.scan(full);
      discards(full);
      Decisions table = engine.decisions();
      int seat = table.seat(slot);
      SeatView view = engine.view(seat);
      batch.clear();
      Observation o = read.bind(batch, writer.capture(view, table, slot, batch));
      assertEquals(o.events(), view.events().size(), "O counts the public events of the log");
      decisions.add(new Decision(seat, o.events()));
      dealer = view.dealer();
      honba = view.honba();
      checkOracle(full, seat, OracleCapture.capture(full, seat, ws));
      decisionCount++;
    }

    @Override
    public void roundEnd(Engine engine) {
      FullView full = engine.full();
      discards(full);
      RoundEnd end = engine.roundEnd();
      RoundOutcome outcome = end.outcome();
      HandEndFacts facts = tracker.end(engine);
      int winners = 0;
      for (int index = 0; index < outcome.winnerCount(); index++) {
        winners |= 1 << outcome.winner(index);
      }
      int tenpai = winners;
      for (int seat = 0; seat < SEATS; seat++) {
        if ((winners & 1 << seat) == 0 && shapeTenpai(full, seat)) {
          tenpai |= 1 << seat;
        }
      }
      String round = "round " + rounds + " " + outcome;
      assertEquals(facts.tenpaiMask(), tenpai, "end tenpai in " + round);
      assertEquals(facts.yakuTenpaiMask() & ~tenpai, 0, "yaku tenpai outside tenpai in " + round);
      assertEquals(facts.yakuTenpaiMask() & winners, winners, round);
      EventLog events = full.seat(0).events();
      for (int seat = 0; seat < SEATS; seat++) {
        if ((tenpai & 1 << seat) != 0 && declaredRiichi(events, seat)) {
          assertTrue(facts.yakuTenpai(seat), "riichi seat " + seat + " in " + round);
        }
      }
      assertEquals(facts.dealer(), dealer, round);
      switch (outcome.kind()) {
        case EXHAUSTIVE -> {
          assertEquals(tenpai, outcome.tenpaiMask(), "engine tenpai in " + round);
          nagashi += outcome.nagashiMask() != 0 ? 1 : 0;
        }
        case ABORTIVE -> {
          aborts.add(outcome.abortive());
          for (int seat = 0; seat < SEATS; seat++) {
            full.copyHand(seat, hand);
            nineTerminals += hand.size() == HandShape.MAX_SIZE ? 1 : 0;
          }
        }
        case RON -> {
          int last = events.size() - 1;
          boolean onRiichi =
              events.kind(last) == EventKind.DISCARD && events.riichi(last) && events.seat(last) == outcome.loser();
          riichiRonned += onRiichi ? 1 : 0;
        }
        case TSUMO -> {}
      }
      for (Decision decision : decisions) {
        checkOutcome(full, end, facts, decision, tenpai);
      }
      rounds++;
      decisions.clear();
      seen = 0;
      for (List<Integer> list : tenpaiEvents) {
        list.clear();
      }
    }

    /** 前回調べた後に増えた打牌ごとに、打牌した席が聴牌かを形の解析で調べ、聴牌ならその打牌を覚える。 */
    private void discards(FullView full) {
      EventLog events = full.seat(0).events();
      for (; seen < events.size(); seen++) {
        if (events.kind(seen) == EventKind.DISCARD) {
          int seat = events.seat(seen);
          if (shapeTenpai(full, seat)) {
            tenpaiEvents.get(seat).add(seen);
          }
        }
      }
    }

    /** 形の上で聴牌かを返す。13 枚相当の手は待ちがあるかで、14 枚相当の手は 1 枚捨てて待ちができるかで決める。 */
    private boolean shapeTenpai(FullView full, int seat) {
      full.copyHand(seat, hand);
      if (hand.size() == HandShape.MAX_SIZE - 1) {
        return shapes.waits(hand) != 0L;
      }
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        if (hand.count(kind) > 0) {
          trial.copyFrom(hand);
          trial.remove(kind);
          if (shapes.waits(trial) != 0L) {
            return true;
          }
        }
      }
      return false;
    }

    private void checkOutcome(FullView full, RoundEnd end, HandEndFacts facts, Decision decision, int tenpai) {
      int seat = decision.seat();
      RoundOutcome outcome = end.outcome();
      codec.write(facts, seat, decision.events(), words, 0);
      Outcome written = codec.outcome(words, 0);
      String where = "seat " + seat + " events " + decision.events() + " " + outcome;
      int winner = -1;
      int winPoints = 0;
      int dealInPoints = 0;
      long yaku = 0L;
      Outcome expected =
          switch (outcome.kind()) {
            case TSUMO -> {
              int tsumo = outcome.winner(0);
              yaku = end.score(0).yaku();
              if (tsumo == seat) {
                Arrays.fill(delta, 0);
                Settlement.tsumo(VARIANT, outcome.base(0), tsumo, dealer, 0, RoundOutcome.NO_SEAT, delta);
                winPoints = delta[tsumo];
                yield Outcome.SELF_TSUMO;
              }
              winner = relative(seat, tsumo);
              yield Outcome.OTHER_TSUMO;
            }
            case RON -> {
              int own = -1;
              for (int index = 0; index < outcome.winnerCount(); index++) {
                own = outcome.winner(index) == seat ? index : own;
              }
              if (own >= 0) {
                yaku = end.score(own).yaku();
                Arrays.fill(delta, 0);
                Settlement.ron(
                    VARIANT, outcome.base(own), seat, outcome.loser(), dealer, 0, RoundOutcome.NO_SEAT, delta);
                winPoints = delta[seat];
                multiRonSelf += outcome.winnerCount() > 1 ? 1 : 0;
                yield Outcome.SELF_RON;
              }
              yaku = end.score(0).yaku();
              winner = relative(seat, outcome.winner(0));
              if (outcome.loser() != seat) {
                yield Outcome.OTHER_RON;
              }
              for (int index = 0; index < outcome.winnerCount(); index++) {
                Arrays.fill(delta, 0);
                Settlement.ron(
                    VARIANT, outcome.base(index), outcome.winner(index), seat, dealer, 0, outcome.pao(index), delta);
                dealInPoints -= delta[seat];
              }
              multiRonDealIn += outcome.winnerCount() > 1 ? 1 : 0;
              honbaDealIn += honba > 0 ? 1 : 0;
              EventLog events = full.seat(0).events();
              chankanDealIn += events.kind(events.size() - 1) == EventKind.KAKAN ? 1 : 0;
              yield Outcome.DEAL_IN;
            }
            case EXHAUSTIVE -> (tenpai & 1 << seat) != 0 ? Outcome.DRAW_TENPAI : Outcome.DRAW_NOTEN;
            case ABORTIVE -> Outcome.ABORTIVE;
          };
      assertEquals(written, expected, where);
      outcomes[written.ordinal()]++;
      assertEquals(codec.winner(words, 0), winner, where);
      assertEquals(codec.winPoints(words, 0), winPoints, where);
      assertEquals(codec.dealInPoints(words, 0), dealInPoints, where);
      assertEquals(codec.yakuBits(words, 0), yaku, where);

      boolean endTenpai = (tenpai & 1 << seat) != 0;
      boolean reached = endTenpai;
      for (int event : tenpaiEvents.get(seat)) {
        reached |= event >= decision.events();
      }
      assertEquals(codec.tenpai(words, 0), reached, "tenpai reach " + where);
      boolean reachedYaku = codec.yakuTenpai(words, 0);
      assertTrue(!reachedYaku || reached, "yaku reach without tenpai " + where);
      assertTrue(reachedYaku || !facts.yakuTenpai(seat), "yaku reach " + where);
      reachedByDiscard += reached && !endTenpai ? 1 : 0;
      tenpaiWithoutYaku += reached && !reachedYaku ? 1 : 0;

      TableState next = end.next();
      for (int relative = 0; relative < SEATS; relative++) {
        assertEquals(codec.finalScore(words, 0, relative), next.score((seat + relative) % SEATS), where);
      }
    }

    /** Oracle の記録を、形の解析と {@code ronScore} で照合する。 */
    private void checkOracle(FullView full, int seat, long[] oracle) {
      assertEquals(oracles.longs(oracle, 0), oracle.length);
      SeatView table = full.seat(seat);
      int index = 0;
      for (int opponent = 1; opponent < SEATS; opponent++) {
        int player = (seat + opponent) % SEATS;
        full.copyHand(player, hand);
        assertEquals(hand.size(), HandShape.MAX_SIZE - 1, "opponent " + player + " at a decision");
        assertEquals(oracles.shanten(oracle, 0, opponent), shapes.shanten(hand));
        long waits = shapes.waits(hand);
        assertEquals(oracles.waits(oracle, 0, opponent), waits);
        long danger = oracles.danger(oracle, 0, opponent);
        assertEquals(danger & ~waits, 0L, "danger outside the waits");
        long held = held(table, player, hand);
        fifthTiles += Long.bitCount(waits & held);
        for (int kind = 0; kind < Tiles.KINDS; kind++) {
          if (Tiles.contains(held, kind)) {
            assertFalse(Tiles.contains(danger, kind), "danger on a fifth tile " + kind);
            continue;
          }
          boolean expected = full.ronScore(player, Tiles.physicalId(kind, PLAIN_COPY), seat, ws, score);
          assertTrue(!expected || Tiles.contains(waits, kind), "ron outside the waits on " + kind);
          assertEquals(Tiles.contains(danger, kind), expected, "danger " + kind);
          if (expected) {
            dangers++;
            int points = ScoreMath.ron(score.base(), player == table.dealer());
            assertEquals(oracles.points(oracle, 0, index++), points, "danger points");
            assertTrue(points % 100 == 0 && points <= MAX_POINTS, "points " + points);
          }
        }
      }
    }

    /** 手牌・面子・抜いた北を合わせて、4 枚とも持っている牌の種類を返す（種類ごとに 1 ビット）。 */
    private static long held(SeatView view, int player, HandShape shape) {
      int[] count = new int[Tiles.KINDS];
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        count[kind] = shape.count(kind);
      }
      for (int index = 0; index < view.meldCount(player); index++) {
        Meld meld = Meld.of(view.meld(player, index));
        for (int at = 0; at < meld.size(); at++) {
          count[meld.tile(at)]++;
        }
      }
      count[Tiles.PEI] += view.nuki(player);
      long full = 0L;
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        full |= count[kind] >= Tiles.COPIES ? Tiles.bit(kind) : 0L;
      }
      return full;
    }

    private static boolean declaredRiichi(EventLog events, int seat) {
      for (int event = 0; event < events.size(); event++) {
        if (events.kind(event) == EventKind.DISCARD && events.riichi(event) && events.seat(event) == seat) {
          return true;
        }
      }
      return false;
    }

    private static int relative(int seat, int player) {
      return Math.floorMod(player - seat, SEATS);
    }

    String summary() {
      StringBuilder out = new StringBuilder("OUTCOME-T8 decisions=").append(decisionCount);
      out.append(" rounds=").append(rounds);
      for (Outcome outcome : Outcome.values()) {
        out.append(' ').append(outcome).append('=').append(outcomes[outcome.ordinal()]);
      }
      out.append(" multiRonSelf=").append(multiRonSelf);
      out.append(" multiRonDealIn=").append(multiRonDealIn);
      out.append(" chankanDealIn=").append(chankanDealIn);
      out.append(" nagashi=").append(nagashi);
      out.append(" aborts=").append(aborts);
      out.append(" nineTerminals14=").append(nineTerminals);
      out.append(" honbaDealIn=").append(honbaDealIn);
      out.append(" riichiRonned=").append(riichiRonned);
      out.append(" reachedByDiscard=").append(reachedByDiscard);
      out.append(" tenpaiWithoutYaku=").append(tenpaiWithoutYaku);
      out.append(" dangers=").append(dangers);
      out.append(" fifthTiles=").append(fifthTiles);
      return out.toString();
    }
  }
}
