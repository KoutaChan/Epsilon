package com.epsilon.prism3p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.facts.WaitScores.Method;
import com.epsilon.agent.facts.WaitScores.WinningTile;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.HandScorer;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.core.SuitSymmetry;
import com.epsilon.prism.input.AfterHand;
import com.epsilon.prism.input.EventTable;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.PassGrade;
import com.epsilon.prism.input.PriorFeatures;
import com.epsilon.prism.input.PublicSeatFacts;
import com.epsilon.prism.input.ScoreLadder;
import com.epsilon.prism.input.ShapeAnalysis;
import com.epsilon.prism.input.SuitCanonicalizer;
import com.epsilon.prism.input.TransitionHands;
import com.epsilon.prism.input.WaitScoring;
import com.epsilon.prism.input.WaitScoring.Stance;
import com.epsilon.prism.input.WinFacts;
import com.epsilon.prism3p.input.InputSchema.TileField;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 三人打ちの Tokenizer が求める値と書いた行を、新しいエンジンと総当たりで照合する。四人打ちのテストと同じことを三人打ちで行い、
 * ツモ損・抜きドラ・27 種類の牌の集合も通す。
 *
 * <ul>
 *   <li>見える枚数と見えない枚数が席のビューと一致し、正準の牌のトークンに書いた枚数も同じになる。
 *   <li>打牌・ポンの後の打牌・北抜きの後の手（純手牌・面子・赤・ドラ・抜きの数）が、エンジンの写しで同じ手を打った後の手と一致する。
 *   <li>形（4 種の向聴・受け入れ・形の待ち・生きた受け入れ・1〜3 ツモ以内に引く確率）が、牌を 1 枚ずつ足す総当たりと一致する。受け入れに
 *       2〜8 萬は出ない。
 *   <li>聴牌の待ちごとの点（赤でない和了牌と赤 5）が、エンジンの写しの手（抜いた北を含む）を直に採点した値と一致する。
 *   <li>追い越しの格が、格ごとに三人打ちの精算の式で点を動かして求めた値と一致する。
 *   <li>即時の和了の得点・順位・終局・連荘が、エンジンの写しで和了した局末と一致する（ツモ損も含む。裏ドラの乗った和了は除く）。
 *   <li>p0 の特徴が、総当たりで求めた向聴と受け入れの枚数を式に通した値とビットで一致する。
 *   <li>全ての行がライターの確かめ（語彙・キーのコード・手のインデックス）を通る。
 *   <li>牌山の筒子と索子を入れ替えた対の局の判断で、入力がビットで同じになる。色のキーが物理の色まで同点になる判断は除く。
 * </ul>
 */
public class DerivationTest {
  private static final Logger log = LoggerFactory.getLogger(DerivationTest.class);

  private static final Variant VARIANT = TENHOU_3P;
  private static final long GREEDY_SEED = 510;
  private static final long CALLS_SEED = 8_210;
  private static final int SEATS = VARIANT.seats();

  /** 筒子と索子だけを入れ替える {@link SuitSymmetry} の番号。 */
  private static final int PIN_SOU_VIEW = 5;

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
    Tensor.TRANSITION_CATEGORY, Tensor.TRANSITION_SCALAR, Tensor.TRANSITION_CODE, Tensor.DISCARD_EXACT
  };
  private static final Method[] METHODS = Method.values();

  @Test(timeOut = 1_200_000)
  public void derivationsMatchTheEngine() {
    Derivations greedy = new Derivations();
    new TableWalk(VARIANT, TableWalk.Policy.GREEDY).run(GREEDY_SEED, 40, greedy);
    Derivations calls = new Derivations();
    new TableWalk(VARIANT, TableWalk.Policy.CALLS).run(CALLS_SEED, 40, calls);
    log.info(greedy.summary("GREEDY"));
    log.info(calls.summary("CALLS"));
    for (Derivations check : List.of(greedy, calls)) {
      assertTrue(check.decisions > 5_000, "decisions " + check.decisions);
      assertTrue(check.grades > 500, "scored pass grades " + check.grades);
      assertTrue(check.engineHands > 3_000, "hands compared with the engine " + check.engineHands);
      assertTrue(check.nukiHands > 100, "nuki hands compared with the engine " + check.nukiHands);
    }
    assertTrue(greedy.priors > 3_000, "p0 features " + greedy.priors);
    assertTrue(greedy.tenpai > 200, "tenpai transitions " + greedy.tenpai);
    assertTrue(greedy.redWaits > 0, "red winning tiles " + greedy.redWaits);
    assertTrue(greedy.nukiWaits > 0, "waits scored with a taken-out north " + greedy.nukiWaits);
    assertTrue(greedy.riichi > 0, "riichi transitions " + greedy.riichi);
    assertTrue(greedy.winChecks > 0 && calls.winChecks > 0, "immediate wins");
    assertTrue(greedy.tsumoChecks > 0 || calls.tsumoChecks > 0, "tsumo wins");
    assertTrue(calls.calls > 300, "call transitions " + calls.calls);
    assertTrue(calls.callHands > 30, "call hands compared with the engine " + calls.callHands);
  }

  /** 筒子と索子を入れ替えた対の局で、入力がビットで同じになる。三人打ちで許される色の入れ替えは、この 2 色だけ。 */
  @Test(timeOut = 1_200_000)
  public void permutedSuitsEncodeTheSameInput() {
    Mirror mirror = new Mirror();
    new TableWalk(VARIANT, TableWalk.Policy.GREEDY).runMirrored(GREEDY_SEED + 1_000, 80, PIN_SOU_VIEW, mirror);
    new TableWalk(VARIANT, TableWalk.Policy.CALLS).runMirrored(CALLS_SEED + 1_000, 80, PIN_SOU_VIEW, mirror);
    log.info(mirror.summary());
    assertTrue(mirror.compared > 5_000, "compared decisions " + mirror.compared);
    assertTrue(mirror.physicallyDifferent * 2 > mirror.compared, "physically different hands " + mirror.physicallyDifferent);
    assertTrue(mirror.tenpai > 50, "compared tenpai transitions " + mirror.tenpai);
  }

  /** Tokenizer が求めた値を、エンジンと総当たりで照合する。 */
  private static final class Derivations implements TableWalk.Visitor {
    private final RowEncoder encoder = new RowEncoder();
    private final TileCounts tiles = new TileCounts(VARIANT);
    private final SeatFacts seatFacts = new SeatFacts(VARIANT);
    private final TransitionHands hands = new TransitionHands();
    private final ShapeAnalysis shapes = new ShapeAnalysis(VARIANT);
    private final WaitScoring waits = new WaitScoring(VARIANT);
    private final ScoreLadder ladder = new ScoreLadder(VARIANT);
    private final WinFacts wins = new WinFacts(InputSchema.UTILITY);
    private final ShapeAnalyzer analyzer = new ShapeAnalyzer(VARIANT);
    private final HandScorer scorer = new HandScorer();
    private final WinScore score = new WinScore();
    private final WinContext mine = new WinContext(VARIANT);
    private final WinContext engineHand = new WinContext(VARIANT);
    private final HandShape probe = new HandShape();
    private final RuleWorkspace ws = new RuleWorkspace(VARIANT);
    private final int[] choices = new int[SEATS - 1];
    private final int[] deltas = new int[SEATS];
    private long decisions;
    private long engineHands;
    private long callHands;
    private long nukiHands;
    private long nukiWaits;
    private long tenpai;
    private long redWaits;
    private long riichi;
    private long calls;
    private long grades;
    private long winChecks;
    private long tsumoChecks;
    private long priors;

    @Override
    public void decision(Engine engine, int slot) {
      Decisions boundary = engine.decisions();
      int seat = boundary.seat(slot);
      Observation read = encoder.encode(engine, slot);
      tiles.load(read);
      seatFacts.load(read);
      SeatView view = engine.view(seat);
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        assertEquals(tiles.hand(kind), view.count(kind), "hand " + kind);
        assertEquals(tiles.visible(kind), view.visible(kind), "visible " + kind);
        assertEquals(tiles.unseen(kind), VARIANT.tiles().copies(kind) - view.visible(kind) - view.count(kind), "unseen " + kind);
        if (VARIANT.tiles().copies(kind) > 0) {
          int token = encoder.tokenizer.canonicalTile(kind);
          assertEquals(encoder.tile(token, TileField.HAND), view.count(kind), "hand token of " + kind);
          assertEquals(encoder.tile(token, TileField.VISIBLE), view.visible(kind), "visible token of " + kind);
          assertEquals(encoder.tile(token, TileField.UNSEEN), tiles.unseen(kind), "unseen token of " + kind);
        }
      }
      for (int other = 0; other < SEATS; other++) {
        assertEquals(encoder.seat(other, InputSchema.SeatField.NUKI), view.nuki(read.absoluteSeat(other)), "nuki of seat " + other);
      }
      hands.load(read, tiles, seatFacts);
      shapes.reset();
      for (int action = 0; action < read.legalActions(); action++) {
        if (read.representative(action) == action) {
          for (int index = 0; index < read.transitionsOf(action); index++) {
            transition(engine, slot, read, action, index);
          }
        }
      }
      ladder(read);
      for (int action = 0; action < read.legalActions(); action++) {
        Action.Type type = read.action(action).type();
        if (type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI) {
          win(engine, slot, read, action);
        }
      }
      decisions++;
    }

    /** 1 つの遷移の後の手・形・待ちの点・p0 を、エンジンと総当たりで照合する。 */
    private void transition(Engine engine, int slot, Observation read, int action, int index) {
      Action chosen = read.action(action);
      Action.Type type = chosen.type();
      AfterHand after = hands.move(action, index);
      calls += type == Action.Type.PON ? 1 : 0;
      if (TransitionHands.terminal(type)) {
        return;
      }
      String where = chosen + " #" + index;
      boolean compared = false;
      if (type == Action.Type.DAHAI
          || type == Action.Type.RIICHI_DAHAI
          || type == Action.Type.PON
          || type == Action.Type.NUKI) {
        compared = playOnCopy(engine, slot, action, hands.discard(), where);
        if (compared) {
          assertSameHand(after, where);
        }
      }
      if (after.effectiveTileCount() != HandShape.MAX_SIZE - 1) {
        return;
      }
      shapes.analyze(after, tiles, ShapeAnalysis.UNSHARED);
      long ukeire = shapeByBruteForce(after, where);
      if (type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI) {
        assertEquals(bits(encoder.batch.priorShanten(action)), bits(PriorFeatures.shanten(shapes.minimum())), where);
        assertEquals(bits(encoder.batch.priorUkeire(action)), bits(PriorFeatures.ukeire(liveOf(ukeire))), where);
        priors++;
      }
      long waitMask = shapes.waits();
      if (waitMask == 0L || !compared) {
        return;
      }
      tenpai++;
      nukiWaits += after.nuki() > 0 ? 1 : 0;
      int seatWind = PublicSeatFacts.seatWindTile(read, 0);
      boolean declares = type == Action.Type.RIICHI_DAHAI && !read.beforeFirstDiscard(0);
      riichi += declares ? 1 : 0;
      waits.score(after, waitMask, tiles, read, seatWind, declares ? RiichiState.RIICHI : RiichiState.NONE);
      assertEquals(waits.waits(), Long.bitCount(waitMask), where);
      for (int wait = 0; wait < waits.waits(); wait++) {
        int kind = waits.waitTile(wait);
        waitPoints(read, wait, kind, seatWind, Stance.DAMA, RiichiState.NONE, where);
        if (declares) {
          waitPoints(read, wait, kind, seatWind, Stance.RIICHI, RiichiState.RIICHI, where);
        }
      }
    }

    /**
     * エンジンの写しで同じ手を打ち（ポンは鳴いた後の打牌まで）、その席の手を採点の条件に書く。ほかの席は見送る。優先の高いほかの席の手に
     * 鳴きが負けるなどして打てないときは false を返す。北抜きでは、補充を引いた後の手から補充の牌を除き、抜いた直後の 13 枚にする。
     */
    private boolean playOnCopy(Engine engine, int slot, int action, Action discard, String where) {
      Decisions boundary = engine.decisions();
      int seat = boundary.seat(slot);
      for (int other = 0; other < boundary.size(); other++) {
        choices[other] = other == slot ? action : passIndex(boundary, other);
        if (choices[other] < 0) {
          return false;
        }
      }
      Engine copy = engine.copy();
      Boundary reached = copy.commit(choices, ws);
      Action.Type type = Action.of(boundary.legal(slot, action)).type();
      if (type == Action.Type.PON) {
        if (reached != Boundary.DECISIONS) {
          return false;
        }
        Decisions next = copy.decisions();
        int nextSlot = next.slotOf(seat);
        assertTrue(nextSlot >= 0 && next.size() == 1, "the caller discards next at " + where);
        int index = -1;
        for (int legal = 0; legal < next.legalCount(nextSlot); legal++) {
          if (next.legal(nextSlot, legal) == discard.code()) {
            index = legal;
          }
        }
        assertTrue(index >= 0, "the call discard " + discard + " is legal at " + where);
        choices[nextSlot] = index;
        copy.commit(choices, ws);
        callHands++;
      }
      if (type == Action.Type.NUKI) {
        if (reached != Boundary.DECISIONS) {
          return false;
        }
        while (copy.decisions().kind(0) == EngineDecisionKind.NUKI_RESPONSE) {
          Decisions next = copy.decisions();
          for (int other = 0; other < next.size(); other++) {
            choices[other] = passIndex(next, other);
          }
          if (copy.commit(choices, ws) != Boundary.DECISIONS) {
            return false;
          }
        }
        assertEquals(copy.decisions().seat(0), seat, "the nuki seat draws next at " + where);
        nukiHands++;
      }
      SeatView view = copy.view(seat);
      engineHand.clear();
      copy.full().copyHand(seat, engineHand.hand());
      for (int meld = 0; meld < view.meldCount(seat); meld++) {
        engineHand.addMeld(Meld.of(view.meld(seat, meld)));
      }
      long red = 0L;
      for (long fives = Tiles.FIVES; fives != 0L; fives &= fives - 1) {
        int kind = Long.numberOfTrailingZeros(fives);
        if (view.holds(Tiles.physicalId(kind, Tiles.RED_COPY))) {
          red |= Tiles.bit(kind);
        }
      }
      if (type == Action.Type.NUKI) {
        TurnEvent drawn = view.last();
        engineHand.hand().remove(drawn.tile());
        if (drawn.red()) {
          red &= ~Tiles.bit(drawn.tile());
        }
      }
      engineHand.setRedFives(red);
      engineHand.setNuki(view.nuki(seat));
      // 表示牌は判断の時点のビューから読む（大明槓・加槓の後の槓ドラは、この打牌の後にめくれる）。
      SeatView before = engine.view(seat);
      for (int index = 0; index < before.indicatorCount(); index++) {
        engineHand.addIndicator(Tiles.kind(before.indicator(index)));
      }
      engineHands++;
      return true;
    }

    /** 手を打った後の手を、エンジンの写しの手（{@link #playOnCopy} が書いたもの）と照合する。 */
    private void assertSameHand(AfterHand after, String where) {
      mine.clear();
      after.writeTo(mine);
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        assertEquals(mine.hand().count(kind), engineHand.hand().count(kind), where + " kind " + kind);
      }
      assertEquals(mine.hand().melds(), engineHand.hand().melds(), where);
      assertEquals(mine.meldCount(), engineHand.meldCount(), where);
      for (int meld = 0; meld < mine.meldCount(); meld++) {
        assertEquals(mine.meld(meld), engineHand.meld(meld), where + " meld " + meld);
      }
      assertEquals(mine.redFives(), engineHand.redFives(), where + " red");
      assertEquals(mine.nuki(), engineHand.nuki(), where + " nuki");
      int dora = Long.bitCount(engineHand.redFives()) + engineHand.nuki() * (1 + tiles.dora(Tiles.PEI));
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        dora += engineHand.hand().count(kind) * tiles.dora(kind);
      }
      for (int meld = 0; meld < engineHand.meldCount(); meld++) {
        Meld m = engineHand.meld(meld);
        dora += m.red() ? 1 : 0;
        for (int tile = 0; tile < m.size(); tile++) {
          dora += tiles.dora(m.tile(tile));
        }
      }
      assertEquals(after.dora(), dora, where + " dora");
    }

    /** 形を総当たりで照合し、受け入れの種類を返す。三人打ちの牌の集合に無い種類は受け入れにならない。 */
    private long shapeByBruteForce(AfterHand after, String where) {
      HandShape shape = after.shape();
      boolean concealed = shape.melds() == 0;
      int standard = analyzer.standardShanten(shape);
      int chiitoi = concealed ? ShapeAnalyzer.chiitoitsuShanten(shape) : 0;
      int kokushi = concealed ? ShapeAnalyzer.kokushiShanten(shape) : 0;
      int minimum = concealed ? Math.min(standard, Math.min(chiitoi, kokushi)) : standard;
      assertEquals(shapes.minimum(), minimum, where);
      assertEquals(shapes.standard(), standard, where);
      assertEquals(shapes.chiitoi(), chiitoi, where);
      assertEquals(shapes.kokushi(), kokushi, where);
      assertEquals(shapes.special(), concealed, where);
      long ukeire = 0L;
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        if (VARIANT.tiles().copies(kind) == 0 || held(after, kind) >= Tiles.COPIES) {
          continue;
        }
        probe.copyFrom(shape);
        probe.add(kind);
        if (analyzer.shanten(probe) < minimum) {
          ukeire |= Tiles.bit(kind);
        }
      }
      assertEquals(shapes.ukeire(), ukeire, where);
      assertEquals(shapes.waits(), minimum == 0 ? ukeire : 0L, where);
      int live = liveOf(ukeire);
      assertEquals(shapes.liveUkeire(), live, where);
      int total = tiles.totalUnseen();
      for (int draws = 1; draws <= 3; draws++) {
        double miss = 1.0;
        for (int draw = 0; draw < draws; draw++) {
          miss *= Math.max(0, total - live - draw) / (double) (total - draw);
        }
        float expected = live == 0 ? 0f : (float) (1.0 - miss);
        assertEquals(shapes.hitWithin(draws), expected, 1e-6f, where + " within " + draws);
      }
      return ukeire;
    }

    /** 待ちの点を、エンジンの写しの手を直に採点した値と照合する。 */
    private void waitPoints(
        Observation read, int wait, int kind, int seatWind, Stance stance, RiichiState state, String where) {
      engineHand.setWinds(seatWind, read.roundWindTile());
      engineHand.setRiichi(state, false);
      long handRed = engineHand.redFives();
      boolean exhausted = engineHand.hand().count(kind) + meldHeld(engineHand, kind) >= Tiles.COPIES;
      boolean red = !exhausted && Tiles.contains(tiles.unseenRed() & ~handRed, kind);
      for (Method method : METHODS) {
        int base = 0;
        int han = 0;
        if (!exhausted) {
          engineHand.setWin(kind, method == Method.RON ? WinContext.Source.RON : WinContext.Source.TSUMO);
          if (scorer.score(engineHand, score)) {
            base = score.base();
            han = score.han();
          }
        }
        assertEquals(waits.basePoints(wait, stance, method, WinningTile.PLAIN), base, where + " " + method + " " + engineHand);
        assertEquals(waits.han(wait, stance, method, WinningTile.PLAIN), han, where + " " + method);
        if (red) {
          engineHand.setRedFives(handRed | Tiles.bit(kind));
          int redBase = scorer.score(engineHand, score) ? score.base() : 0;
          engineHand.setRedFives(handRed);
          assertEquals(waits.basePoints(wait, stance, method, WinningTile.RED), redBase, where + " red " + method);
          redWaits++;
        } else {
          assertEquals(waits.basePoints(wait, stance, method, WinningTile.RED), 0, where + " red " + method);
        }
      }
    }

    /** 追い越しの格を、格ごとに精算の式で点を動かして求めた値と照合する。 */
    private void ladder(Observation read) {
      ladder.load(read);
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

    /**
     * {@code winner} が {@code from} から（同じ席ならツモで）和了したとき、{@code loser} の上に出るのに要る最小の格を返す。
     * もともと上なら NOT_NEEDED、どの格でも届かなければ UNREACHABLE。
     */
    private PassGrade grade(Observation read, int winner, int loser, int from) {
      int winnerSeat = read.absoluteSeat(winner);
      int loserSeat = read.absoluteSeat(loser);
      if (above(read.score(winner), winnerSeat, read.score(loser), loserSeat)) {
        return PassGrade.NOT_NEEDED;
      }
      for (PassGrade grade : PassGrade.values()) {
        if (!grade.scored()) {
          continue;
        }
        settle(grade.basePoints(), winner, from, read.dealer(), read.honba(), read.kyotaku(), deltas);
        if (above(read.score(winner) + deltas[winner], winnerSeat, read.score(loser) + deltas[loser], loserSeat)) {
          return grade;
        }
      }
      return PassGrade.UNREACHABLE;
    }

    /** 即時の和了の得点・順位・終局・連荘を、エンジンの写しで和了した局末と照合する（裏ドラの乗った和了は除く）。 */
    private void win(Engine engine, int slot, Observation read, int action) {
      Decisions boundary = engine.decisions();
      for (int other = 0; other < boundary.size(); other++) {
        choices[other] = other == slot ? action : passIndex(boundary, other);
        if (choices[other] < 0) {
          return;
        }
      }
      Engine copy = engine.copy();
      copy.commit(choices, ws);
      RoundEnd end = copy.roundEnd();
      RoundOutcome outcome = end.outcome();
      int observer = read.observer();
      assertEquals(outcome.winnerCount(), 1, "one winner");
      assertEquals(outcome.winner(0), observer);
      if (end.score(0).ura() > 0) {
        return;
      }
      wins.load(read, read.winOf(action));
      TableState next = end.next();
      boolean ends = next.gameOver();
      int rank = next.rank(observer);
      String where = read.action(action) + " " + read.kyoku() + "/" + read.honba() + "/" + read.kyotaku();
      assertEquals(wins.points(), next.score(observer) - read.score(0), where);
      assertEquals(wins.ends(), ends, where);
      assertEquals(wins.renchan(), !ends && read.dealer() == 0, where);
      assertEquals(wins.rank(), rank, where);
      float utility = ends ? (float) InputSchema.UTILITY.value(VARIANT, rank) : 0f;
      assertEquals(wins.utility(), utility, where);
      winChecks++;
      tsumoChecks += read.action(action).type() == Action.Type.TSUMO_AGARI ? 1 : 0;
    }

    private int liveOf(long kinds) {
      int live = 0;
      for (long rest = kinds; rest != 0L; rest &= rest - 1) {
        live += tiles.unseen(Long.numberOfTrailingZeros(rest));
      }
      return live;
    }

    String summary(String policy) {
      return String.format(
          Locale.ROOT,
          "INPUT-3P %s: decisions %d hands %d callHands %d nukiHands %d calls %d tenpai %d riichi %d redWaits %d"
              + " nukiWaits %d scoredGrades %d wins %d tsumos %d p0Features %d",
          policy,
          decisions,
          engineHands,
          callHands,
          nukiHands,
          calls,
          tenpai,
          riichi,
          redWaits,
          nukiWaits,
          grades,
          winChecks,
          tsumoChecks,
          priors);
    }
  }

  /** 筒子と索子を入れ替えた対の局の判断で、書いた行をビットで照合する。 */
  private static final class Mirror implements TableWalk.MirrorVisitor {
    private final RowEncoder left = new RowEncoder();
    private final RowEncoder right = new RowEncoder();
    private final EventTable events = new EventTable(VARIANT);
    private final TileCounts tiles = new TileCounts(VARIANT);
    private final SuitCanonicalizer suits = new SuitCanonicalizer(VARIANT);
    private long decisions;
    private long compared;
    private long physicalTies;
    private long physicallyDifferent;
    private long tenpai;

    @Override
    public void decision(Engine engine, int slot, Engine mirror, int mirrorSlot) {
      Observation a = left.encode(engine, slot);
      Observation b = right.encode(mirror, mirrorSlot);
      decisions++;
      boolean tie = physicalTie(a);
      assertEquals(physicalTie(b), tie, "physical ties of the pair");
      if (tie) {
        physicalTies++;
        return;
      }
      // 照合が空振りでないことを数える。物理の牌で見れば、対の手はたいてい違う。
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        if (a.handCount(kind) != b.handCount(kind)) {
          physicallyDifferent++;
          break;
        }
      }
      String where = "decision " + decisions + " " + a.decisionKind();
      for (Tensor tensor : ROW_TENSORS) {
        int count = left.batch.plan().count(tensor.axis());
        assertEquals(count, right.batch.plan().count(tensor.axis()), where);
        assertTrue(Arrays.equals(left.bytes(tensor, 0, count), right.bytes(tensor, 0, count)), tensor + " differs at " + where);
      }
      int actions = a.legalActions();
      int[] firstCall = new int[actions];
      int[] mirrorFirstCall = new int[actions];
      for (int action = 1; action < actions; action++) {
        firstCall[action] = firstCall[action - 1] + a.callDiscardsOf(action - 1);
        mirrorFirstCall[action] = mirrorFirstCall[action - 1] + b.callDiscardsOf(action - 1);
      }
      for (int action = 0; action < actions; action++) {
        int mapped = find(b, a.action(action));
        assertTrue(mapped >= 0, where);
        assertTrue(
            Arrays.equals(left.bytes(Tensor.ACTION_EXACT, action, 1), right.bytes(Tensor.ACTION_EXACT, mapped, 1)),
            "action exact differs at " + where + " " + a.action(action));
        assertEquals(left.column(action, ActionColumn.ROLE), right.column(mapped, ActionColumn.ROLE));
        assertEquals(left.column(action, ActionColumn.TRANSITIONS), right.column(mapped, ActionColumn.TRANSITIONS));
        assertMapped(a, b, left.column(action, ActionColumn.BASE), right.column(mapped, ActionColumn.BASE));
        assertMapped(a, b, left.column(action, ActionColumn.REPRESENTATIVE), right.column(mapped, ActionColumn.REPRESENTATIVE));
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
              if (SuitSymmetry.mapped(PIN_SOU_VIEW, discard, b.callDiscard(mirrorFirstCall[mapped] + other))) {
                mirrorIndex = other;
              }
            }
            assertTrue(mirrorIndex >= 0, where);
          }
          for (Tensor tensor : TRANSITION_TENSORS) {
            assertTrue(
                Arrays.equals(left.bytes(tensor, first + index, 1), right.bytes(tensor, mirrorFirst + mirrorIndex, 1)),
                tensor + " differs at " + where + " " + a.action(action) + " #" + index);
          }
          byte[] exact = left.bytes(Tensor.DISCARD_EXACT, first + index, 1);
          tenpai += exact[0] != 0 || exact[1] != 0 || exact[2] != 0 || exact[3] != 0 ? 1 : 0;
        }
      }
      compared++;
    }

    private int find(Observation shadow, Action action) {
      for (int index = 0; index < shadow.legalActions(); index++) {
        if (SuitSymmetry.mapped(PIN_SOU_VIEW, action, shadow.action(index))) {
          return index;
        }
      }
      return -1;
    }

    /** 手の番号の列（無ければ −1）が、色を入れ替えた写しで対になる手を指すことを確かめる。 */
    private void assertMapped(Observation a, Observation b, int index, int mirrorIndex) {
      if (index < 0) {
        assertEquals(mirrorIndex, -1);
        return;
      }
      assertTrue(SuitSymmetry.mapped(PIN_SOU_VIEW, a.action(index), b.action(mirrorIndex)));
    }

    private boolean physicalTie(Observation o) {
      events.load(o);
      tiles.load(o);
      suits.load(o, events, tiles);
      return suits.physicalTie();
    }

    String summary() {
      return String.format(
          Locale.ROOT,
          "INPUT-3P mirrored: decisions %d compared %d physicalTies %d physicallyDifferentHands %d tenpaiTransitions %d",
          decisions,
          compared,
          physicalTies,
          physicallyDifferent,
          tenpai);
    }
  }

  /** 判断する席の合法手のうち、見送りの手が何番目か（無ければ −1）。 */
  private static int passIndex(Decisions boundary, int slot) {
    for (int index = 0; index < boundary.legalCount(slot); index++) {
      if (Action.of(boundary.legal(slot, index)).type() == Action.Type.PASS) {
        return index;
      }
    }
    return -1;
  }

  /** 純手牌と面子と抜いた北にある、その種類の牌の枚数。 */
  private static int held(AfterHand hand, int kind) {
    int held = hand.count(kind) + (kind == Tiles.PEI ? hand.nuki() : 0);
    for (int index = 0; index < hand.meldCount(); index++) {
      Meld meld = hand.meld(index);
      for (int tile = 0; tile < meld.size(); tile++) {
        held += meld.tile(tile) == kind ? 1 : 0;
      }
    }
    return held;
  }

  private static int meldHeld(WinContext context, int kind) {
    int held = 0;
    for (int index = 0; index < context.meldCount(); index++) {
      Meld meld = context.meld(index);
      for (int tile = 0; tile < meld.size(); tile++) {
        held += meld.tile(tile) == kind ? 1 : 0;
      }
    }
    return held;
  }

  /**
   * {@code winner} が {@code from} から（同じ席ならツモで）基本点 {@code base} で和了したときの点の移動を {@code deltas} に書く。
   * 供託は和了者が受け取る。
   */
  static void settle(int base, int winner, int from, int dealer, int honba, int kyotaku, int[] deltas) {
    Arrays.fill(deltas, 0);
    if (from == winner) {
      Settlement.tsumo(VARIANT, base, winner, dealer, honba, RoundOutcome.NO_SEAT, deltas);
    } else {
      Settlement.ron(VARIANT, base, winner, from, dealer, honba, RoundOutcome.NO_SEAT, deltas);
    }
    deltas[winner] += kyotaku * Variant.DEPOSIT;
  }

  private static boolean above(int scoreA, int seatA, int scoreB, int seatB) {
    return scoreA > scoreB || scoreA == scoreB && seatA < seatB;
  }

  private static int bits(float value) {
    return Float.floatToIntBits(value);
  }
}
