package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import java.util.Arrays;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 席のビューだけから求めるルールの関数が、エンジンの判定と合うことを確かめる。手をランダムに選ぶ自己対局の、全ての判断の時点で調べる。
 *
 * <p>合法な鳴きの面子（{@code Legal.callMeld}）は、コピーしたエンジンでその鳴きを通した後の、鳴いた席の最後の面子と同じになる。
 * 鳴いた後の打牌（{@code Legal.postCallDiscards}）は、鳴いた後に打牌を選ぶ判断の合法手と同じ並びになる。和了で終わった局では、ビューから
 * 求めた和了の点（WinSettlementProjection）の翻・符・役満が、局末の点から裏ドラを除いた値と同じで、包の席も局の結果と同じになる。
 * 手番では、実際に引いた種類で求めた仮のツモの点（{@code FullView.tsumoScore}）が、ツモ和了が合法なときだけ書かれ、ビューから求めた
 * 点と同じになる（引いた牌が赤なら、その赤だけを数えない）。和了が起きやすいように、判断する席ごとに半分の確率で最初の合法手（和了・槓・
 * 鳴きが先に並ぶ）を選ぶ。
 */
public class SeatViewRulesTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int STEPS = 500_000;

  private final Engine engine = new Engine(VARIANT);
  private final RuleWorkspace ws = new RuleWorkspace(VARIANT);
  private final Random random = new Random(17);
  private final int[] choices = new int[VARIANT.seats() - 1];
  private final short[] discards = new short[Decisions.MAX_LEGAL];
  private final WinSettlementProjection[] projections = new WinSettlementProjection[VARIANT.seats()];
  private final int[] winCodes = new int[VARIANT.seats()];
  private final WinScore tsumoScore = new WinScore();
  private int calls;
  private int openKans;
  private int tsumo;
  private int ron;
  private int tsumoScores;

  @Test
  public void rulesOfTheSeatViewMatchTheEngine() {
    for (int seat = 0; seat < projections.length; seat++) {
      projections[seat] = new WinSettlementProjection(VARIANT);
    }
    Boundary boundary = Boundary.GAME_END;
    for (int step = 0; step < STEPS; step++) {
      boundary =
          switch (boundary) {
            case DECISIONS -> decide();
            case ROUND_END -> engine.advance(ws);
            case GAME_END -> {
              engine.startGame(random.nextLong(), random.nextBoolean() ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER);
              yield engine.advance(ws);
            }
          };
    }
    Assert.assertTrue(calls > 1_000, "calls " + calls);
    Assert.assertTrue(openKans > 0, "open kans " + openKans);
    Assert.assertTrue(tsumo > 0, "tsumo " + tsumo);
    Assert.assertTrue(ron > 0, "ron " + ron);
    Assert.assertTrue(tsumoScores > 50, "tsumo scores " + tsumoScores);
  }

  /** 判断する席ごとに、鳴きはコピーしたエンジンで確かめ、和了はビューから点を求めておいてから手を選び、局を進める。和了で終わった局は、求めた点を結果と照合する。 */
  private Boundary decide() {
    Decisions decisions = engine.decisions();
    Arrays.fill(winCodes, -1);
    for (int slot = 0; slot < decisions.size(); slot++) {
      int seat = decisions.seat(slot);
      SeatView view = engine.view(seat);
      for (int index = 0; index < decisions.legalCount(slot); index++) {
        int code = decisions.legal(slot, index);
        switch (Action.of(code).type()) {
          case CHI, PON -> checkCall(slot, index, Legal.postCallDiscards(view, code, discards));
          case DAIMINKAN -> checkCall(slot, index, -1);
          case TSUMO_AGARI, RON_AGARI -> {
            projections[seat].load(view, code, ws);
            winCodes[seat] = code;
          }
          default -> {}
        }
      }
      if (decisions.kind(slot) == EngineDecisionKind.TURN) {
        checkTsumo(seat, winCodes[seat] >= 0);
      }
      choices[slot] = random.nextBoolean() ? 0 : random.nextInt(decisions.legalCount(slot));
    }
    Boundary reached = engine.commit(choices, ws);
    if (reached != Boundary.DECISIONS && engine.roundEnd().outcome().kind().win()) {
      checkWins(engine.roundEnd());
    }
    return reached;
  }

  /**
   * 判断する席の 1 つが鳴き、ほかの席が見送ると、鳴いた席の最後の面子は {@code Legal.callMeld} が返す面子と同じになることを確かめる。
   * チー・ポンなら、鳴いた後に打牌を選ぶ判断の合法手が、{@code Legal.postCallDiscards} が並べた打牌と同じになることも確かめる。
   *
   * @param slot 鳴く席（判断する席の何番目か）
   * @param index 鳴く手（合法手の何番目か）
   * @param count 鳴いた後に打てる牌の数。大明槓なら −1
   */
  private void checkCall(int slot, int index, int count) {
    Decisions decisions = engine.decisions();
    int seat = decisions.seat(slot);
    int code = decisions.legal(slot, index);
    int meld = Legal.callMeld(engine.view(seat), code);
    int[] passes = new int[decisions.size()];
    for (int other = 0; other < passes.length; other++) {
      passes[other] = other == slot ? index : decisions.legalCount(other) - 1;
    }
    Engine copy = engine.copy();
    Assert.assertEquals(copy.commit(passes, ws), Boundary.DECISIONS);
    SeatView caller = copy.view(seat);
    Assert.assertEquals(caller.meld(seat, caller.meldCount(seat) - 1), meld, Action.of(code).toString());
    if (count < 0) {
      openKans++;
      return;
    }
    Decisions after = copy.decisions();
    Assert.assertEquals(after.kind(0), EngineDecisionKind.POST_CALL_DAHAI);
    Assert.assertEquals(after.seat(0), decisions.seat(slot));
    short[] expected = new short[after.legalCount(0)];
    for (int legal = 0; legal < expected.length; legal++) {
      expected[legal] = (short) after.legal(0, legal);
    }
    Assert.assertEquals(Arrays.copyOf(discards, count), expected, Action.of(decisions.legal(slot, index)).toString());
    calls++;
  }

  /**
   * 手番の席が実際に引いた種類で求めた仮のツモの点は、ツモ和了が合法なときだけ書かれ、ビューから求めた点と同じになる。引いた牌が赤なら、
   * 仮のツモの点はその赤を数えない（役満なら赤はもともと数えない）。
   *
   * @param seat 手番の席
   * @param legal ツモ和了が合法で、ビューから求めた点を席ごとの WinSettlementProjection に読み込んであるか
   */
  private void checkTsumo(int seat, boolean legal) {
    TurnEvent draw = engine.view(seat).last();
    Assert.assertEquals(engine.full().tsumoScore(seat, draw.tile(), ws, tsumoScore), legal, "seat " + seat);
    if (!legal) {
      return;
    }
    WinScore projected = projections[seat].score();
    if (draw.red()) {
      Assert.assertEquals(tsumoScore.yakuman(), projected.yakuman(), projected.toString());
      Assert.assertEquals(tsumoScore.aka(), projected.yakuman() > 0 ? 0 : projected.aka() - 1, projected.toString());
    } else {
      Assert.assertEquals(tsumoScore.toString(), projected.toString());
    }
    tsumoScores++;
  }

  private void checkWins(RoundEnd end) {
    RoundOutcome outcome = end.outcome();
    for (int i = 0; i < outcome.winnerCount(); i++) {
      int winner = outcome.winner(i);
      Assert.assertTrue(winCodes[winner] >= 0, "seat " + winner + " won without a projected win");
      WinSettlementProjection projection = projections[winner];
      WinScore projected = projection.score();
      WinScore settled = end.score(i);
      Assert.assertEquals(projected.han(), settled.han() - settled.ura(), settled.toString());
      Assert.assertEquals(projected.fu(), settled.fu(), settled.toString());
      Assert.assertEquals(projected.yakuman(), settled.yakuman(), settled.toString());
      Assert.assertEquals(projection.pao(), outcome.pao(i));
      if (outcome.kind() == EndKind.TSUMO) {
        tsumo++;
      } else {
        ron++;
      }
    }
  }
}
