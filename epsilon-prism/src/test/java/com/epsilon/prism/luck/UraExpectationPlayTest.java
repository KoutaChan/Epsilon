package com.epsilon.prism.luck;

import com.epsilon.grp.Phi;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.noise.UraSettings;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 自己対局の局末ごとに、局末の値を数え上げの基準と比べる（四人打ちと三人打ち）。立直の和了では、期待値・分散・基本点の組の数が基準と
 * 一致する。裏ドラ表示牌の数はめくった表示牌の数と、槓の有無は打ち手が選んだ手と一致する。ほかの局末では、期待値は実際の値そのもの。
 */
public class UraExpectationPlayTest {

  private static final double EPSILON = 1e-12;

  private static final int HAND_ENDS = 3_000;

  private static final UraSettings REALIZED = new UraSettings.Realized(0.05), EXPECTED = new UraSettings.Expected();

  @Test
  public void fourPlayerHandEndsMatchTheReference() {
    Tally tally = play(Variant.TENHOU_4P, 41L, HAND_ENDS, null);
    Assert.assertTrue(tally.compared > 100, tally.toString());
    Assert.assertTrue(tally.kanCompared > 0, tally.toString());
  }

  @Test
  public void threePlayerHandEndsMatchTheReference() {
    Tally tally = play(Variant.TENHOU_3P, 43L, HAND_ENDS, null);
    Assert.assertTrue(tally.compared > 100, tally.toString());
    Assert.assertTrue(tally.kanCompared > 0, tally.toString());
  }

  /** テストで使う Φ。四人打ちは点数のなめらかな関数、三人打ちはなめらかな関数の σ が無いので点数の順位で決める。 */
  static Phi phi(Variant variant) {
    return variant.seats() == 4
        ? Phi.scoreSmooth(variant, RankUtility.TENHOU)
        : Phi.scoreRank(variant, RankUtility.TENHOU);
  }

  /** 比べた局末の数と、立直の和了者（局の結果で 1 人目）について集めた「実際の値 − 期待値」。 */
  static final class Tally {
    long handEnds;
    long riichiWins;
    long compared;
    long jointCompared;
    long kanCompared;

    /** 槓の有無（0 は無し、1 は有り）ごとの、行の数・差の和・差の 2 乗の和。 */
    final double[][] winner = new double[2][3];

    double z(boolean kan) {
      double[] cell = winner[kan ? 1 : 0];
      double n = cell[0];
      double mean = cell[1] / n;
      double spread = cell[2] / n - mean * mean;
      return mean / Math.sqrt(spread / (n - 1));
    }

    @Override
    public String toString() {
      return "handEnds="
          + handEnds
          + " riichiWins="
          + riichiWins
          + " compared="
          + compared
          + " joint="
          + jointCompared
          + " kan="
          + kanCompared
          + " rows(no kan)="
          + winner[0][0]
          + " rows(kan)="
          + winner[1][0];
    }
  }

  /**
   * 自己対局を打ち、局末ごとに局末の値を求めて確かめる。
   *
   * @param audit 立直の和了者の行を数える監査。null なら数えない
   */
  static Tally play(Variant variant, long seed, int handEnds, UraAudit audit) {
    Phi phi = phi(variant);
    UraExpectation values = new UraExpectation(phi);
    UraExpectation realizedOnly = new UraExpectation(phi);
    UraReference reference = new UraReference(phi);
    double[] next = new double[variant.seats()];
    double[] out = new double[variant.seats()];
    Tally tally = new Tally();
    new UraGames(variant)
        .play(
            seed,
            handEnds,
            (engine, kan) -> {
              tally.handEnds++;
              check(engine, kan, phi, values, realizedOnly, reference, next, out, tally);
              if (audit != null) {
                audit.handEnd(values);
                if (values.averaged()) {
                  // 局末ごとに別の局として数える
                  audit.rows(
                      values, tally.handEnds, 0, UraAuditTest.rows(DecisionClass.END, firstRiichiWinner(engine)));
                }
              }
            });
    return tally;
  }

  private static void check(
      Engine engine,
      boolean kan,
      Phi phi,
      UraExpectation values,
      UraExpectation realizedOnly,
      UraReference reference,
      double[] next,
      double[] out,
      Tally tally) {
    FullView full = engine.full();
    RoundEnd end = engine.roundEnd();
    TableState start = engine.handStart();
    int seats = next.length;
    boolean averaged = values.evaluate(full, end, start, null, true);
    phi.values(null, end.next(), next);
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(values.actual(seat), next[seat]);
    }
    Assert.assertEquals(averaged, values.riichiWin());
    Assert.assertEquals(values.averaged(), averaged);
    Assert.assertFalse(realizedOnly.evaluate(full, end, start, null, false));
    Assert.assertEquals(realizedOnly.riichiWin(), values.riichiWin());
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(realizedOnly.actual(seat), next[seat]);
      Assert.assertEquals(realizedOnly.expected(seat), next[seat]);
      Assert.assertEquals(realizedOnly.variance(seat), 0.0);
    }
    values.values(REALIZED, out);
    assertSame(out, next);
    if (!averaged) {
      Assert.assertFalse(values.kan());
      Assert.assertEquals(values.groups(), 0);
      values.values(EXPECTED, out);
      assertSame(out, next);
      for (int seat = 0; seat < seats; seat++) {
        Assert.assertEquals(values.variance(seat), 0.0);
      }
      return;
    }
    tally.riichiWins++;
    Assert.assertEquals(values.kan(), kan, "kan in " + end.outcome());
    values.values(EXPECTED, out);
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(out[seat], values.expected(seat));
    }
    Assert.assertThrows(IllegalStateException.class, () -> realizedOnly.values(EXPECTED, out));
    int winner = firstRiichiWinner(engine);
    double difference = values.actual(winner) - values.expected(winner);
    double[] cell = tally.winner[kan ? 1 : 0];
    cell[0]++;
    cell[1] += difference;
    cell[2] += difference * difference;
    if (!reference.compute(full, end, start)) {
      return;
    }
    Assert.assertEquals(reference.uraEvents, full.seat(0).indicatorCount());
    Assert.assertEquals(reference.uraEvents, 1 + reference.indicatorEvents);
    Assert.assertTrue(reference.normalized());
    for (int v = 0; v < reference.varyingCount; v++) {
      Assert.assertEquals(reference.realizedUra(v), end.score(reference.varying(v)).ura());
    }
    Assert.assertEquals(values.groups(), reference.groups(), end.outcome().toString());
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(reference.realized[seat], next[seat], EPSILON, end.outcome().toString());
      Assert.assertEquals(values.expected(seat), reference.expected[seat], EPSILON, end.outcome().toString());
      Assert.assertEquals(values.variance(seat), reference.variance[seat], EPSILON, end.outcome().toString());
    }
    tally.compared++;
    if (reference.varyingCount == 2) {
      tally.jointCompared++;
    }
    if (kan) {
      tally.kanCompared++;
    }
  }

  private static void assertSame(double[] actual, double[] expected) {
    for (int seat = 0; seat < expected.length; seat++) {
      Assert.assertEquals(actual[seat], expected[seat]);
    }
  }

  /** 局の結果に並ぶ順で、最初の立直の和了者の席。 */
  static int firstRiichiWinner(Engine engine) {
    RoundEnd end = engine.roundEnd();
    for (int i = 0; i < end.outcome().winnerCount(); i++) {
      int winner = end.outcome().winner(i);
      if (engine.full().seat(winner).riichi(winner) != RiichiState.NONE) {
        return winner;
      }
    }
    throw new AssertionError("No riichi winner in " + end.outcome());
  }
}
