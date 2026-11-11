package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import java.util.Optional;
import java.util.SplittableRandom;
import org.testng.annotations.Test;

/**
 * 止め方ごとに、判定する時点と判定の結果を確かめる。良い・悪いは互角より先に判定する。標準誤差が 0 なら区間は平均の 1 点になり、
 * 互角の幅が正なら互角、幅が 0 なら最後の時点まで打つ。精度の止め方は標準誤差が目標以下になると止まり、固定の止め方は最後に 1 回だけ
 * 判定する。
 */
public class StoppingTest {

  private static final double FIXED_99 = 2.5758293035489004;

  private static final Sequential SEQUENTIAL = new Sequential(0.01, 3, 512);
  private static final StopRule.Test EVEN_TEST = new StopRule.Test(SEQUENTIAL, 0.01, 0.01);

  @Test
  public void aFixedRuleDecidesOnceAtTheEndWithTheFixedLevel() {
    Stopping stopping = Stopping.of(new StopRule.Fixed(0.01), 1000);
    assertEquals(stopping.looks(), 1);
    assertEquals(stopping.wallsAt(0), 1000);
    assertEquals(stopping.critical(0), FIXED_99, 1e-9);
    assertDecision(stopping.decide(0, 0.01, 0.003), Verdict.BETTER, Stopped.LIMIT);
    assertDecision(stopping.decide(0, -0.01, 0.003), Verdict.WORSE, Stopped.LIMIT);
    Stopping.Decision undecided = assertDecision(stopping.decide(0, 0.001, 0.003), Verdict.UNDECIDED, Stopped.LIMIT);
    assertEquals(undecided.estimate().low(), 0.001 - FIXED_99 * 0.003, 1e-12);
    assertEquals(undecided.estimate().high(), 0.001 + FIXED_99 * 0.003, 1e-12);
  }

  @Test
  public void aPrecisionRuleStopsOnceTheErrorReachesTheTarget() {
    Stopping stopping = Stopping.of(new StopRule.Precision(0.005, 0.01, 512), 4096);
    assertEquals(stopping.looks(), 8);
    for (int look = 0; look < 8; look++) {
      assertEquals(stopping.wallsAt(look), (look + 1) * 512);
      assertEquals(stopping.critical(look), FIXED_99, 1e-9);
    }
    assertTrue(stopping.decide(0, 0.05, 0.006).isEmpty());
    assertDecision(stopping.decide(1, 0.0, 0.005), Verdict.UNDECIDED, Stopped.PRECISE);
    assertDecision(stopping.decide(2, 0.02, 0.004), Verdict.BETTER, Stopped.PRECISE);
    assertDecision(stopping.decide(3, -0.02, 0.004), Verdict.WORSE, Stopped.PRECISE);
    assertDecision(stopping.decide(7, 0.05, 0.006), Verdict.BETTER, Stopped.LIMIT);
    assertDecision(stopping.decide(7, 0.0, 0.006), Verdict.UNDECIDED, Stopped.LIMIT);
    assertThrows(IllegalArgumentException.class, () -> Stopping.of(new StopRule.Precision(0.005, 0.01, 512), 4000));
  }

  @Test
  public void aTestRuleUsesTheSequentialBoundaries() {
    Stopping stopping = Stopping.of(EVEN_TEST, 4096);
    double[] critical = SEQUENTIAL.boundaries(4096);
    assertEquals(stopping.looks(), critical.length);
    for (int look = 0; look < critical.length; look++) {
      assertEquals(stopping.wallsAt(look), (look + 1) * 512);
      assertEquals(stopping.critical(look), critical[look], 0);
    }
    assertThrows(IllegalArgumentException.class, () -> Stopping.of(EVEN_TEST, 1000));
  }

  /** 区間が 0 より上にあれば、互角の幅の内側に収まっていても良いと判定する。0 より下にあれば悪いと判定する。 */
  @Test
  public void betterAndWorseComeBeforeEven() {
    Stopping stopping = Stopping.of(EVEN_TEST, 4096);
    Stopping.Decision better = assertDecision(stopping.decide(0, 0.006, 0.0008), Verdict.BETTER, Stopped.DECIDED);
    assertTrue(better.estimate().low() > 0 && better.estimate().high() < 0.01, better.toString());
    Stopping.Decision worse = assertDecision(stopping.decide(0, -0.006, 0.0008), Verdict.WORSE, Stopped.DECIDED);
    assertTrue(worse.estimate().high() < 0 && worse.estimate().low() > -0.01, worse.toString());
    assertDecision(stopping.decide(0, 0.001, 0.001), Verdict.EVEN, Stopped.DECIDED);
  }

  @Test
  public void aTestRuleWaitsUntilTheLastLookToGiveUp() {
    Stopping stopping = Stopping.of(EVEN_TEST, 4096);
    for (int look = 0; look < 7; look++) {
      assertTrue(stopping.decide(look, 0.004, 0.003).isEmpty(), "look " + look);
    }
    assertDecision(stopping.decide(7, 0.004, 0.003), Verdict.UNDECIDED, Stopped.LIMIT);
  }

  /** 互角の幅は、下側と上側で別々に効く。 */
  @Test
  public void evenMarginsApplyToTheirOwnSide() {
    Stopping lower = Stopping.of(new StopRule.Test(SEQUENTIAL, 0.02, 0.01), 4096);
    Stopping upper = Stopping.of(new StopRule.Test(SEQUENTIAL, 0.01, 0.02), 4096);
    assertDecision(lower.decide(0, -0.004, 0.003), Verdict.EVEN, Stopped.DECIDED);
    assertTrue(upper.decide(0, -0.004, 0.003).isEmpty());
    assertDecision(upper.decide(0, 0.004, 0.003), Verdict.EVEN, Stopped.DECIDED);
    assertTrue(lower.decide(0, 0.004, 0.003).isEmpty());
  }

  /** 同じ値ばかりの牌山では標準誤差がちょうど 0 になり、区間は平均の 1 点になる。 */
  @Test
  public void aZeroErrorIsAPointInterval() {
    WallMean same = constant(512, 0.0);
    assertEquals(same.error(), 0.0);
    Stopping.Decision even = assertDecision(Stopping.of(EVEN_TEST, 4096).decide(0, same), Verdict.EVEN, Stopped.DECIDED);
    assertEquals(even.estimate(), new Estimate(0, 0, 0, 0));

    Stopping noMargin = Stopping.of(new StopRule.Test(SEQUENTIAL, 0, 0), 4096);
    for (int look = 0; look < 7; look++) {
      assertTrue(noMargin.decide(look, constant((look + 1) * 512, 0.0)).isEmpty(), "look " + look);
    }
    assertDecision(noMargin.decide(7, constant(4096, 0.0)), Verdict.UNDECIDED, Stopped.LIMIT);

    WallMean ahead = constant(512, 0.1);
    assertEquals(ahead.error(), 0.0);
    assertDecision(noMargin.decide(0, ahead), Verdict.BETTER, Stopped.DECIDED);
    assertDecision(noMargin.decide(0, -0.1, 0), Verdict.WORSE, Stopped.DECIDED);
    assertDecision(
        Stopping.of(new StopRule.Precision(0, 0.01, 512), 4096).decide(0, same), Verdict.UNDECIDED, Stopped.PRECISE);
  }

  @Test
  public void decisionsNeedTheWallsOfTheirLookAndAValidEstimate() {
    Stopping stopping = Stopping.of(EVEN_TEST, 4096);
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(1, constant(512, 0.0)));
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(8, 0, 0.01));
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(-1, 0, 0.01));
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(0, Double.NaN, 0.01));
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(0, 0, -0.01));
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(0, 0, Double.POSITIVE_INFINITY));
    assertThrows(IllegalArgumentException.class, () -> Stopping.of(EVEN_TEST, 0));
    assertThrows(NullPointerException.class, () -> Stopping.of(null, 512));
    assertThrows(IllegalArgumentException.class, () -> new StopRule.Fixed(0));
    assertThrows(IllegalArgumentException.class, () -> new StopRule.Precision(-0.001, 0.01, 512));
    assertThrows(IllegalArgumentException.class, () -> new StopRule.Precision(0.005, 0.01, 0));
    assertThrows(IllegalArgumentException.class, () -> new StopRule.Test(SEQUENTIAL, -0.01, 0.01));
    assertThrows(IllegalArgumentException.class, () -> new StopRule.Test(SEQUENTIAL, 0.01, Double.NaN));
    assertThrows(NullPointerException.class, () -> new StopRule.Test(null, 0.01, 0.01));
  }

  @Test
  public void anEstimateNegatesIntoTheOtherDirection() {
    Estimate estimate = Estimate.of(0.02, 0.005, 2);
    assertEquals(estimate.low(), 0.01, 1e-15);
    assertEquals(estimate.high(), 0.03, 1e-15);
    Estimate negated = estimate.negate();
    assertEquals(negated, new Estimate(-0.02, 0.005, -estimate.high(), -estimate.low()));
    assertEquals(negated.negate(), estimate);
  }

  /** 平均と標準誤差は、データを 2 回走査して求めた値と一致する。分散は不偏分散。 */
  @Test
  public void aWallMeanMatchesTwoPasses() {
    SplittableRandom random = new SplittableRandom(11);
    double[] values = new double[5000];
    WallMean mean = new WallMean();
    for (int i = 0; i < values.length; i++) {
      values[i] = 3 + 0.2 * random.nextGaussian();
      mean.add(values[i]);
    }
    double sum = 0;
    for (double value : values) {
      sum += value;
    }
    double average = sum / values.length;
    double squares = 0;
    for (double value : values) {
      squares += (value - average) * (value - average);
    }
    assertEquals(mean.walls(), values.length);
    assertEquals(mean.mean(), average, 1e-12);
    assertEquals(mean.error(), Math.sqrt(squares / (values.length - 1) / values.length), 1e-12);
  }

  @Test
  public void aWallMeanNeedsWallsAndFiniteValues() {
    WallMean mean = new WallMean();
    assertThrows(IllegalStateException.class, mean::mean);
    mean.add(1);
    assertEquals(mean.mean(), 1.0);
    assertThrows(IllegalStateException.class, mean::error);
    assertThrows(IllegalArgumentException.class, () -> mean.add(Double.NaN));
    assertThrows(IllegalArgumentException.class, () -> mean.add(Double.NEGATIVE_INFINITY));
    assertEquals(mean.walls(), 1);
  }

  private static Stopping.Decision assertDecision(Optional<Stopping.Decision> decision, Verdict verdict, Stopped stopped) {
    assertFalse(decision.isEmpty(), "no decision, expected " + verdict + " " + stopped);
    assertEquals(decision.get().verdict(), verdict, decision.get().toString());
    assertEquals(decision.get().stopped(), stopped, decision.get().toString());
    return decision.get();
  }

  private static WallMean constant(int walls, double value) {
    WallMean mean = new WallMean();
    for (int wall = 0; wall < walls; wall++) {
      mean.add(value);
    }
    return mean;
  }
}
