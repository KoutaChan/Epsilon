package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SplittableRandom;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.DataProvider;
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
    Stopping.Decision undecided =
        assertDecision(stopping.decide(0, 0.001, 0.003), Verdict.UNDECIDED, Stopped.LIMIT);
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
    assertThrows(
        IllegalArgumentException.class,
        () -> Stopping.of(new StopRule.Precision(0.005, 0.01, 512), 4000));
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
    Stopping.Decision better =
        assertDecision(stopping.decide(0, 0.006, 0.0008), Verdict.BETTER, Stopped.DECIDED);
    assertTrue(better.estimate().low() > 0 && better.estimate().high() < 0.01, better.toString());
    Stopping.Decision worse =
        assertDecision(stopping.decide(0, -0.006, 0.0008), Verdict.WORSE, Stopped.DECIDED);
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
    Stopping.Decision even =
        assertDecision(Stopping.of(EVEN_TEST, 4096).decide(0, same), Verdict.EVEN, Stopped.DECIDED);
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
        Stopping.of(new StopRule.Precision(0, 0.01, 512), 4096).decide(0, same),
        Verdict.UNDECIDED,
        Stopped.PRECISE);
  }

  @Test
  public void decisionsNeedTheWallsOfTheirLookAndAValidEstimate() {
    Stopping stopping = Stopping.of(EVEN_TEST, 4096);
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(1, constant(512, 0.0)));
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(8, 0, 0.01));
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(-1, 0, 0.01));
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(0, Double.NaN, 0.01));
    assertThrows(IllegalArgumentException.class, () -> stopping.decide(0, 0, -0.01));
    assertThrows(
        IllegalArgumentException.class, () -> stopping.decide(0, 0, Double.POSITIVE_INFINITY));
    assertThrows(IllegalArgumentException.class, () -> Stopping.of(EVEN_TEST, 0));
    assertThrows(NullPointerException.class, () -> Stopping.of(null, 512));
    assertThrows(IllegalArgumentException.class, () -> new StopRule.Fixed(0));
    assertThrows(IllegalArgumentException.class, () -> new StopRule.Precision(-0.001, 0.01, 512));
    assertThrows(IllegalArgumentException.class, () -> new StopRule.Precision(0.005, 0.01, 0));
    assertThrows(IllegalArgumentException.class, () -> new StopRule.Test(SEQUENTIAL, -0.01, 0.01));
    assertThrows(
        IllegalArgumentException.class, () -> new StopRule.Test(SEQUENTIAL, 0.01, Double.NaN));
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

  private static Stopping.Decision assertDecision(
      Optional<Stopping.Decision> decision, Verdict verdict, Stopped stopped) {
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

  private static final Logger log = LoggerFactory.getLogger(StoppingTest.class);

  private static final int DUELS = 20_000;
  private static final double ALPHA = 0.01;
  private static final double SIGMA = 0.2;
  private static final double MARGIN = 0.01;
  private static final int WALLS = 4096;
  private static final int CHECK_EVERY = 512;

  /** 誤った判定の割合の上限: 有意水準の半分 ＋ 3 SE。 */
  private static final double FALSE_RATE =
      ALPHA / 2 + 3 * Math.sqrt(ALPHA / 2 * (1 - ALPHA / 2) / DUELS);

  private static final StopRule TEST =
      new StopRule.Test(new Sequential(ALPHA, 3, CHECK_EVERY), MARGIN, MARGIN);
  private static final StopRule FIXED = new StopRule.Fixed(ALPHA);

  /** 精度の止め方。牌山 4,500 個ほどで標準誤差が目標に届く。 */
  private static final StopRule PRECISION = new StopRule.Precision(0.003, ALPHA, 128);

  private static final int PRECISION_WALLS = 8192;

  /** 牌山の値に足す乱数の分布。平均 0・分散 1。 */
  enum Shape {
    NORMAL {
      @Override
      double draw(SplittableRandom random) {
        return random.nextGaussian();
      }
    },
    /** −√2・0・√2 を、それぞれ 1/4・1/2・1/4 の確率で出す。 */
    BOUNDED {
      @Override
      double draw(SplittableRandom random) {
        int quarter = random.nextInt(4);
        return quarter == 0 ? -Math.sqrt(2) : quarter == 1 ? Math.sqrt(2) : 0;
      }
    },
    /** 指数分布 4 つの和から 4 を引いて 2 で割ったもの（形 4 のガンマ分布）。右に裾が長い。 */
    SKEWED {
      @Override
      double draw(SplittableRandom random) {
        double sum = 0;
        for (int i = 0; i < 4; i++) {
          sum -= Math.log(1 - random.nextDouble());
        }
        return (sum - 4) / 2;
      }
    };

    abstract double draw(SplittableRandom random);
  }

  @DataProvider
  public static Object[][] shapes() {
    return new Object[][] {{Shape.NORMAL}, {Shape.BOUNDED}, {Shape.SKEWED}};
  }

  /** 真の差が 0 なら、良い・悪いと言う割合はどちらも上限以内。互角と言う割合は 3 割以上。 */
  @Test(groups = "slow", dataProvider = "shapes", timeOut = 1_800_000)
  public void aTestRuleRarelyCallsANullDifference(Shape shape) {
    Map<Verdict, Double> rates = rates(duels(TEST, WALLS, shape, 0, 1));
    log("test, truth 0", shape, rates);
    assertTrue(rates.get(Verdict.BETTER) <= FALSE_RATE, shape + ": " + rates);
    assertTrue(rates.get(Verdict.WORSE) <= FALSE_RATE, shape + ": " + rates);
    assertTrue(rates.get(Verdict.EVEN) >= 0.3, shape + " rarely finds an even duel: " + rates);
  }

  /** 真の差が互角の幅のちょうど端にあれば、互角と言う割合も、逆向きに言う割合も上限以内。 */
  @Test(groups = "slow", dataProvider = "shapes", timeOut = 1_800_000)
  public void aTestRuleRarelyCallsAnEdgeDifferenceEven(Shape shape) {
    Map<Verdict, Double> above = rates(duels(TEST, WALLS, shape, MARGIN, 2));
    Map<Verdict, Double> below = rates(duels(TEST, WALLS, shape, -MARGIN, 3));
    log("test, truth +margin", shape, above);
    log("test, truth -margin", shape, below);
    assertTrue(above.get(Verdict.EVEN) <= FALSE_RATE, shape + ": " + above);
    assertTrue(below.get(Verdict.EVEN) <= FALSE_RATE, shape + ": " + below);
    assertTrue(above.get(Verdict.WORSE) <= FALSE_RATE, shape + ": " + above);
    assertTrue(below.get(Verdict.BETTER) <= FALSE_RATE, shape + ": " + below);
  }

  /** 真の差が互角の幅の 2 倍なら、95% 以上の決闘で良いと言う。 */
  @Test(groups = "slow", dataProvider = "shapes", timeOut = 1_800_000)
  public void aTestRuleFindsAClearDifference(Shape shape) {
    Map<Verdict, Double> rates = rates(duels(TEST, WALLS, shape, 2 * MARGIN, 4));
    log("test, truth 2 margins", shape, rates);
    assertTrue(rates.get(Verdict.BETTER) >= 0.95, shape + ": " + rates);
  }

  @Test(groups = "slow", dataProvider = "shapes", timeOut = 1_800_000)
  public void aFixedRuleRarelyCallsANullDifference(Shape shape) {
    Map<Verdict, Double> rates = rates(duels(FIXED, WALLS, shape, 0, 5));
    log("fixed, truth 0", shape, rates);
    assertTrue(rates.get(Verdict.BETTER) <= FALSE_RATE, shape + ": " + rates);
    assertTrue(rates.get(Verdict.WORSE) <= FALSE_RATE, shape + ": " + rates);
  }

  /** 精度で止めた区間が真の差を外す割合は、上側・下側とも上限以内。99% 以上の決闘は、標準誤差が目標に届いて止まる。 */
  @Test(groups = "slow", dataProvider = "shapes", timeOut = 1_800_000)
  public void aPrecisionRuleCoversTheTruth(Shape shape) {
    double truth = 0.003;
    List<Stopping.Decision> decisions = duels(PRECISION, PRECISION_WALLS, shape, truth, 6);
    double missBelow = rate(decisions, decision -> decision.estimate().high() < truth);
    double missAbove = rate(decisions, decision -> decision.estimate().low() > truth);
    double precise = rate(decisions, decision -> decision.stopped() == Stopped.PRECISE);
    log.info(
        "STOPPING-SIM precision {}: miss below {} above {}, precise {}",
        shape,
        missBelow,
        missAbove,
        precise);
    assertTrue(missBelow <= FALSE_RATE, shape + ": " + missBelow);
    assertTrue(missAbove <= FALSE_RATE, shape + ": " + missAbove);
    assertTrue(precise >= 0.99, shape + ": " + precise);
  }

  /**
   * 決闘を打つ。牌山 0 から順に値を足し、次の判定の時点までの牌山がそろうたびに判定する。
   *
   * @param walls 計画の牌山の数
   * @param truth 真の差（候補が良い向きが正）
   * @param stream 乱数の種の系列。テストごとに別の値にする
   */
  private static List<Stopping.Decision> duels(
      StopRule rule, int walls, Shape shape, double truth, long stream) {
    Stopping stopping = Stopping.of(rule, walls);
    long base = (stream * 31 + shape.ordinal()) * 0x9E3779B97F4A7C15L;
    return IntStream.range(0, DUELS)
        .parallel()
        .mapToObj(duel -> duel(stopping, shape, truth, base + duel))
        .toList();
  }

  private static Stopping.Decision duel(Stopping stopping, Shape shape, double truth, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    WallMean values = new WallMean();
    for (int look = 0; ; look++) {
      while (values.walls() < stopping.wallsAt(look)) {
        values.add(truth + SIGMA * shape.draw(random));
      }
      Optional<Stopping.Decision> decision = stopping.decide(look, values);
      if (decision.isPresent()) {
        return decision.get();
      }
    }
  }

  private static Map<Verdict, Double> rates(List<Stopping.Decision> decisions) {
    Map<Verdict, Double> rates = new EnumMap<>(Verdict.class);
    for (Verdict verdict : Verdict.values()) {
      rates.put(verdict, rate(decisions, decision -> decision.verdict() == verdict));
    }
    return rates;
  }

  private static double rate(List<Stopping.Decision> decisions, Predicate<Stopping.Decision> test) {
    return (double) decisions.stream().filter(test).count() / decisions.size();
  }

  private static void log(String setting, Shape shape, Map<Verdict, Double> rates) {
    log.info("STOPPING-SIM {} {}: {} (bound {})", setting, shape, rates, FALSE_RATE);
  }
}
