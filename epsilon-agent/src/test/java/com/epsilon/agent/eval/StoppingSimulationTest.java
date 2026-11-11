package com.epsilon.agent.eval;

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
 * 合成の牌山で止め方ごとに 2 万回の決闘を打ち、誤った判定の割合を確かめる。牌山の値は「真の差 ＋ 標準偏差 × 平均 0・分散 1 の乱数」で、
 * 乱数の分布は正規分布、3 点だけの有界な分布、歪んだ分布（形 4 のガンマ分布、歪度 1）の 3 つ。誤って良い・悪い・互角と言う割合と、
 * 精度の止め方で区間が真の差を片側に外す割合は、どれも有意水準の半分 ＋ 3 SE 以内（SE は、割合が有意水準の半分の二項分布の標準誤差）。誤りは少なくても
 * なかなか結論を出さない止め方を通さないように、良い・互角を正しく言える割合の下限も確かめる。
 *
 * <p>区間は正規近似なので、片側の誤りは、歪度 / √(牌山の数) にほぼ比例して増える。歪度 2（指数分布）では、1,600 牌山で止まる精度の
 * 止め方で区間が下側に外れる割合が有意水準の半分の 1.6 倍になり、この上限を越える。
 */
public class StoppingSimulationTest {

  private static final Logger log = LoggerFactory.getLogger(StoppingSimulationTest.class);

  private static final int DUELS = 20_000;
  private static final double ALPHA = 0.01;
  private static final double SIGMA = 0.2;
  private static final double MARGIN = 0.01;
  private static final int WALLS = 4096;
  private static final int CHECK_EVERY = 512;

  /** 誤った判定の割合の上限: 有意水準の半分 ＋ 3 SE。 */
  private static final double FALSE_RATE =
      ALPHA / 2 + 3 * Math.sqrt(ALPHA / 2 * (1 - ALPHA / 2) / DUELS);

  private static final StopRule TEST = new StopRule.Test(new Sequential(ALPHA, 3, CHECK_EVERY), MARGIN, MARGIN);
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
        "STOPPING-SIM precision {}: miss below {} above {}, precise {}", shape, missBelow, missAbove, precise);
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
  private static List<Stopping.Decision> duels(StopRule rule, int walls, Shape shape, double truth, long stream) {
    Stopping stopping = Stopping.of(rule, walls);
    long base = (stream * 31 + shape.ordinal()) * 0x9E3779B97F4A7C15L;
    return IntStream.range(0, DUELS).parallel().mapToObj(duel -> duel(stopping, shape, truth, base + duel)).toList();
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
