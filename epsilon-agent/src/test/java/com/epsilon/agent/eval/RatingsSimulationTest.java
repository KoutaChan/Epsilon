package com.epsilon.agent.eval;

import static org.testng.Assert.assertTrue;

import java.util.List;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 合成の系譜で尺度を 2,000 回当て直し、頑健な標準誤差と、外れた辺を見つけるしきい値が較正されているかを確かめる。系譜は点 40 個・
 * 牌山 1,024 個で、辺は 1 つ前とのラダー、8 つ前とのラダー、基準との辺の 3 種類（{@link SyntheticChain}）。
 */
public class RatingsSimulationTest {

  private static final Logger log = LoggerFactory.getLogger(RatingsSimulationTest.class);

  private static final int NODES = 40;
  private static final int WALLS = 1024;
  private static final int REPEATS = 2000;

  /**
   * 基準との辺が牌山を共有するとき、頑健な標準誤差の平均は点の強さ θ のばらつきの 0.92〜1.08 倍に入る。θ のずれの平均は、0 からその
   * 標準誤差の 4 倍以内に入る。
   */
  @Test(groups = "slow", timeOut = 1_800_000)
  public void robustErrorsMatchTheSpreadOnSharedWalls() {
    SyntheticChain.Spread[] spread = SyntheticChain.spread(NODES, WALLS, REPEATS, true, 50_000);
    for (int v = 1; v < NODES; v++) {
      SyntheticChain.Spread node = spread[v];
      double ratio = node.reported() / node.simulated();
      log.info(
          "RATINGS-SIM v{}: bias {} ({}), simulated {} robust {}",
          v,
          node.bias(),
          node.biasError(),
          node.simulated(),
          node.reported());
      assertTrue(ratio > 0.92 && ratio < 1.08, "v" + v + " ratio " + ratio);
      assertTrue(Math.abs(node.bias()) < 4 * node.biasError(), "v" + v + " bias " + node.bias());
    }
  }

  /**
   * 辺が全て独立で当てはめのモデルが正しいとき、外れた辺が 1 本でも出る当てはめの割合は 2.5% 以上、5% ＋ 3 SE 以下に入る
   * （しきい値は Bonferroni 補正なので、割合は 5% より少し下に出る）。|z| が 3 を超える辺の数の平均は期待値の 0.85〜1.15 倍で、
   * z² の平均は 1 から 0.03 以内。
   */
  @Test(groups = "slow", timeOut = 1_800_000)
  public void theOutlierLineHoldsTheFamilyRateOnIndependentEdges() {
    List<Ratings> fits =
        IntStream.range(0, REPEATS)
            .parallel()
            .mapToObj(
                repeat -> Ratings.fit(SyntheticChain.edges(NODES, WALLS, 0, false, 90_000 + repeat), SyntheticChain.ANCHOR))
            .toList();
    double family = (double) fits.stream().filter(ratings -> !ratings.outliers().isEmpty()).count() / REPEATS;
    double beyondThree = fits.stream().mapToInt(Ratings::beyondThree).average().orElseThrow();
    double expected = fits.get(0).beyondThreeExpected();
    double squares =
        fits.stream()
            .flatMap(ratings -> ratings.residuals().stream())
            .mapToDouble(residual -> residual.z() * residual.z())
            .average()
            .orElseThrow();
    double bound = 0.05 + 3 * Math.sqrt(0.05 * 0.95 / REPEATS);
    log.info(
        "RATINGS-SIM outliers: family {} (bound {}), beyond 3 {} (expected {}), mean z^2 {}",
        family,
        bound,
        beyondThree,
        expected,
        squares);
    assertTrue(family <= bound && family >= 0.025, "family " + family);
    assertTrue(beyondThree > 0.85 * expected && beyondThree < 1.15 * expected, "beyond 3 " + beyondThree);
    assertTrue(Math.abs(squares - 1) < 0.03, "mean z^2 " + squares);
  }
}
