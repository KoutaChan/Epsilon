package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 尺度の当てはめのテスト。辺が木の形なら、辺の平均がそのまま点の強さ θ の差になり、残差は 0。三角形の食い違いは 3 本の辺に等しく分かれる。
 * 同じ牌山の種を持つ辺は牌山ごとにまとめて扱うので、同じ値の辺を重ねても標準誤差は減らない。答えの分かる合成の系譜では θ が
 * 標準誤差の範囲で戻り、頑健な標準誤差は当て直しのばらつきと合う。外れた辺のしきい値は、辺の数が増えるほど上がる。
 */
public class RatingsTest {

  private static final Logger log = LoggerFactory.getLogger(RatingsTest.class);

  /** @return 平均が {@code mean}、平均の標準誤差が {@code error} になる 2 つの牌山の値 */
  private static double[] pair(double mean, double error) {
    return new double[] {mean - error, mean + error};
  }

  @Test
  public void aTreeTakesTheEdgeMeansAsTheRatings() {
    List<Ratings.Edge> edges =
        List.of(
            new Ratings.Edge("b", "a", 1, pair(0.2, 0.1)),
            new Ratings.Edge("c", "b", 2, pair(-0.05, 0.1)),
            new Ratings.Edge("a", "d", 3, pair(0.1, 0.1)));
    Ratings ratings = Ratings.fit(edges, "a");

    List<Ratings.Rating> list = ratings.ratings();
    assertEquals(list.size(), 4);
    assertRating(list.get(0), "a", 0, 0, 2);
    assertRating(list.get(1), "b", 0.2, 0.1, 2);
    assertRating(list.get(2), "c", 0.15, Math.sqrt(0.02), 1);
    assertRating(list.get(3), "d", -0.1, 0.1, 1);
    for (Ratings.Residual residual : ratings.residuals()) {
      assertEquals(residual.observed(), residual.fitted(), 1e-12);
      assertEquals(residual.z(), 0.0, 0);
    }
    assertTrue(Double.isNaN(ratings.chiSquarePerDegree()));
    assertTrue(ratings.outliers().isEmpty());
    assertEquals(ratings.beyondThree(), 0);
  }

  /** 辺が 1 本だけなら、θ はその辺の平均で、標準誤差はふつうの平均の標準誤差になる。辺の向きを逆にすると θ の符号が反転する。 */
  @Test
  public void aSingleEdgeGivesTheOrdinaryMeanAndError() {
    SplittableRandom random = new SplittableRandom(11);
    double[] values = new double[257];
    WallMean expected = new WallMean();
    for (int i = 0; i < values.length; i++) {
      values[i] = 0.03 + 0.4 * random.nextGaussian();
      expected.add(values[i]);
    }
    Ratings.Rating forward = Ratings.fit(List.of(new Ratings.Edge("x", "base", 5, values)), "base").ratings().get(1);
    assertEquals(forward.value(), expected.mean(), 1e-12);
    assertEquals(forward.error(), expected.error(), 1e-12 * expected.error());

    Ratings.Rating backward = Ratings.fit(List.of(new Ratings.Edge("base", "x", 5, values)), "base").ratings().get(1);
    assertEquals(backward.value(), -expected.mean(), 1e-12);
    assertEquals(backward.error(), expected.error(), 1e-12 * expected.error());
  }

  /**
   * 同じ牌山の種で同じ値の辺を 2 本重ねても、牌山ごとにまとめるので標準誤差は 1 本のときと変わらない。別の牌山の種で渡すと
   * 独立な辺とみなされ、標準誤差は 1/√2 倍になる。
   */
  @Test
  public void edgesOnTheSameWallsAreClusteredTogether() {
    double[] values = {0.1, -0.3, 0.25, 0.05, -0.1, 0.4};
    double single = Ratings.fit(List.of(new Ratings.Edge("x", "base", 7, values)), "base").ratings().get(1).error();
    Ratings.Rating shared =
        Ratings.fit(List.of(new Ratings.Edge("x", "base", 7, values), new Ratings.Edge("x", "base", 7, values)), "base")
            .ratings()
            .get(1);
    Ratings.Rating separate =
        Ratings.fit(List.of(new Ratings.Edge("x", "base", 7, values), new Ratings.Edge("x", "base", 8, values)), "base")
            .ratings()
            .get(1);
    assertEquals(shared.error(), single, 1e-12);
    assertEquals(separate.error(), single / Math.sqrt(2), 1e-12);
    assertEquals(shared.value(), separate.value(), 1e-12);
  }

  /**
   * 重みの等しい三角形を 1 周したときの食い違いが r なら、各辺の残差は r/3、てこ比は 2/3 で、z は r / (√3 × 標準誤差) になる。
   * 辺が 0.1、0.1、0.26 なら食い違いは 0.06 で、標準誤差 0.01 のとき z は ±3.464、χ²/自由度は 12。
   */
  @Test
  public void aTriangleSplitsItsMismatchOverItsEdges() {
    Ratings ratings =
        Ratings.fit(
            List.of(
                new Ratings.Edge("b", "a", 1, pair(0.1, 0.01)),
                new Ratings.Edge("c", "b", 2, pair(0.1, 0.01)),
                new Ratings.Edge("c", "a", 3, pair(0.26, 0.01))),
            "a");
    assertEquals(ratings.ratings().get(1).value(), 0.12, 1e-12);
    assertEquals(ratings.ratings().get(2).value(), 0.24, 1e-12);
    double z = 0.06 / (Math.sqrt(3) * 0.01);
    List<Ratings.Residual> residuals = ratings.residuals();
    assertEquals(residuals.get(0).z(), -z, 1e-9);
    assertEquals(residuals.get(1).z(), -z, 1e-9);
    assertEquals(residuals.get(2).z(), z, 1e-9);
    assertEquals(residuals.get(2).fitted(), 0.24, 1e-12);
    assertEquals(ratings.chiSquarePerDegree(), 12, 1e-9);
    assertEquals(ratings.outlierLine(), Normal.upperPoint(0.05 / 6), 0);
    assertEquals(ratings.outliers().size(), 3);
    assertEquals(ratings.beyondThree(), 3);
  }

  /** 食い違いの無い系譜では、外れた辺は出ない。1 本の辺だけを 20 標準誤差ずらすと、その辺が外れた辺になり、|z| が最も大きくなる。 */
  @Test
  public void aShiftedEdgeStandsOutAsTheOutlier() {
    Ratings clean = Ratings.fit(grid(Double.NaN), "x0");
    assertEquals(clean.chiSquarePerDegree(), 0, 1e-12);
    assertTrue(clean.outliers().isEmpty());
    assertEquals(clean.beyondThree(), 0);

    Ratings shifted = Ratings.fit(grid(0.2), "x0");
    Ratings.Residual worst =
        shifted.residuals().stream().max(Comparator.comparingDouble(residual -> Math.abs(residual.z()))).orElseThrow();
    assertEquals(worst.a(), "x15");
    assertEquals(worst.b(), "x0");
    assertTrue(shifted.outliers().contains(worst), shifted.outliers().toString());
    assertTrue(shifted.beyondThree() >= 1);
  }

  /**
   * 点 x0〜x30 をつなぐラダーと、x2 以降の各点から x0 への辺を作る。隣り合う点の真の差は 0.01 で、標準誤差は全て 0.01。
   *
   * @param shift x15 から x0 への辺の平均に足すずれ。NaN ならずらさない
   */
  private static List<Ratings.Edge> grid(double shift) {
    List<Ratings.Edge> edges = new ArrayList<>();
    for (int v = 1; v <= 30; v++) {
      edges.add(new Ratings.Edge("x" + v, "x" + (v - 1), v, pair(0.01, 0.01)));
      if (v >= 2) {
        double mean = 0.01 * v + (v == 15 && !Double.isNaN(shift) ? shift : 0);
        edges.add(new Ratings.Edge("x" + v, "x0", 100 + v, pair(mean, 0.01)));
      }
    }
    return edges;
  }

  /**
   * 外れた辺のしきい値は、どれか 1 本でも外れる確率を全体で 5% に抑える |z| で、辺 100 本で 3.48、500 本で 3.89 になる。|z| が 3 を
   * 超える本数の期待値は 0.0027 × 辺の数。
   */
  @Test
  public void theOutlierLineFollowsTheEdgeCount() {
    double previous = 0;
    for (int count : new int[] {1, 10, 100, 500}) {
      Ratings ratings = Ratings.fit(star(count), "x0");
      assertEquals(ratings.outlierLine(), Normal.upperPoint(0.05 / (2.0 * count)), 0);
      assertTrue(ratings.outlierLine() > previous, count + ": " + ratings.outlierLine());
      previous = ratings.outlierLine();
      assertEquals(ratings.beyondThreeExpected(), 0.0026998 * count, 1e-6 * count);
    }
    assertEquals(Ratings.fit(star(100), "x0").outlierLine(), 3.48, 0.005);
    assertEquals(Ratings.fit(star(500), "x0").outlierLine(), 3.89, 0.005);
  }

  /** 各点が x0 への辺 1 本だけで決まる星の形の辺を作る。どの辺もてこ比は 1 で、z は 0。 */
  private static List<Ratings.Edge> star(int count) {
    SplittableRandom random = new SplittableRandom(count);
    List<Ratings.Edge> edges = new ArrayList<>(count);
    for (int e = 0; e < count; e++) {
      double[] values = {random.nextGaussian(), random.nextGaussian(), random.nextGaussian()};
      edges.add(new Ratings.Edge("x" + (e + 1), "x0", e, values));
    }
    return edges;
  }

  @Test
  public void aStarHasNoResidualToJudge() {
    for (Ratings.Residual residual : Ratings.fit(star(50), "x0").residuals()) {
      assertEquals(residual.z(), 0.0, 0);
    }
  }

  @Test
  public void invalidGraphsAreRejected() {
    double[] values = {0.1, 0.2, 0.4};
    assertThrows(IllegalArgumentException.class, () -> Ratings.fit(List.of(), "a"));
    assertThrows(
        IllegalArgumentException.class, () -> Ratings.fit(List.of(new Ratings.Edge("b", "c", 1, values)), "a"));
    assertThrows(
        IllegalArgumentException.class,
        () -> Ratings.fit(List.of(new Ratings.Edge("b", "a", 1, values), new Ratings.Edge("d", "c", 2, values)), "a"));
    assertThrows(
        IllegalArgumentException.class,
        () -> Ratings.fit(List.of(new Ratings.Edge("b", "a", 1, new double[] {0.1})), "a"));
    assertThrows(
        IllegalArgumentException.class,
        () -> Ratings.fit(List.of(new Ratings.Edge("b", "a", 1, new double[] {0.1, 0.1, 0.1})), "a"));
    assertThrows(IllegalArgumentException.class, () -> new Ratings.Edge("a", "a", 1, values));
    assertThrows(NullPointerException.class, () -> new Ratings.Edge("a", null, 1, values));
    assertThrows(NullPointerException.class, () -> new Ratings.Edge("a", "b", 1, null));
    assertThrows(NullPointerException.class, () -> Ratings.fit(List.of(new Ratings.Edge("b", "a", 1, values)), null));
  }

  /** 答えの分かる系譜（点 40 個、牌山 1,024 個、基準との辺は牌山を共有）を 5 つの種で当てはめ、どの点も θ が真の値から 4 標準誤差以内。 */
  @Test
  public void aKnownChainIsRecoveredWithinItsErrors() {
    for (long seed = 1; seed <= 5; seed++) {
      Ratings ratings = Ratings.fit(SyntheticChain.edges(40, 1024, 0.2, true, seed), SyntheticChain.ANCHOR);
      assertEquals(ratings.ratings().size(), 40);
      for (Ratings.Rating rating : ratings.ratings().subList(1, 40)) {
        double z = (rating.value() - SyntheticChain.theta(SyntheticChain.index(rating.label()))) / rating.error();
        assertTrue(Math.abs(z) < 4, "seed " + seed + " " + rating + " z " + z);
      }
    }
  }

  /**
   * 種を変えて 200 回当て直す（点 20 個、牌山 512 個）。頑健な標準誤差の平均は、θ のばらつきの 0.8〜1.2 倍に入る。同じ値を独立な辺
   * として当てはめると、v10 以降の点では標準誤差がばらつきの 0.8 倍より小さく出る。牌山の共有を無視すると、誤差を小さく見積もる
   * 危険側にずれることも確かめる。
   */
  @Test
  public void robustErrorsMatchTheSimulatedSpread() {
    int nodes = 20;
    SyntheticChain.Spread[] clustered = SyntheticChain.spread(nodes, 512, 200, true, 1000);
    SyntheticChain.Spread[] naive = SyntheticChain.spread(nodes, 512, 200, false, 1000);
    for (int v = 1; v < nodes; v++) {
      double ratio = clustered[v].reported() / clustered[v].simulated();
      log.info(
          "RATINGS v{}: simulated {} robust {} naive {}",
          v,
          clustered[v].simulated(),
          clustered[v].reported(),
          naive[v].reported());
      assertTrue(ratio > 0.8 && ratio < 1.2, "v" + v + " ratio " + ratio);
    }
    for (int v = 10; v < nodes; v++) {
      assertTrue(naive[v].reported() < 0.8 * naive[v].simulated(), "v" + v + " naive " + naive[v].reported());
    }
  }

  private static void assertRating(Ratings.Rating rating, String label, double value, double error, int edges) {
    assertEquals(rating.label(), label);
    assertEquals(rating.value(), value, 1e-12, label);
    assertEquals(rating.error(), error, 1e-12, label);
    assertEquals(rating.edges(), edges, label);
  }
}
