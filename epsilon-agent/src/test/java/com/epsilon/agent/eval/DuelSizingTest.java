package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 王者との決闘の上限と幅を選び直す計算（{@link DuelSizing}）を確かめる。標準偏差・中央値・見つける確率・勧める上限と幅が手計算と合う。
 * 三人打ちの王者の卓（1v2 ＋ 2v1 で牌山 1 つに 6 半荘、512 牌山ごとに判定、有意水準 0.01、消費関数の指数 3）の値で確かめる。
 */
public class DuelSizingTest {

  /** 標準正規分布の上側 5% の点。 */
  private static final double Z95 = 1.6448536269514722;

  private static final Sequential THREE = new Sequential(0.01, 3.0, 512);

  private static DuelResult result(Score score, int walls, double rankError, double utilityError) {
    List<Double> shares = List.of(1 / 3.0, 1 / 3.0, 1 / 3.0);
    return new DuelResult(
        walls,
        walls * 6L,
        Verdict.EVEN,
        Stopped.DECIDED,
        score,
        3.0,
        Estimate.of(0.01, rankError, 3.0),
        Estimate.of(-0.02, utilityError, 3.0),
        List.of(),
        List.of(
            new DuelResult.SlotValue("candidate", 2, rankError, 0, utilityError, shares),
            new DuelResult.SlotValue("champion", 2, 0, 0, 0, shares)),
        new PlayStats.Rates(0, 0, 0, 0, 0),
        0,
        0,
        0);
  }

  @Test
  public void sigmaIsTheErrorOfTheScoreTimesTheRootOfTheWalls() {
    assertEquals(DuelSizing.sigma(result(Score.RANK, 2048, 0.009, 0.5)), 0.009 * Math.sqrt(2048), 1e-15);
    assertEquals(DuelSizing.sigma(result(Score.UTILITY, 1024, 0.009, 0.5)), 0.5 * 32, 1e-12);
  }

  @Test
  public void theMedianTakesTheMiddleOrTheMeanOfTheTwoMiddles() {
    assertEquals(DuelSizing.median(new double[] {0.4}), 0.4);
    assertEquals(DuelSizing.median(new double[] {0.5, 0.1, 0.3}), 0.3);
    assertEquals(DuelSizing.median(new double[] {0.5, 0.1, 0.3, 0.2}), 0.25, 1e-15);
    double[] values = {0.4, 0.2};
    DuelSizing.median(values);
    assertTrue(Arrays.equals(values, new double[] {0.4, 0.2}), "the values are not sorted in place");
    assertThrows(IllegalArgumentException.class, () -> DuelSizing.median(new double[0]));
  }

  @Test
  public void theSuggestionForTheThreePlayerChampionTableMatchesTheHandValue() {
    double sigma = 0.40;
    // 25,000 半荘以内に収める。判定 1 回は 512 牌山 × 6 半荘 = 3,072 半荘で、8 回で 4,096 牌山・24,576 半荘
    DuelSizing.Choice choice = DuelSizing.choose(THREE, sigma, 6, 25_000);
    assertEquals(choice.walls(), 4096);
    double[] critical = THREE.boundaries(4096);
    double c = critical[critical.length - 1];
    assertTrue(c > 2.5758 && c < 3.2, "last critical value " + c);
    double margin = (c + Z95) * sigma / (1.5 * Math.sqrt(4096));
    assertEquals(choice.margin(), margin, 1e-9);
    // その幅の 1.5 倍の真の差を、最後の判定の時点で確率 0.95 で見つける
    assertEquals(DuelSizing.detection(THREE, choice.walls(), sigma, 1.5 * choice.margin()), 0.95, 1e-9);
    // 見つける確率は、最後の時点の区間の下端が 0 を超える確率
    double d = 0.02;
    assertEquals(DuelSizing.detection(THREE, 4096, sigma, d), Normal.upper(c - d * 64 / sigma), 1e-15);
    assertTrue(DuelSizing.detection(THREE, 4096, sigma, 0.0) < 0.01);
    // 幅は標準偏差に比例する
    assertEquals(DuelSizing.choose(THREE, 2 * sigma, 6, 25_000).margin(), 2 * margin, 1e-12);
  }

  @Test
  public void theCostAndTheLookLimitBoundTheWalls() {
    // 判定 1 回分ちょうど
    assertEquals(DuelSizing.choose(THREE, 0.3, 6, 3072).walls(), 512);
    assertEquals(DuelSizing.choose(THREE, 0.3, 6, 6143).walls(), 512);
    // 判定の時点は 32 回まで
    assertEquals(DuelSizing.choose(THREE, 0.3, 6, 10_000_000).walls(), 32 * 512);
    assertEquals(DuelSizing.choose(THREE, 0.3, 1, 12_300).walls(), 24 * 512);
    assertEquals(DuelSizing.choose(THREE, 0.3, 1, 25_000).walls(), 32 * 512);
    // 判定 1 回分も打てない、標準偏差が正でない、牌山 1 つあたりの半荘が 0 のときは拒む
    assertThrows(IllegalArgumentException.class, () -> DuelSizing.choose(THREE, 0.3, 6, 3071));
    assertThrows(IllegalArgumentException.class, () -> DuelSizing.choose(THREE, 0, 6, 25_000));
    assertThrows(IllegalArgumentException.class, () -> DuelSizing.choose(THREE, Double.NaN, 6, 25_000));
    assertThrows(IllegalArgumentException.class, () -> DuelSizing.choose(THREE, 0.3, 0, 25_000));
    assertThrows(IllegalArgumentException.class, () -> DuelSizing.detection(THREE, 4096, -1, 0.01));
  }
}
