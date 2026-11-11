package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 基準の卓の値と 2 つ目の基準とのつなぎ、同じ牌山の対の差を確かめる。2 つ目の基準は、1v2 の値がしきい値を下回ったときに足す。
 * つなぎの差の平均は、両方で測ったバージョンだけで取る。
 */
public class BaselineTest {

  @Test
  public void aSecondBaselineIsNeededOnlyBelowTheLine() {
    assertTrue(Baseline.needsSecond(-0.26, -0.25));
    assertFalse(Baseline.needsSecond(-0.25, -0.25));
    assertFalse(Baseline.needsSecond(0, -0.25));
  }

  /**
   * 1 つ目の基準はバージョン 0〜63、2 つ目は 32〜95 で測り、2 つ目は 1 つ目より 0.3 大きい。つなぎは重なる 32 バージョンの差で −0.3、
   * 標準誤差は 0.003 と 0.004 の二乗和の平方根で 0.005。
   */
  @Test
  public void theLinkMovesTheSecondBaselineOntoTheFirstScale() {
    List<Baseline.Point> first = new ArrayList<>();
    List<Baseline.Point> second = new ArrayList<>();
    for (int v = 0; v < 96; v++) {
      if (v < 64) {
        first.add(new Baseline.Point(v, -0.2 - 0.001 * v, 0.003));
      }
      if (v >= 32) {
        second.add(new Baseline.Point(v, 0.1 - 0.001 * v, 0.004));
      }
    }
    Baseline.Link link = Baseline.link(first, second);
    assertEquals(link.overlap(), 32);
    assertEquals(link.offset(), -0.3, 1e-12);
    assertEquals(link.error(), 0.005, 1e-12);

    Baseline.Point moved = link.apply(new Baseline.Point(80, 0.02, 0.004));
    assertEquals(moved.version(), 80);
    assertEquals(moved.value(), -0.28, 1e-12);
    assertEquals(moved.error(), Math.hypot(0.004, 0.005), 1e-12);
  }

  @Test
  public void baselinesWithoutACommonVersionCannotBeLinked() {
    List<Baseline.Point> first = List.of(new Baseline.Point(1, 0, 0.01));
    List<Baseline.Point> second = List.of(new Baseline.Point(2, 0, 0.01));
    assertThrows(IllegalArgumentException.class, () -> Baseline.link(first, second));
    assertThrows(IllegalArgumentException.class, () -> Baseline.link(first, List.of()));
  }

  /** 牌山ごとの差は、短い方の長さまでで取る。差が全て同じなら標準誤差は厳密に 0。 */
  @Test
  public void pairedWallsTakeTheCommonWalls() {
    WallMean difference = PairedWalls.difference(new double[] {1, 2, 3}, new double[] {0.5, 1.5});
    assertEquals(difference.walls(), 2);
    assertEquals(difference.mean(), 0.5, 0);
    assertEquals(difference.error(), 0.0, 0);
    assertThrows(IllegalArgumentException.class, () -> PairedWalls.difference(new double[0], new double[] {1}));
    assertThrows(
        IllegalArgumentException.class, () -> PairedWalls.difference(new double[] {Double.NaN}, new double[] {1}));
  }

  /** 牌山の運が両方の卓に同じく効くとき、対の差の標準誤差は、別々に測った標準誤差の二乗和の平方根の 1 割より小さい。 */
  @Test
  public void pairingRemovesTheSharedLuck() {
    int walls = 64;
    double[] a = new double[walls];
    double[] b = new double[walls];
    WallMean alone = new WallMean();
    WallMean other = new WallMean();
    for (int i = 0; i < walls; i++) {
      double luck = (i % 5) - 2.0;
      a[i] = luck + 0.01 * (i % 3);
      b[i] = luck - 0.01 * (i % 2);
      alone.add(a[i]);
      other.add(b[i]);
    }
    double paired = PairedWalls.difference(a, b).error();
    assertTrue(paired < 0.1 * Math.hypot(alone.error(), other.error()), paired + "");
  }
}
