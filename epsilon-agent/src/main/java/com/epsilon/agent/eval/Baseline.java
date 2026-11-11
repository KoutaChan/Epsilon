package com.epsilon.agent.eval;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 系譜の主な指標。バージョンごとに基準の卓で測った値（順位の対の量、負が良い）で、一度測った値は後から変わらない。1 つ目の基準の卓で
 * 値が張り付いてきたら、2 つ目の基準の卓を足す。2 つ目の値は、両方で測ったバージョンの平均の差を使って 1 つ目の尺度に直す。
 */
public final class Baseline {

  /** 2 つ目の基準の卓を足すしきい値（今のバージョンを基準の卓で測った、候補 1 席の分け方の値と比べる）。 */
  public static final double SECOND_LINE = -0.25;

  private Baseline() {}

  /**
   * 1 つのバージョンを基準の卓で測った値。
   *
   * @param value 順位の対の量
   * @param error その標準誤差
   */
  public record Point(long version, double value, double error) {}

  /**
   * 2 つ目の基準の値を 1 つ目の尺度に直すために足す量。
   *
   * @param offset 両方で測ったバージョンでの、(1 つ目 − 2 つ目) の平均
   * @param error 足す量の標準誤差の上限
   * @param overlap 両方で測ったバージョンの数
   */
  public record Link(double offset, double error, int overlap) {

    /** @return 1 つ目の尺度に直した値。標準誤差は、元の値と足す量の標準誤差の二乗和の平方根 */
    public Point apply(Point second) {
      return new Point(second.version(), second.value() + offset, Math.hypot(second.error(), error));
    }
  }

  /**
   * @param oneVsTwo 今のバージョンを基準の卓で測った、候補 1 席の分け方の値
   * @param line 2 つ目の基準を足すしきい値（負）
   * @return 2 つ目の基準を足すべきなら true
   */
  public static boolean needsSecond(double oneVsTwo, double line) {
    return oneVsTwo < line;
  }

  /**
   * 両方で測ったバージョンから、2 つ目の基準の値を 1 つ目の尺度に直すために足す量を求める。2 つの基準は別の牌山で打つので、
   * バージョンごとの差の標準誤差は、両方の標準誤差の二乗和の平方根になる。同じ基準の卓で測ったバージョンどうしは牌山が同じで
   * 誤差が相関するので、差の平均の標準誤差には、差ごとの標準誤差の平均を上限として使う。
   *
   * @param first 1 つ目の基準の卓で測った値
   * @param second 2 つ目の基準の卓で測った値
   * @return 足す量
   * @throws IllegalArgumentException 両方で測ったバージョンが無いとき
   */
  public static Link link(List<Point> first, List<Point> second) {
    Map<Long, Point> byVersion = new HashMap<>();
    for (Point point : first) {
      byVersion.put(point.version(), point);
    }
    int overlap = 0;
    double offset = 0;
    double error = 0;
    for (Point point : second) {
      Point other = byVersion.get(point.version());
      if (other == null) {
        continue;
      }
      overlap++;
      offset += other.value() - point.value();
      error += Math.hypot(other.error(), point.error());
    }
    if (overlap == 0) {
      throw new IllegalArgumentException("The two baselines share no version");
    }
    return new Link(offset / overlap, error / overlap, overlap);
  }
}
