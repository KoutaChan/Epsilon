package com.epsilon.prism.probe;

import com.epsilon.prism.core.DecisionClass;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * probe-nano の集計（純粋な関数 {@link #of}）。入力は類ごとの数（床・一致・不一致）と、不一致のラベル行ごとの ΔY = Y[nano] − Y[played]。
 *
 * <ul>
 *   <li>meanDelta ± meanDeltaError: 不一致 1 つあたりの ΔY の平均と、半荘で束ねた標準誤差。半荘 g の残差の和 r_g = Σ_{i∈g} (ΔY_i − 平均)、
 *       行のある半荘の数 G、行の数 n として SE² = G/(G−1)·Σ_g r_g² / n²（G &lt; 2 なら NaN）。
 *   <li>perGame ± perGameError: 半荘 1 本（4 席の和）あたり、その類の全ての判断で PRISM が Nano に負けている効用の推定。半荘 g の S_g =
 *       Σ_{i∈g} ΔY_i / ρ_{c(i)}（行のない半荘は 0）の平均と、S_g の標準偏差 / √games。ラベル率 ρ で割るので、ラベルに予約しなかった判断を含む
 *       その類の全ての判断の推定（床 ε_b の手を打った判断は測らず 0 と数える。その割合は floor / (floor + probed) で、ε_b 程度）。perSeat は
 *       1 席あたり（perGame / 4）。全体の行は類ごとの S_g の和で、ρ = 0 の類（行がない）は含まない。
 *   <li>positiveShare: 不一致のうち ΔY &gt; 0（Nano の手が良かった）の割合。
 * </ul>
 */
public final class ProbeReport {
  private ProbeReport() {}

  /** 不一致の 1 行。 */
  public record Disagreement(long gameId, DecisionClass decisionClass, double delta) {}

  /**
   * 1 つの類か全体の集計。
   *
   * @param decisionClass 類の名前（全体は "ALL"）
   * @param labelRate ラベル率 ρ（全体は NaN）
   * @param floor 床 ε_b の手を打ったので測らなかった判断
   * @param probed Nano に問うた判断（一致 + 不一致）
   * @param agreementRate 一致 / probed
   */
  public record Line(
      String decisionClass,
      double labelRate,
      long floor,
      long probed,
      long agreements,
      long disagreements,
      double agreementRate,
      double meanDelta,
      double meanDeltaError,
      double positiveShare,
      double perGame,
      double perGameError,
      double perSeat,
      double perSeatError) {}

  /** 類ごとの集計（{@link DecisionClass} の順）と全体。 */
  public record Report(int games, List<Line> classes, Line overall) {}

  /**
   * 集計する。配列は {@link DecisionClass} の番号の順。
   *
   * @param games 半荘の数
   * @param labelRates 類ごとのラベル率 ρ
   * @param rows 不一致のラベル行（類ごとの数は disagreements と一致しなければならない）
   */
  public static Report of(
      int games,
      double[] labelRates,
      long[] floor,
      long[] agreements,
      long[] disagreements,
      List<Disagreement> rows) {
    if (games < 1) throw new IllegalArgumentException("games " + games);
    List<List<Disagreement>> byClass = new ArrayList<>(DecisionClass.COUNT);
    for (int c = 0; c < DecisionClass.COUNT; c++) byClass.add(new ArrayList<>());
    for (Disagreement row : rows) byClass.get(row.decisionClass().ordinal()).add(row);
    List<Line> classes = new ArrayList<>(DecisionClass.COUNT);
    long floorTotal = 0, agreeTotal = 0, disagreeTotal = 0;
    for (int c = 0; c < DecisionClass.COUNT; c++) {
      if (byClass.get(c).size() != disagreements[c])
        throw new IllegalStateException(
            DecisionClass.of(c)
                + " has "
                + byClass.get(c).size()
                + " rows but "
                + disagreements[c]
                + " disagreements");
      classes.add(
          line(
              DecisionClass.of(c).name(),
              labelRates[c],
              floor[c],
              agreements[c],
              disagreements[c],
              byClass.get(c),
              games,
              labelRates));
      floorTotal += floor[c];
      agreeTotal += agreements[c];
      disagreeTotal += disagreements[c];
    }
    Line overall =
        line("ALL", Double.NaN, floorTotal, agreeTotal, disagreeTotal, rows, games, labelRates);
    return new Report(games, List.copyOf(classes), overall);
  }

  private static Line line(
      String name,
      double rate,
      long floor,
      long agreements,
      long disagreements,
      List<Disagreement> rows,
      int games,
      double[] labelRates) {
    long probed = agreements + disagreements;
    int n = rows.size();
    double mean = 0;
    int positive = 0;
    for (Disagreement row : rows) {
      mean += row.delta();
      if (row.delta() > 0) positive++;
    }
    mean = n == 0 ? Double.NaN : mean / n;
    // 半荘ごとの残差の和（平均の SE）と、ρ で割った ΔY の和（半荘あたりの推定）
    Map<Long, double[]> perGame = new HashMap<>();
    for (Disagreement row : rows) {
      double[] sums = perGame.computeIfAbsent(row.gameId(), id -> new double[2]);
      sums[0] += row.delta() - mean;
      sums[1] += row.delta() / labelRates[row.decisionClass().ordinal()];
    }
    double residuals = 0, total = 0, squares = 0;
    for (double[] sums : perGame.values()) {
      residuals += sums[0] * sums[0];
      total += sums[1];
      squares += sums[1] * sums[1];
    }
    int clusters = perGame.size();
    double meanError =
        clusters < 2 ? Double.NaN : Math.sqrt(clusters / (clusters - 1.0) * residuals) / n;
    double estimate = Double.NaN, estimateError = Double.NaN;
    // ρ = 0 の類は測っていないので推定しない（全体は ρ > 0 の類だけの和）
    if (!(rate == 0)) {
      estimate = total / games;
      if (games > 1)
        estimateError =
            Math.sqrt(Math.max(0, (squares - games * estimate * estimate) / (games - 1)) / games);
    }
    return new Line(
        name,
        rate,
        floor,
        probed,
        agreements,
        disagreements,
        probed == 0 ? Double.NaN : agreements / (double) probed,
        mean,
        meanError,
        n == 0 ? Double.NaN : positive / (double) n,
        estimate,
        estimateError,
        estimate / 4,
        estimateError / 4);
  }
}
