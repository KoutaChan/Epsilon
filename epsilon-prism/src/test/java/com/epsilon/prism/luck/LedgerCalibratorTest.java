package com.epsilon.prism.luck;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.noise.LedgerCalibration;
import com.epsilon.prism.noise.LedgerCalibrationKind;
import com.epsilon.prism.noise.LedgerCalibrationSource;
import com.epsilon.prism.noise.LedgerCells;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.rows.Row;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Predicate;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 帳簿の補正率の校正器（{@link LedgerCalibrator}）のテスト。確かめるのは次のこと。箱の制約の付いた二次計画は、全ての有効な制約の組を試した
 * 最小と同じ解を返し、大きな問題でも KKT 条件を満たす。格子の点の番号と双線形の重みは、四人打ち・三人打ちとも配給原点と南場の最後の局から
 * 決まり、補間は格子の点の番号の 1 次式を再現する。補正が Y と同じなら補正率は罰の分だけ 1 より小さく、補正が Y と無関係なら 0、行が触れない
 * 係数は 0 になる。罰 ρ を強めると、補正の分散で重みを付けた補正率の和は増えず、十分に強ければ全て 0 になる。係数はいつも [0, 1] に入る。
 * 補正の分散が無ければ校正器を作らない。校正器は JSON に書いて読み戻すとビットまで同じ。2 回目の走査の J と帯は行ごとに直に求めた値と合い、
 * 補正率が無ければ影の腕は今の腕とビットまで同じ。和了の判断の差は、比べる手を s で選び、s が無ければツモを数えない。
 */
public class LedgerCalibratorTest {
  private static final int F = LedgerColumn.COUNT;

  /** 設計の既定値（9 × 3 の格子、二乗の罰 1e-4、縮める罰 0.08）。 */
  static final LedgerCalibration.Shadow DEFAULTS =
      new LedgerCalibration.Shadow(
          LedgerCalibrationKind.COMMON_SCALE,
          LedgerCalibrationSource.PREVIOUS_COHORT,
          9,
          3,
          1e-4,
          0.08);

  @Test
  public void theBoxQuadraticMatchesEveryActiveSet() {
    var random = new SplittableRandom(7);
    for (int trial = 0; trial < 400; trial++) {
      int n = 1 + random.nextInt(6);
      double[][] q = positiveDefinite(n, random);
      double[] c = new double[n];
      for (int j = 0; j < n; j++) c[j] = 2 * random.nextGaussian();
      double[] theta = LedgerCalibrator.BoxQuadratic.minimize(q, c);
      double[] best = bruteForce(q, c);
      for (int j = 0; j < n; j++) {
        Assert.assertTrue(theta[j] >= 0 && theta[j] <= 1, "trial " + trial);
        Assert.assertEquals(theta[j], best[j], 1e-9, "trial " + trial + " coefficient " + j);
      }
      Assert.assertTrue(objective(q, c, theta) <= objective(q, c, best) + 1e-12, "trial " + trial);
    }
  }

  /** 当てはめと同じ形（1 行が 4 つの係数に触れる和と、二乗の罰）の 108 係数の問題でも、KKT 条件を満たす。 */
  @Test
  public void theBoxQuadraticSatisfiesTheOptimalityConditions() {
    var random = new SplittableRandom(11);
    var sums = new LedgerCalibrator.Sums(new LedgerCalibrator.Grid(Variant.TENHOU_4P, 9, 3));
    for (int i = 0; i < 20_000; i++) {
      double a = Math.abs(random.nextGaussian());
      sums.add(standing(random, Variant.TENHOU_4P), 2 * a, a * (2 * random.nextDouble() - 0.5), a);
    }
    int n = sums.size;
    double lambda = 1e-4 * sums.total / n;
    double[][] q = new double[n][n];
    double[] c = new double[n];
    double scale = 0;
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) q[j][k] = sums.h[j * n + k];
      q[j][j] += lambda;
      c[j] = sums.r[j] - 0.08 * sums.a[j];
      scale = Math.max(scale, Math.max(Math.abs(c[j]), q[j][j]));
    }
    double[] theta = LedgerCalibrator.BoxQuadratic.minimize(q, c);
    int free = 0, lower = 0, upper = 0;
    for (int j = 0; j < n; j++) {
      double gradient = -c[j];
      for (int k = 0; k < n; k++) gradient += q[j][k] * theta[k];
      double tolerance = 1e-9 * scale;
      if (theta[j] == 0) {
        lower++;
        Assert.assertTrue(gradient >= -tolerance, j + " " + gradient);
      } else if (theta[j] == 1) {
        upper++;
        Assert.assertTrue(gradient <= tolerance, j + " " + gradient);
      } else {
        free++;
        Assert.assertTrue(theta[j] > 0 && theta[j] < 1, j + " " + theta[j]);
        Assert.assertEquals(gradient, 0, tolerance, "coefficient " + j);
      }
    }
    Assert.assertTrue(free > 0 && lower + upper > 0, free + " " + lower + " " + upper);
  }

  /**
   * 点差 0・東 1 局は順位の区画の最初の点、点差が配給原点・局が南場の最後の局の次（西 1 局）なら点差の点 4・局の点 1 に重み 1。ほかの
   * 局面でも、重みは 0 以上で和が 1、補間は点の番号の 1 次式をそのまま返す。点差と局がどれだけ大きくても、最後の区間の中に入る。
   */
  @Test
  public void theGridFollowsTheStartPointsAndTheLastSouthRound() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      var grid = new LedgerCalibrator.Grid(variant, 9, 3);
      Assert.assertEquals(grid.size(), seats * 27, variant.name());
      int[] index = new int[4];
      double[] weight = new double[4];
      int first = (seats - 1) * 27;
      grid.corners(new LedgerCell.Standing(seats - 1, 0, 0), index, weight);
      Assert.assertEquals(index, new int[] {first, first + 1, first + 3, first + 4});
      Assert.assertEquals(weight, new double[] {1, 0, 0, 0}, 0.0);
      grid.corners(
          new LedgerCell.Standing(1, variant.startPoints(), variant.lastRound() + 1),
          index,
          weight);
      Assert.assertEquals(index[0], 27 + 4 * 3 + 1, variant.name());
      Assert.assertEquals(weight, new double[] {1, 0, 0, 0}, 0.0, variant.name());
      grid.corners(new LedgerCell.Standing(0, 1_000_000_000, variant.finalRound()), index, weight);
      Assert.assertEquals(index[0], 7 * 3 + 1, variant.name());
      Assert.assertTrue(weight[2] + weight[3] > 0.999, Arrays.toString(weight));

      var random = new SplittableRandom(3 + seats);
      List<Double> linear = new ArrayList<>();
      for (int rank = 0; rank < seats; rank++)
        for (int gap = 0; gap < 9; gap++)
          for (int round = 0; round < 3; round++) linear.add((gap + 2 * round) / 13.0);
      LedgerCalibrator calibrator = calibrator(variant, linear);
      for (int i = 0; i < 2000; i++) {
        LedgerCell.Standing standing = standing(random, variant);
        grid.corners(standing, index, weight);
        double total = 0;
        for (double w : weight) {
          Assert.assertTrue(w >= 0, Arrays.toString(weight));
          total += w;
        }
        Assert.assertEquals(total, 1, 1e-12);
        double x = 8.0 * standing.gap() / (variant.startPoints() + standing.gap());
        double t = 2.0 * standing.round() / (variant.lastRound() + 1 + standing.round());
        Assert.assertEquals(
            calibrator.scale(standing), (x + 2 * t) / 13, 1e-12, standing.toString());
      }
    }
  }

  /**
   * 1 行が 1 つの格子点だけに触れるとき、係数は閉じた式になる。補正が Y と同じ行（D = A）の点は (ΣD − ρΣA)/(ΣA + λ)、補正が Y と
   * 無関係な行（D = 0）の点と、行が触れない点は 0。λ = 1e-4 · 全ての行の ΣA / 係数の数。四人打ちも三人打ちも同じ。
   */
  @Test
  public void aFaithfulLedgerKeepsItsScaleAndAnUnrelatedOneLosesIt() {
    for (Variant variant : Variant.values()) {
      var grid = new LedgerCalibrator.Grid(variant, 9, 3);
      var sums = new LedgerCalibrator.Sums(grid);
      var faithful = new LedgerCell.Standing(0, 0, 0);
      var unrelated = new LedgerCell.Standing(1, variant.startPoints(), 0);
      for (int i = 0; i < 300; i++) sums.add(faithful, 0.05, 0.02, 0.02);
      for (int i = 0; i < 200; i++) sums.add(unrelated, 0.05, 0, 0.03);
      LedgerCalibrator fit = fit(sums, DEFAULTS, variant);
      Assert.assertNotNull(fit);
      double total = 300 * 0.02 + 200 * 0.03;
      double lambda = 1e-4 * total / grid.size();
      double expected = (300 * 0.02 - 0.08 * 300 * 0.02) / (300 * 0.02 + lambda);
      double[] theta = fit.theta();
      Assert.assertEquals(theta.length, grid.size());
      Assert.assertEquals(theta[0], expected, 1e-12, variant.name());
      Assert.assertTrue(theta[0] > 0.91 && theta[0] < 0.92, String.valueOf(theta[0]));
      int other = 27 + 4 * 3;
      for (int j = 0; j < theta.length; j++) if (j != 0) Assert.assertEquals(theta[j], 0.0, j + "");
      Assert.assertEquals(fit.scale(faithful), theta[0]);
      Assert.assertEquals(fit.scale(unrelated), theta[other]);
      Assert.assertEquals(fit.rows(), 500);
      Assert.assertEquals(fit.cohort(), 7);
    }
  }

  /**
   * 罰 ρ を強めると、補正の分散で重みを付けた補正率の和 Σ_i A_i h(o_i) は増えない（最小値は ρ について凹で、その傾きがこの和）。
   * ρ が全ての点で r_j/a_j を越えれば、全ての係数が 0。罰が 0 のとき、補正を 2 倍にしたい順位の係数は上限の 1 に、補正を逆に掛けたい
   * 順位の係数は下限の 0 に止まり、どの係数も [0, 1] に入る。
   */
  @Test
  public void shrinkingNeverRaisesTheWeightedScale() {
    var random = new SplittableRandom(5);
    var sums = new LedgerCalibrator.Sums(new LedgerCalibrator.Grid(Variant.TENHOU_4P, 9, 3));
    double[] ratio = {2, -1, 0.5, 0.3};
    for (int i = 0; i < 5000; i++) {
      double a = (0.5 + random.nextDouble()) * 0.01;
      LedgerCell.Standing standing = standing(random, Variant.TENHOU_4P);
      double d = a * (ratio[standing.rank()] + 0.1 * random.nextGaussian());
      sums.add(standing, 0.03, d, a);
    }
    double previous = Double.POSITIVE_INFINITY;
    boolean someOne = false, someZero = false;
    for (double rho : new double[] {0, 0.02, 0.08, 0.2, 0.5, 1, 2}) {
      LedgerCalibrator fit = fit(sums, withShrink(rho), Variant.TENHOU_4P);
      double[] theta = fit.theta();
      double weighted = 0;
      for (int j = 0; j < theta.length; j++) {
        Assert.assertTrue(theta[j] >= 0 && theta[j] <= 1, rho + " " + theta[j]);
        weighted += sums.a[j] * theta[j];
        if (rho == 0 && sums.a[j] > 0) {
          someOne |= theta[j] == 1;
          someZero |= theta[j] == 0;
        }
      }
      Assert.assertTrue(
          weighted <= previous + 1e-9 * Math.abs(weighted),
          rho + ": " + weighted + " > " + previous);
      previous = weighted;
    }
    Assert.assertTrue(someOne && someZero, "the bounds are reached at rho 0");
    double limit = 0;
    for (int j = 0; j < sums.size; j++)
      if (sums.a[j] > 0) limit = Math.max(limit, sums.r[j] / sums.a[j]);
    for (double coefficient : fit(sums, withShrink(limit + 0.01), Variant.TENHOU_4P).theta())
      Assert.assertEquals(coefficient, 0.0);
  }

  /** 行が無いか、どの行も補正の分散が 0 なら、当てはめる情報が無いので校正器を作らない。 */
  @Test
  public void noLedgerVarianceMakesNoCalibrator() {
    for (Variant variant : Variant.values()) {
      var grid = new LedgerCalibrator.Grid(variant, 9, 3);
      Assert.assertNull(fit(new LedgerCalibrator.Sums(grid), DEFAULTS, variant));
      var flat = new LedgerCalibrator.Sums(grid);
      for (int i = 0; i < 100; i++)
        flat.add(new LedgerCell.Standing(i % variant.seats(), 100 * i, i % 9), 0.01, 0, 0);
      Assert.assertNull(fit(flat, DEFAULTS, variant));
    }
    var grid = new LedgerCalibrator.Grid(Variant.TENHOU_4P, 9, 3);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> fit(new LedgerCalibrator.Sums(grid), DEFAULTS, Variant.TENHOU_3P));
  }

  /** 校正器は、係数の数が格子と合い、係数が [0, 1] に入るものだけを作れる。JSON に書いて読み戻すと、係数までビットで同じ。 */
  @Test
  public void theCalibratorRoundTripsThroughJson() {
    var random = new SplittableRandom(9);
    List<Double> coefficients = new ArrayList<>();
    for (int j = 0; j < 81; j++) coefficients.add(j % 5 == 0 ? 0.0 : random.nextDouble());
    LedgerCalibrator calibrator = calibrator(Variant.TENHOU_3P, coefficients);
    LedgerCalibrator back =
        RecordFile.read(LedgerCalibrator.class, RecordFile.json(calibrator), "calibrator.json")
            .value();
    Assert.assertEquals(back, calibrator);
    for (int j = 0; j < 81; j++)
      Assert.assertEquals(
          Double.doubleToRawLongBits(back.coefficients().get(j)),
          Double.doubleToRawLongBits(coefficients.get(j)));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> calibrator(Variant.TENHOU_4P, coefficients));
    List<Double> outside = new ArrayList<>(coefficients);
    outside.set(3, 1.0000001);
    Assert.expectThrows(
        IllegalArgumentException.class, () -> calibrator(Variant.TENHOU_3P, outside));
  }

  /**
   * 2 回目の走査の J は、行ごとに測った手の中で中心化した残差の二乗を測った手の数で割った値の平均で、スロットの R² は手ごとの残差の二乗の和の
   * 比。補正率が無ければ影の腕は今の腕とビットまで同じ。帯は、南場の最後の局より後の行を順位と点差で分け、全ての行・前・後の帯とあわせて
   * 行の数が合う。次の反復の補正率は、同じ行の C・D・A を直に足した和から当てはめたものと同じ。
   */
  @Test
  public void theSecondPassMeasuresEveryArmRowByRow() {
    Variant variant = Variant.TENHOU_4P;
    var rows = scaledRows(variant, 4000, 0, 21);
    var fitter =
        new LedgerFit(LedgerGroup.computed(), 0.001, 32, LedgerCells.CLASS_STAGE_RANK, variant);
    LedgerCorrection current = fitter.fitRows(rows).correction();
    LedgerCalibrator scale = calibrator(variant, Collections.nCopies(108, 0.55));
    for (LedgerCalibrator applied : Arrays.asList(null, scale)) {
      var pass = new LedgerCalibrator.Pass(current, applied, DEFAULTS, variant);
      var direct = new LedgerCalibrator.Sums(new LedgerCalibrator.Grid(variant, 9, 3));
      double[] j = new double[3], slot = new double[3];
      double[] audit = new double[4];
      long last = 0;
      Map<String, Long> bands = new HashMap<>();
      for (var row : rows) {
        pass.add(row);
        double h = applied == null ? 1 : 0.55;
        double[] beta =
            current.coefficients(row.gameId(), row.decisionClass().ordinal(), row.cell());
        double[] y = new double[3], b = new double[3];
        double my = 0, mb = 0;
        for (int a = 0; a < 3; a++) {
          y[a] = row.values()[a];
          for (int f = 0; f < F; f++) b[a] += beta[f] * row.ledger(a, f);
          my += y[a] / 3;
          mb += b[a] / 3;
        }
        double xx = 0, xv = 0, vv = 0, ec = 0, es = 0;
        for (int a = 0; a < 3; a++) {
          double x = y[a] - my, v = b[a] - mb;
          xx += x * x;
          xv += x * v;
          vv += v * v;
          ec += (x - v) * (x - v);
          es += (x - h * v) * (x - h * v);
        }
        direct.add(row.standing(), xx / 3, xv / 3, vv / 3);
        double offset = (b[row.played()] - mb) * 3 / 2;
        audit[0] += mb;
        audit[1] += h * mb;
        audit[2] += offset;
        audit[3] += h * offset;
        j[0] += xx / 3;
        j[1] += ec / 3;
        j[2] += es / 3;
        slot[0] += xx;
        slot[1] += ec;
        slot[2] += es;
        if (row.standing().round() >= variant.lastRound()) {
          last++;
          int band = LedgerCalibrator.band(LedgerCalibrator.GAP_BANDS, row.standing().gap());
          String name =
              "rank"
                  + (row.standing().rank() + 1)
                  + "_"
                  + LedgerCalibrator.bandName(LedgerCalibrator.GAP_BANDS, band, "_");
          bands.merge(name, 1L, Long::sum);
        }
      }
      LedgerCalibrator next = fit(pass.sums(), DEFAULTS, variant);
      LedgerReport.Calibration report =
          pass.report(
              "SHADOW", LedgerCalibrationKind.COMMON_SCALE, LedgerReport.Arm.CLASS, null, next);
      int n = rows.size();
      Assert.assertEquals(report.rows(), n);
      Assert.assertEquals(report.slots(), 3L * n);
      Assert.assertEquals(report.j().raw(), j[0] / n, 1e-12);
      Assert.assertEquals(report.j().current(), j[1] / n, 1e-12);
      Assert.assertEquals(report.j().shadow(), j[2] / n, 1e-12);
      Assert.assertEquals(report.slotR2().current(), 1 - slot[1] / slot[0], 1e-12);
      Assert.assertEquals(report.slotR2().shadow(), 1 - slot[2] / slot[0], 1e-12);
      Assert.assertEquals(pass.shadowR2(), report.slotR2().shadow());
      if (applied == null) {
        Assert.assertEquals(report.j().shadow(), report.j().current());
        Assert.assertEquals(report.slotR2().shadow(), report.slotR2().current());
        Assert.assertNull(report.origin());
        Assert.assertEquals(report.scale().mean(), 1.0);
        Assert.assertEquals((long) report.scale().histogram().get(9), n);
      } else {
        Assert.assertEquals(report.origin().cohort(), 7);
        Assert.assertEquals(report.origin().source(), LedgerCalibrationSource.PREVIOUS_COHORT);
        Assert.assertEquals(report.origin().shrink(), 0.08);
        Assert.assertEquals(report.scale().mean(), 0.55, 1e-12);
        Assert.assertEquals((long) report.scale().histogram().get(5), n);
        Assert.assertTrue(report.j().shadow() < report.j().raw(), report.j().toString());
      }
      Map<String, LedgerReport.Band> byName = new HashMap<>();
      for (LedgerReport.Band band : report.bands()) byName.put(band.name(), band);
      Assert.assertEquals(byName.size(), 3 + 4 * LedgerCalibrator.GAP_BANDS.length);
      Assert.assertEquals(byName.get("all").rows(), n);
      Assert.assertEquals(byName.get("last").rows(), last);
      Assert.assertEquals(byName.get("early").rows(), n - last);
      Assert.assertEquals(byName.get("all").j().shadow(), report.j().shadow(), 1e-12);
      for (var entry : byName.entrySet())
        if (entry.getKey().startsWith("rank"))
          Assert.assertEquals(
              entry.getValue().rows(),
              (long) bands.getOrDefault(entry.getKey(), 0L),
              entry.getKey());
      Assert.assertTrue(byName.get("all").seShadowMinusRaw() > 0);
      Assert.assertEquals(fit(direct, DEFAULTS, variant).theta(), next.theta(), 1e-12);
      Assert.assertEquals(report.next().cohort(), 7);
      Assert.assertEquals(report.next().rows(), n);
      Assert.assertEquals(report.next().j().raw(), j[0] / n, 1e-12);
      Assert.assertEquals(report.next().j().current(), j[1] / n, 1e-9);
      Assert.assertEquals(report.next().coefficients(), 108);
      Assert.assertEquals(report.next().calibrator(), next);
      Map<String, LedgerReport.Audit> audits = new HashMap<>();
      for (LedgerReport.Audit group : report.audits()) audits.put(group.group(), group);
      Assert.assertEquals(
          audits.size(), LedgerCalibrator.AUDIT_GROUPS.size() + DecisionClass.COUNT);
      LedgerReport.Audit all = audits.get("all");
      Assert.assertEquals(all.rows(), n);
      Assert.assertEquals(all.played(), n);
      Assert.assertEquals(all.rowMean().current(), audit[0] / n, 1e-12);
      Assert.assertEquals(all.rowMean().shadow(), audit[1] / n, 1e-12);
      Assert.assertEquals(all.playedMinusOthers().current(), audit[2] / n, 1e-12);
      Assert.assertEquals(all.playedMinusOthers().shadow(), audit[3] / n, 1e-12);
      Assert.assertTrue(all.rowMean().seCurrent() > 0 && all.playedMinusOthers().seShadow() > 0);
      Assert.assertEquals(audits.get("main").rowMean(), all.rowMean());
      Assert.assertEquals(audits.get("rollin").rows(), 0);
      Assert.assertNull(audits.get("rollin").rowMean().current());
      Assert.assertEquals(audits.get("last").rows(), last);
      Assert.assertEquals(audits.get("CALL").rows() + audits.get("DISCARD_MID").rows(), (long) n);
    }
  }

  /**
   * 補正率は 4 点の係数の双線形の補間を [0, 1] に収めた値で、係数が全て 1 の格子ではどの局面でもちょうど 1、全て 0 ではちょうど 0
   * （重みの和の丸めで 1 を越えない）。
   */
  @Test
  public void theScaleStaysInsideTheUnitInterval() {
    for (Variant variant : List.of(Variant.TENHOU_4P, Variant.TENHOU_3P)) {
      int size = new LedgerCalibrator.Grid(variant, 9, 3).size();
      LedgerCalibrator ones = calibrator(variant, Collections.nCopies(size, 1.0));
      LedgerCalibrator zeros = calibrator(variant, Collections.nCopies(size, 0.0));
      for (int rank = 0; rank < variant.seats(); rank++)
        for (int gap = 0; gap <= 60_000; gap += 100)
          for (int round = 0; round <= variant.finalRound(); round++) {
            var standing = new LedgerCell.Standing(rank, gap, round);
            Assert.assertEquals(ones.scale(standing), 1.0, standing.toString());
            Assert.assertEquals(zeros.scale(standing), 0.0, standing.toString());
          }
    }
  }

  /**
   * 局面で補正の効き方が変わる行（点差が小さいほど Y が帳簿を強く含む）から当てはめた補正率は、点差の大きい首位で小さく、点差の小さい局面で
   * 大きい。前のコホートで当てはめた補正率を別の半荘の行に掛けると、帳簿の補正より J が下がる。
   */
  @Test
  public void theScaleFollowsWhereTheLedgerExplainsTheLabel() {
    Variant variant = Variant.TENHOU_4P;
    var fitter =
        new LedgerFit(LedgerGroup.computed(), 0.001, 32, LedgerCells.CLASS_STAGE_RANK, variant);
    var before = scaledRows(variant, 20_000, 0, 31);
    var pass =
        new LedgerCalibrator.Pass(fitter.fitRows(before).correction(), null, DEFAULTS, variant);
    for (var row : before) pass.add(row);
    LedgerCalibrator scale = fit(pass.sums(), DEFAULTS, variant);
    double close = scale.scale(new LedgerCell.Standing(0, 1000, 7));
    double decided = scale.scale(new LedgerCell.Standing(0, 40_000, 7));
    Assert.assertTrue(close > 0.6, "close " + close);
    Assert.assertTrue(decided < 0.25, "decided " + decided);
    var after = scaledRows(variant, 20_000, 1_000_000, 32);
    var held =
        new LedgerCalibrator.Pass(fitter.fitRows(after).correction(), scale, DEFAULTS, variant);
    for (var row : after) held.add(row);
    LedgerReport.Calibration report =
        held.report(
            "SHADOW", LedgerCalibrationKind.COMMON_SCALE, LedgerReport.Arm.CLASS, null, null);
    Assert.assertTrue(report.j().shadow() < report.j().current(), report.j().toString());
  }

  /**
   * 和了の判断の差。卓を打ち進めて集めた、南場の最後の局から後で和了が合法な O から、和了と比べる手を測った主経路の FINAL_END の行を作る。
   * ロンの比べる手は見送り、ツモの比べる手は測った打牌のうち s が最大の手で、3 つの腕の差の平均と標準偏差は直に求めた値と合う。s を渡さなければ
   * ツモの帯は無く、ロンの帯は変わらない。学習用でない行・主経路でない行・FINAL_END でない行は数えない。
   */
  @Test
  public void gatesCompareTheWinWithTheChosenAlternative() {
    Variant variant = Variant.TENHOU_4P;
    var o = new Observation(variant);
    List<long[]> wins =
        observations(
            variant,
            101,
            16,
            view -> {
              if (view.kyoku() < variant.lastRound()) return false;
              boolean tsumo = false, ron = false;
              for (int a = 0; a < view.legalActions(); a++) {
                Action.Type type = view.action(a).type();
                tsumo |= type == Action.Type.TSUMO_AGARI;
                ron |= type == Action.Type.RON_AGARI;
              }
              return tsumo || ron;
            });
    Assert.assertTrue(wins.size() >= 6, "found " + wins.size());
    var random = new SplittableRandom(41);
    double[][][] beta = new double[2][DecisionClass.COUNT][F];
    for (int f = 0; f < F; f++)
      beta[0][DecisionClass.FINAL_END.ordinal()][f] =
          beta[1][DecisionClass.FINAL_END.ordinal()][f] = 0.1 * (f % 4 + 1);
    var current = new LedgerCorrection(beta);
    LedgerCalibrator scale = calibrator(variant, Collections.nCopies(108, 0.25));
    List<Row> rows = new ArrayList<>();
    List<float[]> scores = new ArrayList<>();
    Map<String, List<Double>> expected = new HashMap<>();
    int tsumoRows = 0, ronRows = 0;
    for (int i = 0; i < wins.size(); i++) {
      long[] words = wins.get(i);
      o.bind(words, 0);
      int n = o.legalActions();
      float[] values = new float[n], ledger = new float[n * F], s = new float[n];
      Arrays.fill(values, Float.NaN);
      Arrays.fill(ledger, Float.NaN);
      int win = -1;
      for (int a = 0; a < n; a++) {
        Action.Type type = o.action(a).type();
        s[a] = (float) random.nextGaussian();
        if (win < 0 && (type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI)) win = a;
      }
      boolean tsumo = o.action(win).type() == Action.Type.TSUMO_AGARI;
      int other = -1;
      for (int a = 0; a < n; a++) {
        Action.Type type = o.action(a).type();
        boolean compared =
            tsumo
                ? type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI
                : type == Action.Type.PASS;
        if (a == win || compared) {
          values[a] = half(random.nextDouble());
          for (int f = 0; f < F; f++) ledger[a * F + f] = half(random.nextGaussian());
        }
        if (compared && (other < 0 || s[a] > s[other])) other = a;
      }
      if (other < 0) continue;
      Row row = row(1000 + i, DecisionClass.FINAL_END, Row.Source.MAIN, words, values, ledger, win);
      rows.add(row);
      scores.add(s);
      double db = 0;
      for (int f = 0; f < F; f++)
        db +=
            beta[0][DecisionClass.FINAL_END.ordinal()][f]
                * ((double) ledger[win * F + f] - ledger[other * F + f]);
      double dy = (double) values[win] - values[other];
      String kind = tsumo ? "tsumo" : "ron";
      if (tsumo) tsumoRows++;
      else ronRows++;
      expected.computeIfAbsent("raw " + kind, k -> new ArrayList<>()).add(dy);
      expected.computeIfAbsent("current " + kind, k -> new ArrayList<>()).add(dy - db);
      expected.computeIfAbsent("shadow " + kind, k -> new ArrayList<>()).add(dy - 0.25 * db);
      expected.computeIfAbsent("correction " + kind, k -> new ArrayList<>()).add(db);
    }
    rows.add(
        row(
            5,
            DecisionClass.END,
            Row.Source.MAIN,
            wins.get(0),
            rows.get(0).values(),
            rows.get(0).ledger(),
            rows.get(0).played()));
    rows.add(
        row(
            6,
            DecisionClass.FINAL_END,
            Row.Source.ROLLIN,
            wins.get(0),
            rows.get(0).values(),
            rows.get(0).ledger(),
            rows.get(0).played()));
    scores.add(scores.get(0));
    scores.add(scores.get(0));

    var withScores = new LedgerCalibrator.Gates(current, scale, variant);
    var withoutScores = new LedgerCalibrator.Gates(current, scale, variant);
    for (int i = 0; i < rows.size(); i++) {
      withScores.add(rows.get(i), scores.get(i));
      withoutScores.add(rows.get(i), null);
    }
    Map<String, LedgerReport.Gate> gates = new HashMap<>(), ronOnly = new HashMap<>();
    for (LedgerReport.Gate gate : withScores.gates())
      gates.put(gate.group() + " " + gate.kind(), gate);
    for (LedgerReport.Gate gate : withoutScores.gates())
      ronOnly.put(gate.group() + " " + gate.kind(), gate);
    for (String kind : List.of("tsumo", "ron")) {
      int count = kind.equals("tsumo") ? tsumoRows : ronRows;
      LedgerReport.Gate gate = gates.get("FINAL_END " + kind);
      if (count < 2) {
        Assert.assertNull(gate, kind);
        continue;
      }
      Assert.assertEquals(gate.rows(), count, kind);
      Assert.assertEquals(gate.mean().raw(), mean(expected.get("raw " + kind)), 1e-12);
      Assert.assertEquals(gate.sd().raw(), sd(expected.get("raw " + kind)), 1e-12);
      Assert.assertEquals(gate.sd().current(), sd(expected.get("current " + kind)), 1e-12);
      Assert.assertEquals(gate.sd().shadow(), sd(expected.get("shadow " + kind)), 1e-12);
      Assert.assertEquals(gate.shadowMinusRawSd(), gate.sd().shadow() - gate.sd().raw(), 1e-15);
      double correction = mean(expected.get("correction " + kind));
      Assert.assertEquals(gate.correction().current(), correction, 1e-12);
      Assert.assertEquals(gate.correction().shadow(), 0.25 * correction, 1e-12);
    }
    Assert.assertTrue(tsumoRows + ronRows >= 6, tsumoRows + " " + ronRows);
    for (String key : ronOnly.keySet()) Assert.assertTrue(key.endsWith(" ron"), key);
    for (var entry : gates.entrySet())
      if (entry.getKey().endsWith(" ron"))
        Assert.assertEquals(ronOnly.get(entry.getKey()), entry.getValue(), entry.getKey());
  }

  /** 校正器を作る（学習元は前のコホートでコホート 7、帳簿の全ての列と 5 組、クラスごとの区分、9 × 3 の格子、設計の罰の値）。 */
  static LedgerCalibrator calibrator(Variant variant, List<Double> coefficients) {
    return new LedgerCalibrator(
        LedgerCalibrationKind.COMMON_SCALE,
        LedgerCalibrationSource.PREVIOUS_COHORT,
        variant,
        7,
        LedgerCatalog.of(variant).columns(),
        LedgerGroup.computed(),
        LedgerCells.CLASS,
        9,
        3,
        1e-4,
        0.08,
        500,
        coefficients);
  }

  /** 和から当てはめる（コホート 7、帳簿の全ての列と 5 組、クラスごとの区分）。 */
  static LedgerCalibrator fit(
      LedgerCalibrator.Sums sums, LedgerCalibration.Scaling settings, Variant variant) {
    return LedgerCalibrator.fit(
        sums,
        settings,
        variant,
        7,
        LedgerCatalog.of(variant).columns(),
        LedgerGroup.computed(),
        LedgerCells.CLASS);
  }

  private static LedgerCalibration.Shadow withShrink(double rho) {
    return new LedgerCalibration.Shadow(
        LedgerCalibrationKind.COMMON_SCALE,
        LedgerCalibrationSource.PREVIOUS_COHORT,
        9,
        3,
        1e-4,
        rho);
  }

  /** ランダムな局面（点差は 0〜6 万点、局は西場の最後まで）。 */
  static LedgerCell.Standing standing(SplittableRandom random, Variant variant) {
    return new LedgerCell.Standing(
        random.nextInt(variant.seats()),
        random.nextInt(60_001),
        random.nextInt(variant.finalRound() + 1));
  }

  /**
   * 局面ごとに帳簿の効き方が違う行。3 手を測り、帳簿の列 f は N(0, 1) を fp16 に丸めた値。Y_a = 0.1·a + s(o)·β*·L_a + N(0, 0.2²) で、
   * s(o) は点差 8000 点以下で 1、24000 点以上で 0、間は 1 次式。区分は局面の局と順位から決める。
   */
  static List<LedgerFitTest.SyntheticRow> scaledRows(
      Variant variant, int count, long firstGame, long seed) {
    var random = new SplittableRandom(seed);
    double[] beta = LedgerFitTest.planted();
    LedgerCell cells = LedgerCell.of(variant);
    List<LedgerFitTest.SyntheticRow> rows = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      LedgerCell.Standing standing = standing(random, variant);
      double strength = Math.min(1, Math.max(0, (24_000 - standing.gap()) / 16_000.0));
      float[] values = new float[4], ledger = new float[4 * F];
      Arrays.fill(values, Float.NaN);
      Arrays.fill(ledger, Float.NaN);
      for (int a = 0; a < 3; a++) {
        double y = 0.1 * a + 0.2 * random.nextGaussian();
        for (int f = 0; f < F; f++) {
          float l = half(random.nextGaussian());
          ledger[a * F + f] = l;
          y += strength * beta[f] * l;
        }
        values[a] = (float) y;
      }
      DecisionClass c = i % 3 == 0 ? DecisionClass.CALL : DecisionClass.DISCARD_MID;
      rows.add(
          new LedgerFitTest.SyntheticRow(
              firstGame + i,
              c,
              values,
              ledger,
              cells.index(standing.round(), standing.rank()),
              standing));
    }
    return rows;
  }

  /** fp16 に丸めた値。 */
  static float half(double value) {
    return Float.float16ToFloat(Float.floatToFloat16((float) value));
  }

  /**
   * 卓を打ち進め、条件に合う判断の O を集める（1 半荘ずつ種を変えて、{@code count} 個まで。半荘は最大 200 個）。
   *
   * @param keep O を残す条件
   */
  static List<long[]> observations(
      Variant variant, long seed, int count, Predicate<Observation> keep) {
    var writer = new ObservationWriter(variant);
    var batch = new ObservationBatch(variant, 1 << 12);
    var view = new Observation(variant);
    var walk = new TableWalk(variant, TableWalk.Policy.GREEDY);
    List<long[]> kept = new ArrayList<>();
    for (long game = seed; kept.size() < count && game < seed + 200; game++) {
      walk.run(
          game,
          1,
          (engine, slot) -> {
            if (kept.size() >= count) return;
            int seat = engine.decisions().seat(slot);
            batch.clear();
            int index = writer.capture(engine.view(seat), engine.decisions(), slot, batch);
            int start = batch.start(index);
            long[] words = Arrays.copyOfRange(batch.words(), start, start + batch.longs(index));
            if (keep.test(view.bind(words, 0))) kept.add(words);
          });
    }
    return kept;
  }

  /** 学習用の行（測った手を候補にし、局末の記録と Oracle は持たない）。 */
  static Row row(
      long gameId,
      DecisionClass decisionClass,
      Row.Source source,
      long[] words,
      float[] values,
      float[] ledger,
      int played) {
    int actions = values.length, support = 0;
    for (int a = 0; a < actions; a++) if (!Float.isNaN(values[a])) support |= 1 << a;
    return new Row(
        gameId,
        gameId,
        decisionClass,
        Row.Purpose.TRAIN,
        source,
        played,
        values,
        new float[actions],
        support,
        -1,
        Float.NaN,
        Float.NaN,
        -1,
        ledger,
        new long[actions * OutcomeCodec.WORDS],
        new long[0],
        words);
  }

  private static double[][] positiveDefinite(int n, SplittableRandom random) {
    double[][] a = new double[n][n], q = new double[n][n];
    for (double[] row : a) for (int j = 0; j < n; j++) row[j] = random.nextGaussian();
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) q[i][j] += a[k][i] * a[k][j];
        if (i == j) q[i][j] += 0.1;
      }
    return q;
  }

  /** 全ての変数を下限・上限・自由のどれかに置いた 3^n 通りのうち、自由な変数の連立方程式の解が箱に入り、f が最小のもの。 */
  private static double[] bruteForce(double[][] q, double[] c) {
    int n = c.length, patterns = (int) Math.pow(3, n);
    double[] best = null;
    double bestValue = Double.POSITIVE_INFINITY;
    for (int pattern = 0; pattern < patterns; pattern++) {
      int[] state = new int[n];
      for (int j = 0, p = pattern; j < n; j++, p /= 3) state[j] = p % 3;
      double[] theta = new double[n];
      List<Integer> free = new ArrayList<>();
      for (int j = 0; j < n; j++) {
        if (state[j] == 2) free.add(j);
        else theta[j] = state[j];
      }
      int k = free.size();
      double[][] m = new double[k][k + 1];
      for (int i = 0; i < k; i++) {
        int j = free.get(i);
        double rhs = c[j];
        for (int b = 0; b < n; b++) if (state[b] != 2) rhs -= q[j][b] * theta[b];
        for (int p = 0; p < k; p++) m[i][p] = q[j][free.get(p)];
        m[i][k] = rhs;
      }
      double[] z = gauss(m);
      boolean inside = true;
      for (int i = 0; i < k; i++) {
        inside &= z[i] >= 0 && z[i] <= 1;
        theta[free.get(i)] = z[i];
      }
      if (!inside) continue;
      double value = objective(q, c, theta);
      if (value < bestValue) {
        bestValue = value;
        best = theta;
      }
    }
    return best;
  }

  /** 拡大係数行列のガウスの消去（部分ピボット）。 */
  private static double[] gauss(double[][] m) {
    int k = m.length;
    for (int col = 0; col < k; col++) {
      int pivot = col;
      for (int row = col + 1; row < k; row++)
        if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
      double[] swap = m[col];
      m[col] = m[pivot];
      m[pivot] = swap;
      for (int row = col + 1; row < k; row++) {
        double factor = m[row][col] / m[col][col];
        for (int j = col; j <= k; j++) m[row][j] -= factor * m[col][j];
      }
    }
    double[] z = new double[k];
    for (int row = k - 1; row >= 0; row--) {
      double sum = m[row][k];
      for (int j = row + 1; j < k; j++) sum -= m[row][j] * z[j];
      z[row] = sum / m[row][row];
    }
    return z;
  }

  private static double objective(double[][] q, double[] c, double[] theta) {
    double value = 0;
    for (int j = 0; j < c.length; j++) {
      value -= 2 * c[j] * theta[j];
      for (int k = 0; k < c.length; k++) value += theta[j] * q[j][k] * theta[k];
    }
    return value;
  }

  private static double mean(List<Double> values) {
    double total = 0;
    for (double value : values) total += value;
    return total / values.size();
  }

  private static double sd(List<Double> values) {
    double mean = mean(values), total = 0;
    for (double value : values) total += (value - mean) * (value - mean);
    return Math.sqrt(total / (values.size() - 1));
  }
}
