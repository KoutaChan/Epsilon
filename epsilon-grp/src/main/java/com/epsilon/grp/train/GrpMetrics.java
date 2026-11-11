package com.epsilon.grp.train;

import com.epsilon.grp.GrpPhase;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;

/**
 * GRP を評価する指標（純 Java）。Φ 1 つの較正を測り（{@link GrpReport}）、2 つの Φ を同じ局末で比べた差も求める。数える単位は
 * 「席 × 半荘が続く局末」で、標準誤差は全て半荘をクラスタとして求める（Clusters）。区分、今の順位、点差の決め方は HandEnds に従う。
 * セルの切り方と式はコードの定数で決める。
 */
public final class GrpMetrics {

  /** 偏りの二乗（区分 × 今の順位 × 点差のセル）で点差を区間に分ける境。境の値は上の区間に入る。 */
  static final int[] GAP_EDGES = {2_000, 6_000, 12_000};

  static final int GAP_BINS = GAP_EDGES.length + 1;

  /** 偏りの二乗（区分 × Φ の区間のセル）で Φ を分ける等幅の区間の数。範囲は最小の効用から最大の効用まで。 */
  private static final int PHI_BINS = 10;

  /** 最下位になる確率の ECE で使う、等幅の区間の数。 */
  private static final int ECE_BINS = 15;

  /** 周辺確率の NLL で使う確率の下限。点数だけで決めた順位（確率が 0 か 1）でも NLL を有限にする。 */
  private static final double PROBABILITY_FLOOR = 1e-9;

  /** 終盤とみなす区分の数（最後の区分から数える）。 */
  private static final int LATE_GROUPS = 2;

  /** 端の判定。確率の下の端と上の端、大差とみなす点差。 */
  private static final double LOW_TAIL = 0.01;

  private static final double HIGH_TAIL = 0.95;
  private static final int LEAD = 12_000;

  /** 僅差のセルで点差を分ける区間の境。 */
  private static final int[] CLOSE_EDGES = {0, 2_000, 6_000};

  private static final int CLOSE_BINS = CLOSE_EDGES.length - 1;

  /** 段差を見る点差の区間の幅と数。12,000 点以上は見ない。 */
  private static final int JUMP_STEP = 500;

  private static final int JUMP_BINS = 24;

  /** 段差の検定で 3 次式に入れる点差の単位。 */
  private static final double STEP_UNIT = 10_000;

  /**
   * 端のセル（GrpReport.Tails の 8 つの値のうち、確率と Φ の 6 つ）。確率の上の端（p が 95% 以上）は外れる側（予測は 1 − p、
   * 実測はその順位で終わらなかった率）を足すので、確率の端の基準はどちらも稀な側で比べる。
   */
  private enum TailCell {
    LOW_PROBABILITY,
    HIGH_PROBABILITY,
    LEADER_LAST,
    LAST_FIRST,
    LOW_PHI,
    HIGH_PHI
  }

  private static final int TAIL_CELLS = TailCell.values().length;

  private GrpMetrics() {}

  /** {@link #residual} で、区分・今の順位・点差（HandEnds の点差）から席と局末を選ぶ条件。 */
  @FunctionalInterface
  public interface SeatFilter {
    boolean test(int group, int rank, int gap);
  }

  /** Φ 1 つの較正を、全ての指標で測る。同点での段差は TieJumps で測る。 */
  public static GrpReport measure(HandEnds ends, HandEndValues values) {
    return measure(ends, values, handEnd -> true);
  }

  /**
   * 選んだ局末だけで較正を測る（区間ごとの僅差のセルや傾きを見るときなど）。報告する半荘の数は、選んだ局末を持つ半荘の数。
   * クラスタは変わらず半荘。
   *
   * @param selected 測る局末を選ぶ条件
   */
  public static GrpReport measure(HandEnds ends, HandEndValues values, IntPredicate selected) {
    requireValues(ends, values);
    int seats = ends.seats;
    int last = seats - 1;
    int firstLate = ends.groups - LATE_GROUPS;
    double low = ends.lowestUtility();
    double width = ends.highestUtility() - low;
    Clusters spg = new Clusters(ends.groups * seats * GAP_BINS);
    Clusters sb = new Clusters(ends.groups * PHI_BINS);
    Clusters close = new Clusters(LATE_GROUPS * seats * CLOSE_BINS);
    Clusters tails = new Clusters(TAIL_CELLS);
    double[] predictedTails = new double[TAIL_CELLS];
    AdjacentBins keepJumps = new AdjacentBins(LATE_GROUPS * seats, JUMP_BINS);
    AdjacentBins phiJumps = new AdjacentBins(LATE_GROUPS * seats, JUMP_BINS);
    Fits groupFits = new Fits(ends.groups);
    Fits phaseFits = new Fits(GrpPhase.values().length);
    double[] eceConfidence = new double[ECE_BINS];
    double[] eceOutcome = new double[ECE_BINS];
    double[] eceCount = new double[ECE_BINS];
    double squares = 0;
    double nll = 0;
    double brier = 0;
    double jointNll = 0;
    double minPhi = Double.POSITIVE_INFINITY;
    double maxPhi = Double.NEGATIVE_INFINITY;
    boolean joint = values.joint();
    long games = 0;
    long handEnds = 0;
    for (int game = 0; game < ends.games(); game++) {
      long before = handEnds;
      for (int h = ends.firstHand[game]; h < ends.firstHand[game + 1]; h++) {
        if (!selected.test(h)) {
          continue;
        }
        handEnds++;
        int group = ends.group[h];
        int late = group - firstLate;
        if (joint) {
          jointNll -= values.jointLog[h];
        }
        for (int seat = 0; seat < seats; seat++) {
          int j = h * seats + seat;
          int place = ends.rank[j];
          int finished = ends.finalRank[j];
          int gap = ends.gap[j];
          double phi = values.phi[j];
          double u = ends.utilities[finished];
          double residual = u - phi;
          int base = j * seats;
          squares += residual * residual;
          spg.add(spgCell(ends, group, place, gap), residual);
          sb.add(group * PHI_BINS + phiBin(phi, low, width), residual);
          groupFits.add(group, phi, u, place == last);
          phaseFits.add(ends.phase[h], phi, u, place == last);
          minPhi = Math.min(minPhi, phi);
          maxPhi = Math.max(maxPhi, phi);
          nll -= Math.log(Math.max(values.marginals[base + finished], PROBABILITY_FLOOR));
          for (int rank = 0; rank < seats; rank++) {
            double probability = values.marginals[base + rank];
            double outcome = rank == finished ? 1 : 0;
            brier += (probability - outcome) * (probability - outcome);
            if (probability <= LOW_TAIL) {
              tail(tails, predictedTails, TailCell.LOW_PROBABILITY, probability, outcome);
            } else if (probability >= HIGH_TAIL) {
              tail(tails, predictedTails, TailCell.HIGH_PROBABILITY, 1 - probability, 1 - outcome);
            }
          }
          double fourth = values.marginals[base + last];
          int eceBin = Math.min(ECE_BINS - 1, (int) (fourth * ECE_BINS));
          eceConfidence[eceBin] += fourth;
          eceOutcome[eceBin] += finished == last ? 1 : 0;
          eceCount[eceBin]++;
          double scaled = (phi - low) / width;
          if (scaled <= LOW_TAIL) {
            tail(tails, predictedTails, TailCell.LOW_PHI, phi, u);
          } else if (scaled >= HIGH_TAIL) {
            tail(tails, predictedTails, TailCell.HIGH_PHI, phi, u);
          }
          if (late < 0) {
            continue;
          }
          double keep = finished == place ? 1 : 0;
          double keepResidual = keep - values.marginals[base + place];
          int closeBin = closeBin(gap);
          if (closeBin >= 0) {
            close.add((late * seats + place) * CLOSE_BINS + closeBin, keepResidual);
          }
          if (gap < JUMP_STEP * JUMP_BINS) {
            keepJumps.add(late * seats + place, gap / JUMP_STEP, keepResidual);
            phiJumps.add(late * seats + place, gap / JUMP_STEP, residual);
          }
          if (place == 0 && gap >= LEAD) {
            tail(tails, predictedTails, TailCell.LEADER_LAST, values.marginals[base + last], finished == last ? 1 : 0);
          } else if (place == last && gap >= LEAD) {
            tail(tails, predictedTails, TailCell.LAST_FIRST, values.marginals[base], finished == 0 ? 1 : 0);
          }
        }
      }
      games += handEnds > before ? 1 : 0;
      spg.endGame();
      sb.endGame();
      close.endGame();
      tails.endGame();
      keepJumps.endGame();
      phiJumps.endGame();
    }
    groupFits.solve();
    phaseFits.solve();
    for (int game = 0; game < ends.games(); game++) {
      for (int h = ends.firstHand[game]; h < ends.firstHand[game + 1]; h++) {
        if (!selected.test(h)) {
          continue;
        }
        for (int seat = 0; seat < seats; seat++) {
          int j = h * seats + seat;
          double u = ends.utilities[ends.finalRank[j]];
          groupFits.score(ends.group[h], values.phi[j], u);
          phaseFits.score(ends.phase[h], values.phi[j], u);
        }
      }
      groupFits.endGame();
      phaseFits.endGame();
    }
    double seatHands = (double) handEnds * seats;
    double ece = 0;
    for (int bin = 0; bin < ECE_BINS; bin++) {
      if (eceCount[bin] > 0) {
        ece += eceCount[bin] / seatHands * Math.abs((eceConfidence[bin] - eceOutcome[bin]) / eceCount[bin]);
      }
    }
    GrpReport.Tails tailReport = new GrpReport.Tails(
        tailOf(tails, predictedTails, TailCell.LOW_PROBABILITY),
        tailOf(tails, predictedTails, TailCell.HIGH_PROBABILITY),
        tailOf(tails, predictedTails, TailCell.LEADER_LAST),
        tailOf(tails, predictedTails, TailCell.LAST_FIRST),
        minPhi,
        maxPhi,
        tailOf(tails, predictedTails, TailCell.LOW_PHI),
        tailOf(tails, predictedTails, TailCell.HIGH_PHI));
    return new GrpReport(
        games,
        handEnds * seats,
        squares / seatHands,
        spg.distance(seatHands),
        sb.distance(seatHands),
        groupFits.fits(),
        phaseFits.fits(),
        nll / seatHands,
        brier / seatHands,
        ece,
        joint ? jointNll / handEnds : Double.NaN,
        tailReport,
        closeCells(ends, close),
        jumps(ends, keepJumps, phiJumps));
  }

  /** 残差の二乗の平均。 */
  public static double mse(HandEnds ends, HandEndValues values) {
    requireValues(ends, values);
    double squares = 0;
    for (int j = 0; j < values.phi.length; j++) {
      double residual = ends.utilities[ends.finalRank[j]] - values.phi[j];
      squares += residual * residual;
    }
    return squares / ends.seatHands();
  }

  /** 同じ局末で比べた (u − Φ)² の差（候補 − 基準）の平均と、その標準誤差。 */
  public static Paired pairedMse(HandEnds ends, HandEndValues reference, HandEndValues candidate) {
    return pairedMse(ends, reference, candidate, handEnd -> true);
  }

  /**
   * 選んだ局末だけで比べた (u − Φ)² の差（候補 − 基準）の平均と、その標準誤差。
   *
   * @param selected 比べる局末を選ぶ条件
   */
  public static Paired pairedMse(
      HandEnds ends, HandEndValues reference, HandEndValues candidate, IntPredicate selected) {
    requireValues(ends, reference);
    requireValues(ends, candidate);
    int seats = ends.seats;
    Clusters cluster = new Clusters(1);
    for (int game = 0; game < ends.games(); game++) {
      for (int h = ends.firstHand[game]; h < ends.firstHand[game + 1]; h++) {
        if (!selected.test(h)) {
          continue;
        }
        for (int j = h * seats; j < (h + 1) * seats; j++) {
          double u = ends.utilities[ends.finalRank[j]];
          double ofCandidate = u - candidate.phi[j];
          double ofReference = u - reference.phi[j];
          cluster.add(0, ofCandidate * ofCandidate - ofReference * ofReference);
        }
      }
      cluster.endGame();
    }
    return cluster.paired(0);
  }

  /** 席ごとの最終順位の周辺確率で、NLL の差（候補 − 基準。席・局末あたり）の平均と、その標準誤差。 */
  public static Paired pairedRankNll(HandEnds ends, HandEndValues reference, HandEndValues candidate) {
    requireValues(ends, reference);
    requireValues(ends, candidate);
    int seats = ends.seats;
    Clusters cluster = new Clusters(1);
    for (int game = 0; game < ends.games(); game++) {
      for (int j = ends.firstHand[game] * seats; j < ends.firstHand[game + 1] * seats; j++) {
        int index = j * seats + ends.finalRank[j];
        cluster.add(
            0,
            Math.log(Math.max(reference.marginals[index], PROBABILITY_FLOOR))
                - Math.log(Math.max(candidate.marginals[index], PROBABILITY_FLOOR)));
      }
      cluster.endGame();
    }
    return cluster.paired(0);
  }

  /** 揺れ。2 つの Φ を同じ局末で比べた差の RMS（席・局末あたり）。 */
  public static double wobble(HandEnds ends, HandEndValues first, HandEndValues second) {
    requireValues(ends, first);
    requireValues(ends, second);
    double squares = 0;
    for (int j = 0; j < first.phi.length; j++) {
      double difference = first.phi[j] - second.phi[j];
      squares += difference * difference;
    }
    return Math.sqrt(squares / ends.seatHands());
  }

  /** 選んだ席・局末での残差の平均と、その標準誤差。 */
  public static Paired residual(HandEnds ends, HandEndValues values, SeatFilter filter) {
    requireValues(ends, values);
    Clusters cluster = new Clusters(1);
    for (int game = 0; game < ends.games(); game++) {
      for (int h = ends.firstHand[game]; h < ends.firstHand[game + 1]; h++) {
        for (int seat = 0; seat < ends.seats; seat++) {
          int j = h * ends.seats + seat;
          if (filter.test(ends.group[h], ends.rank[j], ends.gap[j])) {
            cluster.add(0, ends.utilities[ends.finalRank[j]] - values.phi[j]);
          }
        }
      }
      cluster.endGame();
    }
    return cluster.paired(0);
  }

  /** 値がその局末について求めたものでなければ IllegalArgumentException を投げる。 */
  static void requireValues(HandEnds ends, HandEndValues values) {
    if (values.ends != ends) {
      throw new IllegalArgumentException("The values were computed for other hand ends");
    }
  }

  private static List<GrpReport.CloseCell> closeCells(HandEnds ends, Clusters close) {
    int seats = ends.seats;
    List<GrpReport.CloseCell> cells = new ArrayList<>();
    for (int late = 0; late < LATE_GROUPS; late++) {
      for (int place = seats - 1; place >= seats - 2; place--) {
        for (int bin = 0; bin < CLOSE_BINS; bin++) {
          int cell = (late * seats + place) * CLOSE_BINS + bin;
          cells.add(new GrpReport.CloseCell(
              ends.groups - LATE_GROUPS + late,
              place,
              CLOSE_EDGES[bin],
              CLOSE_EDGES[bin + 1],
              (long) close.count(cell),
              close.mean(cell),
              Math.sqrt(close.variance(cell))));
        }
      }
    }
    return cells;
  }

  private static List<GrpReport.JumpSeries> jumps(HandEnds ends, AdjacentBins keep, AdjacentBins phi) {
    int seats = ends.seats;
    Paired[][] keepJumps = keep.differences();
    Paired[][] phiJumps = phi.differences();
    List<GrpReport.JumpSeries> series = new ArrayList<>();
    for (int late = 0; late < LATE_GROUPS; late++) {
      for (int place = 0; place < seats; place++) {
        int line = late * seats + place;
        int group = ends.groups - LATE_GROUPS + late;
        double step = StepTest.chiSquarePerDf(keep, line, JUMP_STEP / STEP_UNIT);
        series.add(jumpSeries(group, place, GrpReport.JumpMeasure.KEEP, keepJumps[line], step));
        series.add(jumpSeries(group, place, GrpReport.JumpMeasure.PHI, phiJumps[line], Double.NaN));
      }
    }
    return series;
  }

  private static GrpReport.JumpSeries jumpSeries(
      int group, int place, GrpReport.JumpMeasure measure, Paired[] jumps, double step) {
    double chiSquare = 0;
    double maxAbsZ = 0;
    int pairs = 0;
    for (Paired jump : jumps) {
      double z = jump.z();
      if (Double.isFinite(z)) {
        chiSquare += z * z;
        maxAbsZ = Math.max(maxAbsZ, Math.abs(z));
        pairs++;
      }
    }
    if (pairs == 0) {
      return new GrpReport.JumpSeries(group, place, measure, 0, Double.NaN, Double.NaN, step);
    }
    return new GrpReport.JumpSeries(group, place, measure, pairs, chiSquare / pairs, maxAbsZ, step);
  }

  private static void tail(Clusters tails, double[] predicted, TailCell cell, double prediction, double outcome) {
    tails.add(cell.ordinal(), outcome);
    predicted[cell.ordinal()] += prediction;
  }

  private static GrpReport.Tail tailOf(Clusters tails, double[] predicted, TailCell cell) {
    double rows = tails.count(cell.ordinal());
    return new GrpReport.Tail(
        (long) rows,
        rows > 0 ? predicted[cell.ordinal()] / rows : Double.NaN,
        tails.mean(cell.ordinal()),
        Math.sqrt(tails.variance(cell.ordinal())));
  }

  /** 偏りの二乗を求めるセル（区分 × 今の順位 × 点差の 4 区間）。 */
  static int spgCell(HandEnds ends, int group, int rank, int gap) {
    int bin = 0;
    while (bin < GAP_EDGES.length && gap >= GAP_EDGES[bin]) {
      bin++;
    }
    return (group * ends.seats + rank) * GAP_BINS + bin;
  }

  private static int phiBin(double phi, double low, double width) {
    return Math.max(0, Math.min(PHI_BINS - 1, (int) ((phi - low) / width * PHI_BINS)));
  }

  /** 僅差のセルの区間。どの区間にも入らなければ −1。 */
  private static int closeBin(int gap) {
    for (int bin = 0; bin < CLOSE_BINS; bin++) {
      if (gap >= CLOSE_EDGES[bin] && gap < CLOSE_EDGES[bin + 1]) {
        return bin;
      }
    }
    return -1;
  }
}
