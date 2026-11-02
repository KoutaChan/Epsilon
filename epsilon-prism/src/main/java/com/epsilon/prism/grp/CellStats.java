package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.ScoreRanking;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * 見張り・関門・信頼の条件で共通の較正の測り。升目はコードの定数（局 0..7、Φ の 10 区間、順位、隣との点差の区間 2000/6000/12000）。
 *
 * <p>局は終わった局の局番号（公開系列の最後から 2 つ目の局境界。{@code PhiCalibration} の段と audit-grp の CSV の kyoku と同じ。南 4 と西入は
 * まとめる）で、{@link HandGroup#of} に渡す。順位と隣との点差は精算後の点数（次局の点数）で決める。隣は 1 つ下の順位の席（4 位は 3 位の席）。
 * 升目の残差の平均の分散は半荘ごとのまとまりで取る: v_c = G_c/(G_c − 1) · Σ_g (S_gc − m_c n_gc)² / n_c²（S_gc・n_gc は半荘 g の升目 c の残差の
 * 和と数、G_c は升目 c に入る半荘の数）。G_c &lt; 2 の升目は D に入れない。
 */
public final class CellStats {
  private static final HandGroup[] GROUPS = HandGroup.values();

  /** 局の区分の数（{@link HandGroup}）。 */
  static final int KYOKU_CELLS = GROUPS.length;

  /** 隣との点差の区間の境（未満で区切る）。 */
  private static final int[] GAP_EDGES = {2000, 6000, 12000};

  /** 局 × 今の順位 × 隣との点差の升目の数（D_SPG）。 */
  static final int SPG_CELLS = KYOKU_CELLS * 4 * (GAP_EDGES.length + 1);

  /** Φ の値の等幅の区間の数（最下位から 1 位の効用まで）。 */
  static final int PHI_BINS = 10;

  /** 4 位の確率の ECE の等幅の区間の数（{@code EpsilonGrpInference} と同じ）。 */
  private static final int ECE_BINS = 15;

  /** NLL の確率の下限（0/1 の点数の順位でも有限にする）。 */
  private static final double PROBABILITY_FLOOR = 1e-9;

  /** 傾きの SE と z を出す区分の半荘の数の下限。 */
  static final int MIN_SLOPE_GAMES = 200;

  private CellStats() {}

  /**
   * 対の差（候補 − 使用中）。
   *
   * @param mseDifference 残差の二乗の平均の差
   * @param standardError その半荘ごとのまとまりの標準誤差
   */
  public record Paired(double mseDifference, double standardError) {}

  /**
   * 局 × Φ の区間の絶対の較正。
   *
   * @param rms 升目の残差の平均の重み付き RMS √(Σ_c (n_c/N) m_c²)
   * @param floor 雑音の床 √(Σ_c (n_c/N) v_c)（較正が完全でも残る分）
   * @param flagged |m_c| &gt; limit かつ |m_c|/√v_c &gt; z の升目の数
   */
  record Absolute(double rms, double floor, int flagged) {}

  /**
   * 升目ごとの残差の平均と標準誤差（局 × 今の順位 × 隣との点差、{@link #spgCell} の順）。
   *
   * @param counts 席局の数
   * @param means 残差の平均
   * @param errors 平均の半荘ごとの標準誤差（半荘が 2 未満なら NaN）
   */
  record CellTable(double[] counts, double[] means, double[] errors) {}

  /** 1 席局の升目の選び（{@link #residual}）。 */
  @FunctionalInterface
  interface SeatHandFilter {
    /**
     * @param group 終わった局の区分
     * @param rank 今の順位（0 = 1 位）
     * @param gap 隣との点差
     */
    boolean test(HandGroup group, int rank, int gap);
  }

  /**
   * 半荘の列と、その半荘が続く全ての局末の Φ（{@link HandEndPhi#gameValues} の順）の較正の測り。
   *
   * @param profile 最終の効用の順位効用
   */
  public static CellReport measure(
      List<RecordedGame> games, PhiValues phi, EpsilonUtilityProfile profile) {
    Layout layout = Layout.of(games, phi, profile);
    int seatHands = layout.seatHands();
    Cells spg = new Cells(SPG_CELLS), sb = new Cells(KYOKU_CELLS * PHI_BINS);
    double[] b4Sum = new double[KYOKU_CELLS], b4Count = new double[KYOKU_CELLS];
    double[][] regression = new double[KYOKU_CELLS][5];
    double[] eceConfidence = new double[ECE_BINS], eceOutcome = new double[ECE_BINS], eceCount = new double[ECE_BINS];
    double squares = 0, nll = 0, brier = 0;
    for (int g = 0; g < layout.games; g++) {
      for (int h = layout.firstHand[g]; h < layout.firstHand[g + 1]; h++) {
        int group = layout.group[h];
        double[] probabilities = phi.rankProbabilities()[h];
        for (int seat = 0; seat < 4; seat++) {
          int j = 4 * h + seat;
          double value = phi.utilities()[h][seat], u = layout.utility[j], r = u - value;
          squares += r * r;
          spg.add(layout.spgCell(j), r);
          sb.add(group * PHI_BINS + layout.phiBin(value), r);
          if (layout.rank[j] == 3) {
            b4Sum[group] += r;
            b4Count[group]++;
          }
          double[] sums = regression[group];
          sums[0]++;
          sums[1] += value;
          sums[2] += u;
          sums[3] += value * value;
          sums[4] += value * u;
          int finalRank = layout.finalRank[j];
          nll -= Math.log(Math.max(probabilities[4 * seat + finalRank], PROBABILITY_FLOOR));
          for (int rank = 0; rank < 4; rank++) {
            double error = probabilities[4 * seat + rank] - (rank == finalRank ? 1 : 0);
            brier += error * error;
          }
          double fourth = probabilities[4 * seat + 3];
          int bin = Math.min(ECE_BINS - 1, (int) (fourth * ECE_BINS));
          eceConfidence[bin] += fourth;
          eceOutcome[bin] += finalRank == 3 ? 1 : 0;
          eceCount[bin]++;
        }
      }
      spg.endGame();
      sb.endGame();
    }
    double[] b4 = new double[KYOKU_CELLS];
    for (int group = 0; group < KYOKU_CELLS; group++) b4[group] = b4Sum[group] / b4Count[group];
    double[][] slopes = slopes(layout, phi, regression);
    double ece = 0;
    for (int bin = 0; bin < ECE_BINS; bin++)
      if (eceCount[bin] > 0)
        ece += eceCount[bin] / seatHands * Math.abs((eceConfidence[bin] - eceOutcome[bin]) / eceCount[bin]);
    return new CellReport(
        layout.games,
        seatHands,
        squares / seatHands,
        spg.distance(seatHands),
        sb.distance(seatHands),
        b4,
        slopes[0],
        slopes[1],
        slopes[2],
        nll / seatHands,
        brier / seatHands,
        ece);
  }

  /** 同じ半荘の 2 つの Φ の対の MSE の差（候補 − 使用中）と半荘ごとの標準誤差。 */
  public static Paired paired(
      List<RecordedGame> games, PhiValues active, PhiValues candidate, EpsilonUtilityProfile profile) {
    Layout layout = Layout.of(games, candidate, profile);
    if (active.size() != candidate.size()) throw new IllegalArgumentException("Paired Φ differ in length");
    Cells cluster = new Cells(1);
    for (int g = 0; g < layout.games; g++) {
      for (int h = layout.firstHand[g]; h < layout.firstHand[g + 1]; h++)
        for (int seat = 0; seat < 4; seat++) {
          double u = layout.utility[4 * h + seat];
          double c = u - candidate.utilities()[h][seat], a = u - active.utilities()[h][seat];
          cluster.add(0, c * c - a * a);
        }
      cluster.endGame();
    }
    return cluster.paired();
  }

  /** 同じ半荘の 2 つの Φ の最終順位の NLL の対の差（候補 − 使用中、席局あたり）と半荘ごとの標準誤差。 */
  static Paired pairedNll(
      List<RecordedGame> games, PhiValues active, PhiValues candidate, EpsilonUtilityProfile profile) {
    Layout layout = Layout.of(games, candidate, profile);
    if (active.size() != candidate.size()) throw new IllegalArgumentException("Paired Φ differ in length");
    Cells cluster = new Cells(1);
    for (int g = 0; g < layout.games; g++) {
      for (int h = layout.firstHand[g]; h < layout.firstHand[g + 1]; h++)
        for (int seat = 0; seat < 4; seat++) {
          int index = 4 * seat + layout.finalRank[4 * h + seat];
          cluster.add(
              0,
              Math.log(Math.max(active.rankProbabilities()[h][index], PROBABILITY_FLOOR))
                  - Math.log(Math.max(candidate.rankProbabilities()[h][index], PROBABILITY_FLOOR)));
        }
      cluster.endGame();
    }
    return cluster.paired();
  }

  /**
   * 選んだ席局の残差の平均と半荘ごとの標準誤差（{@link Paired} の形で、mseDifference に平均を入れる）。
   *
   * @param filter 席局の選び
   */
  static Paired residual(
      List<RecordedGame> games, PhiValues phi, EpsilonUtilityProfile profile, SeatHandFilter filter) {
    Layout layout = Layout.of(games, phi, profile);
    Cells cluster = new Cells(1);
    for (int g = 0; g < layout.games; g++) {
      for (int h = layout.firstHand[g]; h < layout.firstHand[g + 1]; h++)
        for (int seat = 0; seat < 4; seat++) {
          int j = 4 * h + seat;
          if (filter.test(GROUPS[layout.group[h]], layout.rank[j], layout.gap[j]))
            cluster.add(0, layout.utility[j] - phi.utilities()[h][seat]);
        }
      cluster.endGame();
    }
    return cluster.paired();
  }

  /** 局 × 今の順位 × 隣との点差の升目ごとの残差の平均と標準誤差。 */
  static CellTable spgTable(List<RecordedGame> games, PhiValues phi, EpsilonUtilityProfile profile) {
    Layout layout = Layout.of(games, phi, profile);
    Cells spg = new Cells(SPG_CELLS);
    for (int g = 0; g < layout.games; g++) {
      for (int h = layout.firstHand[g]; h < layout.firstHand[g + 1]; h++)
        for (int seat = 0; seat < 4; seat++) {
          int j = 4 * h + seat;
          spg.add(layout.spgCell(j), layout.utility[j] - phi.utilities()[h][seat]);
        }
      spg.endGame();
    }
    double[] means = new double[SPG_CELLS], errors = new double[SPG_CELLS];
    for (int c = 0; c < SPG_CELLS; c++) {
      means[c] = spg.mean(c);
      errors[c] = Math.sqrt(spg.variance(c));
    }
    return new CellTable(spg.count.clone(), means, errors);
  }

  /**
   * 局 × Φ の区間の絶対の較正。
   *
   * @param limit 大きいとみなす升目の残差の平均
   * @param z 大きいとみなす升目の |m_c|/√v_c
   */
  static Absolute absolute(
      List<RecordedGame> games, PhiValues phi, EpsilonUtilityProfile profile, double limit, double z) {
    Layout layout = Layout.of(games, phi, profile);
    Cells sb = new Cells(KYOKU_CELLS * PHI_BINS);
    for (int g = 0; g < layout.games; g++) {
      for (int h = layout.firstHand[g]; h < layout.firstHand[g + 1]; h++)
        for (int seat = 0; seat < 4; seat++) {
          double value = phi.utilities()[h][seat];
          sb.add(layout.group[h] * PHI_BINS + layout.phiBin(value), layout.utility[4 * h + seat] - value);
        }
      sb.endGame();
    }
    double squares = 0, noise = 0;
    int flagged = 0;
    for (int c = 0; c < sb.count.length; c++) {
      double variance = sb.variance(c);
      if (!(variance >= 0)) continue;
      double weight = sb.count[c] / layout.seatHands(), mean = sb.mean(c);
      squares += weight * mean * mean;
      noise += weight * variance;
      if (Math.abs(mean) > limit && Math.abs(mean) > z * Math.sqrt(variance)) flagged++;
    }
    return new Absolute(Math.sqrt(squares), Math.sqrt(noise), flagged);
  }

  /**
   * 半荘のブートストラップでの D_SPG の標準偏差。
   *
   * @param resamples 引き直しの回数
   * @param seed 引き直しの種
   */
  public static double bootstrapDSpgSd(
      List<RecordedGame> games,
      PhiValues phi,
      EpsilonUtilityProfile profile,
      int resamples,
      long seed) {
    if (resamples < 2) throw new IllegalArgumentException("Bootstrap needs at least two resamples");
    Layout layout = Layout.of(games, phi, profile);
    // 半荘ごとの升目の (升目, 残差の和, 数) を疎に持つ
    int[] first = new int[layout.games + 1];
    int[] cells = new int[layout.seatHands()];
    double[] sums = new double[layout.seatHands()], counts = new double[layout.seatHands()];
    double[] gameSum = new double[SPG_CELLS], gameCount = new double[SPG_CELLS];
    int entries = 0;
    for (int g = 0; g < layout.games; g++) {
      first[g] = entries;
      for (int h = layout.firstHand[g]; h < layout.firstHand[g + 1]; h++)
        for (int seat = 0; seat < 4; seat++) {
          int j = 4 * h + seat, cell = layout.spgCell(j);
          if (gameCount[cell] == 0) cells[entries++] = cell;
          gameSum[cell] += layout.utility[j] - phi.utilities()[h][seat];
          gameCount[cell]++;
        }
      for (int e = first[g]; e < entries; e++) {
        sums[e] = gameSum[cells[e]];
        counts[e] = gameCount[cells[e]];
        gameSum[cells[e]] = 0;
        gameCount[cells[e]] = 0;
      }
    }
    first[layout.games] = entries;
    SplittableRandom random = new SplittableRandom(seed);
    int[] weights = new int[layout.games];
    double total = 0, totalSquares = 0;
    for (int b = 0; b < resamples; b++) {
      Arrays.fill(weights, 0);
      for (int i = 0; i < layout.games; i++) weights[random.nextInt(layout.games)]++;
      Cells spg = new Cells(SPG_CELLS);
      double seatHands = 0;
      for (int g = 0; g < layout.games; g++) {
        int weight = weights[g];
        if (weight == 0) continue;
        for (int e = first[g]; e < first[g + 1]; e++) {
          spg.addGame(cells[e], sums[e], counts[e], weight);
          seatHands += weight * counts[e];
        }
      }
      double d = spg.distance(seatHands);
      total += d;
      totalSquares += d * d;
    }
    double mean = total / resamples;
    return Math.sqrt(Math.max(0, (totalSquares - resamples * mean * mean) / (resamples - 1)));
  }

  /**
   * 区分ごとの u_最終 の Φ への傾き β と z = (β − 1)/SE(β)（SE は半荘ごとのまとまりのサンドイッチ: G/(G − 1) · Σ_g (Σ_i x̃_i e_i)² / S_xx²）。
   * 半荘が {@value #MIN_SLOPE_GAMES} 未満の区分は SE と z を NaN にする（まとまりが少ないと SE がほぼ 0 になり z が発散する）。
   *
   * @param regression 区分ごとの n・Σx・Σy・Σx²・Σxy（x = Φ、y = u）
   * @return [0] 傾き、[1] SE、[2] z
   */
  private static double[][] slopes(Layout layout, PhiValues phi, double[][] regression) {
    double[] slope = new double[KYOKU_CELLS], meanX = new double[KYOKU_CELLS], meanY = new double[KYOKU_CELLS];
    double[] sxx = new double[KYOKU_CELLS];
    for (int group = 0; group < KYOKU_CELLS; group++) {
      double[] s = regression[group];
      meanX[group] = s[1] / s[0];
      meanY[group] = s[2] / s[0];
      sxx[group] = s[3] - s[0] * meanX[group] * meanX[group];
      slope[group] = (s[4] - s[0] * meanX[group] * meanY[group]) / sxx[group];
    }
    double[] scores = new double[KYOKU_CELLS], squares = new double[KYOKU_CELLS];
    int[] clusters = new int[KYOKU_CELLS];
    boolean[] touched = new boolean[KYOKU_CELLS];
    for (int g = 0; g < layout.games; g++) {
      for (int h = layout.firstHand[g]; h < layout.firstHand[g + 1]; h++) {
        int group = layout.group[h];
        touched[group] = true;
        for (int seat = 0; seat < 4; seat++) {
          double x = phi.utilities()[h][seat] - meanX[group];
          double e = layout.utility[4 * h + seat] - meanY[group] - slope[group] * x;
          scores[group] += x * e;
        }
      }
      for (int group = 0; group < KYOKU_CELLS; group++) {
        if (!touched[group]) continue;
        squares[group] += scores[group] * scores[group];
        clusters[group]++;
        scores[group] = 0;
        touched[group] = false;
      }
    }
    double[] error = new double[KYOKU_CELLS], z = new double[KYOKU_CELLS];
    for (int group = 0; group < KYOKU_CELLS; group++) {
      double n = clusters[group];
      error[group] = n < MIN_SLOPE_GAMES ? Double.NaN : Math.sqrt(n / (n - 1) * squares[group]) / sxx[group];
      z[group] = (slope[group] - 1) / error[group];
    }
    return new double[][] {slope, error, z};
  }

  /** 局 × 今の順位 × 隣との点差の升目（0..{@value #SPG_CELLS}）。 */
  static int spgCell(HandGroup group, int rank, int gap) {
    return spgCell(group.ordinal(), rank, gap);
  }

  private static int spgCell(int group, int rank, int gap) {
    int bin = 0;
    while (bin < GAP_EDGES.length && gap >= GAP_EDGES[bin]) bin++;
    return (group * 4 + rank) * (GAP_EDGES.length + 1) + bin;
  }

  /** 終わった局の区分（局末 k の次局 root の公開系列の最後から 2 つ目の局境界の局番号）。 */
  static HandGroup endedGroup(RecordedGame game, int k) {
    return HandGroup.of(k == 0 ? HandStart.INITIAL.kyokuIndex() : game.starts().get(k - 1).kyokuIndex());
  }

  /** 席 × 半荘が続く局末の並びの Φ に依らない部分。添字 j = 4h + 席、h は {@link HandEndPhi#gameValues} の順の局末。 */
  private static final class Layout {
    final int games;
    final int[] firstHand;
    final byte[] group;
    final byte[] rank;
    final int[] gap;
    final byte[] finalRank;
    final double[] utility;
    final double low, width;

    private Layout(int games, int handEnds, EpsilonUtilityProfile profile) {
      this.games = games;
      firstHand = new int[games + 1];
      group = new byte[handEnds];
      rank = new byte[4 * handEnds];
      gap = new int[4 * handEnds];
      finalRank = new byte[4 * handEnds];
      utility = new double[4 * handEnds];
      double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
      for (int r = 0; r < 4; r++) {
        lo = Math.min(lo, profile.utilityForRankDouble(r));
        hi = Math.max(hi, profile.utilityForRankDouble(r));
      }
      low = lo;
      width = hi - lo;
    }

    static Layout of(List<RecordedGame> games, PhiValues phi, EpsilonUtilityProfile profile) {
      int handEnds = 0;
      for (RecordedGame game : games) handEnds += game.continuingHandEnds();
      if (handEnds != phi.size())
        throw new IllegalArgumentException("Φ has " + phi.size() + " hand ends, the games have " + handEnds);
      Layout layout = new Layout(games.size(), handEnds, profile);
      int h = 0;
      for (int g = 0; g < games.size(); g++) {
        RecordedGame game = games.get(g);
        layout.firstHand[g] = h;
        int[] finalRanks = game.finalRanks();
        double[] utilities = game.finalUtilities(profile);
        for (int k = 0; k < game.continuingHandEnds(); k++, h++) {
          layout.group[h] = (byte) endedGroup(game, k).ordinal();
          int[] scores = game.starts().get(k).scores();
          int[] ranks = ScoreRanking.byScoreThenSeat(scores);
          int[] seatAt = new int[4];
          for (int seat = 0; seat < 4; seat++) seatAt[ranks[seat]] = seat;
          for (int seat = 0; seat < 4; seat++) {
            int j = 4 * h + seat;
            int neighbour = seatAt[ranks[seat] < 3 ? ranks[seat] + 1 : 2];
            layout.rank[j] = (byte) ranks[seat];
            layout.gap[j] = Math.abs(scores[seat] - scores[neighbour]);
            layout.finalRank[j] = (byte) finalRanks[seat];
            layout.utility[j] = utilities[seat];
          }
        }
      }
      layout.firstHand[games.size()] = h;
      return layout;
    }

    int seatHands() {
      return utility.length;
    }

    int spgCell(int j) {
      return CellStats.spgCell(group[j >> 2], rank[j], gap[j]);
    }

    int phiBin(double value) {
      return Math.max(0, Math.min(PHI_BINS - 1, (int) ((value - low) / width * PHI_BINS)));
    }
  }

  /** 升目ごとの残差の和と、半荘ごとのまとまりの分散のための和。 */
  private static final class Cells {
    final double[] count, sum, sumSquares, sumCross, countSquares, clusters;
    private final double[] gameSum, gameCount;
    private final int[] touched;
    private int touchedCount;

    Cells(int cells) {
      count = new double[cells];
      sum = new double[cells];
      sumSquares = new double[cells];
      sumCross = new double[cells];
      countSquares = new double[cells];
      clusters = new double[cells];
      gameSum = new double[cells];
      gameCount = new double[cells];
      touched = new int[cells];
    }

    void add(int cell, double residual) {
      if (gameCount[cell] == 0) touched[touchedCount++] = cell;
      gameSum[cell] += residual;
      gameCount[cell]++;
    }

    /** 今の半荘を閉じる。 */
    void endGame() {
      for (int i = 0; i < touchedCount; i++) {
        int cell = touched[i];
        addGame(cell, gameSum[cell], gameCount[cell], 1);
        gameSum[cell] = 0;
        gameCount[cell] = 0;
      }
      touchedCount = 0;
    }

    /** 升目に 1 半荘の和 s・数 n を weight 回足す（ブートストラップの重複）。 */
    void addGame(int cell, double s, double n, int weight) {
      count[cell] += weight * n;
      sum[cell] += weight * s;
      sumSquares[cell] += weight * s * s;
      sumCross[cell] += weight * s * n;
      countSquares[cell] += weight * n * n;
      clusters[cell] += weight;
    }

    double mean(int cell) {
      return sum[cell] / count[cell];
    }

    /** 升目の平均の半荘ごとのまとまりの分散（半荘が 2 未満なら NaN）。 */
    double variance(int cell) {
      double g = clusters[cell];
      if (g < 2) return Double.NaN;
      double m = mean(cell);
      double spread = sumSquares[cell] - 2 * m * sumCross[cell] + m * m * countSquares[cell];
      return g / (g - 1) * Math.max(0, spread) / (count[cell] * count[cell]);
    }

    /** Σ_c (n_c/N)(m_c² − v_c)（半荘が 2 未満の升目は入れない）。 */
    double distance(double seatHands) {
      double d = 0;
      for (int cell = 0; cell < count.length; cell++) {
        double variance = variance(cell);
        if (!(variance >= 0)) continue;
        double m = mean(cell);
        d += count[cell] / seatHands * (m * m - variance);
      }
      return d;
    }

    /** 升目 0 の平均と標準誤差。 */
    Paired paired() {
      return new Paired(mean(0), Math.sqrt(variance(0)));
    }
  }
}
