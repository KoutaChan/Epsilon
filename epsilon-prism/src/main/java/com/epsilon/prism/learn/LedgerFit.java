package com.epsilon.prism.learn;

import com.epsilon.prism.collect.EventLedger;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.rows.LedgerRow;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 運の精算の係数の当てはめ（{@link EventLedger} の帳簿 L で Y を補正する β）。行の測った手ごとに行の中で中心化した ỹ = Y − mean_M Y と
 * l̃ = L − mean_M L の最小二乗（特徴ごとの対角に比例する弱い ridge）で、判断の類 c ごとに β を当てる。
 *
 * <p>交差適合: 半荘 ID の偶奇 g（{@code gameId & 1}）の行に掛ける β_{g,c} は、偶奇 1 − g の行だけで当てる（行数が {@link #MIN_ROWS} に足りない類は
 * 偶奇 1 − g の全ての類をまとめた当てはめ、それも足りなければ 0）。半荘は独立（牌山と硬貨は半荘 ID ごとのハッシュ）で、β は行自身の半荘と独立なので、
 * E[β·L_a | o, β] = β·E[L_a | o] = 0 となり、補正した Ỹ_a = Y_a − β·L_a も E[Ỹ_a | o] = Q(o, a)。和は半荘の組（A・B）ごとに足してから組を
 * 足し合わせる（浮動小数の足し算の順はこの順で決まる）。
 *
 * <p>帳簿のない行と、測った手の帳簿が 1 つでも有限でない行は当てはめに入れず、補正もしない（帳簿の NaN は「帳簿を付けなかった」ことだけを表し、o の
 * 時点で決まるので、この選別は偏らない）。重みは付けない（学習の重みは β の不偏性に関係しない）。当てはめは行の集合と順だけの決定的な関数。行は
 * {@link LedgerRow} として読む。
 *
 * <p>報告（{@link Report}）: held-out の残差 SSE_{g,c} = yy − 2β·b + βᵀAβ（偶奇 g の和、β は他方の偶奇で当てたもの）から R² = 1 − ΣSSE/Σyy
 * （行の中で中心化した Y の分散の減り）と同等の分岐の本数 Σyy/ΣSSE、欄の組（{@link EventLedger#GROUPS}）ごとの部分
 * R²、偏りの確かめ（特徴ごとの生の L の平均・標準偏差・0 でない割合・半荘で束ねた z）。
 */
public final class LedgerFit {
  /** ridge の係数（特徴ごとの対角 A_ff に掛ける。標準化した特徴の上の ridge と同じ）。 */
  static final double RIDGE = 1e-3;

  /** 類ごとの当てはめに要る行数（足りなければ類をまとめる）。 */
  static final int MIN_ROWS = 32 * EventLedger.FEATURES;

  private static final int F = EventLedger.FEATURES, CLASSES = DecisionClass.COUNT, GROUPS = 2;

  private LedgerFit() {}

  /**
   * 補正の係数 beta[g][c][f]: 偶奇 g の行・類 c の欄 f（偶奇 1 − g の行で当てたもの）。
   */
  public record Correction(double[][][] beta) {
    /** 補正した Y（測った手は Y_a − β·L_a、測らない手は NaN のまま。帳簿のない行・帳簿の欠けた行は row.values() のまま）。 */
    public float[] values(LedgerRow row) {
      float[] values = row.values();
      if (!row.hasLedger() || row.ledgerFeatures() != F) return values;
      double[] coefficients = beta[(int) (row.gameId() & 1)][row.decisionClass().ordinal()];
      float[] corrected = new float[values.length];
      for (int a = 0; a < values.length; a++) {
        if (Float.isNaN(values[a])) {
          corrected[a] = Float.NaN;
          continue;
        }
        double value = values[a];
        for (int f = 0; f < F; f++) {
          float term = row.ledger(a, f);
          if (!Float.isFinite(term)) return values;
          value -= coefficients[f] * term;
        }
        corrected[a] = (float) value;
      }
      return corrected;
    }
  }

  /**
   * 1 つの区分の当てはめ。
   *
   * @param rows 行の数
   * @param slots 測った手の数
   * @param variance 行の中で中心化した Y の手あたりの平方
   * @param residual 補正した後の手あたりの平方（held-out）
   * @param r2 held-out の R²（1 − residual / variance）
   * @param r2ByFold 偶奇ごとの R²
   * @param equivalentForks 同等の分岐の本数 variance / residual
   * @param r2ByGroup 欄の組（draw・dora・ura・policy）だけで当てたときの R²
   * @param beta 両方の偶奇で当てた係数
   * @param betaByFold 偶奇 g の行に掛ける係数（偶奇 1 − g で当てたもの）
   * @param fallback 偶奇 g の係数が、類ごとの区分では類をまとめた当てはめ（か、それも足りずに 0）か、全体の区分では行が足りずに 0 か
   */
  public record Line(
      long rows,
      long slots,
      double variance,
      double residual,
      double r2,
      double[] r2ByFold,
      double equivalentForks,
      Map<String, Double> r2ByGroup,
      double[] beta,
      double[][] betaByFold,
      boolean[] fallback) {}

  /**
   * 1 つの欄の偏りの確かめ。
   *
   * @param mean 手あたりの生の L の平均
   * @param sd 手あたりの標準偏差
   * @param nonzero 0 でない手の割合
   * @param z 半荘で束ねた平均の z（正しく実装されていれば標準正規）
   */
  public record FeatureBias(String name, double mean, double sd, double nonzero, double z) {}

  /**
   * 偏りの確かめ。
   *
   * @param all 全体（半荘で束ねた z）
   */
  public record Bias(List<FeatureBias> all) {}

  /**
   * 報告。
   *
   * @param rows 当てはめた行の数
   * @param slots 当てはめた測った手の数
   * @param skippedRows 帳簿のない行（と帳簿の欠けた行）の数
   * @param all 全体（全ての類をまとめた交差適合）
   * @param byClass 類ごと（行のない類は載せない）
   * @param seconds 当てはめの全体の秒（行の列を読む時間を含む）
   */
  public record Report(
      long rows,
      long slots,
      long skippedRows,
      Line all,
      Map<String, Line> byClass,
      Bias bias,
      double seconds) {}

  /**
   * 当てはめの結果。
   *
   * @param correction 学習の行に掛ける交差適合の係数
   * @param report 報告
   */
  public record Result(Correction correction, Report report) {}

  /** 行の列から当てる（系譜は貯蔵から塊ずつ読む列を渡す）。 */
  public static Result fit(Iterable<? extends LedgerRow> rows) {
    long started = System.nanoTime();
    var accumulator = new Accumulator();
    for (LedgerRow row : rows) accumulator.add(row);
    return accumulator.fit(started);
  }

  /** (組, 偶奇, 類) の和: A = Σ l̃ l̃ᵀ、b = Σ l̃ ỹ、yy = Σ ỹ²、行と手の数。 */
  static final class Sums {
    final double[] a = new double[F * F];
    final double[] b = new double[F];
    double yy;
    long rows, slots;

    void add(Sums other) {
      for (int i = 0; i < a.length; i++) a[i] += other.a[i];
      for (int i = 0; i < F; i++) b[i] += other.b[i];
      yy += other.yy;
      rows += other.rows;
      slots += other.slots;
    }

    /** held-out の残差 yy − 2β·b + βᵀAβ。 */
    double sse(double[] beta) {
      double cross = 0, quadratic = 0;
      for (int i = 0; i < F; i++) {
        if (beta[i] == 0) continue;
        cross += beta[i] * b[i];
        double row = 0;
        for (int j = 0; j < F; j++) row += a[i * F + j] * beta[j];
        quadratic += beta[i] * row;
      }
      return yy - 2 * cross + quadratic;
    }

    static Sums of(Sums... parts) {
      var sum = new Sums();
      for (Sums part : parts) sum.add(part);
      return sum;
    }
  }

  /**
   * (A + RIDGE·diag(A)) β = b を、欄 features の上で解く（ほかの欄と A_ff = 0 の欄は 0）。d_f = sqrt(A_ff) で割った相関行列の形にしてから
   * Cholesky で解くので、欄の大きさに依らない。
   */
  static double[] solve(Sums sums, int[] features) {
    double[] beta = new double[F];
    int[] used = new int[features.length];
    int k = 0;
    for (int f : features) if (sums.a[f * F + f] > 0) used[k++] = f;
    if (k == 0) return beta;
    double[] scale = new double[k];
    for (int i = 0; i < k; i++) scale[i] = Math.sqrt(sums.a[used[i] * F + used[i]]);
    double[][] m = new double[k][k];
    double[] r = new double[k];
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) m[i][j] = sums.a[used[i] * F + used[j]] / (scale[i] * scale[j]);
      m[i][i] += RIDGE;
      r[i] = sums.b[used[i]] / scale[i];
    }
    // Cholesky m = L Lᵀ（下三角を m に上書き）
    for (int i = 0; i < k; i++) {
      for (int j = 0; j <= i; j++) {
        double sum = m[i][j];
        for (int p = 0; p < j; p++) sum -= m[i][p] * m[j][p];
        if (i == j) {
          if (!(sum > 0)) throw new IllegalStateException("Ledger normal equations are not positive definite");
          m[i][i] = Math.sqrt(sum);
        } else {
          m[i][j] = sum / m[j][j];
        }
      }
    }
    double[] y = new double[k];
    for (int i = 0; i < k; i++) {
      double sum = r[i];
      for (int p = 0; p < i; p++) sum -= m[i][p] * y[p];
      y[i] = sum / m[i][i];
    }
    for (int i = k - 1; i >= 0; i--) {
      double sum = y[i];
      for (int p = i + 1; p < k; p++) sum -= m[p][i] * y[p];
      y[i] = sum / m[i][i];
    }
    for (int i = 0; i < k; i++) beta[used[i]] = y[i] / scale[i];
    return beta;
  }

  private static final int[] ALL_FEATURES = range(0, F);

  private static int[] range(int from, int to) {
    int[] features = new int[to - from];
    for (int f = from; f < to; f++) features[f - from] = f;
    return features;
  }

  /** 行を足していく和（当てはめと偏りの確かめ）。 */
  static final class Accumulator {
    final Sums[][][] sums = new Sums[GROUPS][2][CLASSES];

    /** 類ごとの生の L の和・平方の和・0 でない数。 */
    final double[][] sum = new double[CLASSES][F], square = new double[CLASSES][F], nonzero = new double[CLASSES][F];

    /** 半荘で束ねる和: 鍵 半荘 → [手の数, 欄ごとの L の和]（全体の z）。 */
    final Map<Long, double[]> byGame = new HashMap<>();

    long rows, slots, skipped;

    Accumulator() {
      for (int g = 0; g < GROUPS; g++)
        for (int p = 0; p < 2; p++) for (int c = 0; c < CLASSES; c++) sums[g][p][c] = new Sums();
    }

    void add(LedgerRow row) {
      if (!row.hasLedger()) {
        skipped++;
        return;
      }
      if (row.ledgerFeatures() != F)
        throw new IllegalStateException(
            "Ledger of game " + row.gameId() + " has " + row.ledgerFeatures() + " features, expected " + F);
      float[] values = row.values();
      int n = values.length;
      int[] measured = new int[n];
      int m = 0;
      for (int a = 0; a < n; a++) if (!Float.isNaN(values[a])) measured[m++] = a;
      if (m < 2) {
        skipped++;
        return;
      }
      for (int i = 0; i < m; i++)
        for (int f = 0; f < F; f++)
          if (!Float.isFinite(row.ledger(measured[i], f))) {
            skipped++;
            return;
          }
      double meanY = 0;
      double[] meanL = new double[F];
      for (int i = 0; i < m; i++) {
        int a = measured[i];
        meanY += values[a];
        for (int f = 0; f < F; f++) meanL[f] += row.ledger(a, f);
      }
      meanY /= m;
      for (int f = 0; f < F; f++) meanL[f] /= m;
      int c = row.decisionClass().ordinal();
      Sums target = sums[row.group().ordinal()][(int) (row.gameId() & 1)][c];
      // 偏りの確かめは生の L（束は行ごとに 1 回引く）
      double[] game = byGame.computeIfAbsent(row.gameId(), ignored -> new double[1 + F]);
      double[] centred = new double[F];
      for (int i = 0; i < m; i++) {
        int a = measured[i];
        double y = values[a] - meanY;
        for (int f = 0; f < F; f++) centred[f] = row.ledger(a, f) - meanL[f];
        for (int f = 0; f < F; f++) {
          if (centred[f] == 0) continue;
          target.b[f] += centred[f] * y;
          for (int g = 0; g < F; g++) target.a[f * F + g] += centred[f] * centred[g];
        }
        target.yy += y * y;
        game[0]++;
        for (int f = 0; f < F; f++) {
          double l = row.ledger(a, f);
          game[1 + f] += l;
          sum[c][f] += l;
          square[c][f] += l * l;
          if (l != 0) nonzero[c][f]++;
        }
      }
      target.rows++;
      target.slots += m;
      rows++;
      slots += m;
    }

    Result fit(long started) {
      Domain both = domain();
      Map<String, Line> byClass = new LinkedHashMap<>();
      for (DecisionClass c : DecisionClass.values())
        if (both.classes[c.ordinal()].rows() > 0) byClass.put(c.name(), both.classes[c.ordinal()].line());
      var report =
          new Report(
              rows,
              slots,
              skipped,
              both.all(),
              byClass,
              new Bias(features(totalSums(), totalSquares(), totalNonzero(), slots, byGame)),
              (System.nanoTime() - started) / 1e9);
      return new Result(both.correction(), report);
    }

    /** 両方の組の和の区分（組 A の和に組 B の和を足す）。 */
    private Domain domain() {
      Sums[][] parity = new Sums[2][CLASSES];
      for (int p = 0; p < 2; p++)
        for (int c = 0; c < CLASSES; c++) {
          parity[p][c] = new Sums();
          for (int g = 0; g < GROUPS; g++) parity[p][c].add(sums[g][p][c]);
        }
      return new Domain(parity);
    }

    private double[] totalSums() {
      double[] total = new double[F];
      for (double[] row : sum) for (int f = 0; f < F; f++) total[f] += row[f];
      return total;
    }

    private double[] totalSquares() {
      double[] total = new double[F];
      for (double[] row : square) for (int f = 0; f < F; f++) total[f] += row[f];
      return total;
    }

    private double[] totalNonzero() {
      double[] total = new double[F];
      for (double[] row : nonzero) for (int f = 0; f < F; f++) total[f] += row[f];
      return total;
    }
  }

  /**
   * 欄ごとの偏りの確かめ。z は束ねた平均の線形化: mean = ΣS_g/Σn_g、se² = G/(G−1)·Σ(S_g − mean·n_g)²/(Σn_g)²。
   */
  private static List<FeatureBias> features(
      double[] sums, double[] squares, double[] nonzero, long count, Map<Long, double[]> clusters) {
    List<FeatureBias> result = new ArrayList<>(F);
    long groups = clusters.size();
    for (int f = 0; f < F; f++) {
      double mean = count > 0 ? sums[f] / count : Double.NaN;
      double sd = count == 0 ? Double.NaN : Math.sqrt(Math.max(0, squares[f] / count - mean * mean));
      double share = count == 0 ? Double.NaN : nonzero[f] / count;
      double s2 = 0, sn = 0, n2 = 0, n = 0;
      for (double[] cluster : clusters.values()) {
        double s = cluster[1 + f], k = cluster[0];
        s2 += s * s;
        sn += s * k;
        n2 += k * k;
        n += k;
      }
      double z = Double.NaN;
      if (groups >= 2 && n > 0) {
        double squared = Math.max(0, s2 - 2 * mean * sn + mean * mean * n2);
        double error = Math.sqrt(groups / (groups - 1.0) * squared) / n;
        z = error > 0 ? mean / error : mean == 0 ? 0 : Double.NaN;
      }
      result.add(new FeatureBias(EventLedger.NAMES.get(f), mean, sd, share, z));
    }
    return result;
  }

  /** 両方の組の偶奇 × 類の和と、その交差適合。 */
  private static final class Domain {
    final Sums[] pooledByParity = new Sums[2];
    final ClassFit[] classes = new ClassFit[CLASSES];

    Domain(Sums[][] parity) {
      for (int p = 0; p < 2; p++) pooledByParity[p] = Sums.of(parity[p]);
      for (int c = 0; c < CLASSES; c++) classes[c] = new ClassFit(parity[0][c], parity[1][c], pooledByParity);
    }

    /** 交差適合の係数（偶奇 g・類 c は偶奇 1 − g で当てたもの）。 */
    Correction correction() {
      double[][][] beta = new double[2][CLASSES][];
      for (int p = 0; p < 2; p++) for (int c = 0; c < CLASSES; c++) beta[p][c] = classes[c].byFold[p];
      return new Correction(beta);
    }

    /** 全ての類をまとめた区分（残差は類ごとの係数の残差の和）。 */
    Line all() {
      Sums[] folds = {pooledByParity[0], pooledByParity[1]};
      double[] sse = new double[2];
      double[] groupSse = new double[EventLedger.GROUPS.size()];
      for (ClassFit fit : classes) {
        for (int p = 0; p < 2; p++) sse[p] += fit.sse[p];
        for (int k = 0; k < groupSse.length; k++) groupSse[k] += fit.groupSse[k];
      }
      double[][] byFold = new double[2][];
      boolean[] fallback = new boolean[2];
      for (int p = 0; p < 2; p++) {
        fallback[p] = folds[1 - p].rows < MIN_ROWS;
        byFold[p] = fallback[p] ? new double[F] : solve(folds[1 - p], ALL_FEATURES);
      }
      Sums everything = Sums.of(folds);
      return line(
          folds,
          sse,
          groupSse,
          everything.rows >= MIN_ROWS ? solve(everything, ALL_FEATURES) : new double[F],
          byFold,
          fallback);
    }
  }

  /** 1 つの類の交差適合: 偶奇 g の係数は偶奇 1 − g の類の和（足りなければ類をまとめた和）で当てる。 */
  private static final class ClassFit {
    final Sums[] folds;
    final double[][] byFold = new double[2][];
    final boolean[] fallback = new boolean[2];
    final double[] sse = new double[2];
    final double[] groupSse = new double[EventLedger.GROUPS.size()];

    ClassFit(Sums even, Sums odd, Sums[] pooledByParity) {
      folds = new Sums[] {even, odd};
      for (int p = 0; p < 2; p++) {
        Sums other = folds[1 - p];
        Sums source = other.rows >= MIN_ROWS ? other : pooledByParity[1 - p].rows >= MIN_ROWS ? pooledByParity[1 - p] : null;
        fallback[p] = source != other;
        byFold[p] = source == null ? new double[F] : solve(source, ALL_FEATURES);
        sse[p] = folds[p].sse(byFold[p]);
        for (int k = 0; k < groupSse.length; k++) {
          EventLedger.Group group = EventLedger.GROUPS.get(k);
          double[] partial = source == null ? new double[F] : solve(source, range(group.from(), group.to()));
          groupSse[k] += folds[p].sse(partial);
        }
      }
    }

    long rows() {
      return folds[0].rows + folds[1].rows;
    }

    Line line() {
      Sums both = Sums.of(folds);
      return LedgerFit.line(
          folds, sse, groupSse, both.rows >= MIN_ROWS ? solve(both, ALL_FEATURES) : new double[F], byFold, fallback);
    }
  }

  private static Line line(
      Sums[] folds, double[] sse, double[] groupSse, double[] beta, double[][] byFold, boolean[] fallback) {
    long rows = folds[0].rows + folds[1].rows, slots = folds[0].slots + folds[1].slots;
    double yy = folds[0].yy + folds[1].yy, residual = sse[0] + sse[1];
    Map<String, Double> byGroup = new LinkedHashMap<>();
    for (int k = 0; k < groupSse.length; k++)
      byGroup.put(EventLedger.GROUPS.get(k).name(), yy > 0 ? 1 - groupSse[k] / yy : Double.NaN);
    return new Line(
        rows,
        slots,
        slots > 0 ? yy / slots : Double.NaN,
        slots > 0 ? residual / slots : Double.NaN,
        yy > 0 ? 1 - residual / yy : Double.NaN,
        new double[] {
          folds[0].yy > 0 ? 1 - sse[0] / folds[0].yy : Double.NaN,
          folds[1].yy > 0 ? 1 - sse[1] / folds[1].yy : Double.NaN
        },
        residual > 0 ? yy / residual : Double.NaN,
        byGroup,
        beta,
        byFold,
        fallback);
  }
}
