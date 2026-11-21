package com.epsilon.prism.luck;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.noise.LedgerCalibration;
import com.epsilon.prism.noise.LedgerCalibrationKind;
import com.epsilon.prism.noise.LedgerCalibrationSource;
import com.epsilon.prism.noise.LedgerCells;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.rows.Row;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * 帳簿の補正率 h(o) の校正器。帳簿の補正 β·L に、判断時の公開の局面 o だけから決めた率を掛けて、教師を Ỹ = Y − h(o)·β·L にする
 * （{@link LedgerCorrection#scaled}）。h = 1 は帳簿の補正そのもの、h = 0 は生の Y。
 *
 * <p>形は {@link LedgerCalibrationKind#COMMON_SCALE}。自席の順位 r ごとに、最も近い他家との点差 d と局の番号 k を
 * x = d / (S + d)、t = k / (K + 1 + k)（S は配給原点、K は南場の最後の局の番号）に直す。x の [0, 1] に等間隔の {@code gapGrid} 点、
 * t の [0, 1] に等間隔の {@code roundGrid} 点の格子を置き、h(o) は (x, t) を囲む 4 点の係数 θ を双線形に補間した値。θ は [0, 1] に置くので、
 * h も [0, 1] に入る。
 *
 * <p>当てはめ（{@link Sums}・{@link #fit}）: 行 i ごとに、測った手（m 手）の中で中心化した Y と β·L を x_i・v_i として、C_i = |x_i|²/m、
 * D_i = x_i·v_i/m、A_i = |v_i|²/m とする。行内の残差の二乗は J_i(h) = |x_i − h·v_i|²/m = C_i − 2h·D_i + h²·A_i で、
 *
 * <pre>
 *   minimize  Σ_i [J_i(h(o_i)) + 2ρ·A_i·h(o_i)] + λ·|θ|²   subject to  0 ≤ θ ≤ 1,   λ = regularization · Σ_i A_i / Nθ
 * </pre>
 *
 * を解く（ρ = shrink、Nθ は係数の数）。h は θ の線形関数なので、これは θ の狭義凸な二次計画で、解はただ 1 つ。ρ の項は補正の分散に比例して
 * h を 0 へ引くので、補正が Y と共変しない局面では解がちょうど 0 になれる。行が触れない係数は λ の項で 0 になる。行ごとの D_i/A_i を
 * その行に掛けることはしない（係数はその行の結果に依らない）。
 *
 * <p>偏りのなさ: h を使う行の結果と独立に凍結し（前のコホートの行で当てはめる）、β も使う行と別の半荘で当てはめれば、帳簿の条件付き平均が
 * 0 なので E[h(o)·β·L_a | o] = 0 となり、Ỹ の期待値は Y と同じ。
 *
 * @param kind 補正率の形
 * @param source 当てはめる行を取ったコホートの選び方
 * @param variant 当てはめた行のルール（席の数・配給原点・南場の最後の局の番号を決める）
 * @param cohort 当てはめた行のコホート
 * @param columns 補正 β·L の帳簿の列の並び
 * @param apply 補正 β·L に使った組
 * @param cells β を当てはめた区分
 * @param gapGrid 点差の軸の格子の点の数
 * @param roundGrid 局の番号の軸の格子の点の数
 * @param regularization 係数の二乗の罰の係数 c
 * @param shrink 補正率を 0 へ縮める罰の強さ ρ
 * @param rows 当てはめた行の数
 * @param coefficients 格子の係数 θ（[0, 1]）。番号は（順位 × gapGrid ＋ 点差の点）× roundGrid ＋ 局の点
 */
public record LedgerCalibrator(
    LedgerCalibrationKind kind,
    LedgerCalibrationSource source,
    Variant variant,
    int cohort,
    List<LedgerColumn> columns,
    Set<LedgerGroup> apply,
    LedgerCells cells,
    int gapGrid,
    int roundGrid,
    double regularization,
    double shrink,
    long rows,
    List<Double> coefficients) {

  public LedgerCalibrator {
    Objects.requireNonNull(kind, "kind");
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(variant, "variant");
    columns = List.copyOf(columns);
    apply = LedgerRecording.groups(apply);
    Objects.requireNonNull(cells, "cells");
    coefficients = List.copyOf(coefficients);
    int size = new Grid(variant, gapGrid, roundGrid).size();
    if (coefficients.size() != size) {
      throw new IllegalArgumentException(
          "A "
              + gapGrid
              + "x"
              + roundGrid
              + " calibrator has "
              + size
              + " coefficients, not "
              + coefficients.size());
    }
    for (double coefficient : coefficients) {
      if (!(coefficient >= 0 && coefficient <= 1)) {
        throw new IllegalArgumentException("Calibrator coefficient outside [0, 1]: " + coefficient);
      }
    }
  }

  /**
   * 局面の補正率 h(o)。囲む 4 点の係数の双線形の補間を、局の軸、点差の軸の順に線形補間を 2 回重ねて求める（{@link Grid#corners} の重みと
   * 同じ関数）。どの補間も両端の係数の間に入るので h は 4 点の係数の最小と最大の間にあり、4 点の係数が同じならその値そのもの。
   */
  public double scale(LedgerCell.Standing standing) {
    int[] index = new int[Grid.CORNERS];
    double[] weight = new double[Grid.CORNERS];
    Grid grid = new Grid(variant, gapGrid, roundGrid);
    double[] at = grid.corners(standing, index, weight);
    double u = at[0], v = at[1];
    double low = lerp(coefficients.get(index[0]), coefficients.get(index[1]), v);
    double high = lerp(coefficients.get(index[2]), coefficients.get(index[3]), v);
    return Math.min(1, Math.max(0, lerp(low, high, u)));
  }

  /** a から b への線形補間 a + t·(b − a)（0 ≤ t ≤ 1 なら a と b の間）。 */
  private static double lerp(double a, double b, double t) {
    return a + t * (b - a);
  }

  /**
   * 格子と、局面を囲む 4 点の求め方。
   *
   * @param seats 席の数
   * @param gaps 点差の軸の点の数
   * @param rounds 局の番号の軸の点の数
   * @param origin 配給原点 S（点）
   * @param roundScale K + 1（K は南場の最後の局の番号）
   */
  record Grid(int seats, int gaps, int rounds, double origin, double roundScale) {
    static final int CORNERS = 4;

    Grid(Variant variant, int gaps, int rounds) {
      this(variant.seats(), gaps, rounds, variant.startPoints(), variant.lastRound() + 1);
    }

    Grid {
      if (gaps < 2 || rounds < 2) {
        throw new IllegalArgumentException("A calibration grid needs two points per axis");
      }
    }

    /** 係数の数。 */
    int size() {
      return seats * gaps * rounds;
    }

    /**
     * 局面を囲む 4 点の番号と双線形の重みを書く。順は（点差、局）の（低、低）・（低、高）・（高、低）・（高、高）。重みの和は 1。
     *
     * @return 囲む区間の中の位置（点差の軸 u と局の軸 v。どちらも [0, 1]）
     * @throws IllegalArgumentException 順位か点差か局の番号が範囲の外のとき
     */
    double[] corners(LedgerCell.Standing standing, int[] index, double[] weight) {
      int rank = standing.rank(), gap = standing.gap(), round = standing.round();
      if (rank < 0 || rank >= seats || gap < 0 || round < 0) {
        throw new IllegalArgumentException("Standing out of range: " + standing);
      }
      double x = (gaps - 1) * (double) gap / (origin + gap);
      double t = (rounds - 1) * (double) round / (roundScale + round);
      int i = Math.min((int) x, gaps - 2), j = Math.min((int) t, rounds - 2);
      double u = x - i, v = t - j;
      int base = (rank * gaps + i) * rounds + j;
      index[0] = base;
      index[1] = base + 1;
      index[2] = base + rounds;
      index[3] = base + rounds + 1;
      weight[0] = (1 - u) * (1 - v);
      weight[1] = (1 - u) * v;
      weight[2] = u * (1 - v);
      weight[3] = u * v;
      return new double[] {u, v};
    }
  }

  /**
   * 当てはめの和。H = Σ_i A_i·w_i w_iᵀ、r = Σ_i D_i·w_i、a = Σ_i A_i·w_i（w_i は行 i の局面の双線形の重み）と、Σ_i C_i・Σ_i A_i・行の
   * 数。和は行を足す順で決まる。
   */
  static final class Sums {
    final Grid grid;
    final int size;
    final double[] h, r, a;
    double c, total;
    long rows;
    private final int[] index = new int[Grid.CORNERS];
    private final double[] weight = new double[Grid.CORNERS];

    Sums(Grid grid) {
      this.grid = grid;
      size = grid.size();
      h = new double[size * size];
      r = new double[size];
      a = new double[size];
    }

    /** 行 i の C_i・D_i・A_i を足す。 */
    void add(LedgerCell.Standing standing, double ci, double di, double ai) {
      grid.corners(standing, index, weight);
      for (int p = 0; p < Grid.CORNERS; p++) {
        int row = index[p];
        double w = weight[p];
        r[row] += di * w;
        a[row] += ai * w;
        for (int q = 0; q < Grid.CORNERS; q++) h[row * size + index[q]] += ai * w * weight[q];
      }
      c += ci;
      total += ai;
      rows++;
    }

    /** 係数 θ のときの Σ_i J_i(h(o_i)) = ΣC − 2θ·r + θᵀHθ。 */
    double residual(double[] theta) {
      double cross = 0, quadratic = 0;
      for (int j = 0; j < size; j++) {
        cross += theta[j] * r[j];
        double row = 0;
        for (int k = 0; k < size; k++) row += h[j * size + k] * theta[k];
        quadratic += theta[j] * row;
      }
      return c - 2 * cross + quadratic;
    }
  }

  /**
   * 和から係数を当てはめる（クラスの説明の二次計画）。
   *
   * @param settings 補正率の設定（形・格子・罰の強さ）
   * @param cohort 和を作った行のコホート
   * @param columns 補正 β·L の帳簿の列の並び
   * @param apply 補正 β·L に使った組
   * @param cells β を当てはめた区分
   * @return 校正器。補正の分散がどの行でも 0（行が無いときも含む）なら、当てはめる情報が無いので null
   * @throws IllegalArgumentException 和の格子が設定の格子と違うとき
   */
  static LedgerCalibrator fit(
      Sums sums,
      LedgerCalibration.Scaling settings,
      Variant variant,
      int cohort,
      List<LedgerColumn> columns,
      Set<LedgerGroup> apply,
      LedgerCells cells) {
    Grid grid = new Grid(variant, settings.gapGrid(), settings.roundGrid());
    if (!grid.equals(sums.grid)) {
      throw new IllegalArgumentException("Calibration sums of " + sums.grid + " for " + grid);
    }
    if (!(sums.total > 0)) {
      return null;
    }
    int n = sums.size;
    double lambda = settings.regularization() * sums.total / n;
    double[][] q = new double[n][n];
    double[] linear = new double[n];
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) q[j][k] = sums.h[j * n + k];
      q[j][j] += lambda;
      linear[j] = sums.r[j] - settings.shrink() * sums.a[j];
    }
    double[] theta =
        switch (settings.kind()) {
          case COMMON_SCALE -> BoxQuadratic.minimize(q, linear);
        };
    List<Double> coefficients = new ArrayList<>(n);
    for (double value : theta) coefficients.add(value);
    return new LedgerCalibrator(
        settings.kind(),
        settings.source(),
        variant,
        cohort,
        columns,
        apply,
        cells,
        settings.gapGrid(),
        settings.roundGrid(),
        settings.regularization(),
        settings.shrink(),
        sums.rows,
        coefficients);
  }

  /** 係数を配列で返す。 */
  double[] theta() {
    double[] theta = new double[coefficients.size()];
    for (int j = 0; j < theta.length; j++) theta[j] = coefficients.get(j);
    return theta;
  }

  /**
   * 箱の制約の付いた凸二次計画。0 ≤ θ ≤ 1 の範囲で f(θ) = θᵀQθ − 2c·θ を最小にする（Q は対称な正定値）。有効な制約（下限か上限に置いた
   * 係数）の集合を 1 つずつ動かす主の方法で、自由な係数の連立方程式を Cholesky 分解で解く。返す θ は KKT 条件（自由な係数で Qθ − c = 0、
   * 下限の係数で 0 以上、上限の係数で 0 以下。どれも係数の大きさに比例した許容幅で）を満たし、狭義凸なのでただ 1 つの最小点。手順は
   * 決まった順で進むので、同じ入力からは同じビットの θ を返す。
   */
  static final class BoxQuadratic {
    private static final int LOWER = 0, FREE = 1, UPPER = 2;

    /** 許容幅の、|c| と Q の対角の最大に対する比。 */
    static final double TOLERANCE = 1e-12;

    private BoxQuadratic() {}

    /**
     * @param q 対称な正定値の行列 Q
     * @param c 線形の項 c
     * @return 最小点 θ（全ての成分が [0, 1]）
     * @throws IllegalStateException 決まった回数の内に収束しないか、Q の部分が正定値でないとき
     */
    static double[] minimize(double[][] q, double[] c) {
      int n = c.length;
      double scale = 0;
      for (int j = 0; j < n; j++) scale = Math.max(scale, Math.max(Math.abs(c[j]), q[j][j]));
      double tolerance = TOLERANCE * scale;
      double[] theta = new double[n];
      int[] state = new int[n];
      int limit = 100 + 20 * n;
      for (int iteration = 0; iteration < limit; iteration++) {
        int[] free = free(state);
        double[] target = subspace(q, c, theta, free);
        double step = 1;
        int blocking = -1;
        for (int i = 0; i < free.length; i++) {
          int j = free[i];
          double to = target[i], from = theta[j];
          double reach = to < 0 ? from / (from - to) : to > 1 ? (1 - from) / (to - from) : 1;
          if (reach < step) {
            step = reach;
            blocking = i;
          }
        }
        if (blocking < 0) {
          for (int i = 0; i < free.length; i++) theta[free[i]] = target[i];
          int release = violated(q, c, theta, state, tolerance);
          if (release < 0) {
            return theta;
          }
          state[release] = FREE;
        } else {
          for (int i = 0; i < free.length; i++) {
            int j = free[i];
            theta[j] = Math.min(1, Math.max(0, theta[j] + step * (target[i] - theta[j])));
          }
          int j = free[blocking];
          boolean upper = target[blocking] > 1;
          theta[j] = upper ? 1 : 0;
          state[j] = upper ? UPPER : LOWER;
        }
      }
      throw new IllegalStateException("Ledger calibration did not converge in " + limit + " steps");
    }

    private static int[] free(int[] state) {
      int count = 0;
      for (int s : state) if (s == FREE) count++;
      int[] free = new int[count];
      int k = 0;
      for (int j = 0; j < state.length; j++) if (state[j] == FREE) free[k++] = j;
      return free;
    }

    /** 自由な係数の最小点。ほかの係数を今の値に置いたまま、Q_FF z = c_F − Q_FB θ_B を解く。 */
    private static double[] subspace(double[][] q, double[] c, double[] theta, int[] free) {
      int k = free.length;
      double[][] m = new double[k][k];
      double[] rhs = new double[k];
      boolean[] isFree = new boolean[theta.length];
      for (int j : free) isFree[j] = true;
      for (int i = 0; i < k; i++) {
        int j = free[i];
        for (int p = 0; p < k; p++) m[i][p] = q[j][free[p]];
        double value = c[j];
        for (int b = 0; b < theta.length; b++) if (!isFree[b]) value -= q[j][b] * theta[b];
        rhs[i] = value;
      }
      return cholesky(m, rhs);
    }

    /** m z = rhs を Cholesky 分解で解く（m を書き換える）。 */
    private static double[] cholesky(double[][] m, double[] rhs) {
      int k = rhs.length;
      for (int i = 0; i < k; i++) {
        for (int j = 0; j <= i; j++) {
          double sum = m[i][j];
          for (int p = 0; p < j; p++) sum -= m[i][p] * m[j][p];
          if (i == j) {
            if (!(sum > 0)) {
              throw new IllegalStateException("Ledger calibration matrix is not positive definite");
            }
            m[i][i] = Math.sqrt(sum);
          } else {
            m[i][j] = sum / m[j][j];
          }
        }
      }
      double[] z = new double[k];
      for (int i = 0; i < k; i++) {
        double sum = rhs[i];
        for (int p = 0; p < i; p++) sum -= m[i][p] * z[p];
        z[i] = sum / m[i][i];
      }
      for (int i = k - 1; i >= 0; i--) {
        double sum = z[i];
        for (int p = i + 1; p < k; p++) sum -= m[p][i] * z[p];
        z[i] = sum / m[i][i];
      }
      return z;
    }

    /**
     * 制約に置いた係数のうち、外すと f が下がるもの（下限で (Qθ − c)_j が負、上限で正）の中で、許容幅を最も越えたものを返す。
     *
     * @return 係数の番号。無ければ −1
     */
    private static int violated(double[][] q, double[] c, double[] theta, int[] state, double tol) {
      int worst = -1;
      double most = tol;
      for (int j = 0; j < theta.length; j++) {
        if (state[j] == FREE) continue;
        double gradient = -c[j];
        for (int k = 0; k < theta.length; k++) gradient += q[j][k] * theta[k];
        double violation = state[j] == LOWER ? -gradient : gradient;
        if (violation > most) {
          most = violation;
          worst = j;
        }
      }
      return worst;
    }
  }

  /** 報告に書く、帯の J を数える順位 × 点差の帯の境（点）。最後の帯は上限なし。 */
  static final int[] GAP_BANDS = {0, 4000, 8000, 12000, 18000, 24000, 32000, 48000};

  /** 和了の判断の差を数える、首位と最も近い他家との点差の帯の境（点）。最後の帯は上限なし。 */
  static final int[] GATE_BANDS = {0, 12000, 18000, 24000};

  /** 補正率の分布の柱の数（[0, 1] を等分する）。 */
  static final int HISTOGRAM = 10;

  /**
   * 補正の向きを監査する、事前に決めた群。all は全て、early と last は南場の最後の局より前と後、close と wide は後の局で最も近い他家との
   * 点差が {@link #CLOSE_GAP} 点未満と {@link #WIDE_GAP} 点以上（first は自席が首位）、main・rollin・root は行の種類。この後に判断の
   * クラスごとの群が、クラスの順に続く。
   */
  static final List<String> AUDIT_GROUPS =
      List.of(
          "all",
          "early",
          "last",
          "close_first",
          "wide_first",
          "close_any",
          "wide_any",
          "main",
          "rollin",
          "root");

  /** 監査の群の、接戦と大差の点差の境（点）。 */
  static final int CLOSE_GAP = 4000, WIDE_GAP = 18000;

  /**
   * 補正率を求めるモードで、係数を当てはめた後の行をもう一度順に読む。生（raw）・帳簿の補正 Y − β·L（current）・補正率を掛けた補正
   * Y − h(o)·β·L（shadow）の 3 つの腕で、行内の J とスロットの残差を測り、局面の帯ごとと半荘ごとに束ねる。補正の向きの監査（{@link
   * #AUDIT_GROUPS}）も同じ行で数える。あわせて、次の反復の補正率を当てはめる和を作る。β は行の偶奇が逆の行で当てはめたもの（{@link
   * LedgerCorrection#coefficients}）。
   */
  static final class Pass {
    private final LedgerCorrection current;
    private final boolean byCell;
    private final LedgerCalibrator scale;
    private final Variant variant;
    private final Sums sums;
    private final Group[] groups;
    private final AuditGroup[] audits;
    private final long[] histogram = new long[HISTOGRAM];
    private double scaleSum,
        scaleMin = Double.POSITIVE_INFINITY,
        scaleMax = Double.NEGATIVE_INFINITY;
    private long rows, slots;
    private final double[] rowJ = new double[3], slotResidual = new double[3];
    private int[] measured = new int[0];
    private double[] terms = new double[0];

    /**
     * @param current 帳簿の補正（β）
     * @param scale 今のコホートに掛ける補正率（無ければ null で、補正率は 1）
     * @param next 次の反復の補正率の設定
     */
    Pass(
        LedgerCorrection current,
        LedgerCalibrator scale,
        LedgerCalibration.Scaling next,
        Variant variant) {
      this.current = Objects.requireNonNull(current, "current");
      byCell = current.byCell() != null;
      this.scale = scale;
      this.variant = Objects.requireNonNull(variant, "variant");
      sums = new Sums(new Grid(variant, next.gapGrid(), next.roundGrid()));
      List<Group> named = new ArrayList<>();
      named.add(new Group("all"));
      named.add(new Group("early"));
      named.add(new Group("last"));
      for (int rank = 0; rank < variant.seats(); rank++)
        for (int band = 0; band < GAP_BANDS.length; band++)
          named.add(new Group("rank" + (rank + 1) + "_" + bandName(GAP_BANDS, band, "_")));
      groups = named.toArray(Group[]::new);
      List<AuditGroup> checked = new ArrayList<>();
      for (String name : AUDIT_GROUPS) checked.add(new AuditGroup(name));
      for (DecisionClass decision : DecisionClass.values())
        checked.add(new AuditGroup(decision.name()));
      audits = checked.toArray(AuditGroup[]::new);
    }

    /** 行を 1 つ足す。帳簿の無い行、測った手が 2 つ未満の行、帳簿の欠けた行は飛ばす（係数の当てはめと同じ選び方）。 */
    void add(FitRow row) {
      float[] values = row.values();
      if (measured.length < values.length) {
        measured = new int[values.length];
        terms = new double[values.length];
      }
      int m = LedgerFit.Accumulator.measured(row, values, measured);
      if (m < 0) {
        return;
      }
      double[] beta =
          current.coefficients(
              row.gameId(), row.decisionClass().ordinal(), byCell ? row.cell() : 0);
      LedgerCell.Standing standing = row.standing();
      double h = scale == null ? 1 : scale.scale(standing);
      double meanY = 0, meanB = 0;
      for (int i = 0; i < m; i++) {
        int action = measured[i];
        double b = 0;
        for (int f = 0; f < LedgerColumn.COUNT; f++) b += beta[f] * row.ledger(action, f);
        terms[i] = b;
        meanY += values[action];
        meanB += b;
      }
      meanY /= m;
      meanB /= m;
      double xx = 0, xv = 0, vv = 0, eCurrent = 0, eShadow = 0;
      for (int i = 0; i < m; i++) {
        double x = values[measured[i]] - meanY, v = terms[i] - meanB;
        xx += x * x;
        xv += x * v;
        vv += v * v;
        eCurrent += (x - v) * (x - v);
        eShadow += (x - h * v) * (x - h * v);
      }
      double raw = xx / m, cur = eCurrent / m, shadow = eShadow / m;
      sums.add(standing, raw, xv / m, vv / m);
      rows++;
      slots += m;
      rowJ[0] += raw;
      rowJ[1] += cur;
      rowJ[2] += shadow;
      slotResidual[0] += xx;
      slotResidual[1] += eCurrent;
      slotResidual[2] += eShadow;
      scaleSum += h;
      scaleMin = Math.min(scaleMin, h);
      scaleMax = Math.max(scaleMax, h);
      histogram[Math.min((int) (h * HISTOGRAM), HISTOGRAM - 1)]++;
      long game = row.gameId();
      boolean last = standing.round() >= variant.lastRound();
      groups[0].add(game, raw, cur, shadow);
      groups[last ? 2 : 1].add(game, raw, cur, shadow);
      if (last) {
        int band = band(GAP_BANDS, standing.gap());
        groups[3 + standing.rank() * GAP_BANDS.length + band].add(game, raw, cur, shadow);
      }
      int played = -1;
      for (int i = 0; i < m && played < 0; i++) if (measured[i] == row.played()) played = i;
      double offset = played < 0 ? Double.NaN : (terms[played] - meanB) * m / (m - 1);
      boolean first = standing.rank() == 0;
      boolean close = last && standing.gap() < CLOSE_GAP, wide = last && standing.gap() >= WIDE_GAP;
      audit(0, game, meanB, h, played >= 0, offset);
      audit(last ? 2 : 1, game, meanB, h, played >= 0, offset);
      if (close && first) audit(3, game, meanB, h, played >= 0, offset);
      if (wide && first) audit(4, game, meanB, h, played >= 0, offset);
      if (close) audit(5, game, meanB, h, played >= 0, offset);
      if (wide) audit(6, game, meanB, h, played >= 0, offset);
      audit(7 + row.source().ordinal(), game, meanB, h, played >= 0, offset);
      audit(
          AUDIT_GROUPS.size() + row.decisionClass().ordinal(), game, meanB, h, played >= 0, offset);
    }

    private void audit(int group, long game, double mean, double h, boolean played, double offset) {
      audits[group].add(game, mean, h * mean, played, offset, h * offset);
    }

    /** 足した行の数。 */
    long rows() {
      return rows;
    }

    /** 補正率を掛けた補正のスロットの R²（1 − 残差 / 生の値）。生の値が 0 なら NaN。 */
    double shadowR2() {
      return slotResidual[0] > 0 ? 1 - slotResidual[2] / slotResidual[0] : Double.NaN;
    }

    /** 足した行の和。 */
    Sums sums() {
      return sums;
    }

    /**
     * 報告を作る。和了の判断の差（{@code gates}）は空にしておく。
     *
     * @param mode 補正率のモード（設定に書く名前）
     * @param applied 教師に実際に掛けた腕
     * @param fallback 補正率を 1 にした理由（補正率を掛けたなら null）
     * @param next 次の反復の補正率（当てはめられなければ null）
     */
    LedgerReport.Calibration report(
        String mode,
        LedgerCalibrationKind kind,
        LedgerReport.Arm applied,
        String fallback,
        LedgerCalibrator next) {
      List<LedgerReport.Band> bands = new ArrayList<>(groups.length);
      for (Group group : groups) bands.add(group.band());
      List<LedgerReport.Audit> checked = new ArrayList<>(audits.length);
      for (AuditGroup audit : audits) checked.add(audit.audit());
      double rawSlots = slotResidual[0];
      return new LedgerReport.Calibration(
          mode,
          kind,
          applied.reportName(),
          scale == null ? null : LedgerReport.Origin.of(scale),
          fallback,
          rows,
          slots,
          LedgerReport.Arms.of(rowJ[0] / rows, rowJ[1] / rows, rowJ[2] / rows),
          LedgerReport.Arms.of(0, 1 - slotResidual[1] / rawSlots, 1 - slotResidual[2] / rawSlots),
          new LedgerReport.Scale(
              LedgerReport.finite(scaleSum / rows),
              rows == 0 ? null : scaleMin,
              rows == 0 ? null : scaleMax,
              Arrays.stream(histogram).boxed().toList()),
          bands,
          checked,
          List.of(),
          next == null ? null : LedgerReport.Next.of(next, sums));
    }
  }

  /**
   * 補正の向きの監査の群 1 つの和。行ごとに、測った手の補正の平均 b̄（中心化しない）と、打った手の補正のほかの測った手の平均からの差
   * (b_打った − b̄)·m/(m − 1) を、今（β·L）と影（h(o)·β·L）の腕で足し、半荘ごとにも束ねる。平均の標準誤差は、半荘 g の和 S_g と行の数
   * n_g から se² = G/(G−1)·Σ_g (S_g − 平均·n_g)²/N²。
   */
  private static final class AuditGroup {
    final String name;
    final Map<Long, double[]> games = new HashMap<>();
    long rows, played;

    AuditGroup(String name) {
      this.name = name;
    }

    void add(
        long game,
        double mean,
        double scaledMean,
        boolean hasPlayed,
        double offset,
        double scaled) {
      double[] g = games.computeIfAbsent(game, ignored -> new double[6]);
      g[0]++;
      g[1] += mean;
      g[2] += scaledMean;
      rows++;
      if (hasPlayed) {
        g[3]++;
        g[4] += offset;
        g[5] += scaled;
        played++;
      }
    }

    LedgerReport.Audit audit() {
      return new LedgerReport.Audit(
          name, rows, games.size(), played, signed(0, 1, 2, rows), signed(3, 4, 5, played));
    }

    /** 行の数の列 {@code count}、今と影の和の列 {@code current}・{@code shadow} から、半荘で束ねた平均と標準誤差を求める。 */
    private LedgerReport.Signed signed(int count, int current, int shadow, long n) {
      double totalCurrent = 0, totalShadow = 0;
      for (double[] g : games.values()) {
        totalCurrent += g[current];
        totalShadow += g[shadow];
      }
      double meanCurrent = totalCurrent / n, meanShadow = totalShadow / n;
      double eCurrent = 0, eShadow = 0;
      int clusters = 0;
      for (double[] g : games.values()) {
        if (g[count] == 0) continue;
        clusters++;
        double c = g[current] - meanCurrent * g[count], s = g[shadow] - meanShadow * g[count];
        eCurrent += c * c;
        eShadow += s * s;
      }
      double inflation = clusters >= 2 ? clusters / (clusters - 1.0) : Double.NaN;
      return new LedgerReport.Signed(
          LedgerReport.finite(meanCurrent),
          LedgerReport.finite(Math.sqrt(inflation * eCurrent) / n),
          LedgerReport.finite(meanShadow),
          LedgerReport.finite(Math.sqrt(inflation * eShadow) / n));
    }
  }

  /** 帯の名前（下の境と上の境。最後の帯の上は inf）。 */
  static String bandName(int[] edges, int band, String separator) {
    return edges[band]
        + separator
        + (band + 1 < edges.length ? String.valueOf(edges[band + 1]) : "inf");
  }

  /** 値の入る帯の番号。 */
  static int band(int[] edges, int value) {
    int band = 0;
    while (band + 1 < edges.length && value >= edges[band + 1]) band++;
    return band;
  }

  /**
   * 局面の帯 1 つの和。行の数と腕ごとの J の和を、半荘ごとにも束ねる（差の標準誤差を半荘で束ねて求める）。
   *
   * <p>差 d = mean(J_腕 − J_基準) の標準誤差は、半荘 g の和 S_g と行の数 n_g から se² = G/(G−1)·Σ_g (S_g^腕 − S_g^基準 − d·n_g)²/N²。
   * 比 1 − J_影/J_今 は、R = ΣJ_影/ΣJ_今 で線形化した S_g^影 − R·S_g^今 を J_今 の平均で割って同じ式で求める。
   */
  private static final class Group {
    final String name;
    final Map<Long, double[]> games = new HashMap<>();
    long rows;
    final double[] sums = new double[3];

    Group(String name) {
      this.name = name;
    }

    void add(long game, double raw, double current, double shadow) {
      double[] byGame = games.computeIfAbsent(game, ignored -> new double[4]);
      byGame[0]++;
      byGame[1] += raw;
      byGame[2] += current;
      byGame[3] += shadow;
      rows++;
      sums[0] += raw;
      sums[1] += current;
      sums[2] += shadow;
    }

    LedgerReport.Band band() {
      double n = rows;
      double raw = sums[0] / n, current = sums[1] / n, shadow = sums[2] / n;
      double dCurrent = current - raw, dShadow = shadow - raw;
      double ratio = sums[2] / sums[1];
      double eCurrent = 0, eShadow = 0, eRatio = 0;
      for (double[] g : games.values()) {
        double c = g[2] - g[1] - dCurrent * g[0];
        double s = g[3] - g[1] - dShadow * g[0];
        double r = (g[3] - ratio * g[2]) / current;
        eCurrent += c * c;
        eShadow += s * s;
        eRatio += r * r;
      }
      int count = games.size();
      double inflation = count >= 2 ? count / (count - 1.0) : Double.NaN;
      return new LedgerReport.Band(
          name,
          rows,
          count,
          LedgerReport.Arms.of(raw, current, shadow),
          LedgerReport.finite(dCurrent),
          LedgerReport.finite(Math.sqrt(inflation * eCurrent) / n),
          LedgerReport.finite(dShadow),
          LedgerReport.finite(Math.sqrt(inflation * eShadow) / n),
          LedgerReport.finite(1 - ratio),
          LedgerReport.finite(Math.sqrt(inflation * eRatio) / n));
    }
  }

  /**
   * 和了の判断の教師の差。南場の最後の局から後の主経路の行で、和了（ツモ・ロン）を測った行について、和了と比べる手の教師の差を 3 つの腕で
   * 求める。比べる手は、ロンなら測った見送り、ツモなら測った打牌（立直の打牌を含む）のうちアンカーの s が最大の手で、Y では選ばない。
   * 差は、生が Y_和了 − Y_比べる手、今が その差 − (β·L_和了 − β·L_比べる手)、影が その差 − h(o)·(β·L_和了 − β·L_比べる手)。和了の手が
   * 複数あるときは、手の順で最初の和了を使う。帯は、全ての順位をまとめた FINAL_END と、首位の最も近い他家との点差の帯。
   *
   * <p>腕ごとの標準偏差の、生との差の標準誤差は、標準偏差の影響関数 ((v − μ)² − σ̂²)·n/(n − 1)/(2s) を半荘で束ねて求める。
   */
  static final class Gates {
    private final LedgerCorrection current;
    private final boolean byCell;
    private final LedgerCalibrator scale;
    private final Variant variant;
    private final LedgerCell reader;
    private final Observation observation;
    private final Map<String, List<Entry>> byGroup = new LinkedHashMap<>();

    /** 1 行の、和了と比べる手の教師の差（生・今・影）。 */
    private record Entry(long game, double raw, double current, double shadow) {
      double arm(int arm) {
        return arm == 0 ? raw : arm == 1 ? current : shadow;
      }
    }

    Gates(LedgerCorrection current, LedgerCalibrator scale, Variant variant) {
      this.current = Objects.requireNonNull(current, "current");
      byCell = current.byCell() != null;
      this.scale = scale;
      this.variant = Objects.requireNonNull(variant, "variant");
      reader = LedgerCell.of(variant);
      observation = new Observation(variant);
      for (String kind : List.of("tsumo", "ron")) {
        byGroup.put("FINAL_END " + kind, new ArrayList<>());
        for (int band = 0; band < GATE_BANDS.length; band++)
          byGroup.put("first_" + bandName(GATE_BANDS, band, "_") + " " + kind, new ArrayList<>());
      }
    }

    /**
     * 行を 1 つ数える。
     *
     * @param scores 行のアンカーの s（合法手の順。無ければ null で、ツモの行は比べる手を選べないので数えない）
     */
    void add(Row row, float[] scores) {
      if (row.purpose() != Row.Purpose.TRAIN
          || row.source() != Row.Source.MAIN
          || row.decisionClass() != DecisionClass.FINAL_END
          || row.ledgerFeatures() != LedgerColumn.COUNT) {
        return;
      }
      long[] words = row.observation();
      LedgerCell.Standing standing = reader.standing(words, 0);
      if (standing.round() < variant.lastRound()) {
        return;
      }
      Observation o = observation.bind(words, 0);
      float[] values = row.values();
      int win = -1;
      for (int a = 0; a < values.length && win < 0; a++) {
        Action.Type type = o.action(a).type();
        if (!Float.isNaN(values[a])
            && (type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI)) {
          win = a;
        }
      }
      if (win < 0) {
        return;
      }
      boolean tsumo = o.action(win).type() == Action.Type.TSUMO_AGARI;
      if (tsumo && scores == null) {
        return;
      }
      int other = -1;
      for (int a = 0; a < values.length; a++) {
        if (Float.isNaN(values[a])) continue;
        Action.Type type = o.action(a).type();
        boolean compared =
            tsumo
                ? type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI
                : type == Action.Type.PASS;
        if (compared && (other < 0 || scores != null && scores[a] > scores[other])) other = a;
      }
      if (other < 0) {
        return;
      }
      double[] beta =
          current.coefficients(
              row.gameId(), row.decisionClass().ordinal(), byCell ? reader.of(words, 0) : 0);
      double db = 0;
      for (int f = 0; f < LedgerColumn.COUNT; f++)
        db += beta[f] * ((double) row.ledger(win, f) - row.ledger(other, f));
      double dy = (double) values[win] - values[other];
      double h = scale == null ? 1 : scale.scale(standing);
      Entry entry = new Entry(row.gameId(), dy, dy - db, dy - h * db);
      String kind = tsumo ? " tsumo" : " ron";
      byGroup.get("FINAL_END" + kind).add(entry);
      if (standing.rank() == 0) {
        byGroup
            .get("first_" + bandName(GATE_BANDS, band(GATE_BANDS, standing.gap()), "_") + kind)
            .add(entry);
      }
    }

    /** 帯ごとの結果（2 行以上の帯だけ）。 */
    List<LedgerReport.Gate> gates() {
      List<LedgerReport.Gate> gates = new ArrayList<>();
      byGroup.forEach(
          (key, entries) -> {
            if (entries.size() >= 2) gates.add(gate(key, entries));
          });
      return List.copyOf(gates);
    }

    private static LedgerReport.Gate gate(String key, List<Entry> entries) {
      int n = entries.size();
      double[] mean = new double[3], var = new double[3], sd = new double[3];
      for (Entry e : entries) for (int arm = 0; arm < 3; arm++) mean[arm] += e.arm(arm);
      for (int arm = 0; arm < 3; arm++) mean[arm] /= n;
      for (Entry e : entries)
        for (int arm = 0; arm < 3; arm++) {
          double d = e.arm(arm) - mean[arm];
          var[arm] += d * d;
        }
      for (int arm = 0; arm < 3; arm++) {
        sd[arm] = Math.sqrt(var[arm] / (n - 1));
        var[arm] /= n;
      }
      double correction = mean[0] - mean[1], scaled = mean[0] - mean[2];
      Map<Long, double[]> games = new LinkedHashMap<>();
      for (Entry e : entries) {
        double[] g = games.computeIfAbsent(e.game(), ignored -> new double[4]);
        double raw = influence(e.raw(), mean[0], var[0], sd[0], n);
        g[0] += influence(e.current(), mean[1], var[1], sd[1], n) - raw;
        g[1] += influence(e.shadow(), mean[2], var[2], sd[2], n) - raw;
        g[2] += e.raw() - e.current() - correction;
        g[3] += e.raw() - e.shadow() - scaled;
      }
      double current = 0, shadow = 0, eCorrection = 0, eScaled = 0;
      for (double[] g : games.values()) {
        current += g[0] * g[0];
        shadow += g[1] * g[1];
        eCorrection += g[2] * g[2];
        eScaled += g[3] * g[3];
      }
      int count = games.size();
      double inflation = count >= 2 ? count / (count - 1.0) : Double.NaN;
      int split = key.indexOf(' ');
      return new LedgerReport.Gate(
          key.substring(0, split),
          key.substring(split + 1),
          n,
          count,
          LedgerReport.Arms.of(mean[0], mean[1], mean[2]),
          LedgerReport.Arms.of(sd[0], sd[1], sd[2]),
          new LedgerReport.Signed(
              LedgerReport.finite(correction),
              LedgerReport.finite(Math.sqrt(inflation * eCorrection) / n),
              LedgerReport.finite(scaled),
              LedgerReport.finite(Math.sqrt(inflation * eScaled) / n)),
          LedgerReport.finite(sd[1] - sd[0]),
          LedgerReport.finite(Math.sqrt(inflation * current) / n),
          LedgerReport.finite(sd[2] - sd[0]),
          LedgerReport.finite(Math.sqrt(inflation * shadow) / n));
    }

    private static double influence(double value, double mean, double var, double sd, int n) {
      double d = value - mean;
      return (d * d - var) * n / (n - 1) / (2 * sd);
    }
  }
}
