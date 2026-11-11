package com.epsilon.prism.luck;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * 運の精算の係数 β を当てはめる。β は、帳簿 L（{@link LedgerColumn}）を使って Y を補正するときの係数。行ごとに、測った手の Y と L を
 * その行の中で中心化し（ỹ = Y − mean_M Y、l̃ = L − mean_M L）、判断のクラス c ごとに最小二乗で β を求める。最小二乗には、列ごとの対角に
 * 比例する弱い ridge を付ける。当てはめるのは補正に使う組の列だけで、ほかの列の β は 0。
 *
 * <p>交差適合: 半荘 ID の偶奇が g（{@code gameId & 1}）の行に掛ける β_{g,c} は、偶奇が 1 − g の行だけで当てはめる。区分に要る行数
 * （列 1 つあたりの数 × 当てはめる列の数）に足りないクラスは、偶奇 1 − g の全てのクラスをまとめて当てはめ、それでも足りなければ 0 にする。
 * 半荘どうしは独立で（牌山と乱数は半荘 ID ごとのハッシュで決まる）、β はその行の半荘と独立なので、E[β·L_a | o, β] = β·E[L_a | o] = 0 と
 * なる。よって補正した Ỹ_a = Y_a − β·L_a も E[Ỹ_a | o] = Q(o, a) を満たす。
 *
 * <p>帳簿のない行と、測った手の帳簿に有限でない値が 1 つでもある行は、当てはめに入れず、補正もしない。帳簿の NaN は「帳簿を付けなかった」
 * ことだけを表し、o の時点で決まるので、この選び方で偏りは生じない。行に重みは付けない（学習の重みは β の不偏性に関係しない）。当てはめの
 * 結果は、行の集合と順だけで決まる。
 *
 * <p>報告（{@link LedgerReport}）には次のものを載せる。
 *
 * <ul>
 *   <li>R² = 1 − ΣSSE/Σyy と、同等の分岐の本数 Σyy/ΣSSE。R² は、行の中で中心化した Y の分散がどれだけ減ったかを表す。
 *       SSE_{g,c} = yy − 2β·b + βᵀAβ はホールドアウトの残差で、偶奇 g の行の和に、もう一方の偶奇で当てはめた β を使って求める。
 *   <li>組ごとに、その組の列だけで当てはめたときの R²。
 *   <li>偏りの確かめ。列ごとの生の L の平均・標準偏差・0 でない割合と、半荘で束ねた z。
 * </ul>
 *
 * <p>系譜では、収集しながら和を作る（{@link WriterTap}）。和は行を足す順で決まる。{@link WriterTap} は、確定したコホートから行を選んで
 * 読み直す {@link #fit} と同じ行を同じ順に足すので、結果はビット単位で同じになる。
 */
public final class LedgerFit {

  private static final int F = LedgerColumn.COUNT, CLASSES = DecisionClass.COUNT;

  private static final LedgerColumn[] COLUMNS = LedgerColumn.values();

  /** 報告で組ごとの R² を求める組（宣言の順。列を持つ組だけ）。 */
  private static final List<Part> PARTS =
      Arrays.stream(LedgerGroup.values()).map(Part::of).filter(part -> part.columns().length > 0).toList();

  /** 当てはめる列（補正に使う組の列、宣言の順）。 */
  private final int[] fitted;

  /** ridge の係数。列ごとに対角 A_ff に掛ける（列を標準化してから掛ける ridge と同じになる）。 */
  private final double ridge;

  /** 区分の当てはめに要る行の数。足りなければクラスをまとめて当てはめる。 */
  private final long minRows;

  /**
   * @param apply 補正に使う組（列を持つ組を 1 つ以上含む）
   * @param ridge ridge の係数（正）
   * @param minRowsPerColumn 区分に要る行の数を、当てはめる列 1 つあたりで表した値
   */
  public LedgerFit(Set<LedgerGroup> apply, double ridge, int minRowsPerColumn) {
    fitted = columns(apply);
    if (fitted.length == 0) throw new IllegalArgumentException("No ledger column belongs to " + apply);
    if (!(ridge > 0 && Double.isFinite(ridge))) throw new IllegalArgumentException("Invalid ridge " + ridge);
    if (minRowsPerColumn < 1) throw new IllegalArgumentException("Invalid minRowsPerColumn " + minRowsPerColumn);
    this.ridge = ridge;
    minRows = (long) minRowsPerColumn * fitted.length;
  }

  /**
   * 組ごとの R² を求める 1 つの組。
   *
   * @param group 組
   * @param columns 組に属する列（宣言の順）
   */
  private record Part(LedgerGroup group, int[] columns) {
    static Part of(LedgerGroup group) {
      return new Part(group, LedgerFit.columns(Set.of(group)));
    }
  }

  /** 組に属する列の番号を、宣言の順に返す。 */
  private static int[] columns(Set<LedgerGroup> groups) {
    return Arrays.stream(COLUMNS).filter(c -> groups.contains(c.group())).mapToInt(Enum::ordinal).toArray();
  }

  /**
   * 当てはめの結果。
   *
   * @param correction 学習の行に掛ける交差適合の係数
   * @param report 報告
   */
  public record Result(LedgerCorrection correction, LedgerReport report) {}

  /** 行を順に足して当てはめる（系譜では、行のストアからバッチごとに読み出す行の並びを渡す）。 */
  public Result fit(Iterable<Row> rows) {
    long started = System.nanoTime();
    var accumulator = new Accumulator();
    for (Row row : rows) accumulator.add(FitRow.of(row));
    return result(accumulator, started);
  }

  /** {@link FitRow} の形の行を順に足して当てはめる。和に足す順は {@link #fit} と同じ。 */
  Result fitRows(Iterable<? extends FitRow> rows) {
    long started = System.nanoTime();
    var accumulator = new Accumulator();
    for (FitRow row : rows) accumulator.add(row);
    return result(accumulator, started);
  }

  /**
   * コホートのライターに付けて、収集しながら和を作る受け取り先を返す。
   *
   * @param shards コホートのライターの数（シャードの数）
   * @param rows 和に入れる行を選ぶ条件（行を読み直して当てはめるときと同じ選び方）
   */
  public WriterTap tap(int shards, Predicate<RowCodec.Index> rows) {
    return new WriterTap(this, shards, rows);
  }

  /**
   * 収集しながら和を作る当てはめ（コホートのライターに付ける {@link CohortWriter.Tap}）。呼び出し元が選んだ行を、確定したコホートから
   * 読み直すときと同じ順（シャード 0 の行を書いた順、続いてシャード 1、…）で和に足す。シャード 0 の行は、そのライターのスレッドですぐ足す。
   * ほかのシャードの行は、測った手の Y と帳簿を書いた順に貯めておき（{@link Tape}）、{@link #fit} でシャードの順に足す。
   *
   * <p>{@link #written} は、シャードごとに 1 つのスレッド（そのシャードのライター）から呼ぶ。{@link #fit} は、ライターを止めた後
   * （{@link CohortWriter#seal}）に 1 度だけ呼ぶ。
   */
  public static final class WriterTap implements CohortWriter.Tap {
    private final LedgerFit fit;
    private final Predicate<RowCodec.Index> rows;
    private final Accumulator accumulator = new Accumulator();
    private final Tape[] tapes;
    private boolean fitted;

    private WriterTap(LedgerFit fit, int shards, Predicate<RowCodec.Index> rows) {
      if (shards < 1) throw new IllegalArgumentException("shard count must be positive: " + shards);
      this.fit = fit;
      this.rows = Objects.requireNonNull(rows, "rows");
      tapes = new Tape[shards];
      for (int shard = 1; shard < shards; shard++) tapes[shard] = new Tape();
    }

    @Override
    public void written(int shard, Row row, RowCodec.Index index) {
      if (rows.test(index)) add(shard, FitRow.of(row));
    }

    /** シャード {@code shard} のライターが次に書いた行を足す。シャード 0 ならすぐ足し、ほかのシャードなら貯めておく。 */
    void add(int shard, FitRow row) {
      if (shard < 0 || shard >= tapes.length)
        throw new IllegalArgumentException("shard " + shard + " out of " + tapes.length);
      if (shard == 0) accumulator.add(row);
      else tapes[shard].append(row);
    }

    /**
     * 貯めた行をシャードの順に足して解く。貯めた行はここで解放する。
     *
     * @throws IllegalStateException 2 度目に呼んだとき
     */
    public Result fit() {
      if (fitted) throw new IllegalStateException("Ledger stream is already fitted");
      fitted = true;
      long started = System.nanoTime();
      for (int shard = 1; shard < tapes.length; shard++) {
        tapes[shard].replay(accumulator);
        tapes[shard] = null;
      }
      return fit.result(accumulator, started);
    }
  }

  /**
   * 1 つのシャードで貯めた行。行ごとに半荘・判断のクラスを持ち、測った手の Y と帳簿の列を、行を書いた順・手の順に持つ。帳簿は fp16 で
   * 持つ（コホートの行の帳簿は fp16 で正確に表せる。{@link RowCodec}）。手は {@value #BLOCK} 手ずつのブロックに置き、1 行の手がブロックを
   * またがないようにする。飛ばす行（{@link Accumulator#measured(FitRow, float[], int[])} が −1 を返す行）は、数だけを数える。
   */
  private static final class Tape {
    static final int BLOCK = 1 << 16;
    static final DecisionClass[] CLASS = DecisionClass.values();

    final List<float[]> values = new ArrayList<>();
    final List<short[]> terms = new ArrayList<>();
    long[] games = new long[1024];
    byte[] classes = new byte[1024], counts = new byte[1024];

    /** 行の最初の手の位置（ブロックの番号 × BLOCK + ブロックの中の位置）。 */
    long[] starts = new long[1024];

    int rows;

    /** 次の行の最初の手の位置。 */
    long slot;

    long skipped;

    /** 測った手の番号の作業領域（行ごとに使い回す）。 */
    int[] measured = new int[0];

    void append(FitRow row) {
      float[] source = row.values();
      if (measured.length < source.length) measured = new int[source.length];
      int m = Accumulator.measured(row, source, measured);
      if (m < 0) {
        skipped++;
        return;
      }
      if (slot % BLOCK + m > BLOCK) slot += BLOCK - slot % BLOCK;
      if (slot / BLOCK == values.size()) {
        values.add(new float[BLOCK]);
        terms.add(new short[BLOCK * F]);
      }
      if (rows == games.length) {
        int capacity = rows * 2;
        games = Arrays.copyOf(games, capacity);
        classes = Arrays.copyOf(classes, capacity);
        counts = Arrays.copyOf(counts, capacity);
        starts = Arrays.copyOf(starts, capacity);
      }
      float[] y = values.get((int) (slot / BLOCK));
      short[] l = terms.get((int) (slot / BLOCK));
      int at = (int) (slot % BLOCK);
      for (int i = 0; i < m; i++) {
        y[at + i] = source[measured[i]];
        for (int f = 0; f < F; f++) {
          float term = row.ledger(measured[i], f);
          short half = Float.floatToFloat16(term);
          if (Float.floatToRawIntBits(Float.float16ToFloat(half)) != Float.floatToRawIntBits(term))
            throw new IllegalArgumentException("ledger term is not exact in fp16: " + term);
          l[(at + i) * F + f] = half;
        }
      }
      games[rows] = row.gameId();
      classes[rows] = (byte) row.decisionClass().ordinal();
      counts[rows] = (byte) m;
      starts[rows] = slot;
      rows++;
      slot += m;
    }

    /**
     * 貯めた行を、書いた順に和へ足す。行は測った手だけを持つ形で足すが、測った手の順と値は元の行と同じなので、足す値はビット単位で同じになる。
     */
    void replay(Accumulator accumulator) {
      accumulator.skipped += skipped;
      for (int r = 0; r < rows; r++) accumulator.add(new Stored(r));
    }

    /** 貯めた r 番目の行。 */
    private final class Stored implements FitRow {
      final int row, start;
      final float[] y;
      final short[] l;

      Stored(int row) {
        this.row = row;
        int block = (int) (starts[row] / BLOCK);
        start = (int) (starts[row] % BLOCK);
        y = values.get(block);
        l = terms.get(block);
      }

      @Override
      public long gameId() {
        return games[row];
      }

      @Override
      public DecisionClass decisionClass() {
        return CLASS[classes[row]];
      }

      @Override
      public float[] values() {
        return Arrays.copyOfRange(y, start, start + counts[row]);
      }

      @Override
      public int ledgerFeatures() {
        return F;
      }

      @Override
      public float ledger(int action, int column) {
        return Float.float16ToFloat(l[(start + action) * F + column]);
      }
    }
  }

  /** (偶奇, クラス) ごとの和。A = Σ l̃ l̃ᵀ、b = Σ l̃ ỹ、yy = Σ ỹ² と、行数と手の数を持つ。 */
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

    /** ホールドアウトの残差 yy − 2β·b + βᵀAβ を返す。 */
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
   * (A + ridge·diag(A)) β = b を、指定した列だけで解く。指定しない列と A_ff = 0 の列の係数は 0 にする。d_f = sqrt(A_ff) で割って
   * 相関行列の形にしてから Cholesky 分解で解くので、結果は列の大きさに依らない。
   */
  static double[] solve(Sums sums, int[] features, double ridge) {
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
      m[i][i] += ridge;
      r[i] = sums.b[used[i]] / scale[i];
    }
    // Cholesky 分解 m = L Lᵀ（下三角を m に上書きする）
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

  /** 行を足していく和（当てはめと偏りの確かめに使う）。 */
  static final class Accumulator {
    final Sums[][] sums = new Sums[2][CLASSES];

    /** クラスと列ごとの、生の L の和・平方の和・L が 0 でない手の数。 */
    final double[][] sum = new double[CLASSES][F], square = new double[CLASSES][F], nonzero = new double[CLASSES][F];

    /** 半荘で束ねる和。半荘 ID をキーに [手の数, 列ごとの L の和] を持つ（全体の z に使う）。 */
    final Map<Long, double[]> byGame = new HashMap<>();

    long rows, slots, skipped;

    /** 測った手の番号の作業領域（行ごとに使い回す）。 */
    private int[] measured = new int[0];

    /** 行の中での帳簿の平均と、中心化した帳簿を置く作業領域（行ごとに使い回す）。 */
    private final double[] meanL = new double[F], centred = new double[F];

    /** 前に足した行の半荘と、その半荘で束ねる和。 */
    private long lastGame;
    private double[] lastCluster;

    Accumulator() {
      for (int p = 0; p < 2; p++) for (int c = 0; c < CLASSES; c++) sums[p][c] = new Sums();
    }

    /**
     * 当てはめに入れる行なら、測った手の番号を手の順に書き、その数を返す。帳簿がない行、測った手が 2 つ未満の行、測った手の帳簿に有限で
     * ない値が 1 つでもある行は飛ばすので、−1 を返す。
     *
     * @param row 行
     * @param values 行の合法手ごとの Y
     * @param into 測った手の番号を先頭から書く先（長さは合法手の数以上）
     * @return 測った手の数。飛ばす行なら −1
     * @throws IllegalStateException 帳簿の列の数が {@link LedgerColumn#COUNT} でないとき
     */
    static int measured(FitRow row, float[] values, int[] into) {
      int features = row.ledgerFeatures();
      if (features == 0) return -1;
      if (features != F)
        throw new IllegalStateException(
            "Ledger of game " + row.gameId() + " has " + features + " features, expected " + F);
      int m = 0;
      for (int a = 0; a < values.length; a++) if (!Float.isNaN(values[a])) into[m++] = a;
      if (m < 2) return -1;
      for (int i = 0; i < m; i++)
        for (int f = 0; f < F; f++) if (!Float.isFinite(row.ledger(into[i], f))) return -1;
      return m;
    }

    void add(FitRow row) {
      float[] values = row.values();
      if (measured.length < values.length) measured = new int[values.length];
      int m = measured(row, values, measured);
      if (m < 0) {
        skipped++;
        return;
      }
      double meanY = 0;
      Arrays.fill(meanL, 0);
      for (int i = 0; i < m; i++) {
        int a = measured[i];
        meanY += values[a];
        for (int f = 0; f < F; f++) meanL[f] += row.ledger(a, f);
      }
      meanY /= m;
      for (int f = 0; f < F; f++) meanL[f] /= m;
      int c = row.decisionClass().ordinal();
      Sums target = sums[(int) (row.gameId() & 1)][c];
      // 偏りの確かめには中心化しない生の L を使う（半荘で束ねる和は行ごとに 1 回だけ引く）
      double[] game = cluster(row.gameId());
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

    /** 半荘で束ねる和を返す（無ければ作る）。同じ半荘の行が続く間は表を引かない。 */
    private double[] cluster(long gameId) {
      if (lastCluster == null || gameId != lastGame) {
        lastCluster = byGame.computeIfAbsent(gameId, ignored -> new double[1 + F]);
        lastGame = gameId;
      }
      return lastCluster;
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

  /** 足し終えた和から係数を当てはめ、報告を作る。 */
  private Result result(Accumulator sums, long started) {
    CrossFit both = new CrossFit(sums.sums);
    Map<String, LedgerReport.Line> byClass = new LinkedHashMap<>();
    for (DecisionClass c : DecisionClass.values())
      if (both.classes[c.ordinal()].rows() > 0) byClass.put(c.name(), both.classes[c.ordinal()].line());
    var report =
        new LedgerReport(
            sums.rows,
            sums.slots,
            sums.skipped,
            both.all(),
            byClass,
            new LedgerReport.Bias(
                features(sums.totalSums(), sums.totalSquares(), sums.totalNonzero(), sums.slots, sums.byGame)),
            (System.nanoTime() - started) / 1e9);
    return new Result(both.correction(), report);
  }

  /**
   * 列ごとの偏りを確かめる。z は、半荘で束ねた平均を線形化して求める。mean = ΣS_g/Σn_g、se² = G/(G−1)·Σ(S_g − mean·n_g)²/(Σn_g)²。
   */
  private static List<LedgerReport.FeatureBias> features(
      double[] sums, double[] squares, double[] nonzero, long count, Map<Long, double[]> clusters) {
    List<LedgerReport.FeatureBias> result = new ArrayList<>(F);
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
      result.add(
          new LedgerReport.FeatureBias(
              COLUMNS[f].reportName(),
              LedgerReport.finite(mean),
              LedgerReport.finite(sd),
              LedgerReport.finite(share),
              LedgerReport.finite(z)));
    }
    return result;
  }

  /** 偶奇 × クラスごとの和と、それを使った交差適合。 */
  private final class CrossFit {
    final Sums[] pooledByParity = new Sums[2];
    final ClassFit[] classes = new ClassFit[CLASSES];

    CrossFit(Sums[][] parity) {
      for (int p = 0; p < 2; p++) pooledByParity[p] = Sums.of(parity[p]);
      for (int c = 0; c < CLASSES; c++) classes[c] = new ClassFit(parity[0][c], parity[1][c], pooledByParity);
    }

    /** 交差適合の係数を返す（偶奇 g・クラス c の係数は、偶奇 1 − g の行で当てはめたもの）。 */
    LedgerCorrection correction() {
      double[][][] beta = new double[2][CLASSES][];
      for (int p = 0; p < 2; p++) for (int c = 0; c < CLASSES; c++) beta[p][c] = classes[c].byFold[p];
      return new LedgerCorrection(beta);
    }

    /** 全てのクラスをまとめた区分の結果を返す。残差は、クラスごとの係数で求めた残差の和にする。 */
    LedgerReport.Line all() {
      Sums[] folds = {pooledByParity[0], pooledByParity[1]};
      double[] sse = new double[2];
      double[] groupSse = new double[PARTS.size()];
      for (ClassFit fit : classes) {
        for (int p = 0; p < 2; p++) sse[p] += fit.sse[p];
        for (int k = 0; k < groupSse.length; k++) groupSse[k] += fit.groupSse[k];
      }
      double[][] byFold = new double[2][];
      boolean[] fallback = new boolean[2];
      for (int p = 0; p < 2; p++) {
        fallback[p] = folds[1 - p].rows < minRows;
        byFold[p] = fallback[p] ? new double[F] : solve(folds[1 - p], fitted, ridge);
      }
      Sums everything = Sums.of(folds);
      return line(
          folds,
          sse,
          groupSse,
          everything.rows >= minRows ? solve(everything, fitted, ridge) : new double[F],
          byFold,
          fallback);
    }
  }

  /** 1 つのクラスの交差適合。偶奇 g の係数は、偶奇 1 − g のそのクラスの和（足りなければクラスをまとめた和）で当てはめる。 */
  private final class ClassFit {
    final Sums[] folds;
    final double[][] byFold = new double[2][];
    final boolean[] fallback = new boolean[2];
    final double[] sse = new double[2];
    final double[] groupSse = new double[PARTS.size()];

    ClassFit(Sums even, Sums odd, Sums[] pooledByParity) {
      folds = new Sums[] {even, odd};
      for (int p = 0; p < 2; p++) {
        Sums other = folds[1 - p];
        Sums pooled = pooledByParity[1 - p];
        Sums source = other.rows >= minRows ? other : pooled.rows >= minRows ? pooled : null;
        fallback[p] = source != other;
        byFold[p] = source == null ? new double[F] : solve(source, fitted, ridge);
        sse[p] = folds[p].sse(byFold[p]);
        for (int k = 0; k < groupSse.length; k++) {
          double[] partial = source == null ? new double[F] : solve(source, PARTS.get(k).columns(), ridge);
          groupSse[k] += folds[p].sse(partial);
        }
      }
    }

    long rows() {
      return folds[0].rows + folds[1].rows;
    }

    LedgerReport.Line line() {
      Sums both = Sums.of(folds);
      return LedgerFit.line(
          folds, sse, groupSse, both.rows >= minRows ? solve(both, fitted, ridge) : new double[F], byFold, fallback);
    }
  }

  private static LedgerReport.Line line(
      Sums[] folds, double[] sse, double[] groupSse, double[] beta, double[][] byFold, boolean[] fallback) {
    long rows = folds[0].rows + folds[1].rows, slots = folds[0].slots + folds[1].slots;
    double yy = folds[0].yy + folds[1].yy, residual = sse[0] + sse[1];
    Map<LedgerGroup, Double> byGroup = new EnumMap<>(LedgerGroup.class);
    for (int k = 0; k < groupSse.length; k++)
      byGroup.put(PARTS.get(k).group(), yy > 0 ? 1 - groupSse[k] / yy : Double.NaN);
    return new LedgerReport.Line(
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
