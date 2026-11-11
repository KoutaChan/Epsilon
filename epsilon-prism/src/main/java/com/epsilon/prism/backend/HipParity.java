package com.epsilon.prism.backend;

import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat.Role;
import com.epsilon.prism.input.InputFormat;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * HIP と eager で計算した s が一致するかを測る。同じネットワーク・同じバッチで出した 2 つの s（バッチの手の順に並ぶ）を比べ、差の
 * 中央値・p99・最大と、手の種類（{@link Role}）ごとの符号つきの差の平均と、argmax の一致率を出す。一致率は、全ての行と、上位 2 手の差が
 * {@link #CONFIDENT_GAP} 以上の行の 2 つについて出す。
 */
public final class HipParity {

  /** |Δs| の中央値の上限。 */
  public static final double MEDIAN_LIMIT = 3e-4;

  /** |Δs| の p99 の上限。 */
  public static final double P99_LIMIT = 3e-3;

  /** 手の種類ごとに出した符号つきの差の平均について、絶対値の上限。 */
  public static final double MEAN_LIMIT = 1e-4;

  /** 全ての行の argmax の一致率の下限。 */
  public static final double ARGMAX_LIMIT = 0.99;

  /** 上位 2 手の差が大きい行の argmax の一致率の下限。 */
  public static final double CONFIDENT_LIMIT = 0.999;

  /** 上位 2 手の差が大きいとみなすしきい値。参照の s で測る。 */
  public static final double CONFIDENT_GAP = 5e-3;

  /** 手の種類を enum の宣言の順に並べる。 */
  private static final Comparator<Role> BY_ORDINAL = Comparator.comparingInt(Role::ordinal);

  private HipParity() {}

  /**
   * バッチの手の並び。行ごとの手の範囲と、手ごとの種類を持つ。
   *
   * @param rowFirst 行ごとの最初の手の番号
   * @param rowCount 行ごとの手の数
   * @param roles 手の種類（バッチの手の順）
   */
  public record Actions(int[] rowFirst, int[] rowCount, Role[] roles) {

    /**
     * 確定したバッチのインデックスから読む。手の種類は、バッチの入力の形式が持つ一覧から引く。
     *
     * @param batch 確定したバッチ
     * @return 手の並び
     */
    public static Actions of(HostBatch batch) {
      BatchPlan plan = batch.plan();
      ByteBuffer blob = batch.blob().duplicate().order(ByteOrder.LITTLE_ENDIAN);
      int rows = plan.rows();
      int actions = plan.count(Axis.ACTION);
      InputFormat format = plan.format();
      int rowWidth = format.width(Tensor.ROW_COLUMN);
      int actionWidth = format.width(Tensor.ACTION_COLUMN);
      int[] first = new int[rows];
      int[] count = new int[rows];
      for (int r = 0; r < rows; r++) {
        int at = plan.offset(Tensor.ROW_COLUMN) + r * rowWidth * Integer.BYTES;
        first[r] = blob.getInt(at + RowColumn.FIRST_ACTION.ordinal() * Integer.BYTES);
        count[r] = blob.getInt(at + RowColumn.ACTIONS.ordinal() * Integer.BYTES);
      }
      Role[] roles = new Role[actions];
      List<? extends Role> values = format.roles();
      for (int a = 0; a < actions; a++) {
        int at = plan.offset(Tensor.ACTION_COLUMN) + (a * actionWidth + ActionColumn.ROLE.ordinal()) * Integer.BYTES;
        roles[a] = values.get(blob.getInt(at));
      }
      return new Actions(first, count, roles);
    }
  }

  /**
   * 比べた結果。
   *
   * @param actions 手の数
   * @param median |Δs| の中央値
   * @param p99 |Δs| の p99
   * @param max |Δs| の最大
   * @param signedMean 手の種類ごとの Δs（HIP − 参照）の平均
   * @param counts 手の種類ごとの手の数
   * @param rows 行の数
   * @param argmaxAgreement argmax の一致率
   * @param confidentRows 上位 2 手の差が大きい行の数
   * @param confidentAgreement その行の argmax の一致率。行がなければ 1
   */
  public record Report(
      int actions,
      double median,
      double p99,
      double max,
      Map<Role, Double> signedMean,
      Map<Role, Integer> counts,
      int rows,
      double argmaxAgreement,
      int confidentRows,
      double confidentAgreement) {

    /** このクラスの上限と下限を全て満たせば true を返す。 */
    public boolean passes() {
      boolean means = true;
      for (double mean : signedMean.values()) {
        means &= Math.abs(mean) <= MEAN_LIMIT;
      }
      return median <= MEDIAN_LIMIT
          && p99 <= P99_LIMIT
          && means
          && argmaxAgreement >= ARGMAX_LIMIT
          && confidentAgreement >= CONFIDENT_LIMIT;
    }

    /** 1 行の要約を返す。 */
    public String summary() {
      StringBuilder means = new StringBuilder();
      for (Map.Entry<Role, Double> entry : signedMean.entrySet()) {
        means.append(String.format(Locale.ROOT, " %s=%+.2e(n=%d)", entry.getKey(), entry.getValue(), counts.get(entry.getKey())));
      }
      return String.format(
          Locale.ROOT,
          "actions=%d median=%.3e p99=%.3e max=%.3e argmax=%.5f (rows %d) confident=%.5f (rows %d) mean:%s",
          actions,
          median,
          p99,
          max,
          argmaxAgreement,
          rows,
          confidentAgreement,
          confidentRows,
          means);
    }
  }

  /**
   * 2 つの s を比べる。
   *
   * @param hip HIP の s（バッチの手の順）
   * @param reference 参照の s（同じ並び）
   * @param actions 手の並び
   * @return 比べた結果
   */
  public static Report compare(float[] hip, float[] reference, Actions actions) {
    int n = actions.roles().length;
    if (hip.length < n || reference.length < n) {
      throw new IllegalArgumentException("score arrays are shorter than the " + n + " actions");
    }
    double[] errors = new double[n];
    Map<Role, Double> sums = new TreeMap<>(BY_ORDINAL);
    Map<Role, Integer> counts = new TreeMap<>(BY_ORDINAL);
    double max = 0;
    for (int a = 0; a < n; a++) {
      double delta = (double) hip[a] - reference[a];
      errors[a] = Math.abs(delta);
      max = Math.max(max, errors[a]);
      sums.merge(actions.roles()[a], delta, Double::sum);
      counts.merge(actions.roles()[a], 1, Integer::sum);
    }
    Arrays.sort(errors);
    Map<Role, Double> means = new TreeMap<>(BY_ORDINAL);
    for (Map.Entry<Role, Double> entry : sums.entrySet()) {
      means.put(entry.getKey(), entry.getValue() / counts.get(entry.getKey()));
    }
    int rows = actions.rowFirst().length;
    int agree = 0;
    int confident = 0;
    int confidentAgree = 0;
    for (int r = 0; r < rows; r++) {
      int first = actions.rowFirst()[r];
      int count = actions.rowCount()[r];
      int expected = argmax(reference, first, count);
      boolean same = argmax(hip, first, count) == expected;
      agree += same ? 1 : 0;
      if (gap(reference, first, count) >= CONFIDENT_GAP) {
        confident++;
        confidentAgree += same ? 1 : 0;
      }
    }
    return new Report(
        n,
        quantile(errors, 0.5),
        quantile(errors, 0.99),
        max,
        Collections.unmodifiableMap(means),
        Collections.unmodifiableMap(counts),
        rows,
        rows == 0 ? 1 : (double) agree / rows,
        confident,
        confident == 0 ? 1 : (double) confidentAgree / confident);
  }

  /**
   * 2 つの s がビット単位で同じなら true を返す。−0 と +0 も区別する。
   *
   * @param a 比べる s
   * @param b もう一方の s
   * @return ビット単位で同じなら true
   */
  public static boolean bitwiseEqual(float[] a, float[] b) {
    if (a.length != b.length) {
      return false;
    }
    for (int i = 0; i < a.length; i++) {
      if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
        return false;
      }
    }
    return true;
  }

  /** 行の argmax を返す。同点なら先の手。 */
  private static int argmax(float[] scores, int first, int count) {
    int best = first;
    for (int a = first + 1; a < first + count; a++) {
      if (scores[a] > scores[best]) {
        best = a;
      }
    }
    return best;
  }

  /** 行の上位 2 手の差を返す。手が 1 つなら無限大。 */
  private static double gap(float[] scores, int first, int count) {
    double top = Double.NEGATIVE_INFINITY;
    double second = Double.NEGATIVE_INFINITY;
    for (int a = first; a < first + count; a++) {
      double value = scores[a];
      if (value > top) {
        second = top;
        top = value;
      } else if (value > second) {
        second = value;
      }
    }
    return top - second;
  }

  private static double quantile(double[] sorted, double q) {
    if (sorted.length == 0) {
      return 0;
    }
    int index = (int) Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1);
    return sorted[Math.max(index, 0)];
  }
}
