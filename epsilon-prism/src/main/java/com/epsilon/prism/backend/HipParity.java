package com.epsilon.prism.backend;

import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.input.InputSchema.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;

/**
 * HIP と eager の s の一致。同じ網・同じ塊の 2 つの s（塊の手の順）を比べ、差の中央値・p99・最大、手の役割ごとの符号つきの平均、
 * argmax の一致率（全ての行と、上位 2 手の差が {@link #CONFIDENT_GAP} 以上の行）を出す。
 */
public final class HipParity {

  /** |Δs| の中央値の上限。 */
  public static final double MEDIAN_LIMIT = 3e-4;

  /** |Δs| の p99 の上限。 */
  public static final double P99_LIMIT = 3e-3;

  /** 役割ごとの符号つきの平均の絶対値の上限。 */
  public static final double MEAN_LIMIT = 1e-4;

  /** 全ての行の argmax の一致率の下限。 */
  public static final double ARGMAX_LIMIT = 0.99;

  /** 上位 2 手の差が大きい行の argmax の一致率の下限。 */
  public static final double CONFIDENT_LIMIT = 0.999;

  /** 「上位 2 手の差が大きい」の閾値（参照の s で測る）。 */
  public static final double CONFIDENT_GAP = 5e-3;

  private HipParity() {}

  /**
   * 塊の手の形。
   *
   * @param rowFirst 行の最初の手
   * @param rowCount 行の手の数
   * @param roles 手の役割（塊の手の順）
   */
  public record Actions(int[] rowFirst, int[] rowCount, ActionRole[] roles) {

    /**
     * 封じた塊の索引から読む。
     *
     * @param batch 封じた塊
     * @return 手の形
     */
    public static Actions of(HostBatch batch) {
      BatchPlan plan = batch.plan();
      ByteBuffer blob = batch.blob().duplicate().order(ByteOrder.LITTLE_ENDIAN);
      int rows = plan.rows();
      int actions = plan.count(Axis.ACTION);
      int rowWidth = Tensor.ROW_COLUMN.width();
      int actionWidth = Tensor.ACTION_COLUMN.width();
      int[] first = new int[rows];
      int[] count = new int[rows];
      for (int r = 0; r < rows; r++) {
        int at = plan.offset(Tensor.ROW_COLUMN) + r * rowWidth * Integer.BYTES;
        first[r] = blob.getInt(at + RowColumn.FIRST_ACTION.ordinal() * Integer.BYTES);
        count[r] = blob.getInt(at + RowColumn.ACTIONS.ordinal() * Integer.BYTES);
      }
      ActionRole[] roles = new ActionRole[actions];
      ActionRole[] values = ActionRole.values();
      for (int a = 0; a < actions; a++) {
        int at = plan.offset(Tensor.ACTION_COLUMN) + (a * actionWidth + ActionColumn.ROLE.ordinal()) * Integer.BYTES;
        roles[a] = values[blob.getInt(at)];
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
   * @param signedMean 役割ごとの Δs の平均（HIP − 参照）
   * @param counts 役割ごとの手の数
   * @param rows 行の数
   * @param argmaxAgreement argmax の一致率
   * @param confidentRows 上位 2 手の差が大きい行の数
   * @param confidentAgreement その行の argmax の一致率（行がなければ 1）
   */
  public record Report(
      int actions,
      double median,
      double p99,
      double max,
      Map<ActionRole, Double> signedMean,
      Map<ActionRole, Integer> counts,
      int rows,
      double argmaxAgreement,
      int confidentRows,
      double confidentAgreement) {

    /** HipParity の上限と下限を全て満たすか。 */
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

    /** 1 行の要約。 */
    public String summary() {
      StringBuilder means = new StringBuilder();
      for (Map.Entry<ActionRole, Double> entry : signedMean.entrySet()) {
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
   * @param hip HIP の s（塊の手の順）
   * @param reference 参照の s（同じ並び）
   * @param actions 手の形
   * @return 結果
   */
  public static Report compare(float[] hip, float[] reference, Actions actions) {
    int n = actions.roles().length;
    if (hip.length < n || reference.length < n) {
      throw new IllegalArgumentException("score arrays are shorter than the " + n + " actions");
    }
    double[] errors = new double[n];
    Map<ActionRole, Double> sums = new EnumMap<>(ActionRole.class);
    Map<ActionRole, Integer> counts = new EnumMap<>(ActionRole.class);
    double max = 0;
    for (int a = 0; a < n; a++) {
      double delta = (double) hip[a] - reference[a];
      errors[a] = Math.abs(delta);
      max = Math.max(max, errors[a]);
      sums.merge(actions.roles()[a], delta, Double::sum);
      counts.merge(actions.roles()[a], 1, Integer::sum);
    }
    Arrays.sort(errors);
    Map<ActionRole, Double> means = new EnumMap<>(ActionRole.class);
    for (Map.Entry<ActionRole, Double> entry : sums.entrySet()) {
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
   * 2 つの s がビットで同じか（−0 と +0 も区別する）。
   *
   * @param a s
   * @param b s
   * @return ビットで同じなら true
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

  /** 行の argmax（同点は先の手）。 */
  private static int argmax(float[] scores, int first, int count) {
    int best = first;
    for (int a = first + 1; a < first + count; a++) {
      if (scores[a] > scores[best]) {
        best = a;
      }
    }
    return best;
  }

  /** 行の上位 2 手の差（手が 1 つなら無限大）。 */
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
