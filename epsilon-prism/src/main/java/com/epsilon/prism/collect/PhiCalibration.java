package com.epsilon.prism.collect;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Φ（固定の GRP の期待効用）の効用の空間の較正。GRP は今の順位が続くことに自信過剰で、局の途中で 4 位の席の Φ は実際の最終効用より低く、
 * 点差が大きいほどずれる。終わった局の段ごとに、Φ の値の等幅の区分の平均の残差 c = E[u − Φ] を区分の中心の間で線形に補い（端は端の値）、
 * Φ' = Φ + c(Φ) − （4 席の c の平均）にする（4 席の和を変えない）。
 *
 * <p>表は audit-grp の CSV から外の解析（grp_calib_util.py）で作る JSON: {"stages": [...], "lo": 下端, "hi": 上端, "bins": B,
 * "table": {"段": [B 個の値]}}。段は終わった局の局番号（東1 = 0 … 南4 = 7、それ以降は最後の段）。
 */
public final class PhiCalibration {
  private final double lo, hi;
  private final double[][] table;

  PhiCalibration(double lo, double hi, double[][] table) {
    this.lo = lo;
    this.hi = hi;
    this.table = table;
  }

  /** JSON の表を読む。 */
  public static PhiCalibration load(Path path) throws IOException {
    JsonObject root = JsonParser.parseString(Files.readString(path, StandardCharsets.UTF_8)).getAsJsonObject();
    int stages = root.getAsJsonArray("stages").size();
    int bins = root.get("bins").getAsInt();
    JsonObject cells = root.getAsJsonObject("table");
    double[][] table = new double[stages][bins];
    for (int stage = 0; stage < stages; stage++) {
      var values = cells.getAsJsonArray(Integer.toString(stage));
      if (values == null || values.size() != bins)
        throw new IllegalArgumentException("Φ calibration lacks stage " + stage);
      for (int bin = 0; bin < bins; bin++) table[stage][bin] = values.get(bin).getAsDouble();
    }
    return new PhiCalibration(root.get("lo").getAsDouble(), root.get("hi").getAsDouble(), table);
  }

  /**
   * 較正した 4 席の Φ。段は公開系列の最後から 2 つ目の局境界（終わった局の root）の局番号。
   *
   * @param utilities 4 席の Φ（GRP の期待効用）
   * @param prefix GRP の公開系列（{@link EpsilonGrpFeature#FEATURE_SIZE} ずつの局境界。最後が次局）
   */
  public double[] apply(double[] utilities, float[] prefix) {
    int blocks = prefix.length / EpsilonGrpFeature.FEATURE_SIZE;
    int ended = Math.max(0, blocks - 2) * EpsilonGrpFeature.FEATURE_SIZE;
    int kyoku = Math.round(prefix[ended + EpsilonGrpFeature.KYOKU_INDEX] * EpsilonGrpFeature.KYOKU_SCALE);
    double[] cells = table[Math.max(0, Math.min(kyoku, table.length - 1))];
    double[] corrections = new double[4];
    double mean = 0;
    for (int seat = 0; seat < 4; seat++) {
      corrections[seat] = correction(cells, utilities[seat]);
      mean += corrections[seat] / 4;
    }
    double[] result = new double[4];
    for (int seat = 0; seat < 4; seat++) result[seat] = utilities[seat] + corrections[seat] - mean;
    return result;
  }

  /** 区分の中心の値を線形に補う（区分の外側は端の値）。 */
  private double correction(double[] cells, double value) {
    int bins = cells.length;
    double position = (value - lo) / (hi - lo) * bins - 0.5;
    if (position <= 0) return cells[0];
    if (position >= bins - 1) return cells[bins - 1];
    int low = (int) Math.floor(position);
    double t = position - low;
    return cells[low] + t * (cells[low + 1] - cells[low]);
  }
}
