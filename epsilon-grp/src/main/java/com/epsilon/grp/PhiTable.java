package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;

/**
 * GruSinkhorn の出力にだけ重ねて使う表（Phi が使う）。席ごとの順位の周辺確率に次局の温度を掛け、Φ には終わった局の補正を足す。
 * どのメソッドもメモリを確保しない。温度の表も補正の表も、空なら何もしない。
 *
 * <p>温度: 次局の温度を T とし（表より後の局には最後の値を使う）、席ごとに次のように変える。
 * p'_r ∝ max(p_r, 1e-9)^(1/T) = exp(log(max(p_r, 1e-9)) / T)。T = 1 なら変えない。行の和は 1 のままだが列の和は保たないので、
 * T ≠ 1 の局では Σ_i Φ_i = Σ_r u_r も成り立たない。
 *
 * <p>補正: 補正の表は終わった局ごとに 1 行ある（表より後の局には最後の行を使う）。下端から上端までを等分し、行の値を各区間の
 * 中心に置く。間は線形補間し、範囲の外は端の値を使う。Φ'_i = Φ_i ＋ c(Φ_i) − Σ_j c(Φ_j) / S なので、席の和は変わらない。
 */
public final class PhiTable {

  /** 温度を掛ける前の確率の下限。 */
  private static final double FLOOR = 1e-9;

  private static final String VARIANT = "variant";
  private static final String TEMPERATURES = "temperatures";
  private static final String LOW = "low";
  private static final String HIGH = "high";
  private static final String CORRECTIONS = "corrections";

  private final Variant variant;

  /** 次局ごとの温度。 */
  private final double[] temperatures;

  private final double low;
  private final double high;

  /** 終わった局と区間ごとの補正。添字は [終わった局][区間]。 */
  private final double[][] corrections;

  private PhiTable(Variant variant, double[] temperatures, double low, double high, double[][] corrections) {
    this.variant = variant;
    this.temperatures = temperatures;
    this.low = low;
    this.high = high;
    this.corrections = corrections;
  }

  /**
   * 渡した配列をコピーして表を作る。温度は正の有限の値であること。補正の表が空でなければ、下端は上端より小さく、どの行も同じ数
   * （1 以上）の区間を持ち、値は有限であること。満たさなければ IllegalArgumentException を投げる。
   *
   * @param low 補正の区間の下端
   * @param high 補正の区間の上端
   */
  public static PhiTable of(Variant variant, double[] temperatures, double low, double high, double[][] corrections) {
    Objects.requireNonNull(variant, "variant");
    for (double temperature : temperatures) {
      if (!(temperature > 0) || !Double.isFinite(temperature)) {
        throw new IllegalArgumentException("Temperatures must be positive and finite: " + temperature);
      }
    }
    double[][] copy = new double[corrections.length][];
    for (int stage = 0; stage < corrections.length; stage++) {
      if (corrections[stage].length == 0 || corrections[stage].length != corrections[0].length) {
        throw new IllegalArgumentException("Every stage needs the same positive number of bins");
      }
      copy[stage] = corrections[stage].clone();
      for (double correction : copy[stage]) {
        if (!Double.isFinite(correction)) {
          throw new IllegalArgumentException("Corrections must be finite: " + correction);
        }
      }
    }
    if (copy.length > 0 && !(low < high && Double.isFinite(low) && Double.isFinite(high))) {
      throw new IllegalArgumentException("The bins need a finite low < high: " + low + ", " + high);
    }
    return new PhiTable(variant, temperatures.clone(), low, high, copy);
  }

  /** JSON のファイルから読む。読めないときや形が合わないときは IOException を投げる。 */
  public static PhiTable read(Path file) throws IOException {
    try {
      JsonObject root = JsonParser.parseString(Files.readString(file, StandardCharsets.UTF_8)).getAsJsonObject();
      JsonArray stages = root.getAsJsonArray(CORRECTIONS);
      double[][] corrections = new double[stages.size()][];
      for (int stage = 0; stage < corrections.length; stage++) {
        corrections[stage] = doubles(stages.get(stage).getAsJsonArray());
      }
      return of(
          Variant.valueOf(root.get(VARIANT).getAsString()),
          doubles(root.getAsJsonArray(TEMPERATURES)),
          root.get(LOW).getAsDouble(),
          root.get(HIGH).getAsDouble(),
          corrections);
    } catch (RuntimeException malformed) {
      throw new IOException("Malformed Phi table " + file + ": " + malformed.getMessage(), malformed);
    }
  }

  public void write(Path file) throws IOException {
    JsonObject root = new JsonObject();
    root.addProperty(VARIANT, variant.name());
    root.add(TEMPERATURES, array(temperatures));
    root.addProperty(LOW, low);
    root.addProperty(HIGH, high);
    JsonArray stages = new JsonArray();
    for (double[] stage : corrections) {
      stages.add(array(stage));
    }
    root.add(CORRECTIONS, stages);
    DurableFiles.replace(
        file, (new GsonBuilder().setPrettyPrinting().create().toJson(root) + "\n").getBytes(StandardCharsets.UTF_8));
  }

  /** ファイルの中身の SHA-256 を小文字の 16 進で返す。Φ の項目で、表が同じものかを照合するのに使う。 */
  public static String sha256(Path file) throws IOException {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));
    } catch (NoSuchAlgorithmException missing) {
      throw new IllegalStateException(missing);
    }
  }

  public Variant variant() {
    return variant;
  }

  /**
   * 席ごとの順位の周辺確率に、次局の温度を掛ける。
   *
   * @param nextRound 次局の番号
   * @param marginals 周辺確率。並びは [席 × S ＋ 順位] で、その場で書き換える
   */
  public void temper(int nextRound, double[] marginals) {
    int seats = variant.seats();
    requireRound(nextRound);
    if (marginals.length != seats * seats) {
      throw new IllegalArgumentException(variant + " needs " + seats * seats + " marginals, got " + marginals.length);
    }
    if (temperatures.length == 0) {
      return;
    }
    double temperature = temperatures[Math.min(nextRound, temperatures.length - 1)];
    if (temperature == 1) {
      return;
    }
    double exponent = 1 / temperature;
    for (int row = 0; row < marginals.length; row += seats) {
      double total = 0;
      for (int rank = 0; rank < seats; rank++) {
        // StrictMath.pow は呼ぶたびに作業用の配列を確保するので、exp(log(p)·指数) で求める。
        marginals[row + rank] = StrictMath.exp(StrictMath.log(Math.max(marginals[row + rank], FLOOR)) * exponent);
        total += marginals[row + rank];
      }
      for (int rank = 0; rank < seats; rank++) {
        marginals[row + rank] /= total;
      }
    }
  }

  /**
   * 絶対席ごとの Φ に、終わった局の補正を足す。
   *
   * @param endedRound 終わった局の番号
   * @param phi 絶対席ごとの Φ。その場で書き換える
   */
  public void calibrate(int endedRound, double[] phi) {
    int seats = variant.seats();
    requireRound(endedRound);
    if (phi.length != seats) {
      throw new IllegalArgumentException(variant + " needs " + seats + " values, got " + phi.length);
    }
    if (corrections.length == 0) {
      return;
    }
    double[] cells = corrections[Math.min(endedRound, corrections.length - 1)];
    // c(Φ_i) は Φ_i だけで決まるので、席ごとに求めたらすぐ Φ_i ＋ c(Φ_i) に書き換え、平均は後で引く。
    double total = 0;
    for (int seat = 0; seat < seats; seat++) {
      double correction = correction(cells, phi[seat]);
      phi[seat] += correction;
      total += correction;
    }
    double mean = total / seats;
    for (int seat = 0; seat < seats; seat++) {
      phi[seat] -= mean;
    }
  }

  private double correction(double[] cells, double value) {
    int bins = cells.length;
    double position = (value - low) / (high - low) * bins - 0.5;
    if (position <= 0) {
      return cells[0];
    }
    if (position >= bins - 1) {
      return cells[bins - 1];
    }
    int lower = (int) position;
    return cells[lower] + (position - lower) * (cells[lower + 1] - cells[lower]);
  }

  private void requireRound(int round) {
    if (round < 0 || round > variant.finalRound()) {
      throw new IllegalArgumentException("Round " + round + " is outside " + variant);
    }
  }

  private static double[] doubles(JsonArray array) {
    double[] values = new double[array.size()];
    for (int index = 0; index < values.length; index++) {
      values[index] = array.get(index).getAsDouble();
    }
    return values;
  }

  private static JsonArray array(double[] values) {
    JsonArray array = new JsonArray();
    for (double value : values) {
      array.add(value);
    }
    return array;
  }
}
