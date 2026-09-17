package com.epsilon.ai.decision;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.util.Objects;

/**
 * スカラーの効用値を、正規分布で平滑化した101区間の教師分布に変換するHL-Gaussの共通処理。
 *
 * <p>各区間の確率は順位確率ではなく、効用値の教師分布を表す。推論時は区間の中心値を確率で加重平均し、スカラーの効用値へ戻す。
 * 効用値の範囲の両端には標準偏差の6倍（4.5区間分）の余白を設け、分布の切り詰めによる平均の偏りを抑える。
 */
public final class EpsilonDecisionHlGauss {
  public static final int BIN_COUNT = 101;
  public static final float SIGMA_TO_BIN_RATIO = 0.75f;
  public static final double SUPPORT_PADDING_SIGMAS = 6.0;

  /** 事前分布の対数を取る前に適用する確率の下限。教師分布には適用しない。 */
  public static final float PRIOR_PROBABILITY_FLOOR = 1.0e-8f;

  /** 推論サーバー・学習ワーカーが所有するデバイス上の定数を実行時パラメーターで共有するためのキー。 */
  public static final String DEVICE_CONSTANTS = "hlGaussDeviceConstants";

  private static final double SQRT_TWO = Math.sqrt(2.0);
  private static final ProfileConstants[] PROFILE_CONSTANTS = createProfileConstants();

  private record ProfileConstants(Support support, float[] edges, float[] centers) {}

  private static ProfileConstants[] createProfileConstants() {
    EpsilonUtilityProfile[] profiles = EpsilonUtilityProfile.values();
    ProfileConstants[] constants = new ProfileConstants[profiles.length];
    for (EpsilonUtilityProfile profile : profiles) {
      if (!profile.rankBased()) {
        continue;
      }
      Support support = createSupport(profile);
      float[] edges = new float[BIN_COUNT + 1];
      float[] centers = new float[BIN_COUNT];
      for (int edge = 0; edge <= BIN_COUNT; edge++) {
        edges[edge] = (float) (support.minimum() + edge * support.binWidth());
        if (edge < BIN_COUNT) {
          centers[edge] = (float) (support.minimum() + (edge + 0.5) * support.binWidth());
        }
      }
      constants[profile.ordinal()] = new ProfileConstants(support, edges, centers);
    }
    return constants;
  }

  private EpsilonDecisionHlGauss() {}

  /** ヒストグラムの区間境界と正規分布の標準偏差。最小値と最大値は、余白を含む値域の両端を表す。 */
  public record Support(
      double utilityMinimum,
      double utilityMaximum,
      double minimum,
      double maximum,
      double binWidth,
      double sigma) {
    public double center(int bin) {
      if (bin < 0 || bin >= BIN_COUNT) {
        throw new IllegalArgumentException("HL-Gauss bin must be 0-100: " + bin);
      }
      return minimum + (bin + 0.5) * binWidth;
    }
  }

  /** 順位効用値の定義に応じた、等間隔の区間境界と正規分布の幅を返す。 */
  public static Support support(EpsilonUtilityProfile profile) {
    Objects.requireNonNull(profile, "profile");
    if (!profile.rankBased()) {
      throw new IllegalArgumentException("HL-Gauss requires a bounded rank utility: " + profile);
    }
    return PROFILE_CONSTANTS[profile.ordinal()].support();
  }

  private static Support createSupport(EpsilonUtilityProfile profile) {
    double minimum = Double.POSITIVE_INFINITY;
    double maximum = Double.NEGATIVE_INFINITY;
    for (int rank = 0; rank < com.epsilon.core.GameState.NUM_PLAYERS; rank++) {
      minimum = Math.min(minimum, profile.utilityForRank(rank));
      maximum = Math.max(maximum, profile.utilityForRank(rank));
    }
    double width =
        (maximum - minimum) / (BIN_COUNT - 2.0 * SUPPORT_PADDING_SIGMAS * SIGMA_TO_BIN_RATIO);
    double sigma = SIGMA_TO_BIN_RATIO * width;
    double padding = SUPPORT_PADDING_SIGMAS * sigma;
    return new Support(minimum, maximum, minimum - padding, maximum + padding, width, sigma);
  }

  /** チェックポイントが効用値とヒストグラムの定義を固定するための識別子。 */
  public static String fingerprint(EpsilonUtilityProfile profile) {
    Support support = support(profile);
    return "hl-gauss-v1-bins="
        + BIN_COUNT
        + ";sigma-bin="
        + SIGMA_TO_BIN_RATIO
        + ";padding-sigmas="
        + SUPPORT_PADDING_SIGMAS
        + ";profile="
        + profile.name()
        + ";utility="
        + Float.toHexString(profile.utilityForRank(0))
        + ","
        + Float.toHexString(profile.utilityForRank(1))
        + ","
        + Float.toHexString(profile.utilityForRank(2))
        + ","
        + Float.toHexString(profile.utilityForRank(3))
        + ";minimum="
        + Double.toHexString(support.minimum())
        + ";maximum="
        + Double.toHexString(support.maximum())
        + ";prior-floor="
        + PRIOR_PROBABILITY_FLOOR;
  }

  /** スカラーの教師値を各区間の正規分布確率へ変換し、有限値域内で正規化する。 */
  public static float[] targetProbabilities(float target, EpsilonUtilityProfile profile) {
    Support support = support(profile);
    if (!Float.isFinite(target)
        || target < (float) support.minimum()
        || target > (float) support.maximum()) {
      throw new IllegalArgumentException("HL-Gauss target outside finite support: " + target);
    }
    float[] result = new float[BIN_COUNT];
    double previous = normalCdf((support.minimum() - target) / support.sigma());
    double sum = 0.0;
    for (int bin = 0; bin < BIN_COUNT; bin++) {
      double upper = support.minimum() + (bin + 1) * support.binWidth();
      double cumulative = normalCdf((upper - target) / support.sigma());
      result[bin] = (float) Math.max(0.0, cumulative - previous);
      sum += result[bin];
      previous = cumulative;
    }
    for (int bin = 0; bin < BIN_COUNT; bin++) {
      result[bin] /= (float) sum;
    }
    return result;
  }

  /** 検証済みスカラー教師 [バッチ] を、CPU 転送なしで FLOAT32 教師ラベル [バッチ,101] へ変換する。 */
  public static NDArray targetProbabilities(NDArray scalarTargets, EpsilonUtilityProfile profile) {
    if (scalarTargets.getShape().dimension() != 1) {
      throw new IllegalArgumentException("HL-Gauss scalar targets must have shape [batch]");
    }
    Support support = support(profile);
    float[] edges = PROFILE_CONSTANTS[profile.ordinal()].edges();
    NDArray targets = scalarTargets.toType(DataType.FLOAT32, false).stopGradient().reshape(-1, 1);
    NDArray cumulative =
        scalarTargets
            .getManager()
            .create(edges)
            .reshape(1, BIN_COUNT + 1)
            .sub(targets)
            .div((float) (SQRT_TWO * support.sigma()))
            .erf();
    NDArray labels = cumulative.get(":,1:").sub(cumulative.get(":,0:" + BIN_COUNT)).maximum(0.0f);
    return labels.div(labels.sum(new int[] {1}, true)).stopGradient();
  }

  /** 効用値を区間ごとに表すロジット [バッチ,101] を同じデバイス上でスカラー [バッチ] へ復号する。 */
  public static NDArray decode(NDArray logits, EpsilonUtilityProfile profile) {
    if (logits.getShape().dimension() != 2 || logits.getShape().get(1) != BIN_COUNT) {
      throw new IllegalArgumentException("HL-Gauss logits must have shape [batch,101]");
    }
    support(profile);
    float[] centers = PROFILE_CONSTANTS[profile.ordinal()].centers();
    // この DJL の toType は非微分演算なので、型昇格によって復号の勾配を保つ。
    NDArray fullPrecision =
        logits.getDataType() == DataType.FLOAT32
            ? logits
            : logits.add(logits.getManager().zeros(new Shape(1), DataType.FLOAT32));
    return fullPrecision.softmax(1).mul(logits.getManager().create(centers)).sum(new int[] {1});
  }

  /**
   * 一つの実行デバイスで再利用するFP32定数。学習パラメーターには含めず、サーバー・学習ワーカー終了時に解放する。
   *
   * <p>中間テンソルをバッチメモリ管理オブジェクトへ所有させ、長寿命の定数メモリ管理オブジェクトへ残さない。
   */
  public static final class DeviceConstants implements AutoCloseable {
    private final NDManager manager;
    private final Support support;
    private final NDArray utilityWeights;
    private final NDArray edges;
    private final NDArray centers;

    public DeviceConstants(NDManager owner, EpsilonUtilityProfile profile) {
      support = EpsilonDecisionHlGauss.support(profile);
      ProfileConstants constants = PROFILE_CONSTANTS[profile.ordinal()];
      manager = owner.newSubManager();
      try {
        utilityWeights = manager.create(profile.rankUtility());
        edges = manager.create(constants.edges(), new Shape(1, BIN_COUNT + 1));
        centers = manager.create(constants.centers());
      } catch (RuntimeException | Error failure) {
        manager.close();
        throw failure;
      }
    }

    /** 選択したプレイヤーのGRP順位確率から、従来と同じFP32期待効用値を作る。 */
    public NDArray priorUtility(NDArray rankProbabilities) {
      return rankProbabilities.mul(utilityWeights).sum(new int[] {1});
    }

    /** 固定した区間境界のH2D転送をせず、バッチメモリ管理オブジェクト内で正規分布に基づく事前分布を作る。 */
    public NDArray targetProbabilities(NDArray scalarTargets) {
      if (scalarTargets.getShape().dimension() != 1) {
        throw new IllegalArgumentException("HL-Gauss scalar targets must have shape [batch]");
      }
      NDArray targets = scalarTargets.toType(DataType.FLOAT32, false).stopGradient().reshape(-1, 1);
      NDArray centered = edges.sub(targets);
      // 定数は移動せず、減算で作った一時テンソルだけをバッチの寿命へ戻す。
      centered.attach(scalarTargets.getManager());
      NDArray cumulative = centered.div((float) (SQRT_TWO * support.sigma())).erf();
      NDArray labels = cumulative.get(":,1:").sub(cumulative.get(":,0:" + BIN_COUNT)).maximum(0.0f);
      return labels.div(labels.sum(new int[] {1}, true)).stopGradient();
    }

    /** 固定した区間の中心値のH2D転送をせず、FP32 の確率を使って効用値の加重平均へ復号する。 */
    public NDArray decode(NDArray logits) {
      NDArray fullPrecision =
          logits.getDataType() == DataType.FLOAT32
              ? logits
              : logits.add(logits.getManager().zeros(new Shape(1), DataType.FLOAT32));
      return fullPrecision.softmax(1).mul(centers).sum(new int[] {1});
    }

    @Override
    public void close() {
      manager.close();
    }
  }

  /** 診断用のホストロジットの指定行をスカラー効用値へ復号する。 */
  public static float decodeLogits(float[] logits, int offset, EpsilonUtilityProfile profile) {
    double maximum = maximumLogit(logits, offset);
    Support support = support(profile);
    double sum = 0.0;
    double weighted = 0.0;
    for (int bin = 0; bin < BIN_COUNT; bin++) {
      double probability = Math.exp(logits[offset + bin] - maximum);
      sum += probability;
      weighted += probability * (support.minimum() + (bin + 0.5) * support.binWidth());
    }
    return (float) (weighted / sum);
  }

  /** スカラーの教師値から正規分布に基づく教師分布を作り、ホスト上のロジットとの交差エントロピーを数値的に安定した方法で計算する。 */
  public static double crossEntropyFromLogits(
      float[] logits, int offset, float target, EpsilonUtilityProfile profile) {
    double maximum = maximumLogit(logits, offset);
    float[] labels = targetProbabilities(target, profile);
    double sum = 0.0;
    double expectedLogit = 0.0;
    double labelSum = 0.0;
    for (int bin = 0; bin < BIN_COUNT; bin++) {
      double shifted = logits[offset + bin] - maximum;
      sum += Math.exp(shifted);
      expectedLogit += labels[bin] * shifted;
      labelSum += labels[bin];
    }
    return Math.log(sum) * labelSum - expectedLogit;
  }

  private static double maximumLogit(float[] logits, int offset) {
    Objects.requireNonNull(logits, "logits");
    Objects.checkFromIndexSize(offset, BIN_COUNT, logits.length);
    double maximum = Double.NEGATIVE_INFINITY;
    for (int bin = 0; bin < BIN_COUNT; bin++) {
      float value = logits[offset + bin];
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("HL-Gauss logits must be finite");
      }
      maximum = Math.max(maximum, value);
    }
    return maximum;
  }

  /** 標準正規 CDF の有理近似。絶対誤差は 7.5e-8 以下。 */
  private static double normalCdf(double value) {
    double x = Math.abs(value);
    double t = 1.0 / (1.0 + 0.2316419 * x);
    double tail =
        Math.exp(-0.5 * x * x)
            / Math.sqrt(2.0 * Math.PI)
            * t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return value < 0.0 ? tail : 1.0 - tail;
  }
}
