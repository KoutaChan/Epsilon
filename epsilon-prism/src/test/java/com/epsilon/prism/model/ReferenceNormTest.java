package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.prism.hip.RmsNormKernels;
import com.epsilon.prism.hip.RmsNormKernels.NormShape;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 学習で使う RMSNorm の式（{@link RmsNormKernels}）を CPU でテストする。FP64 で計算する参照実装 {@link ReferenceAttention} が、eager の
 * パス（{@link Context#norm} と同じ cast → rms_norm（FP32）→ cast と自動微分）と同じ値と勾配になることを確かめる。ゲインの勾配は、
 * 行のタイルごとの部分和を足した値で比べる。行の数には、タイルの境界をまたぐものも含める。行ごとに値の大きさを変え、全ての値が 0 の行も
 * 入れる。比べ方と入力は {@link HipNormGpuTest} と共通にする。
 */
public class ReferenceNormTest {
  private static final Logger log = LoggerFactory.getLogger(ReferenceNormTest.class);

  static final float EPSILON = (float) PrismShape.standard().epsilon();

  private static final int[] ROWS = {
    1, RmsNormKernels.GAIN_ROWS, RmsNormKernels.GAIN_ROWS + 1, 300
  };
  private static final int[] WIDTHS = {32, 256};

  /** 入力。x と dy は、比べる配列の型で表せる値にしてある。 */
  record Inputs(NormShape shape, float[] input, float[] gain, float[] gradOut) {}

  /** ホストに読み出した FP32 の出力。{@code gradGain} は、行のタイルごとの部分和を足した [width]。 */
  record Outputs(float[] out, float[] gradInput, float[] gradGain) {}

  /** 2 つのパスの差。BF16 では、出力と入力の勾配について、値が一致する割合と ulp の差の最大を見る。相対 L2 の差はどの型でも見る。 */
  record Difference(
      double outEqual,
      int outUlps,
      double outRelative,
      double inputEqual,
      int inputUlps,
      double inputRelative,
      double gainRelative) {

    @Override
    public String toString() {
      return String.format(
          "out equal=%.6f ulps=%d rel=%.3g input equal=%.6f ulps=%d rel=%.3g gain rel=%.3g",
          outEqual,
          outUlps,
          outRelative,
          inputEqual,
          inputUlps,
          inputRelative,
          gainRelative);
    }
  }

  @Test
  public void referenceMatchesTheEagerPathInBfloat16() {
    check(DataType.BFLOAT16);
  }

  @Test
  public void referenceMatchesTheEagerPathInFloat32() {
    check(DataType.FLOAT32);
  }

  private static void check(DataType type) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      for (int rows : ROWS) {
        for (int width : WIDTHS) {
          Inputs inputs = generate(new NormShape(rows, width, EPSILON), type, rows * 31L + width);
          Outputs expected = eager(manager, inputs, type);
          Difference difference =
              compare(expected, kernels(manager, new ReferenceAttention(), inputs, type));
          log.info("PRISM-NORM-REFERENCE {} rows={} width={} {}", type, rows, width, difference);
          assertWithin(difference, type, "rows " + rows + " width " + width);
        }
      }
    }
  }

  /**
   * 2 つのパスの差がしきい値に収まることを確かめる。BF16 では、出力は 99.9% 以上の要素が同じ値で、差は 1 ulp まで。入力の勾配は
   * 桁落ちのある式なので、99% 以上が同じ値で、相対 L2 の差が 1e-3 まで。FP32 では、出力（1e-6 まで）と入力の勾配（1e-5 まで）を
   * 相対の差で見る。ゲインの勾配はどちらの型でも、FP32 の和の順の違いで生じる相対の差 1e-5 まで。
   */
  static void assertWithin(Difference d, DataType type, String label) {
    if (type == DataType.BFLOAT16) {
      Assert.assertTrue(d.outEqual() >= 0.999, label + " output " + d);
      Assert.assertTrue(d.outUlps() <= 1, label + " output " + d);
      Assert.assertTrue(d.inputEqual() >= 0.99, label + " input gradient " + d);
      Assert.assertTrue(d.inputRelative() <= 1e-3, label + " input gradient " + d);
    } else {
      Assert.assertTrue(d.outRelative() <= 1e-6, label + " output " + d);
      Assert.assertTrue(d.inputRelative() <= 1e-5, label + " input gradient " + d);
    }
    Assert.assertTrue(d.gainRelative() <= 1e-5, label + " gain gradient " + d);
  }

  /**
   * 入力を乱数で作る。行 i の x は、N(0, 1) に行ごとの大きさ 10^u（u は −2〜1 の一様分布）を掛けた値。行 0 は全てが 0 で、r = 1/√ε に
   * なる。dy は N(0, 1)、ゲインは 1 + 0.3·N(0, 1)。x と dy は型 {@code type} に丸める。
   */
  static Inputs generate(NormShape shape, DataType type, long seed) {
    Random random = new Random(seed);
    int rows = shape.rows();
    int width = shape.width();
    float[] x = new float[rows * width];
    float[] dy = new float[rows * width];
    for (int i = 0; i < rows; i++) {
      double scale = i == 0 ? 0 : Math.pow(10, -2 + 3 * random.nextDouble());
      for (int j = 0; j < width; j++) {
        x[i * width + j] = round((float) (scale * random.nextGaussian()), type);
        dy[i * width + j] = round((float) random.nextGaussian(), type);
      }
    }
    float[] gain = new float[width];
    for (int j = 0; j < width; j++) {
      gain[j] = (float) (1 + 0.3 * random.nextGaussian());
    }
    return new Inputs(shape, x, gain, dy);
  }

  /** eager のパスで計算する。x を FP32 に変換し、rms_norm（ゲインは FP32）を通して元の型に戻す。勾配は Σ y·dy から自動微分で求める。 */
  static Outputs eager(NDManager manager, Inputs in, DataType type) {
    try (NDManager scope = manager.newSubManager();
        GradientCollector collector = scope.getEngine().newGradientCollector()) {
      Shape shape = new Shape(in.shape().rows(), in.shape().width());
      NDArray x = scope.create(in.input(), shape).toType(type, false);
      NDArray gain = scope.create(in.gain());
      x.setRequiresGradient(true);
      gain.setRequiresGradient(true);
      NDArray values = cast(x, DataType.FLOAT32);
      long[] axes = {in.shape().width()};
      NDArray y =
          cast(values.getNDArrayInternal().rmsNorm(axes, gain, in.shape().epsilon()), type);
      NDArray dy = scope.create(in.gradOut(), shape).toType(type, false);
      collector.backward(y.mul(dy).sum());
      return new Outputs(floats(y), floats(x.getGradient()), floats(gain.getGradient()));
    }
  }

  /** 演算のパスで計算する。順伝播と逆伝播を {@code kernels} で行い、ゲインの勾配の部分和をタイルの順に足す（{@link AttentionTape} と同じ）。 */
  static Outputs kernels(NDManager manager, RmsNormKernels kernels, Inputs in, DataType type) {
    try (NDManager scope = manager.newSubManager()) {
      NormShape d = in.shape();
      Shape shape = new Shape(d.rows(), d.width());
      NDArray x = scope.create(in.input(), shape).toType(type, false);
      NDArray gain = scope.create(in.gain());
      NDArray dy = scope.create(in.gradOut(), shape).toType(type, false);
      NDArray y = scope.create(shape, type);
      NDArray inverseRms = scope.create(new Shape(d.rows()), DataType.FLOAT32);
      NDArray dx = scope.create(shape, type);
      NDArray partials = scope.create(new Shape(d.partials(), d.width()), DataType.FLOAT32);
      kernels.forward(d, x, gain, y, inverseRms);
      kernels.backward(d, x, gain, inverseRms, dy, dx, partials);
      return new Outputs(floats(y), floats(dx), floats(partials.sum(new int[] {0})));
    }
  }

  static Difference compare(Outputs expected, Outputs actual) {
    return new Difference(
        equal(expected.out(), actual.out()),
        ulps(expected.out(), actual.out()),
        relative(expected.out(), actual.out()),
        equal(expected.gradInput(), actual.gradInput()),
        ulps(expected.gradInput(), actual.gradInput()),
        relative(expected.gradInput(), actual.gradInput()),
        relative(expected.gradGain(), actual.gradGain()));
  }

  /** 値が等しい要素の割合。== で比べるので、+0 と −0 は等しいとみなす。 */
  static double equal(float[] expected, float[] actual) {
    int same = 0;
    for (int i = 0; i < expected.length; i++) {
      // == は +0 と -0 を同じとみなす（NaN は入力に無い）
      if (expected[i] == actual[i]) {
        same++;
      }
    }
    return (double) same / expected.length;
  }

  /** BF16 の値どうしの ulp の差の最大。BF16 の 16 bit を、値の大小と同じ順に並ぶ符号付きの整数に直して比べる。 */
  static int ulps(float[] expected, float[] actual) {
    int worst = 0;
    for (int i = 0; i < expected.length; i++) {
      worst = Math.max(worst, Math.abs(ordered(expected[i]) - ordered(actual[i])));
    }
    return worst;
  }

  static double relative(float[] expected, float[] actual) {
    double difference = 0;
    double norm = 0;
    for (int i = 0; i < expected.length; i++) {
      double d = (double) expected[i] - actual[i];
      difference += d * d;
      norm += (double) expected[i] * expected[i];
    }
    return Math.sqrt(difference / Math.max(norm, 1e-300));
  }

  private static int ordered(float value) {
    int bits = Float.floatToRawIntBits(value) >>> 16;
    return (bits & 0x8000) != 0 ? -(bits & 0x7fff) : bits;
  }

  private static float round(float value, DataType type) {
    if (type == DataType.FLOAT32) {
      return value;
    }
    int bits = Float.floatToRawIntBits(value);
    return Float.intBitsToFloat(((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) << 16);
  }

  private static NDArray cast(NDArray array, DataType type) {
    return array.getDataType() == type
        ? array
        : array.getNDArrayInternal().differentiableCast(type);
  }

  static float[] floats(NDArray array) {
    return array.toType(DataType.FLOAT32, false).toFloatArray();
  }
}
