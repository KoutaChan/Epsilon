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
 * 学習の RMSNorm の式（{@link RmsNormKernels}。参照の実装 {@link ReferenceAttention} は FP64）が、eager の道（{@link Context#norm} と同じ
 * cast → rms_norm（FP32）→ cast と自動微分）と同じ値と勾配になること（CPU）。利得の勾配は塊の部分和を足した値で比べる。行の数は利得の
 * 塊の境目を含み、行ごとに大きさを変え、全てが 0 の行を含む。比べ方と入力は {@link HipNormGpuTest} と共にする。
 */
public class ReferenceNormTest {
  private static final Logger log = LoggerFactory.getLogger(ReferenceNormTest.class);

  static final float EPSILON = (float) PrismShape.standard().epsilon();

  private static final int[] ROWS = {
    1, RmsNormKernels.GAIN_ROWS, RmsNormKernels.GAIN_ROWS + 1, 300
  };
  private static final int[] WIDTHS = {32, 256};

  /** 入力（x・dy は配列の型で表せる値）。 */
  record Inputs(NormShape shape, float[] input, float[] gain, float[] gradOut) {}

  /** 出力（ホストの FP32）。gradGain は塊の部分和を足した [width]。 */
  record Outputs(float[] out, float[] gradInput, float[] gradGain) {}

  /** 比べた差: BF16 は出力と入力の勾配のビットの一致の割合と最大の ulp、全ての型で相対の L2 の差。 */
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
   * 両方の道の差の門。BF16: 出力は 99.9% 以上がビットで同じで差は 1 ulp まで、入力の勾配（打ち消しのある式）は 99% 以上が同じで相対の
   * L2 の差 1e-3 まで、利得の勾配は FP32 の和の順の幅 1e-5 まで。FP32: 全てを相対の差で見る。
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
   * 乱数の入力。行 i の x は N(0, 1) に行の大きさ 10^u（u は −2〜1 の一様）を掛け、行 0 は全てが 0（r = 1/√ε）。dy は N(0, 1)、利得は
   * 1 + 0.3·N(0, 1)。x と dy は型 type に丸めた値。
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

  /** eager の道: x を FP32 に変換 → rms_norm（FP32 の利得）→ 型に変換し、Σ y·dy から自動微分で勾配を取る。 */
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

  /** 演算の道: 順伝播と逆伝播を演算で行い、利得の勾配の部分和を塊の順に足す（{@link AttentionTape} と同じ）。 */
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

  /** ビットで同じ要素の割合（+0 と −0 は同じとみなす）。 */
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

  /** BF16 の値どうしの最大の ulp の差（BF16 の 16 bit を符号つきの順序の整数にして比べる）。 */
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
