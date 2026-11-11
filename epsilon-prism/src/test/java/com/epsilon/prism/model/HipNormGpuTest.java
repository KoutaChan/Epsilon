package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.training.GradientCollector;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.RmsNormKernels.NormShape;
import com.epsilon.prism.model.ReferenceNormTest.Difference;
import com.epsilon.prism.model.ReferenceNormTest.Inputs;
import com.epsilon.prism.model.ReferenceNormTest.Outputs;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 学習で使う RMSNorm の HIP の演算（{@link HipAttention}、{@code t2_norm.hip}）をテストする（rocm のグループ、GPU 1 枚）。BF16 で
 * 表せる乱数の入力（{@link ReferenceNormTest#generate}）で、次を確かめる。
 *
 * <ul>
 *   <li>FP64 の参照（{@link ReferenceAttention}）と、GPU の eager のパス（cast → rms_norm（FP32）→ cast と自動微分）の両方との差が、
 *       {@link ReferenceNormTest#assertWithin} のしきい値に収まること。eager のパスと FP64 の参照の差も同じ行に出し、HIP の差が
 *       その程度に収まっているかの目安にする。
 *   <li>同じ入力で順伝播・逆伝播を 2 回行い、全ての出力（y・r・dx・ゲインの勾配の部分和）がビットまで同じこと。
 *   <li>先頭の行だけのバッチでも、その行の y・r・dx が元のバッチとビットまで同じこと（行の値はバッチの組み方に依らない）。
 * </ul>
 *
 * 幅は、レーンのタイル 1〜4 つ分の 96・256・320・1024。行の数には、ゲインの勾配を部分和で足すタイルの境界をまたぐものも含める。
 * {@link #measureSpeed} は、バックボーンの正規化 1 回分の大きさ（18 万行 × 256）で、eager のパスと HIP の演算の順伝播・逆伝播に
 * かかる時間を ms で出す。
 */
public class HipNormGpuTest {
  private static final Logger log = LoggerFactory.getLogger(HipNormGpuTest.class);

  private static final int[] ROWS = {1, 7, 128, 129, 1000, 5000};
  private static final int[] WIDTHS = {96, 256, 320, 1024};
  private static final int SUBSET_ROWS = 61;
  private static final int SPEED_ROWS = 180_000;
  private static final int SPEED_WIDTH = 256;
  private static final int WARMUP = 3;
  private static final int LAUNCHES = 20;

  /** HIP の演算を 1 回実行したときの全ての出力（ホストに読み出した FP32）。 */
  private record Run(float[] out, float[] inverseRms, float[] gradInput, float[] partials) {}

  @FunctionalInterface
  private interface Body {
    void accept(NDManager manager, PtStream stream, HipAttention kernels);
  }

  @Test(groups = "rocm")
  public void kernelsMatchTheReferenceAndTheEagerPath() {
    withKernels(
        (manager, stream, kernels) -> {
          for (int rows : ROWS) {
            for (int width : WIDTHS) {
              NormShape shape = new NormShape(rows, width, ReferenceNormTest.EPSILON);
              Inputs inputs = ReferenceNormTest.generate(shape, DataType.BFLOAT16, rows * 31L + width);
              Outputs hip = ReferenceNormTest.kernels(manager, kernels, inputs, DataType.BFLOAT16);
              Outputs fp64;
              try (NDManager host = NDManager.newBaseManager(Device.cpu())) {
                fp64 =
                    ReferenceNormTest.kernels(
                        host, new ReferenceAttention(), inputs, DataType.BFLOAT16);
              }
              Outputs eager = ReferenceNormTest.eager(manager, inputs, DataType.BFLOAT16);
              Difference toReference = ReferenceNormTest.compare(fp64, hip);
              Difference toEager = ReferenceNormTest.compare(eager, hip);
              Difference eagerToReference = ReferenceNormTest.compare(fp64, eager);
              log.info(
                  "PRISM-HIPNORM rows={} width={} hip-fp64 {} | hip-eager {} | eager-fp64 {}",
                  rows,
                  width,
                  toReference,
                  toEager,
                  eagerToReference);
              String label = "rows " + rows + " width " + width;
              ReferenceNormTest.assertWithin(toReference, DataType.BFLOAT16, label + " to FP64");
              ReferenceNormTest.assertWithin(toEager, DataType.BFLOAT16, label + " to eager");
            }
          }
        });
  }

  @Test(groups = "rocm")
  public void kernelsAreDeterministicAndRowIndependent() {
    withKernels(
        (manager, stream, kernels) -> {
          for (int width : WIDTHS) {
            int rows = 1000;
            NormShape shape = new NormShape(rows, width, ReferenceNormTest.EPSILON);
            Inputs inputs = ReferenceNormTest.generate(shape, DataType.BFLOAT16, 97L + width);
            Run first = run(manager, kernels, inputs);
            Run second = run(manager, kernels, inputs);
            Inputs subset = prefix(inputs, SUBSET_ROWS);
            Run part = run(manager, kernels, subset);
            int elements = SUBSET_ROWS * width;
            boolean deterministic =
                same(first.out(), second.out())
                    && same(first.inverseRms(), second.inverseRms())
                    && same(first.gradInput(), second.gradInput())
                    && same(first.partials(), second.partials());
            boolean independent =
                same(Arrays.copyOf(first.out(), elements), part.out())
                    && same(Arrays.copyOf(first.inverseRms(), SUBSET_ROWS), part.inverseRms())
                    && same(Arrays.copyOf(first.gradInput(), elements), part.gradInput());
            log.info(
                "PRISM-HIPNORM-BITS width={} deterministic={} rowIndependent={}",
                width,
                deterministic,
                independent);
            Assert.assertTrue(deterministic, "width " + width + " differs between runs");
            Assert.assertTrue(independent, "width " + width + " rows depend on the batch");
          }
        });
  }

  /** バックボーンの正規化 1 回分の大きさで、eager のパスと HIP の演算それぞれの順伝播・逆伝播 1 回にかかる時間を ms で出す。 */
  @Test(groups = "rocm")
  public void measureSpeed() {
    withKernels(
        (manager, stream, kernels) -> {
          NormShape d = new NormShape(SPEED_ROWS, SPEED_WIDTH, ReferenceNormTest.EPSILON);
          Inputs inputs = ReferenceNormTest.generate(d, DataType.BFLOAT16, 5);
          Shape shape = new Shape(d.rows(), d.width());
          PtEvent event = stream.newEvent();
          try (NDManager scope = manager.newSubManager()) {
            NDArray x = scope.create(inputs.input(), shape).toType(DataType.BFLOAT16, false);
            NDArray gain = scope.create(inputs.gain());
            NDArray dy = scope.create(inputs.gradOut(), shape).toType(DataType.BFLOAT16, false);
            NDArray y = scope.create(shape, DataType.BFLOAT16);
            NDArray inverseRms = scope.create(new Shape(d.rows()), DataType.FLOAT32);
            NDArray dx = scope.create(shape, DataType.BFLOAT16);
            NDArray partials = scope.create(new Shape(d.partials(), d.width()), DataType.FLOAT32);
            double hipForward = millis(event, () -> kernels.forward(d, x, gain, y, inverseRms));
            double hipBackward =
                millis(
                    event,
                    () -> {
                      kernels.backward(d, x, gain, inverseRms, dy, dx, partials);
                      partials.sum(new int[] {0}).close();
                    });
            double eagerForward = millis(event, () -> eagerForward(scope, x, gain, d).close());
            double eagerTotal = millis(event, () -> eagerStep(scope, x, gain, dy, d));
            log.info(
                "PRISM-HIPNORM-SPEED rows={} width={} eager forward={} ms forward+backward={} ms"
                    + " | hip forward={} ms backward={} ms forward+backward={} ms",
                d.rows(),
                d.width(),
                String.format("%.3f", eagerForward),
                String.format("%.3f", eagerTotal),
                String.format("%.3f", hipForward),
                String.format("%.3f", hipBackward),
                String.format("%.3f", hipForward + hipBackward));
          }
        });
  }

  private static NDArray eagerForward(NDManager scope, NDArray x, NDArray gain, NormShape d) {
    NDArray values = x.toType(DataType.FLOAT32, false);
    NDArray y = values.getNDArrayInternal().rmsNorm(new long[] {d.width()}, gain, d.epsilon());
    NDArray result = y.toType(DataType.BFLOAT16, false);
    values.close();
    y.close();
    return result;
  }

  /** eager のパスで順伝播と逆伝播（入力とゲインの勾配まで）を行う。測る時間には、損失 Σ y·dy を作る積と和と、その逆伝播も入る。 */
  private static void eagerStep(NDManager scope, NDArray x, NDArray gain, NDArray dy, NormShape d) {
    try (NDManager step = scope.newSubManager();
        GradientCollector collector = step.getEngine().newGradientCollector()) {
      NDArray leaf = x.stopGradient();
      NDArray weight = gain.stopGradient();
      leaf.attach(step);
      weight.attach(step);
      leaf.setRequiresGradient(true);
      weight.setRequiresGradient(true);
      NDArray values = leaf.getNDArrayInternal().differentiableCast(DataType.FLOAT32);
      NDArray y =
          values
              .getNDArrayInternal()
              .rmsNorm(new long[] {d.width()}, weight, d.epsilon())
              .getNDArrayInternal()
              .differentiableCast(DataType.BFLOAT16);
      collector.backward(y.mul(dy).sum());
    }
  }

  private static Run run(NDManager manager, HipAttention kernels, Inputs in) {
    try (NDManager scope = manager.newSubManager()) {
      NormShape d = in.shape();
      Shape shape = new Shape(d.rows(), d.width());
      NDArray x = scope.create(in.input(), shape).toType(DataType.BFLOAT16, false);
      NDArray gain = scope.create(in.gain());
      NDArray dy = scope.create(in.gradOut(), shape).toType(DataType.BFLOAT16, false);
      NDArray y = scope.create(shape, DataType.BFLOAT16);
      NDArray inverseRms = scope.create(new Shape(d.rows()), DataType.FLOAT32);
      NDArray dx = scope.create(shape, DataType.BFLOAT16);
      NDArray partials = scope.create(new Shape(d.partials(), d.width()), DataType.FLOAT32);
      kernels.forward(d, x, gain, y, inverseRms);
      kernels.backward(d, x, gain, inverseRms, dy, dx, partials);
      return new Run(
          ReferenceNormTest.floats(y),
          inverseRms.toFloatArray(),
          ReferenceNormTest.floats(dx),
          partials.toFloatArray());
    }
  }

  private static Inputs prefix(Inputs in, int rows) {
    int elements = rows * in.shape().width();
    return new Inputs(
        new NormShape(rows, in.shape().width(), in.shape().epsilon()),
        Arrays.copyOf(in.input(), elements),
        in.gain(),
        Arrays.copyOf(in.gradOut(), elements));
  }

  private static boolean same(float[] expected, float[] actual) {
    if (expected.length != actual.length) {
      return false;
    }
    for (int i = 0; i < expected.length; i++) {
      if (Float.floatToRawIntBits(expected[i]) != Float.floatToRawIntBits(actual[i])) {
        return false;
      }
    }
    return true;
  }

  /** 演算をウォームアップしてから {@value #LAUNCHES} 回積み、ストリームが終わるまでの時間を 1 回あたりの ms で返す。 */
  private static double millis(PtEvent event, Runnable launch) {
    for (int i = 0; i < WARMUP; i++) {
      launch.run();
    }
    event.record();
    event.synchronize();
    long started = System.nanoTime();
    for (int i = 0; i < LAUNCHES; i++) {
      launch.run();
    }
    event.record();
    event.synchronize();
    return (System.nanoTime() - started) / 1e6 / LAUNCHES;
  }

  private static void withKernels(Body body) {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    Device gpu = Device.gpu(0);
    try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
        PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
        PtStreamScope ignored = stream.openScope()) {
      body.accept(manager, stream, HipAttention.on(HipLibrary.load(), stream));
    }
  }
}
