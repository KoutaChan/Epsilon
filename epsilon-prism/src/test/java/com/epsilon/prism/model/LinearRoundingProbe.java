package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 診断: eager の BF16 の Linear（{@link Context#linear}）が偏りを足して 1 回丸めるか。入力の階（[行, S, H] と [N, H]）ごとに、FP32 の
 * 積和を 1 回丸めた値（約束）と、BF16 の積を丸めてから偏りを足して丸めた値（2 回）のどちらと同じかを数えて書く。HIP は約束の 1 回で
 * 丸める。CPU は既定の組、GPU は rocm の組。
 *
 * <p>背骨（[行, S, H]）と候補の鍵・値（[行, 39, H]）の Linear が 3 階の入力に当たる。
 */
public class LinearRoundingProbe {
  private static final Logger log = LoggerFactory.getLogger(LinearRoundingProbe.class);

  @Test
  public void cpu() {
    probe(Device.cpu());
  }

  @Test(groups = "rocm")
  public void gpu() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    probe(Device.gpu(0));
  }

  private static void probe(Device device) {
    try (NDManager manager = NDManager.newBaseManager(device)) {
      Random random = new Random(5);
      int rows = 24;
      int tokens = 80;
      int hidden = 256;
      int out = 512;
      Parameter weight = parameter(manager, "weight", random, new Shape(out, hidden), 0.06);
      Parameter bias = parameter(manager, "bias", random, new Shape(out), 0.3);
      ParameterStore store = new ParameterStore(manager, false);
      try (NDManager scope = manager.newSubManager()) {
        Context context = new Context(store, scope, Precision.BFLOAT16, false);
        NDArray x3 = scope.create(gaussian(random, rows * tokens * hidden), new Shape(rows, tokens, hidden)).toType(DataType.BFLOAT16, false);
        NDArray x2 = x3.reshape(rows * tokens, hidden);
        NDArray w = context.value(weight);
        NDArray b = context.value(bias);
        NDArray once =
            x2.toType(DataType.FLOAT32, false)
                .matMul(w.toType(DataType.FLOAT32, false).transpose())
                .add(b.toType(DataType.FLOAT32, false))
                .toType(DataType.BFLOAT16, false);
        NDArray twice = x2.matMul(w.transpose()).add(b);
        for (int rank = 2; rank <= 3; rank++) {
          NDArray y = context.linear(rank == 2 ? x2 : x3, weight, bias).reshape(rows * tokens, out);
          log.info(
              "PRISM-LINEAR-ROUNDING {} rank {}: differs from one rounding in {}, from two roundings in {} of {}",
              device,
              rank,
              differ(y, once),
              differ(y, twice),
              rows * tokens * out);
        }
      }
    }
  }

  private static Parameter parameter(NDManager manager, String name, Random random, Shape shape, double sigma) {
    Parameter parameter = Parameter.builder().setName(name).setType(Parameter.Type.WEIGHT).build();
    parameter.setArray(manager.create(gaussian(random, (int) shape.size(), sigma), shape));
    return parameter;
  }

  private static float[] gaussian(Random random, int count) {
    return gaussian(random, count, 1.0);
  }

  private static float[] gaussian(Random random, int count, double sigma) {
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      values[i] = (float) (random.nextGaussian() * sigma);
    }
    return values;
  }

  private static int differ(NDArray a, NDArray b) {
    float[] x = a.toType(DataType.FLOAT32, false).toFloatArray();
    float[] y = b.toType(DataType.FLOAT32, false).toFloatArray();
    int differ = 0;
    for (int i = 0; i < x.length; i++) {
      differ += Float.floatToRawIntBits(x[i]) != Float.floatToRawIntBits(y[i]) ? 1 : 0;
    }
    return differ;
  }
}
