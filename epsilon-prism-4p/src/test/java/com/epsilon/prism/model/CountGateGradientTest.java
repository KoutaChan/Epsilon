package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtEngine;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 同じコードを読む牌と遷移のゲート勾配を、順序を固定した FP32 の部分和と照合する。 */
public class CountGateGradientTest {
  @Test
  public void heldOutInputsDistinguishProductRoundingAndRowOrder() {
    for (int[] c : GateGradientFixture.cases()) {
      var fixture = new GateGradientFixture(c[0], c[1], c[2], c[3]);
      var slope = fixture.hostSlope();
      var expected = fixture.reduce(slope, false, false);
      Assert.assertFalse(java.util.Arrays.equals(expected, fixture.reduce(slope, true, false)));
      Assert.assertFalse(java.util.Arrays.equals(expected, fixture.reduce(slope, false, true)));
    }
  }

  @Test(groups = "rocm")
  public void gateContributionsRoundBeforeTheRowSum() throws Exception {
    try (var manager = NDManager.newBaseManager(Device.gpu(0), "PyTorch");
        var stream = ((PtEngine) manager.getEngine()).newStream(manager.getDevice());
        var scope = stream.openScope()) {
      var kernels = HipAttention.on(HipLibrary.load(), stream);
      for (int[] c : GateGradientFixture.cases()) {
        var fixture = new GateGradientFixture(c[0], c[1], c[2], c[3]);
        fixture.assertMatches(manager, kernels);
      }
    }
  }

  @Test(groups = "rocm")
  public void repeatedCodesAccumulateInFixedOrder() {
    int batch = 11, tiles = 34, transitions = 97, codes = 100, width = 16;
    var random = new Random(421);
    float[] projected = values(random, batch * tiles * width);
    float[] upstream = values(random, transitions * width);
    int[] rows = new int[transitions], ids = new int[transitions * tiles];
    for (int t = 0; t < transitions; t++) rows[t] = t * (batch - 1) / transitions;
    for (int i = 0; i < ids.length; i++) ids[i] = i % 7 == 0 ? codes - 1 : i % 3;
    float[] expected = new float[codes * width];
    for (int row = 0; row < batch; row++) {
      float[] partial = new float[expected.length];
      for (int start = 0; start < tiles; start += 8) {
        for (int t = 0; t < transitions; t++) {
          if (rows[t] != row) continue;
          for (int key = start; key < Math.min(start + 8, tiles); key++) {
            for (int c = 0; c < width; c++) {
              int at = ids[t * tiles + key] * width + c;
              partial[at] +=
                  (upstream[t * width + c] * projected[(row * tiles + key) * width + c]) * 0.25f;
            }
          }
        }
      }
      for (int at = 0; at < expected.length; at++) expected[at] += partial[at];
    }
    try (var manager = NDManager.newBaseManager(Device.gpu(0), "PyTorch");
        var stream = ((PtEngine) manager.getEngine()).newStream(manager.getDevice());
        var scope = stream.openScope()) {
      var kernels = HipAttention.on(HipLibrary.load(), stream);
      var x =
          manager
              .create(projected, new Shape(batch, tiles, width))
              .toType(DataType.BFLOAT16, false);
      var grad =
          manager.create(upstream, new Shape(transitions, width)).toType(DataType.BFLOAT16, false);
      var logits = manager.zeros(new Shape(codes, width));
      var owners = manager.create(rows);
      var codeIds = manager.create(ids, new Shape(transitions, tiles));
      var dx = manager.create(x.getShape(), DataType.BFLOAT16);
      var dg = manager.ones(logits.getShape());
      for (int repeat = 0; repeat < 3; repeat++) {
        kernels.countGradient(x, logits, owners, codeIds, grad, dx, dg);
        Assert.assertEquals(dg.toFloatArray(), expected);
      }
    }
  }

  private static float[] values(Random random, int n) {
    float[] values = new float[n];
    for (int i = 0; i < n; i++)
      values[i] = Math.scalb((float) (random.nextInt(255) - 127), random.nextInt(24) - 15);
    return values;
  }
}
