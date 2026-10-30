package com.epsilon.prism.hip;

import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * K3 の速さの門: 本番の形（H=256・F=512、1 塊 2048 行 × 平均 74 トークン）で、実効 15 TFLOPS 以上。
 *
 * <p>実効 = 2·トークン·(H·H/2 + 2·H·F) / カーネルの時間（印で測る。転送は含めない）。他の仕事のない GPU 1 枚で回す。
 */
@Test(groups = "rocm")
public class MlpBenchmarkTest {

  private static final Logger log = LoggerFactory.getLogger(MlpBenchmarkTest.class);

  private static final int ROWS = 2048;
  private static final int TOKENS_PER_ROW = 74;
  private static final int FFN = 512;
  private static final double GATE_TFLOPS = 15.0;
  private static final int WARMUP = 10;
  private static final int ITERATIONS = 50;

  public void reachesTheGate() {
    int tokens = ROWS * TOKENS_PER_ROW;
    MlpCase c = MlpCase.random(7, FFN, tokens, tokens);
    HipShape shape = c.shape;
    double flops = 2.0 * tokens * ((double) shape.hidden() * shape.attention() + 2.0 * shape.hidden() * shape.ffn());
    float[] millis = new float[ITERATIONS];
    HipLibrary library = HipLibrary.load();
    try (PtStream stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
        HipPlan plan = HipPlan.create(library, stream, shape)) {
      plan.uploadWeights(c.pack(), 1);
      plan.copyIn(HipRegion.RESIDUAL, c.input(HipRegion.RESIDUAL), c.bytes(HipRegion.RESIDUAL));
      plan.copyIn(HipRegion.ATTENTION, c.input(HipRegion.ATTENTION), c.bytes(HipRegion.ATTENTION));
      plan.profile(true);
      for (int i = 0; i < WARMUP + ITERATIONS; i++) {
        plan.run(HipKernel.MLP0, 0, c.batch());
        float ms = plan.kernelMillis()[HipKernel.MLP0.ordinal()];
        if (i >= WARMUP) {
          millis[i - WARMUP] = ms;
        }
      }
    }
    Arrays.sort(millis);
    double median = millis[ITERATIONS / 2];
    double tflops = flops / (median * 1e-3) / 1e12;
    log.info(
        "K3 {} tokens: median {} ms (min {}, max {}) = {} TFLOPS effective; gate {} TFLOPS: {}",
        tokens,
        String.format("%.3f", median),
        String.format("%.3f", millis[0]),
        String.format("%.3f", millis[ITERATIONS - 1]),
        String.format("%.1f", tflops),
        String.format("%.0f", GATE_TFLOPS),
        tflops >= GATE_TFLOPS ? "PASS" : "FAIL");
    assertTrue(tflops >= GATE_TFLOPS, "K3 effective " + tflops + " TFLOPS is below the gate " + GATE_TFLOPS);
  }
}
