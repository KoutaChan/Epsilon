package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import com.epsilon.prism.model.Batches.Batch;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * GPU の煙試験（rocm の組。GPU 1 枚）。標準の形・2048 行で、BF16 の白紙の s ≡ 0、FP32 と BF16 の s の相対 RMS（丸めの約束の目安）、
 * BF16 の学習の 1 歩の時間と有限を確かめる。
 */
public class PrismGpuSmokeTest {
  private static final Logger log = LoggerFactory.getLogger(PrismGpuSmokeTest.class);

  private static final int ROWS = 2048;
  private static final int EVENTS = 35;

  @Test(groups = "rocm")
  public void bf16StepOnGpu() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0));
        NDManager scope = manager.newSubManager()) {
      manager.getEngine().setRandomSeed(3);
      PrismNetwork network = new PrismNetwork(PrismShape.standard(2));
      network.initialize(manager);
      ParameterStore store = new ParameterStore(manager, false);
      List<Batches.Row> rows = new ArrayList<>();
      for (int r = 0; r < ROWS; r++) {
        rows.add(
            switch (r % 8) {
              case 0, 1, 2, 3 -> Batches.turn(EVENTS);
              case 4, 5 -> Batches.response(EVENTS);
              case 6 -> Batches.chankan(EVENTS);
              default -> Batches.postCall(EVENTS);
            });
      }
      Batch batch = Batches.build(rows, 5);
      try (NDManager step = scope.newSubManager()) {
        float[] blank =
            network.forward(store, batch.inputs(step), Pass.inference(Precision.BFLOAT16)).scores().toFloatArray();
        Assert.assertEquals(Batches.maxAbs(blank), 0f);
      }
      Batches.perturb(network, 0.02f, 7);
      try (NDManager step = scope.newSubManager()) {
        float[] full =
            network.forward(store, batch.inputs(step), Pass.inference(Precision.FLOAT32)).scores().toFloatArray();
        float[] half =
            network.forward(store, batch.inputs(step), Pass.inference(Precision.BFLOAT16)).scores().toFloatArray();
        double error = 0;
        double norm = 0;
        for (int i = 0; i < full.length; i++) {
          error += (full[i] - half[i]) * (full[i] - half[i]);
          norm += full[i] * full[i];
        }
        double relative = Math.sqrt(error / Math.max(norm, 1e-30));
        log.info("PRISM-GPU bf16 relative rms={}", String.format("%.4g", relative));
        Assert.assertTrue(relative < 5e-2, "bf16 relative rms " + relative);
      }
      List<ValueLabel> labels = Batches.values(batch, 11);
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 13);
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      Optimizer optimizer = Batches.adam(1e-4f);
      long nanos = 0;
      for (int iteration = 0; iteration < 6; iteration++) {
        long start = System.nanoTime();
        float value;
        try (NDManager step = scope.newSubManager();
            GradientCollector collector = manager.getEngine().newGradientCollector()) {
          Outputs outputs =
              network.forward(
                  store, batch.inputs(step), Pass.training(Precision.BFLOAT16, measured, auxiliary));
          var total = loss.total(step, outputs, labels, auxiliary, 1f, labels.size(), 1f);
          collector.backward(total);
          value = total.getFloat();
        }
        Batches.step(network, optimizer);
        Assert.assertTrue(Float.isFinite(value));
        if (iteration > 0) {
          nanos += System.nanoTime() - start;
        }
      }
      log.info("PRISM-GPU bf16 train rows={} ms/step={}", ROWS, String.format("%.1f", nanos / 1e6 / 5));
    }
  }
}
