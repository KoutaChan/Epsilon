package com.epsilon.prism.learn;

import static org.testng.Assert.assertEquals;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism4p.input.InputSchema;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 系譜と同じ順で 1 つの JVM の中で学習を続けても、学習のデバイスで PyTorch が確保したメモリが増えないことを確かめる（rocm の
 * グループ）。GPU が 2 枚なら、本番と同じく gpu1 を主にしたデータ並列で学ぶ。学習のランタイムは使い回し、毎回、前の学習が保存した
 * ネットワークからウォームスタートして学習し、閉じる。行列積の作業領域は最初の学習で確保されるので、2 回目からの確保量を 1 回目の後と比べる。
 */
public class FitGpuMemoryTest {
  private static final Logger log = LoggerFactory.getLogger(FitGpuMemoryTest.class);

  private static final int FITS = 3;
  private static final int STEPS = 3;

  @Test(groups = "rocm")
  public void allocatedMemoryStaysFlatAcrossFits() throws IOException {
    int gpus = Engine.getInstance().getGpuCount();
    if (gpus == 0) {
      throw new SkipException("no GPU");
    }
    NetworkDevices devices =
        gpus >= 2 ? NetworkDevices.of(Device.gpu(1), Device.gpu(0)) : NetworkDevices.of(Device.gpu(0));
    PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
    Path root = Files.createTempDirectory("prism-fit-memory");
    try {
      SyntheticCohort.write(root.resolve("1"), 1, SyntheticCohort.rows(500, 4, 1));
      FitPlan plan =
          new FitPlan(
              new FitSettings(STEPS, 512, 2e-4, 1e-5, 0.5f, 0.01, 0.02),
              PrismShape.standard(),
              AuxiliaryLoss.standard(),
              FitPlan.TOKEN_BUDGET,
              3);
      Path start = root.resolve("network-0");
      try (Model blank = Fit.blank(InputSchema.FORMAT, plan.shape(), Device.cpu(), plan.seed())) {
        ((PrismNetwork) blank.getBlock()).save(start);
      }
      long[] first = null;
      try (CohortStore store = new CohortStore();
          NetworkRuntime learning = NetworkRuntime.forLearning()) {
        store.addCohort(root.resolve("1"));
        int[] rows = store.select(1, 1, Selection.LABELS);
        for (int fit = 1; fit <= FITS; fit++) {
          try (Fit.Prepared prepared =
              Fit.prepare(store, rows, start, plan, Teacher.RAW.gain(), devices, 8, learning)) {
            prepared.fit(Teacher.RAW);
            start = root.resolve("network-" + fit);
            prepared.network().save(start);
          }
          long[] allocated = new long[devices.size()];
          for (int lane = 0; lane < allocated.length; lane++) {
            allocated[lane] = engine.getMemoryStats(devices.get(lane)).getAllocatedBytes();
          }
          log.info("PRISM-FIT-MEMORY fit={} devices={} allocated={}", fit, devices.devices(), Arrays.toString(allocated));
          if (first == null) {
            first = allocated;
          } else {
            assertEquals(allocated, first, "bytes allocated on " + devices.devices() + " after fit " + fit);
          }
        }
      }
    } finally {
      try (var files = Files.walk(root)) {
        for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
          Files.delete(file);
        }
      }
    }
  }
}
