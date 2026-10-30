package com.epsilon.prism.learn;

import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.rows.Row;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * GPU の煙試験（rocm の組）。合成のコホートを標準の形の白紙の網（前線なしと前線あり）で、見える GPU（2 枚まで、2 枚ならデータ並列）・BF16 で
 * 当て、損失が有限で、1 歩の時間を出す。
 */
public class FitGpuSmokeTest {
  private static final Logger log = LoggerFactory.getLogger(FitGpuSmokeTest.class);

  /** 損失を回収する歩（1・21・41）が 3 つになる歩数。 */
  private static final int STEPS = 2 * Learner.METRIC_EVERY + 1;

  @DataProvider
  public Object[][] frontierArms() {
    return new Object[][] {{false}, {true}};
  }

  @Test(groups = "rocm", dataProvider = "frontierArms")
  public void fitsOnGpus(boolean frontier) throws IOException {
    int gpus = Engine.getInstance().getGpuCount();
    if (gpus == 0) {
      throw new SkipException("no GPU");
    }
    Device[] devices =
        gpus >= 2
            ? new Device[] {Device.gpu(0), Device.gpu(1)}
            : new Device[] {Device.gpu(0)};
    Path root = Files.createTempDirectory("prism-learn-gpu");
    try {
      List<Row> rows = SyntheticCohort.rows(500, 4, 1);
      SyntheticCohort.write(root.resolve("1"), 1, rows);
      FitPlan plan =
          new FitPlan(
              new FitSettings(STEPS, 512, 2e-4, 1e-5, 0.5f, 0.01, 0.02),
              PrismShape.standard().withFrontier(frontier),
              AuxiliaryLoss.standard(),
              FitPlan.TOKEN_BUDGET,
              3);
      try (CohortStore store = new CohortStore();
          Model model = Fit.blank(plan.shape(), devices[0], plan.seed())) {
        store.addCohort(root.resolve("1"));
        int[] selected = store.select(1, 1, Selection.LABELS);
        FitResult result = Fit.run(store, selected, model, plan, Teacher.RAW, NetworkDevices.of(devices), 8);
        log.info(
            "PRISM-FIT-GPU frontier={} devices={} rows={} steps={} loss {} -> {} clipped={} ms/step={} stages={}",
            frontier,
            Arrays.toString(devices),
            result.rows(),
            result.steps(),
            String.format("%.4f", result.firstLoss()),
            String.format("%.4f", result.lastLoss()),
            result.clippedSteps(),
            String.format("%.1f", result.seconds() * 1e3 / result.steps()),
            Arrays.toString(result.stageSeconds()));
        assertTrue(Double.isFinite(result.firstLoss()) && Double.isFinite(result.lastLoss()));
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
