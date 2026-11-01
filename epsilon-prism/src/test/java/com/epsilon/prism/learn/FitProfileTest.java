package com.epsilon.prism.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.learn.FitProfile.Timing;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.rows.CohortWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 当てはめの計測（{@link FitProfile}、CPU）: 設定の形と歩の計り方の順（温めの W 歩の後、K 歩に 1 歩を同期）。計測しても CPU の当てはめの
 * 損失と重みはビットで同じで（印は時計を読むだけ。CPU のレーンには自前の注意の記録がないので木も同じ）、表に歩の窓・微小バッチの形・
 * 学習器と網の段が出る。
 */
public class FitProfileTest {
  private static final Logger log = LoggerFactory.getLogger(FitProfileTest.class);

  private static final PrismShape SHAPE = new PrismShape(64, 128, 64, 2, false);
  private static final int STEPS = 6;
  private static final int BUDGET = 6_000;

  private Path root;

  @BeforeClass
  public void writeCohort() throws IOException {
    root = Files.createTempDirectory("prism-profile");
    SyntheticCohort.write(CohortWriter.directory(root, 1), 1, SyntheticCohort.rows(500, 1, 6));
  }

  @AfterClass(alwaysRun = true)
  public void deleteCohort() throws IOException {
    if (root == null) {
      return;
    }
    try (var files = Files.walk(root)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(file);
      }
    }
  }

  @Test
  public void settingChoosesTheMeasuredSteps() {
    FitProfile off = FitProfile.of(null);
    assertFalse(off.enabled());
    for (int step = 1; step <= 8; step++) {
      assertEquals(off.timing(step), Timing.OFF);
    }
    FitProfile alternating = FitProfile.of("sync:2:3");
    assertTrue(alternating.enabled());
    List<Timing> expected =
        List.of(
            Timing.OFF, Timing.OFF, Timing.OFF, Timing.SYNCED, Timing.HOST, Timing.SYNCED, Timing.HOST);
    for (int step = 1; step <= expected.size(); step++) {
      assertEquals(alternating.timing(step), expected.get(step - 1), "step " + step);
    }
    FitProfile every = FitProfile.of(" sync:1:0 ");
    assertEquals(every.timing(1), Timing.SYNCED);
    assertEquals(every.timing(2), Timing.SYNCED);
    for (String invalid : List.of("sync:0:1", "async:2:1", "sync:2", "sync:-1:2", "sync:2:x")) {
      assertThrows(IllegalArgumentException.class, () -> FitProfile.of(invalid));
    }
  }

  @Test(timeOut = 600_000)
  public void profileKeepsTheFitAndReportsTheStages() throws IOException {
    FitPlan plan =
        new FitPlan(
            new FitSettings(STEPS, 4096, 3e-3, 3e-4, 1f, 0.01, 0.02),
            SHAPE,
            AuxiliaryLoss.standard(),
            BUDGET,
            11);
    try (CohortStore store = new CohortStore()) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] rows = store.select(1, 1, Selection.LABELS);
      FitProfile profile = FitProfile.of("sync:2:2");
      List<FitResult> results = new ArrayList<>();
      List<Map<String, float[]>> weights = new ArrayList<>();
      for (FitProfile used : List.of(FitProfile.of(null), profile)) {
        try (Model model = Fit.blank(SHAPE, Device.cpu(), plan.seed())) {
          results.add(
              Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(Device.cpu()), 2, used));
          weights.add(parameters(model));
        }
      }
      List<String> report = profile.report();
      report.forEach(log::info);
      FitResult plain = results.get(0);
      FitResult profiled = results.get(1);
      assertEquals(Double.doubleToRawLongBits(profiled.loss()), Double.doubleToRawLongBits(plain.loss()));
      assertEquals(
          Double.doubleToRawLongBits(profiled.lastLoss()), Double.doubleToRawLongBits(plain.lastLoss()));
      for (Map.Entry<String, float[]> entry : weights.get(0).entrySet()) {
        assertTrue(
            Arrays.equals(entry.getValue(), weights.get(1).get(entry.getKey())),
            "weights differ: " + entry.getKey());
      }
      assertTrue(report.get(0).contains("synced steps 2, host-only steps 2"), report.get(0));
      String table = String.join("\n", report);
      for (String row :
          List.of(
              "compute",
              "optimizer",
              "PIECES",
              "PADDED_PAIRS",
              "TRANSFER",
              "BATCH_INDEX",
              "EMBED",
              "RELATION",
              "NORM_QKV L0",
              "ATTENTION L1",
              "O_FFN L1",
              "FINAL_NORM",
              "CANDIDATE",
              "VALUE_GRAPH",
              "AUX_HEADS",
              "LOSS",
              "BACKWARD",
              "RELEASE",
              "BUSY")) {
        assertTrue(table.contains("fit profile " + row + " "), "no row " + row);
      }
    }
  }

  private static Map<String, float[]> parameters(Model model) {
    Map<String, float[]> values = new LinkedHashMap<>();
    for (Pair<String, Parameter> pair : model.getBlock().getParameters()) {
      values.put(pair.getKey(), pair.getValue().getArray().toFloatArray());
    }
    return values;
  }
}
