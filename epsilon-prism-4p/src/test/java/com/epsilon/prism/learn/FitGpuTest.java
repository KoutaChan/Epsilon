package com.epsilon.prism.learn;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.PairLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** BF16 の学習器でパラメーターを更新する試験。対応する形では HIP の Attention を記録する。 */
public class FitGpuTest {
  private static final Logger log = LoggerFactory.getLogger(FitGpuTest.class);

  /** GPU の学習器のステップ数。損失は {@code METRIC_EVERY} ステップおきに回収するので、最初と最後の損失が別のステップになる数にする。 */
  private static final int GPU_STEPS = 2 * Learner.METRIC_EVERY + 1;

  private static final int BUDGET = 6_000;

  private Path root;

  @BeforeClass(alwaysRun = true)
  public void writeCohort() throws IOException {
    root = Files.createTempDirectory("prism-fit-gpu");
    SyntheticCohort.write(CohortWriter.directory(root, 1), 1, SyntheticCohort.rows(500, 1, 6));
  }

  @AfterClass(alwaysRun = true)
  public void deleteCohort() throws IOException {
    if (root != null) {
      Snapshot.deleteTree(root);
    }
  }

  /** 乱した重みから学習し、損失が下がり各部の重みが動くことを確かめる。 */
  @Test(groups = "rocm", timeOut = 600_000, dataProvider = "gpuShapes")
  public void gpuLearnerUpdatesTheNetwork(PrismShape shape) throws IOException {
    if (Engine.getInstance().getGpuCount() == 0
        || (shape.supportsAttentionTape() && !HipLibrary.bundled())) {
      throw new SkipException("needs a GPU and the bundled libprism_hip.so");
    }
    FitPlan plan =
        new FitPlan(
            new FitSettings(
                GPU_STEPS,
                4096,
                3e-3,
                3e-4,
                1f,
                0.01,
                0.02,
                PairLoss.Mode.PRISM,
                PairLoss.ClassWeight.INVERSE_LAMBDA,
                AdamEpsilon.FIXED),
            shape,
            AuxiliaryLoss.standard(),
            BUDGET,
            19);
    Device gpu = Device.gpu(0);
    try (CohortStore store = new CohortStore();
        Model model = Fit.blank(FOUR_PLAYER, shape, gpu, plan.seed())) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] rows = store.select(1, 1, Selection.LABELS);
      Map<String, float[]> before = perturb((PrismNetwork) model.getBlock());
      FitResult result = Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(gpu), 2);
      // この小さなコホートの行では、ほぼ全てのゲートで exact のフィールドが 0 なので、ゲートの exact の重みは勾配が 0 になることがある（CPU でも同じ）
      List<String> frozen = frozen((PrismNetwork) model.getBlock(), before);
      frozen.removeIf(
          name -> name.endsWith("gates_exactProjection") || name.endsWith("gates_exactOutput"));
      log.info(
          "PRISM-FIT-GPU shape={} steps={} loss {} -> {} frozen={} {}s",
          shape,
          result.steps(),
          String.format("%.4f", result.firstLoss()),
          String.format("%.4f", result.lastLoss()),
          frozen,
          String.format("%.1f", result.seconds()));
      assertTrue(Double.isFinite(result.firstLoss()) && Double.isFinite(result.lastLoss()));
      assertTrue(
          result.lastLoss() < result.firstLoss(),
          "loss " + result.firstLoss() + " -> " + result.lastLoss());
      assertTrue(frozen.isEmpty(), "parameters the learner did not update: " + frozen);
    }
  }

  @DataProvider
  public Object[][] gpuShapes() {
    return new Object[][] {
      {new PrismShape.V2(64, 128, 64, 2)},
      {new PrismShape.V3(64, 128, 64, 128, 256, 128, true)},
      {ShapeArm.V3.shape()},
      {ShapeArm.V4_S.shape()}
    };
  }

  /** 全てのパラメーターに N(0, 0.05²) を足し、足した後の値を返す。 */
  private static Map<String, float[]> perturb(PrismNetwork network) {
    Random random = new Random(17);
    Map<String, float[]> values = new LinkedHashMap<>();
    for (Pair<String, Parameter> pair : network.getParameters()) {
      NDArray array = pair.getValue().getArray();
      float[] perturbed = array.toFloatArray();
      for (int i = 0; i < perturbed.length; i++) {
        perturbed[i] += (float) random.nextGaussian() * 0.05f;
      }
      array.set(perturbed);
      values.put(pair.getKey(), perturbed);
    }
    return values;
  }

  /** 乱した後の値から動いていないパラメーターの名前。 */
  private static List<String> frozen(PrismNetwork network, Map<String, float[]> before) {
    List<String> names = new ArrayList<>();
    for (Pair<String, Parameter> pair : network.getParameters()) {
      if (Arrays.equals(pair.getValue().getArray().toFloatArray(), before.get(pair.getKey()))) {
        names.add(pair.getKey());
      }
    }
    return names;
  }
}
