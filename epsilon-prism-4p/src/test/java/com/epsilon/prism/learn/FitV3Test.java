package com.epsilon.prism.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism4p.input.InputSchema;
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

/**
 * 型ごとの FFN を持つ形（v3 の構造）を、学習器で小さな合成のコホートに学習させる。学習器は CPU（djl の演算、slow のグループ）と GPU
 * （rocm のグループ。バックボーンの Attention と正規化は、同梱した libprism_hip.so の演算と Attention の記録）の 2 つ。
 */
public class FitV3Test {
  private static final Logger log = LoggerFactory.getLogger(FitV3Test.class);

  private static final PrismShape SHAPE =
      new PrismShape(64, 128, 64, 3, new PrismShape.Typed(128, 256, 128, true));
  private static final int STEPS = 16;

  /** GPU の学習器のステップ数。損失は {@code METRIC_EVERY} ステップおきに回収するので、最初と最後の損失が別のステップになる数にする。 */
  private static final int GPU_STEPS = 2 * Learner.METRIC_EVERY + 1;

  private static final int BUDGET = 6_000;

  private Path root;

  @BeforeClass(alwaysRun = true)
  public void writeCohort() throws IOException {
    root = Files.createTempDirectory("prism-fit-v3");
    SyntheticCohort.write(CohortWriter.directory(root, 1), 1, SyntheticCohort.rows(500, 1, 6));
  }

  @AfterClass(alwaysRun = true)
  public void deleteCohort() throws IOException {
    if (root != null) {
      Snapshot.deleteTree(root);
    }
  }

  @Test(groups = "slow", timeOut = 600_000)
  public void fitLowersTheLossAndTheSnapshotKeepsScores() throws IOException {
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
      try (Model model = Fit.blank(InputSchema.FORMAT, SHAPE, Device.cpu(), plan.seed())) {
        FitResult result =
            Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(Device.cpu()), 2);
        log.info(
            "PRISM-FIT-V3 rows={} steps={} loss {} -> {} {}s",
            result.rows(),
            result.steps(),
            String.format("%.4f", result.firstLoss()),
            String.format("%.4f", result.lastLoss()),
            String.format("%.1f", result.seconds()));
        assertEquals(result.steps(), STEPS);
        assertTrue(
            result.lastLoss() < result.firstLoss() - 0.01,
            "loss " + result.firstLoss() + " -> " + result.lastLoss());
        PrismNetwork network = (PrismNetwork) model.getBlock();
        Path snapshot = root.resolve("out").resolve("snapshot");
        Snapshot.write(
            snapshot,
            network,
            PolicyParams.of(2),
            new Snapshot.FitRecord(root.toString(), 1, 1, 0, plan, result));
        try (StepFeed feed = new StepFeed(store, plan, 1, 2);
            NDManager manager = NDManager.newBaseManager(Device.cpu())) {
          Snapshot.Loaded loaded = Snapshot.load(snapshot, manager);
          assertEquals(loaded.network().shape(), SHAPE);
          assertEquals(loaded.fit().plan(), plan);
          StepFeed.Step step = feed.prepare(Arrays.copyOf(rows, 40), Teacher.RAW).join();
          for (Microbatch part : step.lanes().get(0)) {
            try (NDManager scope = manager.newSubManager()) {
              float[] before = scores(network, part, scope);
              float[] after = scores(loaded.network(), part, scope);
              assertEquals(after, before);
              float max = 0;
              for (float value : before) {
                max = Math.max(max, Math.abs(value));
              }
              assertTrue(max > 0, "trained scores must be non-trivial");
            }
          }
          feed.release(step);
        }
      }
    }
  }

  /** 全てのパラメーターを少し乱してから学習すると、全てのパラメーターのテンソルが動く。 */
  @Test(groups = "slow", timeOut = 600_000)
  public void everyParameterLearnsThroughTheLearner() throws IOException {
    FitPlan plan =
        new FitPlan(
            new FitSettings(4, 4096, 3e-3, 3e-4, 1f, 0.01, 0.02),
            SHAPE,
            AuxiliaryLoss.standard(),
            BUDGET,
            13);
    try (CohortStore store = new CohortStore();
        Model model = Fit.blank(InputSchema.FORMAT, SHAPE, Device.cpu(), plan.seed())) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] rows = store.select(1, 1, Selection.LABELS);
      Map<String, float[]> before = perturb((PrismNetwork) model.getBlock());
      Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(Device.cpu()), 2);
      List<String> frozen = frozen((PrismNetwork) model.getBlock(), before);
      assertTrue(frozen.isEmpty(), "parameters the learner did not update: " + frozen);
    }
  }

  /**
   * GPU の学習器（BF16。バックボーンの Attention と正規化は自前の HIP の演算と Attention の記録）で、乱した型ごとの形（小さな形と V3）を
   * 学習すると、損失が有限のまま下がり、全てのパラメーターのテンソルが動く（GPU 1 枚）。
   */
  @Test(groups = "rocm", timeOut = 600_000, dataProvider = "gpuShapes")
  public void gpuLearnerFitsThroughTheHipTape(PrismShape shape) throws IOException {
    if (Engine.getInstance().getGpuCount() == 0 || !HipLibrary.bundled()) {
      throw new SkipException("needs a GPU and the bundled libprism_hip.so");
    }
    FitPlan plan =
        new FitPlan(
            new FitSettings(GPU_STEPS, 4096, 3e-3, 3e-4, 1f, 0.01, 0.02),
            shape,
            AuxiliaryLoss.standard(),
            BUDGET,
            19);
    Device gpu = Device.gpu(0);
    try (CohortStore store = new CohortStore(); Model model = Fit.blank(InputSchema.FORMAT, shape, gpu, plan.seed())) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] rows = store.select(1, 1, Selection.LABELS);
      Map<String, float[]> before = perturb((PrismNetwork) model.getBlock());
      FitResult result = Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(gpu), 2);
      // この小さなコホートの行では、ほぼ全てのゲートで exact のフィールドが 0 なので、ゲートの exact の重みは勾配が 0 になることがある（CPU でも同じ）
      List<String> frozen = frozen((PrismNetwork) model.getBlock(), before);
      frozen.removeIf(name -> name.endsWith("gates_exactProjection") || name.endsWith("gates_exactOutput"));
      log.info(
          "PRISM-FIT-V3-GPU shape={} steps={} loss {} -> {} frozen={} {}s",
          shape,
          result.steps(),
          String.format("%.4f", result.firstLoss()),
          String.format("%.4f", result.lastLoss()),
          frozen,
          String.format("%.1f", result.seconds()));
      assertTrue(Double.isFinite(result.firstLoss()) && Double.isFinite(result.lastLoss()));
      assertTrue(
          result.lastLoss() < result.firstLoss(), "loss " + result.firstLoss() + " -> " + result.lastLoss());
      assertTrue(frozen.isEmpty(), "parameters the learner did not update: " + frozen);
    }
  }

  @DataProvider
  public Object[][] gpuShapes() {
    return new Object[][] {{SHAPE}, {ShapeArm.V3.shape(ShapeArm.V3_DEPTH)}};
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

  private static float[] scores(PrismNetwork network, Microbatch part, NDManager scope) {
    return network
        .forward(
            new ParameterStore(scope, false),
            part.inputs(scope, PinnedUploads.on(scope)),
            Pass.inference(Precision.FLOAT32))
        .scores()
        .toFloatArray();
  }
}
