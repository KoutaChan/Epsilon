package com.epsilon.prism.learn;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.prism.Limits;
import com.epsilon.prism.Main;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.app.FitCohorts;
import com.epsilon.prism.auxiliary.OracleCapture;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.SuitSymmetry;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.kernel.TrainingKernels;
import com.epsilon.prism.model.AuxiliaryLabels;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.ModelShapes;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.PairLoss;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 学習の流れを確かめる（CPU、FP32、小さなネットワーク）。合成したコホートを 2 つ書く。行は GREEDY で打った実際の判断の O と Oracle、
 * 局末の結果に、乱数の Y を付けたもの。ストアから読んでバッチにし、数ステップ学習すると損失が下がる。バージョンに書いて開き直しても s は
 * ビットで同じ。補助の教師の形は補助のヘッドの出力と合う。読みの教師の牌の軸は、入力と同じように色を並べ替える（色を入れ替えた半荘の対で
 * 入力の牌のトークンが同じなら、読みの教師もビットで同じ）。先に計算したアンカーの s は、行の番号で引いて教師に混ぜる。
 */
public class FitTest {
  private static final Logger log = LoggerFactory.getLogger(FitTest.class);

  private static final PrismShape SHAPE = new PrismShape.V2(64, 128, 64, 2);
  private static final int STEPS = 16;
  private static final int BUDGET = 6_000;
  private static final BatchPlan.Capacity ONE_ROW =
      new BatchPlan.Capacity(
          1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS);

  private Path root;
  private List<Row> first;
  private List<Row> second;

  @Test
  public void defaultAdamMatchesDjlBitForBit() {
    Tracker rate = t -> 0.01f / t;
    var setting = TestSettings.of().fit().fit(3, 1).adamEpsilon();
    assertEquals(setting, AdamEpsilon.FIXED);
    try (var manager = NDManager.newBaseManager(Device.cpu());
        var actual = setting.optimizer(rate, 0.035f);
        var baseline = Optimizer.adamW().optLearningRateTracker(rate).optWeightDecays(0f).build()) {
      NDArray a = manager.create(0f), b = manager.create(0f);
      for (float g : new float[] {2e-9f, -4e-8f, 7e-9f}) {
        NDArray gradient = manager.create(g);
        actual.update("weight", a, gradient);
        baseline.update("weight", b, gradient);
        assertEquals(Float.floatToRawIntBits(a.getFloat()), Float.floatToRawIntBits(b.getFloat()));
      }
    }
  }

  @Test
  public void tempoAdamMatchesBiasCorrectedUnscaledReference() {
    var learner = TestSettings.of("network.shape=V4_S", "fit.adamEpsilon=TEMPO").fit().fit(3, 1);
    for (PrismShape shape : List.of(SHAPE, ShapeArm.V4_S.shape(), ShapeArm.V4_M.shape())) {
      var plan = new FitPlan(learner, shape, new AuxiliaryLoss(0, 0), BUDGET, 0);
      assertEquals(plan.shape(), shape);
    }
    for (float gain : new float[] {1f, 0.035f}) {
      try (var manager = NDManager.newBaseManager(Device.cpu());
          var optimizer = learner.adamEpsilon().optimizer(Tracker.fixed(0.01f), gain)) {
        NDArray weight = manager.create(0f);
        double m = 0, v = 0, reference = 0;
        int t = 0;
        for (float g : new float[] {2e-9f, -4e-8f, 7e-9f}) {
          t++;
          m = 0.9 * m + 0.1 * g;
          v = 0.999 * v + 0.001 * (double) g * g;
          double correctedMean = m / (1 - Math.pow(0.9, t));
          double correctedRms = Math.sqrt(v / (1 - Math.pow(0.999, t)));
          reference -= 0.01 * correctedMean / (correctedRms + 1e-8);
          optimizer.update("weight", weight, manager.create(gain * g));
          assertEquals((double) weight.getFloat(), reference, 3e-9, "gain=" + gain + " step=" + t);
        }
      }
    }
  }

  @BeforeClass(alwaysRun = true)
  public void writeCohorts() throws IOException {
    root = Files.createTempDirectory("prism-learn");
    TestSettings.create(
        root,
        "learner.devices=[cpu]",
        "learner.precision=FLOAT32",
        "learner.decodeThreads=1",
        "fit.batch=4",
        "fit.epochs=1");
    first = SyntheticCohort.rows(500, 1, 6);
    second = SyntheticCohort.rows(700, 1, 6);
    SyntheticCohort.write(CohortWriter.directory(root, 1), 1, first);
    SyntheticCohort.write(CohortWriter.directory(root, 2), 2, second);
  }

  @Test
  public void resumedGpuLossReadbackKeepsTheGlobalCadence() {
    int[] firstMeasured = {1, 9, 17, 5, 13};
    for (int chunk = 0; chunk < 20; chunk++) {
      int start = chunk * 512;
      int local = 1;
      while (!Learner.samplesLoss(start + local, true)) local++;
      assertEquals(local, firstMeasured[chunk % firstMeasured.length]);
      assertTrue(Learner.samplesLoss(start + 1, false));
    }
  }

  @Test
  public void resumedPlateauPassKeepsThePermutationMomentsAndLearningRate() throws IOException {
    int batch = 2, steps = 4;
    var plan =
        new FitPlan(
            new FitSettings(
                steps,
                batch,
                1e-3,
                1e-4,
                0.5f,
                0.01,
                0.02,
                PairLoss.Mode.PRISM,
                PairLoss.ClassWeight.INVERSE_LAMBDA,
                AdamEpsilon.FIXED),
            SHAPE,
            AuxiliaryLoss.standard(),
            BUDGET,
            23);
    var devices = NetworkDevices.of(Device.cpu());
    Path saved = root.resolve("plateau-weights"), adam = root.resolve("plateau-adam");
    try (var store = new CohortStore();
        var reference = Fit.blank(FOUR_PLAYER, SHAPE, Device.cpu(), 23);
        var chunked = Fit.blank(FOUR_PLAYER, SHAPE, Device.cpu(), 23)) {
      store.addCohort(CohortWriter.directory(root, 1));
      store.addCohort(CohortWriter.directory(root, 2));
      int[] rows = Arrays.copyOf(store.select(1, 2, Selection.LABELS), steps * batch - 1);
      int[] order = Fit.shuffle(rows, plan.seed());
      Fit.run(store, rows, reference, plan, Teacher.RAW, devices, 1);
      Fit.chunk(
          store,
          Arrays.copyOf(order, batch * 2),
          chunked,
          plan,
          Teacher.RAW,
          devices,
          1,
          0,
          null,
          adam);
      ((PrismNetwork) chunked.getBlock()).save(saved);
      try (var resumed = Fit.load(saved, Device.cpu())) {
        Fit.chunk(
            store,
            Arrays.copyOfRange(order, batch * 2, order.length),
            resumed,
            plan,
            Teacher.RAW,
            devices,
            1,
            2,
            adam,
            null);
        for (var parameter : reference.getBlock().getParameters()) {
          assertEquals(
              resumed.getBlock().getParameters().get(parameter.getKey()).getArray().toFloatArray(),
              parameter.getValue().getArray().toFloatArray(),
              parameter.getKey());
        }
      }
    }
  }

  @AfterClass(alwaysRun = true)
  public void deleteCohorts() throws IOException {
    if (root == null) {
      return;
    }
    try (var files = Files.walk(root)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(file);
      }
    }
  }

  /**
   * 学習器（{@link Fit#run}。データ並列の部品と AdamW）を通しても、全てのパラメーターのテンソルが動く。勾配が届かず 0 として扱われる
   * パラメーターはない。初期化したままのネットワークでは 0 で始めるヘッドが上流への勾配を止めるので、全てのパラメーターを少し乱してから学習する。
   */
  @Test(dataProvider = "smallShapes", dataProviderClass = ModelShapes.class, timeOut = 600_000)
  public void everyParameterLearnsThroughTheLearner(PrismShape shape) throws IOException {
    FitPlan plan =
        new FitPlan(
            new FitSettings(
                4,
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
            13);
    try (CohortStore store = new CohortStore();
        Model model = Fit.blank(FOUR_PLAYER, shape, Device.cpu(), plan.seed())) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] rows = store.select(1, 1, Selection.LABELS);
      PrismNetwork network = (PrismNetwork) model.getBlock();
      Random random = new Random(17);
      Map<String, float[]> before = new LinkedHashMap<>();
      for (Pair<String, Parameter> pair : network.getParameters()) {
        NDArray array = pair.getValue().getArray();
        float[] values = array.toFloatArray();
        for (int i = 0; i < values.length; i++) {
          values[i] += (float) random.nextGaussian() * 0.05f;
        }
        array.set(values);
        before.put(pair.getKey(), values);
      }
      Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(Device.cpu()), 2);
      List<String> frozen = new ArrayList<>();
      for (Pair<String, Parameter> pair : network.getParameters()) {
        float[] after = pair.getValue().getArray().toFloatArray();
        if (Arrays.equals(after, before.get(pair.getKey()))) {
          frozen.add(pair.getKey());
        }
      }
      log.info("PRISM-FIT-PARAMETERS total={} unchanged={}", before.size(), frozen);
      assertTrue(frozen.isEmpty(), "parameters the learner did not update: " + frozen);
    }
  }

  @Test(
      dataProvider = "smallShapes",
      dataProviderClass = ModelShapes.class,
      groups = "slow",
      timeOut = 600_000)
  public void fitLowersTheLossAndTheSnapshotKeepsScores(PrismShape shape) throws IOException {
    FitPlan plan =
        new FitPlan(
            new FitSettings(
                STEPS,
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
            11);
    try (CohortStore store = new CohortStore()) {
      store.addCohort(CohortWriter.directory(root, 1));
      store.addCohort(CohortWriter.directory(root, 2));
      assertEquals(store.select(1, 1, Selection.LABELS).length, SyntheticCohort.training(first));
      int[] rows = store.select(1, 2, Selection.LABELS);
      assertEquals(rows.length, SyntheticCohort.training(first) + SyntheticCohort.training(second));

      try (Model model = Fit.blank(FOUR_PLAYER, shape, Device.cpu(), plan.seed())) {
        FitResult result =
            Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(Device.cpu()), 2);
        log.info(
            "PRISM-FIT rows={} steps={} loss {} -> {} mean {} clipped={} {}s stages={}",
            result.rows(),
            result.steps(),
            String.format("%.4f", result.firstLoss()),
            String.format("%.4f", result.lastLoss()),
            String.format("%.4f", result.loss()),
            result.clippedSteps(),
            String.format("%.1f", result.seconds()),
            Arrays.toString(result.stageSeconds()));
        assertEquals(result.steps(), STEPS);
        assertTrue(Double.isFinite(result.loss()), "loss " + result.loss());
        assertTrue(
            result.lastLoss() < result.firstLoss() - 0.01,
            "loss " + result.firstLoss() + " -> " + result.lastLoss());

        PrismNetwork network = (PrismNetwork) model.getBlock();
        Path snapshot = Files.createTempDirectory(root, "fit-").resolve("snapshot");
        Snapshot.FitRecord record = new Snapshot.FitRecord(root.toString(), 1, 2, 0, plan, result);
        Snapshot.write(snapshot, network, PolicyParams.of(3), record);
        try (StepFeed feed = new StepFeed(store, plan, 1, 2);
            NDManager manager = NDManager.newBaseManager(Device.cpu())) {
          Snapshot.Loaded loaded = Snapshot.load(snapshot, manager);
          assertEquals(loaded.params().version(), 3);
          assertEquals(loaded.fit().plan(), plan);
          assertEquals(loaded.network().shape(), shape);
          assertEquals(loaded.fit().result().steps(), STEPS);
          StepFeed.Step step = feed.prepare(Arrays.copyOf(rows, 40), Teacher.RAW).join();
          for (Microbatch part : step.lanes().get(0)) {
            assertSameScores(network, loaded.network(), part, manager);
            assertAuxiliaryShapes(loaded.network(), part, manager);
          }
          feed.release(step);
        }
      }
    }
  }

  /**
   * 学習器は学習カーネルをデバイスごとに開き、作り終えたレーンにだけ渡す。2 つ目のデバイスで開くのに失敗しても、レーンを作る前に閉じても、
   * レーンを作った後に順伝播で失敗しても、開いたカーネルの数と閉じた数が合う。
   */
  @Test(timeOut = 600_000)
  public void theLearnerClosesEveryTrainingKernelItOpens() throws IOException {
    FitPlan plan =
        new FitPlan(
            new FitSettings(
                1,
                4096,
                3e-3,
                3e-4,
                1f,
                0.01,
                0.02,
                PairLoss.Mode.PRISM,
                PairLoss.ClassWeight.INVERSE_LAMBDA,
                AdamEpsilon.FIXED),
            SHAPE,
            new AuxiliaryLoss(0f, 0f),
            BUDGET,
            5);
    NetworkDevices two = NetworkDevices.of(Device.cpu(), Device.cpu());
    try (Model model = Fit.blank(FOUR_PLAYER, SHAPE, Device.cpu(), plan.seed());
        NetworkRuntime runtime = NetworkRuntime.forLearning();
        CohortStore store = new CohortStore()) {
      CountingKernels failing = new CountingKernels(1);
      expectThrows(
          IllegalStateException.class, () -> new Learner(model, two, plan, 1f, runtime, failing));
      assertEquals(failing.opened, 1);
      assertEquals(failing.closed, 1);

      CountingKernels unused = new CountingKernels(-1);
      new Learner(model, two, plan, 1f, runtime, unused).close();
      assertEquals(unused.opened, 2);
      assertEquals(unused.closed, 2);

      store.addCohort(CohortWriter.directory(root, 1));
      int[] rows = Arrays.copyOf(store.select(1, 1, Selection.LABELS), 8);
      CountingKernels lanes = new CountingKernels(-1);
      try (Learner learner = new Learner(model, two, plan, 1f, runtime, lanes);
          StepFeed feed = new StepFeed(store, plan, two.size(), 1)) {
        StepFeed.Step step = feed.prepare(rows, Teacher.RAW).join();
        try {
          expectThrows(RuntimeException.class, () -> learner.rehearse(step));
        } finally {
          feed.release(step);
        }
      }
      assertEquals(lanes.opened, 2);
      assertEquals(lanes.closed, 2);
    }
  }

  /** 開いた数と閉じた数を数える学習カーネル。演算を呼ぶと失敗する。{@code failAt} 番目（0 から）の開く処理は失敗する。 */
  private static final class CountingKernels implements Learner.LaneKernels {
    private final int failAt;
    private int opened;
    private int closed;

    CountingKernels(int failAt) {
      this.failAt = failAt;
    }

    @Override
    public TrainingKernels open(Device device, PrismNetwork network, PtStream stream) {
      if (opened == failAt) {
        throw new IllegalStateException("The training kernel provider failed");
      }
      opened++;
      return (TrainingKernels)
          Proxy.newProxyInstance(
              TrainingKernels.class.getClassLoader(),
              new Class<?>[] {TrainingKernels.class},
              (self, method, args) ->
                  switch (method.getName()) {
                    case "close" -> {
                      closed++;
                      yield null;
                    }
                    case "candidateKernels" -> null;
                    case "hashCode" -> System.identityHashCode(self);
                    case "equals" -> self == args[0];
                    case "toString" -> "CountingKernels";
                    default -> throw new UnsupportedOperationException(method.getName());
                  });
    }
  }

  /**
   * データ並列（CPU に 2 レーン）。補助のヘッドの項を外した計画では、2 レーンで学習した損失と重みが、1 レーンで学習したものと許容誤差の内で
   * 合う。主の損失は行ごとの和を 1 ステップの分母で割るので、マイクロバッチの分け方に依らない。補助の項はバッチごとの平均なので、レーンの数で
   * バッチが変わると値も変わる。補助のヘッドの項を入れた計画では、1 行ずつのマイクロバッチを 2 レーンに置いたステップでも落ちない
   * （補助のヘッドの項が届かない行だけのレーンもできる）。
   */
  @Test(timeOut = 600_000)
  public void twoLanesMatchOneLane() throws IOException {
    FitSettings settings =
        new FitSettings(
            4,
            4096,
            3e-3,
            3e-4,
            1f,
            0.01,
            0.02,
            PairLoss.Mode.PRISM,
            PairLoss.ClassWeight.INVERSE_LAMBDA,
            AdamEpsilon.FIXED);
    FitPlan plan = new FitPlan(settings, SHAPE, new AuxiliaryLoss(0f, 0f), BUDGET, 21);
    NetworkDevices two = NetworkDevices.of(Device.cpu(), Device.cpu());
    try (CohortStore store = new CohortStore()) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] rows = store.select(1, 1, Selection.LABELS);
      Map<String, float[]> singleWeights = new LinkedHashMap<>();
      Map<String, float[]> parallelWeights = new LinkedHashMap<>();
      FitResult single =
          fitWeights(store, rows, plan, NetworkDevices.of(Device.cpu()), singleWeights);
      FitResult parallel = fitWeights(store, rows, plan, two, parallelWeights);
      double largest = 0;
      for (var entry : singleWeights.entrySet()) {
        float[] expected = entry.getValue();
        float[] actual = parallelWeights.get(entry.getKey());
        assertEquals(actual.length, expected.length, entry.getKey());
        for (int i = 0; i < expected.length; i++) {
          largest = Math.max(largest, Math.abs(actual[i] - expected[i]));
        }
      }
      log.info(
          "PRISM-FIT-LANES loss {} / {} -> {} / {} largest weight difference {}",
          single.firstLoss(),
          parallel.firstLoss(),
          single.lastLoss(),
          parallel.lastLoss(),
          largest);
      assertEquals(parallel.steps(), single.steps());
      assertEquals(parallel.firstLoss(), single.firstLoss(), 1e-5 * Math.abs(single.firstLoss()));
      assertEquals(parallel.lastLoss(), single.lastLoss(), 1e-4 * Math.abs(single.lastLoss()));
      assertTrue(largest <= 1e-4, "largest weight difference " + largest);

      FitPlan auxiliary =
          new FitPlan(
              settings,
              SHAPE,
              AuxiliaryLoss.standard(),
              FOUR_PLAYER.fixedTokens() + Limits.MAX_EVENT_TOKENS,
              21);
      FitResult split =
          fitWeights(store, Arrays.copyOf(rows, 6), auxiliary, two, new LinkedHashMap<>());
      assertEquals(split.steps(), 4);
      assertTrue(Double.isFinite(split.loss()), "loss " + split.loss());
    }
  }

  private static FitResult fitWeights(
      CohortStore store,
      int[] rows,
      FitPlan plan,
      NetworkDevices devices,
      Map<String, float[]> weights)
      throws IOException {
    try (Model model = Fit.blank(FOUR_PLAYER, SHAPE, Device.cpu(), plan.seed())) {
      FitResult result = Fit.run(store, rows, model, plan, Teacher.RAW, devices, 2);
      for (Pair<String, Parameter> pair : model.getBlock().getParameters()) {
        weights.put(pair.getKey(), pair.getValue().getArray().toFloatArray());
      }
      return result;
    }
  }

  /**
   * ウォームアップしてからの学習（{@link Fit#prepare}）。学習器をウォームアップしてから学習した重みが、ウォームアップせずに学習した重み
   * （{@link Fit#run}）とビットで同じ。ウォームアップは重み・最適化の状態・ステップ数・.grad を残さない。
   */
  @Test(timeOut = 600_000)
  public void preparedFitMatchesTheColdFitBitwise() throws IOException {
    FitPlan plan =
        new FitPlan(
            new FitSettings(
                4,
                4096,
                3e-3,
                3e-4,
                1f,
                0.01,
                0.02,
                PairLoss.Mode.PRISM,
                PairLoss.ClassWeight.INVERSE_LAMBDA,
                AdamEpsilon.FIXED),
            SHAPE,
            AuxiliaryLoss.standard(),
            BUDGET,
            9);
    Path start = root.resolve("start");
    try (Model model = Fit.blank(FOUR_PLAYER, SHAPE, Device.cpu(), 9)) {
      PrismNetwork network = (PrismNetwork) model.getBlock();
      Random random = new Random(29);
      for (Pair<String, Parameter> pair : network.getParameters()) {
        NDArray array = pair.getValue().getArray();
        float[] values = array.toFloatArray();
        for (int i = 0; i < values.length; i++) {
          values[i] += (float) random.nextGaussian() * 0.05f;
        }
        array.set(values);
      }
      network.save(start);
    }
    Teacher teacher = new Teacher(null, null, 0, 0.5f);
    NetworkDevices cpu = NetworkDevices.of(Device.cpu());
    try (CohortStore store = new CohortStore()) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] rows = store.select(1, 1, Selection.LABELS);
      Map<String, float[]> cold = new LinkedHashMap<>();
      try (Model model = Fit.load(start, Device.cpu())) {
        Fit.run(store, rows, model, plan, teacher, cpu, 2);
        for (Pair<String, Parameter> pair : model.getBlock().getParameters()) {
          cold.put(pair.getKey(), pair.getValue().getArray().toFloatArray());
        }
      }
      try (NetworkRuntime runtime = NetworkRuntime.forLearning();
          Fit.Prepared prepared = Fit.prepare(store, rows, start, plan, 0.5f, cpu, 2, runtime)) {
        expectThrows(IllegalArgumentException.class, () -> prepared.fit(Teacher.RAW));
        FitResult result = prepared.fit(teacher);
        assertEquals(result.steps(), 4);
        for (Pair<String, Parameter> pair : prepared.network().getParameters()) {
          float[] warm = pair.getValue().getArray().toFloatArray();
          float[] reference = cold.get(pair.getKey());
          assertEquals(warm.length, reference.length, pair.getKey());
          for (int i = 0; i < warm.length; i++) {
            assertEquals(
                Float.floatToRawIntBits(warm[i]),
                Float.floatToRawIntBits(reference[i]),
                pair.getKey() + " " + i);
          }
        }
      }
    }
  }

  /**
   * アンカーの s（{@link AnchorScores}）。全ての行の s を先に計算して教師に混ぜた値が、同じマイクロバッチをアンカーのネットワークに通して
   * 混ぜた値と合う。eager の CPU のアンカーはバッチの組み方で最後のビットが変わるので、その差だけを許す（行を取り違えれば桁が違う）。
   * 別のバージョンのアンカーと照合すると例外で止まり、無い行は引けない。
   */
  @Test(timeOut = 600_000)
  public void anchorScoresFollowEachRowIntoTheTeacher() throws IOException {
    FitPlan plan =
        new FitPlan(
            new FitSettings(
                1,
                4096,
                3e-3,
                3e-4,
                1f,
                0.01,
                0.02,
                PairLoss.Mode.PRISM,
                PairLoss.ClassWeight.INVERSE_LAMBDA,
                AdamEpsilon.FIXED),
            SHAPE,
            AuxiliaryLoss.standard(),
            BUDGET,
            5);
    Path anchor = root.resolve("anchor");
    try (Model model = Fit.blank(FOUR_PLAYER, SHAPE, Device.cpu(), 5)) {
      PrismNetwork network = (PrismNetwork) model.getBlock();
      Random random = new Random(21);
      for (Pair<String, Parameter> pair : network.getParameters()) {
        NDArray array = pair.getValue().getArray();
        float[] values = array.toFloatArray();
        for (int i = 0; i < values.length; i++) {
          values[i] += (float) random.nextGaussian() * 0.05f;
        }
        array.set(values);
      }
      network.save(anchor);
    }
    try (CohortStore store = new CohortStore()) {
      store.addCohort(CohortWriter.directory(root, 1));
      store.addCohort(CohortWriter.directory(root, 2));
      int[] rows = store.select(1, 2, Selection.LABELS);
      AnchorScores scores =
          AnchorScores.compute(
              store, rows, anchor, 3, Kernels.EAGER, NetworkDevices.of(Device.cpu()), 2);
      scores.requireVersion(3);
      expectThrows(IllegalStateException.class, () -> scores.requireVersion(4));
      expectThrows(IllegalArgumentException.class, () -> scores.scores(-1));
      Teacher teacher = new Teacher(null, scores, 0.5, 1f);
      int[] part = Arrays.copyOf(rows, 40);
      int compared = 0;
      try (StepFeed feed = new StepFeed(store, plan, 1, 2);
          Anchor eager =
              Anchor.open(
                  anchor,
                  Kernels.EAGER,
                  Device.cpu(),
                  null,
                  null,
                  StepFeed.capacity(FOUR_PLAYER, BUDGET))) {
        StepFeed.Step plain = feed.prepare(part, Teacher.RAW).join();
        StepFeed.Step anchored = feed.prepare(part, teacher).join();
        List<Microbatch> references = plain.lanes().get(0);
        List<Microbatch> results = anchored.lanes().get(0);
        assertEquals(results.size(), references.size());
        for (int p = 0; p < references.size(); p++) {
          Microbatch reference = references.get(p);
          Anchor.Scores s = eager.scores(reference.batch());
          for (int row = 0; row < reference.rows(); row++) {
            float[] y = reference.labels().get(row).values();
            int from = s.starts()[row];
            float[] expected =
                Teacher.anchored(
                    y,
                    Arrays.copyOfRange(s.values(), from, from + y.length),
                    0.5,
                    new float[y.length]);
            float[] actual = results.get(p).labels().get(row).values();
            assertEquals(actual.length, expected.length);
            for (int a = 0; a < expected.length; a++) {
              if (Float.isNaN(expected[a])) {
                assertTrue(Float.isNaN(actual[a]), "row " + row + " action " + a);
                continue;
              }
              assertTrue(
                  Math.abs(actual[a] - expected[a]) <= 1e-5 + 1e-4 * Math.abs(expected[a]),
                  "row " + row + " action " + a + ": " + actual[a] + " vs " + expected[a]);
              compared++;
            }
          }
        }
        feed.release(plain);
        feed.release(anchored);
      }
      log.info("PRISM-ANCHOR-SCORES rows={} compared={}", rows.length, compared);
      assertTrue(compared > 0, "no measured actions were compared");
    }
  }

  /** 学習したネットワークと、保存して開き直したネットワークの s がビットで同じ。 */
  private static void assertSameScores(
      PrismNetwork trained, PrismNetwork opened, Microbatch part, NDManager manager) {
    try (NDManager scope = manager.newSubManager()) {
      float[] before = scores(trained, part, scope);
      float[] after = scores(opened, part, scope);
      assertEquals(after.length, before.length);
      for (int index = 0; index < before.length; index++) {
        assertEquals(
            Float.floatToRawIntBits(after[index]),
            Float.floatToRawIntBits(before[index]),
            "score " + index);
      }
      assertTrue(maxAbs(before) > 0, "trained scores must be non-trivial");
    }
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

  /** 補助の教師の形が補助のヘッドの出力と合い、補助の損失が有限で正。 */
  private static void assertAuxiliaryShapes(
      PrismNetwork network, Microbatch part, NDManager manager) {
    AuxiliaryLabels labels = part.auxiliary();
    int candidates = labels.candidateActions().length;
    int readings = labels.readingRows().length;
    assertTrue(
        candidates > 0 && readings > 0, "candidates " + candidates + " readings " + readings);
    try (NDManager scope = manager.newSubManager()) {
      Outputs outputs =
          network.forward(
              new ParameterStore(scope, false),
              part.inputs(scope, PinnedUploads.on(scope)),
              Pass.training(Precision.FLOAT32, part.measured(), labels));
      assertEquals(
          outputs.auxiliary().candidates().getShape(),
          new Shape(candidates, TargetSchema.of(TENHOU_4P).candidateWidth()));
      assertEquals(
          outputs.auxiliary().reading().getShape(),
          new Shape(
              readings,
              TargetSchema.of(TENHOU_4P).opponents(),
              TargetSchema.of(TENHOU_4P).readingWidth()));
      float total =
          AuxiliaryLoss.standard().terms(scope, outputs.auxiliary(), labels, 1f).total().getFloat();
      assertTrue(Float.isFinite(total) && total > 0, "auxiliary loss " + total);
    }
  }

  @Test(timeOut = 600_000)
  public void readingTargetsFollowTheInputSuitMap() {
    FitPlan plan =
        new FitPlan(
            new FitSettings(
                1,
                1,
                1e-4,
                1e-5,
                1f,
                0.01,
                0.02,
                PairLoss.Mode.PRISM,
                PairLoss.ClassWeight.INVERSE_LAMBDA,
                AdamEpsilon.FIXED),
            SHAPE,
            AuxiliaryLoss.standard(),
            BUDGET,
            1);
    Side left = new Side(plan);
    Side right = new Side(plan);
    int[] counts = new int[3];
    for (int view = 1; view < SuitSymmetry.VIEWS; view++) {
      new TableWalk(TENHOU_4P, TableWalk.Policy.GREEDY)
          .runMirrored(
              600 + 100L * view,
              10,
              view,
              (engine, slot, mirror, mirrorSlot) -> {
                AuxiliaryLabels a = left.encode(engine, slot);
                AuxiliaryLabels b = right.encode(mirror, mirrorSlot);
                if (!Arrays.equals(left.tiles(), right.tiles())) {
                  counts[2]++;
                  return;
                }
                assertTrue(
                    Arrays.equals(a.readingValues(), b.readingValues()),
                    "reading targets differ at seat " + engine.decisions().seat(slot));
                counts[0]++;
                if (left.informative && left.physicalTiles != right.physicalTiles) {
                  counts[1]++;
                }
              });
    }
    log.info(
        "PRISM-AUX compared={} physicallyDifferent={} inputTies={}",
        counts[0],
        counts[1],
        counts[2]);
    assertTrue(counts[0] > 1_000, "compared decisions " + counts[0]);
    assertTrue(counts[1] > 50, "pairs whose physical waits or dangers differ " + counts[1]);
    assertTrue(counts[2] * 20 < counts[0], "input ties " + counts[2]);
  }

  /** 色を入れ替えた対の片側。判断を 1 行のバッチに書き、補助の教師を作る。 */
  private static final class Side {
    private final Encoder encoder;
    private final ObservationWriter writer = new ObservationWriter(TENHOU_4P);
    private final ObservationBatch observations = new ObservationBatch(TENHOU_4P, 1 << 12);
    private final RuleWorkspace ws = new RuleWorkspace(TENHOU_4P);
    private final OracleCodec oracles = OracleCodec.of(TENHOU_4P);
    private final Observation observation = new Observation(TENHOU_4P);
    private final HostInputs batch =
        HostInputs.allocate(FOUR_PLAYER, ONE_ROW, ByteBuffer::allocateDirect);
    private boolean informative;
    private long physicalTiles;

    Side(FitPlan plan) {
      encoder = new Encoder(plan, TENHOU_4P);
    }

    AuxiliaryLabels encode(Engine engine, int slot) {
      Decisions decisions = engine.decisions();
      int seat = decisions.seat(slot);
      observations.clear();
      int index = writer.capture(engine.view(seat), decisions, slot, observations);
      long[] o =
          Arrays.copyOfRange(
              observations.words(),
              observations.start(index),
              observations.start(index) + observations.longs(index));
      long[] oracle = OracleCapture.capture(engine.full(), seat, ws);
      int actions = observation.bind(o, 0).legalActions();
      // 全ての相手の待ちと放銃になる牌を、実際の牌のまま重ねる（色の並べ替えが効いた対を数えるため）
      physicalTiles = 0L;
      for (int opponent = 1; opponent <= TargetSchema.of(TENHOU_4P).opponents(); opponent++) {
        physicalTiles |= oracles.waits(oracle, 0, opponent) | oracles.danger(oracle, 0, opponent);
      }
      informative = physicalTiles != 0L;
      float[] values = new float[actions];
      Arrays.fill(values, Float.NaN);
      Row row =
          new Row(
              0,
              0,
              DecisionClass.DISCARD_MID,
              Row.Purpose.TRAIN,
              Row.Source.MAIN,
              0,
              values,
              new float[actions],
              0,
              -1,
              Float.NaN,
              Float.NaN,
              -1,
              new float[0],
              new long[actions * OutcomeCodec.WORDS],
              oracle,
              o);
      batch.plan().clear();
      batch.plan().add(observation.bind(o, 0));
      batch.plan().seal();
      return encoder.encode(batch, List.of(row), new int[1], Teacher.RAW).auxiliary();
    }

    /** 入力の牌のトークンのフィールド（色を並べ替えた後）。 */
    byte[] tiles() {
      BatchPlan plan = batch.plan();
      byte[] bytes = new byte[plan.count(Axis.TILE) * FOUR_PLAYER.width(Tensor.TILE_CATEGORY)];
      batch.input().blob().get(plan.offset(Tensor.TILE_CATEGORY), bytes);
      return bytes;
    }
  }

  private static float maxAbs(float[] values) {
    float max = 0;
    for (float value : values) {
      max = Math.max(max, Math.abs(value));
    }
    return max;
  }

  /** 学習のカーネルが対応しない形では、製品の学習の選び方は GPU の演算を選ばない（学習は eager の演算を使う）。 */
  @Test(dataProvider = "eagerArms", dataProviderClass = ModelShapes.class)
  public void eagerArmsOpenNoTrainingKernels(PrismShape shape) {
    try (Model model = Fit.blank(FOUR_PLAYER, shape, Device.cpu(), 1)) {
      assertNull(Learner.training(Device.gpu(0), (PrismNetwork) model.getBlock(), null));
    }
  }

  @DataProvider
  public Object[][] refitArms() {
    return new Object[][] {
      {ShapeArm.V2}, {ShapeArm.V3}, {ShapeArm.V4_S}, {ShapeArm.V4_M},
    };
  }

  @Test(dataProvider = "refitArms")
  public void fitCohortsSavesTheSelectedArm(ShapeArm arm) throws Exception {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Path out = Files.createTempDirectory(root, "refit-");
      List<String> arguments =
          new ArrayList<>(
              List.of(
                  "fit-cohorts",
                  "--run",
                  root.toString(),
                  "--cohorts",
                  "1..1",
                  "--out",
                  out.toString(),
                  "--shape",
                  arm.name(),
                  "--max-rows",
                  "4",
                  "--set",
                  "fit.loss=PAIR_STAR",
                  "--set",
                  "fit.classWeight=UNIFORM"));
      if (arm.selectableDepth()) {
        arguments.addAll(List.of("--depth", "2"));
      }
      assertEquals(Main.run(arguments.toArray(String[]::new)), 0);
      Snapshot.Loaded loaded = Snapshot.load(out.resolve(FitCohorts.SNAPSHOT), manager);
      assertEquals(loaded.network().shape(), arm.shape());
      assertEquals(loaded.fit().plan().shape(), loaded.network().shape());
      assertEquals(loaded.fit().plan().learner().loss(), PairLoss.Mode.PAIR_STAR);
      assertEquals(loaded.fit().plan().learner().classWeight(), PairLoss.ClassWeight.UNIFORM);
      assertEquals(loaded.fit().result().rows(), 4);
      assertEquals(loaded.fit().result().steps(), 1);
      assertTrue(Double.isFinite(loaded.fit().result().lastLoss()));
    }
  }
}
