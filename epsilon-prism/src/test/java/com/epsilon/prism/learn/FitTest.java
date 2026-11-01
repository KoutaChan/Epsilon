package com.epsilon.prism.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.Limits;
import com.epsilon.prism.auxiliary.OracleCapture;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.SuitSymmetry;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.FrontierBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.model.AuxiliaryLabels;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.Uploads;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationSchema;
import com.epsilon.prism.observe.ObservationWriter;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.run.FitWorkflow;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.workflow.WorkflowArguments;
import com.google.gson.GsonBuilder;
import java.io.IOException;
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
import org.testng.annotations.Test;

/**
 * 学習の道（CPU、FP32、小さな網）。合成のコホート 2 つ（GREEDY の実際の判断の O・神託・局末に乱数の Y を付けた行）を貯蔵から読み、
 * 塊にして数歩当てると損失が下がる。版に書いて開くと s がビットで同じ。補助の教師の形が網の補助の頭と合い、読みの教師の牌の軸は入力と同じ
 * 色の写しになる（色を入れ替えた半荘の対で、入力の牌のトークンが同じ判断は読みの教師もビットで同じ）。前線の腕は命令 refit
 * --frontier on で当てはめ、前線を書いた塊で版の網が動く。
 */
public class FitTest {
  private static final Logger log = LoggerFactory.getLogger(FitTest.class);

  private static final PrismShape SHAPE = new PrismShape(64, 128, 64, 2, false);
  private static final int STEPS = 16;
  private static final int BUDGET = 6_000;
  private static final BatchPlan.Capacity ONE_ROW =
      new BatchPlan.Capacity(
          1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS);

  private Path root;
  private List<Row> first;
  private List<Row> second;

  @BeforeClass
  public void writeCohorts() throws IOException {
    root = Files.createTempDirectory("prism-learn");
    first = SyntheticCohort.rows(500, 1, 6);
    second = SyntheticCohort.rows(700, 1, 6);
    SyntheticCohort.write(CohortWriter.directory(root, 1), 1, first);
    SyntheticCohort.write(CohortWriter.directory(root, 2), 2, second);
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
   * 学習器（{@link Fit#run}。データ並列の部品と AdamW）を通しても、全てのパラメーターのテンソルが動くこと（勾配が届かず 0 として扱われる
   * パラメーターがない）。白紙の網は 0 で始める頭が上流の勾配を止めるので、全てのパラメーターを少し乱してから当てる。
   */
  @Test(timeOut = 600_000)
  public void everyParameterLearnsThroughTheLearner() throws IOException {
    FitPlan plan =
        new FitPlan(
            new FitSettings(4, 4096, 3e-3, 3e-4, 1f, 0.01, 0.02),
            SHAPE,
            AuxiliaryLoss.standard(),
            BUDGET,
            13);
    try (CohortStore store = new CohortStore();
        Model model = Fit.blank(SHAPE, Device.cpu(), plan.seed())) {
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

  @Test(timeOut = 600_000)
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
      store.addCohort(CohortWriter.directory(root, 2));
      assertEquals(store.select(1, 1, Selection.LABELS).length, SyntheticCohort.training(first));
      int[] rows = store.select(1, 2, Selection.LABELS);
      assertEquals(rows.length, SyntheticCohort.training(first) + SyntheticCohort.training(second));

      try (Model model = Fit.blank(SHAPE, Device.cpu(), plan.seed())) {
        FitResult result = Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(Device.cpu()), 2);
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
        Path snapshot = root.resolve("out").resolve("snapshot");
        Snapshot.FitRecord record = new Snapshot.FitRecord(root.toString(), 1, 2, 0, plan, result);
        Snapshot.write(snapshot, network, PolicyParams.of(3), record);
        try (StepFeed feed = new StepFeed(store, plan, Teacher.RAW, 1, 2);
            NDManager manager = NDManager.newBaseManager(Device.cpu())) {
          Snapshot.Loaded loaded = Snapshot.load(snapshot, manager);
          assertEquals(loaded.params().version(), 3);
          assertEquals(loaded.fit().plan(), plan);
          assertEquals(loaded.fit().result().steps(), STEPS);
          StepFeed.Step step = feed.prepare(Arrays.copyOf(rows, 40)).join();
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
   * 前線の腕: 命令 refit --frontier on が run の設定と行で白紙の標準の網（深さ 2・前線あり）を当て、版を書く。版は前線の指紋を照らして
   * 前線の腕の網として開け、行の O から前線を書いた塊で s が有限。
   */
  @Test(timeOut = 600_000)
  public void refitRunsTheFrontierArm() throws Exception {
    PrismConfig config =
        PrismConfig.fromSettings(
            EpsilonSettings.of(
                Map.of(
                    PrismConfig.SETTINGS + "learnerDevice", "cpu",
                    PrismConfig.SETTINGS + "decodeThreads", "2",
                    PrismConfig.SETTINGS + "batchSize", "64")));
    Files.writeString(
        root.resolve("config.json"),
        new GsonBuilder().serializeSpecialFloatingPointValues().create().toJson(config));
    Path snapshot =
        new FitWorkflow()
            .execute(
                new WorkflowArguments(
                    "refit",
                    new String[] {
                      "--rows", root.toString(),
                      "--cohorts", "1-2",
                      "--depth", "2",
                      "--aux-result", "0.1",
                      "--frontier", "on",
                      "--out", root.resolve("refit").toString()
                    },
                    EpsilonSettings.defaults()));
    try (CohortStore store = new CohortStore();
        NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Snapshot.Loaded loaded = Snapshot.load(snapshot, manager);
      FitPlan plan = loaded.fit().plan();
      FitResult result = loaded.fit().result();
      log.info(
          "PRISM-FRONTIER rows={} steps={} loss {} -> {} {}s",
          result.rows(),
          result.steps(),
          String.format("%.4f", result.firstLoss()),
          String.format("%.4f", result.lastLoss()),
          String.format("%.1f", result.seconds()));
      assertEquals(loaded.network().shape(), PrismShape.standard(2).withFrontier(true));
      assertEquals(plan.shape(), loaded.network().shape());
      assertTrue(
          result.steps() > 1 && Double.isFinite(result.loss()),
          "steps " + result.steps() + " loss " + result.loss());
      store.addCohort(CohortWriter.directory(root, 1));
      store.addCohort(CohortWriter.directory(root, 2));
      int[] rows = store.select(1, 2, Selection.LABELS);
      try (StepFeed feed = new StepFeed(store, plan, Teacher.RAW, 1, 2)) {
        StepFeed.Step step = feed.prepare(Arrays.copyOf(rows, 40)).join();
        for (Microbatch part : step.lanes().get(0)) {
          assertTrue(written(part.batch().frontier()), "the frontier features were not written");
          try (NDManager scope = manager.newSubManager()) {
            for (float score : scores(loaded.network(), part, scope)) {
              assertTrue(Float.isFinite(score), "score " + score);
            }
          }
        }
        feed.release(step);
      }
    }
  }

  /** 前線の塊に 0 でない値がある（符号化が前線を書いた）。 */
  private static boolean written(FrontierBatch frontier) {
    ByteBuffer blob = frontier.blob();
    for (int at = 0; at < frontier.bytes(); at += Short.BYTES) {
      if (blob.getShort(at) != 0) {
        return true;
      }
    }
    return false;
  }

  /** 当てた網と開いた網の s がビットで同じ。 */
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
            part.inputs(scope, Uploads.on(scope)),
            Pass.inference(Precision.FLOAT32))
        .scores()
        .toFloatArray();
  }

  /** 補助の教師の形が網の補助の頭の出力と合い、損失が有限。 */
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
              part.inputs(scope, Uploads.on(scope)),
              Pass.training(Precision.FLOAT32, part.measured(), labels));
      assertEquals(
          outputs.auxiliary().candidates().getShape(),
          new Shape(candidates, CandidateTarget.WIDTH));
      assertEquals(
          outputs.auxiliary().reading().getShape(),
          new Shape(readings, AuxiliaryLabels.OPPONENTS, ReadingTarget.WIDTH));
      float total =
          AuxiliaryLoss.standard().terms(scope, outputs.auxiliary(), labels, 1f).total().getFloat();
      assertTrue(Float.isFinite(total) && total > 0, "auxiliary loss " + total);
    }
  }

  @Test(timeOut = 600_000)
  public void readingTargetsFollowTheInputSuitMap() {
    FitPlan plan =
        new FitPlan(
            new FitSettings(1, 1, 1e-4, 1e-5, 1f, 0.01, 0.02),
            SHAPE,
            AuxiliaryLoss.standard(),
            BUDGET,
            1);
    Side left = new Side(plan);
    Side right = new Side(plan);
    int[] counts = new int[3];
    for (int view = 1; view < SuitSymmetry.VIEWS; view++) {
      new DecisionWalk(DecisionWalk.Policy.GREEDY)
          .runMirrored(
              600 + 100L * view,
              1,
              view,
              (engine, point, mirror, mirrorPoint) -> {
                AuxiliaryLabels a = left.encode(engine, point);
                AuxiliaryLabels b = right.encode(mirror, mirrorPoint);
                if (!Arrays.equals(left.tiles(), right.tiles())) {
                  counts[2]++;
                  return;
                }
                assertTrue(
                    Arrays.equals(a.readingValues(), b.readingValues()),
                    "reading targets differ at seat " + point.player());
                counts[0]++;
                if (left.informative && left.physicalTiles != right.physicalTiles) {
                  counts[1]++;
                }
              });
    }
    log.info("PRISM-AUX compared={} physicallyDifferent={} inputTies={}", counts[0], counts[1], counts[2]);
    assertTrue(counts[0] > 1_000, "compared decisions " + counts[0]);
    assertTrue(counts[1] > 50, "pairs whose physical waits or dangers differ " + counts[1]);
    assertTrue(counts[2] * 20 < counts[0], "input ties " + counts[2]);
  }

  /** 色の対の片側: 判断を 1 行の塊に書き、補助の教師を作る。 */
  private static final class Side {
    private final Encoder encoder;
    private final ObservationWriter writer = new ObservationWriter();
    private final ObservationArena arena = new ObservationArena(256);
    private final Observation observation = new Observation();
    private final OracleCapture oracles = new OracleCapture();
    private final HostInputs batch =
        HostInputs.allocate(ONE_ROW, false, ByteBuffer::allocateDirect);
    private boolean informative;
    private long physicalTiles;

    Side(FitPlan plan) {
      encoder = new Encoder(plan, Teacher.RAW);
    }

    AuxiliaryLabels encode(GameEngine engine, EngineDecisionPoint point) {
      int seat = point.player();
      arena.clear();
      int base = writer.capture(engine.observation(seat), point.kind(), point.legalActions(), arena);
      long[] words = arena.words();
      long[] o = Arrays.copyOfRange(words, base, base + ObservationSchema.longs(words, base));
      long[] oracle = oracles.capture(engine.getState(), seat);
      int actions = observation.bind(o, 0).legalActions();
      // 物理の待ちと放銃の牌を全ての相手で重ねる（色の写しが効いた対を数える）
      physicalTiles = 0L;
      for (int opponent = 1; opponent <= AuxiliaryLabels.OPPONENTS; opponent++) {
        physicalTiles |=
            OracleCodec.waits(oracle, 0, opponent) | OracleCodec.danger(oracle, 0, opponent);
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
              Row.Group.A,
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
      return encoder.encode(batch, List.of(row)).auxiliary();
    }

    /** 入力の牌のトークンの欄（色の写しの後）。 */
    byte[] tiles() {
      BatchPlan plan = batch.plan();
      byte[] bytes = new byte[plan.count(Axis.TILE) * Tensor.TILE_CATEGORY.width()];
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
}
