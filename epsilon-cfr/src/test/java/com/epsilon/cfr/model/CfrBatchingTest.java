package com.epsilon.cfr.model;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.network.NetworkDevices;
import com.epsilon.network.NetworkRuntime;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.training.DataParallelGroup;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/** 候補の容量区分ごとに学習しても、損失を合算した場合と全勾配・AdamW更新が一致することを検証する。 */
public class CfrBatchingTest {
  @Test(groups = "native")
  public void mixedBucketsPreserveObjectivesAndUpdates() throws Exception {
    verify(Device.cpu(), DecisionComputePrecision.FLOAT32, 3e-5f, 4e-5f);
  }

  @Test(groups = "rocm")
  public void mixedBucketsPreserveUpdatesOnGpu() throws Exception {
    NetworkRuntime.initialize();
    if (Engine.getInstance().getGpuCount() == 0) throw new SkipException("GPU required");
    verify(Device.gpu(), DecisionComputePrecision.FLOAT32, 2e-4f, 1e-4f);
  }

  @Test(groups = "rocm")
  public void mixedBucketsPreserveUpdatesWithBfloat16() throws Exception {
    NetworkRuntime.initialize();
    if (Engine.getInstance().getGpuCount() == 0) throw new SkipException("GPU required");
    verify(Device.gpu(), DecisionComputePrecision.BFLOAT16, .02f, .0005f);
  }

  private static void verify(
      Device device, DecisionComputePrecision precision, float tolerance, float updateTolerance)
      throws Exception {
    List<CfrTrainingRow> rows = rows();
    var config =
        new CfrModelConfig(
            device.isGpu() ? 256 : 16,
            4,
            .001f,
            .01f,
            .002f,
            1337,
            4,
            precision,
            DecisionTensorTransfer.PINNED_BUFFER,
            EpsilonUtilityProfile.TENHOU);
    boolean bfloat16 = precision == DecisionComputePrecision.BFLOAT16;
    for (var role : CfrModel.Role.values()) {
      try (var expected = new Fixture(config, role, device);
          var actual = new Fixture(config, role, device);
          var execution = new DecisionExecutionContext();
          var reference =
              new CfrOptimizerGroup<>(
                  List.of(new TestLane(expected.model, execution)),
                  execution,
                  config.gradientClip())) {
        DataParallelGroup.copyParameters(expected.model, actual.model);
        var initial = bfloat16 ? snapshot(expected.network) : null;
        reference.clearTrainingGradients();
        Backward want =
            reference
                .<Backward>executeWork(List.of(lane -> backward(expected, rows, false)))
                .getFirst();
        Backward shared = backward(actual, rows, true);
        Assert.assertEquals(shared.loss(), want.loss(), tolerance);
        Assert.assertEquals(shared.logits().keySet(), want.logits().keySet());
        for (var entry : want.logits().entrySet())
          close(shared.logits().get(entry.getKey()), entry.getValue(), tolerance, "logits");
        Assert.assertEquals(shared.gradients().keySet(), want.gradients().keySet());
        double stateGradient = 0;
        var gradientDifference = new VectorDifference();
        for (var entry : want.gradients().entrySet()) {
          close(
              shared.gradients().get(entry.getKey()), entry.getValue(), tolerance, entry.getKey());
          gradientDifference.add(shared.gradients().get(entry.getKey()), entry.getValue(), null);
          if (entry.getKey().contains("stateEncoder"))
            for (float value : entry.getValue()) stateGradient += Math.abs(value);
        }
        Assert.assertTrue(stateGradient > 0, "State encoder gradients must be nontrivial");
        if (bfloat16) gradientDifference.assertSimilar(.01, .9999, role + " full gradients");
        clear(actual.network);
        var optimizer =
            Optimizer.adamW()
                .optLearningRateTracker(Tracker.fixed(config.learningRate()))
                .optWeightDecays(config.weightDecay())
                .build();
        for (int step = 1; step <= 3; step++) {
          if (step > 1)
            want =
                reference
                    .<Backward>executeWork(List.of(lane -> backward(expected, rows, false)))
                    .getFirst();
          reference.applyFlattenedOptimizerStep(optimizer);
          reference.awaitParameterUpdates();
          var got = actual.learner.train(rows, step);
          actual.learner.awaitModelReady();
          Assert.assertEquals(got.samples(), rows.size());
          String label = role + " step=" + step;
          // BF16の行列積では候補マイクロバッチの形状で加算結果が変わるため、損失と勾配ノルムの
          // 相対誤差を許容する。FP32では従来どおり、後続の検証も含めて各要素を厳密に比較する。
          Assert.assertEquals(
              got.loss(),
              want.loss(),
              bfloat16 ? 2e-4 + .01 * Math.abs(want.loss()) : tolerance,
              label + " loss");
          Assert.assertEquals(
              got.gradientNorm(),
              reference.gradientNorm(),
              bfloat16
                  ? 2e-4 + .02 * reference.gradientNorm()
                  : tolerance * Math.max(1, reference.gradientNorm()),
              label + " gradientNorm");
          Assert.assertTrue(reference.gradientNorm() > config.gradientClip());
          var parameterDifference = new VectorDifference();
          var updateDifference = new VectorDifference();
          for (var entry : expected.network.getParameters()) {
            float[] found = floats(actual.network.getParameters().get(entry.getKey()).getArray());
            float[] wanted = floats(entry.getValue().getArray());
            close(
                found,
                wanted,
                bfloat16 ? .5f * step * config.learningRate() : updateTolerance,
                label + " AdamW " + entry.getKey());
            if (bfloat16) {
              parameterDifference.add(found, wanted, null);
              updateDifference.add(found, wanted, initial.get(entry.getKey()));
            }
          }
          if (bfloat16) {
            // AdamWは一部の成分で、ゼロに近い勾配の丸め誤差を増幅することがある。パラメーター全体の
            // 差に加え、同じ初期重みからの累積更新量と方向も検証し、未更新の重みでは通過しないようにする。
            parameterDifference.assertSimilar(5e-4, .99999, label + " parameters");
            updateDifference.assertSimilar(.03, .999, label + " cumulative AdamW update");
          }
        }
      }
    }
  }

  private static Backward backward(
      Fixture fixture, List<CfrTrainingRow> rows, boolean combineObjectives) {
    var entries = CfrSampleMultiplicity.aggregate(rows);
    var observations =
        entries.stream()
            .map(entry -> CfrObservationCodec.decode(entry.row().observation()))
            .toList();
    var buckets = new LinkedHashMap<DecisionBucket, List<Integer>>();
    for (int row = 0; row < entries.size(); row++)
      buckets
          .computeIfAbsent(observations.get(row).host().bucket(), ignored -> new ArrayList<>())
          .add(row);
    Assert.assertTrue(buckets.size() > 1, "Exercise multiple action/transition shapes");
    var logits = new LinkedHashMap<CfrSampleId, float[]>();
    double lossValue = 0;
    try (NDManager scope = fixture.model.getNDManager().newSubManager();
        var collector = scope.getEngine().newGradientCollector()) {
      var store = new ParameterStore(scope, true);
      NDArray total = null;
      for (List<Integer> indices : buckets.values()) {
        var selected = indices.stream().map(observations::get).toList();
        var host = CfrObservationCodec.batch(selected);
        var batch = DecisionBatchTransfer.transferTrainingToDevice(scope, host);
        NDArray scores;
        try (Autocast ignored =
            EpsilonDecisionAutocast.open(scope, fixture.config.precision(scope.getDevice()))) {
          scores = fixture.network.scores(store, batch, true);
        }
        if (scores.getDataType() != DataType.FLOAT32)
          scores = scores.add(scope.zeros(new Shape(1), DataType.FLOAT32));
        int capacity = host.bucket().legalActionCapacity();
        float[] flat = scores.toFloatArray();
        float[] targets = new float[indices.size() * capacity];
        float[] counts = new float[indices.size()];
        for (int row = 0; row < indices.size(); row++) {
          var entry = entries.get(indices.get(row));
          var teacher =
              fixture.role == CfrModel.Role.REGRET
                  ? entry.row().regretTargets()
                  : entry.row().targetPolicy();
          for (int action = 0; action < teacher.length; action++)
            targets[row * capacity + action] = (float) teacher[action];
          counts[row] = entry.count();
          logits.put(
              entry.row().id(),
              Arrays.copyOfRange(flat, row * capacity, row * capacity + teacher.length));
        }
        NDArray loss =
            CfrLearner.objective(
                fixture.role,
                scores,
                scope.create(targets, new Shape(indices.size(), capacity)),
                batch.legalActionMask().toType(DataType.FLOAT32, false),
                scope.create(counts, new Shape(indices.size(), 1)),
                rows.size());
        lossValue += loss.getFloat();
        if (combineObjectives) total = total == null ? loss : total.add(loss);
        else collector.backward(loss);
      }
      if (combineObjectives) collector.backward(total);
    }
    var gradients = new LinkedHashMap<String, float[]>();
    for (var entry : fixture.network.getParameters()) {
      NDArray parameter = entry.getValue().getArray();
      gradients.put(
          entry.getKey(),
          parameter.hasGradient()
              ? floats(parameter.getGradient())
              : new float[Math.toIntExact(parameter.size())]);
    }
    return new Backward(lossValue, logits, gradients);
  }

  private static List<CfrTrainingRow> rows() {
    var contexts = List.of(CfrModelTest.context(981), reaction());
    int[] multiplicities = {3, 1, 2, 4, 1, 2};
    List<CfrTrainingRow> result = new ArrayList<>();
    for (int index = 0; index < multiplicities.length; index++) {
      var base = contexts.get(index % 2);
      var context =
          new DecisionContext(
              index, base.decisionId(), base.seat(), base.observation(), base.legalActions());
      int[] actions = base.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
      double[] policy = new double[actions.length];
      Arrays.fill(policy, .4 / policy.length);
      policy[0] += .6;
      double[] explicit = index == 2 ? new double[actions.length] : null;
      if (explicit != null)
        for (int action = 0; action < explicit.length; action++)
          explicit[action] = (action % 3 - 1) * .3;
      var row =
          new CfrTrainingRow(
              new CfrSampleId(1, index, base.decisionId(), base.seat()),
              actions,
              actions.length - 1,
              policy,
              index % 2 == 0 ? .7 : -.4,
              70 + index,
              -90 - index,
              CfrObservationCodec.encodePayload(context),
              explicit);
      for (int copy = 0; copy < multiplicities[index]; copy++) result.add(row);
    }
    return result;
  }

  private static DecisionContext reaction() {
    var engine = new GameEngine(770);
    GameStepResult step = engine.stepHanchan();
    for (int count = 0; count < 300 && !(step instanceof GameStepResult.HanchanEnded); count++) {
      if (step instanceof GameStepResult.RoundSettled) {
        step = engine.stepHanchan();
        continue;
      }
      var selections = new EngineSelectionBuffer();
      for (var point : ((GameStepResult.AwaitingDecisions) step).decisions()) {
        if (DecisionBatchBuilder.selectInferenceBucket(point.legalActions())
                .actionTransitionCapacity()
            > 1)
          return new DecisionContext(
              770,
              point.id(),
              point.player(),
              engine.observation(point.player()),
              point.legalActions());
        selections.add(point.id(), point.legalActions().getFirst());
      }
      step = engine.commitDecisions(selections);
    }
    throw new AssertionError("No multi-transition fixture");
  }

  private static float[] floats(NDArray array) {
    return array.toType(DataType.FLOAT32, false).toFloatArray();
  }

  private static void clear(CfrNetwork network) {
    for (var entry : network.getParameters()) {
      NDArray parameter = entry.getValue().getArray();
      if (parameter.hasGradient()) parameter.getGradient().fillI(0);
    }
  }

  private static void close(float[] actual, float[] expected, float tolerance, String label) {
    Assert.assertEquals(actual.length, expected.length);
    for (int index = 0; index < actual.length; index++)
      Assert.assertEquals(
          actual[index],
          expected[index],
          tolerance * Math.max(1, Math.abs(expected[index])),
          label + "[" + index + "]");
  }

  private static Map<String, float[]> snapshot(CfrNetwork network) {
    var result = new LinkedHashMap<String, float[]>();
    for (var entry : network.getParameters())
      result.put(entry.getKey(), floats(entry.getValue().getArray()));
    return result;
  }

  static final class VectorDifference {
    private double squaredError, squaredActual, squaredExpected, dot;

    void add(float[] actual, float[] expected, float[] origin) {
      Assert.assertEquals(actual.length, expected.length);
      for (int index = 0; index < actual.length; index++) {
        double initial = origin == null ? 0 : origin[index];
        double a = actual[index] - initial, b = expected[index] - initial;
        squaredError += (a - b) * (a - b);
        squaredActual += a * a;
        squaredExpected += b * b;
        dot += a * b;
      }
    }

    void assertSimilar(double relativeTolerance, double minimumCosine, String label) {
      String detail = summary(label);
      Assert.assertTrue(relativeError() <= relativeTolerance, detail);
      Assert.assertTrue(cosine() >= minimumCosine, detail);
    }

    double relativeError() {
      return Math.sqrt(squaredError / Math.max(1e-30, squaredExpected));
    }

    double cosine() {
      return dot / Math.sqrt(squaredActual * squaredExpected);
    }

    String summary(String label) {
      return label + " relativeL2=" + relativeError() + " cosine=" + cosine();
    }
  }

  private record Backward(
      double loss, Map<CfrSampleId, float[]> logits, Map<String, float[]> gradients) {}

  private static final class TestLane extends DataParallelGroup.Lane {
    TestLane(Model model, DecisionExecutionContext execution) {
      super(model, false, false, execution);
    }
  }

  private static final class Fixture implements AutoCloseable {
    final CfrModelConfig config;
    final CfrModel.Role role;
    final Model model;
    final CfrNetwork network;
    final CfrLearner learner;

    Fixture(CfrModelConfig config, CfrModel.Role role, Device device) {
      NetworkRuntime.initialize();
      this.config = config;
      this.role = role;
      model = Model.newInstance("cfr-batch-test", device, "PyTorch");
      model.getNDManager().getEngine().setRandomSeed(1337);
      network = new CfrNetwork(config);
      model.setBlock(network);
      network.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, 1));
      for (var parameter : network.getParameters().values()) {
        if (network.isOutputParameter(parameter)) {
          float[] values = new float[Math.toIntExact(parameter.getArray().size())];
          for (int index = 0; index < values.length; index++)
            values[index] = (float) Math.sin(index + 1) * .03f;
          parameter.getArray().set(values);
        }
      }
      learner = new CfrLearner(config, role, model, network, NetworkDevices.of(device));
    }

    @Override
    public void close() {
      learner.close();
      model.close();
    }
  }
}
