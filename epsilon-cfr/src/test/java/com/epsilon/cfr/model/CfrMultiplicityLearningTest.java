package com.epsilon.cfr.model;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.network.NetworkDevices;
import com.epsilon.network.NetworkRuntime;
import com.epsilon.training.DataParallelGroup;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import org.testng.Assert;
import org.testng.Reporter;
import org.testng.SkipException;
import org.testng.annotations.Test;

/** 出現数をまとめた教師データでも、重複行を展開した場合と目的関数・勾配・AdamW更新が一致することを検証する。 */
public class CfrMultiplicityLearningTest {
  @Test(groups = "native")
  public void weightedLossAndScoreGradientsMatchExpandedRowsForBothRoles() {
    NetworkRuntime.initialize();
    int[] source = {0, 0, 0, 1, 1, 2};
    float[] logits = {.2f, -.3f, .8f, -2, -.6f, .4f, 1.2f, -1, .1f, .9f, -.5f, 2};
    float[] mask = {1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1};
    for (var role : CfrModel.Role.values()) {
      float[] targets =
          role == CfrModel.Role.REGRET
              ? new float[] {.7f, -.2f, -.5f, 99, -.4f, .4f, 99, 99, -.1f, .8f, -.3f, -.4f}
              : new float[] {.6f, .1f, .3f, 0, .2f, .8f, 0, 0, .1f, .2f, .6f, .1f};
      try (var manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
        NDArray uniqueScores = manager.create(logits, new Shape(3, 4));
        NDArray expandedScores = manager.create(expand(logits, source), new Shape(6, 4));
        uniqueScores.setRequiresGradient(true);
        expandedScores.setRequiresGradient(true);
        NDArray weightedLoss;
        NDArray expandedLoss;
        try (var collector = manager.getEngine().newGradientCollector()) {
          weightedLoss =
              CfrLearner.objective(
                  role,
                  uniqueScores,
                  manager.create(targets, new Shape(3, 4)),
                  manager.create(mask, new Shape(3, 4)),
                  manager.create(new float[] {3, 2, 1}, new Shape(3, 1)),
                  source.length);
          collector.backward(weightedLoss);
        }
        try (var collector = manager.getEngine().newGradientCollector()) {
          expandedLoss =
              CfrLearner.objective(
                  role,
                  expandedScores,
                  manager.create(expand(targets, source), new Shape(6, 4)),
                  manager.create(expand(mask, source), new Shape(6, 4)),
                  manager.ones(new Shape(6, 1)),
                  source.length);
          collector.backward(expandedLoss);
        }
        Assert.assertEquals(weightedLoss.getFloat(), expandedLoss.getFloat(), 2e-6f, role.name());
        float[] weightedGradient = uniqueScores.getGradient().toFloatArray();
        float[] expandedGradient = expandedScores.getGradient().toFloatArray();
        float[] summed = new float[weightedGradient.length];
        for (int row = 0; row < source.length; row++)
          for (int action = 0; action < 4; action++)
            summed[source[row] * 4 + action] += expandedGradient[row * 4 + action];
        assertClose(weightedGradient, summed, 2e-6f, role.name() + " gradient");
      }
    }
  }

  @Test(groups = "native")
  public void collapsedDuplicatesPreserveClippedAdamUpdatesAndOriginalSampleCounts() {
    List<CfrTrainingRow> unique = List.of(row(981, .8), row(982, -.4), row(983, .3));
    List<CfrTrainingRow> repeated =
        List.of(
            unique.get(0),
            unique.get(1),
            unique.get(0),
            unique.get(2),
            unique.get(0),
            unique.get(1),
            unique.get(0),
            unique.get(0));
    List<CfrTrainingRow> expanded = new ArrayList<>();
    for (int index = 0; index < repeated.size(); index++) {
      CfrTrainingRow source = repeated.get(index);
      expanded.add(
          new CfrTrainingRow(
              new CfrSampleId(1, 10000 + index, 0, source.id().seat()),
              source.legalActionIds(),
              source.selectedSlot(),
              source.targetPolicy(),
              source.utility(),
              source.logRegretWeight(),
              source.logAverageWeight(),
              source.observation()));
    }
    var config =
        new CfrModelConfig(
            16,
            2,
            .002f,
            .01f,
            .001f,
            1337,
            2,
            DecisionComputePrecision.FLOAT32,
            DecisionTensorTransfer.DIRECT_BUFFER,
            EpsilonUtilityProfile.TENHOU);
    for (var role : CfrModel.Role.values()) {
      try (var collapsed = new Fixture(config, role);
          var full = new Fixture(config, role)) {
        DataParallelGroup.copyParameters(collapsed.model, full.model);
        collapsed.learner.train(unique, 1);
        full.learner.train(unique, 1);
        for (int update = 2; update <= 4; update++) {
          var got = collapsed.learner.train(repeated, update);
          var want = full.learner.train(expanded, update);
          Assert.assertEquals(got.samples(), repeated.size());
          Assert.assertEquals(got.updates(), update);
          Assert.assertEquals(got.loss(), want.loss(), 2e-5, role.name());
          Assert.assertEquals(
              got.gradientNorm(),
              want.gradientNorm(),
              2e-5 * Math.max(1, want.gradientNorm()),
              role.name());
          Assert.assertTrue(got.gradientNorm() > config.gradientClip(), "exercise clipping");
          collapsed.learner.awaitModelReady();
          full.learner.awaitModelReady();
          for (var entry : collapsed.network.getParameters()) {
            assertClose(
                entry.getValue().getArray().toFloatArray(),
                full.network.getParameters().get(entry.getKey()).getArray().toFloatArray(),
                3e-5f,
                role.name() + " " + entry.getKey());
          }
        }
      }
    }
  }

  @Test(groups = "native")
  public void failedLaterMicrobatchDoesNotLeakGradientsIntoTheNextUpdate() {
    CfrTrainingRow valid = row(981, .8);
    CfrTrainingRow invalidTarget = row(981, Double.MAX_VALUE);
    CfrTrainingRow invalid =
        new CfrTrainingRow(
            new CfrSampleId(1, 9981, 0, invalidTarget.id().seat()),
            invalidTarget.legalActionIds(),
            invalidTarget.selectedSlot(),
            invalidTarget.targetPolicy(),
            invalidTarget.utility(),
            invalidTarget.logRegretWeight(),
            invalidTarget.logAverageWeight(),
            invalidTarget.observation());
    var config =
        new CfrModelConfig(
            16,
            1,
            .002f,
            .01f,
            .001f,
            1337,
            1,
            DecisionComputePrecision.FLOAT32,
            DecisionTensorTransfer.DIRECT_BUFFER,
            EpsilonUtilityProfile.TENHOU);
    try (var recovered = new Fixture(config, CfrModel.Role.REGRET);
        var control = new Fixture(config, CfrModel.Role.REGRET)) {
      DataParallelGroup.copyParameters(control.model, recovered.model);
      // 同じ容量区分の入力順を保ち、有効な教師の逆伝播を終えた後に、不正な教師で失敗する。
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> recovered.learner.train(List.of(valid, invalid), 1));
      for (int update = 1; update <= 3; update++) {
        var actual = recovered.learner.train(List.of(valid), update);
        var expected = control.learner.train(List.of(valid), update);
        Assert.assertEquals(actual.loss(), expected.loss(), 1e-7);
        Assert.assertEquals(actual.gradientNorm(), expected.gradientNorm(), 1e-7);
        recovered.learner.awaitModelReady();
        control.learner.awaitModelReady();
        for (var entry : recovered.network.getParameters()) {
          assertClose(
              entry.getValue().getArray().toFloatArray(),
              control.network.getParameters().get(entry.getKey()).getArray().toFloatArray(),
              1e-7f,
              entry.getKey());
        }
      }
    }
  }

  @Test(groups = "native")
  public void cachedRowsAndSameIdReplacementsMatchFreshDecodingForBothRoles() {
    CfrTrainingRow first = row(981, .8);
    CfrTrainingRow next = row(982, -.4);
    double[] nextPolicy = new double[next.actionCount()];
    Arrays.fill(nextPolicy, 1d / nextPolicy.length);
    CfrTrainingRow replacement =
        new CfrTrainingRow(
            first.id(),
            next.legalActionIds(),
            next.selectedSlot(),
            nextPolicy,
            next.utility(),
            next.logRegretWeight(),
            next.logAverageWeight(),
            next.observation());
    var config =
        new CfrModelConfig(
            16,
            1,
            .002f,
            .01f,
            .001f,
            1337,
            1,
            DecisionComputePrecision.FLOAT32,
            DecisionTensorTransfer.DIRECT_BUFFER,
            EpsilonUtilityProfile.TENHOU);
    for (var role : CfrModel.Role.values()) {
      try (var cached = new Fixture(config, role);
          var decoded = new Fixture(config, role)) {
        DataParallelGroup.copyParameters(cached.model, decoded.model);
        for (int update = 1; update <= 4; update++) {
          CfrTrainingRow row = update < 3 ? first : replacement;
          var actual = cached.learner.train(List.of(row), update);
          var expected = decoded.learner.train(List.of(copyWithId(row, row.id())), update);
          Assert.assertEquals(actual.loss(), expected.loss(), 1e-7, role.name());
          Assert.assertEquals(actual.gradientNorm(), expected.gradientNorm(), 1e-7, role.name());
          cached.learner.awaitModelReady();
          decoded.learner.awaitModelReady();
          for (var entry : cached.network.getParameters()) {
            assertClose(
                entry.getValue().getArray().toFloatArray(),
                decoded.network.getParameters().get(entry.getKey()).getArray().toFloatArray(),
                1e-7f,
                role.name() + " " + entry.getKey());
          }
        }
      }
    }
  }

  @Test(groups = "rocm")
  public void cachedBoardInputsMatchFreshDecodingWithBfloat16() throws IOException {
    NetworkRuntime.initialize();
    if (Engine.getInstance().getGpuCount() == 0) throw new SkipException("GPU required");
    CfrTrainingRow first = row(981, .8);
    List<CfrTrainingRow> retained = List.of(row(982, -.4), row(983, .3), row(984, -.2));
    CfrTrainingRow next = row(985, -.65);
    double[] nextPolicy = new double[next.actionCount()];
    Arrays.fill(nextPolicy, 1d / nextPolicy.length);
    CfrTrainingRow replacement =
        new CfrTrainingRow(
            first.id(),
            next.legalActionIds(),
            next.selectedSlot(),
            nextPolicy,
            next.utility(),
            next.logRegretWeight(),
            next.logAverageWeight(),
            next.observation());
    var config =
        new CfrModelConfig(
            256,
            2,
            .001f,
            .01f,
            .002f,
            1337,
            4,
            DecisionComputePrecision.BFLOAT16,
            DecisionTensorTransfer.PINNED_BUFFER,
            EpsilonUtilityProfile.TENHOU);
    for (var role : CfrModel.Role.values()) {
      var optimizerState = Files.createTempFile("cfr-cache-equivalence-", ".bin");
      try (var cached = new Fixture(config, role, Device.gpu());
          var decoded = new Fixture(config, role, Device.gpu())) {
        // 初回から盤面表現へ勾配を返し、出力層だけの一致では検証を通過させない。
        for (var parameter : cached.network.getParameters().values()) {
          if (cached.network.isOutputParameter(parameter)) {
            float[] values = new float[Math.toIntExact(parameter.getArray().size())];
            for (int index = 0; index < values.length; index++)
              values[index] = (float) Math.sin(index + 1) * .03f;
            parameter.getArray().set(values);
          }
        }
        DataParallelGroup.copyParameters(cached.model, decoded.model);
        double maximumLossDifference = 0, maximumNormDifference = 0;
        double maximumParameterError = 0, maximumUpdateError = 0, minimumUpdateCosine = 1;
        for (int update = 1; update <= 6; update++) {
          // GPUの丸め差を別々の学習経路で増幅させず、同じ重みとAdamW状態からの一更新を比較する。
          // 学習を続ける側の復号キャッシュは保持し、再利用と同ID差替えの双方を繰り返し検証する。
          if (update > 1) {
            cached.learner.saveOptimizer(optimizerState);
            decoded.learner.synchronizeParameters();
            DataParallelGroup.copyParameters(cached.model, decoded.model);
            decoded.learner.loadOptimizer(optimizerState);
          }
          var initial = new LinkedHashMap<String, float[]>();
          for (var entry : cached.network.getParameters())
            initial.put(entry.getKey(), entry.getValue().getArray().toFloatArray());
          CfrTrainingRow active = update < 4 ? first : replacement;
          var rows =
              List.of(
                  active,
                  retained.get(0),
                  active,
                  retained.get(1),
                  retained.get(2),
                  retained.get(0));
          var fresh = rows.stream().map(row -> copyWithId(row, row.id())).toList();
          var actual = cached.learner.train(rows, update);
          cached.learner.awaitModelReady();
          var expected = decoded.learner.train(fresh, update);
          decoded.learner.awaitModelReady();
          String label = role + " cached BF16 step=" + update;
          Assert.assertEquals(actual.samples(), rows.size());
          // 候補バッチの共有を検証する既存BF16テストと同じ基準を、一更新の比較へ適用する。
          Assert.assertEquals(
              actual.loss(),
              expected.loss(),
              2e-4 + .01 * Math.abs(expected.loss()),
              label + " loss");
          Assert.assertEquals(
              actual.gradientNorm(),
              expected.gradientNorm(),
              2e-4 + .02 * expected.gradientNorm(),
              label + " gradientNorm");
          Assert.assertTrue(expected.gradientNorm() > config.gradientClip(), "Exercise clipping");
          var parameters = new CfrBatchingTest.VectorDifference();
          var updates = new CfrBatchingTest.VectorDifference();
          double stateMovement = 0;
          for (var entry : cached.network.getParameters()) {
            float[] found = entry.getValue().getArray().toFloatArray();
            float[] wanted =
                decoded.network.getParameters().get(entry.getKey()).getArray().toFloatArray();
            float[] origin = initial.get(entry.getKey());
            assertClose(found, wanted, .5f * config.learningRate(), label + " " + entry.getKey());
            parameters.add(found, wanted, null);
            updates.add(found, wanted, origin);
            if (entry.getKey().contains("stateEncoder"))
              for (int index = 0; index < found.length; index++)
                stateMovement += Math.abs(found[index] - origin[index]);
          }
          Assert.assertTrue(stateMovement > 0, label + " state encoder must be updated");
          parameters.assertSimilar(5e-4, .99999, label + " parameters");
          updates.assertSimilar(.03, .999, label + " AdamW update");
          maximumLossDifference =
              Math.max(maximumLossDifference, Math.abs(actual.loss() - expected.loss()));
          maximumNormDifference =
              Math.max(
                  maximumNormDifference, Math.abs(actual.gradientNorm() - expected.gradientNorm()));
          maximumParameterError = Math.max(maximumParameterError, parameters.relativeError());
          maximumUpdateError = Math.max(maximumUpdateError, updates.relativeError());
          minimumUpdateCosine = Math.min(minimumUpdateCosine, updates.cosine());
          Reporter.log(parameters.summary(label + " parameters"), true);
          Reporter.log(updates.summary(label + " AdamW update"), true);
        }
        Reporter.log(
            "CFR cached BF16 equivalence role="
                + role
                + " maxLossDifference="
                + maximumLossDifference
                + " maxGradientNormDifference="
                + maximumNormDifference
                + " maxParameterRelativeL2="
                + maximumParameterError
                + " maxUpdateRelativeL2="
                + maximumUpdateError
                + " minUpdateCosine="
                + minimumUpdateCosine,
            true);
      } finally {
        Files.deleteIfExists(optimizerState);
      }
    }
  }

  private static CfrTrainingRow copyWithId(CfrTrainingRow row, CfrSampleId id) {
    return new CfrTrainingRow(
        id,
        row.legalActionIds(),
        row.selectedSlot(),
        row.targetPolicy(),
        row.utility(),
        row.logRegretWeight(),
        row.logAverageWeight(),
        row.observation(),
        row.explicitRegretTargets());
  }

  private static float[] expand(float[] source, int[] rowIds) {
    float[] result = new float[rowIds.length * 4];
    for (int row = 0; row < rowIds.length; row++)
      System.arraycopy(source, rowIds[row] * 4, result, row * 4, 4);
    return result;
  }

  private static CfrTrainingRow row(long seed, double utility) {
    var context = CfrModelTest.context(seed);
    int[] actions = context.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
    double[] policy = new double[actions.length];
    Arrays.fill(policy, .4 / actions.length);
    policy[0] += .6;
    return new CfrTrainingRow(
        new CfrSampleId(1, seed, context.decisionId(), context.seat()),
        actions,
        actions.length - 1,
        policy,
        utility,
        70 + seed,
        -90 - seed,
        CfrObservationCodec.encodePayload(context));
  }

  private static void assertClose(float[] actual, float[] expected, float tolerance, String label) {
    Assert.assertEquals(actual.length, expected.length);
    for (int index = 0; index < actual.length; index++)
      Assert.assertEquals(actual[index], expected[index], tolerance, label + " [" + index + "]");
  }

  private static final class Fixture implements AutoCloseable {
    final Model model;
    final CfrNetwork network;
    final CfrLearner learner;

    Fixture(CfrModelConfig config, CfrModel.Role role) {
      this(config, role, Device.cpu());
    }

    Fixture(CfrModelConfig config, CfrModel.Role role, Device device) {
      NetworkRuntime.initialize();
      model = Model.newInstance("cfr-multiplicity-test", device, "PyTorch");
      model.getNDManager().getEngine().setRandomSeed(1337);
      network = new CfrNetwork(config);
      model.setBlock(network);
      network.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, 1));
      learner = new CfrLearner(config, role, model, network, NetworkDevices.of(device));
    }

    @Override
    public void close() {
      learner.close();
      model.close();
    }
  }
}
