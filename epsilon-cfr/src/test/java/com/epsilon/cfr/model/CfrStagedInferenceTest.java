package com.epsilon.cfr.model;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.runtime.CfrPolicyFleet;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.network.NetworkDevices;
import com.epsilon.spi.PolicyExecutionContext;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class CfrStagedInferenceTest {
  @DataProvider
  public Object[][] roles() {
    return new Object[][] {{CfrModel.Role.AVERAGE}, {CfrModel.Role.REGRET}};
  }

  @Test(groups = "native", dataProvider = "roles")
  public void groupedForwardPreservesSeparateBucketScoresAndPolicies(CfrModel.Role role) {
    verifyGroupedScores(role, Device.cpu(), DecisionComputePrecision.FLOAT32);
  }

  @Test(groups = "rocm", dataProvider = "roles")
  public void groupedForwardPreservesReducedPrecisionScoresAndPolicies(CfrModel.Role role) {
    verifyGroupedScores(role, Device.gpu(0), DecisionComputePrecision.BFLOAT16);
    verifyGroupedScores(role, Device.gpu(0), DecisionComputePrecision.FLOAT16);
  }

  private static void verifyGroupedScores(
      CfrModel.Role role, Device device, DecisionComputePrecision precision) {
    var first = CfrModelTest.context(877);
    var second = CfrModelTest.context(878);
    var call = callContext();
    var contexts = new ArrayList<DecisionContext>();
    for (int index = 0; index < 19; index++) contexts.add(index % 2 == 0 ? first : second);
    for (int index = 0; index < 7; index++) contexts.add(call);
    try (var model = trained(first, role, device, precision, 23);
        var frozen = model.copyForInference(device);
        var workspace = frozen.newInferenceWorkspace()) {
      // 順序を反転すると、共有枠内で単遷移と複数遷移が隣り合う境界も入れ替わる。
      for (var ordered : List.of(contexts, contexts.reversed())) {
        var encoded = frozen.encode(ordered);
        var payloads = CfrInputEncoder.encodePayloads(encoded, null).join();
        var separatePredictions = new Prediction[encoded.size()];
        var reference = new ArrayList<float[]>();
        int length = 0;
        for (var group : encoded.groups) {
          int[] localIndices = new int[group.indices.length];
          Arrays.setAll(localIndices, index -> index);
          var localGroup = new CfrModel.EncodedGroup(group.host, localIndices, group.forced);
          try (var batch =
              workspace.stage(
                  new CfrModel.EncodedBatch(List.of(localGroup), localIndices.length))) {
            batch.upload();
            var output = batch.forward();
            float[] scores = output == null ? new float[0] : output.toFloatArray();
            reference.add(scores);
            length += scores.length;
            // 方策も同じ凍結経路から取り、容量区分をまとめた影響だけを比較する。
            var predictions =
                batch
                    .complete(FloatBuffer.wrap(scores))
                    .withObservations(
                        Arrays.stream(group.indices).mapToObj(payloads::get).toList());
            for (int row = 0; row < group.indices.length; row++)
              separatePredictions[group.indices[row]] = predictions.get(row);
          }
        }
        float[] expected = new float[length];
        int offset = 0;
        for (float[] scores : reference) {
          System.arraycopy(scores, 0, expected, offset, scores.length);
          offset += scores.length;
        }
        var separatePolicies = List.of(separatePredictions);
        try (var batch = workspace.stage(encoded)) {
          batch.upload();
          float[] actual = batch.forward().toFloatArray();
          assertGroupedScores(actual, expected, precision, role.name());
          var policies = batch.complete(FloatBuffer.wrap(actual)).withObservations(payloads);
          if (precision == DecisionComputePrecision.FLOAT32)
            assertPredictions(separatePolicies, policies, 1e-6);
          else
            assertReducedPrecisionPredictions(
                separatePolicies,
                policies,
                precision == DecisionComputePrecision.BFLOAT16 ? 1d / 128 : 1d / 1024);
        }
      }
    }
  }

  static void assertGroupedScores(
      float[] actual, float[] expected, DecisionComputePrecision precision, String role) {
    Assert.assertEquals(actual.length, expected.length);
    if (precision == DecisionComputePrecision.FLOAT32) {
      Assert.assertEquals(actual, expected, 2e-4f);
      return;
    }
    double squaredError = 0, expectedSquares = 0, actualSquares = 0, dot = 0;
    double maximumError = 0, maximumExpected = 0;
    for (int index = 0; index < expected.length; index++) {
      Assert.assertTrue(Float.isFinite(actual[index]));
      double reference = expected[index], value = actual[index], difference = value - reference;
      squaredError += difference * difference;
      expectedSquares += reference * reference;
      actualSquares += value * value;
      dot += value * reference;
      maximumError = Math.max(maximumError, Math.abs(difference));
      maximumExpected = Math.max(maximumExpected, Math.abs(reference));
    }
    double relativeRms = Math.sqrt(squaredError / Math.max(expectedSquares, 1e-30));
    double cosine = dot / Math.sqrt(Math.max(expectedSquares * actualSquares, 1e-30));
    double resolution = precision == DecisionComputePrecision.BFLOAT16 ? 1d / 128 : 1d / 1024;
    String metrics =
        role
            + " "
            + precision
            + " grouped raw scores maxAbs="
            + maximumError
            + " relativeRms="
            + relativeRms
            + " cosine="
            + cosine;
    System.out.println(metrics);
    // 共有特徴・遷移射影の既存試験と同じ低精度基準を使う。
    Assert.assertTrue(relativeRms <= 2 * resolution, metrics);
    Assert.assertTrue(cosine >= .9998, metrics);
    Assert.assertTrue(maximumError <= Math.max(1, maximumExpected) * 4 * resolution, metrics);
  }

  @Test(groups = "native", dataProvider = "roles")
  public void compactTransitionBucketsPreserveDensePoliciesAndPayloads(CfrModel.Role role) {
    var base = CfrModelTest.context(877);
    var other = CfrModelTest.context(878);
    var call = callContext();
    try (var model = trained(base, role, Device.cpu());
        var frozen = model.copyForInference(Device.cpu());
        var workspace = frozen.newInferenceWorkspace()) {
      assertStagedParity(
          frozen, workspace, List.of(base, other, call, context(call, 1, 0), call, base));
      assertStagedParity(frozen, workspace, List.of(call));
    }
  }

  @Test(groups = "rocm", dataProvider = "roles")
  public void compactTransitionBucketsPreserveBfloat16Policies(CfrModel.Role role) {
    assertReducedPrecisionPolicies(role, DecisionComputePrecision.BFLOAT16);
  }

  @Test(groups = "rocm", dataProvider = "roles")
  public void compactTransitionBucketsPreserveFloat16Policies(CfrModel.Role role) {
    assertReducedPrecisionPolicies(role, DecisionComputePrecision.FLOAT16);
  }

  private static void assertReducedPrecisionPolicies(
      CfrModel.Role role, DecisionComputePrecision precision) {
    var base = CfrModelTest.context(877);
    var other = CfrModelTest.context(878);
    var call = callContext();
    try (var model = trained(base, role, Device.gpu(0), precision);
        var frozen = model.copyForInference(Device.gpu(0));
        var workspace = frozen.newInferenceWorkspace()) {
      for (var contexts :
          List.of(
              List.of(base, other, call, context(call, 1, 0), call, base),
              List.of(call),
              List.of(other, base, other))) {
        assertReducedPrecisionPredictions(
            frozen.evaluate(contexts),
            evaluateStaged(frozen, workspace, contexts),
            precision == DecisionComputePrecision.BFLOAT16 ? 1d / 128 : 1d / 1024);
      }
    }
  }

  @Test(groups = "native", dataProvider = "roles")
  public void stagedMixedBucketsAndRepeatedObservationsPreserveOrderPayloadAndPolicy(
      CfrModel.Role role) {
    var base = CfrModelTest.context(877);
    try (var model = trained(base, role, Device.cpu());
        var frozen = model.copyForInference(Device.cpu());
        var workspace = frozen.newInferenceWorkspace()) {
      List<DecisionContext> first = mixed(base, 0);
      assertStagedParity(frozen, workspace, first);
      assertStagedParity(frozen, workspace, mixed(base, 1));
      assertStagedParity(frozen, workspace, mixed(base, 2));
      assertStagedParity(frozen, workspace, List.of(context(base, 1, 0), context(base, 1, 1)));
      assertStagedParity(frozen, workspace, List.of());
    }
  }

  @Test(groups = "native")
  public void cpuFleetStillUsesTheSynchronousFallbackAndReleasesRejectedIngress() {
    var base = CfrModelTest.context(877);
    try (var model = trained(base, CfrModel.Role.AVERAGE, Device.cpu());
        var reference = model.copyForInference(Device.cpu());
        var execution = new PolicyExecutionContext(1, 1, 1);
        var fleet = new CfrPolicyFleet(model, NetworkDevices.of(Device.cpu()), execution)) {
      var invalid =
          new DecisionContext(
              base.episodeId(), base.decisionId(), base.seat(), null, base.legalActions());
      Assert.expectThrows(CompletionException.class, () -> submit(fleet, List.of(invalid)).join());
      var contexts = mixed(base, 1);
      assertPredictions(reference.evaluate(contexts), fleet.evaluate(contexts), 1e-6);
    }
  }

  @Test(groups = "native")
  public void failedCompletionRejectsInvalidOutputAndWorkspaceCanBeReused() {
    var base = CfrModelTest.context(879);
    try (var model = trained(base, CfrModel.Role.AVERAGE, Device.cpu());
        var frozen = model.copyForInference(Device.cpu());
        var workspace = frozen.newInferenceWorkspace()) {
      Assert.expectThrows(IllegalStateException.class, model::newInferenceWorkspace);
      var encoded = frozen.encode(List.of(base));
      try (var batch = workspace.stage(encoded)) {
        Assert.expectThrows(IllegalStateException.class, () -> workspace.stage(encoded));
        Assert.expectThrows(IllegalStateException.class, workspace::close);
        Assert.expectThrows(IllegalStateException.class, batch::forward);
        batch.upload();
        float[] output = batch.forward().toFloatArray();
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> batch.complete(FloatBuffer.wrap(Arrays.copyOf(output, output.length - 1))));
        output[0] = Float.NaN;
        Assert.expectThrows(
            IllegalStateException.class, () -> batch.complete(FloatBuffer.wrap(output)));
      }
      assertStagedParity(frozen, workspace, List.of(base));
    }
  }

  @Test(groups = "native", dataProvider = "roles")
  public void readbackContainsOnlyActionScoresAndRespectsTheBufferPosition(CfrModel.Role role) {
    var base = CfrModelTest.context(880);
    try (var model = trained(base, role, Device.cpu());
        var frozen = model.copyForInference(Device.cpu());
        var workspace = frozen.newInferenceWorkspace()) {
      var encoded = frozen.encode(mixed(base, 1));
      int expectedElements = 0;
      for (var group : encoded.groups)
        if (!group.forced)
          for (int row = 0; row < group.host.size(); row++)
            expectedElements += group.host.legalActionCount(row);
      var payloads = CfrInputEncoder.encodePayloads(encoded, null).join();
      var expected = frozen.evaluateEncoded(encoded).withObservations(payloads);
      try (var batch = workspace.stage(encoded)) {
        batch.upload();
        float[] output = batch.forward().toFloatArray();
        Assert.assertEquals(output.length, expectedElements);
        Assert.assertEquals(batch.outputElements(), expectedElements);
        var buffer = FloatBuffer.allocate(output.length + 3);
        buffer.position(3).put(output).flip().position(3);
        var actual = batch.complete(buffer).withObservations(payloads);
        Assert.assertEquals(buffer.position(), 3);
        assertPredictions(expected, actual, 1e-6);
        for (int row = 0; row < actual.size(); row++)
          Assert.assertSame(actual.get(row).observationPayload(), payloads.get(row));
        Assert.expectThrows(IllegalStateException.class, () -> batch.complete(buffer));
      }
    }
  }

  @Test(groups = "native")
  public void outputAndWorkspaceAreReleasedWhilePayloadEncodingIsStillQueued() throws Exception {
    var engine = new GameEngine(879);
    var awaiting = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    var point = awaiting.decisions().getFirst();
    var base =
        new DecisionContext(
            879,
            point.id(),
            point.player(),
            engine.observation(point.player()),
            point.legalActions());
    var contexts = mixed(base, 1);
    var expectedPayloads = contexts.stream().map(CfrObservationCodec::encodePayload).toList();
    try (var model = trained(base, CfrModel.Role.AVERAGE, Device.cpu());
        var frozen = model.copyForInference(Device.cpu());
        var workspace = frozen.newInferenceWorkspace();
        var execution = new PolicyExecutionContext(1)) {
      var encoded = frozen.encode(contexts);
      var expected = frozen.evaluateEncoded(encoded).withObservations(expectedPayloads);
      var release = new CompletableFuture<Void>();
      var entered = new CountDownLatch(1);
      execution.encodeRowsAsync(
          1,
          (from, to) -> {
            entered.countDown();
            release.join();
          });
      try {
        Assert.assertTrue(entered.await(10, TimeUnit.SECONDS));
        var payloads = encoded.encodePayloadsAsync(execution);
        CfrModel.Policies policies;
        try (var staged = workspace.stage(encoded)) {
          staged.upload();
          var output = staged.forward().toFloatArray();
          policies = staged.complete(FloatBuffer.wrap(output));
          Arrays.fill(output, Float.NaN);
        }
        Assert.assertFalse(payloads.isDone());
        // 出力と作業領域を使い直しても、先の方策配列は独立している。
        assertStagedParity(frozen, workspace, List.of(context(base, 1, 0)));
        // 保存処理は観測元のエンジンではなく、確定済みの盤面入力だけを読む。
        var selections = new EngineSelectionBuffer();
        for (var decision : awaiting.decisions())
          selections.add(decision.id(), decision.legalActions().getFirst());
        engine.commitDecisions(selections);
        release.complete(null);
        var saved = payloads.get(10, TimeUnit.SECONDS);
        Assert.assertEquals(saved, expectedPayloads);
        var actual = policies.withObservations(saved);
        assertPredictions(expected, actual, 1e-6);
        var mutated = actual.getFirst().targetPolicy();
        Arrays.fill(mutated, -1);
        assertPredictions(expected, policies.withObservations(saved), 1e-6);
      } finally {
        release.complete(null);
      }
    }
  }

  @Test(groups = "rocm", dataProvider = "roles")
  public void frozenStateFusionMatchesEagerLowPrecisionWithRepeatedBoardObservations(
      CfrModel.Role role) {
    var base = CfrModelTest.context(877);
    try (var model = trained(base, role, Device.gpu(0));
        var eager = model.copyForInference(Device.gpu(0));
        var fused = model.copyForInference(Device.gpu(0));
        var eagerWorkspace = eager.newInferenceWorkspace(false);
        var fusedWorkspace = fused.newInferenceWorkspace()) {
      for (int step : new int[] {0, 1, 2, 2, 0}) {
        var contexts = mixed(base, step);
        assertBfloat16Predictions(
            evaluateStaged(eager, eagerWorkspace, contexts),
            evaluateStaged(fused, fusedWorkspace, contexts));
      }
      assertBfloat16Predictions(
          evaluateStaged(eager, eagerWorkspace, List.of(base)),
          evaluateStaged(fused, fusedWorkspace, List.of(base)));
    }
  }

  @Test(groups = "rocm", dataProvider = "roles")
  public void sharedGpuPipelinePreservesMixedBucketPoliciesAndDrains(CfrModel.Role role) {
    var base = CfrModelTest.context(877);
    try (var model = trained(base, role, Device.gpu(0));
        var reference = model.copyForInference(Device.gpu(0));
        var execution = new PolicyExecutionContext(2, 3, 4)) {
      try (var fleet = new CfrPolicyFleet(model, NetworkDevices.of(Device.gpu(0)), execution)) {
        for (int step = 0; step < 3; step++) {
          List<DecisionContext> contexts = mixed(base, step);
          var expected =
              reference.evaluateObservations(
                  contexts.stream().map(CfrObservationCodec::encodePayload).toList());
          assertBfloat16Predictions(expected, fleet.evaluate(contexts));
        }
        List<DecisionContext> forced = List.of(context(base, 1, 4));
        assertPredictions(reference.evaluate(forced), fleet.evaluate(forced), 0);
        var invalid =
            new DecisionContext(
                base.episodeId(), base.decisionId(), base.seat(), null, base.legalActions());
        var rejected = submit(fleet, List.of(invalid));
        Assert.expectThrows(CompletionException.class, rejected::join);
        List<List<Prediction>> expected = new ArrayList<>();
        List<CompletableFuture<List<Prediction>>> pending = new ArrayList<>();
        for (int index = 0; index < 6; index++) {
          final long episode = 1000L + index;
          var contexts =
              mixed(base, index).stream()
                  .map(
                      context ->
                          new DecisionContext(
                              episode,
                              context.decisionId(),
                              context.seat(),
                              context.observation(),
                              context.legalActions(),
                              context.roundOrdinal()))
                  .toList();
          expected.add(
              reference.evaluateObservations(
                  contexts.stream().map(CfrObservationCodec::encodePayload).toList()));
          pending.add(submit(fleet, contexts));
        }
        for (int index = 0; index < pending.size(); index++)
          assertBfloat16Predictions(expected.get(index), pending.get(index).join());
      }
      execution.devices().awaitIdle();
    }
  }

  private static CompletableFuture<List<Prediction>> submit(
      CfrPolicyFleet fleet, List<DecisionContext> contexts) {
    while (true) {
      var attempt = fleet.tryAcquireInferenceIngress();
      if (attempt.acquired()) return fleet.submitBatch(contexts, attempt.ingress());
      attempt.available().join();
    }
  }

  private static void assertStagedParity(
      CfrModel model, CfrInferenceWorkspace workspace, List<DecisionContext> contexts) {
    var payloads = contexts.stream().map(CfrObservationCodec::encodePayload).toList();
    var expected = model.evaluateObservations(payloads);
    assertPredictions(expected, evaluateStaged(model, workspace, contexts), 1e-6);
  }

  private static List<Prediction> evaluateStaged(
      CfrModel model, CfrInferenceWorkspace workspace, List<DecisionContext> contexts) {
    var encoded = model.encode(contexts);
    try (var batch = workspace.stage(encoded)) {
      batch.upload();
      var output = batch.forward();
      return batch
          .complete(
              output == null ? FloatBuffer.allocate(0) : FloatBuffer.wrap(output.toFloatArray()))
          .withObservations(CfrInputEncoder.encodePayloads(encoded, null).join());
    }
  }

  static void assertPredictions(
      List<Prediction> expected, List<Prediction> actual, double tolerance) {
    Assert.assertEquals(actual.size(), expected.size());
    for (int row = 0; row < expected.size(); row++) {
      Assert.assertEquals(
          actual.get(row).observationPayload(), expected.get(row).observationPayload());
      double[] found = actual.get(row).targetPolicy(), wanted = expected.get(row).targetPolicy();
      Assert.assertEquals(found.length, wanted.length);
      for (int index = 0; index < found.length; index++)
        Assert.assertEquals(found[index], wanted[index], tolerance);
    }
  }

  private static void assertBfloat16Predictions(
      List<Prediction> expected, List<Prediction> actual) {
    assertReducedPrecisionPredictions(expected, actual, 1d / 128);
  }

  static void assertReducedPrecisionPredictions(
      List<Prediction> expected, List<Prediction> actual, double resolution) {
    // 型の1付近での精度に基づき、分布距離と最大確率の行動が他の候補につけた差も確認する。
    assertPredictions(expected, actual, resolution);
    for (int row = 0; row < expected.size(); row++) {
      double[] p = expected.get(row).targetPolicy(), q = actual.get(row).targetPolicy();
      double kl = 0;
      int expectedWinner = 0, actualWinner = 0;
      for (int action = 0; action < p.length; action++) {
        Assert.assertEquals(q[action] == 0, p[action] == 0, "Inference changed policy support");
        if (p[action] > 0) kl += p[action] * Math.log(p[action] / q[action]);
        if (p[action] > p[expectedWinner]) expectedWinner = action;
        if (q[action] > q[actualWinner]) actualWinner = action;
      }
      Assert.assertTrue(kl <= resolution * resolution, "Inference policy KL=" + kl);
      Assert.assertTrue(
          p[expectedWinner] - p[actualWinner] <= resolution,
          "Inference changed a winner beyond rounding uncertainty");
    }
  }

  private static List<DecisionContext> mixed(DecisionContext base, int step) {
    int all = base.legalActions().size();
    Assert.assertTrue(all > 4);
    return List.of(
        context(base, all, step),
        context(base, 1, step),
        context(base, 2, step + 1),
        context(base, all, step + 2),
        context(base, 3, step + 1),
        context(base, all, step));
  }

  private static DecisionContext context(DecisionContext base, int actions, int step) {
    return new DecisionContext(
        base.episodeId(),
        base.decisionId() + step,
        base.seat(),
        base.observation(),
        base.legalActions().subList(0, actions),
        base.roundOrdinal() + step);
  }

  /** 鳴き後の複数打牌を持つ局面を返し、返却後は観測元のエンジンを進めない。 */
  static DecisionContext callContext() {
    var engine = new GameEngine(770);
    GameStepResult step = engine.stepHanchan();
    for (int boundary = 0;
        boundary < 200 && !(step instanceof GameStepResult.HanchanEnded);
        boundary++) {
      if (step instanceof GameStepResult.RoundSettled) {
        step = engine.stepHanchan();
        continue;
      }
      var selections = new EngineSelectionBuffer();
      for (var point : ((GameStepResult.AwaitingDecisions) step).decisions()) {
        if (DecisionBatchBuilder.selectInferenceBucket(point.legalActions())
                .actionTransitionCapacity()
            > 1) {
          return new DecisionContext(
              770,
              point.id(),
              point.player(),
              engine.observation(point.player()),
              point.legalActions());
        }
        selections.add(point.id(), point.legalActions().getFirst());
      }
      step = engine.commitDecisions(selections);
    }
    throw new AssertionError("Fixed fixture has no call transitions");
  }

  private static CfrModel trained(DecisionContext base, CfrModel.Role role, Device device) {
    return trained(
        base,
        role,
        device,
        device.isGpu() ? DecisionComputePrecision.BFLOAT16 : DecisionComputePrecision.FLOAT32);
  }

  private static CfrModel trained(
      DecisionContext base, CfrModel.Role role, Device device, DecisionComputePrecision precision) {
    return trained(base, role, device, precision, 2);
  }

  private static CfrModel trained(
      DecisionContext base,
      CfrModel.Role role,
      Device device,
      DecisionComputePrecision precision,
      int inferenceBatchSize) {
    var config =
        new CfrModelConfig(
            device.isGpu() ? 256 : 16,
            8,
            .01f,
            0,
            0,
            919,
            inferenceBatchSize,
            precision,
            device.isGpu()
                ? DecisionTensorTransfer.PINNED_BUFFER
                : DecisionTensorTransfer.DIRECT_BUFFER,
            EpsilonUtilityProfile.TENHOU);
    var model = new CfrModel(config, role, device);
    try {
      int[] actions = base.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
      double[] teacher = new double[actions.length];
      if (role == CfrModel.Role.AVERAGE) teacher[0] = 1;
      else Arrays.fill(teacher, 1d / teacher.length);
      var row =
          new CfrTrainingRow(
              new CfrSampleId(1, base.episodeId(), base.decisionId(), base.seat()),
              actions,
              0,
              teacher,
              .5,
              0,
              0,
              CfrObservationCodec.encodePayload(base));
      for (int update = 0; update < 3; update++) model.train(List.of(row));
      double[] policy = model.evaluate(List.of(base)).getFirst().targetPolicy();
      Assert.assertTrue(
          Arrays.stream(policy).max().orElseThrow() - Arrays.stream(policy).min().orElseThrow()
              > 1e-5);
      return model;
    } catch (RuntimeException | Error failure) {
      model.close();
      throw failure;
    }
  }
}
