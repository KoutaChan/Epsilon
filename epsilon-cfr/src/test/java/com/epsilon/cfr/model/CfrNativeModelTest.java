package com.epsilon.cfr.model;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.testng.Assert;
import org.testng.annotations.Test;

/** DJLのCPU実装で順伝播・逆伝播と、オプティマイザーのチェックポイントからの学習再開を検証する。 */
@Test(groups = "native")
public class CfrNativeModelTest {
  private static CfrModelConfig config(int microbatch) {
    return new CfrModelConfig(
        16,
        microbatch,
        .002f,
        0,
        0,
        1337,
        microbatch,
        DecisionComputePrecision.FLOAT32,
        DecisionTensorTransfer.DIRECT_BUFFER,
        EpsilonUtilityProfile.TENHOU);
  }

  public void regretLearnsAndOptimizerResumeMatchesNextUpdate() throws Exception {
    Path checkpoint = Files.createTempDirectory("epsilon-cfr-native-");
    var context = CfrModelTest.context(981);
    var rows = rows(context, 1);
    try (var uninterrupted = new CfrModel(config(2), CfrModel.Role.REGRET, Device.cpu())) {
      double initial = uninterrupted.train(rows).loss();
      double last = initial;
      for (int step = 0; step < 7; step++) last = uninterrupted.train(rows).loss();
      Assert.assertTrue(
          last < initial, "Signed regret regression must improve: " + initial + " -> " + last);
      float[] scores = uninterrupted.rawScores(rows.getFirst().observation());
      Assert.assertTrue(
          Arrays.stream(toDouble(scores)).min().orElseThrow() < 0,
          "Regret output must retain negative predictions");
      uninterrupted.saveTraining(checkpoint);
      try (var resumed = CfrModel.loadTraining(checkpoint, Device.cpu())) {
        Assert.assertEquals(resumed.rawScores(rows.getFirst().observation()), scores);
        Assert.assertEquals(resumed.updates(), uninterrupted.updates());
        var want = uninterrupted.train(rows);
        var got = resumed.train(rows);
        Assert.assertEquals(got.loss(), want.loss(), 1e-6);
        assertClose(
            resumed.rawScores(rows.getFirst().observation()),
            uninterrupted.rawScores(rows.getFirst().observation()),
            2e-6f);
      }
      try (var frozen = CfrModel.loadModel(checkpoint, Device.cpu())) {
        Assert.expectThrows(IllegalStateException.class, () -> frozen.train(rows));
      }
    }
  }

  public void averageLearnsFullDistributionAndExportIsIndependent() throws Exception {
    var context = CfrModelTest.context(982);
    var rows = rows(context, 0);
    Path exported = Files.createTempDirectory("epsilon-cfr-average-");
    try (var average = new CfrModel(config(2), CfrModel.Role.AVERAGE, Device.cpu());
        var regret = new CfrModel(config(2), CfrModel.Role.REGRET, Device.cpu())) {
      float[] before = average.rawScores(rows.getFirst().observation());
      regret.train(rows);
      Assert.assertEquals(average.rawScores(rows.getFirst().observation()), before);
      double initial = average.train(rows).loss();
      double last = initial;
      for (int step = 0; step < 7; step++) last = average.train(rows).loss();
      Assert.assertTrue(last < initial, "Full-distribution cross entropy must improve");
      average.saveModel(exported);
      Assert.assertFalse(Files.exists(exported.resolve("optimizer.bin")));
      Assert.expectThrows(IOException.class, () -> CfrModel.loadTraining(exported, Device.cpu()));
      try (var loaded = CfrModel.loadModel(exported, Device.cpu());
          var snapshot = average.copyForInference(Device.cpu())) {
        Assert.assertEquals(
            loaded.rawScores(rows.getFirst().observation()),
            average.rawScores(rows.getFirst().observation()));
        Assert.assertEquals(
            snapshot.rawScores(rows.getFirst().observation()),
            average.rawScores(rows.getFirst().observation()));
        var predicted = loaded.evaluate(List.of(context)).getFirst().targetPolicy();
        Assert.assertEquals(Arrays.stream(predicted).sum(), 1, 1e-12);
        for (double probability : predicted) Assert.assertTrue(probability >= 0);
        var contexts = List.of(context, CfrModelTest.context(983), context);
        var direct = loaded.evaluate(contexts);
        var encoded =
            loaded.evaluateObservations(
                contexts.stream().map(CfrObservationCodec::encodePayload).toList());
        for (int row = 0; row < contexts.size(); row++)
          Assert.assertEquals(encoded.get(row).targetPolicy(), direct.get(row).targetPolicy());
        Assert.assertTrue(loaded.evaluateObservations(List.of()).isEmpty());
      }
    }
  }

  public void explicitBranchRegretsTrainSignedTargetsWithoutChangingAverageObjective() {
    CfrTrainingRow ordinary = rows(CfrModelTest.context(981), 0).getFirst();
    Assert.assertTrue(ordinary.actionCount() > 1);
    double[] targets = new double[ordinary.actionCount()];
    targets[0] = .75;
    targets[1] = -.25;
    CfrTrainingRow compared =
        new CfrTrainingRow(
            ordinary.id(),
            ordinary.legalActionIds(),
            ordinary.selectedSlot(),
            ordinary.targetPolicy(),
            0,
            ordinary.logRegretWeight(),
            ordinary.logAverageWeight(),
            ordinary.observation(),
            targets);
    var observation = ordinary.observation();
    try (var scalar = new CfrModel(config(1), CfrModel.Role.REGRET, Device.cpu());
        var branch = new CfrModel(config(1), CfrModel.Role.REGRET, Device.cpu())) {
      float[] before = branch.rawScores(observation);
      for (float score : before) Assert.assertEquals(score, 0f);
      var scalarMetrics = scalar.train(List.of(ordinary));
      var branchMetrics = branch.train(List.of(compared));

      Assert.assertEquals(scalarMetrics.loss(), 0d);
      Assert.assertEquals(scalarMetrics.gradientNorm(), 0d);
      Assert.assertEquals(scalar.rawScores(observation), before);
      Assert.assertEquals(branchMetrics.loss(), .75 * .75 + .25 * .25, 1e-6);
      Assert.assertTrue(branchMetrics.gradientNorm() > 0);
      Assert.assertFalse(
          Arrays.equals(branch.rawScores(observation), before),
          "Explicit signed targets must update REGRET even when scalar utility is zero");
    }
    try (var scalar = new CfrModel(config(1), CfrModel.Role.AVERAGE, Device.cpu());
        var branch = new CfrModel(config(1), CfrModel.Role.AVERAGE, Device.cpu())) {
      var scalarMetrics = scalar.train(List.of(ordinary));
      var branchMetrics = branch.train(List.of(compared));
      Assert.assertEquals(branchMetrics.loss(), scalarMetrics.loss(), 1e-6);
      Assert.assertEquals(branchMetrics.gradientNorm(), scalarMetrics.gradientNorm(), 1e-6);
      assertClose(branch.rawScores(observation), scalar.rawScores(observation), 1e-6f);
    }
  }

  public void microbatchChangesPreserveTheObjectiveAndUpdate() {
    var first = CfrModelTest.context(981);
    var second = CfrModelTest.context(983);
    var rows = List.of(rows(first, 1).getFirst(), rows(second, 1).getFirst());
    try (var a = new CfrModel(config(1), CfrModel.Role.REGRET, Device.cpu());
        var b = new CfrModel(config(2), CfrModel.Role.REGRET, Device.cpu())) {
      var one = a.train(rows);
      var two = b.train(rows);
      Assert.assertEquals(one.loss(), two.loss(), 2e-5);
      assertClose(
          a.rawScores(rows.getFirst().observation()),
          b.rawScores(rows.getFirst().observation()),
          1e-4f);
    }
  }

  public void newSolveCopiesOnlyRepresentationAndStartsFromZeroRegrets() {
    var context = CfrModelTest.context(984);
    var examples = rows(context, 1);
    try (var average = new CfrModel(config(2), CfrModel.Role.AVERAGE, Device.cpu());
        var regret = new CfrModel(config(2), CfrModel.Role.REGRET, Device.cpu())) {
      for (int step = 0; step < 3; step++) average.train(examples);
      float[] averageBefore = average.rawScores(examples.getFirst().observation());
      regret.initializeRepresentationFrom(average);
      Assert.assertEquals(regret.updates(), 0);
      for (float score : regret.rawScores(examples.getFirst().observation()))
        Assert.assertEquals(score, 0f);
      for (String name : regret.parameterNames()) {
        String lower = name.toLowerCase(Locale.ROOT);
        Assert.assertFalse(
            lower.contains("valuehead")
                || lower.contains("gat")
                || lower.contains("alternative")
                || lower.contains("policyhead"),
            name);
      }
      regret.train(examples);
      Assert.assertEquals(average.rawScores(examples.getFirst().observation()), averageBefore);
      Assert.expectThrows(
          IllegalStateException.class, () -> regret.initializeRepresentationFrom(average));
    }
  }

  private static List<CfrTrainingRow> rows(DecisionContext context, double utility) {
    int[] actions = context.legalActions().stream().mapToInt(a -> a.toIndex()).toArray();
    double[] policy = new double[actions.length];
    Arrays.fill(policy, .5 / actions.length);
    policy[0] += .5;
    var payload = CfrObservationCodec.encodePayload(context);
    return List.of(
        new CfrTrainingRow(
            new CfrSampleId(1, context.episodeId(), context.decisionId(), context.seat()),
            actions,
            0,
            policy,
            utility,
            0,
            0,
            payload));
  }

  private static void assertClose(float[] actual, float[] expected, float tolerance) {
    Assert.assertEquals(actual.length, expected.length);
    for (int i = 0; i < actual.length; i++)
      Assert.assertEquals(actual[i], expected[i], tolerance, "slot " + i);
  }

  private static double[] toDouble(float[] values) {
    double[] result = new double[values.length];
    for (int i = 0; i < values.length; i++) result[i] = values[i];
    return result;
  }
}
