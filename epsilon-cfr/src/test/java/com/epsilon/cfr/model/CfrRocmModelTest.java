package com.epsilon.cfr.model;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.network.NetworkDevices;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

@Test(groups = "rocm")
public class CfrRocmModelTest {
  public void bfloat16ForwardAndPinnedTransfersPreserveTrainingAndCheckpointResume()
      throws Exception {
    var config =
        new CfrModelConfig(
            256,
            8,
            2e-4f,
            1e-4f,
            .5f,
            1337,
            32,
            DecisionComputePrecision.BFLOAT16,
            DecisionTensorTransfer.PINNED_BUFFER,
            EpsilonUtilityProfile.TENHOU);
    List<CfrTrainingRow> rows = new ArrayList<>();
    for (int index = 0; index < 24; index++) {
      var context = CfrModelTest.context(900 + index);
      int[] actions =
          context.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
      double[] target = new double[actions.length];
      target[0] = 1;
      rows.add(
          new CfrTrainingRow(
              new CfrSampleId(1, index, 0, context.seat()),
              actions,
              0,
              target,
              1,
              0,
              0,
              CfrObservationCodec.encodePayload(context)));
    }
    var directory = Files.createTempDirectory("cfr-bfloat16-");
    try (var model =
        new CfrModel(config, CfrModel.Role.AVERAGE, NetworkDevices.of(Device.gpu(0)))) {
      double initial = model.train(rows).loss();
      CfrModel.TrainingMetrics trained = null;
      for (int update = 0; update < 5; update++) trained = model.train(rows);
      Assert.assertTrue(Double.isFinite(trained.gradientNorm()));
      Assert.assertTrue(trained.loss() < initial);
      model.saveTraining(directory);
      var expected =
          model.evaluateObservations(rows.stream().map(CfrTrainingRow::observation).toList());
      for (var prediction : expected)
        Assert.assertEquals(Arrays.stream(prediction.targetPolicy()).sum(), 1, 1e-9);
      try (var loaded = CfrModel.loadTraining(directory, NetworkDevices.of(Device.gpu(0)))) {
        var actual =
            loaded.evaluateObservations(rows.stream().map(CfrTrainingRow::observation).toList());
        for (int row = 0; row < expected.size(); row++)
          Assert.assertEquals(actual.get(row).targetPolicy(), expected.get(row).targetPolicy());
        var next = loaded.train(rows);
        Assert.assertEquals(next.updates(), 7L);
        Assert.assertTrue(Double.isFinite(next.loss()));
      }
    }
  }
}
