package com.epsilon.prism.grp;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.ai.grp.EpsilonGrpNetwork;
import com.epsilon.ai.grp.EpsilonGrpTrainer;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.prism.collect.GrpPhi;
import com.epsilon.prism.collect.PhiCalibration;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 正確なパスの GRP の Φ は、学習の JVM の {@link GrpPhi}（同じ checkpoint を {@code openInferenceOnly} で開き、同じ温度と較正の表）と最後の
 * ビットまで同じ値を返す。構造が grp-epsilon-v3 でない checkpoint は読まない。
 */
public class CheckpointPhiTest {
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;
  private static final double[] TEMPERATURES = {1.0, 1.15, 0.9, 1.3};

  @Test(groups = "native")
  public void matchesTheTrainingJvmPhiBitForBit() throws Exception {
    Path root = Files.createTempDirectory("grp-phi");
    Path exact = root.resolve("grp").resolve("latest");
    Engine.getEngine("PyTorch").setRandomSeed(9);
    try (Model model =
            NetworkFactory.createGrpModel(
                Device.cpu(), EpsilonGrpNetwork.DEFAULT_HIDDEN, EpsilonGrpNetwork.DEFAULT_LAYERS);
        EpsilonGrpTrainer saver = new EpsilonGrpTrainer(model, 64, 2e-4f, 1e-4f, 0.5f)) {
      EpsilonGrpCheckpointManager.save(model, saver, exact, 0, 0);
    }
    PhiCalibration calibration = PhiCalibration.load(calibrationTable(root.resolve("calib.json")));
    List<HandEndInput> inputs = new ArrayList<>();
    for (RecordedGame game : GrpGameTrainerTest.games(new SplittableRandom(8), 120))
      for (int k = 0; k < game.continuingHandEnds(); k++) inputs.add(game.input(k));
    double[][] expected;
    try (var session = EpsilonGrpTrainingSession.openInferenceOnly(root, Device.cpu())) {
      Assert.assertEquals(
          EpsilonGrpCheckpointManager.resolveExisting(root.resolve("grp")).toAbsolutePath(), exact.toAbsolutePath());
      expected = production(session.inference(), calibration, inputs);
    }
    try (var phi = CheckpointPhi.open(exact, TEMPERATURES, calibration, PROFILE)) {
      PhiValues values = phi.values(inputs);
      for (int i = 0; i < inputs.size(); i++)
        Assert.assertEquals(
            Arrays.stream(values.utilities()[i]).mapToLong(Double::doubleToRawLongBits).toArray(),
            Arrays.stream(expected[i]).mapToLong(Double::doubleToRawLongBits).toArray(),
            "hand end " + i);
      for (double[] ranks : values.rankProbabilities()) {
        double total = 0;
        for (double p : ranks) total += p;
        Assert.assertEquals(total, 4, 1e-4);
      }
    }
  }

  @Test(groups = "native")
  public void refusesAnotherArchitecture() throws IOException {
    Path exact = Files.createTempDirectory("grp-phi").resolve("v2");
    try (Model model = NetworkFactory.createGrpModel(Device.cpu(), 32, 1);
        EpsilonGrpTrainer saver = new EpsilonGrpTrainer(model, 64, 2e-4f, 1e-4f, 0.5f)) {
      EpsilonGrpCheckpointManager.save(model, saver, exact, 0, 0);
    }
    Path manifest = exact.resolve("manifest.json");
    Files.writeString(manifest, Files.readString(manifest).replace("grp-epsilon-v3", "grp-epsilon-v2"));
    Assert.assertThrows(IOException.class, () -> CheckpointPhi.open(exact, new double[0], null, PROFILE));
  }

  /** {@link GrpPhi} に全ての入力を投げて、ドライバのスレッドで受け取った値。 */
  private static double[][] production(
      EpsilonGrpInference inference, PhiCalibration calibration, List<HandEndInput> inputs)
      throws InterruptedException {
    ExecutorService driver = Executors.newSingleThreadExecutor();
    double[][] values = new double[inputs.size()][];
    CountDownLatch done = new CountDownLatch(inputs.size());
    try (GrpPhi grp =
        new GrpPhi(inference, PROFILE, driver::execute, 256, 1 << 12, TEMPERATURES, calibration)) {
      driver.execute(
          () -> {
            for (int i = 0; i < inputs.size(); i++) {
              int row = i;
              HandEndInput input = inputs.get(i);
              grp.request(
                  input.prefix(),
                  input.next().scores(),
                  result -> {
                    values[row] = result;
                    done.countDown();
                  });
            }
          });
      Assert.assertTrue(done.await(2, TimeUnit.MINUTES));
    } finally {
      driver.shutdown();
      driver.awaitTermination(1, TimeUnit.MINUTES);
    }
    return values;
  }

  private static Path calibrationTable(Path file) throws IOException {
    SplittableRandom random = new SplittableRandom(4);
    StringBuilder json =
        new StringBuilder("{\"stages\": [\"E1\",\"E2\",\"E3\",\"E4\",\"S1\",\"S2\",\"S3\",\"S4\",\"W+\"], ");
    json.append("\"lo\": -1.0, \"hi\": 0.6666666666666666, \"bins\": 24, \"table\": {");
    for (int stage = 0; stage < 9; stage++) {
      json.append(stage == 0 ? "" : ", ").append('"').append(stage).append("\": [");
      for (int bin = 0; bin < 24; bin++) json.append(bin == 0 ? "" : ", ").append(0.04 * (random.nextDouble() - 0.5));
      json.append(']');
    }
    Files.writeString(file, json.append("}}").toString(), StandardCharsets.UTF_8);
    return file;
  }
}
