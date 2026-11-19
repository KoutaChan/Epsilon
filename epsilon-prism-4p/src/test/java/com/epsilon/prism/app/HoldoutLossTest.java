package com.epsilon.prism.app;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.Main;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.eval.GameEstimate;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.ValueLabel;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.run.Snapshot;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class HoldoutLossTest {
  @Test
  public void reportsBothLabelSourcesAndExcludesRoots() throws Exception {
    var run = Files.createTempDirectory("holdout-loss");
    var variant = Variant.TENHOU_4P;
    try {
      TestSettings.create(run);
      var snapshot = run.resolve("version");
      try (var manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(snapshot, HipFixtures.network(manager, 2, 11), PolicyParams.of(1));
      }
      long[] observation = HipFixtures.mixedRows(16, 23).getFirst();
      int n = new Observation(variant).bind(observation, 0).legalActions();
      Assert.assertTrue(n >= 2);
      float[] prior = new float[n];
      Arrays.fill(prior, 1f / n);
      try (var writer =
          new CohortWriter(
              CohortWriter.directory(run, 1), variant, 1, 0, 1, CohortWriter.Tap.NONE)) {
        for (var source : Row.Source.values()) {
          writer.accept(
              new Row(
                  source.ordinal(),
                  1,
                  DecisionClass.CALL,
                  Row.Purpose.HOLDOUT,
                  source,
                  0,
                  new float[n],
                  prior,
                  3,
                  source == Row.Source.ROOT ? 1 : -1,
                  1,
                  1,
                  -1,
                  new float[0],
                  new long[n * OutcomeCodec.WORDS],
                  new long[0],
                  observation));
        }
        float[] shifted = new float[n];
        Arrays.fill(shifted, 0.25f);
        writer.accept(
            new Row(
                4,
                1,
                DecisionClass.RIICHI,
                Row.Purpose.TRAIN,
                Row.Source.MAIN,
                0,
                shifted,
                prior,
                3,
                -1,
                1,
                1,
                -1,
                new float[0],
                new long[n * OutcomeCodec.WORDS],
                new long[0],
                observation));
        for (int measured = 0; measured < 2; measured++) {
          float[] sparse = new float[n];
          Arrays.fill(sparse, Float.NaN);
          if (measured == 1) sparse[0] = 0;
          writer.accept(
              new Row(
                  5,
                  measured,
                  DecisionClass.RIICHI,
                  Row.Purpose.TRAIN,
                  Row.Source.ROLLIN,
                  0,
                  sparse,
                  prior,
                  measured,
                  -1,
                  1,
                  1,
                  -1,
                  new float[0],
                  new long[n * OutcomeCodec.WORDS],
                  new long[0],
                  observation));
        }
        writer.seal();
      }
      int exit =
          Main.run(
              new String[] {
                "holdout-loss",
                "--run",
                run.toString(),
                "--cohort",
                "1",
                "--versions",
                snapshot.toString(),
                "--versions",
                snapshot.toString(),
                "--out",
                run.toString(),
                "--set",
                "inference.maxBatch=1"
              });
      Assert.assertEquals(exit, Exit.OK.code());
      var json =
          JsonParser.parseString(Files.readString(run.resolve("holdout-loss.json")))
              .getAsJsonArray();
      var all = json.get(json.size() - 1).getAsJsonObject().getAsJsonObject("loss");
      Assert.assertEquals(all.get("positions").getAsInt(), 3);
      var classes = new HashMap<String, Double>();
      for (var item : json) {
        var line = item.getAsJsonObject();
        Assert.assertEquals(line.get("backend").getAsString(), "EAGER");
        Assert.assertEquals(line.get("precision").getAsString(), "FLOAT32");
        var loss = line.getAsJsonObject("loss");
        if (loss.get("positions").getAsInt() > 0)
          classes.put(line.get("group").getAsString(), loss.get("mean").getAsDouble());
      }
      double call = classes.get("CALL"), riichi = classes.get("RIICHI");
      Assert.assertEquals(all.get("mean").getAsDouble(), (2 * call + riichi) / 3, 1e-12);
      Assert.assertEquals(all.get("se").getAsDouble(), Math.abs(call - riichi) / 3, 1e-12);
      Assert.assertTrue(
          Files.readString(run.resolve("holdout-loss.tsv")).contains("\tALL\t3\t3\t"));
      var diff = json.get(json.size() - 1).getAsJsonObject().getAsJsonObject("difference");
      Assert.assertEquals(diff.get("reference").getAsString(), snapshot.toString());
      Assert.assertEquals(diff.get("meanDifference").getAsDouble(), 0.);
      Assert.assertEquals(diff.get("se").getAsDouble(), 0.);
      Assert.assertTrue(diff.get("z").isJsonNull());
      var differences = Files.readAllLines(run.resolve("holdout-loss-diff.tsv"));
      Assert.assertEquals(differences.size(), json.size() / 2 + 1);
      Assert.assertEquals(
          differences.getLast(), snapshot + "\t" + snapshot + "\tALL\t3\t3\t0.0\t0.0\tnull");
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void pairedUncertaintyUsesTheSumOfDifferencesWithinEachHanchan() throws Exception {
    var baseline = new GameEstimate.Game();
    baseline.add(DecisionClass.CALL, 100);
    baseline.add(DecisionClass.RIICHI, 90);
    var candidate = new GameEstimate.Game();
    candidate.add(DecisionClass.CALL, 101);
    candidate.add(DecisionClass.RIICHI, 93);
    var otherBaseline = new GameEstimate.Game();
    otherBaseline.add(DecisionClass.CALL, -80);
    var otherCandidate = new GameEstimate.Game();
    otherCandidate.add(DecisionClass.CALL, -81);
    byte[] digest = {5};
    var reference = new HoldoutLoss.Scores(Map.of(1L, baseline, 2L, otherBaseline), digest);
    var games = Map.of(2L, otherCandidate, 1L, candidate);
    var scores = new HoldoutLoss.Scores(games, digest);
    var differences = scores.differences(reference);
    var all = GameEstimate.of(differences, null);
    Assert.assertEquals(all, new GameEstimate(3, 2, 1., 4. / 3));
    Assert.assertEquals(new HoldoutLoss.Difference(Path.of("reference"), all).z(), .75);
    Assert.assertEquals(
        GameEstimate.of(differences, DecisionClass.CALL), new GameEstimate(2, 2, 0., 1.));
    Assert.assertEquals(
        GameEstimate.of(differences, DecisionClass.RIICHI), new GameEstimate(1, 1, 3., null));
    Assert.expectThrows(
        IOException.class,
        () -> new HoldoutLoss.Scores(games, new byte[] {6}).differences(reference));
  }

  @Test
  public void measuredLossMatchesTrainingValue() {
    try (var manager = NDManager.newBaseManager(Device.cpu())) {
      float[] scores = {0.01f, 42, -0.03f, 0.08f};
      float[] values = {0.2f, Float.NaN, -0.07f, 0.03f};
      float lambda = 0.05f;
      Shape shape = new Shape(1, scores.length);
      var outputs =
          new Outputs(
              manager.create(scores, shape),
              manager.create(new boolean[] {true, true, true, true}, shape),
              new int[] {scores.length},
              new boolean[scores.length],
              null);
      var loss = new PrismLoss(AuxiliaryLoss.standard());
      double expected =
          loss.value(manager, outputs, List.of(new ValueLabel(values, lambda, 1))).getFloat();
      Assert.assertEquals(PrismLoss.value(scores, values, lambda), expected, 1e-7);
    }
  }
}
