package com.epsilon.prism.play;

import ai.djl.engine.Engine;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.Properties;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;

/** 配布済みの方策と、固定した判断列における各合法手の参照スコア。 */
public record PublishedModel(String directory, Variant variant, PrismShape shape) {
  private static final String ROOT_PROPERTY = "prism.publishedModels";

  public static PublishedModel fourPlayer(int version) {
    return new PublishedModel(
        "prism-v2b-v" + version, Variant.TENHOU_4P, new PrismShape.V2(256, 512, 256, 2));
  }

  @DataProvider
  public static Object[] fourPlayerModels() {
    return new Object[] {fourPlayer(714), fourPlayer(760)};
  }

  @DataProvider
  public static Object[] threePlayerModels() {
    var shape = new PrismShape.V3(256, 512, 512, 768, 1536, 1024, true);
    return new Object[] {
      new PublishedModel("prism3p-u3-v355", Variant.TENHOU_3P, shape),
      new PublishedModel("prism3p-u3-v413", Variant.TENHOU_3P, shape)
    };
  }

  public Path snapshot() {
    String root = System.getProperty(ROOT_PROPERTY);
    if (root == null)
      throw new SkipException("Set -D" + ROOT_PROPERTY + " to the published models directory");
    return Path.of(root).resolve(directory);
  }

  public void assertEagerScores() throws IOException {
    assertEagerScores(snapshot(), shape);
  }

  /** GPU の参照スコアと比べる。GPU か libprism_hip.so（-Pnative-hip で組んだ jar）が無いときは飛ばす。 */
  public void assertHipScores() throws IOException {
    Path snapshot = snapshot();
    if (!HipLibrary.bundled() || Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("needs a GPU and libprism_hip.so built with -Pnative-hip");
    }
    assertHipScores(snapshot, shape);
  }

  /** Compare an exported warm start against this published model's checked-in action score bits. */
  public void assertEagerScores(Path candidate, PrismShape candidateShape) throws IOException {
    assertScores(candidate, candidateShape, "cpu", Kernels.EAGER, Precision.values());
  }

  public void assertHipScores(Path candidate, PrismShape candidateShape) throws IOException {
    assertScores(candidate, candidateShape, "gpu0", Kernels.NATIVE, Precision.BFLOAT16);
  }

  private void assertScores(
      Path snapshot,
      PrismShape expectedShape,
      String device,
      Kernels kernels,
      Precision... precisions)
      throws IOException {
    PrismShape saved = PrismNetwork.readShape(Snapshot.model(snapshot));
    Assert.assertEquals(saved, expectedShape, directory);
    InferencePath path = InferencePath.of(kernels);
    Assert.assertEquals(path.resolve(NetworkDevices.device(device), saved), path, directory);
    Properties scores = new Properties();
    try (var stream = PublishedModel.class.getResourceAsStream("published-scores.properties")) {
      Assert.assertNotNull(stream, "Missing published-scores.properties");
      scores.load(stream);
    }
    ObservationBatch rows = PublishedPositions.sample(variant);
    Assert.assertEquals(
        PublishedPositions.fingerprint(rows),
        scores.getProperty(variant + ".inputs"),
        variant.name());
    for (Precision precision : precisions) {
      String key = directory + "." + path + "." + precision;
      String reference = scores.getProperty(key);
      Assert.assertNotNull(reference, key);
      var options = new PolicyOptions(device, precision, 1, kernels);
      try (var policy = ModelFamily.of(Series.PRISM).open(snapshot, variant, options)) {
        assertScoreBits(policy, rows, reference.split(";"), key);
      }
    }
  }

  private static void assertScoreBits(
      Policy policy, ObservationBatch rows, String[] references, String key) {
    Assert.assertEquals(rows.size(), references.length, key);
    var observation = new Observation(rows.variant());
    for (int row = 0; row < rows.size(); row++) {
      String label = key + " row " + row;
      int[] expected =
          Arrays.stream(references[row].split(" ")).mapToInt(HexFormat::fromHexDigits).toArray();
      float[] actual = policy.evaluate(observation.bind(rows, row)).values();
      Assert.assertEquals(actual.length, expected.length, label);
      for (int action = 0; action < actual.length; action++) {
        Assert.assertEquals(
            Float.floatToRawIntBits(actual[action]), expected[action], label + " action " + action);
      }
    }
  }
}
