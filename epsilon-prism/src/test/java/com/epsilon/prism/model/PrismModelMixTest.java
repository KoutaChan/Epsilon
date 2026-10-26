package com.epsilon.prism.model;

import ai.djl.Device;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 重みを混ぜる: η = 0 は変えない、η = 1 は相手の写し、η = 1/2 はどちらから混ぜても同じ平均になる（保存した重みのバイトで比べる）。 */
public class PrismModelMixTest {
  private static final NetworkDevices CPU = NetworkDevices.of(Device.cpu());

  private static Path blank(Path root, String name, long seed) throws Exception {
    PrismConfig config =
        PrismConfig.fromSettings(
            EpsilonSettings.of(Map.of(PrismConfig.SETTINGS + "seed", Long.toString(seed))));
    Path directory = root.resolve(name);
    try (var model = new PrismModel(config.modelConfig(Device.cpu(), 16), "m0", CPU)) {
      model.save(directory);
    }
    return directory;
  }

  private static byte[] mixed(Path base, Path other, double eta, Path out) throws Exception {
    try (var model = PrismModel.load(base, CPU, false);
        var source = PrismModel.load(other, CPU, false)) {
      model.mixIn(source, eta);
      model.save(out);
    }
    return Files.readAllBytes(out.resolve("weights.bin"));
  }

  @Test
  public void mixesWeights() throws Exception {
    Path root = Files.createTempDirectory("prism-mix");
    try {
      Path a = blank(root, "a", 1), b = blank(root, "b", 2);
      byte[] weightsA = Files.readAllBytes(a.resolve("weights.bin"));
      byte[] weightsB = Files.readAllBytes(b.resolve("weights.bin"));
      Assert.assertNotEquals(weightsA, weightsB);
      Assert.assertEquals(mixed(a, b, 0, root.resolve("zero")), weightsA);
      Assert.assertEquals(mixed(a, b, 1, root.resolve("one")), weightsB);
      Assert.assertEquals(
          mixed(a, b, 0.5, root.resolve("ab")), mixed(b, a, 0.5, root.resolve("ba")));
      try (var model = PrismModel.load(a, CPU, false);
          var source = PrismModel.load(b, CPU, false)) {
        Assert.assertThrows(IllegalArgumentException.class, () -> model.mixIn(source, 1.5));
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
