package com.epsilon.prism.run;

import com.epsilon.prism.config.settings.EpsilonSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * x1n3 の出力の印（.running → tsv・log の差し替え → .done、失敗は .failed）、測っていないバージョンの選び方（間隔・始め・.done と .failed を飛ばす）、
 * x1n3 の鍵の検査（ON では絶対パスの相手と出力先が要る）。
 */
public class X1n3OutputsTest {
  private Path directory;

  @BeforeMethod
  public void createDirectory() throws IOException {
    directory = Files.createTempDirectory("prism-x1n3-outputs").toAbsolutePath();
  }

  @AfterMethod(alwaysRun = true)
  public void deleteDirectory() throws IOException {
    Snapshot.deleteTree(directory);
  }

  private static PrismConfig config(String... pairs) {
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put(PrismConfig.SETTINGS + pairs[i], pairs[i + 1]);
    return PrismConfig.fromSettings(EpsilonSettings.of(values));
  }

  @Test
  public void writesThroughTemporariesAndMarksTheVersion() throws IOException {
    var outputs = new X1n3Outputs(directory.resolve("v2b"), 408);
    Assert.assertEquals(outputs.tsv().getFileName().toString(), "v408-x1n3.tsv");
    Assert.assertFalse(outputs.settled());
    outputs.begin();
    Assert.assertTrue(Files.exists(outputs.running()));
    Assert.assertEquals(
        Files.readString(outputs.running()).trim(), Long.toString(ProcessHandle.current().pid()));
    Files.writeString(outputs.tsv().resolveSibling("v408-x1n3.tsv.tmp"), "partial");
    outputs.commit("1\t2\t0.5\n", "complete\n");
    Assert.assertEquals(Files.readString(outputs.tsv()), "1\t2\t0.5\n");
    Assert.assertEquals(Files.readString(outputs.log()), "complete\n");
    Assert.assertTrue(Files.exists(outputs.done()) && outputs.settled());
    Assert.assertFalse(Files.exists(outputs.running()));
    try (var listed = Files.list(outputs.directory())) {
      Assert.assertTrue(listed.noneMatch(path -> path.toString().endsWith(".tmp")));
    }
    var failing = new X1n3Outputs(outputs.directory(), 416);
    failing.begin();
    failing.fail("failed\n");
    Assert.assertTrue(Files.exists(failing.failed()) && failing.settled());
    Assert.assertFalse(Files.exists(failing.running()) || Files.exists(failing.done()));
    Assert.assertEquals(Files.readString(failing.log()), "failed\n");
  }

  @Test
  public void pendingSkipsDoneAndFailedVersions() throws IOException {
    String dir = directory.toString();
    Assert.assertEquals(X1n3Outputs.pending(config("x1n3Dir", dir), 400), List.of());
    PrismConfig config =
        config("x1n3", "ON", "x1n3Every", "8", "x1n3From", "170", "x1n3Opponent", NanoTable.HEURISTIC, "x1n3Dir", dir);
    config.validate();
    Assert.assertEquals(X1n3Outputs.pending(config, 175), List.of());
    Assert.assertEquals(X1n3Outputs.pending(config, 200), List.of(176, 184, 192, 200));
    Files.writeString(new X1n3Outputs(directory, 184).done(), "");
    Files.writeString(new X1n3Outputs(directory, 192).failed(), "");
    Files.writeString(new X1n3Outputs(directory, 200).running(), "1\n");
    Assert.assertEquals(X1n3Outputs.pending(config, 207), List.of(176, 200));
    Assert.assertEquals(
        X1n3Outputs.pending(config("x1n3", "ON", "x1n3Every", "1", "x1n3From", "3", "x1n3Opponent", "heuristic", "x1n3Dir", dir), 4),
        List.of(3, 4));
  }

  @Test
  public void x1n3KeysAreValidated() {
    String dir = directory.toString(), nano = NanoTable.NANO + directory.resolve("latest");
    PrismConfig defaults = PrismConfig.fromSettings(EpsilonSettings.defaults());
    Assert.assertEquals(defaults.x1n3(), X1n3.OFF);
    Assert.assertEquals(defaults.x1n3Every(), 8);
    Assert.assertEquals(defaults.x1n3From(), 1);
    Assert.assertEquals(defaults.x1n3Games(), 131072);
    Assert.assertEquals(defaults.x1n3Opponent(), "");
    Assert.assertEquals(defaults.x1n3Dir(), "");
    config("x1n3", "ON", "x1n3Opponent", nano, "x1n3Dir", dir).validate();
    config("x1n3Opponent", nano, "x1n3Dir", dir).validate();
    Assert.assertThrows(IllegalArgumentException.class, () -> config("x1n3", "on"));
    Assert.assertThrows(IllegalArgumentException.class, () -> config("x1n3", "ON").validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("x1n3", "ON", "x1n3Opponent", nano).validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("x1n3", "ON", "x1n3Dir", dir).validate());
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> config("x1n3", "ON", "x1n3Opponent", "nano:latest", "x1n3Dir", dir).validate());
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> config("x1n3", "ON", "x1n3Opponent", dir, "x1n3Dir", dir).validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("x1n3Dir", "~/v2b").validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("x1n3Every", "0").validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("x1n3From", "0").validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("x1n3Games", "0").validate());
  }
}
