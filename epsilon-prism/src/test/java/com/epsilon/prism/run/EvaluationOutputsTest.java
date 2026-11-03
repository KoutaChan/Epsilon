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
 * 評価の出力の印（.running → tsv・log の差し替え → .done、失敗は .failed）、測っていない評価の選び方（間隔・始め・every = 0・.done と .failed を
 * 飛ばす・バージョンの順）、評価の節の検査（名前・相手・数の範囲）。
 */
public class EvaluationOutputsTest {
  private Path root;

  @BeforeMethod
  public void createRoot() throws IOException {
    root = Files.createTempDirectory("prism-evaluation-outputs").toAbsolutePath();
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRoot() throws IOException {
    Snapshot.deleteTree(root);
  }

  private static PrismConfig config(String... pairs) {
    return config(pairs(pairs));
  }

  private static PrismConfig config(Map<String, String> fields) {
    Map<String, String> values = new LinkedHashMap<>();
    fields.forEach((key, value) -> values.put(Evaluation.SETTINGS + key, value));
    return PrismConfig.fromSettings(EpsilonSettings.of(values));
  }

  private static Map<String, String> pairs(String... pairs) {
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put(pairs[i], pairs[i + 1]);
    return values;
  }

  private static List<String> names(List<EvaluationOutputs> pending) {
    return pending.stream().map(outputs -> outputs.name() + ":" + outputs.version()).toList();
  }

  @Test
  public void writesThroughTemporariesAndMarksTheVersion() throws IOException {
    var outputs = new EvaluationOutputs(root.resolve(EvaluationOutputs.DIRECTORY), "x1n3", 408);
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
    var failing = new EvaluationOutputs(outputs.directory(), "x1n3", 416);
    failing.begin();
    failing.fail("failed\n");
    Assert.assertTrue(Files.exists(failing.failed()) && failing.settled());
    Assert.assertFalse(Files.exists(failing.running()) || Files.exists(failing.done()));
    Assert.assertEquals(Files.readString(failing.log()), "failed\n");
  }

  @Test
  public void pendingSkipsDoneAndFailedVersions() throws IOException {
    PrismConfig config =
        config(
            "3xH.every", "0",
            "x1n3.opponent", NanoTable.HEURISTIC, "x1n3.games", "131072", "x1n3.every", "8", "x1n3.from", "170");
    config.validate();
    Assert.assertEquals(names(EvaluationOutputs.pending(root, config.evaluations(), 175)), List.of());
    Assert.assertEquals(
        names(EvaluationOutputs.pending(root, config.evaluations(), 200)),
        List.of("x1n3:176", "x1n3:184", "x1n3:192", "x1n3:200"));
    Evaluation x1n3 = config.evaluations().get(1);
    Files.createDirectories(root.resolve(EvaluationOutputs.DIRECTORY));
    Files.writeString(EvaluationOutputs.of(root, x1n3, 184).done(), "");
    Files.writeString(EvaluationOutputs.of(root, x1n3, 192).failed(), "");
    Files.writeString(EvaluationOutputs.of(root, x1n3, 200).running(), "1\n");
    Assert.assertEquals(
        names(EvaluationOutputs.pending(root, config.evaluations(), 207)), List.of("x1n3:176", "x1n3:200"));
    // バージョンの古い順、同じバージョンでは名前の順
    PrismConfig both =
        config(
            "3xH.from", "3",
            "x1n3.opponent", NanoTable.HEURISTIC, "x1n3.games", "2", "x1n3.every", "2", "x1n3.from", "1");
    Assert.assertEquals(
        names(EvaluationOutputs.pending(root.resolve("other"), both.evaluations(), 4)),
        List.of("x1n3:2", "3xH:3", "3xH:4", "x1n3:4"));
  }

  @Test
  public void evaluationSectionsAreValidated() {
    String nano = NanoTable.NANO + root.resolve("latest");
    config("x1n3.opponent", nano, "x1n3.games", "131072", "x1n3.every", "8", "x1n3.from", "1").validate();
    config("3xH.every", "0").validate();
    String[][] wrongs = {
      {"x1n3.opponent", "nano:latest"},
      {"x1n3.opponent", root.toString()},
      {"x1n3.opponent", ""},
      {"x1n3.games", "0"},
      {"x1n3.every", "-1"},
      {"x1n3.from", "0"}
    };
    for (String[] wrong : wrongs) {
      Map<String, String> fields = pairs("x1n3.opponent", nano, "x1n3.games", "2", "x1n3.every", "1", "x1n3.from", "1");
      fields.put(wrong[0], wrong[1]);
      Assert.assertThrows(IllegalArgumentException.class, () -> config(fields).validate());
    }
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> config("x-1.opponent", nano, "x-1.games", "2", "x-1.every", "1", "x-1.from", "1").validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("3xH.games", "many"));
  }
}
