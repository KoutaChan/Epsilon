package com.epsilon.prism.grp;

import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.workflow.WorkflowArguments;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * config.json を読むだけの点検（check-config）: 正しい設定は 0 で Φ の項目を解決し、古い形（phi の鍵）・知らない鍵・検証に落ちる組・読めない
 * GRP は 1（例外を投げない）。何も書かない。
 */
public class CheckConfigWorkflowTest {
  private static final Gson GSON = new GsonBuilder().serializeSpecialFloatingPointValues().create();

  private Path directory;

  @BeforeMethod
  public void createDirectory() throws IOException {
    directory = Files.createTempDirectory("prism-check-config");
  }

  @AfterMethod(alwaysRun = true)
  public void deleteDirectory() throws IOException {
    Snapshot.deleteTree(directory);
  }

  @Test
  public void validConfigsResolveTheirPhi() throws Exception {
    Path scoreRank = write("score-rank.json", tree(Map.of("phiSource", "SCORE_RANK")));
    Assert.assertEquals(execute(scoreRank), 0);
    Assert.assertEquals(CheckConfigWorkflow.check(scoreRank, null), PhiEntry.scoreRank());
    // SELF_GRP の点数の順位の出発は、phi.jsonl がまだなければ g000 の出どころを返す（GRP を読まない）
    Path self = write("self.json", tree(Map.of("phiSource", "SELF_GRP", "phiStart", "SCORE_RANK")));
    PhiEntry initial = CheckConfigWorkflow.check(self, null);
    Assert.assertEquals(initial.id(), PhiEntry.INITIAL_ID);
    Assert.assertEquals(initial.kind(), PhiEntryKind.SCORE_RANK);
    try (var listed = Files.list(directory)) {
      Assert.assertEquals(listed.count(), 2L, "check-config writes nothing");
    }
  }

  @Test
  public void brokenConfigsFailWithStatusOne() throws Exception {
    JsonObject old = tree(Map.of("phiSource", "SCORE_RANK"));
    old.remove("phiSource");
    old.addProperty("phi", "grp");
    Assert.assertEquals(execute(write("old.json", old)), 1);
    JsonObject unknown = tree(Map.of("phiSource", "SCORE_RANK"));
    unknown.addProperty("grpRefitEvery", 48);
    Assert.assertEquals(execute(write("unknown.json", unknown)), 1);
    // 検証に落ちる組（警報は採点が要る）
    Assert.assertEquals(
        execute(write("invalid.json", tree(Map.of("phiSource", "SCORE_RANK", "phiMonitor", "OFF", "phiDrift", "REPORT")))),
        1);
    // 読めない GRP
    Path missing = directory.resolve("no-grp");
    Assert.assertEquals(
        execute(write("fixed.json", tree(Map.of("phiSource", "FIXED_GRP", "grpCheckpoint", missing.toString())))), 1);
    Assert.assertEquals(execute(directory.resolve("absent.json")), 1);
  }

  /** check-config --config file の状態。 */
  private static int execute(Path file) throws Exception {
    return new CheckConfigWorkflow()
        .execute(
            new WorkflowArguments(
                "check-config", new String[] {"--config", file.toString()}, EpsilonSettings.defaults()));
  }

  /** 既定の設定に overrides を重ねた config.json の木。 */
  private static JsonObject tree(Map<String, String> overrides) {
    Map<String, String> values = new LinkedHashMap<>();
    overrides.forEach((key, value) -> values.put(PrismConfig.SETTINGS + key, value));
    return JsonParser.parseString(GSON.toJson(PrismConfig.fromSettings(EpsilonSettings.of(values))))
        .getAsJsonObject();
  }

  private Path write(String name, JsonObject tree) throws IOException {
    return Files.writeString(directory.resolve(name), GSON.toJson(tree));
  }
}
