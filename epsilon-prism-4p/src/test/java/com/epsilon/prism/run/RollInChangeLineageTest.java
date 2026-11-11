package com.epsilon.prism.run;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Change;
import com.epsilon.prism.Main;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 続いている系譜に、反復の区切りで鳴きのアームを入れるテスト（CPU）。鳴ける判断のロールインを切った run を 1 反復回し、
 * settings change で、鳴ける判断の β を 0.15 に、その写しの ρ_br を 0.6 に、上限を 4 にするファイルを重ねて、もう 1 反復回す。
 * 設定の行 2 は反復 2 から効き、変えたキーは COLLECT に分類される。
 * 反復 1 は鳴きの根も写しも作らない。反復 2 は鳴きの写しを作り、写しの中のラベル行は写しごとに 4 行まで。反復の記録の内訳にも
 * 同じことが出る。
 */
public class RollInChangeLineageTest {

  @Test(groups = "slow", timeOut = 3_600_000)
  public void callArmTakesEffectAtTheNextIteration() throws Exception {
    Path run = Files.createTempDirectory("prism-rollin-lineage");
    try {
      TestSettings.create(
          run, TestSettings.cpuLineage("phi.source=SCORE_RANK", "schedule.iterations=2", "labels.rollin.CALL=0"));
      Assert.assertEquals(PrismRun.runner(run, 1, () -> false).train(), Ending.LIMIT);

      Path apply = run.resolve("call-arm.toml");
      Files.writeString(
          apply,
          """
          [labels.rollin]
          CALL = 0.15

          [labels.copies.CALL]
          copyLabelRate = 0.6
          maxCopyLabels = 4
          """,
          StandardCharsets.UTF_8);
      Assert.assertEquals(
          Main.run(
              new String[] {"settings", "change", "--run", run.toString(), "--apply", apply.toString(), "--reason", "test"}),
          Exit.OK.code());
      SettingsRow row = new SettingsRows(new RunDirectory(run)).read().getLast();
      Assert.assertEquals(row.row(), 2);
      Assert.assertEquals(row.from(), 2);
      for (String key : List.of("labels.rollin.CALL", "labels.copies.CALL.copyLabelRate", "labels.copies.CALL.maxCopyLabels"))
        Assert.assertTrue(
            row.changes().stream().anyMatch(changed -> changed.key().equals(key) && changed.change() == Change.COLLECT),
            key + " in " + row.changes());

      Assert.assertEquals(PrismRun.runner(run, null, () -> false).train(), Ending.FINISHED);
      List<JsonObject> iterations = lines(new RunDirectory(run).report(RunDirectory.ITERATIONS));
      Assert.assertEquals(iterations.size(), 2);
      JsonObject before = breakdown(iterations.get(0)), after = breakdown(iterations.get(1));
      Assert.assertEquals(count(before, "copies", "CALL"), 0, before.toString());
      Assert.assertEquals(count(before, "sameDeviations", "CALL"), 0, before.toString());
      Assert.assertFalse(before.getAsJsonObject("copyLabels").has("CALL"), before.toString());
      long copies = count(after, "copies", "CALL");
      Assert.assertTrue(copies > 0, after.toString());
      JsonObject callLabels = after.getAsJsonObject("copyLabels").getAsJsonObject("CALL");
      long labels = 0;
      if (callLabels != null) for (String c : callLabels.keySet()) labels += callLabels.get(c).getAsLong();
      Assert.assertTrue(labels <= 4 * copies, labels + " labels in " + copies + " copies");

      Assert.assertEquals(callCopyRows(CohortWriter.directory(run, 1)).size(), 0);
      Map<String, Integer> perCopy = callCopyRows(CohortWriter.directory(run, 2));
      Assert.assertEquals(perCopy.values().stream().mapToLong(Integer::longValue).sum(), labels);
      for (Map.Entry<String, Integer> copy : perCopy.entrySet())
        Assert.assertTrue(copy.getValue() <= 4, copy.toString());
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /** 反復の記録から、収集の段の内訳を返す。 */
  private static JsonObject breakdown(JsonObject iteration) {
    return iteration.getAsJsonObject("steps").getAsJsonObject(PrismStep.COLLECT.name()).getAsJsonObject("breakdown");
  }

  private static long count(JsonObject breakdown, String table, String c) {
    JsonObject counts = breakdown.getAsJsonObject(table);
    return counts != null && counts.has(c) ? counts.get(c).getAsLong() : 0;
  }

  /** コホートの中で、根が CALL の写しにあるラベル行の数を、写し（半荘と根の判断の組）ごとに数える。 */
  private static Map<String, Integer> callCopyRows(Path cohort) throws Exception {
    List<Row> rows = cohortRows(cohort);
    Map<String, Row> roots = new HashMap<>();
    for (Row row : rows) if (row.source() == Row.Source.ROOT) roots.put(row.gameId() + ":" + row.decisionId(), row);
    Map<String, Integer> perCopy = new HashMap<>();
    for (Row row : rows) {
      if (row.source() != Row.Source.ROLLIN) continue;
      String key = row.gameId() + ":" + row.rootDecisionId();
      Row root = roots.get(key);
      Assert.assertNotNull(root, "root of " + key);
      if (root.decisionClass() == DecisionClass.CALL) perCopy.merge(key, 1, Integer::sum);
    }
    return perCopy;
  }

  /** 確定したコホートの全ての行を返す。 */
  private static List<Row> cohortRows(Path cohort) throws Exception {
    CohortWriter.Manifest manifest = CohortWriter.readManifest(cohort);
    RowCodec codec = new RowCodec(manifest.ledgerFeatures(), manifest.variant());
    List<Row> rows = new ArrayList<>();
    for (CohortWriter.Shard shard : manifest.shards()) {
      var frames = ByteBuffer.wrap(Files.readAllBytes(cohort.resolve(shard.file())));
      while (frames.hasRemaining()) rows.add(codec.decode(frames));
    }
    Assert.assertEquals(rows.size(), manifest.rows());
    return rows;
  }

  private static List<JsonObject> lines(Path path) throws Exception {
    List<JsonObject> result = new ArrayList<>();
    for (String line : Files.readAllLines(path, StandardCharsets.UTF_8))
      if (!line.isBlank()) result.add(JsonParser.parseString(line).getAsJsonObject());
    return result;
  }
}
