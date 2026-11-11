package com.epsilon.prism.grp;

import com.epsilon.agent.run.Ending;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.PrismRun;
import com.epsilon.prism.run.Snapshot;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 半荘の記録を書いても行が変わらないことを確かめる（CPU）。同じ設定・同じ乱数の種で、train を 1 反復ずつ 2 回回す。1 回は点数の順位の Φ で、
 * もう 1 回は点数の順位から始める自前の GRP（半荘を記録する）で回し、コホート 1 の行（rows）のフィンガープリントがビットまで一致することを確かめる。
 * grp/records/c00001.bin を書くのは自前の GRP の run だけ。その半荘の数は iterations.jsonl の収集の段の games と同じで、半荘の数と局末の数の
 * 和は hands と同じ。ヘッダーには使った Φ（g000）が入る。点数の順位の run は grp のディレクトリを作らない。
 */
public class GameLogRowsTest {

  @Test(groups = "slow", timeOut = 1_800_000)
  public void recordingKeepsTheCohortRowsBitwise() throws Exception {
    Path off = Files.createTempDirectory("prism-gamelog-off");
    Path on = Files.createTempDirectory("prism-gamelog-on");
    try {
      String[] lineage = TestSettings.cpuLineage(TestSettings.EMA_FROM_2);
      TestSettings.create(off, TestSettings.concat(lineage, "phi.source=SCORE_RANK"));
      TestSettings.create(on, TestSettings.concat(lineage, GrpFixtures.scoreStartAssignments()));
      Assert.assertEquals(PrismRun.runner(off, 1, () -> false).train(), Ending.LIMIT);
      Assert.assertEquals(PrismRun.runner(on, 1, () -> false).train(), Ending.LIMIT);
      RowHash.Digest unrecorded = RowHash.of(CohortWriter.directory(off, 1));
      RowHash.Digest recorded = RowHash.of(CohortWriter.directory(on, 1));
      Assert.assertTrue(unrecorded.rows() > 0, "no rows");
      Assert.assertEquals(recorded, unrecorded);
      Assert.assertFalse(Files.exists(new GrpFiles(off).directory()), "a SCORE_RANK run records nothing");
      GameCohort games = GameRecordCodec.read(new GrpFiles(on).records(1));
      JsonObject collection = iteration(on).getAsJsonObject("steps").getAsJsonObject("COLLECT");
      Assert.assertEquals(games.games().size(), collection.get("games").getAsLong());
      long handEnds = games.games().size();
      for (GameRecord game : games.games()) handEnds += game.handEnds().size();
      Assert.assertEquals(handEnds, collection.get("hands").getAsLong());
      Assert.assertEquals(games.cohort(), 1);
      Assert.assertEquals(games.phiId(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(phiId(on), PhiEntry.INITIAL_ID);
      Assert.assertEquals(phiId(off), PhiEntry.SCORE_RANK_ID);
    } finally {
      Snapshot.deleteTree(off);
      Snapshot.deleteTree(on);
    }
  }

  /** iterations.jsonl から、1 反復目の行を返す。 */
  private static JsonObject iteration(Path run) throws Exception {
    List<String> lines = Files.readAllLines(run.resolve("reports/iterations.jsonl"), StandardCharsets.UTF_8);
    return JsonParser.parseString(lines.get(0)).getAsJsonObject();
  }

  /** 1 反復目に Φ の段が決めた項目の id を返す。 */
  private static String phiId(Path run) throws Exception {
    return iteration(run)
        .getAsJsonObject("steps")
        .getAsJsonObject("PHI")
        .getAsJsonObject("value")
        .getAsJsonObject("entry")
        .get("id")
        .getAsString();
  }
}
