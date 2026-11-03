package com.epsilon.prism.grp;

import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Lineage;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.run.StopSignal;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 系譜の記録の口（CPU）: 同じ設定・同じ種で train を 1 反復ずつ 2 回、phiMonitor = OFF と RECORD で回すと、コホート 1 の行（rows-v2）の指紋が
 * ビットで同じ。RECORD の run だけが grp/games/c00001.bin.gz を書き、その局末の数は iterations.jsonl の収集の hands と同じで、先頭は使った Φ
 * （点数の順位）と行動方策の床。OFF の run は grp を作らない。
 */
public class GameLogRowsTest {

  @Test(timeOut = 1_800_000)
  public void recordingKeepsTheCohortRowsBitwise() throws Exception {
    Path off = Files.createTempDirectory("prism-gamelog-off");
    Path on = Files.createTempDirectory("prism-gamelog-on");
    try {
      Assert.assertEquals(Lineage.run(off, 1, EpsilonSettings.of(settings("OFF")), StopSignal.none()), 0);
      Assert.assertEquals(Lineage.run(on, 1, EpsilonSettings.of(settings("RECORD")), StopSignal.none()), 0);
      RowHash.Digest unrecorded = RowHash.of(CohortWriter.directory(off, 1));
      RowHash.Digest recorded = RowHash.of(CohortWriter.directory(on, 1));
      Assert.assertTrue(unrecorded.rows() > 0, "no rows");
      Assert.assertEquals(recorded, unrecorded);
      Assert.assertFalse(Files.exists(new GrpFiles(off).directory()), "phiMonitor = OFF records nothing");
      CohortGames games = GameLog.read(new GrpFiles(on).games(1));
      JsonObject collection = iteration(on).getAsJsonObject("collection").getAsJsonObject("counts");
      Assert.assertEquals(games.games().size(), collection.get("games").getAsLong());
      Assert.assertEquals(games.handEnds(), collection.get("hands").getAsLong());
      Assert.assertEquals(games.header().cohort(), 1);
      Assert.assertEquals(games.header().phiId(), PhiEntry.SCORE_RANK_ID);
      Assert.assertEquals(games.header().epsilonB(), 0.005);
      Assert.assertEquals(games.header().mainNonGreedy(), collection.get("mainNonGreedy").getAsLong());
      Assert.assertEquals(iteration(on).getAsJsonObject("phi").get("id").getAsString(), PhiEntry.SCORE_RANK_ID);
    } finally {
      Snapshot.deleteTree(off);
      Snapshot.deleteTree(on);
    }
  }

  /** 1 反復目の iterations.jsonl の行。 */
  private static JsonObject iteration(Path run) throws Exception {
    List<String> lines = Files.readAllLines(run.resolve("reports/iterations.jsonl"), StandardCharsets.UTF_8);
    return JsonParser.parseString(lines.get(0)).getAsJsonObject();
  }

  /**
   * {@code LineageSmokeTest} と同じ CPU の小さな系譜（score-rank の Φ、1 コホート 2 局）に phiMonitor だけを替えたもの。評価は回さない。
   *
   * @param monitor OFF か RECORD（警報は採点が要るので OFF）
   */
  private static Map<String, String> settings(String monitor) {
    String[] pairs = {
      "phiSource", "SCORE_RANK", "phiMonitor", monitor, "phiDrift", "OFF", "gamesPerCohort", "2",
      "gamesSchedule", "", "iterations", "10", "epsilonB", "0.005",
      "collectDevices", "cpu", "learnerDevice", "cpu", "maxActiveGames", "8", "advanceWorkers", "2",
      "encoders", "2", "slots", "1", "readyBatches", "2", "inferBatch", "256",
      "batchWaitMicros", "1000", "rowWriters", "1", "labelRateDiscardEarly", "0.003",
      "labelRateDiscardMid", "0.003", "labelRateDiscardLate", "0.003", "labelRateCommit", "0.03",
      "maxCandidates", "4", "rollinRateDiscard", "0.001", "rollinRateCommit", "0.015", "rhoBranch", "0.2",
      "averaging", "ema", "averageFrom", "2", "batchSize", "16", "decodeThreads", "2",
      "evaluation.3xH.every", "0"
    };
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put("epsilon.prism." + pairs[i], pairs[i + 1]);
    return values;
  }
}
