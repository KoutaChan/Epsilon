package com.epsilon.prism.grp;

import ai.djl.engine.Engine;
import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Lineage;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.run.StopSignal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 本番の GRP のラベル行の指紋（CPU、torch の 1 スレッド＝学習の JVM と同じ）。{@code -Dprism.grp.rows.checkpoint}（grpCheckpoint の
 * ディレクトリ）・{@code -Dprism.grp.rows.calibration}（較正の表、既定は空）・{@code -Dprism.grp.rows.monitor}（phiMonitor、既定 SCORE）・
 * {@code -Dprism.grp.rows.source}（phiSource を読点で区切って並べる、既定 FIXED_GRP。SELF_GRP なら phiStart = GRP_CHECKPOINT の g000）で、
 * phiSource ごとに小さな系譜を 1 反復回し、コホート 1 の行の指紋（{@link RowHash}）を {@code -Dprism.grp.rows.out} に 1 行ずつ書く（1 つなら
 * 「行数 sha」、並べたら「phiSource 行数 sha」）。2 つの jar の指紋が同じならラベル行はビットで同じ。
 *
 * <p>{@code -Dprism.grp.checkConfig=<config.json,...>} は、config.json を check-config と同じ点検（{@link CheckConfigWorkflow#check}、run
 * なし、GPU なし）に通す。名前が *Test でないので、{@code -Dtest} で名指ししたときだけ走る。
 */
public class FixedGrpRowsBench {
  private static final Logger LOG = LoggerFactory.getLogger(FixedGrpRowsBench.class);

  @Test(timeOut = 3_600_000)
  public void fixedGrpRowHash() throws Exception {
    String checkpoint = System.getProperty("prism.grp.rows.checkpoint");
    String out = System.getProperty("prism.grp.rows.out");
    if (checkpoint == null || out == null)
      throw new SkipException("set -Dprism.grp.rows.checkpoint and -Dprism.grp.rows.out");
    String calibration = System.getProperty("prism.grp.rows.calibration", "");
    String monitor = System.getProperty("prism.grp.rows.monitor", "SCORE");
    String[] sources = System.getProperty("prism.grp.rows.source", "FIXED_GRP").split(",");
    Engine.getEngine("PyTorch");
    int threads = JniUtils.getNumThreads();
    JniUtils.setNumThreads(1);
    StringBuilder lines = new StringBuilder();
    try {
      for (String source : sources) lines.append(rowHash(source.trim(), sources.length > 1, checkpoint, calibration, monitor));
      Files.writeString(Path.of(out), lines, StandardCharsets.UTF_8);
    } finally {
      JniUtils.setNumThreads(threads);
    }
  }

  /** phiSource source の 1 反復の行の指紋の 1 行。 */
  private static String rowHash(String source, boolean named, String checkpoint, String calibration, String monitor)
      throws Exception {
    Path run = Files.createTempDirectory("prism-fixed-rows");
    try {
      Assert.assertEquals(
          Lineage.run(
              run, 1, EpsilonSettings.of(settings(source, checkpoint, calibration, monitor)), StopSignal.none()),
          0);
      RowHash.Digest digest = RowHash.of(CohortWriter.directory(run, 1));
      Assert.assertTrue(digest.rows() > 0, "no rows");
      LOG.info(
          "PRISM-GRP-ROWS phiSource {} phiMonitor {} rows {} sha256 {} game log {} phi.jsonl {}",
          source,
          monitor,
          digest.rows(),
          digest.sha256(),
          Files.exists(new GrpFiles(run).games(1)),
          Files.exists(new GrpFiles(run).phiHistory()) ? Files.readString(new GrpFiles(run).phiHistory()).strip() : "-");
      return (named ? source + " " : "") + digest.rows() + " " + digest.sha256() + "\n";
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void checkConfigCopies() throws Exception {
    String files = System.getProperty("prism.grp.checkConfig");
    if (files == null) throw new SkipException("set -Dprism.grp.checkConfig");
    for (String file : files.split(",")) {
      PhiEntry entry = CheckConfigWorkflow.check(Path.of(file.trim()), null);
      LOG.info("PRISM-GRP-CHECK-CONFIG {}: ok, Φ {}", file.trim(), entry);
    }
  }

  /** {@code GameLogRowsTest} の CPU の小さな系譜に、GRP と表と phiMonitor を入れ、1 コホート 4 局にしたもの。 */
  private static Map<String, String> settings(String source, String checkpoint, String calibration, String monitor) {
    String[] pairs = {
      "phiSource", source, "grpCheckpoint", checkpoint, "phiCalibration", calibration,
      "phiMonitor", monitor, "phiDrift", "OFF", "gamesPerCohort", "4",
      "gamesSchedule", "", "iterations", "10", "epsilonB", "0.005",
      "collectDevices", "cpu", "learnerDevice", "cpu", "maxActiveGames", "8", "advanceWorkers", "2",
      "encoders", "2", "slots", "1", "readyBatches", "2", "inferBatch", "256",
      "batchWaitMicros", "1000", "rowWriters", "1", "labelRateDiscardEarly", "0.003",
      "labelRateDiscardMid", "0.003", "labelRateDiscardLate", "0.003", "labelRateCommit", "0.03",
      "maxCandidates", "4", "rollinRateDiscard", "0.001", "rollinRateCommit", "0.015", "rhoBranch", "0.2",
      "averaging", "ema", "averageFrom", "2", "evaluateIterate", "false", "batchSize", "16",
      "decodeThreads", "2", "evalEvery", "0", "evalGames", "1"
    };
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put("epsilon.prism." + pairs[i], pairs[i + 1]);
    return values;
  }
}
