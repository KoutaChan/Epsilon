package com.epsilon.prism.grp;

import ai.djl.engine.Engine;
import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.prism.collect.Behaviour;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.collect.HandValue;
import com.epsilon.prism.collect.RowSink;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.core.CrnScheme;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.learn.LedgerFit;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.run.TrainingResources;
import com.epsilon.util.SeedMixer;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * g000 の煙（CPU）: SELF_GRP＋phiStart = GRP_CHECKPOINT で今の GRP と表（と温度）を指すと、小さな収集のラベル行は FIXED_GRP とビットで同じ。
 * 同じ JVM・torch の 1 スレッド・同じ maxBatch で、学習の資源（{@link TrainingResources}）から反復 1 の Φ を取って同じ半荘を集める。
 *
 * <p>既定は白紙の GRP と合成の表。本番の GRP と表で確かめるときは -Dprism.grp.g000.checkpoint（grpCheckpoint のディレクトリ）・
 * -Dprism.grp.g000.calibration（表）・-Dprism.grp.g000.temperatures（既定は本番と同じ空）を渡す。
 */
public class SelfGrpG000SmokeTest {
  private static final Logger LOG = LoggerFactory.getLogger(SelfGrpG000SmokeTest.class);
  private static final long SEED = 7_001L;

  /** ラベル行・ロールイン・写しの中の行がどれも出る率（本番より高い）。 */
  private static final Sampling SAMPLING =
      new Sampling(0.25, 0.05, 0.05, 0.05, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.V1);

  /** s を手の ID と席のハッシュに置き、σ をその argmax にする inline 方策（収集の試験と同じ。retain なら p0 も書く）。 */
  private static final class HashedScorePolicy implements PolicyService {
    @Override
    public int version() {
      return 1;
    }

    @Override
    public boolean inline() {
      return true;
    }

    @Override
    public void evaluateInline(GameState state, EngineDecisionPoint point, PolicyRequest request) {
      int n = request.legal().size();
      float[] scores = new float[n];
      int best = 0;
      for (int a = 0; a < n; a++) {
        scores[a] = (float) CrnSeeds.unit(SeedMixer.mix64(request.legal().get(a).toIndex() * 31L + request.seat()));
        if (scores[a] > scores[best]) best = a;
      }
      float[] sigma = new float[n];
      sigma[best] = 1f;
      request.scores = scores;
      request.sigma = sigma;
      if (request.retain()) {
        float[] prior = new float[n];
        double sum = 0;
        for (int a = 0; a < n; a++) {
          prior[a] = 1f + (float) CrnSeeds.unit(SeedMixer.mix64(request.legal().get(a).toIndex() * 17L + 5));
          sum += prior[a];
        }
        for (int a = 0; a < n; a++) prior[a] = (float) (prior[a] / sum);
        request.prior = prior;
      }
    }
  }

  @Test(timeOut = 1_800_000)
  public void g000LabelsAreBitwiseTheFixedGrpLabels() throws Exception {
    Path work = Files.createTempDirectory("prism-g000");
    // 学習の JVM と同じ torch の 1 スレッド（prism.sh）。終わったら戻す
    Engine.getEngine("PyTorch");
    int threads = JniUtils.getNumThreads();
    JniUtils.setNumThreads(1);
    try {
      String checkpoint = System.getProperty("prism.grp.g000.checkpoint");
      String calibration = System.getProperty("prism.grp.g000.calibration");
      String temperatures = System.getProperty("prism.grp.g000.temperatures", "");
      if (checkpoint == null) {
        Path decision = work.resolve("decision");
        GrpFixtures.checkpoint(decision);
        checkpoint = decision.toString();
        temperatures = "1.0,1.1,1.2";
      }
      if (calibration == null) {
        Path table = work.resolve("table.json");
        StringBuilder cells = new StringBuilder();
        for (int stage = 0; stage < 9; stage++) {
          if (stage > 0) cells.append(',');
          cells.append('"').append(stage).append("\": [");
          for (int bin = 0; bin < 24; bin++) cells.append(bin == 0 ? "" : ",").append(0.001 * (stage + 1) * (12 - bin));
          cells.append(']');
        }
        Files.writeString(
            table,
            "{\"stages\": [0,1,2,3,4,5,6,7,8], \"lo\": -1, \"hi\": 0.6667, \"bins\": 24, \"table\": {" + cells + "}}");
        calibration = table.toString();
      }
      String[] common = {
        "grpCheckpoint", checkpoint, "phiCalibration", calibration, "phiTemperatures", temperatures,
        "encoders", "2", "slots", "1", "readyBatches", "2", "maxActiveGames", "8", "advanceWorkers", "2",
        "batchWaitMicros", "1000"
      };
      PrismConfig fixed = GrpFixtures.config(concat(common, "phiSource", "FIXED_GRP", "phiMonitor", "OFF", "phiDrift", "OFF"));
      PrismConfig self =
          GrpFixtures.config(
              concat(common, "phiSource", "SELF_GRP", "phiStart", "GRP_CHECKPOINT", "phiMonitor", "RECORD", "phiDrift", "OFF"));
      fixed.validateTraining();
      self.validateTraining();
      Collected expected = collect(fixed, work.resolve("fixed"));
      Collected actual = collect(self, work.resolve("self"));
      Assert.assertEquals(actual.entry().id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(actual.entry().kind(), PhiEntryKind.FIXED_CHECKPOINT);
      Assert.assertEquals(Path.of(actual.entry().path()), Path.of(expected.entry().path()).toAbsolutePath().normalize());
      Assert.assertEquals(actual.entry().sha256(), expected.entry().sha256());
      Assert.assertTrue(expected.evaluated() > 0, "the GRP valued some hand ends");
      Assert.assertEquals(actual.evaluated(), expected.evaluated());
      Assert.assertTrue(expected.rows().size() > 0);
      Assert.assertEquals(actual.rows(), expected.rows(), "SELF_GRP g000 must give the FIXED_GRP label rows bit for bit");
      LOG.info(
          "g000 smoke: {} rows, {} hand ends valued by the GRP, sha {} — identical",
          expected.rows().size(),
          expected.evaluated(),
          expected.entry().sha256());
    } finally {
      JniUtils.setNumThreads(threads);
      Snapshot.deleteTree(work);
    }
  }

  private record Collected(PhiEntry entry, List<String> rows, long evaluated) {}

  /** run の資源から反復 1 の Φ を取り、半荘 100..107 を集めて行を読み戻す（行の並びに依らない正準）。 */
  private static Collected collect(PrismConfig config, Path run) throws Exception {
    Files.createDirectories(run);
    Path cohort = run.resolve("rows");
    List<Long> ids = new ArrayList<>();
    for (long id = 100; id < 108; id++) ids.add(id);
    try (var resources = TrainingResources.open(config, GrpFixtures.PROFILE, run);
        var writer = new CohortWriter(cohort, 1, 0, 2, new LedgerFit.Stream(2))) {
      ActivePhi phi = resources.phi(1);
      HandValue value = phi.value();
      var plan = new CollectionDriver.Cohort(SEED, SEED * 3, ids, GrpFixtures.PROFILE, SAMPLING, FLOOR, null, null);
      resources.driver().collect(plan, new HashedScorePolicy(), value, new RowSink(writer));
      writer.seal();
      CohortWriter.Manifest manifest = CohortWriter.readManifest(cohort);
      RowCodec codec = new RowCodec(manifest.ledgerFeatures());
      List<String> rows = new ArrayList<>();
      for (CohortWriter.Shard shard : manifest.shards()) {
        var frames = ByteBuffer.wrap(Files.readAllBytes(cohort.resolve(shard.file())));
        while (frames.hasRemaining()) {
          Row row = codec.decode(frames);
          rows.add(Base64.getEncoder().encodeToString(new RowCodec(row.ledgerFeatures()).encode(row)));
        }
      }
      Collections.sort(rows);
      return new Collected(phi.entry(), rows, value.stats().evaluated());
    }
  }

  private static String[] concat(String[] head, String... tail) {
    String[] all = new String[head.length + tail.length];
    System.arraycopy(head, 0, all, 0, head.length);
    System.arraycopy(tail, 0, all, head.length, tail.length);
    return all;
  }
}
