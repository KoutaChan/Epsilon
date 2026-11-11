package com.epsilon.prism.grp;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.grp.Phi;
import com.epsilon.grp.PhiTable;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.collect.Collector;
import com.epsilon.prism.collect.RowSink;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.collect.TestSampling;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.run.TrainingResources;
import com.epsilon.prism.settings.PrismSettings;
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
 * g000 のスモークテスト（CPU）。自前の GRP で g000 の元の GRP に今の GRP と表を指定すると、小さな収集で作るラベル行が、固定の GRP の
 * ときとビットで一致する。どちらの設定でも、学習のリソース（{@link TrainingResources}）から反復 1 の Φ を取り、同じ半荘を集める。
 *
 * <p>既定では、乱数で初期化した小さな GruSinkhorn と合成した表を使う。本番の GRP と表で確かめるときは、-Dprism.grp.g000.checkpoint
 * （GRP のディレクトリそのもの）と -Dprism.grp.g000.table（表）を渡す。
 */
public class SelfGrpG000SmokeTest {
  private static final Logger LOG = LoggerFactory.getLogger(SelfGrpG000SmokeTest.class);
  private static final long SEED = 7_001L;

  /** ラベル行、ロールイン、写しの中の行がどれも出るように、本番より高くした率。 */
  private static final Sampling SAMPLING =
      TestSampling.grouped(0.25, 0.05, 0.05, 0.05, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.SINGLE_RACE);

  /** 裏ドラは実際にめくれた値でラベルを作り、局の 5% では期待値も求めて監査する。 */
  private static final UraSettings URA = new UraSettings.Realized(0.05);

  /**
   * 収集のテストと同じ inline の方策。s を手の ID と席から作ったハッシュの値にし、σ をその argmax の手に全て置く。p0 を残す要求なら、
   * p0 も書く。
   */
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
    public void evaluateInline(PolicyRequest request) {
      int n = request.legalCount();
      float[] scores = new float[n];
      int best = 0;
      for (int a = 0; a < n; a++) {
        scores[a] = (float) CrnSeeds.unit(SeedMixer.mix64(request.legal()[a] * 31L + request.seat()));
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
          prior[a] = 1f + (float) CrnSeeds.unit(SeedMixer.mix64(request.legal()[a] * 17L + 5));
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
    try {
      String checkpoint = System.getProperty("prism.grp.g000.checkpoint");
      String table = System.getProperty("prism.grp.g000.table");
      if (checkpoint == null) checkpoint = GrpFixtures.gru(work.resolve("model"), 17).toString();
      if (table == null) {
        Path file = work.resolve("table.json");
        double[][] corrections = new double[9][24];
        for (int stage = 0; stage < 9; stage++)
          for (int bin = 0; bin < 24; bin++) corrections[stage][bin] = 0.001 * (stage + 1) * (12 - bin);
        PhiTable.of(TENHOU_4P, new double[] {1.0, 1.1, 1.2}, -1, 2.0 / 3, corrections).write(file);
        table = file.toString();
      }
      String[] common = {
        "inference.encoders=2", "inference.slots=1", "inference.readyBatches=2", "table.activeGames=8", "table.workers=2",
        "table.batchWaitMicros=1000"
      };
      PrismSettings fixed =
          TestSettings.of(concat(common, "phi.source=FIXED_GRP", "phi.grp=" + checkpoint, "phi.table=" + table));
      PrismSettings self =
          TestSettings.of(
              concat(
                  common, "phi.source=SELF_GRP", "phi.start.kind=GRP_CHECKPOINT", "phi.start.grp=" + checkpoint,
                  "phi.start.table=" + table, "phi.grp.monitor=RECORD", "phi.grp.refit=MANUAL", "phi.grp.switching=PROPOSE"));
      for (PrismSettings settings : List.of(fixed, self))
        Assert.assertEquals(
            PhiControl.problems(new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed())), List.of());
      Collected expected = collect(fixed, work.resolve("fixed"));
      Collected actual = collect(self, work.resolve("self"));
      Assert.assertEquals(expected.entry().id(), PhiEntry.FIXED_ID);
      Assert.assertEquals(actual.entry().id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(actual.entry().kind(), PhiEntryKind.NETWORK);
      Assert.assertTrue(actual.entry().sameValue(expected.entry()), actual.entry() + " vs " + expected.entry());
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
      Snapshot.deleteTree(work);
    }
  }

  private record Collected(PhiEntry entry, List<String> rows, long evaluated) {}

  /** run のリソースから反復 1 の Φ を取り、半荘 100〜107 を集めて行を読み戻す。行は並べ替えて、書いた順に依らない形にして返す。 */
  private static Collected collect(PrismSettings settings, Path run) throws Exception {
    Files.createDirectories(run);
    Path cohort = run.resolve("rows");
    List<Long> ids = new ArrayList<>();
    for (long id = 100; id < 108; id++) ids.add(id);
    try (var resources = TrainingResources.open(settings, run);
        var writer = new CohortWriter(cohort, TENHOU_4P, 1, 0, 2, CohortWriter.Tap.NONE)) {
      ActivePhi phi = resources.phi(1, settings.phi());
      Phi value = phi.phi();
      var plan =
          new Collector.Cohort(
              TENHOU_4P, SEED, SEED * 3, ids, DrawOrder.SEAT_STREAMS, SAMPLING, null, URA, FLOOR, TwinDiscard.TSUMOGIRI,
              null);
      Collector.collect(resources.table(), plan, new HashedScorePolicy(), value, new RowSink(writer));
      writer.seal();
      CohortWriter.Manifest manifest = CohortWriter.readManifest(cohort);
      RowCodec codec = new RowCodec(manifest.ledgerFeatures(), TENHOU_4P);
      List<String> rows = new ArrayList<>();
      for (CohortWriter.Shard shard : manifest.shards()) {
        var frames = ByteBuffer.wrap(Files.readAllBytes(cohort.resolve(shard.file())));
        while (frames.hasRemaining()) {
          Row row = codec.decode(frames);
          rows.add(Base64.getEncoder().encodeToString(new RowCodec(row.ledgerFeatures(), TENHOU_4P).encode(row)));
        }
      }
      Collections.sort(rows);
      return new Collected(phi.entry(), rows, value.stats().requests());
    }
  }

  private static String[] concat(String[] head, String... tail) {
    String[] all = new String[head.length + tail.length];
    System.arraycopy(head, 0, all, 0, head.length);
    System.arraycopy(tail, 0, all, head.length, tail.length);
    return all;
  }
}
