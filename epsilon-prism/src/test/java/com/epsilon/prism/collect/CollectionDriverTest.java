package com.epsilon.prism.collect;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.ObservationPayload;
import com.epsilon.prism.rows.PrismRow;
import com.epsilon.prism.rows.RowFormat;
import com.epsilon.prism.rows.RowSink;
import com.epsilon.util.SeedMixer;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 収集の行が実行順・同時進行数・ワーカー数に依らず、行の種類（MAIN・ROLLIN・ROOT）ごとの形を守ることを確かめる。 */
public class CollectionDriverTest {
  private static final long SEED = 7_001L;
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;

  /** ラベル行・ロールイン・写しの中の行がどれも出る率（本番より高い）。 */
  private static final Sampling SAMPLING =
      new Sampling(0.25, 0.03, 0.03, 0.03, 0.15, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  /** 行を受け取るだけの書き先。 */
  static final class Capture implements RowSink {
    final ConcurrentLinkedQueue<PrismRow> rows = new ConcurrentLinkedQueue<>();

    @Override
    public void accept(PrismRow row) {
      rows.add(row);
    }

    @Override
    public long backlog() {
      return 0;
    }

    @Override
    public RowStats stats() {
      return new RowStats(rows.size(), 0);
    }

    @Override
    public void close() {}

    /** 行の並びに依らない正準（行の形式のバイト列を並べ替えたもの）。 */
    List<String> canonical() {
      List<String> result = new ArrayList<>();
      for (PrismRow row : rows)
        result.add(Base64.getEncoder().encodeToString(RowFormat.encodeRaw(row)));
      Collections.sort(result);
      return result;
    }
  }

  /**
   * s を手の ID と席のハッシュに置き、σ をその argmax にする inline 方策（手の選びは決定的で、打ち筋は手ごとにばらける）。retain なら p0（ハッシュの
   * 正の重みを正規化したもの）と仮の payload も書く（収集の試験だけ。本番の inline の方策は s も p0 も書かない）。
   */
  static final class HashedScorePolicy implements PolicyService {
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
        long bits = SeedMixer.mix64(request.legal().get(a).toIndex() * 31L + request.seat());
        scores[a] = (float) CrnSeeds.unit(bits);
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
          long bits = SeedMixer.mix64(request.legal().get(a).toIndex() * 17L + 5);
          prior[a] = 1f + (float) CrnSeeds.unit(bits);
          sum += prior[a];
        }
        for (int a = 0; a < n; a++) prior[a] = (float) (prior[a] / sum);
        request.prior = prior;
        request.payload = ObservationPayload.takeOwnership(new byte[] {'P', 'R', 'S', '1'});
      }
    }
  }

  private static CollectionDriver.Result collect(int workers, int concurrent, Capture sink) {
    var settings =
        new CollectionDriver.Settings(concurrent, workers, new InferenceBatchingSettings(1000));
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < 4; id++) ids.add(100 + id);
    var cohort =
        new CollectionDriver.Cohort(
            1, SEED, SEED * 3, ids, PROFILE, SAMPLING, new Behaviour(0.05), null, null);
    try (var driver = new CollectionDriver(settings);
        var policy = new HashedScorePolicy();
        var phi = HandValue.scoreRank(PROFILE)) {
      return driver.collect(cohort, policy, phi, sink);
    }
  }

  private static long sum(long[] counts) {
    long total = 0;
    for (long count : counts) total += count;
    return total;
  }

  @Test(timeOut = 300_000)
  public void rowsDoNotDependOnScheduling() {
    var serial = new Capture();
    var a = collect(1, 4, serial);
    var parallel = new Capture();
    var b = collect(4, 2, parallel);
    CollectionMetrics.Snapshot counts = a.counts();
    Assert.assertTrue(sum(counts.labelRows()) > 0, "labels=" + sum(counts.labelRows()));
    Assert.assertTrue(sum(counts.copyLabelRows()) > 0, "rollin=" + sum(counts.copyLabelRows()));
    Assert.assertTrue(counts.copies() > 0, "copies=" + counts.copies());
    Assert.assertTrue(counts.forks() > 0, "forks=" + counts.forks());
    Assert.assertEquals(counts.rowsWritten(), counts.labels() + counts.rootCount());
    Assert.assertEquals(serial.rows.size(), counts.rowsWritten());
    Assert.assertEquals(b.counts().modelDecisions(), counts.modelDecisions());
    Assert.assertEquals(b.counts().forks(), counts.forks());
    Assert.assertEquals(b.counts().copies(), counts.copies());
    Assert.assertEquals(parallel.canonical(), serial.canonical());
    Assert.assertFalse(a.summary().isEmpty());
  }

  /** 行の種類ごとの形: 測った手は C ∪ {played}、根は played と d の枠と傾向 β_c·q、写しの中の行は逸脱した席だけで写しごとに上限まで。 */
  @Test(timeOut = 300_000)
  public void rowsKeepTheirShape() {
    var sink = new Capture();
    var result = collect(2, 4, sink);
    Map<String, PrismRow> roots = new HashMap<>();
    for (PrismRow row : sink.rows)
      if (row.source() == PrismRow.Source.ROOT)
        roots.put(row.gameId() + ":" + row.decisionId(), row);
    Map<String, Integer> perCopy = new HashMap<>();
    long measuredTotal = 0;
    for (PrismRow row : sink.rows) {
      Assert.assertEquals(row.values().length, row.actions());
      double priorSum = 0;
      for (float p : row.prior()) {
        Assert.assertTrue(p > 0);
        priorSum += p;
      }
      Assert.assertEquals(priorSum, 1.0, 1e-4);
      int[] support = new int[row.support().length];
      for (int i = 0; i < support.length; i++) support[i] = Byte.toUnsignedInt(row.support()[i]);
      for (int i = 1; i < support.length; i++) Assert.assertTrue(support[i - 1] < support[i]);
      Assert.assertTrue(support.length >= 1 && support.length <= SAMPLING.maxCandidates());
      Assert.assertFalse(Float.isNaN(row.playedValue()));
      switch (row.source()) {
        case MAIN, ROLLIN -> {
          int measured = 0;
          for (int a = 0; a < row.actions(); a++) {
            boolean expected = a == row.played() || contains(support, a);
            Assert.assertEquals(!Float.isNaN(row.values()[a]), expected, "action " + a);
            if (expected) measured++;
          }
          Assert.assertTrue(measured >= 2, "measured=" + measured);
          measuredTotal += measured;
          Assert.assertEquals(row.deviation(), -1);
          Assert.assertTrue(Float.isNaN(row.propensity()));
          if (row.source() == PrismRow.Source.MAIN) {
            Assert.assertEquals(row.rootDecisionId(), -1L);
          } else {
            String key = row.gameId() + ":" + row.rootDecisionId();
            PrismRow root = roots.get(key);
            Assert.assertNotNull(root, "root of " + key);
            Assert.assertNotEquals(root.deviation(), root.played());
            Assert.assertEquals(row.seat(), root.seat());
            perCopy.merge(key, 1, Integer::sum);
          }
        }
        case ROOT -> {
          int d = row.deviation();
          Assert.assertTrue(contains(support, d), "deviation outside the support");
          for (int a = 0; a < row.actions(); a++)
            Assert.assertEquals(
                !Float.isNaN(row.values()[a]), a == row.played() || a == d, "action " + a);
          Assert.assertTrue(
              row.deviationProbability() >= SAMPLING.epsilonQ() / support.length - 1e-6
                  && row.deviationProbability() <= 1);
          Assert.assertEquals(
              row.propensity(),
              (float) (SAMPLING.rollinRate(row.decisionClass()) * row.deviationProbability()));
          Assert.assertEquals(row.rootDecisionId(), -1L);
        }
      }
    }
    for (int count : perCopy.values()) Assert.assertTrue(count <= SAMPLING.maxCopyLabels());
    Assert.assertEquals(result.counts().measured(), measuredTotal);
  }

  /** q は C の中の s の順位（同点は番号の小さい手が上）に 0.5^r で沿い、一様な床 ε_q を混ぜる。d は順位の順の累積の逆関数。 */
  @Test
  public void rollInFollowsRanksWithAFloor() {
    int[] support = {1, 3, 4, 7};
    float[] scores = new float[8];
    scores[1] = 0.2f;
    scores[3] = 0.9f;
    scores[4] = 0.2f;
    scores[7] = -1f;
    double[] q = RollIn.distribution(support, scores, null, 0.1);
    double normalizer = 1 + 0.5 + 0.25 + 0.125;
    // 順位: 手 3 → 0、手 1 → 1（手 4 と同点で番号が小さい）、手 4 → 2、手 7 → 3
    Assert.assertEquals(q[1], 0.9 / normalizer + 0.025, 1e-12);
    Assert.assertEquals(q[0], 0.9 * 0.5 / normalizer + 0.025, 1e-12);
    Assert.assertEquals(q[2], 0.9 * 0.25 / normalizer + 0.025, 1e-12);
    Assert.assertEquals(q[3], 0.9 * 0.125 / normalizer + 0.025, 1e-12);
    Assert.assertEquals(q[0] + q[1] + q[2] + q[3], 1.0, 1e-12);
    RollIn.Deviation top = RollIn.draw(support, scores, null, 0.1, 0.0);
    Assert.assertEquals(top.action(), 3);
    Assert.assertEquals(top.probability(), (float) q[1]);
    Assert.assertEquals(RollIn.draw(support, scores, null, 0.1, q[1] + 1e-9).action(), 1);
    Assert.assertEquals(RollIn.draw(support, scores, null, 0.1, 0.999999).action(), 7);
  }

  private static boolean contains(int[] actions, int action) {
    for (int a : actions) if (a == action) return true;
    return false;
  }
}
