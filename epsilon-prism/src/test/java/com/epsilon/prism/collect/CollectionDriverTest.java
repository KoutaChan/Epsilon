package com.epsilon.prism.collect;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundResult;
import com.epsilon.engine.WinClaim;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.CrnScheme;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.luck.LuckSample;
import com.epsilon.prism.luck.LuckSampleStore;
import com.epsilon.prism.luck.LuckSampleWriter;
import com.epsilon.prism.luck.LuckSampling;
import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.policy.Capacity;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.policy.PriorSettings;
import com.epsilon.prism.policy.ServedGoldenPolicy;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 収集の行が実行順・同時進行数・ワーカー数に依らず、行の種類（MAIN・ROLLIN・ROOT）ごとの形を守ることを確かめる。 */
public class CollectionDriverTest {
  private static final Logger log = LoggerFactory.getLogger(CollectionDriverTest.class);

  private static final long SEED = 7_001L;
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;

  /** ラベル行・ロールイン・写しの中の行がどれも出る率（本番より高い）。 */
  private static final Sampling SAMPLING =
      new Sampling(0.25, 0.03, 0.03, 0.03, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  /** 床 ε_b = 0.05 の行動方策（評価と同じ乱数の使い方）。 */
  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.V1);

  /**
   * s を手の ID と席のハッシュに置き、σ をその argmax にする inline 方策（手の選びは決定的で、打ち筋は手ごとにばらける）。retain なら p0（ハッシュの
   * 正の重みを正規化したもの）も書く（収集の試験だけ。本番の inline の方策は s も p0 も書かない）。記録 O は読まない（行の O は収集が捕捉する）。
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
      }
    }
  }

  /**
   * 行を一時ディレクトリのコホートに書き（書き手 2 本）、封印して {@link RowCodec} で読み戻す道具。閉じるとディレクトリを消す。
   */
  static final class TemporaryCohort implements AutoCloseable {
    final Path directory;
    final CohortWriter writer;
    final RowSink sink;

    /** 収集の率 sampling の帳簿の有無に合わせた欄の数で書き手を作る。 */
    TemporaryCohort(Sampling sampling) throws IOException {
      directory = Files.createTempDirectory("prism-rows");
      writer = new CohortWriter(directory, 1, sampling.eventLedger() ? EventLedger.FEATURES : 0, 2);
      sink = new RowSink(writer);
    }

    /** 封印して全ての行を読み戻す（目録とフレームの数の食い違いは試験の失敗）。 */
    List<Row> sealAndRead() throws IOException {
      writer.seal();
      CohortWriter.Manifest manifest = CohortWriter.readManifest(directory);
      RowCodec codec = new RowCodec(manifest.ledgerFeatures());
      List<Row> rows = new ArrayList<>();
      for (CohortWriter.Shard shard : manifest.shards()) {
        var frames = ByteBuffer.wrap(Files.readAllBytes(directory.resolve(shard.file())));
        while (frames.hasRemaining()) rows.add(codec.decode(frames));
      }
      Assert.assertEquals(rows.size(), manifest.rows());
      Assert.assertEquals(rows.size(), sink.rows());
      return rows;
    }

    @Override
    public void close() throws IOException {
      writer.close();
      Snapshot.deleteTree(directory);
    }
  }

  /**
   * 1 つの収集の結果と、封印して読み戻した行。
   *
   * @param rows 行（受け口に渡した順ではない）
   */
  record Collected(CollectionDriver.Result result, List<Row> rows) {}

  /** 半荘 first..first+games−1 の 1 つの収集。行は実行順に依らない。 */
  static Collected collectGames(
      Sampling sampling,
      long iterationSeed,
      long first,
      int games,
      Behaviour behaviour,
      PolicyService policy,
      int workers,
      int concurrent) {
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < games; id++) ids.add(first + id);
    try (var driver =
            new CollectionDriver(
                new CollectionDriver.Settings(concurrent, workers, new InferenceBatchingSettings(1000)));
        var phi = HandValue.scoreRank(PROFILE);
        var temporary = new TemporaryCohort(sampling)) {
      var cohort = new CollectionDriver.Cohort(SEED, iterationSeed, ids, PROFILE, sampling, behaviour, null, null);
      var result = driver.collect(cohort, policy, phi, temporary.sink);
      List<Row> rows = temporary.sealAndRead();
      Assert.assertEquals(result.counts().rowsWritten(), rows.size());
      Assert.assertEquals(result.rows().rows(), rows.size());
      return new Collected(result, rows);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /** 半荘 100..103 を {@link #SAMPLING} の率・床 0.05 で集める。 */
  private static Collected collect(int workers, int concurrent) {
    return collectGames(SAMPLING, SEED * 3, 100, 4, FLOOR, new HashedScorePolicy(), workers, concurrent);
  }

  /** 行の並びに依らない正準（直列化したフレームを並べ替えたもの）。 */
  static List<String> encoded(List<Row> rows) {
    List<String> result = new ArrayList<>();
    for (Row row : rows)
      result.add(Base64.getEncoder().encodeToString(new RowCodec(row.ledgerFeatures()).encode(row)));
    Collections.sort(result);
    return result;
  }

  /** 判断の席（O の観測席）。 */
  private static int seat(Row row) {
    return new Observation().bind(row.observation(), 0).observer();
  }

  /**
   * 主経路の探索: 温度が正なら床 0 でも主経路で貪欲でない手を打ち、温度 0・床 0 なら打たない。softmax(s/τ) は和 1 で s の大きい手ほど重い。
   * 行の形（ラベル行と根の数の和が受け口の行）は保つ。
   */
  @Test(timeOut = 300_000)
  public void mainPathExplorationPlaysNonGreedyMovesOnlyWhenEnabled() {
    float[] b = new Behaviour(0.01, 0.1, CrnScheme.V1).explore(new float[] {0.3f, 0.1f, 0.2f}, 3);
    Assert.assertEquals(b[0] + b[1] + b[2], 1f, 1e-5f);
    Assert.assertTrue(b[0] > b[2] && b[2] > b[1]);
    var explored =
        collectGames(SAMPLING, SEED * 3, 100, 4, new Behaviour(0, 0.2, CrnScheme.V1), new HashedScorePolicy(), 1, 4).result();
    var greedy =
        collectGames(SAMPLING, SEED * 3, 100, 4, new Behaviour(0, 0, CrnScheme.V1), new HashedScorePolicy(), 1, 4).result();
    Assert.assertTrue(explored.counts().mainNonGreedy() > 0);
    Assert.assertEquals(greedy.counts().mainNonGreedy(), 0);
    Assert.assertEquals(
        explored.counts().rowsWritten(), explored.counts().labels() + explored.counts().rootCount());
  }

  private static long sum(long[] counts) {
    long total = 0;
    for (long count : counts) total += count;
    return total;
  }

  @Test(timeOut = 300_000)
  public void rowsDoNotDependOnScheduling() {
    Collected serial = collect(1, 4);
    Collected parallel = collect(4, 2);
    CollectionMetrics.Snapshot counts = serial.result().counts();
    Assert.assertTrue(sum(counts.labelRows()) > 0, "labels=" + sum(counts.labelRows()));
    Assert.assertTrue(sum(counts.copyLabelRows()) > 0, "rollin=" + sum(counts.copyLabelRows()));
    Assert.assertTrue(counts.copies() > 0, "copies=" + counts.copies());
    Assert.assertTrue(counts.forks() > 0, "forks=" + counts.forks());
    Assert.assertEquals(counts.rowsWritten(), counts.labels() + counts.rootCount());
    Assert.assertEquals(serial.rows().size(), counts.rowsWritten());
    Assert.assertEquals(parallel.result().counts().modelDecisions(), counts.modelDecisions());
    Assert.assertEquals(parallel.result().counts().forks(), counts.forks());
    Assert.assertEquals(parallel.result().counts().copies(), counts.copies());
    Assert.assertEquals(encoded(parallel.rows()), encoded(serial.rows()));
    Assert.assertFalse(serial.result().summary().isEmpty());
  }

  /**
   * 受付から受けたバッチで答える方策: 行ごとに {@link HashedScorePolicy} と同じ s・σ・p0 を書き、バッチごとの行の容量区分の組を控える
   * （ドライバのスレッドから呼ばれる）。merged なら受付の区分を 1 つにする（網の方策と同じ {@link PolicyService#admissionKey}）。
   */
  static final class BatchedHashedPolicy implements PolicyService {
    private final HashedScorePolicy scores = new HashedScorePolicy();
    private final boolean merged;

    /** バッチごとの行の容量区分の組（受け付けた順）。 */
    final List<EnumSet<Capacity>> batches = new ArrayList<>();

    BatchedHashedPolicy(boolean merged) {
      this.merged = merged;
    }

    @Override
    public int version() {
      return 1;
    }

    @Override
    public boolean inline() {
      return false;
    }

    @Override
    public Capacity admissionKey(PolicyRequest request) {
      return merged ? Capacity.SINGLE_TRANSITION : PolicyService.super.admissionKey(request);
    }

    @Override
    public CompletableFuture<Void> submit(List<PolicyRequest> batch, DecisionInferenceIngress ingress) {
      ingress.handoff(this).release();
      EnumSet<Capacity> capacities = EnumSet.noneOf(Capacity.class);
      for (PolicyRequest request : batch) {
        capacities.add(request.capacity());
        scores.evaluateInline(null, null, request);
      }
      batches.add(capacities);
      return CompletableFuture.completedFuture(null);
    }
  }

  /**
   * 受付から受けたバッチで答える方策で半荘 100..103 を集め（同時に 4 局・進行ワーカー 2 本）、バッチごとの行の容量区分の組と行を返す。待ちの期限は
   * 長く取り、バッチは進行が全て止まってから出す（バッチの組みが実行の速さに依らない）。
   */
  private static Map.Entry<List<EnumSet<Capacity>>, List<Row>> collectBatched(boolean merged) throws Exception {
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < 4; id++) ids.add(100 + id);
    var cohort = new CollectionDriver.Cohort(SEED, SEED * 3, ids, PROFILE, SAMPLING, FLOOR, null, null);
    try (var driver = new CollectionDriver(new CollectionDriver.Settings(4, 2, new InferenceBatchingSettings(1_000_000)));
        var policy = new BatchedHashedPolicy(merged);
        var phi = HandValue.scoreRank(PROFILE);
        var temporary = new TemporaryCohort(SAMPLING)) {
      var result = driver.collect(cohort, policy, phi, temporary.sink);
      List<Row> rows = temporary.sealAndRead();
      Assert.assertEquals(result.counts().rowsWritten(), rows.size());
      Assert.assertEquals(result.batches(), policy.batches.size());
      return Map.entry(policy.batches, rows);
    }
  }

  /**
   * 受付の区分（{@link PolicyService#admissionKey}）: 既定では行の容量区分ごとに受け付け、鳴きの行（{@link Capacity#MULTIPLE_TRANSITIONS}）と
   * ほかの行は同じバッチに入らない。区分を 1 つにする方策（網の方策）では両方の行が 1 つのバッチに入る。行は区分に依らない。
   */
  @Test(timeOut = 300_000)
  public void admissionKeyDecidesWhichRowsShareABatch() throws Exception {
    var separate = collectBatched(false);
    var merged = collectBatched(true);
    Assert.assertTrue(
        separate.getKey().stream().anyMatch(batch -> batch.contains(Capacity.MULTIPLE_TRANSITIONS)), "no call rows");
    for (EnumSet<Capacity> batch : separate.getKey()) Assert.assertEquals(batch.size(), 1, "the default key mixed " + batch);
    Assert.assertTrue(
        merged.getKey().stream().anyMatch(batch -> batch.size() == Capacity.values().length),
        "the merged key never put both capacities in one batch");
    Assert.assertFalse(separate.getValue().isEmpty());
    Assert.assertEquals(encoded(merged.getValue()), encoded(separate.getValue()));
  }

  /** 行の種類ごとの形: 測った手は C ∪ {played}、根は played と d の枠と傾向 β_c·q、写しの中の行は逸脱した席だけで写しごとに上限まで。 */
  @Test(timeOut = 300_000)
  public void rowsKeepTheirShape() {
    Collected collected = collect(2, 4);
    Map<String, Row> roots = new HashMap<>();
    for (Row row : collected.rows())
      if (row.source() == Row.Source.ROOT) roots.put(row.gameId() + ":" + row.decisionId(), row);
    Map<String, Integer> perCopy = new HashMap<>();
    long measuredTotal = 0;
    for (Row row : collected.rows()) {
      double priorSum = 0;
      for (float p : row.prior()) {
        Assert.assertTrue(p > 0);
        priorSum += p;
      }
      Assert.assertEquals(priorSum, 1.0, 1e-4);
      int supportSize = Integer.bitCount(row.support());
      Assert.assertTrue(supportSize >= 1 && supportSize <= SAMPLING.maxCandidates());
      Assert.assertFalse(Float.isNaN(row.values()[row.played()]));
      switch (row.source()) {
        case MAIN, ROLLIN -> {
          int measured = 0;
          for (int a = 0; a < row.actions(); a++) {
            boolean expected = a == row.played() || (row.support() >>> a & 1) != 0;
            Assert.assertEquals(row.measured(a), expected, "action " + a);
            if (expected) measured++;
          }
          Assert.assertTrue(measured >= 2, "measured=" + measured);
          measuredTotal += measured;
          Assert.assertEquals(row.deviation(), -1);
          Assert.assertTrue(Float.isNaN(row.propensity()));
          if (row.source() == Row.Source.MAIN) {
            Assert.assertEquals(row.rootDecisionId(), -1L);
          } else {
            String key = row.gameId() + ":" + row.rootDecisionId();
            Row root = roots.get(key);
            Assert.assertNotNull(root, "root of " + key);
            Assert.assertNotEquals(root.deviation(), root.played());
            Assert.assertEquals(seat(row), seat(root));
            perCopy.merge(key, 1, Integer::sum);
          }
        }
        case ROOT -> {
          int d = row.deviation();
          Assert.assertTrue((row.support() >>> d & 1) != 0, "deviation outside the support");
          for (int a = 0; a < row.actions(); a++)
            Assert.assertEquals(row.measured(a), a == row.played() || a == d, "action " + a);
          Assert.assertTrue(
              row.deviationProbability() >= SAMPLING.epsilonQ() / supportSize - 1e-6
                  && row.deviationProbability() <= 1);
          Assert.assertEquals(
              row.propensity(),
              (float) (SAMPLING.rollinRate(row.decisionClass()) * row.deviationProbability()));
          Assert.assertEquals(row.rootDecisionId(), -1L);
        }
      }
    }
    for (int count : perCopy.values()) Assert.assertTrue(count <= SAMPLING.maxCopyLabels());
    Assert.assertEquals(collected.result().counts().measured(), measuredTotal);
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

  /** 帳簿を外した行の正準（帳簿の欄 0 で直列化）。 */
  private static List<String> strippedCanonical(List<Row> rows) {
    var codec = new RowCodec(0);
    List<String> result = new ArrayList<>();
    for (Row row : rows)
      result.add(
          Base64.getEncoder()
              .encodeToString(
                  codec.encode(
                      new Row(
                          row.gameId(),
                          row.decisionId(),
                          row.decisionClass(),
                          row.purpose(),
                          row.group(),
                          row.source(),
                          row.played(),
                          row.values(),
                          row.prior(),
                          row.support(),
                          row.deviation(),
                          row.deviationProbability(),
                          row.propensity(),
                          row.rootDecisionId(),
                          new float[0],
                          row.outcomes(),
                          row.oracle(),
                          row.observation()))));
    Collections.sort(result);
    return result;
  }

  /** 帳簿の有無だけを変えて比べる収集（率・床と主経路の探索）。 */
  private static List<Collected> everyCollection(boolean ledger) {
    Sampling sampling = SAMPLING.withEventLedger(ledger);
    List<Collected> result = new ArrayList<>();
    result.add(collectGames(sampling, SEED * 3, 100, 4, FLOOR, new HashedScorePolicy(), 1, 4));
    result.add(collectGames(sampling, SEED * 7, 300, 4, FLOOR, new HashedScorePolicy(), 1, 4));
    result.add(collectGames(sampling, SEED * 3, 100, 4, new Behaviour(0, 0.2, CrnScheme.V1), new HashedScorePolicy(), 1, 4));
    return result;
  }

  /**
   * record は帳簿を足すだけ: 同じ収集を帳簿なしとありで打ち、帳簿つきの行から帳簿を外した正準が帳簿なしと一致する（帳簿の計算は乱数の消費も手も
   * 変えない）。帳簿つきの行は全て帳簿を持ち（ラベル行と根）、欄は {@value EventLedger#FEATURES}、測った手の帳簿は有限で fp16 で厳密、測って
   * いない手は NaN。帳簿の段は帳簿つきの収集でだけ記録に出る。
   */
  @Test(timeOut = 600_000)
  public void recordModeOnlyAddsTheLedger() {
    List<Collected> off = everyCollection(false);
    List<Collected> recorded = everyCollection(true);
    long rows = 0, roots = 0, nonzero = 0;
    for (int i = 0; i < off.size(); i++) {
      Assert.assertEquals(strippedCanonical(recorded.get(i).rows()), encoded(off.get(i).rows()), "collection " + i);
      Assert.assertFalse(off.get(i).result().stages().containsKey(StageTimer.Stage.LEDGER), "collection " + i);
      Assert.assertTrue(recorded.get(i).result().stages().containsKey(StageTimer.Stage.LEDGER), "collection " + i);
      for (Row row : off.get(i).rows()) Assert.assertFalse(row.hasLedger());
      for (Row row : recorded.get(i).rows()) {
        Assert.assertTrue(row.hasLedger());
        Assert.assertEquals(row.ledgerFeatures(), EventLedger.FEATURES);
        for (int a = 0; a < row.actions(); a++)
          for (int f = 0; f < EventLedger.FEATURES; f++) {
            float value = row.ledger(a, f);
            if (!row.measured(a)) {
              Assert.assertTrue(Float.isNaN(value), "unmeasured action " + a);
            } else {
              Assert.assertTrue(Float.isFinite(value), "measured action " + a + " feature " + f);
              Assert.assertEquals(Float.float16ToFloat(Float.floatToFloat16(value)), value, 0f);
              if (value != 0) nonzero++;
            }
          }
        rows++;
        if (row.source() == Row.Source.ROOT) roots++;
      }
    }
    Assert.assertTrue(rows > 0 && roots > 0, "rows " + rows + " roots " + roots);
    Assert.assertTrue(nonzero > 0, "every ledger entry was zero");
  }

  /** 帳簿つきの行は実行順・同時進行数・ワーカー数に依らない（席の表のキャッシュは進行ごと）。 */
  @Test(timeOut = 600_000)
  public void recordModeIsDeterministic() {
    Sampling sampling = SAMPLING.withEventLedger(true);
    var serial = collectGames(sampling, SEED * 3, 100, 4, FLOOR, new HashedScorePolicy(), 1, 4).rows();
    var parallel = collectGames(sampling, SEED * 3, 100, 4, FLOOR, new HashedScorePolicy(), 4, 2).rows();
    Assert.assertFalse(serial.isEmpty());
    Assert.assertEquals(encoded(parallel), encoded(serial));
  }

  /**
   * 本番の符号化の道を通す方策の収集の結果（並びに依らない正準）。
   *
   * @param rows 行（{@link #encoded}）
   * @param luck 運の網の標本（並べ替えた文字列）
   * @param queries 方策が答えた問い合わせの数
   * @param observations 方策が受けた O のハッシュの和
   * @param tokens 方策が符号化した塊の行のハッシュの和
   */
  private record Served(
      CollectionDriver.Result result,
      List<String> rows,
      List<String> luck,
      long queries,
      long observations,
      long tokens) {}

  /**
   * 半荘 100..105 を帳簿つきの率・床 0.05 で、本番の符号化の道を通す方策（{@link ServedGoldenPolicy}）と運の網の標本（率 1）つきで
   * 集める。
   */
  private static Served collectServed(int workers, int concurrent) throws IOException {
    Sampling sampling = SAMPLING.withEventLedger(true);
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < 6; id++) ids.add(100 + id);
    var cohort = new CollectionDriver.Cohort(SEED, SEED * 3, ids, PROFILE, sampling, FLOOR, null, null);
    try (var driver =
            new CollectionDriver(
                new CollectionDriver.Settings(concurrent, workers, new InferenceBatchingSettings(1000)));
        var policy = new ServedGoldenPolicy(new PriorSettings(4.0, 0.05));
        var phi = HandValue.scoreRank(PROFILE);
        var temporary = new TemporaryCohort(sampling);
        var writer = new LuckSampleWriter(temporary.directory, 1)) {
      var luck = new LuckSampleSink(new LuckSampling(SEED * 3, 1.0), writer);
      var result = driver.collect(cohort, policy, phi, temporary.sink, luck);
      if (luck.failure() != null) throw new AssertionError("luck samples failed", luck.failure());
      List<Row> rows = temporary.sealAndRead();
      writer.seal();
      Assert.assertTrue(rows.stream().allMatch(Row::hasLedger), "a row has no ledger");
      List<String> samples = new ArrayList<>();
      try (LuckSampleStore.Cursor cursor = LuckSampleStore.open(temporary.directory).cursor()) {
        while (cursor.next()) {
          LuckSample sample = cursor.sample();
          samples.add(
              sample.game() + " " + Arrays.toString(sample.observation()) + " " + Arrays.toString(sample.y()) + " "
                  + sample.stratum() + " " + sample.weight());
        }
      }
      Assert.assertEquals(samples.size(), luck.counts().samples());
      Collections.sort(samples);
      return new Served(result, encoded(rows), samples, policy.queries(), policy.observations(), policy.tokens());
    }
  }

  /**
   * 本番の符号化の道（捕捉・計画・符号化・答え）を通す方策・運の精算の帳簿・運の網の標本つきの収集も、実行順・同時進行数・ワーカー数に
   * 依らない: 同じ種で 2 回集めた行・運の標本と、方策が受けた O・符号化した塊の行が同じ。
   */
  @Test(timeOut = 1_800_000)
  public void servedCollectionIsDeterministic() throws IOException {
    Served serial = collectServed(1, 4);
    Served parallel = collectServed(4, 2);
    Assert.assertTrue(serial.result().counts().forkDecisions() > 0, "no fork decisions");
    Assert.assertTrue(serial.result().counts().copyDecisions() > 0, "no copy decisions");
    Assert.assertFalse(serial.rows().isEmpty(), "no rows");
    Assert.assertFalse(serial.luck().isEmpty(), "no luck samples");
    Assert.assertTrue(serial.queries() > serial.rows().size(), "the policy answered too few queries");
    Assert.assertEquals(parallel.rows(), serial.rows());
    Assert.assertEquals(parallel.luck(), serial.luck());
    Assert.assertEquals(parallel.queries(), serial.queries());
    Assert.assertEquals(parallel.observations(), serial.observations());
    Assert.assertEquals(parallel.tokens(), serial.tokens());
  }

  /** 1 局の進行の帳簿（{@link #forkTrackerDoesNotTouchItsParent}）: 主経路と、境界 fork で作る分岐（-1 なら作らない）。 */
  private static List<Slot> mainLedger(long seed, int fork) {
    var shapes = new HandShapeAnalyzer();
    var engine = new GameEngine(SeedMixer.mix64(seed));
    engine.getState().enableSeatStreamDraws();
    var tracker = new EventLedger.Tracker();
    List<Slot> slots = LedgerGames.slots();
    var random = new SplittableRandom(seed);
    int boundary = 0;
    var step = engine.stepHanchan();
    while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
      List<EngineDecisionPoint> points = awaiting.decisions();
      tracker.observe(engine.getState(), points, slots, null);
      var selections = LedgerGames.select(engine.getState(), points, LedgerGames.KANS, random, shapes, null);
      if (boundary++ == fork) {
        // 分岐: 親の計算器を写し、別の乱数の手で局末まで進めて、分岐自身の枠に帳簿を付ける
        var branch = engine.forkAtDecisionBoundary();
        var branchTracker = new EventLedger.Tracker(tracker, engine.getState());
        List<Slot> branchSlots = LedgerGames.slots();
        var branchRandom = new SplittableRandom(seed + 1);
        var branchStep = branch.commitDecisions(LedgerGames.select(branch.getState(), points, LedgerGames.CALLS, branchRandom, shapes, null));
        while (branchStep instanceof GameStepResult.AwaitingDecisions next) {
          branchTracker.observe(branch.getState(), next.decisions(), branchSlots, null);
          branchStep = branch.commitDecisions(LedgerGames.select(branch.getState(), next.decisions(), LedgerGames.CALLS, branchRandom, shapes, null));
        }
      }
      step = engine.commitDecisions(selections);
    }
    return slots;
  }

  /**
   * 分岐の計算器は親に触らない: 主経路の計算器を写した分岐を局末まで進めてから主経路を進めた帳簿と、分岐を作らずに進めた帳簿がビット単位で一致する
   * （席の表を作った後に変えない約束。分岐は親の表の配列の写しで始め、新しい表は自分の配列にだけ入れる）。
   */
  @Test(timeOut = 300_000)
  public void forkTrackerDoesNotTouchItsParent() {
    for (long seed = 1; seed <= 40; seed++) {
      List<Slot> alone = mainLedger(seed, -1), forked = mainLedger(seed, 8 + (int) (seed % 5));
      for (int seat = 0; seat < alone.size(); seat++)
        for (int f = 0; f < EventLedger.FEATURES; f++)
          Assert.assertEquals(
              Double.doubleToRawLongBits(forked.get(seat).ledger()[f]),
              Double.doubleToRawLongBits(alone.get(seat).ledger()[f]),
              "seed " + seed + " seat " + seat + " " + EventLedger.NAMES.get(f));
    }
  }

  /**
   * 局末の 3 語はその（行, 手）の経路の局末から来る。効用を素点（{@link EpsilonUtilityProfile#SCORE}）にした収集では Y が判断の席の局末の点で
   * 決まるので、3 語を記録した全ての（行, 手）で、3 語の自分の局末の点が Y を再現する（分岐・写し・主経路の全ての枠）。主経路の枠（MAIN・ROOT の
   * 打った手）では、3 語の 4 席の局末の点と聴牌への到達が、主経路を外から数えた値（局末の点、和了・局末の聴牌・判断の後の打牌の後の聴牌）と一致する
   * （河の打牌の数え・局ごとの消去・枠の席の選びの確かめ。分岐と写しの到達は親の写しから始め、写し損ねると境界の数えが例外で止まる）。
   */
  @Test(timeOut = 600_000)
  public void outcomesComeFromTheirOwnPath() throws Exception {
    EpsilonUtilityProfile profile = EpsilonUtilityProfile.SCORE;
    Sampling sampling = SAMPLING.withEventLedger(true);
    var truth = new MainPath();
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < 8; id++) ids.add(700 + id);
    List<Row> rows;
    try (var driver = new CollectionDriver(new CollectionDriver.Settings(4, 2, new InferenceBatchingSettings(1000)));
        var policy = new HashedScorePolicy();
        var phi = HandValue.scoreRank(profile);
        var temporary = new TemporaryCohort(sampling)) {
      var cohort = new CollectionDriver.Cohort(SEED, SEED * 13, ids, profile, sampling, FLOOR, null, truth);
      driver.collect(cohort, policy, phi, temporary.sink);
      rows = temporary.sealAndRead();
    }
    var observation = new Observation();
    int slots = 0, mainSlots = 0, reached = 0, reachedByDiscard = 0;
    for (Row row : rows) {
      observation.bind(row.observation(), 0);
      int seat = observation.observer();
      for (int a = 0; a < row.actions(); a++) {
        int at = a * OutcomeCodec.WORDS;
        if (OutcomeCodec.outcome(row.outcomes(), at) == null) continue;
        float y = (float) EpsilonUtilityTargets.utilityDouble(profile, OutcomeCodec.finalScore(row.outcomes(), at, 0), 0);
        Assert.assertEquals(Float.floatToRawIntBits(row.values()[a]), Float.floatToRawIntBits(y), row.source() + " action " + a);
        slots++;
      }
      int at = row.played() * OutcomeCodec.WORDS;
      boolean mainSlot = row.source() == Row.Source.MAIN || row.source() == Row.Source.ROOT;
      if (!mainSlot || OutcomeCodec.outcome(row.outcomes(), at) == null) continue;
      MainPath.Hand hand = truth.hand(row.gameId(), observation.kyoku(), observation.honba());
      for (int relative = 0; relative < GameState.NUM_PLAYERS; relative++)
        Assert.assertEquals(
            OutcomeCodec.finalScore(row.outcomes(), at, relative),
            hand.scores[(seat + relative) % GameState.NUM_PLAYERS],
            "main path score");
      boolean expected = hand.reached(seat, observation.events());
      Assert.assertEquals(OutcomeCodec.tenpai(row.outcomes(), at), expected, "tenpai reach of " + row.source());
      if (expected) reached++;
      if (expected && (hand.endTenpai & 1 << seat) == 0) reachedByDiscard++;
      mainSlots++;
    }
    Assert.assertTrue(slots > 0 && mainSlots > 0 && reached > 0, slots + " slots, " + mainSlots + " main slots");
    log.info(
        "outcomesComeFromTheirOwnPath: rows={} slots={} mainSlots={} reached={} reachedByDiscard={}",
        rows.size(),
        slots,
        mainSlots,
        reached,
        reachedByDiscard);
  }

  /**
   * 主経路を外から見て、局ごとに局末の 4 席の点、局末の聴牌（和了を含む）、打牌ごとの（席、出来事の番号、打牌の後の聴牌）を数える
   * （{@link #outcomesComeFromTheirOwnPath}）。進行ワーカーから並行に呼ばれる（同じ半荘は 1 スレッドずつ）。
   */
  static final class MainPath implements CollectionDriver.GameObserver {
    private static final ThreadLocal<HandShapeAnalyzer> SHAPES =
        ThreadLocal.withInitial(HandShapeAnalyzer::new);

    /** 1 局の事実。 */
    static final class Hand {
      final List<int[]> discards = new ArrayList<>();
      int[] scores;
      int endTenpai;

      /** 判断の時点の出来事の数が events の席 seat が、局の終わりまでに聴牌に届いたか。 */
      boolean reached(int seat, int events) {
        if ((endTenpai & 1 << seat) != 0) return true;
        for (int[] discard : discards) if (discard[0] == seat && discard[1] >= events && discard[2] != 0) return true;
        return false;
      }
    }

    private final Map<String, Hand> hands = new ConcurrentHashMap<>();
    private final Map<Long, Hand> ended = new ConcurrentHashMap<>();

    Hand hand(long gameId, int kyoku, int honba) {
      Hand hand = hands.get(gameId + ":" + kyoku + ":" + honba);
      Assert.assertNotNull(hand, "main path hand " + gameId + ":" + kyoku + ":" + honba);
      return hand;
    }

    private Hand hand(long gameId, GameState state) {
      PublicObservation o = state.publicObservation(0);
      return hands.computeIfAbsent(gameId + ":" + o.roundIndex() + ":" + o.honba(), key -> new Hand());
    }

    @Override
    public void gameEnded(long gameId, int[] finalScores) {}

    @Override
    public void decided(long gameId, GameState state, EngineDecisionPoint point, Action action) {
      if (action.type() != Action.Type.DAHAI && action.type() != Action.Type.RIICHI_DAHAI) return;
      int seat = point.player();
      int tenpai = SHAPES.get().calculateMinimumAfterRemoving(state.hand(seat), action.tileType()) <= 0 ? 1 : 0;
      // 打牌の出来事の番号 = それまでの公開の出来事（打牌・副露・加槓）の数
      PublicObservation o = state.publicObservation(seat);
      int events = 0;
      for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
        events += o.river(player).size();
        PublicObservation.VisibleHand hand = o.hand(player);
        for (int meld = 0; meld < hand.meldCount(); meld++) events += hand.meldKakanEvent(meld) < 0 ? 1 : 2;
      }
      hand(gameId, state).discards.add(new int[] {seat, events, tenpai});
    }

    @Override
    public void handEnded(long gameId, GameState state, RoundResult result) {
      Hand hand = hand(gameId, state);
      int tenpai = 0;
      if (result instanceof RoundResult.Winning winning)
        for (WinClaim claim : winning.claims()) tenpai |= 1 << claim.winner();
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
        if (SHAPES.get().calculateMinimum(state.hand(seat)) <= 0) tenpai |= 1 << seat;
      hand.endTenpai = tenpai;
      ended.put(gameId, hand);
    }

    @Override
    public void handValued(long gameId, float[] nextRootPrefix, int[] scores) {
      ended.get(gameId).scores = scores.clone();
    }
  }
}
