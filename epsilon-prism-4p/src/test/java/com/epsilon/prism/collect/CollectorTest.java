package com.epsilon.prism.collect;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.learn.Selection;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerFit;
import com.epsilon.prism.luck.LedgerFitTest;
import com.epsilon.prism.luck.LedgerRecording;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.policy.Capacity;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.policy.PriorSettings;
import com.epsilon.prism.policy.ServedGoldenPolicy;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 収集した行が、実行の順・同時に進める半荘の数・ワーカーの数に依らず、行の種類（MAIN・ROLLIN・ROOT）ごとの形を守ることを確かめる。 */
public class CollectorTest {
  private static final Logger log = LoggerFactory.getLogger(CollectorTest.class);

  private static final long SEED = 7_001L;
  private static final RankUtility UTILITY = RankUtility.TENHOU;

  /** 局末の点数の順位を、そのまま値にする Φ（どのスレッドからも呼べる）。 */
  private static final Phi SCORE_RANK = Phi.scoreRank(TENHOU_4P, UTILITY);

  /** ラベル行、ロールイン、写しの中の行がどれも出るようにした抽出率（本番より高い）。 */
  private static final Sampling SAMPLING =
      TestSampling.grouped(0.25, 0.03, 0.03, 0.03, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  /** 行の帳簿の 17 列が属する 5 つの組を全て計算し、点の列を 32 千点、方策の列の s を 8 で切る帳簿（同梱の設定の既定値と同じ）。 */
  static final LedgerRecording LEDGER = new LedgerRecording(LedgerGroup.computed(), 32, 8);

  /** 裏ドラは実際の値で学び、局の 5% では期待値も求めて監査する（同梱の設定の既定値と同じ）。 */
  static final UraSettings URA = new UraSettings.Realized(0.05);

  /** 一様分布を ε_b = 0.05 だけ混ぜる行動方策（乱数の使い方は評価と同じ）。 */
  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.SINGLE_RACE);

  /**
   * s を手の ID と席から作るハッシュにし、σ をその argmax に置く inline の方策。手の選び方は決定的で、打ち筋は手ごとにばらける。
   * {@code retain} のときは p0（ハッシュから作った正の重みを正規化したもの）も書く。これは収集のテストのためだけで、本番の inline の方策は
   * s も p0 も書かない。観測の記録 O は読まない（行の O は収集の側が取り込む）。
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
    public void evaluateInline(PolicyRequest request) {
      int n = request.legalCount();
      float[] scores = new float[n];
      int best = 0;
      for (int a = 0; a < n; a++) {
        long bits = SeedMixer.mix64(request.legal()[a] * 31L + request.seat());
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
          long bits = SeedMixer.mix64(request.legal()[a] * 17L + 5);
          prior[a] = 1f + (float) CrnSeeds.unit(bits);
          sum += prior[a];
        }
        for (int a = 0; a < n; a++) prior[a] = (float) (prior[a] / sum);
        request.prior = prior;
      }
    }
  }

  /**
   * 行を一時ディレクトリのコホートに書き、確定してから {@link RowCodec} で読み戻すテスト用のヘルパー。ライターは 2 本で、書いた行から
   * 運の精算の和も作る。閉じるとディレクトリを消す。
   */
  static final class TemporaryCohort implements AutoCloseable {
    final Path directory;
    final LedgerFit.WriterTap ledger = LedgerFitTest.CLASSIC.tap(2, Selection.LABELS_AND_ROOTS);
    final CohortWriter writer;
    final RowSink sink;

    /** 帳簿の有無に合わせた列の数でライターを作る。帳簿を付けない収集では {@code recording} が null。 */
    TemporaryCohort(LedgerRecording recording) throws IOException {
      directory = Files.createTempDirectory("prism-rows");
      writer = new CohortWriter(directory, TENHOU_4P, 1, recording == null ? 0 : LedgerColumn.COUNT, 2, ledger);
      sink = new RowSink(writer);
    }

    /**
     * コホートを確定して全ての行を読み戻す。マニフェストとフレームの数が食い違えばテストは失敗する。ライターが作った運の精算の和が、確定した
     * コホートを選んで読み直して当てはめた結果とビットまで一致することも確かめる。
     */
    List<Row> sealAndRead() throws IOException {
      writer.seal();
      CohortWriter.Manifest manifest = CohortWriter.readManifest(directory);
      RowCodec codec = new RowCodec(manifest.ledgerFeatures(), TENHOU_4P);
      List<Row> rows = new ArrayList<>();
      for (CohortWriter.Shard shard : manifest.shards()) {
        var frames = ByteBuffer.wrap(Files.readAllBytes(directory.resolve(shard.file())));
        while (frames.hasRemaining()) rows.add(codec.decode(frames));
      }
      Assert.assertEquals(rows.size(), manifest.rows());
      Assert.assertEquals(rows.size(), sink.rows());
      try (var store = new CohortStore();
          var readers = Executors.newFixedThreadPool(2)) {
        store.addCohort(directory);
        int[] selected = store.select(1, 1, Selection.LABELS_AND_ROOTS);
        LedgerFit.Result streamed = ledger.fit();
        Assert.assertEquals(
            streamed.report().rows() + streamed.report().skippedRows(), selected.length);
        LedgerFitTest.assertBitwiseEqual(
            streamed, LedgerFitTest.CLASSIC.fit(store.iterate(selected, 7, readers, 2)));
      }
      return rows;
    }

    @Override
    public void close() throws IOException {
      writer.close();
      Snapshot.deleteTree(directory);
    }
  }

  /**
   * 1 つの収集の結果と、確定して読み戻した行。
   *
   * @param result 収集の結果
   * @param rows 行（受け取り先に渡した順ではない）
   */
  record Collected(Collector.Result result, List<Row> rows) {}

  /**
   * 番号 {@code first} から {@code first + games − 1} までの半荘を、1 回の収集で集める。行は実行の順に依らない。
   *
   * @param ledger 帳簿の付け方（帳簿を付けない収集では null）
   */
  static Collected collectGames(
      Sampling sampling,
      LedgerRecording ledger,
      long iterationSeed,
      long first,
      int games,
      Behaviour behaviour,
      PolicyService policy,
      int workers,
      int concurrent) {
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < games; id++) ids.add(first + id);
    try (var table = new TableLoop(new TableSettings(concurrent, workers, 1000L, 4));
        var temporary = new TemporaryCohort(ledger)) {
      var cohort =
          new Collector.Cohort(
              TENHOU_4P, SEED, iterationSeed, ids, DrawOrder.SEAT_STREAMS, sampling, ledger, URA, behaviour,
              TwinDiscard.TSUMOGIRI, null);
      var result = Collector.collect(table, cohort, policy, SCORE_RANK, temporary.sink);
      List<Row> rows = temporary.sealAndRead();
      Assert.assertEquals(result.counts().rowsWritten(), rows.size());
      Assert.assertEquals(result.rows().rows(), rows.size());
      return new Collected(result, rows);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /** 半荘 100〜103 を、{@link #SAMPLING} の抽出率と、一様分布を 0.05 混ぜる行動方策で集める。 */
  private static Collected collect(int workers, int concurrent) {
    return collectGames(SAMPLING, null, SEED * 3, 100, 4, FLOOR, new HashedScorePolicy(), workers, concurrent);
  }

  /** 行の並びに依らない正準形（直列化したフレームを Base64 にして並べ替えたもの）。 */
  static List<String> encoded(List<Row> rows) {
    List<String> result = new ArrayList<>();
    for (Row row : rows)
      result.add(Base64.getEncoder().encodeToString(new RowCodec(row.ledgerFeatures(), TENHOU_4P).encode(row)));
    Collections.sort(result);
    return result;
  }

  /** 判断した席（O を観測した席）。 */
  private static int seat(Row row) {
    return new Observation(TENHOU_4P).bind(row.observation(), 0).observer();
  }

  /**
   * 主経路の探索を確かめる。温度が正なら、一様分布を混ぜなくても主経路で貪欲でない手を打つ。温度 0 で一様分布も混ぜなければ打たない。
   * softmax(s/τ) は和が 1 で、s の大きい手ほど重い。行の形（ラベル行と根の数の和が、受け取り先に渡した行の数）は保たれる。
   */
  @Test(timeOut = 300_000)
  public void mainPathExplorationPlaysNonGreedyMovesOnlyWhenEnabled() {
    float[] b = new Behaviour(0.01, 0.1, CrnScheme.SINGLE_RACE).explore(new float[] {0.3f, 0.1f, 0.2f}, 3);
    Assert.assertEquals(b[0] + b[1] + b[2], 1f, 1e-5f);
    Assert.assertTrue(b[0] > b[2] && b[2] > b[1]);
    var explored =
        collectGames(SAMPLING, null, SEED * 3, 100, 4, new Behaviour(0, 0.2, CrnScheme.SINGLE_RACE), new HashedScorePolicy(), 1, 4)
            .result();
    var greedy =
        collectGames(SAMPLING, null, SEED * 3, 100, 4, new Behaviour(0, 0, CrnScheme.SINGLE_RACE), new HashedScorePolicy(), 1, 4)
            .result();
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
   * 推論の受付から受け取ったバッチに答える方策。行ごとに {@link HashedScorePolicy} と同じ s・σ・p0 を書き、バッチごとに、含む行の
   * 容量区分の組を控える（ドライバのスレッドから呼ばれる）。{@code merged} なら受付の区分を 1 つにする（ネットワークの方策と同じ
   * {@link PolicyService#admission}）。
   */
  static final class BatchedHashedPolicy implements PolicyService {
    private final HashedScorePolicy scores = new HashedScorePolicy();
    private final boolean merged;

    /** バッチごとの、含む行の容量区分の組（受け付けた順）。 */
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
    public int admission(Observation observation) {
      return merged ? 0 : PolicyService.super.admission(observation);
    }

    @Override
    public CompletableFuture<Void> submit(List<PolicyRequest> batch, DecisionInferenceIngress ingress) {
      ingress.handoff(this).release();
      EnumSet<Capacity> capacities = EnumSet.noneOf(Capacity.class);
      Observation reader = new Observation(TENHOU_4P);
      for (PolicyRequest request : batch) {
        capacities.add(Capacity.of(request.bind(reader)));
        scores.evaluateInline(request);
      }
      batches.add(capacities);
      return CompletableFuture.completedFuture(null);
    }
  }

  /**
   * バッチで答える方策で半荘 100〜103 を集め（同時に 4 半荘、進行のワーカー 2 本）、バッチごとの行の容量区分の組と、行を返す。待ちの期限を
   * 長く取るので、バッチは全ての進行が止まってから出る。そのため、バッチの組み方は実行の速さに依らない。
   */
  private static Map.Entry<List<EnumSet<Capacity>>, List<Row>> collectBatched(boolean merged) throws Exception {
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < 4; id++) ids.add(100 + id);
    var cohort =
        new Collector.Cohort(
            TENHOU_4P, SEED, SEED * 3, ids, DrawOrder.SEAT_STREAMS, SAMPLING, null, URA, FLOOR, TwinDiscard.TSUMOGIRI, null);
    try (var table = new TableLoop(new TableSettings(4, 2, 1000000L, 4));
        var policy = new BatchedHashedPolicy(merged);
        var temporary = new TemporaryCohort(null)) {
      var result = Collector.collect(table, cohort, policy, SCORE_RANK, temporary.sink);
      List<Row> rows = temporary.sealAndRead();
      Assert.assertEquals(result.counts().rowsWritten(), rows.size());
      Assert.assertEquals(result.batches(), policy.batches.size());
      return Map.entry(policy.batches, rows);
    }
  }

  /**
   * 受付の区分（{@link PolicyService#admission}）で、どの行が同じバッチに入るかが決まることを確かめる。既定では行の容量区分ごとに受け付ける
   * ので、鳴きの行（{@link Capacity#MULTIPLE_TRANSITIONS}）とほかの行は同じバッチに入らない。区分を 1 つにする方策（ネットワークの方策）では、
   * 両方の行が 1 つのバッチに入る。行そのものは区分に依らない。
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

  /**
   * 行の種類ごとの形を確かめる。MAIN と ROLLIN の行で測った手は C ∪ {played}。根（ROOT）の行で測った手は played と d の 2 つで、
   * 傾向スコアは β_c·q。写しの中の行は逸脱した席の判断だけで、写しごとに上限の数まで。
   */
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
            int count = perCopy.merge(key, 1, Integer::sum);
            Assert.assertTrue(count <= SAMPLING.rollIn(root.decisionClass()).maxLabels(), key);
          }
        }
        case ROOT -> {
          int d = row.deviation();
          RollInRule rule = SAMPLING.rollIn(row.decisionClass());
          Assert.assertTrue((row.support() >>> d & 1) != 0, "deviation outside the support");
          for (int a = 0; a < row.actions(); a++)
            Assert.assertEquals(row.measured(a), a == row.played() || a == d, "action " + a);
          Assert.assertTrue(
              row.deviationProbability() >= rule.epsilonQ() / supportSize - 1e-6
                  && row.deviationProbability() <= 1);
          Assert.assertEquals(row.propensity(), (float) (rule.rate() * row.deviationProbability()));
          Assert.assertEquals(row.rootDecisionId(), -1L);
        }
      }
    }
    Assert.assertEquals(collected.result().counts().measured(), measuredTotal);
  }

  /**
   * q は、C の中での s の順位 r（同点なら番号の小さい手が上）に応じて 0.5^r に比例し、そこに一様分布を ε_q だけ混ぜる。d は、順位の順に
   * 並べた累積分布の逆関数で引く。
   */
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

  /** 帳簿を外した行の正準形（帳簿の列の数を 0 にして直列化する）。 */
  private static List<String> strippedCanonical(List<Row> rows) {
    var codec = new RowCodec(0, TENHOU_4P);
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

  /** 帳簿の有無だけを変えて比べる収集（一様分布を混ぜる行動方策で 2 つ、主経路の探索で 1 つ）。 */
  private static List<Collected> everyCollection(boolean ledger) {
    LedgerRecording recording = ledger ? LEDGER : null;
    List<Collected> result = new ArrayList<>();
    result.add(collectGames(SAMPLING, recording, SEED * 3, 100, 4, FLOOR, new HashedScorePolicy(), 1, 4));
    result.add(collectGames(SAMPLING, recording, SEED * 7, 300, 4, FLOOR, new HashedScorePolicy(), 1, 4));
    result.add(
        collectGames(
            SAMPLING, recording, SEED * 3, 100, 4, new Behaviour(0, 0.2, CrnScheme.SINGLE_RACE), new HashedScorePolicy(), 1, 4));
    return result;
  }

  /**
   * 帳簿を付ける収集（record）は、行に帳簿を足すだけであることを確かめる。同じ収集を帳簿なしと帳簿つきで打つと、帳簿つきの行から帳簿を
   * 外した正準形が、帳簿なしの行と一致する（帳簿の計算は、乱数の消費も打つ手も変えない）。帳簿つきの行（ラベル行と根）は全て帳簿を持ち、
   * 列の数は {@link LedgerColumn#COUNT}。測った手の帳簿は有限で、fp16 で厳密に表せる。測っていない手は NaN。帳簿の段は、帳簿つきの収集で
   * だけ段ごとの集計に出る。
   */
  @Test(timeOut = 600_000)
  public void recordModeOnlyAddsTheLedger() {
    List<Collected> off = everyCollection(false);
    List<Collected> recorded = everyCollection(true);
    long rows = 0, roots = 0, nonzero = 0;
    for (int i = 0; i < off.size(); i++) {
      Assert.assertEquals(strippedCanonical(recorded.get(i).rows()), encoded(off.get(i).rows()), "collection " + i);
      Assert.assertFalse(off.get(i).result().stages().containsKey(CollectStage.LEDGER), "collection " + i);
      Assert.assertTrue(recorded.get(i).result().stages().containsKey(CollectStage.LEDGER), "collection " + i);
      for (Row row : off.get(i).rows()) Assert.assertFalse(row.hasLedger());
      for (Row row : recorded.get(i).rows()) {
        Assert.assertTrue(row.hasLedger());
        Assert.assertEquals(row.ledgerFeatures(), LedgerColumn.COUNT);
        for (int a = 0; a < row.actions(); a++)
          for (int f = 0; f < LedgerColumn.COUNT; f++) {
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

  /** 帳簿つきの行も、実行の順・同時に進める半荘の数・ワーカーの数に依らない（席の表のキャッシュは進行ごとに持つ）。 */
  @Test(timeOut = 600_000)
  public void recordModeIsDeterministic() {
    var serial = collectGames(SAMPLING, LEDGER, SEED * 3, 100, 4, FLOOR, new HashedScorePolicy(), 1, 4).rows();
    var parallel = collectGames(SAMPLING, LEDGER, SEED * 3, 100, 4, FLOOR, new HashedScorePolicy(), 4, 2).rows();
    Assert.assertFalse(serial.isEmpty());
    Assert.assertEquals(encoded(parallel), encoded(serial));
  }

  /**
   * 本番と同じ符号化の処理を通す方策で集めた結果。行は、並びに依らない正準形で持つ。
   *
   * @param result 収集の結果
   * @param rows 行（{@link #encoded}）
   * @param queries 方策が答えた要求の数
   * @param observations 方策が受けた O のハッシュの和
   * @param tokens 方策が符号化したバッチの行のハッシュの和
   */
  private record Served(Collector.Result result, List<String> rows, long queries, long observations, long tokens) {}

  /**
   * 半荘 100〜105 を、帳簿つきで、{@link #SAMPLING} の抽出率と一様分布を 0.05 混ぜる行動方策で集める。方策は、本番と同じ符号化の処理を通す
   * {@link ServedGoldenPolicy}。
   */
  private static Served collectServed(int workers, int concurrent) throws IOException {
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < 6; id++) ids.add(100 + id);
    var cohort =
        new Collector.Cohort(
            TENHOU_4P, SEED, SEED * 3, ids, DrawOrder.SEAT_STREAMS, SAMPLING, LEDGER, URA, FLOOR, TwinDiscard.TSUMOGIRI,
            null);
    try (var table = new TableLoop(new TableSettings(concurrent, workers, 1000L, 4));
        var policy = new ServedGoldenPolicy(new PriorSettings(4.0, 0.05));
        var temporary = new TemporaryCohort(LEDGER)) {
      var result = Collector.collect(table, cohort, policy, SCORE_RANK, temporary.sink);
      List<Row> rows = temporary.sealAndRead();
      Assert.assertTrue(rows.stream().allMatch(Row::hasLedger), "a row has no ledger");
      return new Served(result, encoded(rows), policy.queries(), policy.observations(), policy.tokens());
    }
  }

  /**
   * 本番と同じ符号化の処理（取り込み・計画・符号化・答え）を通す方策で、運の精算の帳簿を付けて集めても、結果は実行の順・同時に進める半荘の
   * 数・ワーカーの数に依らない。同じ種で 2 回集めた行も、方策が受けた O も、符号化したバッチの行も一致する。
   */
  @Test(timeOut = 1_800_000)
  public void servedCollectionIsDeterministic() throws IOException {
    Served serial = collectServed(1, 4);
    Served parallel = collectServed(4, 2);
    Assert.assertTrue(serial.result().counts().forkDecisions() > 0, "no fork decisions");
    Assert.assertTrue(serial.result().counts().copyDecisions() > 0, "no copy decisions");
    Assert.assertFalse(serial.rows().isEmpty(), "no rows");
    Assert.assertTrue(serial.queries() > serial.rows().size(), "the policy answered too few queries");
    Assert.assertEquals(parallel.rows(), serial.rows());
    Assert.assertEquals(parallel.queries(), serial.queries());
    Assert.assertEquals(parallel.observations(), serial.observations());
    Assert.assertEquals(parallel.tokens(), serial.tokens());
  }

  /**
   * 局末の 3 ワードが、その（行, 手）の通った経路の局末から来ることを確かめる。点数の順位の Φ で集めると、Y は局末の点数の順位で決まる。
   * そのため、3 ワードを記録した全ての（行, 手）で、3 ワードにある 4 席の局末の点数から求めた順位が Y を再現する（分岐・写し・主経路の全て）。
   * 主経路の（行, 手）（MAIN と ROOT の打った手）では、3 ワードにある 4 席の局末の点数と聴牌への到達が、主経路を外から数えた値と一致する。
   * 外から数えるのは、局末の点数と、和了か局末の聴牌か、判断の後の打牌で聴牌したか。これで、河の打牌の数え方、局ごとの消去、判断した席の
   * 選び方を確かめる。分岐と写しの到達の数えは親の写しから始めるので、写し損ねると境界の数えが例外で止まる。
   */
  @Test(timeOut = 600_000)
  public void outcomesComeFromTheirOwnPath() throws Exception {
    var truth = new MainPath();
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < 8; id++) ids.add(700 + id);
    List<Row> rows;
    try (var table = new TableLoop(new TableSettings(4, 2, 1000L, 4));
        var policy = new HashedScorePolicy();
        var temporary = new TemporaryCohort(LEDGER)) {
      var cohort =
          new Collector.Cohort(
              TENHOU_4P, SEED, SEED * 13, ids, DrawOrder.SEAT_STREAMS, SAMPLING, LEDGER, URA, FLOOR,
              TwinDiscard.TSUMOGIRI, truth);
      Collector.collect(table, cohort, policy, SCORE_RANK, temporary.sink);
      rows = temporary.sealAndRead();
    }
    var observation = new Observation(TENHOU_4P);
    var outcomes = OutcomeCodec.of(TENHOU_4P);
    int slots = 0, mainSlots = 0, reached = 0, reachedByDiscard = 0;
    for (Row row : rows) {
      observation.bind(row.observation(), 0);
      int seat = observation.observer();
      for (int a = 0; a < row.actions(); a++) {
        int at = a * OutcomeCodec.WORDS;
        if (outcomes.outcome(row.outcomes(), at) == null) continue;
        int[] scores = new int[TENHOU_4P.seats()];
        for (int relative = 0; relative < scores.length; relative++)
          scores[(seat + relative) % scores.length] = outcomes.finalScore(row.outcomes(), at, relative);
        float y = (float) UTILITY.value(TENHOU_4P, Placement.rank(scores, seat));
        Assert.assertEquals(Float.floatToRawIntBits(row.values()[a]), Float.floatToRawIntBits(y), row.source() + " action " + a);
        slots++;
      }
      int at = row.played() * OutcomeCodec.WORDS;
      boolean mainSlot = row.source() == Row.Source.MAIN || row.source() == Row.Source.ROOT;
      if (!mainSlot || outcomes.outcome(row.outcomes(), at) == null) continue;
      MainPath.Hand hand = truth.hand(row.gameId(), observation.kyoku(), observation.honba());
      for (int relative = 0; relative < TENHOU_4P.seats(); relative++)
        Assert.assertEquals(
            outcomes.finalScore(row.outcomes(), at, relative),
            hand.scores[(seat + relative) % TENHOU_4P.seats()],
            "main path score");
      boolean expected = hand.reached(seat, observation.events());
      Assert.assertEquals(outcomes.tenpai(row.outcomes(), at), expected, "tenpai reach of " + row.source());
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
   * 主経路を外から見て、局ごとに局末の 4 席の点数、局末の聴牌（和了を含む）、打牌ごとの（席、イベントの番号、打牌の後に聴牌か）を記録する
   * （{@link #outcomesComeFromTheirOwnPath} で使う）。進行のワーカーから並行に呼ばれる（同じ半荘は 1 度に 1 スレッドから）。
   */
  static final class MainPath implements GameObserver {
    private static final ThreadLocal<Tools> TOOLS = ThreadLocal.withInitial(Tools::new);

    /** スレッドごとのヘルパー（手の形の解析と、O の取り込み）。 */
    private static final class Tools {
      final ShapeAnalyzer shapes = new ShapeAnalyzer(TENHOU_4P);
      final HandShape hand = new HandShape();
      final ObservationWriter writer = new ObservationWriter(TENHOU_4P);
      final ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
      final Observation reader = new Observation(TENHOU_4P);
    }

    /** 1 局で起きたことの記録。 */
    static final class Hand {
      final List<int[]> discards = new ArrayList<>();
      int[] scores;
      int endTenpai;

      /** 席 {@code seat} が、判断の時点（それまでのイベントの数が {@code events}）から局の終わりまでに聴牌に届いたか。 */
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

    private Hand hand(long gameId, SeatView view) {
      return hands.computeIfAbsent(gameId + ":" + view.round() + ":" + view.honba(), key -> new Hand());
    }

    private static int[] scores(TableState table) {
      int[] scores = new int[TENHOU_4P.seats()];
      for (int seat = 0; seat < scores.length; seat++) scores[seat] = table.score(seat);
      return scores;
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {
      ended.get(game.wall()).scores = scores(last);
    }

    @Override
    public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
      Action action = Action.of(decisions.legal(slot, choice));
      if (action.type() != Action.Type.DAHAI && action.type() != Action.Type.RIICHI_DAHAI) return;
      Tools tools = TOOLS.get();
      tools.hand.clear();
      for (int kind = 0; kind < Tiles.KINDS; kind++)
        for (int copy = view.count(kind); copy > 0; copy--) tools.hand.add(kind);
      for (int meld = view.meldCount(view.seat()); meld > 0; meld--) tools.hand.addMeld();
      tools.hand.remove(action.tile());
      int tenpai = tools.shapes.shanten(tools.hand) <= 0 ? 1 : 0;
      // 打牌のイベントの番号は、判断の O にあるイベントの数と同じ
      tools.batch.clear();
      int index = tools.writer.capture(view, decisions, slot, tools.batch);
      int events = tools.reader.bind(tools.batch, index).events();
      hand(game.wall(), view).discards.add(new int[] {view.seat(), events, tenpai});
    }

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      Hand hand = hand(game.wall(), full.seat(0));
      Tools tools = TOOLS.get();
      int tenpai = 0;
      for (int i = 0; i < end.outcome().winnerCount(); i++) tenpai |= 1 << end.outcome().winner(i);
      for (int seat = 0; seat < TENHOU_4P.seats(); seat++) {
        full.copyHand(seat, tools.hand);
        if (tools.shapes.shanten(tools.hand) <= 0) tenpai |= 1 << seat;
      }
      hand.endTenpai = tenpai;
      hand.scores = scores(end.next());
      ended.put(game.wall(), hand);
    }
  }
}
