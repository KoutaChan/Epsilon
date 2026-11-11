package com.epsilon.prism.play;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.play.PolicySeat;
import com.epsilon.agent.play.Query;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.policy.PrismPolicy;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism4p.input.InputSchema;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * PRISM の系列のテスト。系列は ServiceLoader で見つかり、開いたバージョンで新しいエンジンの実際の判断を評価できる（CPU は FP32 の
 * eager）。確率の和は 1、価値は s で、貪欲に選ぶ手は、s が最大の手（同点なら p0 の大きい手）を同値な打牌の切る方の手に替えたもの。
 * 合法手が 1 つなら、確率 {1}・価値 {0}・選ぶ手 0 を返す。ほかのルール（三人打ちなど）は拒む。
 */
public class PrismFamilyTest {
  private static final PolicyOptions CPU = PolicyOptions.defaults();

  @Test
  public void serviceLoaderFindsThePrismFamily() {
    Assert.assertTrue(ModelFamily.of(Series.PRISM) instanceof PrismFamily);
  }

  @Test(timeOut = 300_000)
  public void snapshotEvaluatesRealDecisions() throws Exception {
    Path root = Files.createTempDirectory("prism-family");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(snapshot, HipFixtures.network(manager, 2, 17), PolicyParams.of(1));
      }
      int[] counts = new int[2];
      float[] maxAbs = new float[1];
      try (Policy policy = ModelFamily.of(Series.PRISM).open(snapshot, TENHOU_4P, CPU)) {
        Assert.assertTrue(policy instanceof PrismPolicy, policy.getClass().getName());
        Assert.assertEquals(policy.label(), "prism:1");
        ObservationWriter writer = new ObservationWriter(TENHOU_4P);
        ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
        Observation reader = new Observation(TENHOU_4P);
        new TableWalk(TENHOU_4P, TableWalk.Policy.CALLS)
            .run(
                11,
                1,
                (engine, slot) -> {
                  if (counts[0] >= 8 && counts[1] >= 1) {
                    return;
                  }
                  Decisions decisions = engine.decisions();
                  batch.clear();
                  int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch);
                  Observation o = reader.bind(batch, index);
                  Evaluation evaluation = policy.evaluate(o);
                  int n = o.legalActions();
                  Assert.assertEquals(evaluation.probabilities().length, n);
                  Assert.assertEquals(evaluation.values().length, n);
                  if (n == 1) {
                    Assert.assertEquals(evaluation.probabilities(), new float[] {1f});
                    Assert.assertEquals(evaluation.values(), new float[] {0f});
                    Assert.assertEquals(evaluation.greedy(), 0);
                    counts[1]++;
                    return;
                  }
                  double sum = 0;
                  for (int a = 0; a < n; a++) {
                    sum += evaluation.probabilities()[a];
                    maxAbs[0] = Math.max(maxAbs[0], Math.abs(evaluation.values()[a]));
                  }
                  Assert.assertEquals(sum, 1, Evaluation.SUM_TOLERANCE);
                  int[] legal = new int[n];
                  for (int a = 0; a < n; a++) legal[a] = o.action(a).code();
                  Assert.assertEquals(
                      evaluation.greedy(), TwinDiscard.TSUMOGIRI.play(legal, Greedy.argmax(evaluation.values(), null, n)));
                  counts[0]++;
                });
      }
      Assert.assertTrue(counts[0] >= 8, "evaluated " + counts[0]);
      Assert.assertTrue(counts[1] >= 1, "single-action decisions " + counts[1]);
      Assert.assertTrue(maxAbs[0] > 1e-3f, "values must be non-trivial: " + maxAbs[0]);
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /**
   * 同値な打牌: 系列の入口（天鳳・RiichiLab・Reviewer）は既定のツモ切りで、評価の卓の入口（評価の卓・決闘・牌譜）は渡した切り方で切る。1 判断の
   * 評価・クライアントの席・評価の卓の席が、同じ手を打つ（{@link TwinPolicyWalk#walk}）。
   */
  @Test(timeOut = 600_000)
  public void everyEntryPlaysTheChosenTwin() throws Exception {
    Path root = Files.createTempDirectory("prism-family-twins");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(snapshot, HipFixtures.network(manager, 2, 17), PolicyParams.of(1));
      }
      try (Policy family = ModelFamily.of(Series.PRISM).open(snapshot, TENHOU_4P, CPU)) {
        assertTwins(TwinPolicyWalk.walk(family, TwinDiscard.TSUMOGIRI, TableWalk.Policy.GREEDY, 5, 4, 80, 16));
      }
      for (TwinDiscard twins : TwinDiscard.values()) {
        try (Policy table = TwinPolicyWalk.open(snapshot, TENHOU_4P, twins)) {
          assertTwins(TwinPolicyWalk.walk(table, twins, TableWalk.Policy.GREEDY, 5, 4, 80, 16));
        }
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /**
   * 初期化したままのネットワーク（s ≡ 0）では全ての手が同点になる。1 判断の評価の貪欲な手は、評価の卓の席が打つ手（p0 が最大の手）と同じで、
   * 合法手の先頭とは限らない。
   */
  @Test(timeOut = 300_000)
  public void blankNetworkEvaluatesTheMoveTheTablePlays() throws Exception {
    Path root = Files.createTempDirectory("prism-family-blank");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, PrismShape.standard(2));
        network.initialize(manager);
        Snapshot.write(snapshot, network, PolicyParams.of(1));
      }
      int[] counts = new int[2];
      try (Policy policy = ModelFamily.of(Series.PRISM).open(snapshot, TENHOU_4P, CPU)) {
        PolicySeat table = new PolicySeat(policy);
        ObservationWriter writer = new ObservationWriter(TENHOU_4P);
        ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
        Observation reader = new Observation(TENHOU_4P);
        new TableWalk(TENHOU_4P, TableWalk.Policy.GREEDY)
            .run(
                5,
                1,
                (engine, slot) -> {
                  Decisions decisions = engine.decisions();
                  int n = decisions.legalCount(slot);
                  if (n < 2 || counts[0] >= 48) {
                    return;
                  }
                  batch.clear();
                  int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch);
                  Evaluation evaluation = policy.evaluate(reader.bind(batch, index));
                  for (float value : evaluation.values()) Assert.assertTrue(value == 0f, "blank score " + value);
                  Query query = new Query(null, 0, slot, 0, batch, index, reader.bind(batch, index));
                  var attempt = table.tryAcquireIngress(0);
                  Assert.assertTrue(attempt.acquired());
                  table.submit(List.of(query), attempt.ingress()).join();
                  float[] sigma = new float[n];
                  sigma[evaluation.greedy()] = 1f;
                  Assert.assertEquals(query.sigma, sigma);
                  int[] legal = new int[n];
                  for (int a = 0; a < n; a++) legal[a] = decisions.legal(slot, a);
                  counts[0]++;
                  if (evaluation.greedy() != TwinDiscard.TSUMOGIRI.play(legal, 0)) {
                    counts[1]++;
                  }
                });
      }
      Assert.assertEquals(counts[0], 48);
      Assert.assertTrue(counts[1] > 0, "the prior never moved the greedy action off the first one");
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  private static void assertTwins(TwinPolicyWalk.Counts counts) {
    Assert.assertEquals(counts.pairs(), 80, counts.toString());
    Assert.assertEquals(counts.plain(), 16, counts.toString());
    Assert.assertTrue(counts.twins() > 0, counts.toString());
  }

  @Test
  public void otherVariantsAreRefused() throws Exception {
    Path root = Files.createTempDirectory("prism-family-variant");
    try {
      Path snapshot = root.resolve("1");
      IllegalStateException failure =
          Assert.expectThrows(
              IllegalStateException.class,
              () -> ModelFamily.of(Series.PRISM).open(snapshot, Variant.TENHOU_3P, CPU));
      Assert.assertEquals(failure.getMessage(), "No PRISM module for TENHOU_3P on the classpath");
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
