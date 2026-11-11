package com.epsilon.prism.play;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;

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
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.backend.PerturbedNetwork;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.policy.PrismPolicy;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism3p.input.InputSchema;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの PRISM を、モデルの系列の入口から使えることを確かめる。三人打ちの入力形式のネットワークを保存し、CPU（FP32 の eager）で開いて、
 * 実際の判断（北抜きを含む）を評価できる。v2 と v3 の形の両方で通る。model.json に書く入力とヘッドのフィンガープリントは、三人打ちのもの。
 * 初期化したままのネットワーク（s ≡ 0）なので、貪欲に選ぶ手は評価の卓の席が打つ手（p0 が最大の手を、同値な打牌の切る方の手に替えたもの）。
 */
public class ThreePlayerPolicyTest {

  private static final PolicyOptions CPU = PolicyOptions.defaults();

  private static final int DECISIONS = 8;

  @Test(timeOut = 300_000)
  public void aV2SnapshotEvaluatesRealDecisions() throws Exception {
    evaluate(PrismShape.standard(2));
  }

  @Test(timeOut = 300_000)
  public void aV3SnapshotEvaluatesRealDecisions() throws Exception {
    evaluate(new PrismShape(64, 128, 64, 3, new PrismShape.Typed(128, 256, 128, true)));
  }

  /**
   * 同値な打牌: 系列の入口（天鳳・RiichiLab・Reviewer）は既定のツモ切りで、評価の卓の入口（評価の卓・決闘・牌譜）は渡した切り方で切る。1 判断の
   * 評価・クライアントの席・評価の卓の席が、同じ手を打つ（{@link TwinPolicyWalk#walk}）。
   */
  @Test(timeOut = 600_000)
  public void everyEntryPlaysTheChosenTwin() throws Exception {
    Path root = Files.createTempDirectory("prism-3p-twins");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(
            snapshot, PerturbedNetwork.of(manager, InputSchema.FORMAT, PrismShape.standard(2), 19), PolicyParams.of(1));
      }
      try (Policy family = ModelFamily.of(Series.PRISM).open(snapshot, TENHOU_3P, CPU)) {
        assertTwins(TwinPolicyWalk.walk(family, TwinDiscard.TSUMOGIRI, TableWalk.Policy.GREEDY, 7, 4, 80, 16));
      }
      for (TwinDiscard twins : TwinDiscard.values()) {
        try (Policy table = TwinPolicyWalk.open(snapshot, TENHOU_3P, twins)) {
          assertTwins(TwinPolicyWalk.walk(table, twins, TableWalk.Policy.GREEDY, 7, 4, 80, 16));
        }
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  private static void assertTwins(TwinPolicyWalk.Counts counts) {
    Assert.assertEquals(counts.pairs(), 80, counts.toString());
    Assert.assertEquals(counts.plain(), 16, counts.toString());
    Assert.assertTrue(counts.twins() > 0, counts.toString());
  }

  private static void evaluate(PrismShape shape) throws Exception {
    Path root = Files.createTempDirectory("prism-3p-policy");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
        network.initialize(manager);
        Snapshot.write(snapshot, network, PolicyParams.of(1));
      }
      JsonObject model = JsonParser.parseString(Files.readString(snapshot.resolve("m0/model.json"))).getAsJsonObject();
      Assert.assertEquals(model.get("inputSchema").getAsString(), InputSchema.fingerprint());
      Assert.assertEquals(model.get("headSchema").getAsString(), TargetSchema.of(TENHOU_3P).fingerprint());
      Assert.assertSame(PrismNetwork.readFormat(snapshot.resolve("m0")), InputSchema.FORMAT);
      int[] counts = new int[3];
      try (Policy policy = ModelFamily.of(Series.PRISM).open(snapshot, TENHOU_3P, CPU)) {
        Assert.assertTrue(policy instanceof PrismPolicy, policy.getClass().getName());
        PolicySeat table = new PolicySeat(policy);
        ObservationWriter writer = new ObservationWriter(TENHOU_3P);
        ObservationBatch batch = new ObservationBatch(TENHOU_3P, 1 << 12);
        Observation reader = new Observation(TENHOU_3P);
        new TableWalk(TENHOU_3P, TableWalk.Policy.CALLS)
            .run(
                12,
                2,
                (engine, slot) -> {
                  Decisions decisions = engine.decisions();
                  boolean nuki = decisions.kind(slot) == EngineDecisionKind.TURN && hasNuki(decisions, slot);
                  if (counts[0] >= DECISIONS && (!nuki || counts[2] >= 2)) {
                    return;
                  }
                  batch.clear();
                  int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch);
                  Observation o = reader.bind(batch, index);
                  Evaluation evaluation = policy.evaluate(o);
                  int n = o.legalActions();
                  Assert.assertEquals(evaluation.probabilities().length, n);
                  Assert.assertEquals(evaluation.values().length, n);
                  double sum = 0;
                  for (int a = 0; a < n; a++) {
                    sum += evaluation.probabilities()[a];
                  }
                  Assert.assertEquals(sum, 1, Evaluation.SUM_TOLERANCE);
                  if (n == 1) {
                    Assert.assertEquals(evaluation.greedy(), 0);
                  } else {
                    Query query = new Query(null, 0, slot, 0, batch, index, reader.bind(batch, index));
                    var attempt = table.tryAcquireIngress(0);
                    Assert.assertTrue(attempt.acquired());
                    table.submit(List.of(query), attempt.ingress()).join();
                    float[] sigma = new float[n];
                    sigma[evaluation.greedy()] = 1f;
                    Assert.assertEquals(query.sigma, sigma);
                  }
                  counts[0]++;
                  counts[1] += n > 1 ? 1 : 0;
                  counts[2] += nuki ? 1 : 0;
                });
      }
      Assert.assertTrue(counts[0] >= DECISIONS, "evaluated " + counts[0]);
      Assert.assertTrue(counts[2] >= 1, "nuki decisions " + counts[2]);
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  private static boolean hasNuki(Decisions decisions, int slot) {
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      if (Action.of(decisions.legal(slot, index)).type() == Action.Type.NUKI) {
        return true;
      }
    }
    return false;
  }
}
