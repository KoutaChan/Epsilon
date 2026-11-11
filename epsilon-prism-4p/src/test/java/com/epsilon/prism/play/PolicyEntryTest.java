package com.epsilon.prism.play;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.policy.PrismPolicy;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 打つ側の入口（評価の相手・牌譜・天鳳・RiichiLab・Reviewer が使う {@link ModelFamily#open}）のテスト。PRISM の方策は、渡した設定の
 * まま開く。バッチの行数の上限はそのまま使い、計算の型は順伝播の型になる（CPU の BFLOAT16 は FLOAT32 と s が違う）。HIP の無い CPU では、
 * 専用のカーネルの代わりに eager で計算する。公開した v760 のバージョンも、同じ入口で開いて判断を評価できる。
 */
public class PolicyEntryTest {

  /** 公開した v760 のバージョン（読むだけで書かない）。 */
  private static final Path PUBLISHED =
      Path.of(System.getProperty("user.home"), "epsilon-ai/crr/runs/prism-v2b/snapshots/00760");

  private Path root;
  private Path snapshot;

  @BeforeClass
  public void writeSnapshot() throws Exception {
    root = Files.createTempDirectory("prism-policy-entry");
    snapshot = root.resolve("1");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Snapshot.write(snapshot, HipFixtures.network(manager, 2, 17), PolicyParams.of(1));
    }
  }

  @AfterClass(alwaysRun = true)
  public void deleteSnapshot() throws Exception {
    Snapshot.deleteTree(root);
  }

  @Test(timeOut = 300_000)
  public void theFamilyOpensWithTheGivenBatchAndPrecision() throws Exception {
    List<float[]> single;
    List<float[]> half;
    try (Policy policy = open(snapshot, new PolicyOptions("cpu", Precision.FLOAT32, 3, Kernels.EAGER))) {
      Assert.assertEquals(((PrismPolicy) policy).maxBatch(0), 3);
      single = values(policy, 12);
    }
    try (Policy policy = open(snapshot, new PolicyOptions("cpu", Precision.BFLOAT16, 5, Kernels.EAGER))) {
      Assert.assertEquals(((PrismPolicy) policy).maxBatch(0), 5);
      half = values(policy, 12);
    }
    Assert.assertEquals(half.size(), single.size());
    boolean differs = false;
    for (int i = 0; i < single.size(); i++) {
      Assert.assertEquals(half.get(i).length, single.get(i).length);
      for (int a = 0; a < single.get(i).length; a++) {
        Assert.assertEquals(half.get(i)[a], single.get(i)[a], 0.05f + 0.05f * Math.abs(single.get(i)[a]), "decision " + i);
        differs |= half.get(i)[a] != single.get(i)[a];
      }
    }
    Assert.assertTrue(differs, "BFLOAT16 scored exactly like FLOAT32: the precision was not used");
  }

  @Test(timeOut = 300_000)
  public void nativeKernelsRunEagerOnTheCpu() throws Exception {
    for (Precision precision : Precision.values()) {
      try (NetworkPolicy network =
          Snapshot.open(snapshot, List.of(Device.cpu()), null, 4, precision, InferencePath.of(Kernels.NATIVE))) {
        Assert.assertEquals(network.inference(), InferencePath.EAGER, precision.toString());
        Assert.assertEquals(network.maxBatch(0), 4);
      }
    }
    try (Policy policy = open(snapshot, new PolicyOptions("cpu", Precision.FLOAT32, 8, Kernels.NATIVE))) {
      Assert.assertFalse(values(policy, 2).isEmpty());
    }
  }

  @Test(timeOut = 600_000)
  public void thePublishedV760OpensThroughTheEntryPoint() throws Exception {
    if (!Files.isRegularFile(PUBLISHED.resolve("m0").resolve("weights.bin"))) {
      throw new SkipException("published weights are not here: " + PUBLISHED);
    }
    try (Policy policy = open(PUBLISHED, PolicyOptions.defaults())) {
      Assert.assertEquals(policy.label(), "prism:00760");
      List<float[]> values = values(policy, 8);
      Assert.assertEquals(values.size(), 8);
      Assert.assertTrue(values.stream().anyMatch(v -> v.length > 1 && v[0] != v[1]), "the values are not trivial");
    }
  }

  private static Policy open(Path checkpoint, PolicyOptions options) throws Exception {
    Policy policy = ModelFamily.of(Series.PRISM).open(checkpoint, TENHOU_4P, options);
    Assert.assertTrue(policy instanceof PrismPolicy, policy.getClass().getName());
    return policy;
  }

  /** ルールベースの打ち手の卓で、合法手が 2 つ以上ある判断を先頭から指定した数だけ評価し、その価値を返す。乱数の種が同じなので、判断の列は毎回同じ。 */
  private static List<float[]> values(Policy policy, int count) {
    ObservationWriter writer = new ObservationWriter(TENHOU_4P);
    ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
    Observation reader = new Observation(TENHOU_4P);
    List<float[]> values = new ArrayList<>();
    new TableWalk(TENHOU_4P, TableWalk.Policy.CALLS)
        .run(
            11,
            1,
            (engine, slot) -> {
              if (values.size() >= count) {
                return;
              }
              Decisions decisions = engine.decisions();
              batch.clear();
              int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch);
              Observation o = reader.bind(batch, index);
              if (o.legalActions() < 2) {
                return;
              }
              Evaluation evaluation = policy.evaluate(o);
              values.add(evaluation.values().clone());
            });
    return values;
  }
}
