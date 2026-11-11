package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputFormat.Role;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism3p.input.InputSchema;
import com.epsilon.prism3p.input.InputSchema.ActionRole;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 三人打ちの HIP の CPU 参照実装（{@link HipReference}）が、eager のネットワークと一致するかを確かめる（CPU、エンジンで打った三人打ちの行、
 * v2 の形と V3）。参照の各段に eager の段の入力を与えて実行し、eager の出力と比べる。続けて通しの s を eager の BF16 と比べる。初期化した
 * ままのネットワークでは s が全て +0.0f になること、行を並べ替え、前に別の行を足してバッチを組み直しても同じ行の s がビット単位で同じことも
 * 確かめる。
 */
public class ThreePlayerHipReferenceTest {
  private static final Logger log = LoggerFactory.getLogger(ThreePlayerHipReferenceTest.class);

  /** 段ごとの相対 RMS の上限。 */
  private static final double STAGE_LIMIT = 2e-2;

  /** 通しの s の相対 RMS の上限。埋め込みの表を層 0 の QKV に折り込んだため、丸める位置が eager と違う分の差も含む。 */
  private static final double SCORE_LIMIT = 4e-2;

  private static final int ROWS = 24;

  @DataProvider
  public Object[][] arms() {
    return new Object[][] {{ThreePlayerRows.Arm.V2}, {ThreePlayerRows.Arm.V3}};
  }

  @Test(dataProvider = "arms")
  public void referenceMatchesEagerStages(ThreePlayerRows.Arm arm) {
    HostBatch batch = ThreePlayerRows.encode(ThreePlayerRows.mixedRows(ROWS, 900));
    Set<ActionRole> roles = EnumSet.noneOf(ActionRole.class);
    for (Role role : HipParity.Actions.of(batch).roles()) {
      roles.add((ActionRole) role);
    }
    log.info("PRISM-3P-HIP-REF arm={} roles in the batch: {}", arm.label(), roles);
    Assert.assertTrue(
        roles.containsAll(EnumSet.of(ActionRole.RIICHI, ActionRole.CALL, ActionRole.NUKI, ActionRole.PASS))
            && (roles.contains(ActionRole.KAN) || roles.contains(ActionRole.TSUMO)),
        "the batch must exercise the riichi, response and turn gates with nuki: " + roles);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = ThreePlayerRows.network(manager, arm.shape(), 21 + arm.ordinal());
      ParameterStore store = new ParameterStore(manager, false);
      WeightLayout layout = WeightLayout.of(network);
      HipArchitecture architecture = layout.architecture();
      WeightPack pack = WeightPack.build(layout, network, 1);
      EagerStages.Stages eager;
      try (NDManager scope = manager.newSubManager()) {
        eager = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.BFLOAT16);
      }
      HipReference reference = new HipReference(pack);
      HipReference.Input in = new HipReference.Input(batch);
      List<String> failures = new ArrayList<>();
      String prefix = "PRISM-3P-HIP-REF arm=" + arm.label() + " ";
      int fixedValues = architecture.counts().fixed() * in.rows * architecture.hidden();

      reference.allocate(in);
      reference.embed(in);
      check(failures, prefix + "K1 x", Diff.bf16(eager.embedded(), reference.residual));
      check(failures, prefix + "K1 t_s", Diff.bf16(eager.statics(), reference.statics));
      int allTokens = architecture.allTokenLayers();
      for (int layer = 0; layer < allTokens; layer++) {
        reference.residual = (layer == 0 ? eager.embedded() : eager.layers().get(layer - 1)).clone();
        reference.attention(in, layer);
        check(failures, prefix + "K2 o layer " + layer, Diff.bf16(eager.attention().get(layer), reference.attentionOut));
        reference.attentionOut = eager.attention().get(layer).clone();
        reference.mlp(in, layer);
        check(failures, prefix + "K3 x layer " + layer, Diff.bf16(eager.layers().get(layer), reference.residual));
      }
      float[] backbone = eager.layers().get(allTokens - 1);
      reference.residual = backbone.clone();
      reference.eventKeyValue(in);
      for (int f = 0; f < architecture.fixedLayers(); f++) {
        check(failures, prefix + "K8 kv layer " + (allTokens + f),
            Diff.bf16(eager.eventKeyValues().get(f), reference.eventKeyValues[f]));
      }
      for (int layer = allTokens; layer < architecture.depth(); layer++) {
        reference.residual = backbone.clone();
        System.arraycopy(eager.layers().get(layer - 1), 0, reference.residual, 0, fixedValues);
        reference.attention(in, layer);
        float[] attended = eager.attention().get(layer);
        check(failures, prefix + "K4 o layer " + layer,
            Diff.bf16(attended, Arrays.copyOf(reference.attentionOut, attended.length)));
        System.arraycopy(attended, 0, reference.attentionOut, 0, attended.length);
        if (layer + 1 < architecture.depth()) {
          reference.mlp(in, layer);
          check(failures, prefix + "K3 fixed x layer " + layer,
              Diff.bf16(eager.layers().get(layer), Arrays.copyOf(reference.residual, fixedValues)));
        } else {
          reference.mlpLast(in, layer);
          check(failures, prefix + "K5 y", Diff.bf16(eager.states(), reference.state));
          check(failures, prefix + "K5 KV", Diff.bf16(eager.memory(), reference.memory));
        }
      }
      if (architecture.reread()) {
        reference.state = eager.states().clone();
        reference.rereadKeyValue(in);
        check(failures, prefix + "K8 reread", Diff.bf16(eager.rereadMemory(), reference.rereadMemory));
      }
      reference.statics = eager.statics().clone();
      reference.state = eager.states().clone();
      reference.memory = eager.memory().clone();
      reference.candidate(in);
      check(failures, prefix + (architecture.reread() ? "K6 z1" : "K6 z"),
          Diff.bf16(eager.candidates(), reference.transitions));
      if (architecture.reread()) {
        reference.transitions = eager.candidates().clone();
        reference.rereadMemory = eager.rereadMemory().clone();
        reference.reread(in);
        check(failures, prefix + "K9 z", Diff.bf16(eager.transitions(), reference.transitions));
      }
      check(failures, prefix + "B", Diff.fp32(eager.values(), reference.values));
      reference.transitions = eager.transitions().clone();
      reference.values = eager.values().clone();
      reference.valueGraph(in);
      check(failures, prefix + "K7 s", Diff.fp32(eager.scores(), reference.scores));

      float[] scores = new HipReference(pack).forward(batch);
      log.info("{}s vs eager BF16: {}", prefix,
          HipParity.compare(scores, eager.scores(), HipParity.Actions.of(batch)).summary());
      Diff diff = Diff.fp32(eager.scores(), scores);
      log.info(diff.line(prefix + "s vs eager BF16"));
      if (!(diff.relativeRms() <= SCORE_LIMIT)) {
        failures.add(prefix + "s vs eager BF16 " + diff.relativeRms());
      }
      Assert.assertTrue(failures.isEmpty(), "stages above " + STAGE_LIMIT + ": " + failures);
    }
  }

  @Test(dataProvider = "arms")
  public void blankNetworkScoresPositiveZero(ThreePlayerRows.Arm arm) {
    HostBatch batch = ThreePlayerRows.encode(ThreePlayerRows.mixedRows(ROWS, 900));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      manager.getEngine().setRandomSeed(5);
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, arm.shape());
      network.initialize(manager);
      WeightPack pack = WeightPack.build(WeightLayout.of(network), network, 0);
      float[] scores = new HipReference(pack).forward(batch);
      for (int a = 0; a < scores.length; a++) {
        Assert.assertEquals(Float.floatToRawIntBits(scores[a]), 0, arm.label() + " blank score " + a + " = " + scores[a]);
      }
    }
  }

  /** 行を並べ替え、前に別の行を足してバッチを組み直しても、同じ行の s はビット単位で同じ（イベントの多い行も含む）。 */
  @Test(dataProvider = "arms")
  public void rebatchingKeepsRowScores(ThreePlayerRows.Arm arm) {
    List<long[]> rows = new ArrayList<>(ThreePlayerRows.mixedRows(ROWS - 4, 900));
    rows.addAll(ThreePlayerRows.longRows(4));
    List<long[]> shuffled = new ArrayList<>(rows);
    Collections.shuffle(shuffled, new Random(3));
    List<long[]> prefixed = new ArrayList<>(ThreePlayerRows.simulatedRows(7, 700));
    prefixed.addAll(shuffled);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = ThreePlayerRows.network(manager, arm.shape(), 31 + arm.ordinal());
      WeightPack pack = WeightPack.build(WeightLayout.of(network), network, 1);
      HostBatch first = ThreePlayerRows.encode(rows);
      HostBatch second = ThreePlayerRows.encode(prefixed);
      float[] a = new HipReference(pack).forward(first);
      float[] b = new HipReference(pack).forward(second);
      for (int r = 0; r < rows.size(); r++) {
        int s = prefixed.indexOf(rows.get(r));
        int count = first.plan().actions(r);
        Assert.assertEquals(second.plan().actions(s), count);
        for (int k = 0; k < count; k++) {
          Assert.assertEquals(
              Float.floatToRawIntBits(b[second.plan().firstAction(s) + k]),
              Float.floatToRawIntBits(a[first.plan().firstAction(r) + k]),
              arm.label() + " row " + r + " action " + k);
        }
      }
    }
  }

  private static void check(List<String> failures, String label, Diff diff) {
    log.info(diff.line(label));
    if (!(diff.relativeRms() <= STAGE_LIMIT)) {
      failures.add(label + " " + diff.relativeRms());
    }
  }
}
