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
import com.epsilon.prism4p.input.InputSchema;
import com.epsilon.prism4p.input.InputSchema.ActionRole;
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
 * HIP の CPU の参照実装（{@link HipReference}）が eager のネットワークと一致することを、CPU で全てのアームについて確かめる。WeightLayout、
 * WeightPack、ヘッダーの書き方、関係のコードの式、埋め込みの組の表（W·E を折り込んだ表）、層の種類、型ごとの FFN、候補層の 2 段目、
 * 価値グラフの合成を、GPU を使わずに確かめる。段ごとに eager の入力を入れて参照実装の段を回し、eager の出力と比べる（GPU のカーネルごとのテストと同じ組み方）。
 * 続けて、全段を通した s を eager の BF16 と FP32 の s と比べる。さらに、初期化したままのネットワークでは s が全て +0.0f になること、バッチを
 * 組み直しても行ごとの s がビットまで一致することを確かめる。
 */
public class HipReferenceTest {
  private static final Logger log = LoggerFactory.getLogger(HipReferenceTest.class);

  /** 段ごとの相対 RMS の上限。 */
  private static final double STAGE_LIMIT = 2e-2;

  /** 全段を通した s の相対 RMS の上限。層 0 の QKV を表へ折り込むと（L0-FOLD）丸める位置が変わるので、その差も含む。 */
  private static final double SCORE_LIMIT = 4e-2;

  private static final int ROWS = 24;

  @DataProvider
  public Object[][] arms() {
    return HipFixtures.Arm.selected().stream().map(arm -> new Object[] {arm}).toArray(Object[][]::new);
  }

  @Test(dataProvider = "arms")
  public void referenceMatchesEagerStages(HipFixtures.Arm arm) {
    List<long[]> rows = mixedRows(ROWS);
    HostBatch batch = HipFixtures.encode(rows);
    Set<ActionRole> roles = EnumSet.noneOf(ActionRole.class);
    for (Role role : HipParity.Actions.of(batch).roles()) {
      roles.add((ActionRole) role);
    }
    log.info("PRISM-HIP-REF arm={} roles in the batch: {}", arm.label(), roles);
    Assert.assertTrue(
        roles.contains(ActionRole.RIICHI)
            && roles.contains(ActionRole.CALL)
            && (roles.contains(ActionRole.KAN) || roles.contains(ActionRole.TSUMO)),
        "the batch must exercise the riichi, response and turn gates: " + roles);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = HipFixtures.network(manager, arm.shape(), 21 + arm.ordinal());
      ParameterStore store = new ParameterStore(manager, false);
      WeightLayout layout = WeightLayout.of(network);
      HipArchitecture architecture = layout.architecture();
      WeightPack pack = WeightPack.build(layout, network, 1);
      EagerStages.Stages eager;
      EagerStages.Stages full;
      try (NDManager scope = manager.newSubManager()) {
        eager = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.BFLOAT16);
      }
      try (NDManager scope = manager.newSubManager()) {
        full = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.FLOAT32);
      }
      HipReference reference = new HipReference(pack);
      HipReference.Input in = new HipReference.Input(batch);
      List<String> failures = new ArrayList<>();
      String prefix = "PRISM-HIP-REF arm=" + arm.label() + " ";
      int fixedValues = architecture.counts().fixed() * in.rows * architecture.hidden();
      int attention = architecture.attention();

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
      HipParity.Actions actions = HipParity.Actions.of(batch);
      log.info("{}T5 vs eager BF16: {}", prefix, HipParity.compare(scores, eager.scores(), actions).summary());
      log.info("{}T5 vs eager FP32: {}", prefix, HipParity.compare(scores, full.scores(), actions).summary());
      log.info("{}eager BF16 vs FP32: {}", prefix, HipParity.compare(eager.scores(), full.scores(), actions).summary());
      Diff diff = Diff.fp32(eager.scores(), scores);
      log.info(diff.line(prefix + "s vs eager BF16"));
      if (!(diff.relativeRms() <= SCORE_LIMIT)) {
        failures.add(prefix + "s vs eager BF16 " + diff.relativeRms());
      }
      Assert.assertTrue(failures.isEmpty(), "stages above " + STAGE_LIMIT + ": " + failures);
    }
  }

  @Test(dataProvider = "arms")
  public void blankNetworkScoresPositiveZero(HipFixtures.Arm arm) {
    HostBatch batch = HipFixtures.encode(mixedRows(ROWS));
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

  @Test(dataProvider = "arms")
  public void rebatchingKeepsRowScores(HipFixtures.Arm arm) {
    List<long[]> rows = mixedRows(ROWS);
    List<long[]> shuffled = new ArrayList<>(rows);
    Collections.shuffle(shuffled, new Random(3));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = HipFixtures.network(manager, arm.shape(), 31 + arm.ordinal());
      WeightPack pack = WeightPack.build(WeightLayout.of(network), network, 1);
      HostBatch first = HipFixtures.encode(rows);
      HostBatch second = HipFixtures.encode(shuffled);
      float[] a = new HipReference(pack).forward(first);
      float[] b = new HipReference(pack).forward(second);
      for (int r = 0; r < rows.size(); r++) {
        int s = shuffled.indexOf(rows.get(r));
        int count = HipFixtures.actions(first, r);
        Assert.assertEquals(HipFixtures.actions(second, s), count);
        for (int k = 0; k < count; k++) {
          Assert.assertEquals(
              Float.floatToRawIntBits(b[HipFixtures.firstAction(second, s) + k]),
              Float.floatToRawIntBits(a[HipFixtures.firstAction(first, r) + k]),
              arm.label() + " row " + r + " action " + k);
        }
      }
    }
  }

  /** 価値グラフの作り方が全て現れる行の列（{@link HipFixtures#mixedRows}）。 */
  static List<long[]> mixedRows(int rows) {
    return HipFixtures.mixedRows(rows, 900);
  }

  private static void check(List<String> failures, String label, Diff diff) {
    log.info(diff.line(label));
    if (!(diff.relativeRms() <= STAGE_LIMIT)) {
      failures.add(label + " " + diff.relativeRms());
    }
  }
}
