package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
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
 * HIP の CPU の参照（{@link HipReference}）と eager の網の一致（CPU）。WeightLayout・WeightPack・頭の書き方・関係の符号の式・畳んだ組の
 * 表・価値グラフの合成を、GPU を使わずに確かめる。段ごとに eager の入力を入れて参照の段を回し、eager の出力と比べる（GPU のカーネルごとの
 * 試験と同じ組み方）。続けて通しの s を eager の BF16 と FP32 の s と比べ、白紙で s が全て +0.0f、塊を組み直しても s がビットで同じことを
 * 確かめる。
 */
public class HipReferenceTest {
  private static final Logger log = LoggerFactory.getLogger(HipReferenceTest.class);

  /** 段ごとの相対 RMS の上限。 */
  private static final double STAGE_LIMIT = 2e-2;

  private static final int ROWS = 24;

  @DataProvider
  public Object[][] depths() {
    return new Object[][] {{2}, {3}};
  }

  @Test(dataProvider = "depths")
  public void referenceMatchesEagerStages(int depth) {
    List<long[]> rows = mixedRows(ROWS);
    HostBatch batch = HipFixtures.encode(rows);
    Set<ActionRole> roles = EnumSet.noneOf(ActionRole.class);
    roles.addAll(Arrays.asList(HipParity.Actions.of(batch).roles()));
    log.info("PRISM-HIP-REF depth={} roles in the batch: {}", depth, roles);
    Assert.assertTrue(
        roles.contains(ActionRole.RIICHI)
            && roles.contains(ActionRole.CALL)
            && (roles.contains(ActionRole.KAN) || roles.contains(ActionRole.TSUMO)),
        "the batch must exercise the riichi, response and turn gates: " + roles);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = HipFixtures.network(manager, depth, 21 + depth);
      ParameterStore store = new ParameterStore(manager, false);
      WeightLayout layout = WeightLayout.of(network);
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
      String prefix = "PRISM-HIP-REF depth=" + depth + " ";

      reference.allocate(in);
      reference.embed(in);
      check(failures, prefix + "K1 x", Diff.bf16(eager.embedded(), reference.residual));
      check(failures, prefix + "K1 t_s", Diff.bf16(eager.statics(), reference.statics));
      int fixedTokens = 39 * in.rows;
      int attention = layout.shape().attention();
      for (int layer = 0; layer + 1 < depth; layer++) {
        reference.residual = (layer == 0 ? eager.embedded() : eager.layers().get(layer - 1)).clone();
        reference.attention(in, layer, false);
        check(failures, prefix + "K2 o layer " + layer, Diff.bf16(eager.attention().get(layer), reference.attentionOut));
        reference.attentionOut = eager.attention().get(layer).clone();
        reference.mlp(in, layer);
        check(failures, prefix + "K3 x layer " + layer, Diff.bf16(eager.layers().get(layer), reference.residual));
      }
      int last = depth - 1;
      reference.residual = eager.layers().get(last - 1).clone();
      reference.attention(in, last, true);
      check(
          failures,
          prefix + "K4 o",
          Diff.bf16(eager.attention().get(last), Arrays.copyOf(reference.attentionOut, fixedTokens * attention)));
      System.arraycopy(eager.attention().get(last), 0, reference.attentionOut, 0, fixedTokens * attention);
      reference.mlpKv(in, last);
      check(failures, prefix + "K5 y", Diff.bf16(eager.states(), reference.state));
      check(failures, prefix + "K5 KV", Diff.bf16(eager.memory(), reference.memory));
      reference.statics = eager.statics().clone();
      reference.state = eager.states().clone();
      reference.memory = eager.memory().clone();
      reference.candidate(in);
      check(failures, prefix + "K6 z", Diff.bf16(eager.transitions(), reference.transitions));
      check(failures, prefix + "K6 B", Diff.fp32(eager.values(), reference.values));
      reference.transitions = eager.transitions().clone();
      reference.values = eager.values().clone();
      reference.valueGraph(in);
      check(failures, prefix + "K7 s", Diff.fp32(eager.scores(), reference.scores));

      float[] scores = new HipReference(pack).forward(batch);
      HipParity.Actions actions = HipParity.Actions.of(batch);
      log.info(
          "PRISM-HIP-REF depth={} T5 vs eager BF16: {}", depth, HipParity.compare(scores, eager.scores(), actions).summary());
      log.info(
          "PRISM-HIP-REF depth={} T5 vs eager FP32: {}", depth, HipParity.compare(scores, full.scores(), actions).summary());
      log.info(
          "PRISM-HIP-REF depth={} eager BF16 vs FP32: {}",
          depth,
          HipParity.compare(eager.scores(), full.scores(), actions).summary());
      check(failures, prefix + "s vs eager BF16", Diff.fp32(eager.scores(), scores));
      Assert.assertTrue(failures.isEmpty(), "stages above " + STAGE_LIMIT + ": " + failures);
    }
  }

  @Test
  public void blankNetworkScoresPositiveZero() {
    HostBatch batch = HipFixtures.encode(mixedRows(ROWS));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      manager.getEngine().setRandomSeed(5);
      PrismNetwork network = new PrismNetwork(PrismShape.standard());
      network.initialize(manager);
      WeightPack pack = WeightPack.build(WeightLayout.of(network), network, 0);
      float[] scores = new HipReference(pack).forward(batch);
      for (int a = 0; a < scores.length; a++) {
        Assert.assertEquals(Float.floatToRawIntBits(scores[a]), 0, "blank score " + a + " = " + scores[a]);
      }
    }
  }

  @Test
  public void rebatchingKeepsRowScores() {
    List<long[]> rows = mixedRows(ROWS);
    List<long[]> shuffled = new ArrayList<>(rows);
    Collections.shuffle(shuffled, new Random(3));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = HipFixtures.network(manager, 2, 31);
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
              "row " + r + " action " + k);
        }
      }
    }
  }

  /** 価値グラフの全ての作り方が出る行（{@link HipFixtures#mixedRows}）。 */
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
