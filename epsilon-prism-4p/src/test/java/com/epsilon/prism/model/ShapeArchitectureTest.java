package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.prism.backend.WeightLayout;
import com.epsilon.prism.hip.CandidateStages;
import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipCounts;
import com.epsilon.prism.hip.HipFfn;
import com.epsilon.prism.hip.HipLayer;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.hip.HipShape;
import com.epsilon.prism.hip.LayerRole;
import com.epsilon.prism4p.input.InputSchema;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * ネットワークの形から作る HIP の形の記述のテスト（CPU、native なし）。PrismShape が受ける全ての形で、層の種類、FFN の幅、候補層、
 * ヘッド、四人打ちで決まる数（席・牌の種類など）が期待した値になる。eager のネットワークも、同じ層の並びでパラメーターを登録する。
 * v2 の形の計画では、各領域の大きさが変わらず、2 段目の領域を持たない。
 */
public class ShapeArchitectureTest {

  private static final HipCounts FOUR_PLAYERS = new HipCounts(4, 34, 109, 23, 110);

  private static final HipLayer ALL_512 = new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(512));

  private static HipLayer fixed(HipFfn ffn) {
    return new HipLayer(LayerRole.FIXED_TOKENS, ffn);
  }

  /** 形、層の並び、候補層の FFN の幅、候補層の段の数、G。 */
  @DataProvider
  public Object[][] shapes() {
    HipLayer v3Fixed = fixed(new HipFfn.Typed(1536, 768));
    return new Object[][] {
      {PrismShape.standard(2), List.of(ALL_512, fixed(new HipFfn.Shared(512))), 512, CandidateStages.ONE, 256},
      {PrismShape.standard(3), List.of(ALL_512, ALL_512, fixed(new HipFfn.Shared(512))), 512, CandidateStages.ONE, 256},
      {
        new PrismShape(256, 768, 256, 2, null),
        List.of(
            new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(768)), fixed(new HipFfn.Shared(768))),
        768,
        CandidateStages.ONE,
        256
      },
      {ShapeArm.V3.shape(3), List.of(ALL_512, v3Fixed, v3Fixed), 1024, CandidateStages.REREAD, 512},
      {ShapeArm.V3_ONE_STAGE.shape(3), List.of(ALL_512, v3Fixed, v3Fixed), 1024, CandidateStages.ONE, 512},
      {
        ShapeArm.V3_CONTEXT_512.shape(3),
        List.of(ALL_512, fixed(new HipFfn.Typed(512, 768)), fixed(new HipFfn.Typed(512, 768))),
        1024,
        CandidateStages.REREAD,
        512
      },
      {
        new PrismShape(256, 512, 256, 2, new PrismShape.Typed(768, 1536, 512, false)),
        List.of(ALL_512, v3Fixed),
        512,
        CandidateStages.ONE,
        256
      },
    };
  }

  @Test(dataProvider = "shapes")
  public void everyShapeDescribesItsLayersCandidateAndHeads(
      PrismShape shape, List<HipLayer> layers, int candidateFfn, CandidateStages stages, int gate) {
    HipArchitecture architecture = shape.architecture(InputSchema.FORMAT);
    Assert.assertEquals(shape.layers(), layers);
    Assert.assertEquals(architecture.layers(), layers);
    Assert.assertEquals(architecture.depth(), shape.depth());
    Assert.assertEquals(architecture.candidateFfn(), candidateFfn);
    Assert.assertEquals(architecture.stages(), stages);
    Assert.assertEquals(architecture.reread(), shape.reread());
    Assert.assertEquals(architecture.gate(), gate);
    Assert.assertEquals(architecture.hidden(), HipArchitecture.HIDDEN);
    Assert.assertEquals(architecture.epsilon(), (float) shape.epsilon());
    Assert.assertEquals((double) architecture.scale(), EagerStages.scale(), 0.0);
    Assert.assertEquals(architecture.tau(), EagerStages.tau());
    Assert.assertEquals(architecture.gateKinds(), 6);
    Assert.assertEquals(architecture.roles(), InputSchema.ActionRole.values().length);
    Assert.assertEquals(architecture.counts(), FOUR_PLAYERS);
    Assert.assertEquals(architecture.counts().fixed(), InputSchema.FORMAT.fixedTokens());
    Assert.assertEquals(architecture.counts().keyCodes(), InputSchema.FORMAT.keys().codes());
    Assert.assertEquals(architecture.allTokenLayers() + architecture.fixedLayers(), shape.depth());
    Assert.assertEquals(architecture.encode(), shape.architecture(InputSchema.FORMAT).encode());
    Assert.assertTrue(WeightLayout.supports(shape));
  }

  /** H が 256 でない形は記述を作らない（HIP のカーネルは H 256 に特化している）。 */
  @Test
  public void otherWidthsHaveNoDescription() {
    PrismShape narrow = new PrismShape(128, 512, 256, 2, null);
    Assert.assertThrows(IllegalArgumentException.class, () -> narrow.architecture(InputSchema.FORMAT));
    Assert.assertFalse(WeightLayout.supports(narrow));
  }

  /**
   * eager のネットワークは記述と同じ層の並びで FFN を登録する。共有の FFN の層は up を、型ごとの FFN の層は contextUp と tileUp を持ち、
   * 最後の正規化の重みも最後の層の FFN の種類に合わせる。
   */
  @Test(dataProvider = "shapes")
  public void networkRegistersTheFfnOfEachLayer(
      PrismShape shape, List<HipLayer> layers, int candidateFfn, CandidateStages stages, int gate) {
    Set<String> names = new HashSet<>();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
      network.initialize(manager);
      for (Pair<String, Parameter> pair : network.getParameters()) {
        names.add(pair.getKey().substring(2));
      }
    }
    for (int layer = 0; layer < layers.size(); layer++) {
      boolean typed = layers.get(layer).ffn() instanceof HipFfn.Typed;
      String encoder = "encoder" + layer + "_";
      Assert.assertEquals(names.contains(encoder + "up"), !typed, encoder + "up");
      Assert.assertEquals(names.contains(encoder + "contextUp"), typed, encoder + "contextUp");
      Assert.assertEquals(names.contains(encoder + "tileUp"), typed, encoder + "tileUp");
    }
    boolean typedLast = layers.getLast().ffn() instanceof HipFfn.Typed;
    Assert.assertEquals(names.contains("finalNorm_gain"), !typedLast);
    Assert.assertEquals(names.contains("finalNorm_contextGain"), typedLast);
    Assert.assertEquals(names.contains("reread_output"), stages == CandidateStages.REREAD);
  }

  /**
   * v2 の形の計画では、各領域の大きさが変わらない（関係のコードは行ごとに 160² ワード）。イベントのキーとバリューの領域と、2 段目の
   * 領域は持たない。局・席・牌の層が 1 つだけで、その Attention がイベントの k・v を n から射影するため。型ごとの FFN を持つ形の
   * アーム（V3 など）は、2 つ目の局・席・牌の層が使うイベントのキーとバリューを持ち、2 段目があれば全てのトークンの 2 段目のキーと
   * バリューも持つ。
   */
  @Test
  public void v2RegionsKeepTheirSizes() {
    int rows = 64;
    int events = 3000;
    int transitions = 900;
    int actions = 700;
    long input = 1 << 20;
    for (PrismShape shape : List.of(PrismShape.standard(2), PrismShape.standard(3))) {
      HipShape plan = plan(shape, rows, events, transitions, actions, input);
      long hidden = 256;
      long attention = 128;
      long fixed = 39L * rows;
      long tokens = fixed + events;
      Assert.assertEquals(HipRegion.INPUT.capacityBytes(plan), input);
      Assert.assertEquals(HipRegion.RESIDUAL.capacityBytes(plan), tokens * hidden * 2);
      Assert.assertEquals(HipRegion.NORMALIZED.capacityBytes(plan), tokens * hidden * 2);
      Assert.assertEquals(HipRegion.QKV0.capacityBytes(plan), tokens * 3 * attention * 2);
      Assert.assertEquals(HipRegion.ATTENTION.capacityBytes(plan), tokens * attention * 2);
      Assert.assertEquals(HipRegion.RELATION_CODE.capacityBytes(plan), rows * (160L / 16) * (160L / 16) * 256 * 4);
      Assert.assertEquals(HipRegion.STATE.capacityBytes(plan), fixed * hidden * 2);
      Assert.assertEquals(HipRegion.MEMORY.capacityBytes(plan), fixed * 2 * attention * 2);
      Assert.assertEquals(HipRegion.STATIC.capacityBytes(plan), transitions * hidden * 2);
      Assert.assertEquals(HipRegion.TRANSITION.capacityBytes(plan), transitions * hidden * 2);
      Assert.assertEquals(HipRegion.VALUE.capacityBytes(plan), transitions * 4L);
      Assert.assertEquals(HipRegion.SCORES.capacityBytes(plan), actions * 4L);
      Assert.assertEquals(HipRegion.EVENT_KEY_VALUE.capacityBytes(plan), 0L);
      Assert.assertEquals(HipRegion.REREAD_MEMORY.capacityBytes(plan), 0L);
    }
    for (ShapeArm arm : List.of(ShapeArm.V3, ShapeArm.V3_ONE_STAGE, ShapeArm.V3_CONTEXT_512)) {
      PrismShape shape = arm.shape(ShapeArm.V3_DEPTH);
      HipShape plan = plan(shape, rows, events, transitions, actions, input);
      long tokens = 39L * rows + events;
      Assert.assertEquals(HipRegion.EVENT_KEY_VALUE.capacityBytes(plan), events * 2 * 128 * 2L, arm.name());
      Assert.assertEquals(HipRegion.REREAD_MEMORY.capacityBytes(plan), shape.reread() ? tokens * 2 * 128 * 2 : 0, arm.name());
    }
  }

  private static HipShape plan(PrismShape shape, int rows, int events, int transitions, int actions, long input) {
    return new HipShape(
        shape.architecture(InputSchema.FORMAT), new HipShape.Capacity(rows, events, transitions, actions, input), 1);
  }
}
