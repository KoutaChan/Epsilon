package com.epsilon.prism.model;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.Limits;
import com.epsilon.prism.backend.WeightLayout;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.kernel.CandidateStages;
import com.epsilon.prism.kernel.CollectionArchitecture;
import com.epsilon.prism.kernel.EncoderLayerShape;
import com.epsilon.prism.kernel.FeedForwardShape;
import com.epsilon.prism.kernel.LayerRole;
import com.epsilon.prism.kernel.TokenCounts;
import com.epsilon.prism.play.PublishedPositions;
import java.nio.ByteBuffer;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * ネットワークの形から作る HIP の形の記述のテスト（CPU、native なし）。トークンの幹では層の種類、FFN の幅、候補層、
 * ヘッド、四人打ちで決まる数（席・牌の種類など）が期待した値になる。eager のネットワークも、同じ層の並びでパラメーターを登録する。
 * v2 の形の計画では、各領域の大きさが変わらず、2 段目の領域を持たない。
 */
public class ShapeArchitectureTest {

  @DataProvider
  public Object[][] parameterCounts() {
    return new Object[][] {
      {ShapeArm.V2, 1939604L, 1766610L}, {ShapeArm.V3, 4885780L, 4712786L},
      {ShapeArm.V3_ONE_STAGE, 4719508L, 4546514L}, {ShapeArm.V3_CONTEXT_512, 3835156L, 3662162L},
      {ShapeArm.V4_S, 1985748L, 1812754L}, {ShapeArm.V4_M, 4099220L, 3843538L},
    };
  }

  @Test(dataProvider = "parameterCounts")
  public void separatesInferenceFromTrainingParameters(ShapeArm arm, long total, long inference) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(FOUR_PLAYER, arm.shape());
      network.initialize(manager);
      Assert.assertEquals(
          network.getParameters().values().stream().mapToLong(p -> p.getArray().size()).sum(),
          total);
      Assert.assertEquals(
          network.inferenceParameters().values().stream().mapToLong(p -> p.getArray().size()).sum(),
          inference);
    }
  }

  @Test
  public void countsCoreInferenceMacsOnPublishedRows() {
    var format = FOUR_PLAYER;
    var rows = PublishedPositions.sample(format.variant());
    var observation = new Observation(format.variant());
    var writer = format.writer();
    var host =
        HostInputs.allocate(
            format,
            new BatchPlan.Capacity(
                1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS),
            ByteBuffer::allocateDirect);
    var arms = List.of(new PrismShape.V4(), ModelShapes.bidirectional(256), new PrismShape.V4(384));
    long[] totals = new long[arms.size()];
    for (int r = 0; r < rows.size(); r++) {
      host.plan().clear();
      Assert.assertEquals(host.plan().add(observation.bind(rows, r)), 0);
      host.plan().seal();
      writer.write(observation, host, 0);
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        NetworkInputs inputs = NetworkInputs.of(manager, PinnedUploads.on(manager), host.input());
        for (int arm = 0; arm < arms.size(); arm++) {
          totals[arm] += coreMacs(arms.get(arm), inputs);
        }
      }
    }
    Assert.assertEquals(rows.size(), 32);
    Assert.assertEquals(totals, new long[] {1926289698L, 1948031522L, 4228307682L});
  }

  /** 単独推論の積和。表の参照・K の加算・活性化・正規化・埋め草を除く。 */
  private static long coreMacs(PrismShape.V4 shape, NetworkInputs row) {
    var format = row.format();
    var index = row.index(Pass.inference(Precision.FLOAT32));
    long fixed = format.fixedTokens(), events = row.events(), tokens = fixed + events;
    long h = shape.hidden(), a = shape.attention(), g = shape.gate();
    long embeddings = 0;
    for (TokenType type : TokenType.values()) {
      long count = type == TokenType.EVENT ? events : type.perRow(format);
      embeddings += count * type.inputWidth(format) * h;
    }
    long trunk = 0;
    for (int layer = 0; layer < shape.depth(); layer++) {
      long queries = layer == 0 ? tokens : fixed;
      long pairs = queries * tokens;
      if (layer == 0 && shape.causalEvents()) pairs = fixed * tokens + events * (events + 1) / 2;
      long ffn = layer == 0 ? shape.layer0Ffn() : shape.ffn();
      trunk += 2 * h * a * (tokens + queries) + 2 * a * pairs + 2 * h * ffn * queries;
    }
    int discard = format.width(Tensor.DISCARD_EXACT), exact = format.width(Tensor.ACTION_EXACT);
    long fields =
        FieldPack.of(format)
            .transition()
            .inputWidth(CandidateLayer.DIMENSION, discard + format.width(Tensor.TRANSITION_SCALAR));
    long transitions = row.transitions(), tiles = format.tiles();
    long candidates =
        2 * a * h * fixed
            + transitions
                * ((fields + h) * h + 2 * a * h + 3 * a * fixed + 2 * h * shape.candidateFfn());
    long counting =
        com.epsilon.prism.input.CandidateCodes.COUNT_WIDTH
            * (tiles * h + transitions * (tiles + h));
    long heads =
        transitions * (h * g + g + discard) + index.gateCount * (3 * h * g + exact * g + g + exact);
    if (index.gateCount > 0) {
      heads +=
          h
              * (index.gateChosen.width * index.gateChosen.spreadCount
                  + index.gateBase.width * index.gateBase.spreadCount);
    }
    return embeddings + trunk + candidates + counting + heads;
  }

  @DataProvider
  public Object[][] invalidShapes() {
    return new Object[][] {
      {(Runnable) () -> new PrismShape.V2(64, 128, 64, 1)},
      {(Runnable) () -> new PrismShape.V4(336)},
      {(Runnable) () -> new PrismShape.V4(384, 255, 2, PrismShape.EventAttention.CAUSAL)},
      {(Runnable) () -> new PrismShape.V4(384, 768, 4, PrismShape.EventAttention.CAUSAL)},
      {(Runnable) () -> new PrismShape.V3(64, 128, 64, 32, 128, 128, true)},
    };
  }

  @Test(dataProvider = "invalidShapes")
  public void rejectsInvalidTrunkDimensions(Runnable build) {
    Assert.assertThrows(IllegalArgumentException.class, build::run);
  }

  @Test
  public void armsKeepTheirLayerCounts() {
    Assert.assertThrows(IllegalArgumentException.class, () -> ShapeArm.V3.shape(2));
    Assert.assertEquals(ShapeArm.V4_S.shape().layers(), ShapeArm.V2.shape().layers());
  }

  private static final TokenCounts FOUR_PLAYERS = new TokenCounts(4, 34, 109, 23, 110);

  private static final EncoderLayerShape ALL_512 =
      new EncoderLayerShape(LayerRole.ALL_TOKENS, new FeedForwardShape.Shared(512));

  private static EncoderLayerShape fixed(FeedForwardShape ffn) {
    return new EncoderLayerShape(LayerRole.FIXED_TOKENS, ffn);
  }

  /** 形、層の並び、候補層の FFN の幅、候補層の段の数、G。 */
  @DataProvider
  public Object[][] shapes() {
    EncoderLayerShape v3Fixed = fixed(new FeedForwardShape.Typed(1536, 768));
    return new Object[][] {
      {
        ShapeArm.V4_S.shape(),
        List.of(ALL_512, fixed(new FeedForwardShape.Shared(512))),
        512,
        CandidateStages.ONE,
        256
      },
      {
        PrismShape.standard(2),
        List.of(ALL_512, fixed(new FeedForwardShape.Shared(512))),
        512,
        CandidateStages.ONE,
        256
      },
      {
        PrismShape.standard(3),
        List.of(ALL_512, ALL_512, fixed(new FeedForwardShape.Shared(512))),
        512,
        CandidateStages.ONE,
        256
      },
      {
        new PrismShape.V2(256, 768, 256, 2),
        List.of(
            new EncoderLayerShape(LayerRole.ALL_TOKENS, new FeedForwardShape.Shared(768)),
            fixed(new FeedForwardShape.Shared(768))),
        768,
        CandidateStages.ONE,
        256
      },
      {ShapeArm.V3.shape(3), List.of(ALL_512, v3Fixed, v3Fixed), 1024, CandidateStages.REREAD, 512},
      {
        ShapeArm.V3_ONE_STAGE.shape(3),
        List.of(ALL_512, v3Fixed, v3Fixed),
        1024,
        CandidateStages.ONE,
        512
      },
      {
        ShapeArm.V3_CONTEXT_512.shape(3),
        List.of(
            ALL_512,
            fixed(new FeedForwardShape.Typed(512, 768)),
            fixed(new FeedForwardShape.Typed(512, 768))),
        1024,
        CandidateStages.REREAD,
        512
      },
    };
  }

  @Test(dataProvider = "shapes")
  public void everyShapeDescribesItsLayersCandidateAndHeads(
      PrismShape.TokenTrunk shape,
      List<EncoderLayerShape> layers,
      int candidateFfn,
      CandidateStages stages,
      int gate) {
    CollectionArchitecture architecture = architecture(shape);
    Assert.assertEquals(shape.layers(), layers);
    Assert.assertEquals(architecture.layers(), layers);
    Assert.assertEquals(architecture.depth(), shape.depth());
    Assert.assertEquals(architecture.candidateFfn(), candidateFfn);
    Assert.assertEquals(architecture.stages(), stages);
    Assert.assertEquals(architecture.reread(), shape.reread());
    Assert.assertEquals(architecture.gate(), gate);
    Assert.assertEquals(architecture.hidden(), CollectionArchitecture.HIDDEN);
    Assert.assertEquals(architecture.epsilon(), (float) shape.epsilon());
    Assert.assertEquals((double) architecture.scale(), EagerStages.scale(), 0.0);
    Assert.assertEquals(architecture.tau(), EagerStages.tau());
    Assert.assertEquals(architecture.gateKinds(), 6);
    Assert.assertEquals(architecture.roles(), FOUR_PLAYER.roles().size());
    Assert.assertEquals(architecture.counts(), FOUR_PLAYERS);
    Assert.assertEquals(architecture.counts().fixed(), FOUR_PLAYER.fixedTokens());
    Assert.assertEquals(architecture.counts().keyCodes(), FOUR_PLAYER.keys().codes());
    Assert.assertEquals(architecture.allTokenLayers() + architecture.fixedLayers(), shape.depth());
    Assert.assertEquals(architecture.encode(), architecture(shape).encode());
    Assert.assertTrue(WeightLayout.supports(shape));
  }

  /** 収集カーネルが対応しない幅の形は、重みの束の表を作らない。 */
  @Test
  public void otherWidthsHaveNoDescription() {
    PrismShape.V2 narrow = new PrismShape.V2(128, 512, 256, 2);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> WeightLayout.of(new PrismNetwork(FOUR_PLAYER, narrow)));
    Assert.assertFalse(WeightLayout.supports(narrow));
  }

  /**
   * eager のネットワークは記述と同じ層の並びで FFN を登録する。共有の FFN の層は up を、型ごとの FFN の層は contextUp と tileUp を持ち、
   * 最後の正規化の重みも最後の層の FFN の種類に合わせる。
   */
  @Test(dataProvider = "shapes")
  public void networkRegistersTheFfnOfEachLayer(
      PrismShape.TokenTrunk shape,
      List<EncoderLayerShape> layers,
      int candidateFfn,
      CandidateStages stages,
      int gate) {
    Set<String> names = new HashSet<>();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(FOUR_PLAYER, shape);
      network.initialize(manager);
      for (Pair<String, Parameter> pair : network.getParameters()) {
        names.add(pair.getKey().substring(2));
      }
    }
    for (int layer = 0; layer < layers.size(); layer++) {
      boolean typed = layers.get(layer).ffn() instanceof FeedForwardShape.Typed;
      String encoder = "encoder" + layer + "_";
      Assert.assertEquals(names.contains(encoder + "up"), !typed, encoder + "up");
      Assert.assertEquals(names.contains(encoder + "contextUp"), typed, encoder + "contextUp");
      Assert.assertEquals(names.contains(encoder + "tileUp"), typed, encoder + "tileUp");
    }
    boolean typedLast = layers.getLast().ffn() instanceof FeedForwardShape.Typed;
    Assert.assertEquals(names.contains("finalNorm_gain"), !typedLast);
    Assert.assertEquals(names.contains("finalNorm_contextGain"), typedLast);
    Assert.assertEquals(names.contains("reread_output"), stages != CandidateStages.ONE);
  }

  /** 四人打ちのネットワークの宣言から作る、収集カーネルの形の記述。 */
  private static CollectionArchitecture architecture(PrismShape shape) {
    return CollectionArchitecture.from(new PrismNetwork(FOUR_PLAYER, shape).declaration());
  }
}
