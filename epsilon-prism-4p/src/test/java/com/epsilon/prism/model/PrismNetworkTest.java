package com.epsilon.prism.model;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.prism.Limits;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputSamples;
import com.epsilon.prism.model.Batches.Batch;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * ネットワークを確かめる（CPU、FP32）。見るのは、初期化したままで s ≡ 0 になること、アンカーの勾配の表、乱数の Q を学習できること、
 * 2 ステップで全てのパラメーターに勾配が届くこと、補助のヘッドの有無で s がビットで同じこと、補助と差分整合の勾配の分離、イベント 0 個と
 * 最長の行で値が有限なこと、保存と読み込み。全ての形で同じ契約を確かめる。
 */
public class PrismNetworkTest {
  private static final Logger log = LoggerFactory.getLogger(PrismNetworkTest.class);

  /** ネットワークとパラメーターのマネージャー。 */
  private static final class Net implements AutoCloseable {

    final NDManager manager;
    final PrismNetwork network;
    final ParameterStore store;

    Net(PrismShape shape, int seed) {
      this(shape, seed, Device.cpu());
    }

    Net(PrismShape shape, int seed, Device device) {
      manager = NDManager.newBaseManager(device);
      manager.getEngine().setRandomSeed(seed);
      network = new PrismNetwork(FOUR_PLAYER, shape);
      network.initialize(manager);
      store = new ParameterStore(manager, false);
    }

    Outputs forward(NDManager scope, Batch batch, Pass pass) {
      return network.forward(store, batch.inputs(scope), pass);
    }

    NDArray scores(HostBatch batch) {
      NetworkInputs inputs = NetworkInputs.of(manager, PinnedUploads.on(manager), batch);
      return network.forward(store, inputs, Pass.inference(Precision.FLOAT32)).scores();
    }

    NDArray parameter(String suffix) {
      return network.getParameters().stream()
          .filter(p -> p.getKey().endsWith(suffix))
          .findFirst()
          .orElseThrow()
          .getValue()
          .getArray();
    }

    void fill(String suffix, float value) {
      NDArray parameter = parameter(suffix);
      float[] values = new float[Math.toIntExact(parameter.size())];
      Arrays.fill(values, value);
      parameter.set(values);
    }

    void unitWeight(String suffix, int index) {
      NDArray parameter = parameter(suffix);
      float[] values = new float[Math.toIntExact(parameter.size())];
      values[index] = 1f;
      parameter.set(values);
    }

    @Override
    public void close() {
      manager.close();
    }
  }

  @Test(dataProvider = "smallShapes", dataProviderClass = ModelShapes.class)
  public void blankNetworkScoresZeroForEveryDecisionKind(PrismShape shape) {
    try (Net net = new Net(shape, 1);
        NDManager scope = net.manager.newSubManager()) {
      Batch batch = Batches.build(Batches.allKinds(5), 11);
      float[] scores =
          net.forward(scope, batch, Pass.inference(Precision.FLOAT32)).scores().toFloatArray();
      for (float score : scores) {
        Assert.assertEquals(score, 0f, "blank score");
      }
    }
  }

  /** FP32 スコアのビット列と、関係の表の無効行も含めて登録順に連結した FP32 勾配のビット列の SHA-256。 */
  @DataProvider
  public Object[][] scoreAndGradientBits() {
    return new Object[][] {
      {
        ShapeArm.V2.shape(),
        Precision.FLOAT32,
        "552836acf68520ffe21e5f9562d73b423b288e82d1cdf6014c9165cd6888fe26",
        "9c698296f85a41192ac1b394251d22ea3e1dae6e3cc87b887227a3dd468b8391"
      },
      {
        ShapeArm.V2.shape(),
        Precision.BFLOAT16,
        "df166f94f1ea1b6fd276d104836a2c42c9ad028609a28dfbfcbbbe5294d04ee2",
        "39a399ce293ba138fe73c708190242038ac4efc202bb3b2092672f0af4427769"
      },
      {
        ShapeArm.V2.shape(3),
        Precision.FLOAT32,
        "562ef583cf9b0f7e06ee1eefbcc2c74ba9562f899c59f2be6b8bd218e86d04f8",
        "b64ba5292b4dee4136d0dd7f17a8ae319235114bac792f5886aedda84b26ca3b"
      },
      {
        ShapeArm.V2.shape(3),
        Precision.BFLOAT16,
        "696b6e4394d0a53248ef20bd51014d578e7941142d32fdee86816310eaf20296",
        "3a671d85c0310a6684fb8fe55df36219c0cf8e7d42e9927b356ba1a8095bfd64"
      },
      {
        ShapeArm.V3.shape(),
        Precision.FLOAT32,
        "091e13b97db12c4bd5cb0a3f620e664c455017adb004ed9c19c1f5eef772be80",
        "9abf5028cbef317664e09f97f06e0ee8c78126d8bc2915a0da3d7b798e6645d7"
      },
      {
        ShapeArm.V3.shape(),
        Precision.BFLOAT16,
        "057452f04885e5abc5aade5e426ed69a961c4b2c2a7b53bd4facc5868c357247",
        "873fb98479483fbb7984b14acfff1796374cf38c8b4d8451556f56ee658a1eb9"
      },
      {
        ShapeArm.V3_ONE_STAGE.shape(),
        Precision.FLOAT32,
        "e5538d785cd5ee9962ef467c66066103bc66474897c3862f8c3167c058de551c",
        "4bac0dd896bea05615af1bb198618940c08ee330511d643a07afab01d47b8383"
      },
      {
        ShapeArm.V3_ONE_STAGE.shape(),
        Precision.BFLOAT16,
        "9262f9b6e918a0bc1e02e2b2e8fcad54cbb3808c2225e890d9cbf89f21da23aa",
        "94149f268d0834996668f23ad392d96a3750a3f8911ea23a0660e550424edbe5"
      },
      {
        ShapeArm.V3_CONTEXT_512.shape(),
        Precision.FLOAT32,
        "9cea2f18e24d73040d935d3db0231bc50bef41afeba4c43546f5db71cd69699f",
        "df99d54bd91728d4f64dd095b01b9ef9a1af10626e77f326a0f1670ebfe195e5"
      },
      {
        ShapeArm.V3_CONTEXT_512.shape(),
        Precision.BFLOAT16,
        "420591450abc9f5715c863dd4c7b9ab676943f89a23bf6f1c60376cec7b8f263",
        "6f807a26ca04a9cf870934909f6745defb1a9636940a4c8f3ef9b21abd34a3c5"
      },
    };
  }

  @Test(dataProvider = "scoreAndGradientBits")
  public void preservesScoreAndGradientBits(
      PrismShape shape, Precision precision, String scoreBits, String gradientBits)
      throws NoSuchAlgorithmException {
    try (Net net = new Net(shape, 31)) {
      Batches.perturb(net.network, 0.05f, 37);
      Batch batch = Batches.build(Batches.allKinds(4), 41);
      List<ValueLabel> labels = Batches.values(batch, 43);
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 47);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      MessageDigest sha = MessageDigest.getInstance("SHA-256");
      NDArray inferred = net.forward(net.manager, batch, Pass.inference(precision)).scores();
      sha.update(inferred.toByteBuffer());
      Assert.assertEquals(HexFormat.of().formatHex(sha.digest()), scoreBits, "inference scores");
      try (GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
        Outputs trained =
            net.forward(
                net.manager,
                batch,
                Pass.training(precision, ValueLabel.measured(labels), auxiliary));
        sha.update(trained.scores().toByteBuffer());
        Assert.assertEquals(HexFormat.of().formatHex(sha.digest()), scoreBits, "training scores");
        collector.backward(
            loss.total(net.manager, trained, labels, auxiliary, 1f, labels.size(), 1f));
      }
      for (Pair<String, Parameter> p : net.network.getParameters()) {
        NDArray gradient = p.getValue().getArray().getGradient();
        sha.update(gradient.toByteBuffer());
      }
      Assert.assertEquals(
          HexFormat.of().formatHex(sha.digest()), gradientBits, "parameter gradients");
    }
  }

  @DataProvider
  public Object[][] precisions() {
    return Arrays.stream(Precision.values())
        .map(precision -> new Object[] {precision})
        .toArray(Object[][]::new);
  }

  @DataProvider
  public Object[][] candidateStages() {
    return new Object[][] {{ShapeArm.V3_ONE_STAGE, ShapeArm.V3}};
  }

  @Test(dataProvider = "candidateStages")
  public void blankSecondStageKeepsTheOneStageFunction(ShapeArm oneStage, ShapeArm twoStages) {
    try (Net one = new Net(oneStage.shape(), 11);
        Net two = new Net(twoStages.shape(), 13)) {
      Batches.perturb(one.network, 0.05f, 43);
      Map<String, Parameter> shared = new HashMap<>();
      one.network
          .getParameters()
          .forEach(pair -> shared.put(pair.getKey().substring(2), pair.getValue()));
      for (Pair<String, Parameter> pair : two.network.getParameters()) {
        String name = pair.getKey().substring(2);
        NDArray target = pair.getValue().getArray();
        if (name.startsWith("reread_output")) {
          Assert.assertTrue(target.eq(0).all().getBoolean(), name);
        }
        if (!name.startsWith("reread_")) {
          target.set(shared.get(name).getArray().toFloatArray());
        }
      }
      Batch batch = Batches.build(Batches.allKinds(5), 47);
      for (Precision precision : Precision.values()) {
        Pass pass = Pass.inference(precision);
        Assert.assertEquals(
            two.forward(two.manager, batch, pass).scores().toByteBuffer(),
            one.forward(one.manager, batch, pass).scores().toByteBuffer());
      }
    }
  }

  private static HostBatch sample(Net net) {
    return InputSamples.encode(
        HostInputs.allocate(
            net.network.format(), InputSamples.CAPACITY, ByteBuffer::allocateDirect));
  }

  @Test
  public void smallBidirectionalCoreUsesBothCandidatePathsAndMatchesV2WhenMuted() {
    checkSmallCore(Device.cpu());
  }

  @Test(groups = "rocm")
  public void gpuSmallBidirectionalCoreUsesBothCandidatePathsAndMatchesV2WhenMuted() {
    checkSmallCore(Device.gpu());
  }

  /** 比較時だけ追加経路の出力を消す。通常の S は両方の経路を学習可能な初期値で使う。 */
  private static void checkSmallCore(Device device) {
    try (Net control = new Net(ShapeArm.V2.shape(), 311, device);
        Net core = new Net(ModelShapes.bidirectional(256), 311, device)) {
      for (var shared : control.network.getParameters()) {
        Assert.assertEquals(
            core.network.getParameters().get(shared.getKey()).getArray().toByteBuffer(),
            shared.getValue().getArray().toByteBuffer(),
            shared.getKey());
      }
      Batches.perturb(control.network, 0.05f, 313);
      control
          .network
          .getParameters()
          .forEach(
              shared ->
                  core.network
                      .getParameters()
                      .get(shared.getKey())
                      .getArray()
                      .set(shared.getValue().getArray().toFloatArray()));
      NetworkInputs inputs =
          NetworkInputs.of(core.manager, PinnedUploads.on(core.manager), sample(core));
      NetworkInputs base =
          NetworkInputs.of(control.manager, PinnedUploads.on(control.manager), sample(control));
      backwardScores(core, inputs);
      for (var pair : core.network.getParameters()) {
        if (!pair.getKey().contains("candidateFeatures_")) continue;
        NDArray parameter = pair.getValue().getArray();
        Assert.assertTrue(parameter.abs().max().getFloat() > 0f, pair.getKey());
        Assert.assertTrue(parameter.getGradient().abs().max().getFloat() > 0f, pair.getKey());
        if (pair.getKey().contains("segments_") || pair.getKey().endsWith("count_output")) {
          core.fill(pair.getKey(), 0f);
        }
      }
      for (Precision precision : Precision.values()) {
        Pass pass = Pass.inference(precision);
        NDArray expected = control.network.forward(control.store, base, pass).scores();
        Assert.assertTrue(expected.abs().max().getFloat() > 0f);
        Assert.assertEquals(
            core.network.forward(core.store, inputs, pass).scores().toByteBuffer(),
            expected.toByteBuffer(),
            precision.name());
      }
    }
  }

  private static void backwardScores(Net net, NetworkInputs inputs) {
    boolean[] measured = new boolean[inputs.actions()];
    Arrays.fill(measured, true);
    try (GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
      var scores =
          net.network
              .forward(net.store, inputs, Pass.training(Precision.FLOAT32, measured, null))
              .scores();
      collector.backward(scores.square().sum());
    }
  }

  @Test
  public void causalEventStatesReadThePastAndIgnoreFutureAndFixedTokens() {
    checkCausalStates(Device.cpu(), ShapeArm.V4_S);
  }

  @Test(groups = "rocm")
  public void gpuCausalEventStatesReadThePastAndIgnoreFutureAndFixedTokens() {}

  private static void checkCausalStates(Device device, ShapeArm arm) {
    try (Net net = new Net(arm.shape(), 229, device)) {
      Batches.perturb(net.network, 0.05f, 233);
      HostBatch host = sample(net);
      NetworkInputs inputs = NetworkInputs.of(net.manager, PinnedUploads.on(net.manager), host);
      BatchIndex index = inputs.index(Pass.inference(Precision.FLOAT32));
      Context ctx = new Context(net.store, net.manager, Precision.FLOAT32, false);
      int fixed = inputs.format().fixedTokens();
      int row = 0;
      while (host.plan().events(row) < 3) row++;
      NDArray tokens = net.network.tokens(ctx, inputs, index);
      var layer = (EncoderLayer) net.network.getChildren().values().get(1);
      var mask = index.tokenMask.array();
      var codes = RelationClass.codes(inputs, index, index.tokens);
      NDArray original = layer.forward(ctx, tokens, mask, codes, index.tokens);
      int last = fixed + host.plan().events(row) - 1;
      NDArray perturbed = tokens.duplicate();
      perturbed.set(new NDIndex("{}, {}, 0", row, last), 9f);
      NDArray changed = layer.forward(ctx, perturbed, mask, codes, index.tokens);
      NDIndex prefix = new NDIndex("{}, {}:{}", row, fixed, last);
      Assert.assertEquals(changed.get(prefix).toByteBuffer(), original.get(prefix).toByteBuffer());
      NDIndex round = new NDIndex("{}, 0", row);
      Assert.assertTrue(changed.get(round).sub(original.get(round)).abs().max().getFloat() > 0f);
      perturbed.set(new NDIndex("{}, :{}, 1", row, fixed), 7f);
      changed = layer.forward(ctx, perturbed, mask, codes, index.tokens);
      Assert.assertEquals(changed.get(prefix).toByteBuffer(), original.get(prefix).toByteBuffer());
      perturbed.set(new NDIndex("{}, {}, 2", row, fixed), 8f);
      changed = layer.forward(ctx, perturbed, mask, codes, index.tokens);
      NDIndex second = new NDIndex("{}, {}", row, fixed + 1);
      Assert.assertTrue(changed.get(second).sub(original.get(second)).abs().max().getFloat() > 0f);
      backwardScores(net, inputs);
      Assert.assertTrue(net.parameter("encoder0_qkv").getGradient().abs().max().getFloat() > 0f);
    }
  }

  @DataProvider
  public Object[][] eventUpdates() {
    return new Object[][] {{ShapeArm.V2, 3f}};
  }

  @Test(dataProvider = "eventUpdates")
  public void firstLayerAppliesTheFfnToTheSelectedTokens(ShapeArm arm, float eventUpdate) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismShape shape = arm.shape();
      EncoderLayer layer = new EncoderLayer(shape, 0, FOUR_PLAYER);
      layer.initialize(manager, DataType.FLOAT32);
      // 射影の重みを 0 にすると、Attention と FFN はそれぞれの出力バイアスだけを残差に足す。
      for (String name : List.of("output", "down")) {
        NDArray weight = layer.getParameters().get(name).getArray();
        weight.set(new float[Math.toIntExact(weight.size())]);
      }
      float[] bias = new float[shape.hidden()];
      Arrays.fill(bias, 1f);
      layer.getParameters().get("outputBias").getArray().set(bias);
      Arrays.fill(bias, 2f);
      layer.getParameters().get("downBias").getArray().set(bias);
      NetworkInputs inputs = Batches.build(List.of(Batches.turn(3)), 59).inputs(manager);
      BatchIndex index = inputs.index(Pass.inference(Precision.FLOAT32));
      NDArray codes = RelationClass.codes(inputs, index, index.tokens);
      int fixed = inputs.format().fixedTokens();
      for (Precision precision : Precision.values()) {
        NDArray tokens =
            manager.ones(new Shape(1, index.tokens, shape.hidden()), precision.compute());
        Context context =
            new Context(new ParameterStore(manager, false), manager, precision, false);
        NDArray result =
            layer.forward(context, tokens, index.tokenMask.array(), codes, index.tokens);
        Assert.assertEquals(result.getShape(), tokens.getShape());
        Assert.assertEquals(result.getDataType(), precision.compute());
        Assert.assertTrue(result.get(new NDIndex(":, :{}", fixed)).eq(4f).all().getBoolean());
        Assert.assertTrue(
            result.get(new NDIndex(":, {}:", fixed)).eq(1f + eventUpdate).all().getBoolean());
      }
    }
  }

  @Test(dataProvider = "eagerArms", dataProviderClass = ModelShapes.class)
  public void legalScoresAreIndependentOfBatchPadding(PrismShape shape) {
    List<long[]> rows = HipFixtures.mixedRows(2, 900);
    HostBatch mixed = HipFixtures.encode(rows, shape);
    Assert.assertNotEquals(mixed.plan().events(0), mixed.plan().events(1));
    try (Net net = new Net(shape, 61)) {
      Batches.perturb(net.network, 0.05f, 67);
      NDArray together = net.scores(mixed);
      for (int r = 0; r < rows.size(); r++) {
        HostBatch single = HipFixtures.encode(List.of(rows.get(r)), shape);
        float[] alone = net.scores(single).get(0).toFloatArray();
        float[] padded = together.get(r).toFloatArray();
        Assert.assertTrue(Batches.maxAbs(alone) > 1e-4f);
        for (int a = 0; a < single.plan().actions(0); a++) {
          Assert.assertEquals(padded[a], alone[a], 2e-5f, "row " + r + " action " + a);
        }
      }
    }
  }

  @Test
  public void anchorGradientsFollowTheMeasuredSet() {
    // 手番の行: 0 ダマ x、1 y、2 z、3 立直 x、4 暗槓、5 ツモ、6 九種、7 y と同値な打牌。遷移は手 0〜6 に 1 つずつ。
    // ゲートは 0 立直、1 暗槓、2 ツモ、3 九種。Δ_立直 < 0 なので M の最大はダマ x（d*）。
    float[] discardValues = {0.30f, 0.10f, -0.20f, 0f, 0f, 0f, 0f};
    float[] gateValues = {-0.10f, -0.05f, 0.20f, -0.30f};
    float nan = Float.NaN;
    // C = {ダマ x, 立直 x}: B(x) の勾配は g_d + g_r = 0、Δ_立直 は g_r
    Anchor both =
        anchor(discardValues, gateValues, new float[] {0.4f, nan, nan, 0.1f, nan, nan, nan, nan});
    Assert.assertEquals(both.discard[0], 0f, 1e-6f);
    Assert.assertEquals(both.gate[0], both.score[3], 1e-6f);
    Assert.assertTrue(Math.abs(both.score[3]) > 1e-3f);
    // C = {立直 x, y}: ダマを測っていないので B(x) は 0（sg）、Δ_立直 は g_r、B(y) は g_y
    Anchor noDama =
        anchor(discardValues, gateValues, new float[] {nan, 0.2f, nan, 0.3f, nan, nan, nan, nan});
    Assert.assertEquals(noDama.discard[0], 0f, 0f);
    Assert.assertEquals(noDama.gate[0], noDama.score[3], 1e-6f);
    Assert.assertEquals(noDama.discard[1], noDama.score[1], 1e-6f);
    // C = {ダマ x, 立直 x, y}: B(x) は g_d + g_r = −g_y
    Anchor three =
        anchor(discardValues, gateValues, new float[] {0.4f, 0.2f, nan, 0.1f, nan, nan, nan, nan});
    Assert.assertEquals(three.discard[0], three.score[0] + three.score[3], 1e-6f);
    Assert.assertEquals(three.discard[0], -three.score[1], 1e-6f);
    Assert.assertTrue(Math.abs(three.discard[0]) > 1e-3f);
    // C = {d*, ツモ}: B(d*) は g_d + g_t = 0、Δ_ツモ は g_t
    Anchor tsumo =
        anchor(discardValues, gateValues, new float[] {0.4f, nan, nan, nan, nan, 0.1f, nan, nan});
    Assert.assertEquals(tsumo.discard[0], 0f, 1e-6f);
    Assert.assertEquals(tsumo.gate[2], tsumo.score[5], 1e-6f);
    Assert.assertTrue(Math.abs(tsumo.score[5]) > 1e-3f);
  }

  /**
   * 同値な打牌は、組のどちらの手を測っても同じ勾配になる。測った印がツモ切りの側（番号の大きい手）だけにあっても、M と立直のアンカーは
   * 組が C にあるものとして勾配を流す。
   */
  @Test
  public void anchorGradientsTreatTwinsAsOneAction() {
    float nan = Float.NaN;
    // Batches.turn の行: 最大の打牌は y（手 1。手 7 が同値）。C = {y, ツモ}: B(y) は g_y + g_t = 0、Δ_ツモ は g_t
    float[] discardValues = {0.10f, 0.30f, -0.20f, 0f, 0f, 0f, 0f};
    float[] gateValues = {-0.10f, -0.05f, 0.20f, -0.30f};
    Anchor fromHand =
        anchor(discardValues, gateValues, new float[] {nan, 0.4f, nan, nan, nan, 0.1f, nan, nan});
    Anchor fromDraw =
        anchor(discardValues, gateValues, new float[] {nan, nan, nan, nan, nan, 0.1f, nan, 0.4f});
    for (Anchor twin : List.of(fromHand, fromDraw)) {
      Assert.assertEquals(twin.discard[1], 0f, 1e-6f);
      Assert.assertEquals(twin.gate[2], twin.score[5], 1e-6f);
      Assert.assertTrue(Math.abs(twin.score[5]) > 1e-3f);
    }
    Assert.assertEquals(fromDraw.score[5], fromHand.score[5], 1e-6f);

    // 手番の行: 0 ダマ x、1 その同値な打牌、2 y、3 立直 x（基準は 0）、4 その同値な立直（基準は 1）、5 ツモ。遷移は手 0・2・3・5 に 1 つずつ。
    // ゲートは 0・1 立直、2 ツモ。Δ_立直 > 0 なので M の最大は立直 x。
    // C = {ダマ x, 立直 x, ツモ}: B(x) は g_d + g_r + g_t = 0、Δ_立直 は 2 つのゲートを合わせて g_r + g_t、Δ_ツモ は g_t
    Batch riichi =
        Batches.build(
            List.of(
                new Batches.Row(
                    EngineDecisionKind.TURN,
                    3,
                    List.of(
                        new Batches.Move(Action.Type.DAHAI, 1, -1, 0),
                        new Batches.Move(Action.Type.DAHAI, 0, -1, 0),
                        new Batches.Move(Action.Type.DAHAI, 1, -1, 2),
                        new Batches.Move(Action.Type.RIICHI_DAHAI, 1, 0, 3),
                        new Batches.Move(Action.Type.RIICHI_DAHAI, 0, 1, 3),
                        new Batches.Move(Action.Type.TSUMO_AGARI, 1, -1, 5)))),
            5);
    float[] riichiDiscards = {0.30f, 0.10f, 0f, 0f};
    float[] riichiGates = {0.15f, 0.15f, 0.20f};
    Anchor riichiFromHand =
        anchor(riichi, riichiDiscards, riichiGates, new float[] {0.4f, nan, nan, 0.1f, nan, 0.2f});
    Anchor riichiFromDraw =
        anchor(riichi, riichiDiscards, riichiGates, new float[] {nan, 0.4f, nan, nan, 0.1f, 0.2f});
    Assert.assertEquals(
        riichiFromHand.gate[0], riichiFromHand.score[3] + riichiFromHand.score[5], 1e-6f);
    Assert.assertEquals(riichiFromDraw.gate[0], riichiFromDraw.score[5], 1e-6f);
    Assert.assertEquals(riichiFromDraw.gate[1], riichiFromDraw.score[4], 1e-6f);
    for (Anchor twin : List.of(riichiFromHand, riichiFromDraw)) {
      Assert.assertEquals(twin.discard[0], 0f, 1e-6f);
      Assert.assertEquals(twin.gate[2], twin.score[5], 1e-6f);
      Assert.assertTrue(Math.abs(twin.score[5]) > 1e-3f);
    }
    Assert.assertEquals(
        riichiFromDraw.gate[0] + riichiFromDraw.gate[1],
        riichiFromHand.gate[0] + riichiFromHand.gate[1],
        1e-6f);
  }

  /**
   * 行の平均のアンカー。差分整合は行の定数に依らず、アンカーは行の定数だけを 0 に引き止める。アンカーは測った打牌 D だけで取り、勾配は
   * 行の D の手で全て同じ（D の外、応答の行、パディングは 0）。D の上の s の平均が 0 ならアンカーは 0 で、そこから D のある行に定数 c を
   * 足すとアンカーは w·c² になる。
   */
  @Test
  public void rowMeanPinsOnlyTheRowConstant() {
    Batch batch = Batches.build(List.of(Batches.turn(3), Batches.response(2)), 5);
    List<ValueLabel> labels = Batches.values(batch, 23);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      NetworkInputs inputs = batch.inputs(manager);
      BatchIndex index =
          BatchIndex.send(
              inputs, Pass.training(Precision.FLOAT32, ValueLabel.measured(labels), null));
      int rows = labels.size();
      int capacity = 0;
      for (ValueLabel label : labels) {
        capacity = Math.max(capacity, label.values().length);
      }
      Random random = new Random(29);
      float[] exact = new float[rows * capacity];
      float[] shifted = new float[rows * capacity];
      float[] noisy = new float[rows * capacity];
      float[] shifts = {0.7f, -1.3f};
      for (int row = 0; row < rows; row++) {
        float[] values = labels.get(row).values();
        // 測った打牌の上の平均を 0 にした s（ほかの手は乱数のまま）
        double sum = 0;
        int count = 0;
        for (int a = 0; a < values.length; a++) {
          if (!Float.isNaN(values[a]) && index.discardValueSlots[row * capacity + a]) {
            sum += values[a];
            count++;
          }
        }
        float mean = count == 0 ? 0f : (float) (sum / count);
        for (int a = 0; a < values.length; a++) {
          float base = Float.isNaN(values[a]) ? (float) random.nextGaussian() : values[a] - mean;
          exact[row * capacity + a] = base;
          shifted[row * capacity + a] = base + shifts[row];
          noisy[row * capacity + a] = base + (float) random.nextGaussian() * 0.1f;
        }
      }
      Shape shape = new Shape(rows, capacity);
      float atTarget =
          loss.rowMean(manager, outputs(manager.create(exact, shape), index), labels).getFloat();
      Assert.assertEquals(atTarget, 0f, 1e-10f);
      float pinned =
          loss.rowMean(manager, outputs(manager.create(shifted, shape), index), labels).getFloat();
      double expected = 0;
      for (int row = 0; row < rows; row++) {
        if (hasMeasuredDiscard(labels.get(row), index, row, capacity)) {
          expected += labels.get(row).weight() * shifts[row] * shifts[row];
        }
      }
      Assert.assertTrue(expected > 0, "the turn row must have a measured discard");
      Assert.assertEquals(pinned, (float) expected, 1e-5f);
      float valueExact =
          loss.value(manager, outputs(manager.create(exact, shape), index), labels).getFloat();
      float valueShifted =
          loss.value(manager, outputs(manager.create(shifted, shape), index), labels).getFloat();
      Assert.assertEquals(valueShifted, valueExact, 1e-5f);
      NDArray leaf = manager.create(noisy, shape);
      leaf.setRequiresGradient(true);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        collector.backward(loss.rowMean(manager, outputs(leaf, index), labels));
      }
      float[] gradient = leaf.getGradient().toFloatArray();
      for (int row = 0; row < rows; row++) {
        float[] values = labels.get(row).values();
        Float common = null;
        for (int a = 0; a < capacity; a++) {
          float g = gradient[row * capacity + a];
          if (a >= values.length
              || Float.isNaN(values[a])
              || !index.discardValueSlots[row * capacity + a]) {
            Assert.assertEquals(g, 0f, 0f, "row " + row + " action " + a + " is outside D");
          } else if (common == null) {
            common = g;
          } else {
            Assert.assertEquals(g, common, 1e-6f, "row " + row + " action " + a);
          }
        }
      }
    }
  }

  /** 行に測った打牌があるか。 */
  private static boolean hasMeasuredDiscard(
      ValueLabel label, BatchIndex index, int row, int capacity) {
    float[] values = label.values();
    for (int a = 0; a < values.length; a++) {
      if (!Float.isNaN(values[a]) && index.discardValueSlots[row * capacity + a]) {
        return true;
      }
    }
    return false;
  }

  /** 葉の s をネットワークの出力の形にする（行の手の数と合法手はバッチのインデックスから取る）。 */
  private static Outputs outputs(NDArray scores, BatchIndex index) {
    return new Outputs(
        scores, index.legal.array(), index.rowActions, index.discardValueSlots, null);
  }

  /** アンカーのテストの結果: B・Δ の勾配と、同じ s を葉にしたときの s の勾配 g。 */
  private record Anchor(float[] discard, float[] gate, float[] score) {}

  private static Anchor anchor(float[] discardValues, float[] gateValues, float[] targets) {
    return anchor(Batches.build(List.of(Batches.turn(3)), 5), discardValues, gateValues, targets);
  }

  private static Anchor anchor(
      Batch batch, float[] discardValues, float[] gateValues, float[] targets) {
    List<ValueLabel> labels = List.of(new ValueLabel(targets, 0.1f, 1f));
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      NetworkInputs inputs = batch.inputs(manager);
      BatchIndex index =
          BatchIndex.send(
              inputs, Pass.training(Precision.FLOAT32, ValueLabel.measured(labels), null));
      NDArray discard = manager.create(discardValues);
      NDArray gate = manager.create(gateValues);
      discard.setRequiresGradient(true);
      gate.setRequiresGradient(true);
      float[] composed;
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray scores = ValueGraphHead.compose(manager, index, discard, gate);
        composed = scores.toFloatArray();
        collector.backward(
            loss.value(
                manager,
                new Outputs(
                    scores, index.legal.array(), index.rowActions, index.discardValueSlots, null),
                labels));
      }
      NDArray leaf = manager.create(composed, new Shape(1, composed.length));
      leaf.setRequiresGradient(true);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        collector.backward(
            loss.value(
                manager,
                new Outputs(
                    leaf, index.legal.array(), index.rowActions, index.discardValueSlots, null),
                labels));
      }
      return new Anchor(
          discard.getGradient().toFloatArray(),
          gate.getGradient().toFloatArray(),
          leaf.getGradient().toFloatArray());
    }
  }

  @Test(dataProvider = "smallShapes", dataProviderClass = ModelShapes.class)
  public void fitsRandomValues(PrismShape shape) {
    try (Net net = new Net(shape, 3);
        NDManager scope = net.manager.newSubManager()) {
      Batch batch = Batches.build(Batches.allKinds(4), 13);
      List<ValueLabel> labels = new ArrayList<>(Batches.values(batch, 17));
      // 同値な手は同じ s になるので、教師にも同じ Q を置く
      float[] turn = labels.get(0).values();
      turn[7] = turn[1];
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      Optimizer optimizer = Batches.adam(3e-3f);
      for (int step = 0; step < 400; step++) {
        try (NDManager stepScope = scope.newSubManager();
            GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
          Outputs outputs =
              net.forward(stepScope, batch, Pass.training(Precision.FLOAT32, measured, null));
          collector.backward(loss.total(stepScope, outputs, labels, null, 1f, labels.size(), 1f));
        }
        Batches.step(net.network, optimizer);
      }
      float[] scores =
          net.forward(scope, batch, Pass.inference(Precision.FLOAT32)).scores().toFloatArray();
      int capacity = scores.length / labels.size();
      float worst = 0;
      for (int row = 0; row < labels.size(); row++) {
        float[] target = labels.get(row).values();
        double mean = 0;
        for (int a = 0; a < target.length; a++) {
          mean += scores[row * capacity + a] - target[a];
        }
        mean /= target.length;
        for (int a = 0; a < target.length; a++) {
          worst = Math.max(worst, (float) Math.abs(scores[row * capacity + a] - target[a] - mean));
        }
      }
      log.info(
          "PRISM-T4 shape={} fit centered residual max={}", shape, String.format("%.4f", worst));
      Assert.assertTrue(worst < 0.05f, "centered residual " + worst);
    }
  }

  @Test(dataProvider = "smallShapes", dataProviderClass = ModelShapes.class)
  public void everyParameterReceivesGradientWithinTwoSteps(PrismShape shape) {
    try (Net net = new Net(shape, 5);
        NDManager scope = net.manager.newSubManager()) {
      Batch batch = Batches.build(Batches.allKinds(6), 19);
      List<ValueLabel> labels = Batches.values(batch, 23);
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 29);
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      Optimizer optimizer = Batches.adam(1e-3f);
      Set<String> reached = new HashSet<>();
      for (int step = 0; step < 2; step++) {
        try (NDManager stepScope = scope.newSubManager();
            GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
          Outputs outputs =
              net.forward(stepScope, batch, Pass.training(Precision.FLOAT32, measured, auxiliary));
          collector.backward(
              loss.total(stepScope, outputs, labels, auxiliary, 1f, labels.size(), 1f));
        }
        for (Pair<String, Parameter> pair : net.network.getParameters()) {
          try (NDArray gradient = pair.getValue().getArray().getGradient()) {
            float[] values = gradient.toFloatArray();
            if (Batches.maxAbs(values) > 0) {
              reached.add(pair.getKey());
            }
          }
        }
        Batches.step(net.network, optimizer);
      }
      List<String> missing = new ArrayList<>();
      for (Pair<String, Parameter> pair : net.network.getParameters()) {
        if (!reached.contains(pair.getKey())) {
          missing.add(pair.getKey());
        }
      }
      Assert.assertTrue(missing.isEmpty(), "no gradient: " + missing);
    }
  }

  @Test(dataProvider = "smallShapes", dataProviderClass = ModelShapes.class)
  public void auxiliaryHeadsDoNotChangeScores(PrismShape shape) {
    try (Net net = new Net(shape, 7);
        NDManager scope = net.manager.newSubManager()) {
      Batches.perturb(net.network, 0.05f, 31);
      Batch batch = Batches.build(Batches.allKinds(5), 37);
      List<ValueLabel> labels = Batches.values(batch, 41);
      boolean[] measured = ValueLabel.measured(labels);
      float[] plain =
          net.forward(scope, batch, Pass.inference(Precision.FLOAT32)).scores().toFloatArray();
      float[] trained =
          net.forward(scope, batch, Pass.training(Precision.FLOAT32, measured, null))
              .scores()
              .toFloatArray();
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 43);
      Outputs withHeads =
          net.forward(scope, batch, Pass.training(Precision.FLOAT32, measured, auxiliary));
      Assert.assertNotNull(withHeads.auxiliary());
      Assert.assertEquals(withHeads.scores().getShape(), new Shape(batch.rows().size(), 8));
      Assert.assertEquals(
          withHeads.auxiliary().reading().getShape(),
          new Shape(
              auxiliary.readingRows().length,
              auxiliary.heads().opponents(),
              auxiliary.heads().readingWidth()));
      Assert.assertEquals(
          withHeads.auxiliary().candidates().getShape(),
          new Shape(auxiliary.candidateActions().length, auxiliary.heads().candidateWidth()));
      Assert.assertTrue(Batches.maxAbs(plain) > 1e-4f, "scores must be non-trivial");
      Assert.assertTrue(Arrays.equals(plain, trained));
      Assert.assertTrue(Arrays.equals(plain, withHeads.scores().toFloatArray()));
    }
  }

  @Test(dataProvider = "smallShapes", dataProviderClass = ModelShapes.class)
  public void auxiliaryAndValueGradientsAreSeparated(PrismShape shape) {
    try (Net net = new Net(shape, 9);
        NDManager scope = net.manager.newSubManager()) {
      Batches.perturb(net.network, 0.05f, 47);
      Batch batch = Batches.build(Batches.allKinds(5), 53);
      List<ValueLabel> labels = Batches.values(batch, 59);
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 61);
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());

      try (NDManager stepScope = scope.newSubManager();
          GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
        Outputs outputs =
            net.forward(stepScope, batch, Pass.training(Precision.FLOAT32, measured, auxiliary));
        collector.backward(
            loss.auxiliary().terms(stepScope, outputs.auxiliary(), auxiliary, 1f).total());
      }
      Map<String, float[]> fromAuxiliary = Batches.takeGradients(net.network);
      try (NDManager stepScope = scope.newSubManager();
          GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
        Outputs outputs =
            net.forward(stepScope, batch, Pass.training(Precision.FLOAT32, measured, auxiliary));
        collector.backward(loss.value(stepScope, outputs, labels));
      }
      Map<String, float[]> fromValue = Batches.takeGradients(net.network);

      for (Map.Entry<String, float[]> entry : fromAuxiliary.entrySet()) {
        String name = entry.getKey();
        float auxiliaryMax = Batches.maxAbs(entry.getValue());
        float valueMax = Batches.maxAbs(fromValue.get(name));
        if (name.contains("discardValue_") || name.contains("gates_")) {
          Assert.assertEquals(auxiliaryMax, 0f, name + " from the auxiliary loss");
          Assert.assertTrue(valueMax > 0, name + " from the value loss");
        } else if (name.contains("reading_") || name.contains("outcome_")) {
          Assert.assertEquals(valueMax, 0f, name + " from the value loss");
          Assert.assertTrue(auxiliaryMax > 0, name + " from the auxiliary loss");
        }
      }
      Assert.assertTrue(
          Batches.maxAbs(fromAuxiliary.get(firstOf(fromAuxiliary, "encoder0_qkv"))) > 0);
    }
  }

  @Test(dataProvider = "smallShapes", dataProviderClass = ModelShapes.class)
  public void emptyAndLongestRowsStayFinite(PrismShape shape) {
    try (Net net = new Net(shape, 11);
        NDManager scope = net.manager.newSubManager()) {
      Batches.perturb(net.network, 0.05f, 67);
      List<List<Batches.Row>> shapes =
          List.of(
              List.of(
                  Batches.turn(0),
                  Batches.turn(Limits.MAX_EVENT_TOKENS),
                  Batches.response(0),
                  Batches.postCall(Limits.MAX_EVENT_TOKENS)),
              List.of(Batches.turn(0), Batches.chankan(0)));
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      for (List<Batches.Row> rows : shapes) {
        Batch batch = Batches.build(rows, 71);
        List<ValueLabel> labels = Batches.values(batch, 73);
        AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 79);
        try (NDManager stepScope = scope.newSubManager();
            GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
          Outputs outputs =
              net.forward(
                  stepScope,
                  batch,
                  Pass.training(Precision.FLOAT32, ValueLabel.measured(labels), auxiliary));
          assertFinite("scores", outputs.scores().toFloatArray());
          assertFinite("candidates", outputs.auxiliary().candidates().toFloatArray());
          assertFinite("reading", outputs.auxiliary().reading().toFloatArray());
          NDArray total = loss.total(stepScope, outputs, labels, auxiliary, 1f, labels.size(), 1f);
          Assert.assertTrue(Float.isFinite(total.getFloat()));
          collector.backward(total);
        }
        for (Map.Entry<String, float[]> entry : Batches.takeGradients(net.network).entrySet()) {
          assertFinite(entry.getKey(), entry.getValue());
        }
      }
    }
  }

  private static String firstOf(Map<String, float[]> gradients, String suffix) {
    for (String name : gradients.keySet()) {
      if (name.endsWith(suffix)) {
        return name;
      }
    }
    throw new AssertionError("no parameter " + suffix);
  }

  private static void assertFinite(String what, float[] values) {
    for (float value : values) {
      Assert.assertTrue(Float.isFinite(value), what + " is not finite");
    }
  }
}
