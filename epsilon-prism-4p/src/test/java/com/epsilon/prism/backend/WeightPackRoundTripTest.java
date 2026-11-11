package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism4p.input.InputSchema;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * WeightPack の中身を読み戻して確かめる（CPU、native なし。{@link PackReadBack}）。全ての形で、推論に使う全てのパラメーターが、表か
 * 埋め込みのどちらかにちょうど 1 回置かれる。置いた値は、変換（BF16・BF16 に丸めた FP32・FP32）のとおりにビットまで読み戻せる。記述にある
 * 場所は全て埋まり、場所どうしは重ならない。四人打ちの束では、形式から作る牌の関係の表が {@link MahjongTileRelation#relationIds()} と一致し、
 * 束全体のバイトも変わらない。公開した v760 は、今のコードのまま v2 の深さ 2 の記述として読め、同じように読み戻せる（-Dprism.snapshot に
 * snapshot のディレクトリを渡す。無ければ飛ばす）。
 */
public class WeightPackRoundTripTest {
  private static final Logger log = LoggerFactory.getLogger(WeightPackRoundTripTest.class);

  @DataProvider
  public Object[][] shapes() {
    return new Object[][] {
      {PrismShape.standard(2)},
      {PrismShape.standard(3)},
      {ShapeArm.V3.shape(ShapeArm.V3_DEPTH)},
      {ShapeArm.V3_ONE_STAGE.shape(ShapeArm.V3_DEPTH)},
      {ShapeArm.V3_CONTEXT_512.shape(ShapeArm.V3_DEPTH)},
    };
  }

  @Test(dataProvider = "shapes")
  public void everyPlacedValueReadsBack(PrismShape shape) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = perturbed(manager, shape);
      WeightLayout layout = WeightLayout.of(network);
      WeightPack pack = WeightPack.build(layout, network, 11);
      Assert.assertEquals(pack.version(), 11);
      PackReadBack.assertReadsBack(layout, pack, network.inferenceParameters());
    }
  }

  /**
   * 四人打ちの束のバイトが変わらないことを確かめる。形式・記述・重みを別々に渡して束を作り、その牌の関係の表を
   * {@link MahjongTileRelation#relationIds()}（形式から作る前の表）で書き直す。それが、ネットワークから作った束と全てのバイトで一致する。
   * 同じ束を 2 度作っても一致する。
   */
  @Test(dataProvider = "shapes")
  public void fourPlayerPackKeepsItsBytes(PrismShape shape) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = perturbed(manager, shape);
      WeightLayout layout = WeightLayout.of(network);
      ByteBuffer built = WeightPack.build(layout, network, 11).buffer();
      Assert.assertEquals(WeightPack.build(WeightLayout.of(network), network, 11).buffer(), built, "a second build");

      WeightLayout given =
          WeightLayout.of(InputSchema.FORMAT, shape.architecture(InputSchema.FORMAT), network.inferenceParameters());
      Assert.assertEquals(given.architecture(), layout.architecture());
      Assert.assertEquals(given.bytes(), layout.bytes());
      ByteBuffer expected = copy(WeightPack.build(given, network, 11).buffer());
      long[] relations = MahjongTileRelation.relationIds();
      int tiles = Math.toIntExact(given.header().weight(HipWeight.TILE_RELATION, 0));
      Assert.assertEquals(given.header().elements(HipWeight.TILE_RELATION, 0), relations.length);
      for (int i = 0; i < relations.length; i++) {
        expected.put(tiles + i, (byte) relations[i]);
      }
      Assert.assertEquals(built, expected, "the four-player pack with the MahjongTileRelation table");
    }
  }

  /** 公開した v760 の model.json が、今の PrismShape の標準の形（v2 の深さ 2）として読め、束の記述が v2 の記述と一致することを確かめる。 */
  @Test
  public void publishedV760ReadsAsTheV2Description() throws IOException {
    String given = System.getProperty("prism.snapshot");
    if (given == null) {
      throw new SkipException("needs -Dprism.snapshot (the v760 snapshot directory)");
    }
    Path model = Snapshot.model(Path.of(given));
    Assert.assertEquals(PrismNetwork.readShape(model), PrismShape.standard());
    WeightPack pack = WeightPack.load(model, 760);
    WeightLayout layout = pack.layout();
    Assert.assertEquals(layout.architecture(), PrismShape.standard().architecture(InputSchema.FORMAT));
    Assert.assertEquals(new PackView(pack.buffer()).architecture(), layout.architecture().encode());
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = PrismNetwork.load(model, manager);
      PackReadBack.assertReadsBack(layout, pack, network.inferenceParameters());
    }
    log.info("PRISM-HIP-V760 {} reads as {} ({} bytes)", given, layout.architecture().layers(), layout.bytes());
  }

  /**
   * 乱数の種を固定した四人打ちのネットワークを作り、全てのパラメーターに乱数を足す。初期化したままの 0 のバイアスや 1 のゲインも、
   * 読み戻しで区別できるようにするため。
   */
  private static PrismNetwork perturbed(NDManager manager, PrismShape shape) {
    manager.getEngine().setRandomSeed(23);
    PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
    network.initialize(manager);
    Random random = new Random(29);
    for (Pair<String, Parameter> pair : network.getParameters()) {
      NDArray array = pair.getValue().getArray();
      float[] values = array.toFloatArray();
      for (int i = 0; i < values.length; i++) {
        values[i] += (float) (random.nextGaussian() * 0.05);
      }
      array.set(values);
    }
    return network;
  }

  private static ByteBuffer copy(ByteBuffer source) {
    ByteBuffer copy = ByteBuffer.allocate(source.remaining()).order(source.order());
    copy.put(source.duplicate());
    return copy.flip();
  }
}
