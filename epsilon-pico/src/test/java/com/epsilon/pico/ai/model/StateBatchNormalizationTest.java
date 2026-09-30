package com.epsilon.pico.ai.model;

import ai.djl.Device;
import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class StateBatchNormalizationTest {
  @DataProvider
  public Object[][] batchShapes() {
    return new Object[][] {{2, 128}, {3, 128}, {2, 256}, {3, 256}};
  }

  /** バッチの配置とプレイヤーごとの連結方法を変えても、要素の順序と入力・共有 LayerNorm の勾配が保たれることを検証する。 */
  @Test(groups = "native", dataProvider = "batchShapes")
  public void batchedMemoryMatchesIndependentRowsAndTheirGradients(int rows, int width) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      var encoder = initializedEncoder(manager, width);
      var store = new ParameterStore(manager, true);
      NDArray categories = categoriesWithDifferentMasks(manager, rows);
      NDArray numerics = numericInput(manager, rows);
      numerics.setRequiresGradient(true);
      LayerNorm normalization = entityNormalization(encoder);
      NDArray gamma = normalization.getDirectParameters().get("gamma").getArray();
      NDArray beta = normalization.getDirectParameters().get("beta").getArray();
      NDArray tileWeights = weights(manager, new Shape(rows, 34, width));
      NDArray roundWeights = weights(manager, new Shape(rows, width));
      NDArray entityWeights =
          weights(manager, new Shape(rows, EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT, width));
      EpsilonMahjongStateEncoder.EncodedMemory batched;
      try (var collector = manager.getEngine().newGradientCollector()) {
        batched = encoder.encodeMemory(store, categories, numerics, true, new PairList<>());
        collector.backward(
            batched
                .tileEmbeddings()
                .mul(tileWeights)
                .sum()
                .add(batched.roundEmbedding().mul(roundWeights).sum())
                .add(batched.entityEmbeddings().mul(entityWeights).sum()));
      }
      float[] inputGradient = numerics.getGradient().toFloatArray();
      float[] gammaGradient = gamma.getGradient().toFloatArray();
      float[] betaGradient = beta.getGradient().toFloatArray();
      assertNonzero(inputGradient);
      assertNonzero(gammaGradient);
      assertNonzero(betaGradient);
      numerics.getGradient().muli(0);
      gamma.getGradient().muli(0);
      beta.getGradient().muli(0);

      NDList rounds = new NDList();
      NDList tiles = new NDList();
      NDList projections = new NDList();
      NDList entities = new NDList();
      try (var collector = manager.getEngine().newGradientCollector()) {
        NDArray loss = manager.zeros(new Shape());
        for (int row = 0; row < rows; row++) {
          var single =
              encoder.encodeMemory(
                  store,
                  categories.get("{}:{},:", row, row + 1),
                  numerics.get("{}:{},:", row, row + 1),
                  true,
                  new PairList<>());
          rounds.add(single.roundEmbedding());
          tiles.add(single.tileEmbeddings());
          projections.add(single.tileProjectionEmbeddings());
          NDArray expectedEntities = referenceEntities(single, width);
          entities.add(expectedEntities);
          Assert.assertEquals(
              single.entityEmbeddings().toFloatArray(), expectedEntities.toFloatArray());
          loss =
              loss.add(
                      single.tileEmbeddings().mul(tileWeights.get("{}:{},:,:", row, row + 1)).sum())
                  .add(single.roundEmbedding().mul(roundWeights.get("{}:{},:", row, row + 1)).sum())
                  .add(expectedEntities.mul(entityWeights.get("{}:{},:,:", row, row + 1)).sum());
        }
        collector.backward(loss);
      }
      Assert.assertEquals(
          batched.roundEmbedding().toFloatArray(),
          NDArrays.concat(rounds, 0).toFloatArray(),
          0.0002f);
      Assert.assertEquals(
          batched.tileEmbeddings().toFloatArray(),
          NDArrays.concat(tiles, 0).toFloatArray(),
          0.0002f);
      Assert.assertEquals(
          batched.tileProjectionEmbeddings().toFloatArray(),
          NDArrays.concat(projections, 0).toFloatArray(),
          0.0002f);
      Assert.assertEquals(
          batched.entityEmbeddings().toFloatArray(),
          NDArrays.concat(entities, 0).toFloatArray(),
          0.0002f);
      Assert.assertEquals(inputGradient, numerics.getGradient().toFloatArray(), 0.001f);
      Assert.assertEquals(gammaGradient, gamma.getGradient().toFloatArray(), 0.001f);
      Assert.assertEquals(betaGradient, beta.getGradient().toFloatArray(), 0.001f);
    }
  }

  /** GPU の共有正規化処理が FP32 の結果と、その結果を BF16 に丸めて射影した牌表現を返すことを検証する。 */
  @Test(groups = "rocm", dataProvider = "batchShapes")
  public void gpuTileProjectionIsTheBfloat16ViewOfNormalizedTiles(int rows, int width) {
    try (NDManager manager = NDManager.newBaseManager(Device.gpu())) {
      var encoder = initializedEncoder(manager, width);
      var store = new ParameterStore(manager, false);
      NDArray categories =
          manager.ones(new Shape(rows, DecisionInputSchema.STATE_INT_COUNT), DataType.INT32);
      NDArray numerics = numericInput(manager, rows).toType(DataType.BFLOAT16, false);
      try (Autocast ignored =
          manager.getEngine().newAutocast(Device.gpu(), DataType.BFLOAT16, true)) {
        var actual = encoder.encodeMemory(store, categories, numerics, false, new PairList<>());
        Assert.assertEquals(
            actual.entityEmbeddings().toFloatArray(),
            referenceEntities(actual, width).toFloatArray());
        Assert.assertEquals(actual.tileEmbeddings().getDataType(), DataType.FLOAT32);
        Assert.assertEquals(actual.tileProjectionEmbeddings().getDataType(), DataType.BFLOAT16);
        Assert.assertEquals(
            actual.tileProjectionEmbeddings().toType(DataType.FLOAT32, false).toFloatArray(),
            actual
                .tileEmbeddings()
                .toType(DataType.BFLOAT16, false)
                .toType(DataType.FLOAT32, false)
                .toFloatArray());
      }
    }
  }

  /** 旧配置の形状変更と連結を参照式にして、牌・家・河・面子の対応と勾配を比較する。 */
  private static NDArray referenceEntities(
      EpsilonMahjongStateEncoder.EncodedMemory memory, int width) {
    long rows = memory.roundEmbedding().getShape().get(0);
    int riverEnd = 1 + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER;
    int meldEnd = riverEnd + DecisionInputSchema.MAX_MELDS_PER_PLAYER;
    return NDArrays.concat(
        new NDList(
            memory.roundEmbedding().reshape(rows, 1, width),
            memory.playerMemory().get(":,:,0,:"),
            memory.tileEmbeddings(),
            memory.playerMemory().get(":,:,1:{},:", riverEnd).reshape(rows, 96, width),
            memory.playerMemory().get(":,:,{}:{},:", riverEnd, meldEnd).reshape(rows, 16, width)),
        1);
  }

  private static EpsilonMahjongStateEncoder initializedEncoder(NDManager manager, int width) {
    manager.getEngine().setRandomSeed(4091);
    var encoder = new EpsilonMahjongStateEncoder(width);
    encoder.initialize(
        manager,
        DataType.FLOAT32,
        new Shape(-1, DecisionInputSchema.STATE_INT_COUNT),
        new Shape(-1, DecisionInputSchema.STATE_FLOAT_COUNT));
    return encoder;
  }

  private static NDArray numericInput(NDManager manager, int rows) {
    float[] values = new float[rows * DecisionInputSchema.STATE_FLOAT_COUNT];
    for (int i = 0; i < values.length; i++) values[i] = (i % 19 - 9) * 0.01f;
    return manager.create(values, new Shape(rows, DecisionInputSchema.STATE_FLOAT_COUNT));
  }

  private static NDArray weights(NDManager manager, Shape shape) {
    float[] values = new float[(int) shape.size()];
    for (int i = 0; i < values.length; i++) values[i] = (i % 11 - 5) * 0.03f;
    return manager.create(values, shape);
  }

  private static NDArray categoriesWithDifferentMasks(NDManager manager, int rows) {
    int[] values = new int[rows * DecisionInputSchema.STATE_INT_COUNT];
    Arrays.fill(values, 1);
    int riverOffset =
        DecisionInputSchema.ROUND_INT_COUNT
            + 4 * DecisionInputSchema.PLAYER_INT_STRIDE
            + 34 * DecisionInputSchema.TILE_INT_STRIDE;
    int meldOffset = riverOffset + 96 * DecisionInputSchema.RIVER_INT_STRIDE;
    for (int row = 0; row < rows; row++) {
      int base = row * DecisionInputSchema.STATE_INT_COUNT;
      for (int token = 0; token < 96; token++) {
        int present =
            base
                + riverOffset
                + token * DecisionInputSchema.RIVER_INT_STRIDE
                + DecisionInputSchema.RiverInt.PRESENT.ordinal();
        values[present] = row == 0 || (row + token) % 3 == 0 ? 0 : 1;
      }
      for (int token = 0; token < 16; token++) {
        int present =
            base
                + meldOffset
                + token * DecisionInputSchema.MELD_INT_STRIDE
                + DecisionInputSchema.MeldInt.PRESENT.ordinal();
        values[present] = row == 0 || (row + token) % 2 == 0 ? 0 : 1;
      }
    }
    return manager.create(values, new Shape(rows, DecisionInputSchema.STATE_INT_COUNT));
  }

  private static LayerNorm entityNormalization(EpsilonMahjongStateEncoder encoder) {
    for (var child : encoder.getChildren()) {
      if (child.getKey().endsWith("entityTokenLayerNorm")) return (LayerNorm) child.getValue();
    }
    throw new AssertionError("entity token normalization is absent");
  }

  private static void assertNonzero(float[] gradient) {
    double magnitude = 0;
    for (float value : gradient) magnitude += Math.abs(value);
    Assert.assertTrue(magnitude > 1.0e-5, "gradient must reach the tested input or parameter");
  }
}
