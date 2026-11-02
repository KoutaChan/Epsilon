package com.epsilon.nano.ai.decision.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.nano.ai.decision.input.DecisionCategoryLayout;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import java.nio.ShortBuffer;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public final class TransitionFeaturePackTest {
  private static final int CATEGORIES = DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE;
  private static final int NUMERICS = DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE;
  private static final int EMBEDDING = 8;

  @Test(groups = "native")
  public void frozenAndMutableCpuInputsPreserveValuesAndEmptyRows() {
    verifyValues(Device.cpu(), DataType.FLOAT32, DataType.FLOAT32);
  }

  @DataProvider
  public Object[][] computeTypes() {
    return new Object[][] {
      {DataType.BFLOAT16, DataType.BFLOAT16},
      {DataType.FLOAT16, DataType.FLOAT16},
      {DataType.FLOAT32, DataType.FLOAT32},
      {DataType.BFLOAT16, DataType.FLOAT32}
    };
  }

  @Test(groups = "rocm", dataProvider = "computeTypes")
  public void frozenAndMutableGpuInputsPreserveValuesAndTypes(
      DataType parameterType, DataType numericType) {
    if (Engine.getEngine("PyTorch").getGpuCount() == 0)
      throw new SkipException("This test requires a ROCm GPU");
    verifyValues(Device.gpu(0), parameterType, numericType);
  }

  private static void verifyValues(Device device, DataType parameterType, DataType numericType) {
    try (var manager = NDManager.newBaseManager(device, "PyTorch");
        var inference = manager.getEngine().newInferenceMode()) {
      var encoder = new EpsilonDecisionCandidateEncoder(16);
      encoder.initialize(manager, parameterType);
      var embedding = embedding(encoder);
      var store = new ParameterStore(manager, false);
      float[] weights = values(embedding.getValue(store, device, false));
      for (boolean frozen : new boolean[] {false, true}) {
        embedding.freezeParameters(frozen);
        for (int count : new int[] {0, 1, 37, 9, 0}) {
          try (var scope = manager.newSubManager()) {
            NDArray categories = categories(scope, count);
            NDArray numerics = numerics(scope, count, numericType);
            short[] beforeCategories = categories.toShortArray();
            float[] beforeNumerics = values(numerics);
            NDArray expected = original(embedding, store, categories, numerics, false);
            NDArray actual =
                encoder.encodeCompactedTransitionFeatures(
                    store, categories, numerics, false, new PairList<>());
            Assert.assertEquals(actual.getDataType(), expected.getDataType());
            Assert.assertEquals(
                actual.getShape(), new Shape(count, 1, 1, CATEGORIES * EMBEDDING + NUMERICS));
            // 加算は整数インデックスだけなので、低精度でも特徴値の丸めや許容誤差は生じない。
            Assert.assertEquals(values(actual), values(expected));
            Assert.assertEquals(categories.toShortArray(), beforeCategories);
            Assert.assertEquals(values(numerics), beforeNumerics);
          }
        }
      }
      Assert.assertEquals(values(embedding.getValue(store, device, false)), weights);
    }
  }

  @Test(groups = "native")
  public void trainingPreservesEmbeddingAndNumericGradients() {
    try (var manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      var encoder = new EpsilonDecisionCandidateEncoder(16);
      encoder.initialize(manager, DataType.FLOAT32);
      var embedding = embedding(encoder);
      var store = new ParameterStore(manager, false);
      NDArray table = embedding.getValue(store, Device.cpu(), true);
      NDArray categories = categories(manager, 7);
      NDArray numerics = numerics(manager, 7, DataType.FLOAT32);
      numerics.setRequiresGradient(true);
      float[] expected, tableGradient, numericGradient;
      try (var collector = manager.getEngine().newGradientCollector()) {
        NDArray output = original(embedding, store, categories, numerics, true);
        expected = values(output);
        collector.backward(output.square().sum());
        tableGradient = values(table.getGradient());
        numericGradient = values(numerics.getGradient());
      }
      table.getGradient().fillI(0);
      numerics.getGradient().fillI(0);
      try (var collector = manager.getEngine().newGradientCollector()) {
        NDArray output =
            encoder.encodeCompactedTransitionFeatures(
                store, categories, numerics, true, new PairList<>());
        Assert.assertEquals(values(output), expected);
        collector.backward(output.square().sum());
        Assert.assertEquals(values(table.getGradient()), tableGradient);
        Assert.assertEquals(values(numerics.getGradient()), numericGradient);
      }
    }
  }

  private static IdEmbedding embedding(EpsilonDecisionCandidateEncoder encoder) {
    IdEmbedding[] result = new IdEmbedding[1];
    encoder.registerWith(
        (name, block) -> {
          if (name.equals("actionFeatureEmbedding")) result[0] = (IdEmbedding) block;
        });
    return result[0];
  }

  private static NDArray categories(NDManager manager, int count) {
    short[] values = new short[count * CATEGORIES];
    int[] offsets = DecisionCategoryLayout.transitionOffsets();
    for (int row = 1; row < count; row++)
      for (int field = 0; field < CATEGORIES; field++) {
        int end =
            field + 1 == CATEGORIES
                ? DecisionCategoryLayout.ACTION_DICTIONARY_SIZE
                : offsets[field + 1];
        values[row * CATEGORIES + field] = (short) ((row + field) % (end - offsets[field]));
      }
    // 先頭行は全カテゴリ0。各名前空間の0を正しく参照し、PRESENT=0も勝手に省略しない。
    return manager.create(
        ShortBuffer.wrap(values), new Shape(count, 1, 1, CATEGORIES), DataType.INT16);
  }

  private static NDArray numerics(NDManager manager, int count, DataType type) {
    float[] values = new float[count * NUMERICS];
    for (int index = NUMERICS; index < values.length; index++)
      values[index] = (index % 41 - 20) / 32f;
    return manager.create(values).reshape(count, 1, 1, NUMERICS).toType(type, false);
  }

  private static NDArray original(
      IdEmbedding embedding,
      ParameterStore store,
      NDArray categories,
      NDArray numerics,
      boolean training) {
    NDArray offsets =
        categories
            .getManager()
            .create(DecisionCategoryLayout.transitionOffsets())
            .reshape(1, 1, 1, CATEGORIES);
    return embedding
        .forward(store, new NDList(categories.add(offsets)), training, new PairList<>())
        .singletonOrThrow()
        .reshape(categories.getShape().get(0), 1, 1, CATEGORIES * EMBEDDING)
        .concat(numerics, 3);
  }

  private static float[] values(NDArray array) {
    return array.toType(DataType.FLOAT32, false).toFloatArray();
  }
}
