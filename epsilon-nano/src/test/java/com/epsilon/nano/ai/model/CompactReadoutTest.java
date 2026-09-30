package com.epsilon.nano.ai.model;

import ai.djl.Device;
import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class CompactReadoutTest {

  @DataProvider
  public Object[][] layouts() {
    return new Object[][] {{2, 32, false}, {3, 128, false}, {2, 32, true}};
  }

  @Test(groups = "native", dataProvider = "layouts")
  public void compactProjectionPreservesReadoutAndGradients(int rows, int width, boolean dense) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      var readout = new EpsilonMahjongStateReadout(width);
      readout.initialize(manager, DataType.FLOAT32, new Shape(rows, width));
      var store = new ParameterStore(manager, false);
      Fixture fixture = fixture(manager, rows, width, dense);
      NDArray entities = fixture.entities();
      entities.setRequiresGradient(true);
      NDArray lossWeights = values(manager, new Shape(rows, width));
      EpsilonMahjongStateEncoder.EncodedMemory memory;
      NDArray denseOutput;
      try (var collector = manager.getEngine().newGradientCollector()) {
        memory = memory(entities, fixture.mask());
        denseOutput = readout.read(store, memory, true, new PairList<>());
        collector.backward(denseOutput.mul(lossWeights).sum());
      }
      float[] expectedInputGradient = entities.getGradient().toFloatArray();
      Map<String, float[]> expectedParameterGradients = gradients(readout);
      clearGradients(readout, entities);

      NDArray compactOutput;
      EpsilonMahjongStateReadout.ReadoutContext context;
      try (var collector = manager.getEngine().newGradientCollector()) {
        memory = memory(entities, fixture.mask());
        context = readout.summarize(memory, fixture.playerIndices());
        compactOutput = readout.read(store, memory, context, true, new PairList<>());
        collector.backward(compactOutput.mul(lossWeights).sum());
      }
      Assert.assertEquals(compactOutput.toFloatArray(), denseOutput.toFloatArray(), 0.0002f);
      Assert.assertEquals(entities.getGradient().toFloatArray(), expectedInputGradient, 0.0002f);
      Map<String, float[]> actualParameterGradients = gradients(readout);
      for (var entry : expectedParameterGradients.entrySet()) {
        Assert.assertEquals(
            actualParameterGradients.get(entry.getKey()),
            entry.getValue(),
            0.0002f,
            entry.getKey());
      }
      Assert.assertEquals(
          context.projectionInput().getShape().get(0), (long) fixture.presentEntities());

      clearGradients(readout, entities);
      try (var collector = manager.getEngine().newGradientCollector()) {
        NDArray value =
            readout.read(
                store, memory.stopGradient(), context.stopGradient(), true, new PairList<>());
        collector.backward(value.mul(lossWeights).sum());
      }
      Assert.assertEquals(entities.getGradient().abs().sum().getFloat(), 0f);
      double gradientMagnitude = 0;
      for (var parameter : readout.getParameters()) {
        gradientMagnitude += parameter.getValue().getArray().getGradient().abs().sum().getFloat();
      }
      Assert.assertTrue(
          gradientMagnitude > 0, "Detached value readout still learns its own weights");
    }
  }

  @DataProvider
  public Object[][] gpuLayouts() {
    return new Object[][] {{3, false, 384}, {17, false, 256}, {3, true, 384}};
  }

  /** BF16 の重みと autocast を使い、有効行の順序変更による丸め誤差と勾配経路を検証する。 */
  @SuppressWarnings("try")
  @Test(groups = "rocm", dataProvider = "gpuLayouts")
  public void gpuBfloat16CompactProjectionPreservesReadoutAndGradients(
      int rows, boolean dense, int maximumFeedForwardWidth) {
    Engine engine = Engine.getInstance();
    if (engine.getGpuCount() == 0) {
      throw new SkipException("This test requires a ROCm GPU.");
    }
    int width = 256;
    Device device = Device.gpu(0);
    try (NDManager manager = engine.newBaseManager(device)) {
      engine.setRandomSeed(7391);
      var readout = new EpsilonMahjongStateReadout(width, maximumFeedForwardWidth);
      readout.initialize(manager, DataType.BFLOAT16, new Shape(rows, width));
      var store = new ParameterStore(manager, false);
      Fixture fixture = fixture(manager, rows, width, dense);
      NDArray entities = fixture.entities();
      entities.setRequiresGradient(true);
      NDArray lossWeights = values(manager, new Shape(rows, width));
      EpsilonMahjongStateEncoder.EncodedMemory memory;
      NDArray denseOutput;
      try (var collector = engine.newGradientCollector();
          Autocast ignored = engine.newAutocast(device, DataType.BFLOAT16, true)) {
        memory = memory(entities, fixture.mask());
        denseOutput = readout.read(store, memory, true, new PairList<>());
        collector.backward(denseOutput.mul(lossWeights).sum());
      }
      float[] expectedInputGradient = floats(entities.getGradient());
      Map<String, float[]> expectedParameterGradients = gradients(readout);
      clearGradients(readout, entities);

      NDArray compactOutput;
      EpsilonMahjongStateReadout.ReadoutContext context;
      try (var collector = engine.newGradientCollector();
          Autocast ignored = engine.newAutocast(device, DataType.BFLOAT16, true)) {
        memory = memory(entities, fixture.mask());
        context = readout.summarize(memory, fixture.playerIndices());
        compactOutput = readout.read(store, memory, context, true, new PairList<>());
        collector.backward(compactOutput.mul(lossWeights).sum());
      }
      // BF16 の相対精度は約 1/128。前向きは約2単位、還元順序が変わる勾配は約4単位を許容する。
      // ゼロ付近では相対誤差が定義しにくいため、小さい絶対誤差枠も設ける。
      float outputError =
          assertBfloat16Close(
              floats(compactOutput), floats(denseOutput), 1f / 128, 1f / 64, "output");
      float inputGradientError =
          assertBfloat16Close(
              floats(entities.getGradient()),
              expectedInputGradient,
              1f / 256,
              1f / 32,
              "input gradient");
      float parameterGradientError = 0;
      Map<String, float[]> actualParameterGradients = gradients(readout);
      for (var entry : expectedParameterGradients.entrySet()) {
        parameterGradientError =
            Math.max(
                parameterGradientError,
                assertBfloat16Close(
                    actualParameterGradients.get(entry.getKey()),
                    entry.getValue(),
                    1f / 256,
                    1f / 32,
                    entry.getKey()));
      }
      Assert.assertEquals(
          context.projectionInput().getShape().get(0), (long) fixture.presentEntities());

      clearGradients(readout, entities);
      try (var collector = engine.newGradientCollector();
          Autocast ignored = engine.newAutocast(device, DataType.BFLOAT16, true)) {
        NDArray value =
            readout.read(
                store, memory.stopGradient(), context.stopGradient(), true, new PairList<>());
        collector.backward(value.mul(lossWeights).sum());
      }
      Assert.assertEquals(entities.getGradient().abs().sum().getFloat(), 0f);
      double gradientMagnitude = 0;
      for (var parameter : readout.getParameters()) {
        for (float gradient : floats(parameter.getValue().getArray().getGradient())) {
          gradientMagnitude += Math.abs(gradient);
        }
      }
      Assert.assertTrue(
          gradientMagnitude > 0, "Detached value readout still learns its own weights");
      System.out.printf(
          "CompactReadout BF16 rows=%d dense=%s ffnWidth=%d maxOutputAbs=%g maxInputGradientAbs=%g"
              + " maxParameterGradientAbs=%g%n",
          rows,
          dense,
          maximumFeedForwardWidth,
          outputError,
          inputGradientError,
          parameterGradientError);
    }
  }

  private static float assertBfloat16Close(
      float[] actual,
      float[] expected,
      float absoluteTolerance,
      float relativeTolerance,
      String label) {
    Assert.assertEquals(actual.length, expected.length, label);
    float maximumError = 0;
    for (int index = 0; index < actual.length; index++) {
      float error = Math.abs(actual[index] - expected[index]);
      float tolerance = absoluteTolerance + relativeTolerance * Math.abs(expected[index]);
      Assert.assertTrue(
          error <= tolerance,
          label
              + "["
              + index
              + "] expected="
              + expected[index]
              + " actual="
              + actual[index]
              + " tolerance="
              + tolerance);
      maximumError = Math.max(maximumError, error);
    }
    return maximumError;
  }

  private static float[] floats(NDArray values) {
    return values.toType(DataType.FLOAT32, false).toFloatArray();
  }

  private static Map<String, float[]> gradients(EpsilonMahjongStateReadout readout) {
    Map<String, float[]> result = new LinkedHashMap<>();
    for (var parameter : readout.getParameters()) {
      result.put(parameter.getKey(), floats(parameter.getValue().getArray().getGradient()));
    }
    return result;
  }

  private static void clearGradients(EpsilonMahjongStateReadout readout, NDArray entities) {
    entities.getGradient().fillI(0);
    for (var parameter : readout.getParameters()) {
      parameter.getValue().getArray().getGradient().fillI(0);
    }
  }

  private static EpsilonMahjongStateEncoder.EncodedMemory memory(NDArray entities, NDArray mask) {
    NDArray tiles = entities.get(":,5:39,:");
    return new EpsilonMahjongStateEncoder.EncodedMemory(
        entities.get(":,0,:"), entities, mask, tiles, tiles, mask, entities);
  }

  private static Fixture fixture(NDManager manager, int rows, int width, boolean dense) {
    int count = EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT;
    int rivers = DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER;
    int melds = DecisionInputSchema.MAX_MELDS_PER_PLAYER;
    int playerTokens = 1 + rivers + melds;
    int publicStart = 1 + GameState.NUM_PLAYERS + Tile.NUM_TILE_TYPES;
    float[] mask = new float[rows * count];
    var indices = new ArrayList<Integer>();
    int present = 0;
    for (int row = 0; row < rows; row++) {
      for (int token = 0; token < count; token++) {
        if (token < publicStart || dense || (token + row) % 5 == 0) {
          mask[row * count + token] = 1;
          present++;
        }
      }
      for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
        for (int token = 0; token < playerTokens; token++) {
          int entity =
              token == 0
                  ? 1 + player
                  : token <= rivers
                      ? publicStart + player * rivers + token - 1
                      : publicStart
                          + GameState.NUM_PLAYERS * rivers
                          + player * melds
                          + token
                          - 1
                          - rivers;
          if (mask[row * count + entity] != 0) {
            indices.add((row * GameState.NUM_PLAYERS + player) * playerTokens + token);
          }
        }
      }
    }
    NDArray entityMask = manager.create(mask, new Shape(rows, count));
    NDArray entities = values(manager, new Shape(rows, count, width)).mul(entityMask.expandDims(2));
    return new Fixture(
        entities,
        entityMask,
        manager.create(indices.stream().mapToInt(Integer::intValue).toArray()),
        present);
  }

  private static NDArray values(NDManager manager, Shape shape) {
    float[] values = new float[Math.toIntExact(shape.size())];
    for (int index = 0; index < values.length; index++) {
      values[index] = ((index % 31) - 15) * 0.013f;
    }
    return manager.create(values, shape);
  }

  private record Fixture(
      NDArray entities, NDArray mask, NDArray playerIndices, int presentEntities) {}
}
