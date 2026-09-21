package com.epsilon.major.ai.decision.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.Linear;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.Action;
import com.epsilon.major.ai.decision.input.DecisionFeatureCodec;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import org.testng.Assert;
import org.testng.annotations.Test;

public final class EpsilonWaitPointContextTest {

  @Test(groups = "native")
  public void selectedPointEncoderPreservesParameterGradients() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      EpsilonPointOutcomeEncoder encoder = new EpsilonPointOutcomeEncoder(16);
      encoder.initialize(
          manager, DataType.FLOAT32, new Shape(3, EpsilonPointProjection.FEATURE_WIDTH));
      ParameterStore store = new ParameterStore(manager, true);
      NDArray weight = encoder.getParameters().valueAt(0).getArray();
      for (boolean empty : new boolean[] {false, true}) {
        NDArray features = manager.zeros(new Shape(3, EpsilonPointProjection.FEATURE_WIDTH));
        if (!empty) features.set(new ai.djl.ndarray.index.NDIndex("1,:"), 0.5f);
        NDArray valid = manager.create(empty ? new float[] {0, 0, 0} : new float[] {0, 1, 0});
        NDArray indices = manager.create(empty ? new int[0] : new int[] {1});
        weight.getGradient().fillI(0);
        float[] expected;
        float[] gradients;
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
          NDArray output = encoder.encode(store, features, true, new PairList<>());
          expected = output.toFloatArray();
          collector.backward(output.sum());
          gradients = weight.getGradient().toFloatArray();
        }
        weight.getGradient().fillI(0);
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
          NDArray output =
              encoder.encodeSelected(store, features, valid, indices, true, new PairList<>());
          assertClose(output.toFloatArray(), expected);
          collector.backward(output.sum());
          assertClose(weight.getGradient().toFloatArray(), gradients);
        }
      }
    }
  }

  @Test(groups = "native")
  public void compactWaitContextsPreserveOutputsAndParameterGradients() {
    for (boolean empty : new boolean[] {false, true}) {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
        EpsilonDecisionPolicyHead head =
            new EpsilonDecisionPolicyHead(384, EpsilonUtilityProfile.TENHOU);
        head.initialize(manager, DataType.FLOAT32, new Shape(3, 384));
        EpsilonPointProjection projection = head.pointProjection();
        EpsilonPointOutcomeEncoder encoder = head.pointOutcomeEncoder();
        int factWidth = DecisionInputSchema.WinFact.values().length;
        int[] facts = new int[3 * 3 * 2 * factWidth];
        if (!empty) {
          System.arraycopy(new int[] {1, 3, 4, 0, 0}, 0, facts, 0, factWidth);
          System.arraycopy(new int[] {1, 2, 3, 0, 0}, 0, facts, factWidth, factWidth);
          System.arraycopy(new int[] {1, 5, 4, 0, 1}, 0, facts, 16 * factWidth, factWidth);
        }
        int[][] states = new int[3][DecisionInputSchema.RoundInt.values().length];
        for (int[] state : states) {
          state[DecisionInputSchema.RoundInt.PLAYER_SEAT.ordinal()] = 1;
          state[DecisionInputSchema.RoundInt.DEALER_RELATIVE_SEAT.ordinal()] = 2;
          state[DecisionInputSchema.RoundInt.KYOKU_INDEX.ordinal()] = 1;
        }
        NDArray ledger =
            manager.create(
                new int[][] {{250, 250, 250, 250}, {300, 200, 250, 250}, {100, 400, 200, 300}});
        NDArray state = manager.create(states);
        NDArray action =
            manager.full(
                new Shape(3, 1),
                DecisionFeatureCodec.actionType(Action.Type.DAHAI),
                DataType.INT32);
        NDArray tiles = manager.ones(new Shape(3, 1, 1, 34), DataType.INT32);
        NDArray waitIds = manager.ones(new Shape(3, 1, 1, 3), DataType.INT32);
        NDArray waitFacts = manager.create(facts).reshape(3, 1, 1, 3, 2, factWidth);
        ParameterStore store = new ParameterStore(manager, true);
        NDArray weight = encoder.getParameters().valueAt(0).getArray();
        float[] expected;
        float[] expectedGradient;
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
          var projected =
              projection.projectWaits(
                  ledger,
                  state,
                  action,
                  waitFacts,
                  manager.zeros(new Shape(3, 1, 1, 3), DataType.INT32));
          NDArray contexts = encoder.encode(store, projected.features(), true, new PairList<>());
          NDArray output =
              contexts
                  .mul(projected.validMask().expandDims(5))
                  .sum(new int[] {4})
                  .div(projected.validMask().sum(new int[] {4}).maximum(1).expandDims(4));
          expected = output.toFloatArray();
          collector.backward(output.sum());
          expectedGradient = weight.getGradient().toFloatArray();
        }
        weight.getGradient().fillI(0);
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
          NDArray actual =
              head.encodeWaitPointContexts(
                  store,
                  ledger,
                  state,
                  action,
                  tiles,
                  waitIds,
                  waitFacts,
                  DataType.FLOAT32,
                  true,
                  new PairList<>());
          assertClose(actual.toFloatArray(), expected);
          collector.backward(actual.sum());
          assertClose(weight.getGradient().toFloatArray(), expectedGradient);
        }
        try (var inference = manager.getEngine().newInferenceMode()) {
          NDArray actual =
              head.encodeWaitPointContexts(
                  store,
                  ledger,
                  state,
                  action,
                  tiles,
                  waitIds,
                  waitFacts,
                  DataType.FLOAT32,
                  false,
                  new PairList<>());
          assertClose(actual.toFloatArray(), expected);
        }
      }
    }
  }

  private static void assertClose(float[] actual, float[] expected) {
    Assert.assertEquals(actual.length, expected.length);
    for (int index = 0; index < actual.length; index++) {
      Assert.assertEquals(actual[index], expected[index], 2e-5f, "element=" + index);
    }
  }

  @Test(groups = "native")
  public void selectedWaitProjectionMatchesDenseFloat32() {
    assertSelectedWaitProjection(Device.cpu(), DataType.FLOAT32);
  }

  @Test(groups = "rocm")
  public void selectedWaitProjectionMatchesDenseBfloat16() {
    assertSelectedWaitProjection(Device.gpu(), DataType.BFLOAT16);
  }

  private static void assertSelectedWaitProjection(Device device, DataType dataType) {
    Engine engine = Engine.getEngine("PyTorch");
    engine.setRandomSeed(6413);
    try (NDManager manager = engine.newBaseManager(device);
        var inference = engine.newInferenceMode()) {
      EpsilonDecisionPolicyHead head =
          new EpsilonDecisionPolicyHead(384, EpsilonUtilityProfile.TENHOU);
      head.initialize(manager, dataType, new Shape(3, 384));
      head.freezeParameters(true);
      Linear projection =
          (Linear)
              head.getChildren().stream()
                  .filter(child -> child.getKey().endsWith("transitionWaitKeyValueProjection"))
                  .findFirst()
                  .orElseThrow()
                  .getValue();
      NDArray bias = projection.getDirectParameters().get("bias").getArray();
      bias.set(
          new NDIndex(":"),
          manager.arange(bias.size()).mul(0.0625f).sub(1).toType(dataType, false));
      ParameterStore store = new ParameterStore(manager, false);
      int slots = DecisionInputSchema.MAX_WAIT_TILE_TYPES;
      int factWidth = DecisionInputSchema.WinFact.values().length;
      int[][] states = new int[3][DecisionInputSchema.RoundInt.values().length];
      for (int[] state : states) {
        state[DecisionInputSchema.RoundInt.PLAYER_SEAT.ordinal()] = 1;
        state[DecisionInputSchema.RoundInt.DEALER_RELATIVE_SEAT.ordinal()] = 2;
        state[DecisionInputSchema.RoundInt.KYOKU_INDEX.ordinal()] = 1;
      }
      NDArray ledger =
          manager.create(
              new int[][] {{250, 250, 250, 250}, {300, 200, 250, 250}, {100, 400, 200, 300}});
      NDArray state = manager.create(states);
      NDArray actions =
          manager.full(
              new Shape(3, 1), DecisionFeatureCodec.actionType(Action.Type.DAHAI), DataType.INT32);
      NDArray tiles = manager.ones(new Shape(3, 1, 1, 34), DataType.INT32);
      // 無効な事実でも待ち牌IDは非ゼロになり得るため、未選択行のバイアスも比較する。
      NDArray waitIds = manager.ones(new Shape(3, 1, 1, slots), DataType.INT32);
      NDArray permutation = manager.create(new int[] {2, 0, 1});
      NDArray sourceLedger = com.epsilon.ai.model.EpsilonMaskedRows.gather(ledger, permutation);
      NDArray sourceStates = com.epsilon.ai.model.EpsilonMaskedRows.gather(state, permutation);
      NDArray batchIds = manager.create(new int[] {1, 2, 0});
      try (var autocast =
          dataType == DataType.FLOAT32
              ? null
              : engine.newAutocast(device, DataType.BFLOAT16, false)) {
        for (boolean empty : new boolean[] {false, true}) {
          int[] facts = new int[3 * slots * 2 * factWidth];
          if (!empty) {
            System.arraycopy(new int[] {1, 3, 4, 0, 0}, 0, facts, 0, factWidth);
            System.arraycopy(new int[] {1, 2, 3, 0, 0}, 0, facts, factWidth, factWidth);
            System.arraycopy(
                new int[] {1, 5, 4, 0, 1}, 0, facts, (3 * slots - 1) * 2 * factWidth, factWidth);
          }
          NDArray waitFacts = manager.create(facts).reshape(3, 1, 1, slots, 2, factWidth);
          NDArray indices = manager.create(empty ? new int[0] : new int[] {0, 3 * slots - 1});
          NDArray contexts =
              head.encodeWaitPointContexts(
                  store,
                  ledger,
                  state,
                  actions,
                  tiles,
                  waitIds,
                  waitFacts,
                  dataType,
                  false,
                  new PairList<>());
          NDArray expected =
              projection.forward(store, new NDList(contexts), false).singletonOrThrow();
          NDArray actual =
              head.encodeSelectedWaitKeyValues(
                  store,
                  sourceLedger,
                  sourceStates,
                  actions,
                  tiles,
                  waitIds,
                  waitFacts,
                  dataType,
                  new PairList<>(),
                  indices,
                  batchIds);
          Assert.assertEquals(actual.getDataType(), expected.getDataType());
          Assert.assertEquals(actual.getShape(), expected.getShape());
          float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
          float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
          float tolerance = dataType == DataType.FLOAT32 ? 2e-5f : 0.02f;
          for (int i = 0; i < actualValues.length; ++i) {
            Assert.assertEquals(actualValues[i], expectedValues[i], tolerance, "element=" + i);
          }
          Assert.assertEquals(
              actual.get("0,0,0,1,:").toType(DataType.FLOAT32, false).toFloatArray(),
              expected.get("0,0,0,1,:").toType(DataType.FLOAT32, false).toFloatArray());
          Assert.assertFalse(indices.isReleased(), "caller retains wait-index ownership");
          Assert.assertFalse(batchIds.isReleased(), "caller retains batch-index ownership");
        }
      }
    }
  }
}
