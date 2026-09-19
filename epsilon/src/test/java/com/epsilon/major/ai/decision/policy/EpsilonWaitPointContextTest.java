package com.epsilon.major.ai.decision.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
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
  public void compactWaitContextsPreserveOutputsAndParameterGradients() {
    for (boolean empty : new boolean[] {false, true}) {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
        EpsilonDecisionPolicyHead head =
            new EpsilonDecisionPolicyHead(384, EpsilonUtilityProfile.TENHOU);
        head.initialize(manager, DataType.FLOAT32, new Shape(3, 384));
        EpsilonPointProjection projection = head.pointProjection();
        EpsilonPointOutcomeEncoder encoder = head.pointOutcomeEncoder();
        int factWidth = DecisionInputSchema.WinFact.values().length;
        int[] facts = new int[3 * 2 * factWidth];
        if (!empty) {
          System.arraycopy(new int[] {1, 3, 4, 0, 0}, 0, facts, 0, factWidth);
          System.arraycopy(new int[] {1, 2, 3, 0, 0}, 0, facts, factWidth, factWidth);
          System.arraycopy(new int[] {1, 5, 4, 0, 1}, 0, facts, 4 * factWidth, factWidth);
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
        NDArray waitIds = manager.ones(new Shape(3, 1, 1, 1), DataType.INT32);
        NDArray waitFacts = manager.create(facts).reshape(3, 1, 1, 1, 2, factWidth);
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
                  manager.zeros(new Shape(3, 1, 1, 1), DataType.INT32));
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
      }
    }
  }

  private static void assertClose(float[] actual, float[] expected) {
    Assert.assertEquals(actual.length, expected.length);
    for (int index = 0; index < actual.length; index++) {
      Assert.assertEquals(actual[index], expected[index], 2e-5f, "element=" + index);
    }
  }
}
