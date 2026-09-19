package com.epsilon.major.ai.model;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.major.ai.network.NetworkFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

public class EpsilonDecisionReadoutWidthTest {

  @Test(groups = "native")
  public void policyAndValueReadoutsUse128AttentionWidth() {
    int hiddenSize = 384;
    try (Model model =
        NetworkFactory.createDecisionModel(
            Device.cpu(), false, hiddenSize, EpsilonUtilityProfile.TENHOU)) {
      EpsilonDecisionNetwork network = (EpsilonDecisionNetwork) model.getBlock();
      NDManager manager = model.getNDManager();
      ParameterStore parameterStore = new ParameterStore(manager, false);
      assertAttentionWidth(
          readout(network, "policyStateReadout"), parameterStore, manager, hiddenSize);
      assertAttentionWidth(
          readout(network, "valueStateReadout"), parameterStore, manager, hiddenSize);

      long parameterElements = 0L;
      for (var parameter : network.getParameters()) {
        parameterElements += parameter.getValue().getArray().getShape().size();
      }
      Assert.assertEquals(parameterElements, 10_699_636L);
    }
  }

  private static EpsilonMahjongStateReadout readout(
      EpsilonDecisionNetwork network, String childName) {
    for (var child : network.getChildren()) {
      if (child.getKey().endsWith(childName)) {
        return (EpsilonMahjongStateReadout) child.getValue();
      }
    }
    throw new AssertionError("Missing child block: " + childName);
  }

  private static void assertAttentionWidth(
      EpsilonMahjongStateReadout readout,
      ParameterStore parameterStore,
      NDManager manager,
      int hiddenSize) {
    EpsilonMahjongStateReadout.FrozenParameterView parameters =
        readout.frozenParameterView(parameterStore, manager);
    Assert.assertEquals(parameters.queryWeight().getShape(), new Shape(128, hiddenSize));
    Assert.assertEquals(parameters.keyValueWeight().getShape(), new Shape(256, hiddenSize));
    Assert.assertEquals(parameters.contextWeight().getShape(), new Shape(hiddenSize, 128));
  }
}
