package com.epsilon.nano.ai.model;

import com.epsilon.nano.ai.model.EpsilonDecisionNetwork;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 自己対局学習と牌譜学習で、方策と価値の学習対象パラメーターが重複しないことを検証する。 */
public class ParameterOwnershipTest {
  @DataProvider
  public Object[][] parameterOwners() {
    return new Object[][] {
      {"stateEncoder.tileEmbedding_weight", true, false},
      {"policyStateReadout.query_weight", true, false},
      {"policyHead.discardScorer_weight", true, false},
      {"valueStateReadout.query_weight", false, true},
      {"valueHiddenHead_weight", false, true},
      {"valueHead_bias", false, true},
      {"beliefNetwork.output_weight", false, false},
      {"unrelated_weight", false, false}
    };
  }

  @Test(dataProvider = "parameterOwners")
  public void onlineAndOfflineUpdatesKeepOriginalDisjointOwnership(
      String name, boolean actor, boolean value) {
    Assert.assertEquals(EpsilonDecisionNetwork.isOnlineActorParameterName(name), actor);
    Assert.assertEquals(EpsilonDecisionNetwork.isOfflinePolicyParameterName(name), actor);
    Assert.assertEquals(EpsilonDecisionNetwork.isValueParameterName(name), value);
    Assert.assertEquals(EpsilonDecisionNetwork.isOfflineValueParameterName(name), value);
    Assert.assertEquals(EpsilonDecisionNetwork.isOnlineTrainingParameterName(name), actor || value);
    Assert.assertEquals(
        EpsilonDecisionNetwork.isOfflinePretrainingParameterName(name), actor || value);
  }
}
