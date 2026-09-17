package com.epsilon.major.ai.model;

import com.epsilon.major.ai.model.EpsilonDecisionNetwork;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 方策と価値関数の更新対象パラメータが分離され、共通の最適化処理で正しく選択されることを検証する。 */
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
