package com.epsilon.prism.run;

import com.google.gson.JsonObject;
import org.testng.Assert;
import org.testng.annotations.Test;

/** refit の --overrides は PrismConfig の欄の型で上書きし、知らない欄・列の欄・形式・型の合わない値は拒む。 */
public class FitWorkflowTest {
  @Test
  public void overridesKeepTheFieldTypes() {
    JsonObject tree = new JsonObject();
    FitWorkflow.override(tree, "lrPeak=0.0012, fitSteps=10000,frontier=true,filterLrMode=FULL");
    Assert.assertEquals(tree.get("lrPeak").getAsDouble(), 0.0012);
    Assert.assertTrue(tree.get("lrPeak").getAsJsonPrimitive().isNumber());
    Assert.assertEquals(tree.get("fitSteps").getAsInt(), 10000);
    Assert.assertTrue(tree.get("frontier").getAsBoolean());
    Assert.assertEquals(tree.get("filterLrMode").getAsString(), "FULL");
    FitWorkflow.override(tree, "auxOracle=0,depth=3,gamesSchedule=1:8192,inference=EAGER");
    Assert.assertEquals(tree.get("auxOracle").getAsDouble(), 0.0);
    Assert.assertEquals(tree.get("depth").getAsInt(), 3);
    Assert.assertEquals(tree.get("gamesSchedule").getAsString(), "1:8192");
    Assert.assertEquals(tree.get("inference").getAsString(), "EAGER");
    for (String wrong :
        new String[] {
          "nonsense=1", "trainRoots=true", "collectDevices=gpu0", "format=1", "auxOracle=x", "fitSteps=many",
          "frontier=yes", "inference=hip", "lrPeak"
        })
      Assert.assertThrows(IllegalArgumentException.class, () -> FitWorkflow.override(tree, wrong));
  }
}
