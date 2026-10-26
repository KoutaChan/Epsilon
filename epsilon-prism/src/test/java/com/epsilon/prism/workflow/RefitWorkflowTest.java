package com.epsilon.prism.workflow;

import com.google.gson.JsonObject;
import org.testng.Assert;
import org.testng.annotations.Test;

/** refit の --overrides は元の欄の型で上書きし、知らない欄と型の合わない値は拒む。 */
public class RefitWorkflowTest {
  @Test
  public void overridesKeepTheFieldTypes() {
    JsonObject tree = new JsonObject();
    tree.addProperty("memoryDecay", 0.8);
    tree.addProperty("fitSteps", 2000);
    tree.addProperty("certify", false);
    tree.addProperty("learner", "crossfit");
    RefitWorkflow.override(tree, "memoryDecay=1.0, fitSteps=10000,certify=true,learner=single");
    Assert.assertEquals(tree.get("memoryDecay").getAsDouble(), 1.0);
    Assert.assertEquals(tree.get("fitSteps").getAsInt(), 10000);
    Assert.assertTrue(tree.get("certify").getAsBoolean());
    Assert.assertEquals(tree.get("learner").getAsString(), "single");
    Assert.assertThrows(IllegalArgumentException.class, () -> RefitWorkflow.override(tree, "hidden=1"));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> RefitWorkflow.override(tree, "fitSteps=many"));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> RefitWorkflow.override(tree, "certify=yes"));
  }
}
