package com.epsilon.prism.app;

import com.epsilon.agent.settings.Layer;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.model.ModelShapes;
import com.epsilon.prism.model.PairLoss;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.noise.TempoLrMode;
import com.epsilon.prism.noise.TempoSettings;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * fit-cohorts の設定のテスト。run に記録した最新の設定に {@code --set} を重ねて、学習に効くキーだけを変えられる。ほかの種類のキー、
 * 知らないキー、型の合わない値は拒む。{@code --set} が無ければ、記録した設定をそのまま使う。
 */
public class FitCohortsSettingsTest {

  @Test
  public void setChangesOnlyFitKeys() throws Exception {
    Path run = Files.createTempDirectory("prism-refit-settings");
    try {
      TestSettings.create(run, "phi.source=SCORE_RANK");
      Assert.assertEquals(FitCohorts.settings(run, List.of()).value(), PrismApp.latest(run));
      PrismSettings changed =
          FitCohorts.settings(
                  run,
                  List.of(set("fit.lrPeak=0.002", "fit.maxSteps=10000", "noise.tempo.lrMode=FULL")))
              .value();
      Assert.assertEquals(changed.fit().lrPeak(), 0.002);
      Assert.assertEquals(changed.fit().maxSteps(), 10000);
      Assert.assertEquals(((TempoSettings.On) changed.noise().tempo()).lrMode(), TempoLrMode.FULL);
      for (String wrong :
          new String[] {
            "nonsense=1",
            "fit.trainRoots=true",
            "inference.devices=[\"gpu0\"]",
            "fit.auxOracle=x",
            "fit.maxSteps=many",
            "inference.kernels=hip",
            "network.depth=3",
            "labels.maxCandidates=4"
          })
        Assert.assertThrows(
            IllegalArgumentException.class, () -> FitCohorts.settings(run, List.of(set(wrong))));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test(dataProvider = "defaultShapes", dataProviderClass = ModelShapes.class)
  public void absentPlanFieldsUseTheirDefaults(ModelShapes.DefaultShape fixture) {
    PrismShape shape = fixture.arm().shape();
    Gson gson = new Gson();
    JsonObject saved = gson.toJsonTree(shape).getAsJsonObject();
    fixture.omitted().forEach(saved::remove);
    String json =
        """
        {"learner":{"steps":1,"batchSize":128,"lrPeak":0.001,"lrFloor":0.0001,
                    "gradientClip":1,"lambdaDiscard":0.01,"lambdaCommit":0.02},
         "shape":%s,"auxiliary":{},
         "tokenBudget":160000,"seed":31}
        """
            .formatted(saved);
    FitPlan plan = gson.fromJson(json, FitPlan.class);
    Assert.assertEquals(plan.shape(), shape);
    Assert.assertEquals(plan.learner().adamEpsilon(), com.epsilon.prism.learn.AdamEpsilon.FIXED);
    Assert.assertEquals(plan.learner().loss(), PairLoss.Mode.PRISM);
    Assert.assertEquals(plan.learner().classWeight(), PairLoss.ClassWeight.INVERSE_LAMBDA);
    JsonObject header = gson.toJsonTree(plan).getAsJsonObject();
    JsonObject record = header.getAsJsonObject("shape");
    record.addProperty("relationKeys", false);
    JsonParseException error =
        Assert.expectThrows(JsonParseException.class, () -> gson.fromJson(header, FitPlan.class));
    Assert.assertTrue(error.getMessage().contains("relationKeys"), error.getMessage());
  }

  private static Layer set(String... assignments) {
    return Layer.assignments("--set", List.of(assignments));
  }
}
