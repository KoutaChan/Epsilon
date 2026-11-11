package com.epsilon.prism.app;

import com.epsilon.agent.settings.Layer;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.noise.TempoLrMode;
import com.epsilon.prism.noise.TempoSettings;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
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
          FitCohorts.settings(run, List.of(set("fit.lrPeak=0.002", "fit.maxSteps=10000", "noise.tempo.lrMode=FULL"))).value();
      Assert.assertEquals(changed.fit().lrPeak(), 0.002);
      Assert.assertEquals(changed.fit().maxSteps(), 10000);
      Assert.assertEquals(((TempoSettings.On) changed.noise().tempo()).lrMode(), TempoLrMode.FULL);
      for (String wrong :
          new String[] {
            "nonsense=1", "fit.trainRoots=true", "inference.devices=[\"gpu0\"]", "fit.auxOracle=x", "fit.maxSteps=many",
            "inference.kernels=hip", "network.depth=3", "labels.maxCandidates=4"
          })
        Assert.assertThrows(IllegalArgumentException.class, () -> FitCohorts.settings(run, List.of(set(wrong))));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  private static Layer set(String... assignments) {
    return Layer.assignments("--set", List.of(assignments));
  }
}
