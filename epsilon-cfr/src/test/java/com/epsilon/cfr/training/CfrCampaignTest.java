package com.epsilon.cfr.training;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.Main;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.runtime.CfrDeployment;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 実エンジンでの収集、リプレイ保存、ネイティブ学習、整合性を保った再開、配布までを通して検証する。 */
@Test(groups = {"native", "slow"})
public class CfrCampaignTest {
  public void handCampaignAndCompletedResume() throws Exception {
    exercise("HAND_GRP");
  }

  public void matchCampaignAndCompletedResume() throws Exception {
    exercise("MATCH_MC");
  }

  private void exercise(String mode) throws Exception {
    Path root = Files.createTempDirectory("epsilon-cfr-campaign-");
    Path run = root.resolve("run");
    CfrConfig config =
        new CfrConfig(
            31007,
            1,
            mode.equals("HAND_GRP") ? 2 : 1,
            EpsilonUtilityProfile.PLACEMENT,
            new CfrConfig.Collection(
                mode,
                2,
                2,
                8,
                .2,
                true,
                8,
                2,
                4,
                new DecisionDuelArenaSettings(
                    Math.min(32, Runtime.getRuntime().availableProcessors()), 2, 64),
                new InferenceBatchingSettings(0),
                new DecisionBranchComparisonSettings(false, false, true, 1, 8, .01)),
            new CfrConfig.Learning(
                2,
                2,
                1,
                1,
                1,
                .001f,
                0,
                0,
                DecisionComputePrecision.FLOAT32,
                DecisionTensorTransfer.DIRECT_BUFFER),
            new CfrConfig.Replay(32, 32, 64L << 20, 1L << 20, 4),
            new CfrConfig.Network(16, new DeviceSettings(true, "cpu", "cpu", "cpu")),
            new CfrConfig.Boundary("", 2, 2),
            new CfrConfig.Evaluation(
                2,
                new DecisionChampionDuelSettings(1, 2, .01, .25),
                new DecisionEvalVsSettings(4, 16, 1, 40000),
                new DecisionDuelArenaSettings(1, 1, 4),
                new InferenceBatchingSettings(0)),
            CfrConfig.defaults().initialization());
    CfrCampaign.train(config, run);
    CfrRunState state = CfrRunStore.inspect(run);
    Assert.assertEquals(state.phase(), CfrRunState.Phase.COMPLETE);
    Assert.assertEquals(state.iteration(), config.iterations() + 1L);
    byte[] pointer = Files.readAllBytes(run.resolve("current.json"));
    CfrCampaign.resume(run);
    Assert.assertEquals(Files.readAllBytes(run.resolve("current.json")), pointer);
    CfrDeployment.export(run, root.resolve("deploy"), "average");
    CfrRunStore.verifyArtifact(root.resolve("deploy"));
    ByteArrayOutputStream report = new ByteArrayOutputStream();
    Path startup = Files.writeString(root.resolve("startup.toml"), "");
    Assert.assertEquals(
        Main.run(
            new String[] {"--settings", startup.toString(), "verify", "--run", run.toString()},
            new PrintStream(report)),
        0);
    Assert.assertTrue(report.toString().contains("\"verified\": true"));
    Assert.assertTrue(Files.isRegularFile(root.resolve("deploy/deployment.json")));
  }
}
