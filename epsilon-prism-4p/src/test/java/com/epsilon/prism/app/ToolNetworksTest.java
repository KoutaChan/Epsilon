package com.epsilon.prism.app;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.play.PublishedModel;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ToolNetworksTest {
  @Test
  public void splitDuelOpensThePublishedContinuation() throws Exception {
    var snapshot = PublishedModel.fourPlayer(760).snapshot();
    try (var networks = new ToolNetworks(TestSettings.of().inference());
        var policy = networks.openDuel(snapshot)) {
      Assert.assertEquals(policy.version(), 760);
      Assert.assertEquals(policy.inference(), Kernels.EAGER);
    }
  }

  @Test
  public void cpuOpensThePublishedPolicy() throws Exception {
    var root = Files.createTempDirectory("evaluation-network");
    try {
      var path = root.resolve("version");
      try (var manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(path, HipFixtures.network(manager, 2, 3), PolicyParams.of(11));
      }
      try (var networks = new ToolNetworks(TestSettings.of().inference());
          var policy = networks.open(path)) {
        Assert.assertEquals(policy.inference(), Kernels.EAGER);
        Assert.assertEquals(policy.version(), 11);
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
