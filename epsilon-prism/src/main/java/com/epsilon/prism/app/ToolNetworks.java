package com.epsilon.prism.app;

import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 評価に使う凍結した方策。GPU の専用カーネルが使える形だけ HIP で開く。 */
record ToolNetworks(InferenceSettings settings, NetworkRuntime runtime) implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(ToolNetworks.class);

  record Execution(String version, InferencePath backend, Precision precision) {}

  enum Role {
    HOLDOUT,
    A,
    B,
    MU
  }

  Execution execution(Role role, Path version, NetworkPolicy network) {
    LOG.info(
        "Evaluation {} network {} backend={} precision={}",
        role,
        version,
        network.inference(),
        settings.precision());
    return new Execution(version.toString(), network.inference(), settings.precision());
  }

  ToolNetworks(InferenceSettings settings) {
    this(settings, new NetworkRuntime(settings));
  }

  NetworkPolicy open(Path version) throws IOException {
    return open(version, false);
  }

  /** eager ではバッチ内の位置による丸めの違いで打牌が変わり得るため、対戦で使う方策は常に一行で評価する。 */
  NetworkPolicy openDuel(Path version) throws IOException {
    return open(version, true);
  }

  private NetworkPolicy open(Path version, boolean duel) throws IOException {
    var devices = settings.networkDevices();
    InferencePath path =
        settings.precision() == Precision.BFLOAT16
                && devices.primary().isGpu()
                && HipLibrary.bundled()
            ? InferencePath.HIP.resolve(
                devices.primary(), PrismNetwork.readShape(Snapshot.model(version)))
            : InferencePath.EAGER;
    int capacity = duel && path == InferencePath.EAGER ? 1 : settings.maxBatch();
    return Snapshot.open(version, devices.devices(), runtime, capacity, settings.precision(), path);
  }

  @Override
  public void close() {
    runtime.close();
  }
}
