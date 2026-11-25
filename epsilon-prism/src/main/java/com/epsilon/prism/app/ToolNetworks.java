package com.epsilon.prism.app;

import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.backend.KernelDevices;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 評価に使う凍結した方策。GPU の BF16 では専用カーネルを選ぶ（方策と同じく、宣言が対応形でないモデルは eager）。FP32 と CPU は eager で
 * 計算する。
 */
record ToolNetworks(InferenceSettings settings, NetworkRuntime runtime) implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(ToolNetworks.class);

  /**
   * 評価の記録に書く、開いたネットワーク。
   *
   * @param version バージョンのディレクトリ
   * @param backend 実際に使う計算（{@link KernelDevices#label}）
   * @param precision 計算の精度
   */
  record Execution(String version, String backend, Precision precision) {}

  enum Role {
    HOLDOUT,
    A,
    B,
    MU
  }

  Execution execution(Role role, Path version, NetworkPolicy network) {
    String backend = KernelDevices.label(network.backend());
    LOG.info(
        "Evaluation {} network {} backend={} precision={}",
        role,
        version,
        backend,
        settings.precision());
    return new Execution(version.toString(), backend, settings.precision());
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
    var path =
        settings.precision() == Precision.BFLOAT16 && devices.primary().isGpu()
            ? Kernels.NATIVE
            : Kernels.EAGER;
    return duel
        ? Snapshot.openDuel(
            version, devices.devices(), runtime, settings.maxBatch(), settings.precision(), path)
        : Snapshot.open(
            version, devices.devices(), runtime, settings.maxBatch(), settings.precision(), path);
  }

  @Override
  public void close() {
    runtime.close();
  }
}
