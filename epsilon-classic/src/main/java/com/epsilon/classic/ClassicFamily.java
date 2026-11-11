package com.epsilon.classic;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Nano・Major の系列。チェックポイントのネットワークを開き、CPU なら {@link ClassicPolicy.Inline}、GPU なら {@link ClassicPolicy.Batched}
 * の方策を返す。実装は系列ごとに 1 つ（{@link Nano}・{@link Major}）あり、META-INF/services に書いておくと ServiceLoader が見つける。
 */
public abstract sealed class ClassicFamily implements ModelFamily permits ClassicFamily.Nano, ClassicFamily.Major {

  private final Architecture architecture;

  private ClassicFamily(Architecture architecture) {
    this.architecture = architecture;
  }

  @Override
  public Series series() {
    return architecture.series();
  }

  /** この系列のアーキテクチャのネットワークの方策を開く（{@link #open(Architecture, Path, Variant, PolicyOptions, NetworkRuntime)}）。 */
  @Override
  public Policy open(Path checkpoint, Variant variant, PolicyOptions options, NetworkRuntime runtime)
      throws IOException {
    return open(architecture, checkpoint, variant, options, runtime).asPolicy();
  }

  /**
   * チェックポイントのネットワークを設定したデバイスに読み込み、設定した精度で動く方策を開く。ヘッダーは重みを読む前に照合する。
   *
   * @param architecture チェックポイントが持つはずのアーキテクチャ
   * @param runtime GPU の方策を動かす、呼び出し元の推論のランタイム（null なら方策が自分で作って持つ）。CPU の方策では使わない
   * @return CPU なら {@link ClassicPolicy.Inline}、GPU なら {@link ClassicPolicy.Batched}
   * @throws IllegalArgumentException チェックポイントが null か、ファイル名の無いパスのとき。または推論にネイティブのカーネルを選んだとき（Nano・Major には無い）
   * @throws IllegalStateException チェックポイントのアーキテクチャが違うとき。またはルールか入力のフィンガープリントが合わないとき
   */
  public static ClassicPolicy open(
      Architecture architecture, Path checkpoint, Variant variant, PolicyOptions options, NetworkRuntime runtime)
      throws IOException {
    Objects.requireNonNull(options, "options");
    if (options.kernels() == Kernels.NATIVE) {
      throw new IllegalArgumentException(architecture + " has no native kernels: kernels must be EAGER");
    }
    String label = architecture.series().label(checkpoint);
    ClassicHeader header = ClassicCheckpoint.verify(checkpoint, variant);
    if (header.architecture() != architecture) {
      throw new IllegalStateException(
          "Checkpoint " + checkpoint + " holds a " + header.architecture() + " network, not " + architecture);
    }
    Device device = NetworkDevices.device(options.device());
    NetworkRuntime.initialize();
    NDManager manager = NDManager.newBaseManager(device, PtEngine.ENGINE_NAME);
    try {
      ClassicNetwork network = ClassicCheckpoint.read(checkpoint, variant, manager);
      return device.isGpu()
          ? new ClassicPolicy.Batched(network, manager, options.precision(), label, runtime)
          : new ClassicPolicy.Inline(network, manager, options.precision(), label);
    } catch (IOException | RuntimeException | Error failure) {
      try {
        manager.close();
      } catch (RuntimeException | Error closing) {
        failure.addSuppressed(closing);
      }
      throw failure;
    }
  }

  public static final class Nano extends ClassicFamily {

    public Nano() {
      super(Architecture.NANO);
    }
  }

  public static final class Major extends ClassicFamily {

    public Major() {
      super(Architecture.MAJOR);
    }
  }
}
