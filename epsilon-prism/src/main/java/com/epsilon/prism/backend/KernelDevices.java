package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.jni.LibUtils;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.kernel.KernelProvider;
import com.epsilon.prism.kernel.KernelProvider.Backend;
import com.epsilon.prism.kernel.ModelSchema;
import com.epsilon.prism.model.PrismNetwork;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/** DJL が読み込んだ PyTorch のライブラリから GPU のランタイムを判定する。 */
public final class KernelDevices {
  private KernelDevices() {}

  public static Backend backend() {
    Engine.getEngine(PtEngine.ENGINE_NAME);
    var dir = Path.of(LibUtils.getLibtorchPath());
    if (Files.isRegularFile(dir.resolve(System.mapLibraryName("torch_hip")))) return Backend.HIP;
    if (Files.isRegularFile(dir.resolve(System.mapLibraryName("torch_cuda")))) return Backend.CUDA;
    throw new IllegalStateException("Cannot identify the PyTorch GPU runtime in " + dir);
  }

  public static KernelProvider.Request request(
      PrismNetwork network, Device device, BatchPlan.Capacity capacity, KernelProvider.Use use) {
    return new KernelProvider.Request(
        network.declaration(),
        backend(),
        device.getDeviceId(),
        ModelSchema.Element.BF16,
        use,
        KernelInputs.capacity(network.format(), capacity),
        KernelProvider.Cache.OFF);
  }

  /**
   * 記録とログに書く、実際に使う計算の名前を返す。
   *
   * @param backend 専用カーネルのバックエンド（eager なら空）
   * @return バックエンドの名前か {@code EAGER}
   */
  public static String label(Optional<Backend> backend) {
    return backend.map(Enum::name).orElse(Kernels.EAGER.name());
  }
}
