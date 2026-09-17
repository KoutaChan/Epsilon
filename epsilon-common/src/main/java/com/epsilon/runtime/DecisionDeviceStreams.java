package com.epsilon.runtime;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;

/** 同じGPUでフェーズをまたいで借用する三本のストリーム。モデルやテンソルを保持しない。 */
public final class DecisionDeviceStreams {

  private final Device device;
  private final PtStream compute;
  private final PtStream h2d;
  private final PtStream d2h;
  // このオブジェクトのロックで保護する。実行コンテキストが、このデバイスの共有パイプラインの生成と終了を管理する。
  DecisionDevicePipeline pipeline;

  DecisionDeviceStreams(Device device) {
    this.device = device;
    PtEngine engine = (PtEngine) Engine.getEngine("PyTorch");
    compute = engine.newStream(device);
    PtStream createdH2d = null;
    try {
      createdH2d = engine.newStream(device);
      d2h = engine.newStream(device);
      h2d = createdH2d;
    } catch (RuntimeException | Error failure) {
      if (createdH2d != null) {
        closeAfter(createdH2d, failure);
      }
      closeAfter(compute, failure);
      throw failure;
    }
  }

  public Device device() {
    return device;
  }

  public PtStream compute() {
    return compute;
  }

  public PtStream h2d() {
    return h2d;
  }

  public PtStream d2h() {
    return d2h;
  }

  void awaitIdle() {
    awaitStream(h2d);
    awaitStream(compute);
    awaitStream(d2h);
  }

  private static void awaitStream(PtStream stream) {
    try (PtEvent completion = stream.newEvent()) {
      try (PtStreamScope ignored = stream.openScope()) {
        completion.record();
      }
      completion.synchronize();
    }
  }

  void closeOwned() {
    Throwable failure = closeAfter(d2h, null);
    failure = closeAfter(h2d, failure);
    failure = closeAfter(compute, failure);
    if (failure instanceof RuntimeException runtime) {
      throw runtime;
    }
    if (failure instanceof Error error) {
      throw error;
    }
  }

  private static Throwable closeAfter(PtStream stream, Throwable failure) {
    try {
      stream.close();
    } catch (RuntimeException | Error closeFailure) {
      if (failure == null) {
        return closeFailure;
      }
      failure.addSuppressed(closeFailure);
    }
    return failure;
  }
}
