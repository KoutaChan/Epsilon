package com.epsilon.prism.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.model.FrozenNetwork;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.Uploads;
import com.epsilon.runtime.DecisionDevicePipeline;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * eager の推論の道（1 つのデバイス）。保存した網を凍結して持ち（{@link FrozenNetwork}。GPU は BF16 の丸め、CPU は FP32）、
 * 実行枠の仕事を足す。
 *
 * <p>実行枠では、入力を共通の H2D のストリームで作り、順伝播を実行枠ごとの計算のストリームで出す（網の索引の転送が同期するので、ほかの実行枠の
 * 計算を待たないため）。出力の D2H は共通のパイプラインが受け持つ（{@code Slot.prepareOutput}）。実行枠の計算のストリームは提出のスレッドだけが
 * 作り、使う。
 *
 * <p>1 回の仕事は索引の送り出しの置き場（{@link Uploads}）を池から借り、終わると返す（呼び手のスレッドがいくつあっても、置き場は同時に 1 つの
 * 仕事だけが使う）。
 */
final class EagerInference implements DeviceInference {

  private final FrozenNetwork network;
  private final Map<DecisionDevicePipeline.Slot, PtStream> streams = new IdentityHashMap<>();
  private final ConcurrentLinkedQueue<Uploads> uploads = new ConcurrentLinkedQueue<>();

  private EagerInference(FrozenNetwork network) {
    this.network = network;
  }

  /**
   * 保存した網を読む。
   *
   * @param model 網のディレクトリ（weights.bin と model.json）
   * @param device デバイス
   * @return 推論の道
   * @throws IOException 読めないか、形式・指紋・重みが合わないとき
   */
  static EagerInference load(Path model, Device device) throws IOException {
    return new EagerInference(FrozenNetwork.load(model, device));
  }

  @Override
  public Device device() {
    return network.device();
  }

  /** 1 回の仕事の配列を付ける管理元。 */
  NDManager newScope() {
    return network.newScope();
  }

  /** 自動微分を記録しない区間（順伝播と出力の整形をこの中で行う）。 */
  InferenceMode inference() {
    return network.inference();
  }

  /** 1 回の仕事の送り出しの置き場を借りる（池が空なら網に付けて作る）。{@link #releaseUploads} で返す。 */
  Uploads acquireUploads() {
    Uploads borrowed = uploads.poll();
    return borrowed == null ? network.newUploads() : borrowed;
  }

  /** 借りた置き場を池に返す（送った区画の写しは同期で終わっているので、次の仕事がすぐ使える）。 */
  void releaseUploads(Uploads borrowed) {
    uploads.add(borrowed);
  }

  /**
   * 1 つの塊の s を出す。順伝播の配列は入力の管理元に付く。
   *
   * @param inputs 塊の入力
   * @return s [行, 手の容量]（FP32。列 k は行の k 番目の手）
   */
  NDArray scores(NetworkInputs inputs) {
    return network.scores(inputs);
  }

  @Override
  public FloatBuffer score(Encoding encoding) {
    try (EagerPass pass = new EagerPass(this, encoding)) {
      pass.upload();
      return FloatBuffer.wrap(pass.forward().toFloatArray());
    }
  }

  @Override
  public SlotWork work(Encoding encoding) {
    return new Work(encoding);
  }

  /** 実行枠の計算のストリーム（初めて使うときに作る）。 */
  private synchronized PtStream stream(DecisionDevicePipeline.Slot slot) {
    return streams.computeIfAbsent(
        slot, ignored -> ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(device()));
  }

  @Override
  public synchronized void close() {
    Throwable failure = null;
    for (PtStream stream : streams.values()) {
      try {
        stream.close();
      } catch (RuntimeException | Error problem) {
        if (failure == null) {
          failure = problem;
        } else {
          failure.addSuppressed(problem);
        }
      }
    }
    streams.clear();
    uploads.clear();
    try {
      network.close();
    } catch (RuntimeException | Error problem) {
      if (failure == null) {
        failure = problem;
      } else {
        failure.addSuppressed(problem);
      }
    }
    if (failure instanceof RuntimeException problem) {
      throw problem;
    }
    if (failure instanceof Error problem) {
      throw problem;
    }
  }

  /** 実行枠の 1 回の受け付け: 入力を H2D のストリームで作り、順伝播を実行枠の計算のストリームで出す。 */
  private final class Work implements SlotWork {
    private final Encoding encoding;
    private EagerPass pass;
    private PtStream stream;
    private boolean h2dStarted;
    private boolean h2dRecorded;
    private boolean computeStarted;
    private boolean computeRecorded;

    Work(Encoding encoding) {
      this.encoding = encoding;
    }

    @Override
    public void submit(DecisionDevicePipeline.Slot slot) {
      try {
        h2dStarted = true;
        try (var ignored = slot.h2dStream().openScope()) {
          pass = new EagerPass(EagerInference.this, encoding);
          pass.upload();
          slot.h2dReady().record();
          h2dRecorded = true;
        }
        computeStarted = true;
        stream = stream(slot);
        try (var ignored = stream.openScope()) {
          slot.h2dReady().waitOnStream();
          NDArray output = pass.forward();
          slot.computeReady().record();
          computeRecorded = true;
          slot.prepareOutput((PtNDArray) output);
        }
      } catch (RuntimeException | Error problem) {
        recordLatestCompletion(slot, problem);
        throw problem;
      }
    }

    /** 失敗した提出でも、積んだ転送と順伝播の後に印を残す（回収で待てるように）。 */
    private void recordLatestCompletion(DecisionDevicePipeline.Slot slot, Throwable problem) {
      if (h2dStarted && !h2dRecorded) {
        try (var ignored = slot.h2dStream().openScope()) {
          slot.h2dReady().record();
          h2dRecorded = true;
        } catch (Throwable recordingFailure) {
          problem.addSuppressed(recordingFailure);
        }
      }
      if (h2dRecorded && !computeRecorded) {
        try (var ignored = (stream == null ? stream(slot) : stream).openScope()) {
          slot.h2dReady().waitOnStream();
          slot.computeReady().record();
          computeRecorded = true;
        } catch (Throwable recordingFailure) {
          problem.addSuppressed(recordingFailure);
        }
      }
    }

    @Override
    public boolean settle(DecisionDevicePipeline.Slot slot, Throwable problem) {
      if ((h2dStarted && !h2dRecorded) || (computeStarted && !computeRecorded)) {
        problem.addSuppressed(
            new IllegalStateException("PRISM stream completion could not be recorded"));
        return false;
      }
      try {
        if (computeRecorded) {
          slot.computeReady().synchronize();
        } else if (h2dRecorded) {
          slot.h2dReady().synchronize();
        }
        return true;
      } catch (Throwable synchronizationFailure) {
        problem.addSuppressed(synchronizationFailure);
        return false;
      }
    }

    @Override
    public FloatBuffer scores(DecisionDevicePipeline.Slot slot) {
      return slot.outputFloats();
    }

    @Override
    public void close() {
      if (pass != null) {
        pass.close();
        pass = null;
      }
    }
  }
}
