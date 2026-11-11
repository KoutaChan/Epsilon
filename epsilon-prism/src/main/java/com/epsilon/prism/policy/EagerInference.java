package com.epsilon.prism.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.DecisionDevicePipeline;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.model.FrozenNetwork;
import com.epsilon.prism.model.NetworkInputs;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * eager で推論するバックエンド（デバイス 1 つ分）。保存したネットワークを凍結して持ち（{@link FrozenNetwork}。計算の精度は開くときに
 * 選ぶ）、スロットで行うジョブを作る。
 *
 * <p>スロットでは、入力を共通の H2D ストリームで作り、順伝播をスロットごとの計算ストリームで出す。ネットワークのインデックスの転送は
 * 同期するので、こうするとほかのスロットの計算を待たずに済む。出力の D2H は共通のパイプラインが受け持つ（{@code Slot.prepareOutput}）。
 * スロットの計算ストリームは、提出のスレッドだけが作って使う。
 *
 * <p>1 回のジョブは、インデックスを送るバッファ（{@link PinnedUploads}）をプールから借り、終わったら返す。呼び出し元のスレッドがいくつ
 * あっても、1 つのバッファを同時に使うのは 1 つのジョブだけ。
 */
final class EagerInference implements DeviceInference {

  private final FrozenNetwork network;
  private final Map<DecisionDevicePipeline.Slot, PtStream> streams = new IdentityHashMap<>();
  private final ConcurrentLinkedQueue<PinnedUploads> uploads = new ConcurrentLinkedQueue<>();

  private EagerInference(FrozenNetwork network) {
    this.network = network;
  }

  /**
   * 保存したネットワークを読む。
   *
   * @param model ネットワークのディレクトリ（weights.bin と model.json）
   * @param device デバイス
   * @param precision 順伝播の計算の精度
   * @return 推論のバックエンド
   * @throws IOException 読めないか、形式・フィンガープリント・重みが合わないとき
   */
  static EagerInference load(Path model, Device device, Precision precision) throws IOException {
    return new EagerInference(FrozenNetwork.load(model, device, precision));
  }

  @Override
  public Device device() {
    return network.device();
  }

  /** 1 回のジョブで作る配列を持たせるマネージャーを作る。 */
  NDManager newScope() {
    return network.newScope();
  }

  /** 自動微分を記録しない区間。順伝播と出力の整形はこの中で行う。 */
  InferenceMode inference() {
    return network.inference();
  }

  /** 1 回のジョブで送り出しに使うバッファを借りる（プールが空なら、ネットワークに付けて作る）。{@link #releaseUploads} で返す。 */
  PinnedUploads acquireUploads() {
    PinnedUploads borrowed = uploads.poll();
    return borrowed == null ? network.newUploads() : borrowed;
  }

  /** 借りたバッファをプールに返す。送った領域のコピーは同期して終わっているので、次のジョブがすぐに使える。 */
  void releaseUploads(PinnedUploads borrowed) {
    uploads.add(borrowed);
  }

  /**
   * 1 つのバッチの s を出す。順伝播の配列は入力のマネージャーに付く。
   *
   * @param inputs バッチの入力
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

  /** スロットの計算ストリームを返す。初めて使うときに作る。 */
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

  /** スロットで行う 1 回の受付。入力を H2D ストリームで作り、順伝播をスロットの計算ストリームで出す。 */
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

    /** 提出が途中で失敗しても、積んだ転送と順伝播の後に印を記録する。回収のときに待てるようにするため。 */
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
