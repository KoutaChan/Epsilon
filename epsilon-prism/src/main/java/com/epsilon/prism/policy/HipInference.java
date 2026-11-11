package com.epsilon.prism.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.DecisionDevicePipeline;
import com.epsilon.prism.backend.HipBackend;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * HIP で推論するバックエンド（デバイス 1 つ分）。ネットワークの重み（{@link WeightPack}）をスロットごとのレーン（{@link Lane}: 計算
 * ストリームと実行計画）にコピーし、バッチごとに blob の H2D 1 回 → K1〜K7 → s の D2H 1 回を積む。s はレーンごとのページ固定のバッファに、
 * バッチの順に、バッチの中では手の順に詰める。
 *
 * <p>レーンは開くときにスロットの数だけ作り、重みをコピーする。作業領域の確保とコピーの同期は、この準備のときだけ行う。スロットには、初めて
 * 使うときにレーンを 1 つずつ割り当てる。スロットごとにストリームと作業領域が別なので、一方の転送ともう一方の順伝播が重なる。順伝播には毎回
 * 方策のバージョン番号を渡し、レーンが持つ重みのバージョンと照合する。これで古い重みのまま推論することを防ぐ。
 */
final class HipInference implements DeviceInference {

  /**
   * スロット 1 つ分の HIP の順伝播。本番では GPU の計算ストリームと実行計画を使い、テストでは CPU の参照実装を使う。スレッドセーフではない。
   */
  interface Lane extends AutoCloseable {

    /**
     * バッチ 1 つの順伝播を積む: blob の H2D → K1〜K7 → s の D2H。
     *
     * @param batch 確定したバッチ（blob はページ固定の direct ByteBuffer）
     * @param out s を書く先（位置 0 から、バッチの手の数だけの FP32 を書く。little-endian の direct ByteBuffer）
     * @param version 期待するネットワークのバージョン（コピーした重みのバージョンと違えば例外）
     */
    void submit(HostBatch batch, ByteBuffer out, long version);

    /**
     * 積んだ処理の後にイベントを記録する。
     *
     * @param event 記録するイベント
     */
    void record(PtEvent event);

    /** 積んだ処理が終わるのを待つ。 */
    void synchronize();

    @Override
    void close();
  }

  /** 重みをコピーしたレーンを作るファクトリ。 */
  @FunctionalInterface
  interface Lanes {

    /**
     * @param device デバイス
     * @param pack コピーする重み
     * @return 重みをコピーしたレーン
     */
    Lane open(Device device, WeightPack pack);
  }

  private final Device device;
  private final long version;
  private final int capacity;
  private final NDManager pinned;
  private final Lane[] lanes;
  private final ByteBuffer[] outputs;
  private final FloatBuffer[] scores;
  private final Map<DecisionDevicePipeline.Slot, Integer> bound = new IdentityHashMap<>();

  private HipInference(
      Device device, long version, int capacity, NDManager pinned, Lane[] lanes, ByteBuffer[] outputs) {
    this.device = device;
    this.version = version;
    this.capacity = capacity;
    this.pinned = pinned;
    this.lanes = lanes;
    this.outputs = outputs;
    scores = new FloatBuffer[outputs.length];
    for (int lane = 0; lane < outputs.length; lane++) {
      scores[lane] = outputs[lane].asFloatBuffer();
    }
  }

  /**
   * レーンを作り、重みをコピーする。
   *
   * @param device デバイス
   * @param pack 重み（バージョン番号は方策のバージョン）
   * @param count レーンの数（GPU ではスロットの数。呼び出しスレッドから直接呼ぶだけなら 1）
   * @param capacity 1 回の受付で返す s の数の上限
   * @param lanes レーンを作るファクトリ
   * @return HIP の推論のバックエンド
   */
  static HipInference open(Device device, WeightPack pack, int count, int capacity, Lanes lanes) {
    if (count < 1 || capacity < 1) {
      throw new IllegalArgumentException("HIP inference needs lanes and scores: " + count + ", " + capacity);
    }
    PtNDManager pinned = (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME);
    Lane[] opened = new Lane[count];
    ByteBuffer[] outputs = new ByteBuffer[count];
    try {
      for (int lane = 0; lane < count; lane++) {
        outputs[lane] =
            pinned.allocatePinned(capacity, DataType.FLOAT32).getByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        opened[lane] = lanes.open(device, pack);
      }
      return new HipInference(device, pack.version(), capacity, pinned, opened, outputs);
    } catch (RuntimeException | Error failure) {
      for (Lane lane : opened) {
        if (lane != null) {
          try {
            lane.close();
          } catch (RuntimeException | Error problem) {
            failure.addSuppressed(problem);
          }
        }
      }
      try {
        pinned.close();
      } catch (RuntimeException | Error problem) {
        failure.addSuppressed(problem);
      }
      throw failure;
    }
  }

  /**
   * GPU のレーンを作るファクトリを返す。libprism_hip.so を読み込む。レーンごとに計算ストリームを作り、その上に実行計画を作って重みをコピーする。
   *
   * @param capacity バッチの容量（入力を符号化するときのバッチと同じ）
   * @return レーンを作るファクトリ
   * @throws IllegalStateException libprism_hip.so が読めないか、DJL の libtorch や HIP のランタイムと合わないとき
   */
  static Lanes gpu(BatchPlan.Capacity capacity) {
    HipLibrary library;
    try {
      library = HipLibrary.load();
    } catch (RuntimeException | LinkageError failure) {
      throw new IllegalStateException(
          "inference = HIP, but libprism_hip.so cannot be used"
              + " (build epsilon-prism-hip with -Pnative-hip on the server): "
              + failure.getMessage(),
          failure);
    }
    PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
    return (device, pack) -> GpuLane.open(library, engine, device, pack, capacity);
  }

  @Override
  public Device device() {
    return device;
  }

  /** 呼び出しスレッドから直接呼ぶときは最初のレーンを使い、呼び出しを 1 つずつ通す。 */
  @Override
  public synchronized FloatBuffer score(Encoding encoding) {
    enqueue(0, encoding);
    lanes[0].synchronize();
    float[] copy = new float[encoding.actions()];
    scores[0].get(0, copy);
    return FloatBuffer.wrap(copy);
  }

  @Override
  public SlotWork work(Encoding encoding) {
    return new Work(encoding);
  }

  /** バッチごとの順伝播をレーンに積む。s はバッチの順に詰める。 */
  private void enqueue(int lane, Encoding encoding) {
    int offset = 0;
    for (Encoding.Chunk chunk : encoding.chunks()) {
      HostBatch batch = chunk.batch().input();
      int actions = batch.plan().count(Axis.ACTION);
      if (offset + actions > capacity) {
        throw new IllegalStateException("a policy batch holds more than " + capacity + " actions");
      }
      ByteBuffer out =
          outputs[lane].slice(offset * Float.BYTES, actions * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      lanes[lane].submit(batch, out, version);
      offset += actions;
    }
  }

  /** スロットに割り当てたレーンの番号を返す。初めてのスロットには、まだ使っていない次のレーンを割り当てる。 */
  private synchronized int bind(DecisionDevicePipeline.Slot slot) {
    Integer lane = bound.get(slot);
    if (lane == null) {
      if (bound.size() == lanes.length) {
        throw new IllegalStateException(
            "PRISM HIP inference has " + lanes.length + " lanes for more device slots");
      }
      lane = bound.size();
      bound.put(slot, lane);
    }
    return lane;
  }

  /** レーンとページ固定のバッファを解放する。レーンは実行計画を先に、そのストリームを後に閉じる。 */
  @Override
  public void close() {
    Throwable failure = null;
    for (Lane lane : lanes) {
      try {
        lane.close();
      } catch (RuntimeException | Error problem) {
        if (failure == null) {
          failure = problem;
        } else {
          failure.addSuppressed(problem);
        }
      }
    }
    try {
      pinned.close();
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

  /** スロットで行う 1 回の受付。バッチごとの順伝播を、スロットに割り当てたレーンのストリームに積む（H2D も同じストリーム）。 */
  private final class Work implements SlotWork {
    private final Encoding encoding;
    private int lane = -1;
    private boolean recorded;

    Work(Encoding encoding) {
      this.encoding = encoding;
    }

    @Override
    public void submit(DecisionDevicePipeline.Slot slot) {
      lane = bind(slot);
      try {
        enqueue(lane, encoding);
      } catch (RuntimeException | Error problem) {
        // 前のバッチの処理はもう積まれているので、その後にイベントを記録してから投げる
        try {
          lanes[lane].record(slot.computeReady());
          recorded = true;
        } catch (RuntimeException | Error recordingFailure) {
          problem.addSuppressed(recordingFailure);
        }
        throw problem;
      }
      lanes[lane].record(slot.computeReady());
      recorded = true;
    }

    @Override
    public boolean settle(DecisionDevicePipeline.Slot slot, Throwable problem) {
      if (lane < 0) {
        return true;
      }
      if (!recorded) {
        problem.addSuppressed(
            new IllegalStateException("PRISM HIP stream completion could not be recorded"));
        return false;
      }
      try {
        slot.computeReady().synchronize();
        return true;
      } catch (Throwable synchronizationFailure) {
        problem.addSuppressed(synchronizationFailure);
        return false;
      }
    }

    @Override
    public FloatBuffer scores(DecisionDevicePipeline.Slot slot) {
      return scores[lane];
    }

    /** レーンはスロットに割り当てたまま使い回すので、受付ごとに解放するものはない。 */
    @Override
    public void close() {}
  }

  /** GPU のレーン。自分の計算ストリームと、その上の実行計画（作業領域と重みのコピー）を持つ。閉じるときは実行計画が先。 */
  private record GpuLane(PtStream stream, HipBackend backend) implements Lane {

    static GpuLane open(
        HipLibrary library, PtEngine engine, Device device, WeightPack pack, BatchPlan.Capacity capacity) {
      PtStream stream = engine.newStream(device);
      try {
        HipBackend backend = HipBackend.open(library, stream, pack.layout(), capacity);
        try {
          backend.publish(pack);
          return new GpuLane(stream, backend);
        } catch (RuntimeException | Error failure) {
          try {
            backend.close();
          } catch (RuntimeException | Error problem) {
            failure.addSuppressed(problem);
          }
          throw failure;
        }
      } catch (RuntimeException | Error failure) {
        try {
          stream.close();
        } catch (RuntimeException | Error problem) {
          failure.addSuppressed(problem);
        }
        throw failure;
      }
    }

    @Override
    public void submit(HostBatch batch, ByteBuffer out, long version) {
      backend.submit(batch, out, version);
    }

    @Override
    public void record(PtEvent event) {
      try (var ignored = stream.openScope()) {
        event.record();
      }
    }

    @Override
    public void synchronize() {
      backend.synchronize();
    }

    @Override
    public void close() {
      try {
        backend.close();
      } finally {
        stream.close();
      }
    }
  }
}
