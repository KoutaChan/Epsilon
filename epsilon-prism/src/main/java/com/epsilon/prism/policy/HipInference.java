package com.epsilon.prism.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.backend.HipBackend;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.runtime.DecisionDevicePipeline;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * HIP の推論の道（1 つのデバイス）。網の重みの束（{@link WeightPack}）を実行枠ごとの道具（{@link Lane}: 計算ストリームと計画）に写し、
 * 塊ごとに blob の H2D 1 回 → K1〜K7 → s の D2H 1 回を積む。s は道具ごとのページ固定の置き場に、塊の順に塊の手の順で詰める。
 *
 * <p>道具は開くときに実行枠の数だけ作って重みを写し（作業域の確保と写しの同期は準備のときだけ）、実行枠には初めて使うときに 1 つずつ結ぶ。実行枠
 * ごとにストリームと作業域が別なので、片方の転送ともう片方の順伝播が重なる。順伝播には毎回方策の版の番号を渡し、道具の写しの版と照らす（古い写し
 * が残る穴を塞ぐ）。
 */
final class HipInference implements DeviceInference {

  /** 1 つの実行枠の HIP の順伝播（本番は GPU の計算ストリームと計画、試験は CPU の参照）。スレッド安全でない。 */
  interface Lane extends AutoCloseable {

    /**
     * 1 塊の順伝播を積む: blob の H2D → K1〜K7 → s の D2H。
     *
     * @param batch 封じた塊（blob はページ固定の直の ByteBuffer）
     * @param out s を書く先（位置 0 から塊の手の数の FP32、little-endian の直の ByteBuffer）
     * @param version 期待する網の版（写した版と違えば例外）
     */
    void submit(HostBatch batch, ByteBuffer out, long version);

    /**
     * 積んだ仕事の後に印を記録する。
     *
     * @param event 印
     */
    void record(PtEvent event);

    /** 積んだ仕事が終わるのを待つ。 */
    void synchronize();

    @Override
    void close();
  }

  /** 重みの束を写した道具を作る口。 */
  @FunctionalInterface
  interface Lanes {

    /**
     * @param device デバイス
     * @param pack 写す重みの束
     * @return 重みを写した道具
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
   * 道具を作り、重みを写す。
   *
   * @param device デバイス
   * @param pack 重みの束（版の番号は方策の版）
   * @param count 道具の数（GPU は実行枠の数。呼び出しスレッドの口だけなら 1）
   * @param capacity 1 回の受け付けの s の数の上限
   * @param lanes 道具を作る口
   * @return 推論の道
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
   * GPU の計画の道具を作る口。libprism_hip.so を読み込み、道具ごとに計算ストリームを作って、その上に計画を作り重みを写す。
   *
   * @param capacity 塊の容量（入力の符号化の塊と同じ）
   * @return 口
   * @throws IllegalStateException libprism_hip.so が読めないか、DJL の libtorch・HIP の実行系と合わないとき
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

  /** 呼び出しスレッドの口は最初の道具を使う（呼び出しを 1 つずつ通す）。 */
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

  /** 塊ごとの順伝播を道具に積む（s は塊の順に詰める）。 */
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

  /** 実行枠に結んだ道具の番号（初めての実行枠には次の道具を結ぶ）。 */
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

  /** 道具（計画が先、そのストリームが後）とページ固定の置き場を返す。 */
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

  /** 実行枠の 1 回の受け付け: 塊ごとの順伝播を、実行枠に結んだ道具のストリームに積む（H2D も同じストリーム）。 */
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
        // 先の塊の仕事は積まれているので、その後に印を残してから投げる
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

    /** 道具は実行枠に結んだまま使い回すので、受け付けごとに返すものはない。 */
    @Override
    public void close() {}
  }

  /** GPU の実行枠の道具: 自分の計算ストリームと、その上の計画（作業域と重みの写し）。閉じるのは計画が先。 */
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
