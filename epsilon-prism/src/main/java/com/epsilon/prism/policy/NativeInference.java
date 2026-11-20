package com.epsilon.prism.policy;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtNDManager;
import com.epsilon.agent.runtime.DecisionDevicePipeline;
import com.epsilon.agent.runtime.EventCacheSettings;
import com.epsilon.prism.backend.EventPages;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.kernel.KernelProvider.LaneCapacity;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 専用カーネルで推論するバックエンド（デバイス 1 つ分）。重みを各スロットの実行計画に公開し、バッチの入力転送からスコアの読み出しまでをレーンのストリームに積む。s はスロットごとのページ固定のバッファに、
 * バッチの順に、バッチの中では手の順に詰める。
 *
 * <p>レーンを開くときに作業領域を確保し、重みをコピーする。通常はスロットごとに別のレーンを使う。
 * イベントを保存するときは、卓の受付区分でレーンを選ぶ。同じレーンの準備順と提出順は等しい。
 * スコアの出力先と完了イベントは受付スロットが持つため、その前の受付の結果を返す前にも次の受付を積める。
 */
final class NativeInference implements DeviceInference {
  private static final Logger LOG = LoggerFactory.getLogger(NativeInference.class);

  /**
   * スロット 1 つ分の順伝播。本番では GPU の計算ストリームと実行計画を使い、テストでは CPU の参照実装を使う。スレッドセーフではない。
   */
  interface Lane extends AutoCloseable {

    /**
     * バッチ 1 つの順伝播を積む: blob の H2D、実行計画の順伝播、s の D2H。
     *
     * @param batch 確定したバッチ（blob はページ固定の direct ByteBuffer）
     * @param out s を書く先（位置 0 から、バッチの手の数だけの FP32 を書く。little-endian の direct ByteBuffer）
     * @param version 期待するネットワークのバージョン（コピーした重みのバージョンと違えば例外）
     */
    void submit(HostBatch batch, ByteBuffer out, long version);

    default void submit(HostBatch batch, ByteBuffer out, long version, EventPages pages) {
      submit(batch, out, version);
    }

    default Runnable prepare(HostBatch batch, ByteBuffer out, long version, EventPages pages) {
      return () -> submit(batch, out, version, pages);
    }

    default void invalidatePages() {}

    /**
     * レーンのイベントのページを、空きメモリを残りのレーンと分けて確保する。
     *
     * @param remaining このレーンを含め、まだ確保していないレーンの数
     * @param settings 履歴の設定
     * @return 確保したページと、確保の前後のデバイスのメモリ。ページを持たないレーンは空
     */
    default Optional<LaneCapacity> configurePages(int remaining, EventCacheSettings settings) {
      return Optional.empty();
    }

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
  private final EventCacheSettings eventCache;
  private volatile int configuredLanes;
  private final ByteBuffer[] outputs;
  private final FloatBuffer[] scores;
  private final Map<DecisionDevicePipeline.Slot, Integer> bound = new IdentityHashMap<>();

  private NativeInference(
      Device device,
      long version,
      int capacity,
      NDManager pinned,
      Lane[] lanes,
      ByteBuffer[] outputs,
      EventCacheSettings eventCache) {
    this.device = device;
    this.version = version;
    this.capacity = capacity;
    this.pinned = pinned;
    this.lanes = lanes;
    this.eventCache = eventCache;
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
   * @param eventCache 履歴の設定。保存しない方策は null
   * @return 専用カーネルの推論バックエンド
   */
  static NativeInference open(
      Device device,
      WeightPack pack,
      int count,
      int capacity,
      Lanes lanes,
      EventCacheSettings eventCache) {
    return open(
        device, pack.version(), count, capacity, () -> lanes.open(device, pack), eventCache);
  }

  static NativeInference open(
      Device device,
      long generation,
      int count,
      int capacity,
      Supplier<Lane> lanes,
      EventCacheSettings eventCache) {
    if (count < 1 || capacity < 1) {
      throw new IllegalArgumentException(
          "native inference needs lanes and scores: " + count + ", " + capacity);
    }
    PtNDManager pinned = (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME);
    Lane[] opened = null;
    try {
      ByteBuffer[] outputs = new ByteBuffer[count];
      opened =
          openLanes(
              count,
              lane -> {
                outputs[lane] =
                    pinned
                        .allocatePinned(capacity, DataType.FLOAT32)
                        .getByteBuffer()
                        .order(ByteOrder.LITTLE_ENDIAN);
                return lanes.get();
              });
      return new NativeInference(device, generation, capacity, pinned, opened, outputs, eventCache);
    } catch (RuntimeException | Error failure) {
      if (opened != null) closeAfterFailure(opened, failure);
      try {
        pinned.close();
      } catch (RuntimeException | Error problem) {
        failure.addSuppressed(problem);
      }
      throw failure;
    }
  }

  /** 全スロットへの公開を終えた世代だけを提出側に返す。 */
  static Lane[] openLanes(int count, IntFunction<Lane> publish) {
    Lane[] lanes = new Lane[count];
    try {
      for (int slot = 0; slot < count; ++slot) lanes[slot] = publish.apply(slot);
      return lanes;
    } catch (RuntimeException | Error failure) {
      closeAfterFailure(lanes, failure);
      throw failure;
    }
  }

  private static void closeAfterFailure(Lane[] lanes, Throwable failure) {
    for (Lane lane : lanes) {
      if (lane == null) continue;
      try {
        lane.close();
      } catch (RuntimeException | Error problem) {
        failure.addSuppressed(problem);
      }
    }
  }

  @Override
  public Device device() {
    return device;
  }

  /** 呼び出しスレッドから直接呼ぶときは最初のレーンを使い、呼び出しを 1 つずつ通す。 */
  @Override
  public synchronized FloatBuffer score(Encoding encoding) {
    try {
      for (Runnable command : prepare(0, 0, encoding)) command.run();
      lanes[0].synchronize();
    } catch (RuntimeException | Error failure) {
      lanes[0].invalidatePages();
      try {
        lanes[0].synchronize();
      } catch (RuntimeException | Error pending) {
        failure.addSuppressed(pending);
      }
      throw failure;
    }
    float[] copy = new float[encoding.actions()];
    scores[0].get(0, copy);
    return FloatBuffer.wrap(copy);
  }

  @Override
  public SlotWork work(Encoding encoding) {
    return new Work(encoding);
  }

  /** 順伝播を提出するための引数をバッチごとに用意する。スコアの出力先は受付内のバッチ順に並ぶ。 */
  private Runnable[] prepare(int lane, int outputSlot, Encoding encoding) {
    if (eventCache != null && configuredLanes < lanes.length) configurePages();
    Runnable[] commands = new Runnable[encoding.chunks().size()];
    int index = 0;
    int offset = 0;
    for (Encoding.Chunk chunk : encoding.chunks()) {
      HostBatch batch = chunk.batch().input();
      int actions = batch.plan().count(Axis.ACTION);
      if (offset + actions > capacity) {
        throw new IllegalStateException("a policy batch holds more than " + capacity + " actions");
      }
      ByteBuffer out =
          outputs[outputSlot]
              .slice(offset * Float.BYTES, actions * Float.BYTES)
              .order(ByteOrder.LITTLE_ENDIAN);
      commands[index++] = lanes[lane].prepare(batch, out, version, chunk.pages());
      offset += actions;
    }
    return commands;
  }

  /**
   * 最初の受付で空き容量を分ける。対戦などで一緒に開いた方策の重みと作業領域も、この時点では確保済み。確保したページと前後の空きを、
   * GPU とレーンごとに 1 回ログに書く。
   */
  private synchronized void configurePages() {
    while (configuredLanes < lanes.length) {
      int lane = configuredLanes;
      lanes[lane]
          .configurePages(lanes.length - lane, eventCache)
          .ifPresent(
              memory ->
                  LOG.info(
                      "Event cache GPU {} lane {} slots={} projectionRows={} freeBefore={}"
                          + " freeAfter={} total={} batchBytes={} slotBytes={} projectionBytes={}"
                          + " dictionary={} splitCandidate={}",
                      device.getDeviceId(),
                      lane,
                      memory.slots(),
                      memory.projectionRows(),
                      memory.freeBefore(),
                      memory.freeAfter(),
                      memory.total(),
                      memory.batchBytes(),
                      memory.slotBytes(),
                      memory.projectionBytes(),
                      memory.dictionary(),
                      memory.splitCandidate()));
      configuredLanes++;
    }
  }

  /** スロットに割り当てた出力バッファの番号を返す。通常推論では同じ番号のレーンも使う。 */
  private synchronized int bind(DecisionDevicePipeline.Slot slot) {
    Integer lane = bound.get(slot);
    if (lane == null) {
      if (bound.size() == lanes.length) {
        throw new IllegalStateException(
            "PRISM native inference has " + lanes.length + " lanes for more device slots");
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

  /** スロットで行う 1 回の受付。バッチごとの順伝播を、その受付のレーンのストリームに積む（H2D も同じストリーム）。 */
  private final class Work implements SlotWork {
    private final Encoding encoding;
    private int lane = -1;
    private int outputSlot;
    private Runnable[] commands;
    private boolean recorded;

    Work(Encoding encoding) {
      this.encoding = encoding;
    }

    @Override
    public void prepare(DecisionDevicePipeline.Slot slot) {
      outputSlot = bind(slot);
      lane =
          encoding.chunks().getFirst().pages() == null
              ? outputSlot
              : encoding.admission() % lanes.length;
      try {
        commands = NativeInference.this.prepare(lane, outputSlot, encoding);
      } catch (RuntimeException | Error failure) {
        lanes[lane].invalidatePages();
        throw failure;
      }
    }

    @Override
    public void submit(DecisionDevicePipeline.Slot slot) {
      if (commands == null) prepare(slot);
      try {
        for (Runnable command : commands) command.run();
      } catch (RuntimeException | Error problem) {
        lanes[lane].invalidatePages();
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
      lanes[lane].invalidatePages();
      if (!recorded) {
        problem.addSuppressed(
            new IllegalStateException("PRISM native stream completion could not be recorded"));
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
      encoding.completeEventPages();
      return scores[outputSlot];
    }

    /** レーンはスロットに割り当てたまま使い回すので、受付ごとに解放するものはない。 */
    @Override
    public void close() {}
  }
}
