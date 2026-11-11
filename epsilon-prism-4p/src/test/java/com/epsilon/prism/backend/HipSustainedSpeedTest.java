package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism4p.input.InputSchema;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 本番の負荷をかけ続けたときの推論の速さを測る（rocm のグループ。見える GPU を全て同時に使い、GPU ごとに測る）。
 *
 * <p>GPU ごとに {@link #LANES} 本のレーン（計算ストリームと計画、ページ固定の blob と s のバッファ）を作る。それぞれのスレッドは、バッチの
 * 順伝播を積んでは待つことを休みなく続ける。電力の上限でクロックが下がった状態を測るため、最初の {@link #WARM_SECONDS} 秒は捨てる。続く
 * {@link #MEASURE_SECONDS} 秒に終えたバッチの判断の数から、1 判断あたりの経過時間（区間の長さ / 判断の数）を出す。その後も負荷を切らずに
 * {@link #PROFILE_SECONDS} 秒だけカーネルの計時を入れ、カーネルごとの時間の割合で経過時間を分ける。ストリームが重なるので、計時の印の間には
 * 他のストリームのカーネルも入る。そのため割合だけを使う。区間の端をまたぐバッチはレーンごとに高々 1 つなので、区間の判断の数の誤差は
 * 3 バッチ分以下になる。
 *
 * <p>ネットワークの形はシステムプロパティ {@value #ARM_PROPERTY} で選ぶ（{@link HipFixtures.Arm} の名前。既定は v2 の本番の形）。
 * バッチは模擬の半荘の 2048 行で、全てのレーンが同じバッチを回す。測った区間の両端をエポックからのミリ秒でログに書くので、同じ時刻のクロック・電力・温度のサンプル（テストの外で sysfs を読んで取る）から
 * 1 判断あたりのジュールを出せる。ログの行の先頭は「PRISM-HIP-SUSTAINED」。
 */
public class HipSustainedSpeedTest {
  private static final Logger log = LoggerFactory.getLogger(HipSustainedSpeedTest.class);

  /**
   * GPU ごとのレーンの数（システムプロパティ {@code prism.lanes}。既定は本番のスロットの数 3）。レーンは他のストリームより先に GPU ごとに
   * 作るので、{@code GPU_MAX_HW_QUEUES}（既定 4）以下なら、各レーンが専用の HW キューを持つ。
   */
  private static final int LANES = Integer.getInteger("prism.lanes", 3);

  private static final String ARM_PROPERTY = "prism.arm";
  private static final int WARM_SECONDS = 10;
  private static final int MEASURE_SECONDS = 30;
  private static final int PROFILE_SECONDS = 5;
  private static final int SPEED_ROWS = 2048;

  /** レーンが回す 1 つのバッチ（blob はページ固定）。 */
  private record Batch(ByteBuffer blob, HipBatch batch) {}

  /** 区間の両端の時刻と、計時や終了の指示（メインスレッドが書き、レーンのスレッドが読む）。 */
  private static final class Phases {
    volatile long from = Long.MAX_VALUE;
    volatile long to = Long.MAX_VALUE;
    volatile boolean profiling;
    volatile boolean finished;

    /** バッチを終えた時刻（System.nanoTime）が、測る区間に入るか。 */
    boolean measured(long t) {
      return t >= from && t < to;
    }
  }

  /** 1 本のレーンと、そのスレッドの集計。 */
  private static final class Lane implements Runnable, AutoCloseable {
    final int gpu;
    final PtStream stream;
    final HipBackend backend;
    final ByteBuffer out;
    final Batch batch;
    final Phases phases;
    final long version;
    long measured;
    long profiled;
    long profiledNanos;
    final double[] kernelMillis = new double[HipKernel.values().length];
    Throwable failure;

    Lane(int gpu, WeightPack pack, BatchPlan.Capacity capacity, Batch batch, ByteBuffer out, Phases phases) {
      this.gpu = gpu;
      this.batch = batch;
      this.out = out;
      this.phases = phases;
      version = pack.version();
      stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(gpu));
      HipBackend opened = null;
      try {
        opened = HipBackend.open(HipLibrary.load(), stream, pack.layout(), capacity);
        opened.publish(pack);
      } catch (RuntimeException | Error problem) {
        if (opened != null) {
          opened.close();
        }
        stream.close();
        throw problem;
      }
      backend = opened;
    }

    @Override
    public void run() {
      try {
        long profileStart = 0;
        while (!phases.finished) {
          if (profileStart == 0 && phases.profiling) {
            backend.plan().profile(true);
            profileStart = System.nanoTime();
          }
          backend.plan().forward(batch.blob().duplicate().position(0), batch.batch(), out.duplicate().position(0), version);
          backend.synchronize();
          long done = System.nanoTime();
          int rows = batch.batch().rows();
          if (phases.measured(done)) {
            measured += rows;
          }
          if (profileStart != 0) {
            float[] kernels = backend.plan().kernelMillis();
            for (HipKernel kernel : HipKernel.values()) {
              if (!Float.isNaN(kernels[kernel.ordinal()])) {
                kernelMillis[kernel.ordinal()] += kernels[kernel.ordinal()];
              }
            }
            profiled += rows;
            profiledNanos = done - profileStart;
          }
        }
      } catch (Throwable problem) {
        failure = problem;
      }
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

  /** 選んだ形のネットワークを、GPU ごとに {@link #LANES} 本のストリームで回し続けたときの 1 判断あたりの時間を測る。 */
  @Test(groups = "rocm", timeOut = 600_000)
  public void measureSustained() throws Exception {
    int gpus = Engine.getInstance().getGpuCount();
    Assert.assertTrue(gpus > 0, "no visible GPU");
    Assert.assertTrue(LANES > 0, "prism.lanes must be positive: " + LANES);
    HipFixtures.Arm arm = HipFixtures.Arm.valueOf(System.getProperty(ARM_PROPERTY, HipFixtures.Arm.V2_DEPTH_2.name()));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtNDManager pinned = (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME)) {
      PrismNetwork network = HipFixtures.network(manager, arm.shape(), 72);
      WeightPack pack = WeightPack.build(WeightLayout.of(network), network, 5);
      HostBatch encoded =
          HipFixtures.encode(
              HipFixtures.simulatedRows(SPEED_ROWS, 800),
              bytes -> pinned.allocatePinned(bytes, DataType.UINT8).getByteBuffer());
      Batch batch = new Batch(encoded.blob(), HipBackend.batchOf(encoded));
      BatchPlan.Capacity capacity = BatchPlan.Capacity.standard(InputSchema.FORMAT);
      Phases phases = new Phases();
      List<Lane> lanes = new ArrayList<>();
      long fromEpoch;
      long toEpoch;
      try {
        for (int gpu = 0; gpu < gpus; gpu++) {
          for (int lane = 0; lane < LANES; lane++) {
            ByteBuffer out =
                pinned
                    .allocatePinned(batch.batch().actions(), DataType.FLOAT32)
                    .getByteBuffer()
                    .order(ByteOrder.LITTLE_ENDIAN);
            lanes.add(new Lane(gpu, pack, capacity, batch, out, phases));
          }
        }
        List<Thread> threads = new ArrayList<>();
        for (Lane lane : lanes) {
          Thread thread = new Thread(lane, "sustained-gpu" + lane.gpu + "-" + threads.size());
          threads.add(thread);
          thread.start();
        }
        Thread.sleep(WARM_SECONDS * 1000L);
        fromEpoch = System.currentTimeMillis();
        phases.from = System.nanoTime();
        Thread.sleep(MEASURE_SECONDS * 1000L);
        toEpoch = System.currentTimeMillis();
        phases.to = System.nanoTime();
        phases.profiling = true;
        Thread.sleep(PROFILE_SECONDS * 1000L);
        phases.finished = true;
        for (Thread thread : threads) {
          thread.join();
        }
      } finally {
        phases.finished = true;
        for (Lane lane : lanes) {
          lane.close();
        }
      }
      for (Lane lane : lanes) {
        if (lane.failure != null) {
          throw new AssertionError("lane on GPU " + lane.gpu + " failed", lane.failure);
        }
      }
      double seconds = (phases.to - phases.from) / 1e9;
      for (int gpu = 0; gpu < gpus; gpu++) {
        report(gpu, lanes, arm.label(), seconds, fromEpoch, toEpoch);
      }
    }
  }

  /**
   * GPU ごとに 1 行をログに書く。中身は、区間に終えた判断の数、1 判断あたりの経過時間、それをカーネルの割合で分けた時間、計時を入れていた間の
   * 1 判断あたりの経過時間。
   */
  private static void report(int gpu, List<Lane> lanes, String arm, double seconds, long fromEpoch, long toEpoch) {
    long measured = 0;
    long profiled = 0;
    double profiledSeconds = 0;
    double[] kernels = new double[HipKernel.values().length];
    int count = 0;
    for (Lane lane : lanes) {
      if (lane.gpu != gpu) {
        continue;
      }
      measured += lane.measured;
      profiled += lane.profiled;
      profiledSeconds += lane.profiledNanos / 1e9;
      for (int k = 0; k < kernels.length; k++) {
        kernels[k] += lane.kernelMillis[k];
      }
      count++;
    }
    Assert.assertTrue(measured > 0, "GPU " + gpu + " finished no batch in the measured window");
    Assert.assertTrue(profiled > 0, "GPU " + gpu + " finished no profiled batch");
    double wall = seconds * 1e6 / measured;
    double spans = 0;
    for (double millis : kernels) {
      spans += millis;
    }
    StringBuilder line = new StringBuilder();
    for (HipKernel kernel : HipKernel.values()) {
      if (kernels[kernel.ordinal()] > 0) {
        line.append(String.format(Locale.ROOT, " %s=%.3fus", kernel, wall * kernels[kernel.ordinal()] / spans));
      }
    }
    // 計時を入れていた区間の長さをレーンで平均し、その間に終えた判断の数で割る
    double profiledWall = profiledSeconds / count * 1e6 / profiled;
    log.info(
        "PRISM-HIP-SUSTAINED gpu {} lanes {} arm {}: measured {}..{} epochMs"
            + " decisions {} wall={} us/decision; by kernel share:{}; profiled wall={} us/decision",
        gpu,
        count,
        arm,
        fromEpoch,
        toEpoch,
        measured,
        String.format(Locale.ROOT, "%.4f", wall),
        line,
        String.format(Locale.ROOT, "%.4f", profiledWall));
  }
}
