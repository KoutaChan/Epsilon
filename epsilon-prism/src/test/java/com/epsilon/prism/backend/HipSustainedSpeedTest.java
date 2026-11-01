package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.BatchShape;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
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
 * 本番の負荷での推論の速さ（rocm の組。見える GPU を全て同時に使い、GPU ごとに測る。1 枚だけなら HIP_VISIBLE_DEVICES で絞る）。
 *
 * <p>GPU ごとに本番の実行枠と同じ {@link #LANES} 本の道具（計算ストリームと計画、ページ固定の blob と s の置き場）を作り、それぞれのスレッドが
 * 塊の順伝播を積んでは待つことを休みなく続ける。電力の上限で時計が下がった状態を測るため、最初の {@link #WARM_SECONDS} 秒は捨て、続く
 * {@link #MEASURE_SECONDS} 秒に終えた塊の判断の数から 1 判断あたりの壁の時間（区間の長さ / 判断）を出す。その後も負荷を切らずに
 * {@link #PROFILE_SECONDS} 秒カーネルの計時を入れ、カーネルごとの時間の割合で壁の時間を分ける（ストリームが重なるので、計時の印の間には他の
 * ストリームのカーネルも入る。だから割合だけを使う）。区間の端をまたぐ塊は道具ごとに高々 1 つで、区間の判断の数の誤差は 3 塊ぶん以下。
 *
 * <p>塊は {@link HipFixtures#speedBatches}（書き出した本番の塊か、模擬の半荘の 2048 行）で、道具ごとにずらして順に回す。測った区間の両端を
 * エポックのミリ秒で記録に書くので、同じ時刻の時計・電力・温度の標本（試験の外で sysfs を読む標本取り）から 1 判断あたりのジュールを出せる。記録の行の頭は
 * 「PRISM-HIP-SUSTAINED」。
 */
public class HipSustainedSpeedTest {
  private static final Logger log = LoggerFactory.getLogger(HipSustainedSpeedTest.class);

  private static final int LANES = 3;
  private static final int WARM_SECONDS = 10;
  private static final int MEASURE_SECONDS = 30;
  private static final int PROFILE_SECONDS = 5;
  private static final int SPEED_ROWS = 2048;
  private static final long VERSION = 5;

  /** 区間の境（主スレッドが書き、道具のスレッドが読む）。 */
  private static final class Phases {
    volatile long from = Long.MAX_VALUE;
    volatile long to = Long.MAX_VALUE;
    volatile boolean profiling;
    volatile boolean finished;

    /** 塊を終えた時刻 t（System.nanoTime）が測る区間にあるか。 */
    boolean measured(long t) {
      return t >= from && t < to;
    }
  }

  /** 1 本の道具とそのスレッドの集計。 */
  private static final class Lane implements Runnable, AutoCloseable {
    final int gpu;
    final PtStream stream;
    final HipBackend backend;
    final ByteBuffer out;
    final List<HostBatch> batches;
    final Phases phases;
    int next;
    long measured;
    long profiled;
    long profiledNanos;
    final double[] kernelMillis = new double[HipKernel.values().length];
    Throwable failure;

    Lane(int gpu, int first, WeightPack pack, List<HostBatch> batches, ByteBuffer out, Phases phases) {
      this.gpu = gpu;
      this.batches = batches;
      this.out = out;
      this.phases = phases;
      next = first;
      stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(gpu));
      HipBackend opened = null;
      try {
        opened = HipBackend.open(HipLibrary.load(), stream, pack.layout(), BatchPlan.Capacity.DEFAULT);
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
          HostBatch batch = batches.get(next++ % batches.size());
          backend.submit(batch, out, VERSION);
          backend.synchronize();
          long done = System.nanoTime();
          int rows = batch.plan().rows();
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

  /** 本番の形（深さ {@link PrismShape#DEFAULT_DEPTH}）の網を GPU ごとに 3 本のストリームで回し続けたときの 1 判断あたりの時間。 */
  @Test(groups = "rocm", timeOut = 600_000)
  public void measureSustained() throws Exception {
    int gpus = Engine.getInstance().getGpuCount();
    Assert.assertTrue(gpus > 0, "no visible GPU");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtNDManager pinned = (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME)) {
      PrismNetwork network = HipFixtures.network(manager, PrismShape.DEFAULT_DEPTH, 72);
      WeightLayout layout = WeightLayout.of(network);
      WeightPack pack = WeightPack.build(layout, network, VERSION);
      List<HostBatch> batches = new ArrayList<>();
      BatchShape shape = new BatchShape();
      for (HipFixtures.SpeedBatch source : HipFixtures.speedBatches(SPEED_ROWS, 800)) {
        HostBatch batch =
            HipFixtures.encode(
                source.rows(), bytes -> pinned.allocatePinned(bytes, DataType.UINT8).getByteBuffer());
        batches.add(batch);
        shape.add(batch.plan());
      }
      Phases phases = new Phases();
      List<Lane> lanes = new ArrayList<>();
      long fromEpoch;
      long toEpoch;
      try {
        for (int gpu = 0; gpu < gpus; gpu++) {
          for (int lane = 0; lane < LANES; lane++) {
            ByteBuffer out =
                pinned
                    .allocatePinned(BatchPlan.Capacity.DEFAULT.actions(), DataType.FLOAT32)
                    .getByteBuffer()
                    .order(ByteOrder.LITTLE_ENDIAN);
            lanes.add(new Lane(gpu, lane, pack, batches, out, phases));
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
        report(gpu, lanes, shape, seconds, fromEpoch, toEpoch);
      }
    }
  }

  /** GPU ごとの 1 行: 区間の判断・壁の時間 / 判断・カーネルの割合で分けた時間 / 判断・計時を入れた間の壁の時間 / 判断。 */
  private static void report(
      int gpu, List<Lane> lanes, BatchShape shape, double seconds, long fromEpoch, long toEpoch) {
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
      line.append(
          String.format(Locale.ROOT, " %s=%.3fus", kernel, wall * kernels[kernel.ordinal()] / spans));
    }
    // 道具ごとの計時の区間の長さの平均を、その間に終えた判断で割る
    double profiledWall = profiledSeconds / count * 1e6 / profiled;
    log.info(
        "PRISM-HIP-SUSTAINED gpu {} lanes {} depth {} batches {} {}: measured {}..{} epochMs"
            + " decisions {} wall={} us/decision; by kernel share:{}; profiled wall={} us/decision",
        gpu,
        count,
        PrismShape.DEFAULT_DEPTH,
        shape.summary().batches(),
        shape.summary().line(),
        fromEpoch,
        toEpoch,
        measured,
        String.format(Locale.ROOT, "%.4f", wall),
        line,
        String.format(Locale.ROOT, "%.4f", profiledWall));
  }
}
