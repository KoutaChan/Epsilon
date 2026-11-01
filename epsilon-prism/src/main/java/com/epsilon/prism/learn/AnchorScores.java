package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.network.NetworkDevices;
import com.epsilon.network.NetworkRuntime;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.rows.Row;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 当てはめの全ての行の錨の s（ホストに置く）。錨の網で当てはめの前に 1 度だけ出し（{@link #compute}）、符号化のスレッドが行の貯蔵の番号で
 * 引いて教師に混ぜる（{@link Teacher#values}）。
 *
 * <p>行の入力は生徒の微小バッチと同じ書き手（{@link Encoder.Inputs}）で書き、錨は学習のレーンと同じ道と同じ塊の容量で開く（{@link Anchor}）。
 * HIP の推論の行の s は塊の組み方に依らないので、HIP の錨の s は微小バッチの中で出した s とビットで同じ。eager の錨の s は塊の詰め物で
 * 丸めの違いが出る。
 *
 * <p>錨の網は重みの指紋（{@link PrismNetwork#weightsFingerprint}）で表し、使う側が {@link #requireNetwork} で照らす。
 */
public final class AnchorScores {

  private static final Logger LOG = LoggerFactory.getLogger(AnchorScores.class);

  /** 1 つの仕事で読む行の数（貯蔵の番号の昇順で続いた行。s の置き場もこの単位）。 */
  private static final int SLICE = 1024;

  /**
   * 1 デバイスあたりの書き手のスレッドの上限。書き手は行の読み出し・展開・符号化をしてから、デバイスの錨を順番に使う。符号化は錨の推論より
   * 1 行あたり重いので、1 デバイスに数本の書き手を回してデバイスを埋める。書き手ごとにページ固定の塊を 1 つ持つ。
   */
  private static final int WRITERS_PER_DEVICE = 8;

  private final String network;

  /** s を持つ行の貯蔵の番号（昇順）。 */
  private final int[] rows;

  /** {@link #SLICE} 行ごとの s（行の順に手の s を並べる）。 */
  private final float[][] values;

  /** {@link #SLICE} 行ごとの行の先頭（長さは行の数 + 1）。 */
  private final int[][] starts;

  private AnchorScores(String network, int[] rows, float[][] values, int[][] starts) {
    this.network = network;
    this.rows = rows;
    this.values = values;
    this.starts = starts;
  }

  /**
   * 行の錨の s を出す。行を {@link #SLICE} 行ずつ書き手のスレッドで読んで塊に書き、デバイスごとの錨で s を出す。錨とストリームは返す前に
   * 閉じる。
   *
   * @param store 行の貯蔵
   * @param rows 行の貯蔵の番号（当てはめの行）
   * @param directory 錨の網のディレクトリ
   * @param path 推論の道（設定の inference。GPU・前線なし・特化した幅のときだけ HIP になる）
   * @param devices 錨を開くデバイス
   * @param threads 書き手のスレッドの上限（{@link #WRITERS_PER_DEVICE} × デバイスの数も超えない）
   * @return s
   * @throws IOException 錨の網か行を読めないとき
   */
  public static AnchorScores compute(
      CohortStore store,
      int[] rows,
      Path directory,
      InferencePath path,
      NetworkDevices devices,
      int threads)
      throws IOException {
    long started = System.nanoTime();
    NetworkRuntime.initialize();
    String network = PrismNetwork.weightsFingerprint(directory);
    boolean frontier = PrismNetwork.readShape(directory).frontier();
    int[] sorted = rows.clone();
    Arrays.sort(sorted);
    int slices = Math.ceilDiv(sorted.length, SLICE);
    float[][] values = new float[slices][];
    int[][] starts = new int[slices][];
    BatchPlan.Capacity capacity = StepFeed.capacity(FitPlan.TOKEN_BUDGET);
    int writers = Math.max(1, Math.min(threads, WRITERS_PER_DEVICE * devices.size()));
    Slices job = new Slices(store, sorted, values, starts, new AtomicInteger());
    List<Lane> lanes = new ArrayList<>(devices.size());
    try (PtNDManager pinned =
        (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME)) {
      try {
        for (Device device : devices) {
          lanes.add(Lane.open(directory, path, device, capacity));
        }
        List<Future<?>> work = new ArrayList<>(writers);
        try (ExecutorService pool = Executors.newFixedThreadPool(writers)) {
          for (int writer = 0; writer < writers; writer++) {
            Lane lane = lanes.get(writer % lanes.size());
            HostInputs batch =
                HostInputs.allocate(
                    capacity,
                    frontier,
                    bytes -> pinned.allocatePinned(bytes, DataType.UINT8).getByteBuffer());
            work.add(pool.submit(() -> job.write(batch, new Encoder.Inputs(frontier), lane)));
          }
          for (Future<?> future : work) {
            await(future);
          }
        }
      } finally {
        close(lanes);
      }
    }
    long actions = 0;
    for (float[] slice : values) {
      actions += slice.length;
    }
    LOG.info(
        "anchor scores rows={} actions={} devices={} writers={} {}s",
        sorted.length,
        actions,
        devices.size(),
        writers,
        String.format("%.1f", (System.nanoTime() - started) / 1e9));
    return new AnchorScores(network, sorted, values, starts);
  }

  /**
   * 錨の網の重みの指紋を返す。
   *
   * @return {@link PrismNetwork#weightsFingerprint} の値
   */
  public String network() {
    return network;
  }

  /**
   * s を出した錨の網が directory の網と同じ重みであることを確かめる。
   *
   * @param directory 使う側の錨の網のディレクトリ
   * @throws IOException 重みを読めないとき
   * @throws IllegalStateException 重みの指紋が違うとき
   */
  public void requireNetwork(Path directory) throws IOException {
    String expected = PrismNetwork.weightsFingerprint(directory);
    if (!expected.equals(network)) {
      throw new IllegalStateException(
          "anchor scores came from the network " + network + " but the anchor " + directory
              + " is " + expected);
    }
  }

  /**
   * 行の錨の s（合法手の順）。
   *
   * @param row 行の貯蔵の番号
   * @return s の写し
   * @throws IllegalArgumentException この行の s がないとき
   */
  float[] scores(int row) {
    int position = Arrays.binarySearch(rows, row);
    if (position < 0) {
      throw new IllegalArgumentException("no anchor scores for row " + row);
    }
    int[] offsets = starts[position / SLICE];
    int local = position % SLICE;
    return Arrays.copyOfRange(values[position / SLICE], offsets[local], offsets[local + 1]);
  }

  /**
   * 書き手たちが分け合う仕事: 昇順の行を {@link #SLICE} 行の区間に分け、空いた書き手が次の区間を取って s を置き場に書く。
   *
   * @param store 行の貯蔵
   * @param rows 行の貯蔵の番号（昇順）
   * @param values 区間ごとの s の置き場
   * @param starts 区間ごとの行の先頭の置き場
   * @param next 次に取る区間
   */
  private record Slices(
      CohortStore store, int[] rows, float[][] values, int[][] starts, AtomicInteger next) {

    /** 区間がなくなるまで取って書く。失敗した書き手は、ほかの書き手に残りの区間を取らせない。 */
    Void write(HostInputs batch, Encoder.Inputs inputs, Lane lane) {
      try {
        for (int slice = next.getAndIncrement();
            slice < values.length;
            slice = next.getAndIncrement()) {
          int from = slice * SLICE;
          int[] ids = Arrays.copyOfRange(rows, from, Math.min(rows.length, from + SLICE));
          Row[] read = readRows(store, ids);
          int[] offsets = new int[read.length + 1];
          for (int row = 0; row < read.length; row++) {
            offsets[row + 1] = Math.addExact(offsets[row], read[row].values().length);
          }
          float[] scores = new float[offsets[read.length]];
          score(read, offsets, scores, batch, inputs, lane);
          values[slice] = scores;
          starts[slice] = offsets;
        }
        return null;
      } catch (RuntimeException | Error failure) {
        next.set(values.length);
        throw failure;
      }
    }
  }

  /**
   * 行を塊に足せるだけ足して錨に通し、行ごとの s を scores の行の先頭（offsets）から写す。塊に入らなくなったら次の塊にする。
   */
  private static void score(
      Row[] read,
      int[] offsets,
      float[] scores,
      HostInputs batch,
      Encoder.Inputs inputs,
      Lane lane) {
    BatchPlan plan = batch.plan();
    int first = 0;
    while (first < read.length) {
      plan.clear();
      int end = first;
      while (end < read.length && plan.add(inputs.bind(read[end])) >= 0) {
        end++;
      }
      if (end == first) {
        throw new IllegalStateException("a row does not fit an empty batch");
      }
      plan.seal();
      for (int row = first; row < end; row++) {
        inputs.write(batch, row - first, read[row]);
      }
      Anchor.Scores result = lane.scores(batch);
      for (int row = first; row < end; row++) {
        int count = offsets[row + 1] - offsets[row];
        if (plan.actions(row - first) != count) {
          throw new IllegalStateException(
              "a row has " + count + " values but " + plan.actions(row - first) + " legal actions");
        }
        System.arraycopy(
            result.values(), result.starts()[row - first], scores, offsets[row], count);
      }
      first = end;
    }
  }

  private static Row[] readRows(CohortStore store, int[] ids) {
    try {
      return store.read(ids, Runnable::run, 1);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /** 書き手の完了を待つ。書き手の例外は包みを外して投げる。 */
  private static void await(Future<?> future) throws IOException {
    try {
      future.get();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("interrupted while computing anchor scores", interrupted);
    } catch (ExecutionException failure) {
      Throwable cause = failure.getCause();
      if (cause instanceof UncheckedIOException unreadable) {
        throw unreadable.getCause();
      }
      if (cause instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw new IOException("anchor scores failed", cause);
    }
  }

  private static void close(List<Lane> lanes) {
    RuntimeException failure = null;
    for (Lane lane : lanes) {
      try {
        lane.close();
      } catch (RuntimeException problem) {
        if (failure == null) {
          failure = problem;
        } else {
          failure.addSuppressed(problem);
        }
      }
    }
    if (failure != null) {
      throw failure;
    }
  }

  /** 1 つのデバイスの錨（書き手が順番に使う）と、GPU ならそのストリーム。 */
  private static final class Lane implements AutoCloseable {

    private final Anchor anchor;

    /** HIP の錨の計算ストリーム（CPU と eager の錨は null）。 */
    private final PtStream stream;

    private Lane(Anchor anchor, PtStream stream) {
      this.anchor = anchor;
      this.stream = stream;
    }

    /**
     * デバイスに錨を開く。eager の錨は背骨の注意を djl-rocm の演算にする（自前の演算は専用のストリームに積むので、eager の配列と同じ流れに
     * 乗らない）。
     */
    static Lane open(Path directory, InferencePath path, Device device, BatchPlan.Capacity capacity)
        throws IOException {
      PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
      PtStream stream = device.isGpu() ? engine.newStream(device) : null;
      try {
        return new Lane(Anchor.open(directory, path, device, stream, null, capacity), stream);
      } catch (IOException | RuntimeException | Error failure) {
        if (stream != null) {
          stream.close();
        }
        throw failure;
      }
    }

    synchronized Anchor.Scores scores(HostInputs batch) {
      return anchor.scores(batch);
    }

    @Override
    public void close() {
      try {
        anchor.close();
      } finally {
        if (stream != null) {
          stream.close();
        }
      }
    }
  }
}
