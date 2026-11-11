package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputFormat;
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
 * 学習に使う全ての行について、アンカーの s をホストのメモリに持つ。s はアンカーのネットワークで学習の前に 1 度だけ求め
 * （{@link #compute}）、エンコードのスレッドが行のストアの番号で引いて教師に混ぜる（{@link Teacher#values}）。
 *
 * <p>行の入力は、生徒のマイクロバッチと同じライター（{@link Encoder.Inputs}）で書く。アンカーを開くときは、推論のバックエンドと
 * バッチの容量を学習のレーンにそろえる（{@link Anchor}）。HIP の推論では行の s がバッチの組み方に依らないので、HIP のアンカーの s は、マイクロバッチの中で
 * 求めた s とビット単位で一致する。eager のアンカーの s は、バッチのパディングによって丸めが変わることがある。
 *
 * <p>アンカーのネットワークはバージョンで見分け、使う側が {@link #requireVersion} で照合する。
 */
public final class AnchorScores {

  private static final Logger LOG = LoggerFactory.getLogger(AnchorScores.class);

  /** 書き込みスレッドが 1 度に取る区間の行の数。区間は行のストアの番号の昇順に続く行で、s のバッファもこの単位で持つ。 */
  private static final int SLICE = 1024;

  /**
   * 1 デバイスあたりの書き込みスレッドの上限。書き込みスレッドは行を読み出して展開・エンコードし、デバイスのアンカーを順番に使う。
   * 1 行あたりではエンコードのほうがアンカーの推論より重いので、1 デバイスに数本の書き込みスレッドを回してデバイスを休ませない。
   * 書き込みスレッドは、それぞれページ固定のバッチを 1 つ持つ。
   */
  private static final int WRITERS_PER_DEVICE = 8;

  /** s を求めたアンカーのネットワークのバージョン。 */
  private final int version;

  /** s を持つ行の、行のストアでの番号（昇順）。 */
  private final int[] rows;

  /** {@link #SLICE} 行ごとの s。行の順に、各行の手の s を並べる。 */
  private final float[][] values;

  /** {@link #SLICE} 行ごとの、各行の s の開始位置（長さは行の数 + 1）。 */
  private final int[][] starts;

  private AnchorScores(int version, int[] rows, float[][] values, int[][] starts) {
    this.version = version;
    this.rows = rows;
    this.values = values;
    this.starts = starts;
  }

  /**
   * 行ごとにアンカーの s を求める。行を {@link #SLICE} 行ずつ書き込みスレッドで読んでバッチに書き、デバイスごとのアンカーで s を求める。
   * アンカーとストリームは返す前に閉じる。
   *
   * @param store 行のストア
   * @param rows 学習に使う行の、行のストアでの番号
   * @param directory アンカーのネットワークのディレクトリ
   * @param version アンカーのネットワークのバージョン
   * @param path 推論のバックエンド（設定の inference）。HIP になるのは、GPU で、HIP のカーネルがその幅に特化しているときだけ
   * @param devices アンカーを開くデバイス
   * @param threads 書き込みスレッドの上限（{@link #WRITERS_PER_DEVICE} × デバイスの数も超えない）
   * @return 行ごとの s
   * @throws IOException アンカーのネットワークか行を読めないとき
   */
  public static AnchorScores compute(
      CohortStore store,
      int[] rows,
      Path directory,
      int version,
      InferencePath path,
      NetworkDevices devices,
      int threads)
      throws IOException {
    long started = System.nanoTime();
    NetworkRuntime.initialize();
    int[] sorted = rows.clone();
    Arrays.sort(sorted);
    int slices = Math.ceilDiv(sorted.length, SLICE);
    float[][] values = new float[slices][];
    int[][] starts = new int[slices][];
    InputFormat format = InputFormat.of(store.variant());
    BatchPlan.Capacity capacity = StepFeed.capacity(format, FitPlan.TOKEN_BUDGET);
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
                    format, capacity, bytes -> pinned.allocatePinned(bytes, DataType.UINT8).getByteBuffer());
            work.add(pool.submit(() -> job.write(batch, new Encoder.Inputs(store.variant()), lane)));
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
    return new AnchorScores(version, sorted, values, starts);
  }

  /**
   * s を求めたアンカーのネットワークが、使う側のアンカーと同じバージョンかを確かめる。
   *
   * @param version 使う側のアンカーのネットワークのバージョン
   * @throws IllegalStateException バージョンが違うとき
   */
  public void requireVersion(int version) {
    if (version != this.version) {
      throw new IllegalStateException(
          "anchor scores came from version " + this.version + " but the anchor is version " + version);
    }
  }

  /**
   * 行のアンカーの s を、合法手の順で返す。
   *
   * @param row 行のストアでの番号
   * @return s のコピー
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
   * 書き込みスレッドが分け合うジョブ。昇順の行を {@link #SLICE} 行ずつの区間に分け、空いた書き込みスレッドが次の区間を取って
   * s をバッファに書く。
   *
   * @param store 行のストア
   * @param rows 行のストアでの番号（昇順）
   * @param values 区間ごとの s のバッファ
   * @param starts 区間ごとの、各行の s の開始位置のバッファ
   * @param next 次に取る区間
   */
  private record Slices(
      CohortStore store, int[] rows, float[][] values, int[][] starts, AtomicInteger next) {

    /** 区間がなくなるまで取って書く。失敗した書き込みスレッドは、ほかのスレッドにも残りの区間を取らせない。 */
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
   * 行をバッチに入るだけ足してアンカーに通し、各行の s を {@code scores} の、その行の開始位置（{@code offsets}）からコピーする。
   * バッチに入らなくなったら、次のバッチに移る。
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

  /** 書き込みスレッドの完了を待つ。書き込みスレッドで起きた例外は、包んだ例外から取り出して投げる。 */
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

  /** 1 つのデバイスのアンカー（書き込みスレッドが順番に使う）と、GPU ならそのストリーム。 */
  private static final class Lane implements AutoCloseable {

    private final Anchor anchor;

    /** GPU のデバイスで作る計算ストリーム。HIP のアンカーが使い、eager のアンカーは使わない。CPU では null。 */
    private final PtStream stream;

    private Lane(Anchor anchor, PtStream stream) {
      this.anchor = anchor;
      this.stream = stream;
    }

    /**
     * デバイスにアンカーを開く。eager のアンカーでは、バックボーンの Attention に djl-rocm の演算を使う。自前のカーネルは専用の
     * ストリームに積むので、eager の配列と同じストリームに乗らないからである。
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
