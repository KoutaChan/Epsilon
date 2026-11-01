package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ValueLabel;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.rows.Row;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 学習の 1 歩分の行を、デバイスごとの微小バッチにする（ホストの仕事。学習のスレッドと重ねる）。
 *
 * <ol>
 *   <li>行を貯蔵から読む（作業のスレッドで並列）。
 *   <li>行をトークンの数 S（局・席・牌の 39 ＋ 公開の出来事）の順に並べ、S の近い行から塊に詰める（参照の分け方）。詰めた
 *       トークン（行の数 × 最も長い行の S）は計画の上限まで。本当のトークンで塊の数をデバイスの数の倍数に見積もり、1 つの塊のトークンを
 *       ほぼ同じにするが、詰めたトークンの上限が先に効くと塊が 1 つ増えることがある。行を並べるので網の詰め物（出来事の数を最も長い行に
 *       そろえる分）が小さい。
 *   <li>塊の数がデバイスの数の倍数でなければ、最後の塊（S の最も長い行）の行をデバイスの数の続いた部分に詰め直す（1 つのデバイスだけが塊を
 *       1 つ多く持つ偏りを小さくする）。部分は元の塊の行の割合と補助の頭の分母を持つので（{@link Microbatch#ofPiece}）、1 歩の損失は分けない
 *       ときと同じ（FP32 の和の順の違いだけ）。
 *   <li>塊ごとに Tokenizer（前線の腕なら前線も）で書き、教師を作る（作業のスレッドで並列。{@link Teacher}。錨の s は学習のレーンが
 *       混ぜる）。
 *   <li>重い微小バッチから、仕事の少ないデバイスへ割り当てる。
 * </ol>
 *
 * <p>塊は使い回す（{@link #release} で返す）。足りなければ新しく取るので、同時に作る歩の数だけ増えて止まる。塊の blob はページ固定の
 * 領域に取り、学習のレーンの H2D と HIP の錨は固定メモリから読む。前線の腕（網の形の {@link PrismShape#frontier()}）では、塊ごとに
 * 前線の塊も取る。歩はここで 1 つずつ順に作る。
 */
final class StepFeed implements AutoCloseable {

  private static final Logger LOG = LoggerFactory.getLogger(StepFeed.class);

  /** 閉じるときに、先読みと作業のスレッドがそれぞれの書きかけの仕事を終えるまで待つ上限。 */
  private static final long STOP_SECONDS = 60;

  /** 1 塊の遷移と手の上限は、トークンの上限のこの分の 1（1 行あたりの平均は約 10 で、超える塊は分ける）。 */
  private static final int TOKENS_PER_TRANSITION = 4;

  /**
   * 1 歩分の微小バッチ。
   *
   * @param lanes デバイスの順の微小バッチ（空のデバイスもある）
   * @param rows 1 歩の行の数
   * @param denominator 差分整合の分母 B_C（|C| ≥ 2 の行の数、なければ 1）
   * @param batches 使った塊（学習の後に {@link #release} で返す）
   */
  record Step(List<List<Microbatch>> lanes, int rows, int denominator, List<HostInputs> batches) {}

  /** 詰め方を決めた塊（符号化の前）。 */
  private record Piece(HostInputs batch, List<Row> rows) {}

  private final CohortStore store;
  private final int lanes;
  private final int budget;
  private final int threads;
  private final boolean frontier;
  private final boolean balanced;
  private final BatchPlan.Capacity capacity;
  private final ExecutorService coordinator;
  private final ExecutorService workers;
  private final ThreadLocal<Encoder> encoders;
  private final Observation observation = new Observation();
  private final ConcurrentLinkedQueue<HostInputs> free = new ConcurrentLinkedQueue<>();
  private final PtNDManager pinned;
  private final LongAdder nanos = new LongAdder();
  private final LongAdder uneven = new LongAdder();

  /**
   * 歩の作り手を作る（塊の数がデバイスの数の倍数でない歩は、最後の塊をデバイスの数の部分に分ける）。
   *
   * @param store 行の貯蔵
   * @param plan 当てはめの計画
   * @param teacher 教師の作り方
   * @param lanes デバイスの数
   * @param threads 読み出しと符号化のスレッドの数
   */
  StepFeed(CohortStore store, FitPlan plan, Teacher teacher, int lanes, int threads) {
    this(store, plan, teacher, lanes, threads, true);
  }

  /**
   * 歩の作り手を作る。最後の塊を分けない作り手は、分けた歩と 1 歩の損失を比べる試験の参照。
   *
   * @param store 行の貯蔵
   * @param plan 当てはめの計画
   * @param teacher 教師の作り方
   * @param lanes デバイスの数
   * @param threads 読み出しと符号化のスレッドの数
   * @param balanced 塊の数がデバイスの数の倍数でない歩の最後の塊を、デバイスの数の部分に分けるなら true
   */
  StepFeed(
      CohortStore store, FitPlan plan, Teacher teacher, int lanes, int threads, boolean balanced) {
    if (lanes < 1 || threads < 1) {
      throw new IllegalArgumentException(
          "lanes and threads must be positive: " + lanes + ", " + threads);
    }
    this.store = store;
    this.lanes = lanes;
    this.threads = threads;
    this.balanced = balanced;
    frontier = plan.shape().frontier();
    budget = plan.tokenBudget();
    capacity = capacity(budget);
    pinned = (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME);
    coordinator = Executors.newSingleThreadExecutor(daemons("prism-feed-"));
    workers = Executors.newFixedThreadPool(threads, daemons("prism-lrn-"));
    encoders = ThreadLocal.withInitial(() -> new Encoder(plan, teacher));
  }

  /**
   * 1 歩分の行の微小バッチを、先読みのスレッドで作り始める。
   *
   * @param rows 行の番号
   * @return 1 歩分
   */
  CompletableFuture<Step> prepare(int[] rows) {
    return CompletableFuture.supplyAsync(() -> step(rows), coordinator);
  }

  /**
   * 学習を終えた歩の塊を返す。
   *
   * @param step 歩
   */
  void release(Step step) {
    free.addAll(step.batches());
  }

  /**
   * 歩を作るのにかかった秒数の和を返す。
   *
   * @return 秒
   */
  double seconds() {
    return nanos.sum() / 1e9;
  }

  /**
   * 参照の分け方の塊の数がデバイスの数の倍数でなかった歩の数を返す（分けるかに依らず数える）。
   *
   * @return 歩の数
   */
  int unevenSteps() {
    return uneven.intValue();
  }

  private Step step(int[] rows) {
    long started = System.nanoTime();
    Row[] read;
    try {
      read = store.read(rows, workers, threads);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    List<List<Piece>> pieces = balance(pack(read));
    List<List<CompletableFuture<Microbatch>>> encoding = new ArrayList<>(pieces.size());
    for (List<Piece> parts : pieces) {
      List<CompletableFuture<Microbatch>> encoded = new ArrayList<>(parts.size());
      for (Piece part : parts) {
        encoded.add(CompletableFuture.supplyAsync(() -> encode(part), workers));
      }
      encoding.add(encoded);
    }
    List<Microbatch> microbatches = new ArrayList<>();
    List<HostInputs> batches = new ArrayList<>();
    int denominator = 0;
    for (List<CompletableFuture<Microbatch>> encoded : encoding) {
      List<Microbatch> parts = new ArrayList<>(encoded.size());
      for (CompletableFuture<Microbatch> future : encoded) {
        parts.add(future.join());
      }
      for (Microbatch microbatch : Microbatch.ofPiece(parts, read.length)) {
        microbatches.add(microbatch);
        batches.add(microbatch.batch());
        for (ValueLabel label : microbatch.labels()) {
          if (PrismLoss.contributes(label)) {
            denominator++;
          }
        }
      }
    }
    Step step = new Step(assign(microbatches), read.length, Math.max(1, denominator), batches);
    nanos.add(System.nanoTime() - started);
    return step;
  }

  /** 微小バッチを書く。 */
  private Microbatch encode(Piece piece) {
    return encoders.get().encode(piece.batch(), piece.rows());
  }

  /** 行を S の順に並べて塊に詰め、塊の計画を封じる（参照の分け方）。 */
  private List<Piece> pack(Row[] rows) {
    long[] keys = new long[rows.length];
    long total = 0;
    for (int index = 0; index < rows.length; index++) {
      int tokens =
          BatchPlan.FIXED_TOKENS + observation.bind(rows[index].observation(), 0).publicEvents();
      keys[index] = (long) tokens << Integer.SIZE | index;
      total += tokens;
    }
    Arrays.sort(keys);
    int count = Math.max(lanes, (int) Math.ceilDiv(total, budget));
    long target = Math.ceilDiv(total, Math.ceilDiv(count, lanes) * (long) lanes);
    List<Piece> pieces = new ArrayList<>();
    HostInputs batch = null;
    List<Row> members = null;
    long filled = 0;
    for (long key : keys) {
      int tokens = (int) (key >>> Integer.SIZE);
      Row row = rows[(int) key];
      Observation o = observation.bind(row.observation(), 0);
      boolean room =
          batch != null && filled < target && (long) (members.size() + 1) * tokens <= budget;
      if (!room || batch.plan().add(o) < 0) {
        if (batch != null) {
          batch.plan().seal();
          pieces.add(new Piece(batch, members));
        }
        batch = acquire();
        members = new ArrayList<>();
        filled = 0;
        if (batch.plan().add(o) < 0) {
          throw new IllegalStateException("a row does not fit an empty batch");
        }
      }
      members.add(row);
      filled += tokens;
    }
    if (batch != null) {
      batch.plan().seal();
      pieces.add(new Piece(batch, members));
    }
    return pieces;
  }

  /**
   * 参照の分け方の塊を、元の塊ごとの部分の並びにする。塊の数がデバイスの数の倍数でない歩を数え、分けるなら最後の塊（S の最も長い行の塊）
   * だけをデバイスの数の部分にする。ほかの塊は部分 1 つ。
   */
  private List<List<Piece>> balance(List<Piece> pieces) {
    List<List<Piece>> parts = new ArrayList<>(pieces.size());
    for (Piece piece : pieces) {
      parts.add(List.of(piece));
    }
    if (pieces.size() % lanes != 0) {
      uneven.increment();
      if (balanced) {
        int last = pieces.size() - 1;
        parts.set(last, split(pieces.get(last)));
      }
    }
    return parts;
  }

  /**
   * 塊の行（S の順）を、行の数がほぼ同じデバイスの数の続いた部分に詰め直し、元の塊は返す（端数の行は前の、S の短い部分に入れる）。どの部分も
   * 元の塊の計画に収まった行の一部なので空の塊に入る。行が 1 つなら分けない。
   */
  private List<Piece> split(Piece piece) {
    List<Row> rows = piece.rows();
    int count = Math.min(lanes, rows.size());
    if (count < 2) {
      return List.of(piece);
    }
    free.add(piece.batch());
    List<Piece> parts = new ArrayList<>(count);
    for (int part = 0; part < count; part++) {
      List<Row> members =
          List.copyOf(
              rows.subList(
                  Math.ceilDiv(rows.size() * part, count),
                  Math.ceilDiv(rows.size() * (part + 1), count)));
      HostInputs batch = acquire();
      for (Row row : members) {
        if (batch.plan().add(observation.bind(row.observation(), 0)) < 0) {
          throw new IllegalStateException("a part of a packed batch does not fit an empty batch");
        }
      }
      batch.plan().seal();
      parts.add(new Piece(batch, members));
    }
    return parts;
  }

  /** 重い微小バッチから、詰めたトークンの少ないデバイスへ割り当てる。 */
  private List<List<Microbatch>> assign(List<Microbatch> microbatches) {
    List<List<Microbatch>> assigned = new ArrayList<>(lanes);
    for (int lane = 0; lane < lanes; lane++) {
      assigned.add(new ArrayList<>());
    }
    List<Microbatch> heavyFirst = new ArrayList<>(microbatches);
    heavyFirst.sort(Comparator.comparingLong(Microbatch::tokens).reversed());
    long[] load = new long[lanes];
    for (Microbatch microbatch : heavyFirst) {
      int lane = 0;
      for (int candidate = 1; candidate < lanes; candidate++) {
        if (load[candidate] < load[lane]) {
          lane = candidate;
        }
      }
      assigned.get(lane).add(microbatch);
      load[lane] += microbatch.tokens();
    }
    return assigned;
  }

  /** 空いた塊を借りる（なければ新しく取る）。計画は空にして渡す。 */
  private HostInputs acquire() {
    HostInputs batch = free.poll();
    if (batch == null) {
      batch = HostInputs.allocate(capacity, frontier, this::pinnedBlob);
    }
    batch.plan().clear();
    return batch;
  }

  private ByteBuffer pinnedBlob(int bytes) {
    return pinned.allocatePinned(bytes, DataType.UINT8).getByteBuffer();
  }

  /** トークンの上限から 1 塊の上限を決める（行は全て出来事のない行でも入る数）。 */
  static BatchPlan.Capacity capacity(int budget) {
    int transitions = Math.max(Limits.MAX_TRANSITIONS, budget / TOKENS_PER_TRANSITION);
    int actions = Math.max(Limits.MAX_LEGAL_ACTIONS, budget / TOKENS_PER_TRANSITION);
    return new BatchPlan.Capacity(budget / BatchPlan.FIXED_TOKENS, budget, transitions, actions);
  }

  private static ThreadFactory daemons(String prefix) {
    AtomicInteger index = new AtomicInteger();
    return task -> {
      Thread thread = new Thread(task, prefix + index.getAndIncrement());
      thread.setDaemon(true);
      return thread;
    };
  }

  /**
   * 先読みを止めてから塊のページ固定の領域を返す（学習のレーンを閉じた後に呼ぶ）。先読みの歩は作業のスレッドの仕事を待つので、先読みが
   * 終わるまで作業のスレッドを止めない。どちらかが上限の間に終わらなければ、書きかけの塊を壊さないように領域を返さずに残す。
   */
  @Override
  public void close() {
    coordinator.shutdownNow();
    boolean stopped = stopped(coordinator);
    workers.shutdownNow();
    stopped &= stopped(workers);
    free.clear();
    if (stopped) {
      pinned.close();
    } else {
      LOG.warn(
          "step feed threads did not stop in {} s; leaving their pinned batches allocated",
          STOP_SECONDS);
    }
  }

  private static boolean stopped(ExecutorService pool) {
    try {
      return pool.awaitTermination(STOP_SECONDS, TimeUnit.SECONDS);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      return false;
    }
  }
}
