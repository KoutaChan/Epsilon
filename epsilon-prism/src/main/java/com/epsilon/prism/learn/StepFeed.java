package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.ValueLabel;
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
 * 学習の 1 ステップ分の行を、デバイスごとのマイクロバッチにする。ホストで行う処理で、学習のスレッドと並行して進める。
 *
 * <ol>
 *   <li>行のストアから行を読む（作業スレッドで並列）。
 *   <li>行をトークン数 S（局・席・牌のトークンと公開のイベント）の順に並べ、S の近い行からバッチに詰める（分ける前の詰め方）。パディングを
 *       含むトークン数（行数 × 最も長い行の S）は、計画の上限を超えない。パディングを除いたトークン数で、バッチの数がデバイスの数の倍数に
 *       なるように見積もり、どのバッチのトークン数もほぼ同じにする。ただし、パディングを含むトークン数の上限が先に効くと、バッチが 1 つ
 *       増えることがある。行を並べるので、ネットワークのパディング（イベントの数を最も長い行にそろえる分）は小さい。
 *   <li>バッチをデバイスに割り当てる。重いバッチから順に、時間の見積もり（{@link StepCost}）の少ないデバイスへ割り当てる。バッチの数が
 *       デバイスの数の倍数でなければ、最後のバッチ（S の最も長い行）は、ほかのバッチを割り当てた後で連続した部分に分け、デバイスの見積もりの
 *       最大が最も小さくなるように割り当てる。1 つのデバイスだけがバッチを 1 つ多く持つ偏りを小さくするためである。部分は元のバッチの行の
 *       割合と補助のヘッドの分母を持つので（{@link Microbatch#ofPiece}）、1 ステップの損失は分けないときと同じになる（違うのは FP32 で足す
 *       順だけ）。
 *   <li>割り当てたバッチごとに、入力の形式の行のライターで行を書き、教師を作る（作業スレッドで並列。{@link Teacher}）。アンカーの s は、
 *       行のストアの番号で引いて混ぜる。
 *   <li>マイクロバッチごとに、ネットワークの入力・インデックス・損失の入力をページ固定のバッファ（{@link PinnedUploads}）に並べる（作業
 *       スレッドで並列。{@link Microbatch#staged}）。学習のレーンは、それを非同期に転送するだけにする。
 * </ol>
 *
 * <p>バッチとバッファは使い回す（{@link #release} で返す）。足りなければ新しく取るので、同時に作るステップの数の分だけ増えて止まる。バッチの
 * blob はページ固定の領域に取る。バッファは前の転送が終わるのを待ってから書き直す（{@link PinnedUploads#await}）。ステップはここで 1 つずつ
 * 順に作る。
 */
final class StepFeed implements AutoCloseable {

  private static final Logger LOG = LoggerFactory.getLogger(StepFeed.class);

  /** 閉じるときに、先読みのスレッドと作業スレッドがそれぞれ書きかけの処理を終えるまで待つ時間の上限（秒）。 */
  private static final long STOP_SECONDS = 60;

  /** 1 バッチの遷移と手の数の上限は、トークンの上限をこの値で割ったもの（1 行あたりの平均は約 10。上限を超えるバッチは分ける）。 */
  private static final int TOKENS_PER_TRANSITION = 4;

  /**
   * 1 ステップ分のマイクロバッチ。
   *
   * @param lanes デバイスの順に並べたマイクロバッチ（マイクロバッチの無いデバイスもある）
   * @param rows 1 ステップの行数
   * @param denominator 差分整合の分母 B_C（|C| ≥ 2 の行の数。無ければ 1）
   * @param batches 使ったバッチ（学習の後に {@link #release} で返す）
   */
  record Step(List<List<Microbatch>> lanes, int rows, int denominator, List<HostInputs> batches) {}

  /**
   * 詰め方を決めたバッチ（符号化の前）。
   *
   * @param batch 計画を確定したバッチ
   * @param rows 行（S の順）
   * @param ids 行と同じ順に並べた、行のストアの番号
   * @param tokens 行ごとの S（昇順）
   */
  private record Piece(HostInputs batch, List<Row> rows, int[] ids, int[] tokens) {

    /** デバイスの時間の見積もり（ns）。 */
    long cost() {
      return StepCost.nanos(rows.size(), tokens[tokens.length - 1]);
    }
  }

  /**
   * デバイスに割り当てたバッチ。
   *
   * @param part バッチ（分ける前のバッチか、その部分）
   * @param reference 分ける前のバッチの番号
   * @param lane デバイス
   */
  private record Placement(Piece part, int reference, int lane) {}

  private final CohortStore store;
  private final int lanes;
  private final int budget;
  private final int threads;
  private final InputFormat format;
  private final BatchPlan.Capacity capacity;
  private final ExecutorService coordinator;
  private final ExecutorService workers;
  private final ThreadLocal<Encoder> encoders;
  private final Observation observation;
  private final ConcurrentLinkedQueue<HostInputs> free = new ConcurrentLinkedQueue<>();

  /** 空いたバッファ。 */
  private final ConcurrentLinkedQueue<PinnedUploads> freeRegions = new ConcurrentLinkedQueue<>();

  /** 取った全てのバッファ（閉じるときに転送が終わるのを待って返す）。 */
  private final ConcurrentLinkedQueue<PinnedUploads> regions = new ConcurrentLinkedQueue<>();

  private final PtNDManager pinned;
  private final LongAdder nanos = new LongAdder();
  private final LongAdder uneven = new LongAdder();

  /**
   * バッチの数がデバイスの数の倍数でないステップでは、最後のバッチをデバイスの数までの部分に分ける。
   *
   * @param store 行のストア
   * @param plan 学習の計画
   * @param lanes デバイスの数
   * @param threads 読み出しと符号化のスレッドの数
   */
  StepFeed(CohortStore store, FitPlan plan, int lanes, int threads) {
    if (lanes < 1 || threads < 1) {
      throw new IllegalArgumentException(
          "lanes and threads must be positive: " + lanes + ", " + threads);
    }
    this.store = store;
    this.lanes = lanes;
    this.threads = threads;
    budget = plan.tokenBudget();
    Variant variant = store.variant();
    format = InputFormat.of(variant);
    capacity = capacity(format, budget);
    pinned = (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME);
    coordinator = Executors.newSingleThreadExecutor(daemons("prism-feed-"));
    workers = Executors.newFixedThreadPool(threads, daemons("prism-lrn-"));
    observation = new Observation(variant);
    encoders = ThreadLocal.withInitial(() -> new Encoder(plan, variant));
  }

  /**
   * 1 ステップ分の行からマイクロバッチを作る処理を、先読みのスレッドで始める。
   *
   * @param rows 行の番号
   * @param teacher 教師の作り方
   * @return 1 ステップ分
   */
  CompletableFuture<Step> prepare(int[] rows, Teacher teacher) {
    return CompletableFuture.supplyAsync(() -> step(rows, teacher), coordinator);
  }

  /**
   * 学習を終えたステップのバッチとバッファを返す。
   *
   * @param step ステップ
   */
  void release(Step step) {
    free.addAll(step.batches());
    for (List<Microbatch> lane : step.lanes()) {
      for (Microbatch part : lane) {
        freeRegions.add(part.staged().region());
      }
    }
  }

  /**
   * ステップを作るのにかかった秒数の和を返す。
   *
   * @return 秒数
   */
  double seconds() {
    return nanos.sum() / 1e9;
  }

  /**
   * 分ける前のバッチの数がデバイスの数の倍数でなかったステップの数を返す（最後のバッチが 1 つの部分のままになったステップも数える）。
   *
   * @return ステップの数
   */
  int unevenSteps() {
    return uneven.intValue();
  }

  private Step step(int[] rows, Teacher teacher) {
    long started = System.nanoTime();
    Row[] read;
    try {
      read = store.read(rows, workers, threads);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    List<Piece> pieces = pack(read, rows);
    List<Placement> placements = place(pieces);
    List<CompletableFuture<Microbatch>> encoding = new ArrayList<>(placements.size());
    List<List<Integer>> byReference = new ArrayList<>(pieces.size());
    for (int reference = 0; reference < pieces.size(); reference++) {
      byReference.add(new ArrayList<>());
    }
    for (int index = 0; index < placements.size(); index++) {
      Placement placement = placements.get(index);
      encoding.add(CompletableFuture.supplyAsync(() -> encode(placement.part(), teacher), workers));
      byReference.get(placement.reference()).add(index);
    }
    Microbatch[] placed = new Microbatch[placements.size()];
    for (List<Integer> indices : byReference) {
      List<Microbatch> parts = new ArrayList<>(indices.size());
      for (int index : indices) {
        parts.add(encoding.get(index).join());
      }
      List<Microbatch> shared = Microbatch.ofPiece(parts, read.length);
      for (int part = 0; part < indices.size(); part++) {
        placed[indices.get(part)] = shared.get(part);
      }
    }
    List<Microbatch> staged = stage(Arrays.asList(placed));
    List<List<Microbatch>> assigned = new ArrayList<>(lanes);
    for (int lane = 0; lane < lanes; lane++) {
      assigned.add(new ArrayList<>());
    }
    List<HostInputs> batches = new ArrayList<>(staged.size());
    int denominator = 0;
    for (int index = 0; index < staged.size(); index++) {
      Microbatch microbatch = staged.get(index);
      assigned.get(placements.get(index).lane()).add(microbatch);
      batches.add(microbatch.batch());
      for (ValueLabel label : microbatch.labels()) {
        if (PrismLoss.contributes(label)) {
          denominator++;
        }
      }
    }
    Step step = new Step(assigned, read.length, Math.max(1, denominator), batches);
    nanos.add(System.nanoTime() - started);
    return step;
  }

  /** マイクロバッチのデバイスへの入力を、作業スレッドでバッファに並べる（バッファは先読みのスレッドで借りる）。 */
  private List<Microbatch> stage(List<Microbatch> microbatches) {
    List<CompletableFuture<Microbatch>> staging = new ArrayList<>(microbatches.size());
    for (Microbatch microbatch : microbatches) {
      PinnedUploads region = region();
      staging.add(CompletableFuture.supplyAsync(() -> microbatch.staged(region), workers));
    }
    List<Microbatch> staged = new ArrayList<>(staging.size());
    for (CompletableFuture<Microbatch> future : staging) {
      staged.add(future.join());
    }
    return staged;
  }

  /** 空いたバッファを借りる（無ければ新しく取る）。 */
  private PinnedUploads region() {
    PinnedUploads region = freeRegions.poll();
    if (region == null) {
      region = PinnedUploads.on(pinned);
      regions.add(region);
    }
    return region;
  }

  /** バッチの行を書いて、マイクロバッチを作る。 */
  private Microbatch encode(Piece piece, Teacher teacher) {
    return encoders.get().encode(piece.batch(), piece.rows(), piece.ids(), teacher);
  }

  /**
   * 行を S の順に並べてバッチに詰め、バッチの計画を確定する（分ける前の詰め方）。{@code ids} は {@code rows} と同じ順に並べた、行のストアの
   * 番号。
   */
  private List<Piece> pack(Row[] rows, int[] ids) {
    long[] keys = new long[rows.length];
    long total = 0;
    for (int index = 0; index < rows.length; index++) {
      int tokens = format.fixedTokens() + observation.bind(rows[index].observation(), 0).publicEvents();
      keys[index] = (long) tokens << Integer.SIZE | index;
      total += tokens;
    }
    Arrays.sort(keys);
    int[] widths = new int[keys.length];
    for (int index = 0; index < keys.length; index++) {
      widths[index] = (int) (keys[index] >>> Integer.SIZE);
    }
    int count = Math.max(lanes, (int) Math.ceilDiv(total, budget));
    long target = Math.ceilDiv(total, Math.ceilDiv(count, lanes) * (long) lanes);
    List<Piece> pieces = new ArrayList<>();
    HostInputs batch = null;
    List<Row> members = null;
    int[] memberIds = new int[rows.length];
    int first = 0;
    long filled = 0;
    for (int index = 0; index < keys.length; index++) {
      int tokens = widths[index];
      int at = (int) keys[index];
      Row row = rows[at];
      Observation o = observation.bind(row.observation(), 0);
      boolean room =
          batch != null && filled < target && (long) (members.size() + 1) * tokens <= budget;
      if (!room || batch.plan().add(o) < 0) {
        if (batch != null) {
          batch.plan().seal();
          pieces.add(
              new Piece(
                  batch,
                  members,
                  Arrays.copyOfRange(memberIds, first, index),
                  Arrays.copyOfRange(widths, first, index)));
        }
        batch = acquire();
        members = new ArrayList<>();
        first = index;
        filled = 0;
        if (batch.plan().add(o) < 0) {
          throw new IllegalStateException("a row does not fit an empty batch");
        }
      }
      members.add(row);
      memberIds[index] = ids[at];
      filled += tokens;
    }
    if (batch != null) {
      batch.plan().seal();
      pieces.add(
          new Piece(
              batch,
              members,
              Arrays.copyOfRange(memberIds, first, keys.length),
              Arrays.copyOfRange(widths, first, keys.length)));
    }
    return pieces;
  }

  /**
   * 分ける前のバッチをデバイスに割り当てる。重いバッチから順に、見積もりの和が最も小さいデバイス（同じなら番号の小さい方）へ割り当てる。
   * バッチの数がデバイスの数の倍数でないステップを数え、そのステップでは最後のバッチ（S の最も長い行のバッチ）だけを、ほかのバッチの後に
   * {@link #split} で割り当てる。
   *
   * @return 割り当てたバッチ（デバイスごとのマイクロバッチはこの順に並ぶ）
   */
  private List<Placement> place(List<Piece> pieces) {
    int whole = pieces.size();
    if (whole % lanes != 0) {
      uneven.increment();
      whole--;
    }
    Integer[] heavyFirst = new Integer[whole];
    for (int reference = 0; reference < whole; reference++) {
      heavyFirst[reference] = reference;
    }
    Arrays.sort(heavyFirst, Comparator.comparingLong(index -> -pieces.get(index).cost()));
    long[] load = new long[lanes];
    List<Placement> placements = new ArrayList<>(pieces.size() + lanes - 1);
    for (int reference : heavyFirst) {
      int lane = 0;
      for (int candidate = 1; candidate < lanes; candidate++) {
        if (load[candidate] < load[lane]) {
          lane = candidate;
        }
      }
      placements.add(new Placement(pieces.get(reference), reference, lane));
      load[lane] += pieces.get(reference).cost();
    }
    if (whole < pieces.size()) {
      split(pieces.get(whole), whole, load, placements);
    }
    return placements;
  }

  /**
   * バッチの行（S の順）を連続した部分に分けて割り当て、デバイスの見積もりの最大を最も小さくする。デバイスを見積もりの少ない順に並べ、
   * それぞれが残りの行の先頭から、見積もりの和が上限 T を超えない所まで取る（取らないデバイスもある）。全ての行が入る最小の T（ns）を
   * 二分探索で求める。T を上げると各デバイスが取る範囲の終わりは後ろへしか動かないので、全ての行が入るかどうかは T について単調である。
   * 部分が 1 つなら元のバッチのまま割り当てる。2 つ以上なら元のバッチは返し、部分ごとにバッチを取る。どの部分も元のバッチの計画に収まった
   * 行の一部なので、空のバッチに入る。
   *
   * @param piece バッチ
   * @param reference 分ける前のバッチの番号
   * @param load デバイスごとの、ほかのバッチの見積もりの和（ns）
   * @param placements 部分を足す先
   */
  private void split(Piece piece, int reference, long[] load, List<Placement> placements) {
    Integer[] lightFirst = new Integer[lanes];
    for (int lane = 0; lane < lanes; lane++) {
      lightFirst[lane] = lane;
    }
    Arrays.sort(lightFirst, Comparator.comparingLong(lane -> load[lane]));
    long low = load[lightFirst[0]];
    long high = low + piece.cost();
    while (high - low > 1) {
      long middle = (low + high) >>> 1;
      if (cut(piece.tokens(), lightFirst, load, middle) == null) {
        low = middle;
      } else {
        high = middle;
      }
    }
    int[] ends = cut(piece.tokens(), lightFirst, load, high);
    int parts = 0;
    int only = 0;
    for (int index = 0; index < lanes; index++) {
      if (ends[index] > (index == 0 ? 0 : ends[index - 1])) {
        parts++;
        only = index;
      }
    }
    if (parts == 1) {
      placements.add(new Placement(piece, reference, lightFirst[only]));
      return;
    }
    free.add(piece.batch());
    int start = 0;
    for (int index = 0; index < lanes; index++) {
      int end = ends[index];
      if (end > start) {
        List<Row> members = List.copyOf(piece.rows().subList(start, end));
        HostInputs batch = acquire();
        for (Row row : members) {
          if (batch.plan().add(observation.bind(row.observation(), 0)) < 0) {
            throw new IllegalStateException("a part of a packed batch does not fit an empty batch");
          }
        }
        batch.plan().seal();
        Piece part =
            new Piece(
                batch,
                members,
                Arrays.copyOfRange(piece.ids(), start, end),
                Arrays.copyOfRange(piece.tokens(), start, end));
        placements.add(new Placement(part, reference, lightFirst[index]));
      }
      start = end;
    }
  }

  /**
   * 上限 T のもとで、並べたデバイスが順に、残りの行の先頭から取る範囲を決める。
   *
   * @param tokens 行ごとの S（昇順）
   * @param order デバイスの順
   * @param load デバイスごとの見積もりの和（ns）
   * @param limit 上限 T（ns）
   * @return 順に i 番目のデバイスが取る行の終わり（取る行は前のデバイスの終わりから）。全ての行が入らなければ null
   */
  private static int[] cut(int[] tokens, Integer[] order, long[] load, long limit) {
    int[] ends = new int[order.length];
    int start = 0;
    for (int index = 0; index < order.length; index++) {
      long room = limit - load[order[index]];
      int end = start;
      while (end < tokens.length && StepCost.nanos(end + 1 - start, tokens[end]) <= room) {
        end++;
      }
      ends[index] = end;
      start = end;
    }
    return start == tokens.length ? ends : null;
  }

  /** 空いたバッチを借りる（無ければ新しく取る）。計画は空にして渡す。 */
  private HostInputs acquire() {
    HostInputs batch = free.poll();
    if (batch == null) {
      batch = HostInputs.allocate(format, capacity, this::pinnedBlob);
    }
    batch.plan().clear();
    return batch;
  }

  private ByteBuffer pinnedBlob(int bytes) {
    return pinned.allocatePinned(bytes, DataType.UINT8).getByteBuffer();
  }

  /** トークンの上限から 1 バッチの上限を決める。行数の上限は、全ての行がイベントの無い行でも入る数にする。 */
  static BatchPlan.Capacity capacity(InputFormat format, int budget) {
    int transitions = Math.max(Limits.MAX_TRANSITIONS, budget / TOKENS_PER_TRANSITION);
    int actions = Math.max(Limits.MAX_LEGAL_ACTIONS, budget / TOKENS_PER_TRANSITION);
    return new BatchPlan.Capacity(budget / format.fixedTokens(), budget, transitions, actions);
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
   * 先読みを止め、バッファの最後の転送が終わるのを待ってから、バッチとバッファのページ固定のメモリを返す（学習のレーンを閉じた後に呼ぶ）。
   * 先読みのステップは作業スレッドの処理を待つので、先読みが終わるまで作業スレッドを止めない。どちらかが上限の時間内に終わらないか、転送の
   * 終わりを待てなければ（デバイスの失敗）、メモリを返さずに残す。書きかけのバッチや、転送が読むかもしれないバッファを壊さないためである。
   */
  @Override
  public void close() {
    coordinator.shutdownNow();
    boolean stopped = stopped(coordinator);
    workers.shutdownNow();
    stopped &= stopped(workers);
    free.clear();
    freeRegions.clear();
    if (!stopped) {
      LOG.warn(
          "step feed threads did not stop in {} s; leaving their pinned batches allocated",
          STOP_SECONDS);
      return;
    }
    for (PinnedUploads region : regions) {
      try {
        region.close();
      } catch (RuntimeException failure) {
        LOG.warn(
            "cannot await a copy from a pinned region; leaving the pinned memory allocated",
            failure);
        return;
      }
    }
    pinned.close();
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
