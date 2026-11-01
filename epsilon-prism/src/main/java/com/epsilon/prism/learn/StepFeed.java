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
import com.epsilon.prism.model.UploadRegion;
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
 *   <li>塊をデバイスに置く。重い塊から、仕事の見積もり（{@link StepCost}）の少ないデバイスへ置く。塊の数がデバイスの数の倍数で
 *       なければ、最後の塊（S の最も長い行）はほかの塊を置いた後に、デバイスの見積もりの最大が最も小さくなる続いた部分に分けて置く（1 つの
 *       デバイスだけが塊を 1 つ多く持つ偏りを小さくする）。部分は元の塊の行の割合と補助の頭の分母を持つので（{@link Microbatch#ofPiece}）、
 *       1 歩の損失は分けないときと同じ（FP32 の和の順の違いだけ）。
 *   <li>置いた塊ごとに Tokenizer（前線の腕なら前線も）で書き、教師を作る（作業のスレッドで並列。{@link Teacher}。錨の s は行の貯蔵の
 *       番号で引いて混ぜる）。
 *   <li>微小バッチごとに、網の入力・索引・損失の入力をページ固定の領域（{@link UploadRegion}）に並べる（作業のスレッドで並列。
 *       {@link Microbatch#staged}）。学習のレーンはそれを非同期の写しで送るだけにする。
 * </ol>
 *
 * <p>塊と領域は使い回す（{@link #release} で返す）。足りなければ新しく取るので、同時に作る歩の数だけ増えて止まる。塊の blob はページ固定の
 * 領域に取る。領域は前の写しの終わりを待ってから書き直す（{@link UploadRegion#await}）。前線の腕（網の形の
 * {@link PrismShape#frontier()}）では、塊ごとに前線の塊も取る。歩はここで 1 つずつ順に作る。
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

  /**
   * 詰め方を決めた塊（符号化の前）。
   *
   * @param batch 封じた塊
   * @param rows 行（S の順）
   * @param ids 行と同じ順の行の貯蔵の番号
   * @param tokens 行ごとの S（昇順）
   */
  private record Piece(HostInputs batch, List<Row> rows, int[] ids, int[] tokens) {

    /** デバイスの時間の見積もり（ns）。 */
    long cost() {
      return StepCost.nanos(rows.size(), tokens[tokens.length - 1]);
    }
  }

  /**
   * デバイスに置いた塊。
   *
   * @param part 塊（参照の分け方の塊か、その部分）
   * @param reference 参照の分け方の塊の番号
   * @param lane デバイス
   */
  private record Placement(Piece part, int reference, int lane) {}

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

  /** 空いた領域。 */
  private final ConcurrentLinkedQueue<UploadRegion> freeRegions = new ConcurrentLinkedQueue<>();

  /** 取った全ての領域（閉じるときに写しの終わりを待って返す）。 */
  private final ConcurrentLinkedQueue<UploadRegion> regions = new ConcurrentLinkedQueue<>();

  private final PtNDManager pinned;
  private final LongAdder nanos = new LongAdder();
  private final LongAdder uneven = new LongAdder();

  /**
   * 歩の作り手を作る（塊の数がデバイスの数の倍数でない歩は、最後の塊をデバイスの数までの部分に分ける）。
   *
   * @param store 行の貯蔵
   * @param plan 当てはめの計画
   * @param lanes デバイスの数
   * @param threads 読み出しと符号化のスレッドの数
   */
  StepFeed(CohortStore store, FitPlan plan, int lanes, int threads) {
    this(store, plan, lanes, threads, true);
  }

  /**
   * 歩の作り手を作る。最後の塊を分けない作り手は、分けた歩と 1 歩の損失を比べる試験の参照。
   *
   * @param store 行の貯蔵
   * @param plan 当てはめの計画
   * @param lanes デバイスの数
   * @param threads 読み出しと符号化のスレッドの数
   * @param balanced 塊の数がデバイスの数の倍数でない歩の最後の塊を、デバイスの数までの部分に分けるなら true
   */
  StepFeed(CohortStore store, FitPlan plan, int lanes, int threads, boolean balanced) {
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
    encoders = ThreadLocal.withInitial(() -> new Encoder(plan));
  }

  /**
   * 1 歩分の行の微小バッチを、先読みのスレッドで作り始める。
   *
   * @param rows 行の番号
   * @param teacher 教師の作り方
   * @return 1 歩分
   */
  CompletableFuture<Step> prepare(int[] rows, Teacher teacher) {
    return CompletableFuture.supplyAsync(() -> step(rows, teacher), coordinator);
  }

  /**
   * 学習を終えた歩の塊を返す。
   *
   * @param step 歩
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

  /** 微小バッチのデバイスへの入力を、作業のスレッドで領域に並べる（領域は先読みのスレッドで借りる）。 */
  private List<Microbatch> stage(List<Microbatch> microbatches) {
    List<CompletableFuture<Microbatch>> staging = new ArrayList<>(microbatches.size());
    for (Microbatch microbatch : microbatches) {
      UploadRegion region = region();
      staging.add(CompletableFuture.supplyAsync(() -> microbatch.staged(region), workers));
    }
    List<Microbatch> staged = new ArrayList<>(staging.size());
    for (CompletableFuture<Microbatch> future : staging) {
      staged.add(future.join());
    }
    return staged;
  }

  /** 空いた領域を借りる（なければ新しく取る）。 */
  private UploadRegion region() {
    UploadRegion region = freeRegions.poll();
    if (region == null) {
      region = UploadRegion.on(pinned);
      regions.add(region);
    }
    return region;
  }

  /** 微小バッチを書く。 */
  private Microbatch encode(Piece piece, Teacher teacher) {
    return encoders.get().encode(piece.batch(), piece.rows(), piece.ids(), teacher);
  }

  /** 行を S の順に並べて塊に詰め、塊の計画を封じる（参照の分け方。ids は rows と同じ順の行の貯蔵の番号）。 */
  private List<Piece> pack(Row[] rows, int[] ids) {
    long[] keys = new long[rows.length];
    long total = 0;
    for (int index = 0; index < rows.length; index++) {
      int tokens =
          BatchPlan.FIXED_TOKENS + observation.bind(rows[index].observation(), 0).publicEvents();
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
   * 参照の分け方の塊をデバイスに置く。重い塊から、見積もりの和の最も小さいデバイス（同じなら番号の小さい方）へ置く。塊の数がデバイスの数の
   * 倍数でない歩を数え、分けるなら最後の塊（S の最も長い行の塊）だけはほかの塊の後に {@link #split} で置く。
   *
   * @return 置いた塊（デバイスごとの微小バッチの順はこの順）
   */
  private List<Placement> place(List<Piece> pieces) {
    int whole = pieces.size();
    if (whole % lanes != 0) {
      uneven.increment();
      if (balanced) {
        whole--;
      }
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
   * 塊の行（S の順）を続いた部分に分けて置き、デバイスの見積もりの最大を最も小さくする。デバイスを見積もりの少ない順に並べ、それぞれが残りの
   * 行の頭から、見積もりの和が上限 T を超えない所まで取る（取らないデバイスもある）。全ての行が入る最小の T（ns）を二分探索で
   * 求める（T を上げると各デバイスの取る所は後ろへしか動かないので、入るかは T で単調）。部分が 1 つなら元の塊のまま置き、2 つ以上なら元の
   * 塊は返して部分ごとに塊を取る（どの部分も元の塊の計画に収まった行の一部なので空の塊に入る）。
   *
   * @param piece 塊
   * @param reference 参照の分け方の塊の番号
   * @param load デバイスごとの、ほかの塊の見積もりの和（ns）
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
   * 上限 T で、並べたデバイスが順に残りの行の頭から取る所を決める。
   *
   * @param tokens 行ごとの S（昇順）
   * @param order デバイスの順
   * @param load デバイスごとの見積もりの和（ns）
   * @param limit 上限 T（ns）
   * @return 順の i 番目のデバイスが取る行の終わり（取る行は前の終わりから）。全ての行が入らなければ null
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
   * 先読みを止めてから、領域の最後の写しの終わりを待って塊と領域のページ固定のメモリを返す（学習のレーンを閉じた後に呼ぶ）。先読みの歩は
   * 作業のスレッドの仕事を待つので、先読みが終わるまで作業のスレッドを止めない。どちらかが上限の間に終わらないか、写しの終わりを待てなければ
   * （デバイスの失敗）、書きかけの塊や写しが読むかもしれない領域を壊さないようにメモリを返さずに残す。
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
    for (UploadRegion region : regions) {
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
