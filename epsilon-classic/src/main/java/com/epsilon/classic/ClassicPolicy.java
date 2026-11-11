package com.epsilon.classic;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.policy.BatchedPolicy;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.runtime.DecisionDevicePipeline;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.agent.runtime.InferenceDispatcher;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.classic.input.Buckets;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicExtras;
import com.epsilon.classic.input.ClassicInputs;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicTokenizer;
import com.epsilon.classic.model.ClassicBatch;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.classic.model.PolicyGraph;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Nano・Major のネットワークで手を選ぶ方策。判断の記録を 1 つずつ入力の 1 行に書き（{@link ClassicTokenizer}）、方策のスコアを
 * 方策のグラフ（{@link PolicyGraph}）で合法手の確率に直し、同じグラフで手を貪欲に選ぶ。CPU では呼び出し元のスレッドでその場で答え
 * （{@link Inline}）、GPU では判断をバッチにまとめて推論のランタイムへ出す（{@link Batched}）。
 *
 * <p>価値のヘッドが出すのは局面の効用で、手ごとの値は無い。そのため評価には価値を入れない。CPU の推論では、入力の行と、入力をデバイスへ
 * 送るページ固定のバッファ（{@link PinnedUploads}）をスレッドごとに使い回す。確率は推論のたびに新しいホストの配列へコピーする。GPU の
 * バッチでは、入力のバッチと確率のコピー先をプールで使い回し、バッファはスロットごとに持つ。ネットワークのパラメーターは方策が持ち、
 * 閉じると解放する。閉じた後は呼ばない。
 */
public abstract sealed class ClassicPolicy permits ClassicPolicy.Inline, ClassicPolicy.Batched {

  /** classic の入力のフィールドはどれも判断の記録から求めるので、追加の入力はいつも空。 */
  private static final ClassicExtras EXTRAS = ClassicExtras.empty();

  private final ClassicNetwork network;

  /** パラメーターのマネージャー（デバイスもこのマネージャーのもの）。 */
  final NDManager manager;

  final ClassicSchema schema;
  private final Precision precision;
  private final String label;
  private final ThreadLocal<Workspace> workspaces;

  /**
   * @param network 読んだネットワーク（パラメーターはマネージャーが持つ）
   * @param manager パラメーターのマネージャー。方策が閉じる
   * @param precision 計算の型
   * @param label 評価の結果の行の名前
   */
  private ClassicPolicy(ClassicNetwork network, NDManager manager, Precision precision, String label) {
    this.network = Objects.requireNonNull(network, "network");
    this.manager = Objects.requireNonNull(manager, "manager");
    this.precision = Objects.requireNonNull(precision, "precision");
    this.label = Objects.requireNonNull(label, "label");
    schema = ClassicSchema.of(network.architecture());
    workspaces =
        ThreadLocal.withInitial(() -> new Workspace(network.architecture(), network.variant(), schema, manager));
  }

  public Variant variant() {
    return network.variant();
  }

  public String label() {
    return label;
  }

  public Architecture architecture() {
    return network.architecture();
  }

  /** 同じ方策を {@link Policy} の型で返す。 */
  public final Policy asPolicy() {
    return switch (this) {
      case Inline inline -> inline;
      case Batched batched -> batched;
    };
  }

  /**
   * 合法手の確率と、方策のグラフで貪欲に選んだ手を返す。確率は 0〜1 に収めてから、和が 1 になるように割り直す。
   *
   * @throws IllegalArgumentException 判断の記録のルールがネットワークと違うとき
   */
  public Evaluation evaluate(Observation observation) {
    Workspace workspace = workspaces.get();
    int greedy = infer(observation, workspace);
    int legal = observation.legalActions();
    float[] probabilities = new float[legal];
    double sum = 0;
    for (int action = 0; action < legal; action++) {
      probabilities[action] = Math.min(1f, Math.max(0f, workspace.probabilities[action]));
      sum += probabilities[action];
    }
    for (int action = 0; action < legal; action++) {
      probabilities[action] = (float) (probabilities[action] / sum);
    }
    return new Evaluation(probabilities, null, greedy);
  }

  /**
   * 合法手の確率を、呼び出し元のスレッドで推論して書く。値は方策のグラフの確率をそのまま写す（0〜1 に抑えず、和もそろえない）。
   *
   * @param probabilities 合法手の順に確率を書く先（合法手の数以上の長さ）
   * @return 方策のグラフで貪欲に選んだ手（合法手の何番目か）
   * @throws IllegalArgumentException ほかのルールの判断の記録
   */
  public final int probabilities(Observation observation, float[] probabilities) {
    Workspace workspace = workspaces.get();
    int greedy = infer(observation, workspace);
    System.arraycopy(workspace.probabilities, 0, probabilities, 0, observation.legalActions());
    return greedy;
  }

  /** パラメーターを解放する。 */
  public void close() {
    manager.close();
  }

  /**
   * 判断の記録を呼び出し元のスレッドで推論し、貪欲に選んだ手を返す。合法手の確率は推論ごとに新しいホストの配列に入れ、作業領域に置く。
   * 入力は判断が入る最小の区分の 1 行で、作業領域のバッファからデバイスへ送る。
   *
   * @throws IllegalArgumentException 判断の記録のルールがネットワークと違うとき
   */
  final int infer(Observation observation, Workspace workspace) {
    if (observation.variant() != network.variant()) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + network.variant() + " but found " + observation.variant());
    }
    ClassicInputs inputs = workspace.row(observation);
    workspace.tokenizer.write(observation, EXTRAS, inputs, 0);
    try (NDManager working = manager.newSubManager()) {
      // バッファは推論モードの外で取る。推論モードの中で取ると推論用のテンソルになる
      ClassicBatch batch = ClassicBatch.send(working, workspace.uploads(), inputs, 1);
      try (InferenceMode ignored = manager.getEngine().newInferenceMode()) {
        workspace.probabilities = probabilities(batch).toFloatArray();
      }
    }
    return workspace.greedy.select(inputs, 0, workspace.probabilities, 0);
  }

  /** バッチの各行の合法手の確率 [行, 合法手] を FP32 で返す。配列はバッチのマネージャーに付く。 */
  final NDArray probabilities(ClassicBatch batch) {
    ClassicNetwork.Output output =
        network.forward(
            new ParameterStore(batch.manager(), false), batch, precision, false, ClassicNetwork.Heads.POLICY);
    return PolicyGraph.probabilities(PolicyGraph.logProbabilities(output.policy(), batch), batch);
  }

  final Workspace workspace() {
    return workspaces.get();
  }

  /** CPU の方策。対局を進めるスレッドでその場で答える。作業領域はスレッドごとに持つので、複数のスレッドから同時に呼んでよい。 */
  public static final class Inline extends ClassicPolicy implements InlinePolicy {

    Inline(ClassicNetwork network, NDManager manager, Precision precision, String label) {
      super(network, manager, precision, label);
    }

    /** 貪欲に選んだ手を返す。入力の行は使い回す。順伝播の配列と、確率のホストへのコピーは推論ごとに作り、順伝播の配列はこの中で閉じる。 */
    @Override
    public int act(Observation observation) {
      return infer(observation, workspace());
    }
  }

  /**
   * GPU の方策。同じ区分の判断の記録をバッチにまとめ、推論のランタイム（受付と GPU のパイプライン）で順伝播する。受付の区分は推論の区分
   * （{@link Buckets#inference}）の通し番号。バッチの行数は 2048 までで、合法手 × 遷移 × 行が 32768 を超えないように抑える。
   *
   * <p>推論のランタイムは、開くときに呼び出し元が渡したものを使い、渡されなければ方策が作って持つ。バッチを出す前に {@link #tryAcquire}
   * で受付のホストの容量を予約し、その予約を {@link #choose} に渡す。入力はランタイムのエンコードのスレッドで書く。書いた入力は、
   * パイプラインの準備のスレッドがスロットのページ固定のバッファへコピーし、GPU の転送のストリームに非同期のコピーを積む。順伝播は
   * スロットごとの計算のストリームで、コピーが終わるのを待ってから行う。入力のバッチは、区分と行数（2 の冪に切り上げる）ごとにプールで
   * 使い回す。パラメーターが CPU にあるときは、バッチを呼び出し元のスレッドで推論し、予約は容量を数えない。
   */
  public static final class Batched extends ClassicPolicy implements BatchedPolicy {

    private static final int MAX_ROWS = 2_048;

    /** 1 つのバッチの合法手 × 遷移 × 行の上限。 */
    private static final int MAX_CELLS = 32_768;

    /** 行数の区切りの数。区切りは 1〜2048 の 2 の冪で、12 個ある。 */
    private static final int ROW_STEPS = Integer.numberOfTrailingZeros(MAX_ROWS) + 1;

    private static final int ENCODING_WORKERS = 4;
    private static final int SLOTS = 2;
    private static final int READY_BATCHES = 4;

    private final Device device;
    private final NetworkRuntime runtime;

    /** ランタイムを方策が作ったなら true。方策を閉じるときに一緒に閉じる。 */
    private final boolean ownsRuntime;

    private final DecisionDevicePipeline.Lease lease;
    private final InferenceDispatcher.Registration<Job, Void> registration;

    /** [区分の通し番号 × 区切りの数 ＋ 区切りの番号] → 空いている入力のバッチ。 */
    private final List<Queue<Encoding>> pools;

    /** スロットごとの計算のストリーム。この方策のロックで守る。 */
    private final Map<DecisionDevicePipeline.Slot, PtStream> streams = new IdentityHashMap<>();

    /**
     * スロットごとの入力のバッファ。この方策のロックで守り、パラメーターのマネージャーに付ける。スロットは前のバッチを終えてから次の
     * バッチに渡るので、書き直すときには前のコピーが終わっている。
     */
    private final Map<DecisionDevicePipeline.Slot, PinnedUploads> regions = new IdentityHashMap<>();

    private boolean closed;

    /**
     * @param shared 呼び出し元の推論のランタイム（null なら方策が作って持つ）。パラメーターが CPU にあるときは読まない
     */
    Batched(ClassicNetwork network, NDManager manager, Precision precision, String label, NetworkRuntime shared) {
      super(network, manager, precision, label);
      device = manager.getDevice();
      pools = new ArrayList<>(Buckets.count() * ROW_STEPS);
      for (int pool = 0; pool < Buckets.count() * ROW_STEPS; pool++) {
        pools.add(new ConcurrentLinkedQueue<>());
      }
      if (!device.isGpu()) {
        runtime = null;
        ownsRuntime = false;
        lease = null;
        registration = null;
        return;
      }
      NetworkRuntime used = shared != null ? shared : new NetworkRuntime(ENCODING_WORKERS, SLOTS, READY_BATCHES);
      DecisionDevicePipeline.Lease leased = null;
      InferenceDispatcher.Registration<Job, Void> registered;
      try {
        leased = used.lease(device);
        registered = used.dispatcher().register(List.of(new Backend(leased)));
      } catch (RuntimeException | Error failure) {
        if (leased != null) {
          closeAfter(leased, failure);
        }
        if (shared == null) {
          closeAfter(used, failure);
        }
        throw failure;
      }
      runtime = used;
      ownsRuntime = shared == null;
      lease = leased;
      registration = registered;
    }

    /** 判断を推論する区分の通し番号を返す。 */
    @Override
    public int admission(Observation observation) {
      return Buckets.ordinal(Buckets.inference(observation));
    }

    /** 区分ごとのバッチの行数の上限。2048 と、合法手 × 遷移 × 行が 32768 に収まる行数の小さい方で、少なくとも 1。 */
    @Override
    public int maxBatch(int admission) {
      Bucket bucket = Buckets.of(admission);
      return Math.max(1, Math.min(MAX_ROWS, MAX_CELLS / (bucket.actions() * bucket.transitions())));
    }

    /**
     * バッチを推論する容量を予約する。予約は区分に依らない。GPU では受付の登録でホストの容量を予約し、空きが無ければ空いたときの通知を
     * 返す。パラメーターが CPU にあるときは、容量を数えない予約を返す。
     *
     * @throws IllegalStateException 方策が閉じているか、GPU が壊れたとき
     */
    @Override
    public DecisionInferenceIngress.Attempt tryAcquire(int admission) {
      requireOpen();
      if (registration == null) {
        return BatchedPolicy.super.tryAcquire(admission);
      }
      InferenceDispatcher.Attempt<Job, Void> attempt = registration.tryReserve();
      if (!attempt.acquired()) {
        return DecisionInferenceIngress.Attempt.blocked(attempt.available());
      }
      InferenceDispatcher.Reservation<Job, Void> reservation = attempt.reservation();
      return DecisionInferenceIngress.Attempt.acquired(
          new DecisionInferenceIngress(this, reservation, reservation::close));
    }

    /** GPU では、予約済みのバッチを数えても GPU のスロットが埋まらないなら true。パラメーターが CPU にあるときはいつも true。 */
    @Override
    public boolean needsWork() {
      return registration == null || registration.needsWork();
    }

    /**
     * バッチの全ての判断について、貪欲に選んだ手を書く。入力の形は、全ての判断が入る推論の区分に合わせる（バッチの中で区分が混ざって
     * いてもよい）。予約は方策が受け取る。受け付けないバッチでは、予約を返してから例外を投げる。
     *
     * @param chosen 判断の記録の番号の順に、貪欲に選んだ手（合法手の何番目か）を書く先
     * @param reservation この方策の {@link #tryAcquire} で取った予約
     * @throws IllegalArgumentException 予約がほかの方策のもの、ルールが違う、バッチが空、区分の行数の上限を超える、または書く先が短いとき
     * @throws IllegalStateException 予約を渡し済みか、方策が閉じているとき
     */
    @Override
    public CompletableFuture<Void> choose(ObservationBatch batch, int[] chosen, DecisionInferenceIngress reservation) {
      return submit(batch, new Greedy(chosen), reservation);
    }

    /**
     * バッチの全ての判断の記録について、合法手の確率を書く。値は方策のグラフの確率をそのまま写す（0〜1 に抑えず、和もそろえない）。
     * バッチと予約の扱いは {@link #choose} と同じ。
     *
     * @param probabilities 判断の記録の番号の順に、合法手の確率を書く先（行の長さは、その判断の合法手の数）
     * @param reservation この方策の {@link #tryAcquire} で取った予約
     * @throws IllegalArgumentException ほかの方策の予約か、ルールが違うか、バッチが空か、区分の行の上限を超えるか、書き込み先の形が
     *     合わない
     * @throws IllegalStateException 渡し済みの予約か、閉じた方策
     */
    public CompletableFuture<Void> probabilities(
        ObservationBatch batch, float[][] probabilities, DecisionInferenceIngress reservation) {
      return submit(batch, new Probabilities(probabilities), reservation);
    }

    /** バッチを推論して答えを書く。予約は方策が受け取り、受け付けないバッチでは予約を返してから例外を投げる。 */
    private CompletableFuture<Void> submit(ObservationBatch batch, Answer answer, DecisionInferenceIngress reservation) {
      DecisionInferenceIngress.Ownership ownership = reservation.handoff(this);
      int rows;
      Encoding encoding;
      try {
        requireOpen();
        if (batch.variant() != variant()) {
          throw new IllegalArgumentException(
              "Variant mismatch: expected " + variant() + " but found " + batch.variant());
        }
        rows = batch.size();
        Bucket bucket = cover(batch);
        int limit = maxBatch(Buckets.ordinal(bucket));
        if (rows > limit) {
          throw new IllegalArgumentException(rows + " observations exceed the batch limit " + limit + " of " + bucket);
        }
        answer.check(batch, workspace().reader);
        encoding = borrow(bucket, rows);
      } catch (RuntimeException | Error failure) {
        releaseReservation(ownership, failure);
        throw failure;
      }
      if (registration == null) {
        ownership.move();
        try {
          encode(batch, encoding, 0, rows);
          try (NDManager working = manager.newSubManager()) {
            ClassicBatch input = ClassicBatch.send(working, workspace().uploads(), encoding.inputs, rows);
            try (InferenceMode inference = manager.getEngine().newInferenceMode()) {
              float[] probabilities = probabilities(input).toFloatArray();
              answer.write(encoding, probabilities, rows);
            }
          }
          return CompletableFuture.completedFuture(null);
        } catch (RuntimeException | Error failure) {
          return CompletableFuture.failedFuture(failure);
        } finally {
          giveBack(encoding);
        }
      }
      return run(ownership, batch, encoding, rows, answer).whenComplete((ignored, failure) -> giveBack(encoding));
    }

    /** 受付の登録を外し、出したバッチが全て終わるのを待ってから、GPU のリソースとパラメーターを解放する。 */
    @Override
    public void close() {
      synchronized (this) {
        if (closed) {
          return;
        }
        closed = true;
      }
      Throwable failure = null;
      if (registration != null) {
        failure = closeAfter(registration, failure);
        failure = closeAfter(lease, failure);
        failure = closeAfter(this::closeRegions, failure);
        failure = closeAfter(this::closeStreams, failure);
        if (ownsRuntime) {
          failure = closeAfter(runtime, failure);
        }
      }
      failure = closeAfter(super::close, failure);
      if (failure instanceof RuntimeException problem) {
        throw problem;
      }
      if (failure instanceof Error problem) {
        throw problem;
      }
      if (failure != null) {
        throw new IllegalStateException("Cannot close the classic policy " + label(), failure);
      }
    }

    private synchronized void requireOpen() {
      if (closed) {
        throw new IllegalStateException("The classic policy " + label() + " is closed");
      }
    }

    /** バッチの全ての判断が入る推論の区分を返す。 */
    private Bucket cover(ObservationBatch batch) {
      if (batch.size() == 0) {
        throw new IllegalArgumentException("The batch holds no observation");
      }
      Observation reader = workspace().reader;
      int legal = 1;
      int transitions = 1;
      for (int row = 0; row < batch.size(); row++) {
        reader.bind(batch, row);
        legal = Math.max(legal, reader.legalActions());
        transitions = Math.max(transitions, Buckets.maxTransitions(reader));
      }
      return Buckets.inference(legal, transitions);
    }

    /**
     * バッチの判断の記録のうち、範囲に入るものを同じ番号の入力の行に書く。呼んだスレッドの作業領域を使う。
     *
     * @param from 範囲の最初の番号
     * @param to 範囲の終わり（この番号は含まない）
     */
    private void encode(ObservationBatch batch, Encoding encoding, int from, int to) {
      Workspace workspace = workspace();
      for (int row = from; row < to; row++) {
        workspace.tokenizer.write(workspace.reader.bind(batch, row), EXTRAS, encoding.inputs, row);
      }
    }

    /** エンコードのスレッドで入力を書き、予約した容量でバッチを GPU へ出す。書けなければ予約を返す。 */
    private CompletableFuture<Void> run(
        DecisionInferenceIngress.Ownership ownership,
        ObservationBatch batch,
        Encoding encoding,
        int rows,
        Answer answer) {
      CompletableFuture<Void> encoded;
      try {
        encoded = runtime.encodeRows(rows, (from, to) -> encode(batch, encoding, from, to));
      } catch (RuntimeException | Error failure) {
        releaseReservation(ownership, failure);
        return CompletableFuture.failedFuture(failure);
      }
      return encoded
          .handle(
              (ignored, failure) -> {
                if (failure != null) {
                  releaseReservation(ownership, failure);
                  return CompletableFuture.<Void>failedFuture(failure);
                }
                return dispatch(ownership, new Job(encoding, rows, answer));
              })
          .thenCompose(future -> future);
    }

    /** 予約を実行先へ移してバッチを出す。容量を数えない予約のように、受付の予約でなければ IllegalArgumentException で完了する。 */
    private static CompletableFuture<Void> dispatch(DecisionInferenceIngress.Ownership ownership, Job job) {
      if (!(ownership.move() instanceof InferenceDispatcher.Reservation<?, ?> route)) {
        return CompletableFuture.failedFuture(
            new IllegalArgumentException("The reservation holds no capacity of the inference dispatcher"));
      }
      @SuppressWarnings("unchecked")
      InferenceDispatcher.Reservation<Job, Void> reservation = (InferenceDispatcher.Reservation<Job, Void>) route;
      return reservation.submit(job);
    }

    /** 渡されなかった予約を返す。返すときに出た例外は、先の失敗に付ける。 */
    private static void releaseReservation(DecisionInferenceIngress.Ownership ownership, Throwable failure) {
      try {
        ownership.release();
      } catch (RuntimeException | Error problem) {
        failure.addSuppressed(problem);
      }
    }

    /** 区分と行数の区切りに合う入力のバッチをプールから借りる。プールが空なら作る。 */
    private Encoding borrow(Bucket bucket, int rows) {
      int capacity = rows == 1 ? 1 : Integer.highestOneBit(rows - 1) << 1;
      int pool = Buckets.ordinal(bucket) * ROW_STEPS + Integer.numberOfTrailingZeros(capacity);
      Encoding encoding = pools.get(pool).poll();
      return encoding != null ? encoding : new Encoding(new ClassicInputs(schema, bucket, capacity), pool);
    }

    private void giveBack(Encoding encoding) {
      pools.get(encoding.pool).add(encoding);
    }

    /** スロットの計算のストリームを返す。初めて使うときに作る。 */
    private synchronized PtStream stream(DecisionDevicePipeline.Slot slot) {
      PtStream stream = streams.get(slot);
      if (stream == null) {
        stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(device);
        streams.put(slot, stream);
      }
      return stream;
    }

    /** スロットの入力のバッファを返す。初めて使うときに作る。 */
    private synchronized PinnedUploads region(DecisionDevicePipeline.Slot slot) {
      return regions.computeIfAbsent(slot, ignored -> PinnedUploads.on(manager));
    }

    /** バッファごとに、最後のコピーが終わるのを待ってから返す。 */
    private synchronized void closeRegions() throws Exception {
      closeAll(regions);
    }

    private synchronized void closeStreams() throws Exception {
      closeAll(streams);
    }

    /** スロットごとのリソースを全て閉じて、表を空にする。閉じられないものがあれば、最初の例外にほかの例外を付けて投げる。 */
    private static void closeAll(Map<DecisionDevicePipeline.Slot, ? extends AutoCloseable> resources) throws Exception {
      Throwable failure = null;
      for (AutoCloseable resource : resources.values()) {
        failure = closeAfter(resource, failure);
      }
      resources.clear();
      if (failure instanceof Exception problem) {
        throw problem;
      }
      if (failure instanceof Error problem) {
        throw problem;
      }
    }

    /** バッチの答えの書き方。書き込み先の形は受け付ける前に確かめ、答えは確率が出た後に書く。 */
    private sealed interface Answer permits Greedy, Probabilities {

      /**
       * 書き込み先がバッチの判断の記録に合うかを確かめる。
       *
       * @param reader 呼んだスレッドのリーダー（付け替える）
       * @throws IllegalArgumentException 書き込み先の形が合わない
       */
      void check(ObservationBatch batch, Observation reader);

      /**
       * 先頭の行から答えを書く。
       *
       * @param probabilities 合法手の確率 [行, 合法手の軸]
       */
      void write(Encoding encoding, float[] probabilities, int rows);
    }

    /** 貪欲に選んだ手（合法手の何番目か）を、判断の記録の番号の順に書く。 */
    private record Greedy(int[] chosen) implements Answer {

      @Override
      public void check(ObservationBatch batch, Observation reader) {
        if (chosen.length < batch.size()) {
          throw new IllegalArgumentException(batch.size() + " observations need choices, got " + chosen.length);
        }
      }

      @Override
      public void write(Encoding encoding, float[] probabilities, int rows) {
        encoding.answer(probabilities, rows, chosen);
      }
    }

    /** 合法手の確率を、判断の記録の番号の順の行に写す。行の長さはその判断の合法手の数。 */
    private record Probabilities(float[][] rows) implements Answer {

      @Override
      public void check(ObservationBatch batch, Observation reader) {
        if (rows.length < batch.size()) {
          throw new IllegalArgumentException(batch.size() + " observations need probability rows, got " + rows.length);
        }
        for (int row = 0; row < batch.size(); row++) {
          int legal = reader.bind(batch, row).legalActions();
          if (rows[row].length != legal) {
            throw new IllegalArgumentException(
                "Row " + row + " needs " + legal + " probabilities, got " + rows[row].length);
          }
        }
      }

      @Override
      public void write(Encoding encoding, float[] probabilities, int count) {
        int actions = encoding.inputs.bucket().actions();
        for (int row = 0; row < count; row++) {
          System.arraycopy(probabilities, row * actions, rows[row], 0, rows[row].length);
        }
      }
    }

    /** 1 つの GPU のバックエンド。 */
    private final class Backend implements InferenceDispatcher.Backend<Job, Void> {

      private final DecisionDevicePipeline.Lease lease;

      private Backend(DecisionDevicePipeline.Lease lease) {
        this.lease = lease;
      }

      @Override
      public InferenceDispatcher.Device device() {
        return lease.inferenceDevice();
      }

      @Override
      public CompletableFuture<Void> trySubmit(Job job, Runnable slotHandoff) {
        DecisionDevicePipeline.HostReadyCell cell = lease.acquireHostReadyCell();
        if (cell == null) {
          return null;
        }
        try {
          cell.onSlotHandoff(slotHandoff);
          lease.claim(cell);
          lease.submitClaimed(job, cell);
        } catch (RuntimeException | Error failure) {
          cell.abortIfActive();
          job.result.completeExceptionally(failure);
        }
        return job.result;
      }
    }

    /**
     * スロットで 1 回推論するバッチ。入力をスロットのバッファに書いて転送のストリームで送り、スロットの計算のストリームで順伝播する。
     * 共通のパイプラインが確率をホストへコピーした後に、答えを書く。
     */
    private final class Job implements DecisionDevicePipeline.Work {

      private final Encoding encoding;
      private final int rows;
      private final Answer answer;
      private final CompletableFuture<Void> result = new CompletableFuture<>();

      /** スロットの入力のバッファ。入力を書いたときに入る。 */
      private PinnedUploads region;

      private NDManager working;
      private PtStream stream;
      private Throwable failure;
      private boolean copyStarted;

      /** コピーを積み終え、その終わりをバッファが預かっていれば true。 */
      private boolean copied;

      /** コピーを積む途中で失敗した後、転送のストリームにスロットの転送のイベントを記録したら true。 */
      private boolean copyFenced;

      private boolean computeStarted;
      private boolean computeRecorded;

      private Job(Encoding encoding, int rows, Answer answer) {
        this.encoding = encoding;
        this.rows = rows;
        this.answer = answer;
      }

      /** パディングを含む遷移の位置の数。 */
      @Override
      public long estimatedWorkUnits() {
        Bucket bucket = encoding.inputs.bucket();
        return (long) rows * bucket.actions() * bucket.transitions();
      }

      @Override
      public DecisionDevicePipeline.BatchMetadata batchMetadata() {
        return new DecisionDevicePipeline.BatchMetadata(rows, maxBatch(Buckets.ordinal(encoding.inputs.bucket())));
      }

      /** 書いた入力をスロットのバッファにコピーする。バッファは、このスロットの前のバッチのコピーが終わるのを待ってから書かせる。 */
      @Override
      public void stage(DecisionDevicePipeline.Slot slot) {
        region = region(slot);
        ClassicBatch.write(region, encoding.inputs);
      }

      @Override
      public void submit(DecisionDevicePipeline.Slot slot) {
        working = manager.newSubManager();
        try {
          copyStarted = true;
          ClassicBatch.Upload upload = ClassicBatch.enqueue(working, region, encoding.inputs, rows, slot.h2dStream());
          copied = true;
          computeStarted = true;
          stream = stream(slot);
          NDArray probabilities;
          try (PtStreamScope ignored = stream.openScope()) {
            ClassicBatch input = upload.receive();
            try (InferenceMode inference = manager.getEngine().newInferenceMode()) {
              probabilities = probabilities(input);
            }
            slot.computeReady().record();
            computeRecorded = true;
          }
          // コピー先のページ固定の配列は推論モードの外で作る。推論モードの中で作ると推論用のテンソルになり、
          // 完了のスレッドからのコピーが拒まれる
          slot.prepareOutput((PtNDArray) probabilities);
        } catch (RuntimeException | Error problem) {
          recordLatestCompletion(slot, problem);
          throw problem;
        }
      }

      @Override
      public boolean isComplete(DecisionDevicePipeline.Slot slot) {
        return slot.computeReady().isComplete();
      }

      @Override
      public void complete(DecisionDevicePipeline.Slot slot) {
        slot.outputFloats().get(encoding.probabilities, 0, rows * encoding.inputs.bucket().actions());
        answer.write(encoding, encoding.probabilities, rows);
      }

      @Override
      public void release(DecisionDevicePipeline.Slot slot) {
        if (working != null) {
          working.close();
          working = null;
        }
      }

      /** 失敗したバッチを後始末する。積んだコピーと順伝播の終わりを確かめられなければ false を返し、スロットは隔離される。 */
      @Override
      public boolean abort(DecisionDevicePipeline.Slot slot, Throwable problem) {
        fail(problem);
        if ((copyStarted && !copied && !copyFenced) || (computeStarted && !computeRecorded)) {
          problem.addSuppressed(new IllegalStateException("Classic stream completion could not be recorded"));
          return false;
        }
        try {
          if (computeRecorded) {
            slot.computeReady().synchronize();
          }
          if (copyFenced) {
            slot.h2dReady().synchronize();
          }
          if (region != null) {
            region.await();
          }
          release(slot);
          return true;
        } catch (Throwable cleanup) {
          problem.addSuppressed(cleanup);
          return false;
        }
      }

      @Override
      public void fail(Throwable problem) {
        if (failure == null) {
          failure = problem;
        } else if (failure != problem) {
          failure.addSuppressed(problem);
        }
      }

      @Override
      public void publish() {
        if (failure == null) {
          result.complete(null);
        } else {
          result.completeExceptionally(failure);
        }
      }

      /**
       * 提出に失敗しても、積んだ転送と順伝播の後にイベントを記録して、後始末で待てるようにする。積み終えたコピーの終わりはバッファが
       * 預かるので、転送のイベントはコピーを積む途中で失敗したときだけ記録する。
       */
      private void recordLatestCompletion(DecisionDevicePipeline.Slot slot, Throwable problem) {
        if (copyStarted && !copied) {
          try (PtStreamScope ignored = slot.h2dStream().openScope()) {
            slot.h2dReady().record();
            copyFenced = true;
          } catch (Throwable recording) {
            problem.addSuppressed(recording);
          }
        }
        if (computeStarted && !computeRecorded) {
          try (PtStreamScope ignored = (stream == null ? stream(slot) : stream).openScope()) {
            slot.computeReady().record();
            computeRecorded = true;
          } catch (Throwable recording) {
            problem.addSuppressed(recording);
          }
        }
      }
    }
  }

  /**
   * 1 つのスレッドの作業領域。入力を書くトークナイザー、バッチの判断の記録を読むリーダー、区分ごとの 1 行の入力、その場で推論する
   * 入力を送るバッファ、手の貪欲な選択、最後の確率を持つ。
   */
  static final class Workspace {

    private final ClassicSchema schema;
    private final ClassicTokenizer tokenizer;
    private final Observation reader;
    private final PolicyGraph.Greedy greedy = new PolicyGraph.Greedy();

    /** バッファを付けるパラメーターのマネージャー。方策を閉じるとバッファも返る。 */
    private final NDManager owner;

    /** [区分の通し番号] → 1 行の入力。初めて使うときに作る。 */
    private final ClassicInputs[] rows = new ClassicInputs[Buckets.count()];

    /** その場で推論する入力を送るバッファ。作る前は null。 */
    private PinnedUploads uploads;

    /** 最後に推論した行の合法手の確率。長さは区分の合法手の軸の大きさ。 */
    private float[] probabilities;

    private Workspace(Architecture architecture, Variant variant, ClassicSchema schema, NDManager owner) {
      this.schema = schema;
      this.owner = owner;
      tokenizer = new ClassicTokenizer(architecture, variant);
      reader = new Observation(variant);
    }

    /** バッファを返す。初めて呼ばれたときに、パラメーターのマネージャーに付けて作る。 */
    private PinnedUploads uploads() {
      if (uploads == null) {
        uploads = PinnedUploads.on(owner);
      }
      return uploads;
    }

    /** 判断が入る最小の区分の、1 行の入力を返す。 */
    private ClassicInputs row(Observation observation) {
      Bucket bucket = Buckets.required(observation);
      int ordinal = Buckets.ordinal(bucket);
      ClassicInputs inputs = rows[ordinal];
      if (inputs == null) {
        inputs = new ClassicInputs(schema, bucket, 1);
        rows[ordinal] = inputs;
      }
      return inputs;
    }
  }

  /** 入力のバッチ、GPU の確率のコピー先、手の貪欲な選択をまとめたもの。一度に 1 つのバッチだけが使う。 */
  private static final class Encoding {

    private final ClassicInputs inputs;
    private final int pool;

    /** GPU のバッチの確率をホストへコピーする先 [行, 合法手の軸]。CPU のバッチは推論ごとの配列を使い、ここは使わない。 */
    private final float[] probabilities;
    private final PolicyGraph.Greedy greedy = new PolicyGraph.Greedy();

    private Encoding(ClassicInputs inputs, int pool) {
      this.inputs = inputs;
      this.pool = pool;
      probabilities = new float[inputs.rows() * inputs.bucket().actions()];
    }

    /**
     * 先頭の行から順に、各行で貪欲に選んだ手を書く。
     *
     * @param probabilities 合法手の確率 [行, 合法手の軸]
     * @param rows 書く行数
     * @param chosen 行の順に貪欲に選んだ手を書き込む先
     */
    private void answer(float[] probabilities, int rows, int[] chosen) {
      int actions = inputs.bucket().actions();
      for (int row = 0; row < rows; row++) {
        chosen[row] = greedy.select(inputs, row, probabilities, row * actions);
      }
    }
  }

  private static Throwable closeAfter(AutoCloseable resource, Throwable failure) {
    try {
      resource.close();
    } catch (Exception | Error problem) {
      if (failure == null) {
        return problem;
      }
      failure.addSuppressed(problem);
    }
    return failure;
  }
}
