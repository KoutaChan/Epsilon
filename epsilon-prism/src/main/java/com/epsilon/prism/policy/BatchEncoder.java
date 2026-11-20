package com.epsilon.prism.policy;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.prism.Limits;
import com.epsilon.prism.backend.EventPages;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.learn.FitPlan;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.IntFunction;

/**
 * 受け付けた方策の要求を、ネットワークの入力のバッチにする（{@link Encoding}）。入力は要求が運ぶ記録 O だけで、O の取り込みは
 * 要求を出す側が済ませている。O にリーダーを付ける処理と符号化は、符号化のプールで範囲ごとに並列に行う。この 2 つの間に、バッチの
 * 計画を 1 つのスレッドで立てる。CPU のネットワーク（プールなし）では、全てを呼び出しスレッドで行う。
 *
 * <p>専用カーネルの入力とページ配置は受付とスロットの数に合わせて先に確保する。各受付の分割は {@link BatchPlan.Capacity#standard} の上限を使う。入力の形式の
 * 1 行のライターは、符号化のスレッドごとに 1 つ持つ（O のリーダーは {@link Encoding} が行ごとに持つ）。
 */
final class BatchEncoder implements AutoCloseable {

  /**
   * eager の 1 回の順伝播で扱うトークン数（パディング込み）の上限。学習のマイクロバッチと同じ値で、関係のコード [行, S, S] がデバイスに
   * 収まる大きさ。
   */
  static final int TOKEN_BUDGET = FitPlan.TOKEN_BUDGET;

  /**
   * eager の 1 回の順伝播で扱うトークン数（パディング込み）の上限を、デバイスに合わせて決める。GPU ではスロットごとの計算ストリームで順伝播が
   * 同時に走り、PyTorch のメモリのキャッシュもストリームごとに分かれる。そのため {@link #TOKEN_BUDGET}（デバイス 1 つ分）をスロットの数で割る。
   *
   * @param device デバイス
   * @param runtime 実行のリソース（GPU のときだけスロットの数を読む）
   * @return 1 回の順伝播で扱うトークン数（パディング込み）の上限
   */
  static int eagerBudget(Device device, NetworkRuntime runtime) {
    return device.isGpu() ? TOKEN_BUDGET / runtime.slots() : TOKEN_BUDGET;
  }

  /** トークン数の上限なし。専用カーネルは行をパディングせずに並べるので、バッチの容量だけで分ける。 */
  static final int UNBOUNDED = Integer.MAX_VALUE;

  /** 入力の blob を置くバッファの種類。 */
  enum Memory {
    /** ダイレクトバッファ。eager はバッチごとに入力の配列へコピーする。 */
    DIRECT,
    /** ページ固定のホストメモリ。デバイスへの非同期転送で使う。 */
    PINNED
  }

  /** 符号化のスレッドごとのヘルパー（どれもスレッドセーフでない）。 */
  private static final class Tools {
    final InputFormat.RowWriter writer;

    Tools(InputFormat format) {
      writer = format.writer();
    }
  }

  final boolean eventPages;
  private final ConcurrentHashMap<HostInputs, EventPages> pageBatches = new ConcurrentHashMap<>();
  private final InputFormat format;
  private final BatchPlan.Capacity capacity;
  private final int budget;
  private final PtNDManager pinned;
  private final IntFunction<ByteBuffer> blobs;
  private final ThreadLocal<Tools> tools;
  private final ConcurrentLinkedQueue<Encoding> encodings = new ConcurrentLinkedQueue<>();
  private final ConcurrentLinkedQueue<HostInputs> batches = new ConcurrentLinkedQueue<>();
  private boolean reserved;
  private boolean closed;
  private CompletableFuture<Void> available;
  private final LongAdder nanos = new LongAdder();

  /**
   * @param format 入力の形式
   * @param budget 1 回の順伝播で扱うトークン数（パディング込み）の上限（{@link #TOKEN_BUDGET} か {@link #UNBOUNDED}）
   * @param memory blob を置くバッファの種類
   * @throws IllegalArgumentException 最も長い 1 行が上限に入らないとき
   */
  BatchEncoder(InputFormat format, int budget, Memory memory) {
    this(format, budget, memory, false);
  }

  BatchEncoder(InputFormat format, int budget, Memory memory, boolean eventPages) {
    if (eventPages && !format.eventsSettleOnPublication()) {
      throw new IllegalArgumentException("event pages require immutable public event columns");
    }
    this.eventPages = eventPages;
    if (budget < format.fixedTokens() + format.maxEvents()) {
      throw new IllegalArgumentException("token budget cannot hold the longest row: " + budget);
    }
    this.format = format;
    capacity = BatchPlan.Capacity.standard(format);
    this.budget = budget;
    pinned =
        memory == Memory.PINNED
            ? (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME)
            : null;
    blobs =
        switch (memory) {
          case DIRECT -> ByteBuffer::allocateDirect;
          case PINNED -> bytes -> pinned.allocatePinned(bytes, DataType.UINT8).getByteBuffer();
        };
    tools = ThreadLocal.withInitial(() -> new Tools(format));
  }

  /**
   * 受付中と実行中のバッチが使う入力とページ配置を、公開前に確保する。
   *
   * @throws IllegalArgumentException 受付か行の上限が 1 未満のときと、バッチの容量に全ての軸で最大の行が 1 つも入らないとき
   */
  void reserve(int admissions, int maxRows) {
    if (admissions < 1 || maxRows < 1) throw new IllegalArgumentException("invalid input capacity");
    int events = format.maxEvents();
    int minimumRows =
        Math.min(capacity.rows(), budget / (format.fixedTokens() + format.maxEvents()));
    minimumRows = Math.min(minimumRows, capacity.events() / events);
    minimumRows = Math.min(minimumRows, capacity.transitions() / Limits.MAX_TRANSITIONS);
    minimumRows = Math.min(minimumRows, capacity.actions() / Limits.MAX_LEGAL_ACTIONS);
    if (minimumRows < 1)
      throw new IllegalArgumentException("the batch capacity cannot hold the largest row");
    // 最後以外の分割には、各行が全ての軸で最大でも minimumRows 行が入る。
    for (int cell = 0; cell < admissions; ++cell) {
      var inputs = new ArrayList<HostInputs>();
      for (int remaining = maxRows; remaining > 0; remaining -= minimumRows) {
        int rows = Math.min(remaining, Math.min(capacity.rows(), budget / format.fixedTokens()));
        var limits =
            new BatchPlan.Capacity(
                rows,
                (int)
                    Math.max(
                        Limits.MAX_EVENT_TOKENS, Math.min(capacity.events(), (long) rows * events)),
                (int) Math.min(capacity.transitions(), (long) rows * Limits.MAX_TRANSITIONS),
                (int) Math.min(capacity.actions(), (long) rows * Limits.MAX_LEGAL_ACTIONS));
        inputs.add(allocate(limits));
      }
      encodings.add(new Encoding(format, inputs::get, eventPages ? pageBatches::get : null));
    }
    reserved = true;
  }

  /** 入力に空きがなければ、返却時に完了する通知を返す。 */
  synchronized CompletableFuture<Void> whenFull() {
    if (closed) throw new IllegalStateException("batch encoder is closed");
    if (!reserved || !encodings.isEmpty()) return null;
    if (available == null) available = new CompletableFuture<>();
    return available;
  }

  /** 受付が所有する作業領域を一つ借りる。 */
  synchronized Encoding borrow() {
    if (closed) throw new IllegalStateException("batch encoder is closed");
    Encoding encoding = encodings.poll();
    if (encoding != null) return encoding;
    if (reserved) throw new IllegalStateException("no reserved input available");
    return new Encoding(format, ignored -> batch(), eventPages ? pageBatches::get : null);
  }

  /** 呼び出しスレッドの作業領域を借り、要求を付ける。 */
  Encoding acquire(List<PolicyRequest> rows) {
    Encoding encoding = borrow();
    try {
      encoding.reset(rows);
      return encoding;
    } catch (RuntimeException | Error failure) {
      release(encoding);
      throw failure;
    }
  }

  /**
   * 要求をバッチに書く。
   *
   * @param encoding 借りた作業領域
   * @param runtime 符号化のプールを持つ実行のリソース（null なら呼び出しスレッドで書く）
   * @return 全てのバッチを書き終えたら完了する future
   */
  CompletableFuture<Void> encode(Encoding encoding, NetworkRuntime runtime) {
    int rows = encoding.rows().size();
    return ranges(runtime, rows, encoding::bind)
        .thenCompose(
            ignored -> {
              long started = System.nanoTime();
              encoding.plan(budget);
              nanos.add(System.nanoTime() - started);
              return ranges(
                  runtime,
                  rows,
                  (from, to) -> {
                    Tools own = tools.get();
                    encoding.tokenize(from, to, own.writer);
                  });
            });
  }

  /** 作業領域とバッチをプールに返す。バッチの blob を読む処理が全て終わった後に呼ぶ。 */
  void release(Encoding encoding) {
    for (Encoding.Chunk chunk : encoding.chunks()) {
      if (chunk.pages() != null) {
        chunk.pages().clear(chunk.batch().plan().rows());
        chunk.batch().input().releaseEvents();
      }
      if (!reserved) batches.add(chunk.batch());
    }
    encoding.clear();
    CompletableFuture<Void> wake;
    synchronized (this) {
      encodings.add(encoding);
      wake = available;
      available = null;
    }
    if (wake != null) wake.completeAsync(() -> null);
  }

  /** リーダーを付ける処理・計画・符号化に、各スレッドがかけた時間の合計（秒）。 */
  double seconds() {
    return nanos.sum() / 1e9;
  }

  /** 範囲に分けて書く。各範囲にかかった時間を足す。 */
  private CompletableFuture<Void> ranges(
      NetworkRuntime runtime, int rows, NetworkRuntime.RangeEncoder range) {
    NetworkRuntime.RangeEncoder timed =
        (from, to) -> {
          long started = System.nanoTime();
          try {
            range.encode(from, to);
          } finally {
            nanos.add(System.nanoTime() - started);
          }
        };
    return runtime == null
        ? CompletableFuture.runAsync(() -> timed.encode(0, rows), Runnable::run)
        : runtime.encodeRows(rows, timed);
  }

  private HostInputs batch() {
    HostInputs batch = batches.poll();
    return batch != null ? batch : allocate(capacity);
  }

  private HostInputs allocate(BatchPlan.Capacity limits) {
    HostInputs batch = HostInputs.allocate(format, limits, blobs);
    if (eventPages) {
      batch.input().enablePublishedEvents();
      pageBatches.put(batch, new EventPages(format, limits, blobs));
    }
    return batch;
  }

  /** プールを空にし、ページ固定のバッファを解放する。blob を読むデバイスの処理が全て終わった後に呼ぶ。 */
  @Override
  public void close() {
    CompletableFuture<Void> wake;
    synchronized (this) {
      closed = true;
      encodings.clear();
      wake = available;
      available = null;
    }
    if (wake != null)
      wake.completeExceptionally(new IllegalStateException("batch encoder is closed"));
    batches.clear();
    pageBatches.clear();
    if (pinned != null) {
      pinned.close();
    }
  }
}
