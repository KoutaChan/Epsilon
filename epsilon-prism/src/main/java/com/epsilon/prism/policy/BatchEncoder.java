package com.epsilon.prism.policy;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.learn.FitPlan;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.IntFunction;

/**
 * 受け付けた方策の要求を、ネットワークの入力のバッチにする（{@link Encoding}）。入力は要求が運ぶ記録 O だけで、O の取り込みは
 * 要求を出す側が済ませている。O にリーダーを付ける処理と符号化は、符号化のプールで範囲ごとに並列に行う。この 2 つの間に、バッチの
 * 計画を 1 つのスレッドで立てる。CPU のネットワーク（プールなし）では、全てを呼び出しスレッドで行う。
 *
 * <p>作業領域と入力のバッチはプールで使い回し、足りなければ足す（同時に受け付けた数まで増えて止まる）。バッチの容量は推論のバッチと
 * 同じ（{@link BatchPlan.Capacity#standard}）。blob のバッファの種類は推論のバックエンドで決める（{@link Memory}）。入力の形式の
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

  /** トークン数の上限なし。HIP は行をパディングせずに並べるので、バッチの容量だけで分ける。 */
  static final int UNBOUNDED = Integer.MAX_VALUE;

  /** 入力の blob を置くバッファの種類。 */
  enum Memory {
    /** ダイレクトバッファ。eager はバッチごとに入力の配列へコピーする。 */
    DIRECT,
    /** ページ固定のホストメモリ。HIP の H2D が非同期に読む。 */
    PINNED
  }

  /** 符号化のスレッドごとのヘルパー（どれもスレッドセーフでない）。 */
  private static final class Tools {
    final InputFormat.RowWriter writer;

    Tools(InputFormat format) {
      writer = format.writer();
    }
  }

  private final InputFormat format;
  private final BatchPlan.Capacity capacity;
  private final int budget;
  private final PtNDManager pinned;
  private final IntFunction<ByteBuffer> blobs;
  private final ThreadLocal<Tools> tools;
  private final ConcurrentLinkedQueue<Encoding> encodings = new ConcurrentLinkedQueue<>();
  private final ConcurrentLinkedQueue<HostInputs> batches = new ConcurrentLinkedQueue<>();
  private final LongAdder nanos = new LongAdder();

  /**
   * @param format 入力の形式
   * @param budget 1 回の順伝播で扱うトークン数（パディング込み）の上限（{@link #TOKEN_BUDGET} か {@link #UNBOUNDED}）
   * @param memory blob を置くバッファの種類
   * @throws IllegalArgumentException 最も長い 1 行が上限に入らないとき
   */
  BatchEncoder(InputFormat format, int budget, Memory memory) {
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
   * 作業領域を借りる。結果を書き終えたら {@link #release} で返す。
   *
   * @param rows 方策の要求（1 つ以上）
   * @return 作業領域
   */
  Encoding acquire(List<PolicyRequest> rows) {
    Encoding encoding = encodings.poll();
    if (encoding == null) {
      encoding = new Encoding(format);
    }
    encoding.reset(rows);
    return encoding;
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
              encoding.plan(this::batch, budget);
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
      batches.add(chunk.batch());
    }
    encoding.clear();
    encodings.add(encoding);
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

  /** 空いたバッチを借りる（無ければ新しく確保する）。計画は空にして渡す。 */
  private HostInputs batch() {
    HostInputs batch = batches.poll();
    if (batch == null) {
      batch = HostInputs.allocate(format, capacity, blobs);
    }
    batch.plan().clear();
    return batch;
  }

  /** プールを空にし、ページ固定のバッファを解放する。blob を読むデバイスの処理が全て終わった後に呼ぶ。 */
  @Override
  public void close() {
    encodings.clear();
    batches.clear();
    if (pinned != null) {
      pinned.close();
    }
  }
}
