package com.epsilon.prism.policy;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.FrontierFeatures;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputWriter;
import com.epsilon.prism.input.Tokenizer;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.spi.PolicyExecutionContext;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.LongAdder;

/**
 * 問い合わせの塊を網の入力にする（{@link Encoding}）。入力は問い合わせが運ぶ記録 O だけで、捕捉は出す側が済ませている（{@link
 * PolicyRequest#capture}）。読み手の付けと符号化は符号化のプールで範囲ごとに並列に、塊の計画はその間に 1 つのスレッドで行う。CPU の網（プール
 * なし）では呼び出しスレッドで全てを行う。
 *
 * <p>作業域と入力の塊は池で使い回し、足りなければ足す（同時に受け付けた数だけ増えて止まる）。塊の上限は推論の塊
 * （{@link BatchPlan.Capacity#DEFAULT}）。blob の置き場は推論の道で決める（{@link Memory}）。道具（Tokenizer・書き手、前線の腕なら前線の書き手）は
 * 符号化のスレッドに 1 つずつ持つ（O の読み手は {@link Encoding} が行ごとに持つ）。前線の腕の網では、塊ごとに前線の塊も取る。
 */
final class BatchEncoder implements AutoCloseable {

  /** eager の 1 回の順伝播の詰めたトークンの上限（学習の微小バッチと同じ。関係の符号の [行, S, S] がデバイスに収まる大きさ）。 */
  static final int TOKEN_BUDGET = FitPlan.TOKEN_BUDGET;

  /** 詰めたトークンの上限なし（HIP は行を詰めずに並べるので、塊の容量だけで分ける）。 */
  static final int UNBOUNDED = Integer.MAX_VALUE;

  /** 入力の blob の置き場。 */
  enum Memory {
    /** ダイレクトの領域（eager は塊ごとに入力の配列へ写す）。 */
    DIRECT,
    /** ページ固定のホストの領域（HIP の H2D が非同期に読む）。前線の塊は持たない。 */
    PINNED
  }

  /** 符号化のスレッドの道具（どれもスレッド安全でない）。 */
  private static final class Tools {
    final Tokenizer tokenizer = new Tokenizer();
    final InputWriter writer = new InputWriter();
    final FrontierFeatures frontier;

    Tools(boolean frontier) {
      this.frontier = frontier ? new FrontierFeatures() : null;
    }
  }

  private final int budget;
  private final boolean frontier;
  private final Memory memory;
  private final PtNDManager pinned;
  private final ThreadLocal<Tools> tools;
  private final ConcurrentLinkedQueue<Encoding> encodings = new ConcurrentLinkedQueue<>();
  private final ConcurrentLinkedQueue<HostInputs> batches = new ConcurrentLinkedQueue<>();
  private final LongAdder nanos = new LongAdder();

  /**
   * @param budget 1 回の順伝播の詰めたトークンの上限（{@link #TOKEN_BUDGET} か {@link #UNBOUNDED}）
   * @param frontier 前線の腕の網なら true（前線の塊も書く）
   * @param memory blob の置き場
   * @throws IllegalArgumentException 最も長い 1 行が入らない上限か、ページ固定の置き場で前線の塊を求めたとき
   */
  BatchEncoder(int budget, boolean frontier, Memory memory) {
    if (budget < BatchPlan.FIXED_TOKENS + Limits.MAX_EVENT_TOKENS) {
      throw new IllegalArgumentException("token budget cannot hold the longest row: " + budget);
    }
    if (frontier && memory == Memory.PINNED) {
      throw new IllegalArgumentException("pinned input blobs do not carry frontier batches");
    }
    this.budget = budget;
    this.frontier = frontier;
    this.memory = memory;
    pinned =
        memory == Memory.PINNED
            ? (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME)
            : null;
    tools = ThreadLocal.withInitial(() -> new Tools(frontier));
  }

  /**
   * 作業域を借りる。結果を書き終えたら {@link #release} で返す。
   *
   * @param rows 問い合わせ（1 つ以上）
   * @return 作業域
   */
  Encoding acquire(List<PolicyRequest> rows) {
    Encoding encoding = encodings.poll();
    if (encoding == null) {
      encoding = new Encoding();
    }
    encoding.reset(rows);
    return encoding;
  }

  /**
   * 問い合わせを塊に書く。
   *
   * @param encoding 借りた作業域
   * @param execution 符号化のプール（null なら呼び出しスレッドで書く）
   * @return 全ての塊を書き終えたら完了する
   */
  CompletableFuture<Void> encode(Encoding encoding, PolicyExecutionContext execution) {
    int rows = encoding.rows().size();
    return ranges(execution, rows, encoding::bind)
        .thenCompose(
            ignored -> {
              long started = System.nanoTime();
              encoding.plan(this::batch, budget);
              nanos.add(System.nanoTime() - started);
              return ranges(
                  execution,
                  rows,
                  (from, to) -> {
                    Tools own = tools.get();
                    encoding.tokenize(from, to, own.tokenizer, own.writer, own.frontier);
                  });
            });
  }

  /** 作業域と塊を池に返す（塊の blob を読む仕事が全て終わった後に呼ぶ）。 */
  void release(Encoding encoding) {
    for (Encoding.Chunk chunk : encoding.chunks()) {
      batches.add(chunk.batch());
    }
    encoding.clear();
    encodings.add(encoding);
  }

  /** 付け・計画・符号化にかけたスレッドの秒の和。 */
  double seconds() {
    return nanos.sum() / 1e9;
  }

  /** 範囲に分けて書く（各範囲の時間を足す）。 */
  private CompletableFuture<Void> ranges(
      PolicyExecutionContext execution, int rows, PolicyExecutionContext.RangeEncoder range) {
    PolicyExecutionContext.RangeEncoder timed =
        (from, to) -> {
          long started = System.nanoTime();
          try {
            range.encode(from, to);
          } finally {
            nanos.add(System.nanoTime() - started);
          }
        };
    return execution == null
        ? CompletableFuture.runAsync(() -> timed.encode(0, rows), Runnable::run)
        : execution.encodeRowsAsync(rows, timed);
  }

  /** 空いた塊を借りる（なければ新しく取る）。計画は空にして渡す。 */
  private HostInputs batch() {
    HostInputs batch = batches.poll();
    if (batch == null) {
      batch =
          switch (memory) {
            case DIRECT -> HostInputs.allocate(BatchPlan.Capacity.DEFAULT, frontier);
            case PINNED -> {
              int bytes = BatchPlan.maxBytes(BatchPlan.Capacity.DEFAULT);
              yield new HostInputs(
                  new HostBatch(
                      BatchPlan.Capacity.DEFAULT,
                      pinned.allocatePinned(bytes, DataType.UINT8).getByteBuffer()),
                  null);
            }
          };
    }
    batch.plan().clear();
    return batch;
  }

  /** 池を空にし、ページ固定の置き場を返す（blob を読むデバイスの仕事が全て終わった後に呼ぶ）。 */
  @Override
  public void close() {
    encodings.clear();
    batches.clear();
    if (pinned != null) {
      pinned.close();
    }
  }
}
