package com.epsilon.prism.policy;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.RowTokens;
import com.epsilon.prism4p.input.InputSchema;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 本番と同じ符号化の経路を通す、inline でないテスト用の方策。ネットワークのバージョンの方策（{@link NetworkPolicy}）と同じく、要求が
 * 運ぶ記録 O だけを読む（O は収集を進めるスレッドが取り込む）。受付は {@link BatchEncoder} で受け取り、計画を立てて符号化する。
 * トークンの上限は、本番の HIP と同じく設けない。答えは {@link Encoding#answer} で s・p0・σ を書く。
 *
 * <p>s はネットワークで求めず、符号化した行のハッシュ（{@link RowTokens#row}）と手の位置だけで決める。だから s はバッチの組み方に
 * 依らず、収集は決定的になり、入力のバイトが変われば打ち方も変わる。
 *
 * <p>受付ごとに、バッチの行のハッシュと、要求が運んだ O のハッシュを、それぞれ順に依らない和に足す。また、同じ O を 1 行のバッチに
 * 書いた入力のハッシュの和が、バッチの行のハッシュの和と同じことを確かめる。これで、符号化の書き込みが行の中身だけで決まることを確かめる。
 */
public final class ServedGoldenPolicy implements PolicyService {
  private final BatchEncoder encoder =
      new BatchEncoder(InputSchema.FORMAT, BatchEncoder.UNBOUNDED, BatchEncoder.Memory.DIRECT);

  /** スレッドごとの 1 行のライター（スレッドセーフでない）。 */
  private final ThreadLocal<RowTokens> oneRow = ThreadLocal.withInitial(RowTokens::new);

  private final ThreadLocal<Observation> reader = ThreadLocal.withInitial(() -> new Observation(TENHOU_4P));
  private final PriorSettings prior;
  private final AtomicLong observations = new AtomicLong();
  private final AtomicLong tokens = new AtomicLong();
  private final AtomicLong queries = new AtomicLong();

  /**
   * @param prior p0 の係数
   */
  public ServedGoldenPolicy(PriorSettings prior) {
    this.prior = prior;
  }

  /** 要求ごとの O のハッシュの和（順に依らない）。 */
  public long observations() {
    return observations.get();
  }

  /** 符号化したバッチの行のハッシュの和（順に依らない）。 */
  public long tokens() {
    return tokens.get();
  }

  /** 答えた要求の数。 */
  public long queries() {
    return queries.get();
  }

  @Override
  public int version() {
    return 1;
  }

  @Override
  public boolean inline() {
    return false;
  }

  /** ネットワークのバージョンの方策と同じく、全ての行を 1 つの区分で受け付ける。 */
  @Override
  public int admission(Observation observation) {
    return 0;
  }

  @Override
  public CompletableFuture<Void> submit(List<PolicyRequest> rows, DecisionInferenceIngress ingress) {
    ingress.handoff(this).move();
    Encoding encoding = encoder.acquire(rows);
    try {
      encoder.encode(encoding, null).join();
      long served = 0;
      FloatBuffer scores = FloatBuffer.allocate(encoding.actions());
      for (Encoding.Chunk chunk : encoding.chunks()) {
        HostBatch batch = chunk.batch().input();
        BatchPlan plan = batch.plan();
        for (int row = 0; row < plan.rows(); row++) {
          long bytes = RowTokens.row(batch, row);
          served += SeedMixer.mix64(bytes);
          for (int action = 0; action < plan.actions(row); action++) scores.put(score(bytes, action));
        }
      }
      encoding.answer(scores, prior);
      RowTokens own = oneRow.get();
      long captured = 0;
      long written = 0;
      for (PolicyRequest request : rows) {
        long[] words = request.recordWords();
        int base = request.recordBase();
        captured += SeedMixer.mix64(observation(words, base));
        written += SeedMixer.mix64(own.tokenize(words, base));
      }
      if (written != served) {
        throw new IllegalStateException("the served rows differ from one-row encodings of the captured observations");
      }
      observations.addAndGet(captured);
      tokens.addAndGet(served);
      queries.addAndGet(rows.size());
      return CompletableFuture.completedFuture(null);
    } catch (RuntimeException | Error failure) {
      return CompletableFuture.failedFuture(failure);
    } finally {
      encoder.release(encoding);
    }
  }

  @Override
  public void close() {
    encoder.close();
  }

  /** 行のハッシュと手の位置で決まる s。[-1, 1) を 2^-23 刻みにした値で、float で厳密に表せる。 */
  private static float score(long row, int action) {
    long bits = SeedMixer.mix64(row + action) >>> 40;
    return (bits - (1 << 23)) * 0x1p-23f;
  }

  /** O のハッシュ。{@code base} から O の長さの分の {@code long} を読む。 */
  private long observation(long[] words, int base) {
    int longs = reader.get().bind(words, base).longs();
    long hash = SeedMixer.mix64(longs);
    for (int word = base; word < base + longs; word++) hash = SeedMixer.mix64(hash ^ words[word]);
    return hash;
  }
}
