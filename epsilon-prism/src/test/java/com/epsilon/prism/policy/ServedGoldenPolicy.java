package com.epsilon.prism.policy;

import com.epsilon.engine.GoldenHash;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.RowTokens;
import com.epsilon.prism.observe.ObservationSchema;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 本番の符号化の道を通す inline でない試験の方策。網の版の方策（{@link NetworkPolicy}）と同じく問い合わせが運ぶ記録 O だけを読み
 * （{@link #readsRecord}。O は収集の進行のスレッドが捕捉する）、受け付けを {@link BatchEncoder} で付け・計画・符号化し（本番の HIP と同じ詰めた
 * トークンの上限なし）、{@link Encoding#answer} で s・p0・σ を書く。網の代わりに、s は符号化した行のハッシュ（{@link RowTokens#row}）と
 * 手の位置だけで決まる値にする。だから s は塊の組み方に依らず、収集は決定的で、入力のバイトが変われば打ち方も変わる。
 *
 * <p>受け付けごとに、塊の行のハッシュと問い合わせが運んだ O のハッシュを順に依らない和に足し、その O を 1 行の塊に書いた入力の和が塊の行の
 * 和と同じことを確かめる（符号化の書き込みが、行の中身だけで決まることの確かめ）。
 */
public final class ServedGoldenPolicy implements PolicyService {
  private final BatchEncoder encoder =
      new BatchEncoder(BatchEncoder.UNBOUNDED, false, BatchEncoder.Memory.DIRECT);

  /** スレッドの 1 行の書き手（スレッド安全でない）。 */
  private final ThreadLocal<RowTokens> oneRow = ThreadLocal.withInitial(RowTokens::new);
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

  /** 問い合わせごとの O のハッシュの和（順に依らない）。 */
  public long observations() {
    return observations.get();
  }

  /** 符号化した塊の行のハッシュの和（順に依らない）。 */
  public long tokens() {
    return tokens.get();
  }

  /** 答えた問い合わせの数。 */
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

  @Override
  public boolean readsRecord() {
    return true;
  }

  /** 網の版の方策と同じく、全ての行を 1 つの区分で受け付ける。 */
  @Override
  public Capacity admissionKey(PolicyRequest request) {
    return Capacity.SINGLE_TRANSITION;
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
          served += GoldenHash.mix(bytes);
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
        captured += GoldenHash.mix(observation(words, base));
        written += GoldenHash.mix(own.tokenize(words, base));
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

  /** 行のハッシュと手の位置で決まる s（[-1, 1) の 2^-23 の格子。float で厳密）。 */
  private static float score(long row, int action) {
    long bits = GoldenHash.mix(row + action) >>> 40;
    return (bits - (1 << 23)) * 0x1p-23f;
  }

  /** O（{@code base} から {@link ObservationSchema#longs} の数の {@code long}）のハッシュ。 */
  private static long observation(long[] words, int base) {
    GoldenHash hash = new GoldenHash();
    int longs = ObservationSchema.longs(words, base);
    hash.add(longs);
    for (int word = base; word < base + longs; word++) hash.add(words[word]);
    return hash.value();
  }
}
