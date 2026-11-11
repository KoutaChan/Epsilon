package com.epsilon.agent.play;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.policy.BatchedPolicy;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * agent の方策を卓の席に座らせる。打ち手の分布は方策が選んだ手だけを 1 にしたもので、価値は書かない。
 *
 * <p>まとめて推論する方策では、受付の区分、区分ごとの上限、予約、実行先の空きの判断を方策に任せる。卓が集めたバッチの記録を
 * 送り出し用のバッチにコピーして、方策に選ばせる。その場で答える方策は、進行のスレッドで問い合わせの記録から手を選ぶ。
 */
public final class PolicySeat implements TableSeat<Query> {

  /** 送り出し用のバッチを確保するときに見込む、1 行あたりの {@code long} の数。 */
  private static final int LONGS_PER_ROW = 64;

  private final Policy policy;
  private final ThreadLocal<Observation> readers;

  /**
   * @param policy 席に座らせる方策。所有権を受け取り、{@link #close} で閉じる
   */
  public PolicySeat(Policy policy) {
    this.policy = Objects.requireNonNull(policy, "policy");
    readers = ThreadLocal.withInitial(() -> new Observation(policy.variant()));
  }

  /** 結果の行に付ける名前。 */
  public String label() {
    return policy.label();
  }

  @Override
  public boolean inline() {
    return policy instanceof InlinePolicy;
  }

  @Override
  public void evaluateInline(Query query) {
    if (!(policy instanceof InlinePolicy inline)) {
      throw new UnsupportedOperationException(policy.label() + " is not inline");
    }
    query.sigma = oneHot(query.legalCount(), inline.act(query.bind(readers.get())));
  }

  @Override
  public int admission(Observation observation) {
    return policy instanceof BatchedPolicy batched ? batched.admission(observation) : 0;
  }

  @Override
  public int maxBatch(int admission) {
    return batched().maxBatch(admission);
  }

  @Override
  public boolean needsWork() {
    return !(policy instanceof BatchedPolicy batched) || batched.needsWork();
  }

  @Override
  public DecisionInferenceIngress.Attempt tryAcquireIngress(int admission) {
    return batched().tryAcquire(admission);
  }

  @Override
  public CompletableFuture<Void> submit(List<Query> batch, DecisionInferenceIngress ingress) {
    BatchedPolicy batched = batched();
    ObservationBatch out;
    try {
      Observation reader = readers.get();
      out = new ObservationBatch(policy.variant(), Math.max(1, batch.size() * LONGS_PER_ROW));
      for (Query query : batch) out.add(query.bind(reader));
    } catch (RuntimeException | Error failure) {
      ingress.abort();
      throw failure;
    }
    int[] chosen = new int[batch.size()];
    return batched
        .choose(out, chosen, ingress)
        .thenRun(
            () -> {
              for (int row = 0; row < batch.size(); row++) {
                Query query = batch.get(row);
                query.sigma = oneHot(query.legalCount(), chosen[row]);
              }
            });
  }

  @Override
  public void close() {
    policy.close();
  }

  private BatchedPolicy batched() {
    if (!(policy instanceof BatchedPolicy batched)) {
      throw new UnsupportedOperationException(policy.label() + " answers inline");
    }
    return batched;
  }

  private static float[] oneHot(int n, int chosen) {
    float[] sigma = new float[n];
    sigma[Objects.checkIndex(chosen, n)] = 1f;
    return sigma;
  }
}
