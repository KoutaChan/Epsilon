package com.epsilon.prism.eval;

import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.policy.Capacity;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * 評価の合議の席（測定専用）: 2 つの網の方策 a と b の s を重み w で混ぜた貪欲（s = (1 − w)·s_a + w·s_b、同点は p0、それも同じなら番号の小さい手）。
 * 系譜の違う網の、系譜に固有の誤りが打ち消し合うかを見る。どちらも s を書く網の方策で、s は同じ効用の単位であること。
 *
 * <p>受付の容量は a のものを使い（{@link #tryAcquireIngress}）、b には同じ判断の写しの問い合わせを、b の容量が空いたときに出す。問い合わせの観測と
 * 合法手は両方の結果がそろうまで持ち主の対局が進めないので、写しから読んでよい。行は作らない（評価だけ）。
 */
public final class EnsembleService implements PolicyService {
  private final PolicyService a;
  private final PolicyService b;
  private final double weight;

  /**
   * @param a 主の方策（受付の容量と版の番号はこちら）
   * @param b 混ぜる方策
   * @param weight b の重み w（[0, 1]）
   */
  public EnsembleService(PolicyService a, PolicyService b, double weight) {
    if (!(weight >= 0 && weight <= 1)) throw new IllegalArgumentException("weight must be in [0, 1]: " + weight);
    if (a.inline() || b.inline()) throw new IllegalArgumentException("Ensemble members must be network policies");
    this.a = a;
    this.b = b;
    this.weight = weight;
  }

  @Override
  public int version() {
    return a.version();
  }

  @Override
  public boolean inline() {
    return false;
  }

  /** どちらかが記録 O を読むなら、問い合わせに O を載せてもらう（写しも同じ O を指す）。 */
  @Override
  public boolean readsRecord() {
    return a.readsRecord() || b.readsRecord();
  }

  @Override
  public int maxBatch(Capacity capacity) {
    return Math.min(a.maxBatch(capacity), b.maxBatch(capacity));
  }

  @Override
  public boolean needsWork() {
    return a.needsWork() || b.needsWork();
  }

  @Override
  public DecisionInferenceIngress.Attempt tryAcquireIngress() {
    return a.tryAcquireIngress();
  }

  @Override
  public CompletableFuture<Void> submit(List<PolicyRequest> batch, DecisionInferenceIngress ingress) {
    List<PolicyRequest> copies = new ArrayList<>(batch.size());
    for (PolicyRequest request : batch) copies.add(request.copy(false));
    CompletableFuture<Void> first = a.submit(batch, ingress);
    CompletableFuture<Void> second = submitOther(copies);
    return CompletableFuture.allOf(first, second).thenRun(() -> mix(batch, copies));
  }

  /** b の容量を取って写しを出す（空いていなければ空きの通知を待って取り直す）。 */
  private CompletableFuture<Void> submitOther(List<PolicyRequest> copies) {
    DecisionInferenceIngress.Attempt attempt = b.tryAcquireIngress();
    if (attempt.acquired()) return b.submit(copies, attempt.ingress());
    return attempt.available().thenCompose(ignored -> submitOther(copies));
  }

  private void mix(List<PolicyRequest> batch, List<PolicyRequest> copies) {
    for (int i = 0; i < batch.size(); i++) {
      PolicyRequest request = batch.get(i);
      float[] own = request.scores, other = copies.get(i).scores;
      if (own == null || other == null) throw new IllegalStateException("Ensemble members must write scores");
      int n = request.legal().size();
      float[] mixed = new float[n];
      for (int k = 0; k < n; k++) mixed[k] = (float) ((1 - weight) * own[k] + weight * other[k]);
      float[] sigma = new float[n];
      sigma[Greedy.argmax(mixed, request.prior, n)] = 1f;
      request.scores = mixed;
      request.sigma = sigma;
    }
  }
}
