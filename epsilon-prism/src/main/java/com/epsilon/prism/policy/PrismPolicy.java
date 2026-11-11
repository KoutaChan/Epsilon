package com.epsilon.prism.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.policy.BatchedPolicy;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.TwinDiscard;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.concurrent.CompletableFuture;

/**
 * 公開したバージョンを、agent の方策として打たせる（Reviewer、クライアント、ほかの系列との卓）。手は s で貪欲に選び、収集のように一様分布を
 * 混ぜることはしない。貪欲な手が同値な打牌なら、組の中で切る方の手（{@link TwinDiscard}）を打つ。
 *
 * <p>受付の区分は 1 つだけ。まとめた推論は、バージョンのネットワークの方策（{@link NetworkPolicy}）にそのまま渡す。1 判断の評価
 * （{@link #evaluate}）は、1 行のバッチを排他制御の中で 1 判断ずつ通し、受付が満杯なら空くまで待つ。価値は s、確率は表示用の温度での
 * softmax(s/τ)。貪欲な手は、まとめた推論と同じ手で、s が最大の手（同点なら p0 の大きい手、それも同じなら番号の小さい手）を、切る方の手に
 * 替えたもの。
 */
public final class PrismPolicy implements BatchedPolicy {

  /** 確率を表示する温度 τ（効用の単位）。 */
  public static final double DISPLAY_TEMPERATURE = 0.002;

  /** 1 判断のバッチが初めに確保する {@code long} の数。 */
  private static final int SINGLE_LONGS = 64;

  private final NetworkPolicy network;
  private final Variant variant;
  private final String label;
  private final NetworkRuntime owned;
  private final TwinDiscard twinDiscard;
  private final ThreadLocal<Observation> readers;
  private final ObservationBatch single;
  private final Observation singleReader;

  /**
   * @param network バージョンのネットワークの方策（所有権を受け取る）
   * @param variant 打つルール。ネットワークの形式のルールと同じでなければならない
   * @param label 結果の行の名前
   * @param owned この方策が持つ実行のリソース（呼び出し元のリソースで動かすなら null）。ネットワークの方策を閉じた後に閉じる
   * @param twinDiscard 同値な打牌のうち切る方
   */
  public PrismPolicy(
      NetworkPolicy network, Variant variant, String label, NetworkRuntime owned, TwinDiscard twinDiscard) {
    this.network = Objects.requireNonNull(network, "network");
    this.variant = Objects.requireNonNull(variant, "variant");
    if (network.format().variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + variant + " but found " + network.format().variant() + " in " + label);
    }
    this.label = Objects.requireNonNull(label, "label");
    this.owned = owned;
    this.twinDiscard = Objects.requireNonNull(twinDiscard, "twinDiscard");
    readers = ThreadLocal.withInitial(() -> new Observation(variant));
    single = new ObservationBatch(variant, SINGLE_LONGS);
    singleReader = new Observation(variant);
  }

  @Override
  public Variant variant() {
    return variant;
  }

  @Override
  public String label() {
    return label;
  }

  @Override
  public int admission(Observation observation) {
    return 0;
  }

  @Override
  public int maxBatch(int admission) {
    return network.maxBatch(admission);
  }

  @Override
  public DecisionInferenceIngress.Attempt tryAcquire(int admission) {
    return network.tryAcquireIngress(admission);
  }

  @Override
  public boolean needsWork() {
    return network.needsWork();
  }

  @Override
  public CompletableFuture<Void> choose(
      ObservationBatch batch, int[] chosen, DecisionInferenceIngress reservation) {
    if (batch.variant() != variant) {
      reservation.abort();
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + batch.variant());
    }
    Observation reader = readers.get();
    List<PolicyRequest> rows = new ArrayList<>(batch.size());
    for (int index = 0; index < batch.size(); index++) {
      rows.add(new PolicyRequest(this, 0, index, null, null, false, 0, batch, index, reader.bind(batch, index)));
    }
    return network
        .submit(rows, reservation)
        .thenRun(
            () -> {
              for (int index = 0; index < rows.size(); index++) {
                PolicyRequest row = rows.get(index);
                chosen[index] = twinDiscard.play(row.legal(), Greedy.argmax(row.sigma, null, row.sigma.length));
              }
            });
  }

  @Override
  public synchronized Evaluation evaluate(Observation observation) {
    if (observation.variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + variant + " but found " + observation.variant());
    }
    int n = observation.legalActions();
    if (n == 1) {
      return new Evaluation(new float[] {1}, new float[] {0}, 0);
    }
    single.clear();
    int index = single.add(observation);
    var request = new PolicyRequest(this, 0, 0, null, null, false, 0, single, index, singleReader.bind(single, index));
    while (true) {
      var attempt = network.tryAcquireIngress(0);
      if (attempt.acquired()) {
        network.submit(List.of(request), attempt.ingress()).join();
        break;
      }
      attempt.available().join();
    }
    float[] scores = request.scores;
    return new Evaluation(
        softmax(scores, DISPLAY_TEMPERATURE),
        scores,
        twinDiscard.play(request.legal(), Greedy.argmax(scores, request.prior, n)));
  }

  @Override
  public OptionalDouble temperature() {
    return OptionalDouble.of(DISPLAY_TEMPERATURE);
  }

  /** softmax(s/τ) を求める。FP64 で最大値を引いてから指数を取る。 */
  static float[] softmax(float[] scores, double temperature) {
    double max = Double.NEGATIVE_INFINITY;
    for (float score : scores) max = Math.max(max, score);
    double[] e = new double[scores.length];
    double sum = 0;
    for (int a = 0; a < scores.length; a++) sum += e[a] = Math.exp((scores[a] - max) / temperature);
    float[] probabilities = new float[scores.length];
    for (int a = 0; a < scores.length; a++) probabilities[a] = (float) (e[a] / sum);
    return probabilities;
  }

  @Override
  public void close() {
    try {
      network.close();
    } finally {
      if (owned != null) owned.close();
    }
  }
}
