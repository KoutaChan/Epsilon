package com.epsilon.prism.eval;

import com.epsilon.core.Action;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.nano.NanoModelProvider;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import com.epsilon.spi.PolicyExecutionContext;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Epsilon-Nano の貪欲の打ち手を、まとめて推論する方策として席に置く（測定専用。学習データには入れない）。Nano のチェックポイントを
 * {@link NanoModelProvider} の {@link BatchedPolicy} で開き、ドライバが集めた判断を Nano の分類キーごとに束ねて推論し、選んだ手に
 * σ = 1 を書く。Arena の席（Nano の卓と、{@link HybridSeat} の Nano の側）に使う。
 */
public final class NanoService implements PolicyService {
  private final BatchedPolicy policy;

  private NanoService(BatchedPolicy policy) {
    this.policy = policy;
  }

  /**
   * @param checkpoint Nano の Decision チェックポイントのディレクトリ
   * @param device Nano のデバイス（cpu か gpu:N。gpuN も gpu:N に読み替える）
   * @param execution 推論の実行環境（符号化のスレッドなど）
   */
  public static NanoService open(Path checkpoint, String device, PolicyExecutionContext execution) {
    String name = device.matches("gpu\\d+") ? "gpu:" + device.substring(3) : device;
    return new NanoService(new NanoModelProvider().open(checkpoint, Map.of("device", name), execution));
  }

  @Override
  public int version() {
    return 0;
  }

  @Override
  public boolean inline() {
    return false;
  }

  @Override
  public boolean needsWork() {
    return policy.needsInferenceWork();
  }

  @Override
  public CompletableFuture<Void> submit(List<PolicyRequest> batch, DecisionInferenceIngress ingress) {
    ingress.handoff(this).move(); // 直接の予約で、解放するものはない
    Map<Object, List<PolicyRequest>> rows = new LinkedHashMap<>();
    Map<Object, List<DecisionRequest>> decisions = new LinkedHashMap<>();
    for (PolicyRequest request : batch) {
      DecisionRequest decision =
          new DecisionRequest(
              System.identityHashCode(request.owner()),
              request.point(),
              request.seat(),
              kind(request),
              request.observation(),
              request.legal());
      Object key = policy.batchKey(decision);
      rows.computeIfAbsent(key, k -> new ArrayList<>()).add(request);
      decisions.computeIfAbsent(key, k -> new ArrayList<>()).add(decision);
    }
    List<CompletableFuture<Void>> futures = new ArrayList<>();
    for (var entry : rows.entrySet()) {
      Object key = entry.getKey();
      List<PolicyRequest> own = entry.getValue();
      List<DecisionRequest> asked = decisions.get(key);
      int max = Math.max(1, policy.maxBatchSize(key));
      for (int from = 0; from < own.size(); from += max) {
        int to = Math.min(own.size(), from + max);
        futures.add(infer(key, own.subList(from, to), asked.subList(from, to)));
      }
    }
    return CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new));
  }

  /** 容量を予約して推論し、選んだ手に σ = 1 を書く（満杯なら空きの通知を待って予約し直す）。 */
  private CompletableFuture<Void> infer(
      Object key, List<PolicyRequest> own, List<DecisionRequest> asked) {
    CompletableFuture<Void> available = new CompletableFuture<>();
    BatchedPolicy.Ingress ingress = policy.tryAcquire(key, () -> available.complete(null));
    if (ingress == null) return available.thenCompose(ignored -> infer(key, own, asked));
    return ingress
        .submit(asked)
        .thenAccept(
            selected -> {
              for (int i = 0; i < own.size(); i++) {
                float[] sigma = new float[own.get(i).legal().size()];
                sigma[selected[i]] = 1f;
                own.get(i).sigma = sigma;
              }
            });
  }

  /** Nano の判断の種類（鳴いた後の打牌、応答、手番）。 */
  static EngineDecisionKind kind(PolicyRequest request) {
    if (request.kind() == Kind.POSTCALL) return EngineDecisionKind.POST_CALL_DAHAI;
    for (Action action : request.legal())
      if (action.type() == Action.Type.PASS || action.type() == Action.Type.RON_AGARI)
        return EngineDecisionKind.RESPONSE;
    return EngineDecisionKind.TURN;
  }

  @Override
  public void close() {
    try {
      policy.close();
    } catch (Exception failure) {
      throw new IllegalStateException("Could not close the Nano policy", failure);
    }
  }
}
