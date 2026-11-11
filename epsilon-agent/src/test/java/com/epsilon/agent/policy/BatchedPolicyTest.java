package com.epsilon.agent.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.agent.runtime.InferenceAdmission;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import org.testng.annotations.Test;

/**
 * まとめて推論する方策で、予約の既定の動きを確かめる。既定の予約は容量を数えず、実行先も持たない。受付が予約したバッチは予約と一緒に
 * {@code choose} へ渡り、渡した予約の所有者は方策に移る。
 */
public class BatchedPolicyTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  @Test
  public void theDefaultReservationIsDirect() {
    FirstLegalBatched policy = new FirstLegalBatched();
    assertTrue(policy.needsWork());
    DecisionInferenceIngress.Attempt attempt = policy.tryAcquire(0);
    assertTrue(attempt.acquired());
    assertNull(attempt.available());
    DecisionInferenceIngress.Ownership ownership = attempt.ingress().handoff(policy);
    assertNull(ownership.move());
    expectThrows(IllegalStateException.class, ownership::release);
    DecisionInferenceIngress other = policy.tryAcquire(0).ingress();
    expectThrows(IllegalArgumentException.class, () -> other.handoff(new FirstLegalBatched()));
  }

  @Test
  public void theAdmissionsReservationGoesToChoose() {
    FirstLegalBatched policy = new FirstLegalBatched();
    InferenceAdmission<BatchedPolicy, Integer, Integer> admission =
        new InferenceAdmission<>(0, BatchedPolicy::maxBatch, BatchedPolicy::tryAcquire, BatchedPolicy::needsWork);
    Observation o = ModelFamilyTest.firstDecision();
    ObservationBatch batch = new ObservationBatch(VARIANT, 64);
    for (int row = 0; row < 3; row++) {
      admission.add(policy, policy.admission(o), batch.add(o), 0);
    }
    InferenceAdmission.Admitted<BatchedPolicy, Integer, Integer> admitted = admission.admit(0, false);
    assertEquals(admitted.rows().size(), 3);
    int[] chosen = {-1, -1, -1};
    policy.choose(batch, chosen, admitted.ingress()).join();
    admission.inferenceFinished();
    assertEquals(chosen, new int[] {0, 0, 0});
    expectThrows(IllegalStateException.class, () -> policy.choose(batch, chosen, admitted.ingress()));
  }

  /** 最初の合法手を選ぶ方策。予約には既定のメソッドだけを使う。 */
  private static final class FirstLegalBatched implements BatchedPolicy {

    @Override
    public Variant variant() {
      return VARIANT;
    }

    @Override
    public String label() {
      return Series.RULE_BOT.label(null);
    }

    @Override
    public int admission(Observation observation) {
      return 0;
    }

    @Override
    public int maxBatch(int admission) {
      return 8;
    }

    @Override
    public CompletableFuture<Void> choose(ObservationBatch batch, int[] chosen, DecisionInferenceIngress reservation) {
      if (reservation.handoff(this).move() != null) {
        throw new IllegalStateException("A direct reservation has no executor to move to");
      }
      Arrays.fill(chosen, 0, batch.size(), 0);
      return CompletableFuture.completedFuture(null);
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      float[] probabilities = new float[observation.legalActions()];
      probabilities[0] = 1;
      return new Evaluation(probabilities, null, 0);
    }

    @Override
    public void close() {}
  }
}
