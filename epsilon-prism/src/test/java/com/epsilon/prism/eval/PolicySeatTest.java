package com.epsilon.prism.eval;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.play.PolicySeat;
import com.epsilon.agent.play.Query;
import com.epsilon.agent.policy.BatchedPolicy;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyRequests;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * agent の方策を卓の席に置くテスト。バッチでまとめる方策では、区分・上限・予約がそのまま通り、送り出すバッチの O は要求の O と同じで、
 * σ は選んだ手の one-hot になる。失敗は答えの失敗として伝わり、席を閉じると方策も閉じる。その場で答える方策は進行ワーカーの上で σ を書き、
 * 2 つのスレッドから呼んでも同じ答えを返す。
 */
public class PolicySeatTest {

  private static List<PolicyRequest> requests(int limit) {
    return PolicyRequests.requests(PolicyRequests.observations(9, TableWalk.Policy.CALLS, limit));
  }

  /** 最後の合法手を選ぶ方策（区分は合法手の数の偶奇、上限は区分 + 3）。受け取ったバッチの O のワードを控える。 */
  private static final class LastLegal implements BatchedPolicy {
    final List<long[]> received = new ArrayList<>();
    boolean failing;
    boolean closed;

    @Override
    public Variant variant() {
      return TENHOU_4P;
    }

    @Override
    public String label() {
      return "last";
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      throw new UnsupportedOperationException();
    }

    @Override
    public int admission(Observation observation) {
      return observation.legalActions() % 2;
    }

    @Override
    public int maxBatch(int admission) {
      return admission + 3;
    }

    @Override
    public boolean needsWork() {
      return false;
    }

    @Override
    public CompletableFuture<Void> choose(ObservationBatch batch, int[] chosen, DecisionInferenceIngress reservation) {
      reservation.handoff(this).release();
      if (failing) {
        return CompletableFuture.failedFuture(new IllegalStateException("no answer"));
      }
      Observation o = new Observation(TENHOU_4P);
      for (int index = 0; index < batch.size(); index++) {
        int start = batch.start(index);
        received.add(Arrays.copyOfRange(batch.words(), start, start + batch.longs(index)));
        chosen[index] = o.bind(batch, index).legalActions() - 1;
      }
      return CompletableFuture.completedFuture(null);
    }

    @Override
    public void close() {
      closed = true;
    }
  }

  /** 席、公開のイベントの数、合法手の数で決まる手を選ぶ方策。 */
  private record Hashed() implements InlinePolicy {
    @Override
    public Variant variant() {
      return TENHOU_4P;
    }

    @Override
    public String label() {
      return "hashed";
    }

    @Override
    public int act(Observation observation) {
      return (observation.observer() * 7 + observation.publicEvents()) % observation.legalActions();
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  @Test
  public void batchedPoliciesChooseFromCopiesOfTheRequests() {
    var policy = new LastLegal();
    var seat = new PolicySeat(policy);
    Assert.assertFalse(seat.inline());
    Assert.assertFalse(seat.needsWork());
    Assert.assertEquals(seat.maxBatch(1), 4);
    List<PolicyRequest> batch = requests(12);
    Observation o = new Observation(TENHOU_4P);
    for (PolicyRequest request : batch) {
      Assert.assertEquals(seat.admission(request.bind(o)), request.legalCount() % 2);
    }
    var attempt = seat.tryAcquireIngress(0);
    Assert.assertTrue(attempt.acquired());
    seat.submit(List.<Query>copyOf(batch), attempt.ingress()).join();
    Assert.assertEquals(policy.received.size(), batch.size());
    for (int row = 0; row < batch.size(); row++) {
      PolicyRequest request = batch.get(row);
      int start = request.recordBase();
      long[] words = Arrays.copyOfRange(request.recordWords(), start, start + request.bind(o).longs());
      Assert.assertEquals(policy.received.get(row), words, "row " + row);
      float[] sigma = new float[request.legalCount()];
      sigma[sigma.length - 1] = 1f;
      Assert.assertEquals(request.sigma, sigma, "row " + row);
      Assert.assertNull(request.scores);
      Assert.assertNull(request.prior);
    }
    Assert.assertThrows(UnsupportedOperationException.class, () -> seat.evaluateInline(batch.getFirst()));
    seat.close();
    Assert.assertTrue(policy.closed);
  }

  @Test
  public void failuresReachTheAnswer() {
    var policy = new LastLegal();
    policy.failing = true;
    var seat = new PolicySeat(policy);
    var attempt = seat.tryAcquireIngress(0);
    CompletionException failure =
        Assert.expectThrows(
            CompletionException.class, () -> seat.submit(List.<Query>copyOf(requests(3)), attempt.ingress()).join());
    Assert.assertTrue(failure.getCause() instanceof IllegalStateException, String.valueOf(failure.getCause()));
  }

  @Test
  public void inlinePoliciesAnswerOnAnyThread() throws Exception {
    var seat = new PolicySeat(new Hashed());
    Assert.assertTrue(seat.inline());
    Assert.assertEquals(seat.admission(null), 0);
    Assert.assertThrows(UnsupportedOperationException.class, () -> seat.maxBatch(0));
    List<long[]> rows = PolicyRequests.observations(9, TableWalk.Policy.CALLS, 200);
    ExecutorService threads = Executors.newFixedThreadPool(2);
    try {
      List<Future<List<float[]>>> answers = new ArrayList<>();
      for (int thread = 0; thread < 2; thread++) {
        answers.add(
            threads.submit(
                () -> {
                  List<float[]> sigmas = new ArrayList<>();
                  for (PolicyRequest request : PolicyRequests.requests(rows)) {
                    seat.evaluateInline(request);
                    sigmas.add(request.sigma);
                  }
                  return sigmas;
                }));
      }
      List<float[]> first = answers.get(0).get();
      List<float[]> second = answers.get(1).get();
      Observation o = new Observation(TENHOU_4P);
      List<PolicyRequest> expected = PolicyRequests.requests(rows);
      for (int row = 0; row < rows.size(); row++) {
        Assert.assertEquals(first.get(row), second.get(row), "row " + row);
        float[] sigma = new float[expected.get(row).legalCount()];
        sigma[new Hashed().act(expected.get(row).bind(o))] = 1f;
        Assert.assertEquals(first.get(row), sigma, "row " + row);
      }
    } finally {
      threads.shutdownNow();
    }
  }
}
