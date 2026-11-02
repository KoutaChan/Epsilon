package com.epsilon.prism.eval;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.util.SeedMixer;
import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 混成の席: 全ての類を相手の方策に任せると、その方策を候補にした対局と半荘ごとに同じ結果になり、どの類も任せなければ元の方策の対局と同じに
 * なる。鳴きと副露の間を任せると、両方の方策が打つ。
 */
public class HybridSeatTest {
  /** 塩ごとに違う、決定的な貪欲の方策（その場で評価）。 */
  static final class SaltedPolicy implements PolicyService {
    private final long salt;

    SaltedPolicy(long salt) {
      this.salt = salt;
    }

    @Override
    public int version() {
      return 1;
    }

    @Override
    public boolean inline() {
      return true;
    }

    @Override
    public void evaluateInline(GameState state, EngineDecisionPoint point, PolicyRequest request) {
      int n = request.legal().size();
      float[] scores = new float[n];
      int best = 0;
      for (int a = 0; a < n; a++) {
        long bits = SeedMixer.mix64(request.legal().get(a).toIndex() * 31L + request.seat() + salt * 1_000_003L);
        scores[a] = (float) CrnSeeds.unit(bits);
        if (scores[a] > scores[best]) best = a;
      }
      float[] sigma = new float[n];
      sigma[best] = 1f;
      request.scores = scores;
      request.sigma = sigma;
    }
  }

  private static Arena.Game[] play(PolicyService candidate, PolicyService opponent) {
    try (var driver = new CollectionDriver(new CollectionDriver.Settings(4, 2, new InferenceBatchingSettings(1000)))) {
      return Arena.detailed(driver, candidate, opponent, "test", 8, 7L, 1L << 40, EpsilonUtilityProfile.TENHOU, null)
          .games();
    }
  }

  /** 同じ s をまとめて書く非同期の方策（合議の席の試験。受付は直接）。 */
  static final class BatchedSalted implements PolicyService {
    private final SaltedPolicy inner;

    BatchedSalted(long salt) {
      inner = new SaltedPolicy(salt);
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
    public CompletableFuture<Void> submit(
        List<PolicyRequest> batch, DecisionInferenceIngress ingress) {
      ingress.handoff(this).move();
      for (PolicyRequest request : batch) inner.evaluateInline(null, null, request);
      return CompletableFuture.completedFuture(null);
    }
  }

  /** 合議の席: 重み 0 は主の方策、重み 1 は混ぜる方策と、半荘ごとに同じ結果になる。 */
  @Test(timeOut = 300_000)
  public void ensembleWeightsSelectEitherMember() {
    var opponent = new SaltedPolicy(3);
    var a = new BatchedSalted(1);
    var b = new BatchedSalted(2);
    Assert.assertEquals(play(new EnsembleService(a, b, 0), opponent), play(a, opponent));
    Assert.assertEquals(play(new EnsembleService(a, b, 1), opponent), play(b, opponent));
  }

  @Test(timeOut = 300_000)
  public void routesClassesToTheOtherPolicy() {
    var base = new SaltedPolicy(1);
    var other = new SaltedPolicy(2);
    Arena.Game[] pure = play(base, base);
    Arena.Game[] replaced = play(other, base);
    Assert.assertEquals(play(new HybridSeat(base, other, EnumSet.allOf(DecisionClass.class), false), base), replaced);
    Assert.assertEquals(play(new HybridSeat(base, other, EnumSet.noneOf(DecisionClass.class), false), base), pure);
    // 分け方: 割合 0 は任せない（元の対局と同じ）、割合 1 は全て任せる。半分と門前だけは、どちらも一部だけを任せる
    var all = EnumSet.allOf(DecisionClass.class);
    Assert.assertEquals(play(new HybridSeat(base, other, all, false, false, 0), base), pure);
    Assert.assertEquals(play(new HybridSeat(base, other, all, false, false, 1), base), replaced);
    for (var split :
        new HybridSeat[] {new HybridSeat(base, other, all, false, false, 0.5), new HybridSeat(base, other, all, false, true, 1)}) {
      play(split, base);
      long[] counts = split.routed();
      Assert.assertTrue(counts[0] > 0 && counts[1] > 0, "split " + counts[0] + "/" + counts[1]);
    }
    var calls = new HybridSeat(base, other, EnumSet.of(DecisionClass.CALL), true);
    play(calls, base);
    long[] routed = calls.routed();
    Assert.assertTrue(routed[0] > 0 && routed[1] > 0, "routed " + routed[0] + "/" + routed[1]);
  }
}
