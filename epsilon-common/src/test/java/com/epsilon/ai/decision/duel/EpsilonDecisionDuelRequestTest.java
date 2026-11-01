package com.epsilon.ai.decision.duel;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import org.testng.Assert;
import org.testng.annotations.Test;

public class EpsilonDecisionDuelRequestTest {
  @Test(timeOut = 30000)
  public void parallelRequestsRetainEpisodeIdentityAndMatchDirectGameDecisions() {
    Map<Long, List<Input>> observed = new ConcurrentHashMap<>();
    BatchedPolicy policy =
        new BatchedPolicy() {
          @Override
          public Object batchKey(DecisionRequest request) {
            return 0;
          }

          @Override
          public int maxBatchSize(Object key) {
            return 8;
          }

          @Override
          public Ingress tryAcquire(Object key, Runnable available) {
            return new Ingress() {
              @Override
              public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
                for (DecisionRequest request : requests) {
                  Assert.assertEquals(request.observation().observer(), request.player());
                  Assert.expectThrows(
                      IllegalArgumentException.class,
                      () -> request.observation().hand((request.player() + 1) % 4).count(0));
                  observed
                      .computeIfAbsent(request.episodeId(), ignored -> new CopyOnWriteArrayList<>())
                      .add(
                          new Input(
                              request.id(),
                              request.player(),
                              request.legalActions().size(),
                              request.legalActions().getFirst().toIndex()));
                }
                return CompletableFuture.completedFuture(new int[requests.size()]);
              }

              @Override
              public void close() {}
            };
          }

          @Override
          public void close() {}
        };
    var result =
        EpsilonDecisionDuelArena.evaluateDuel(
            Path.of("candidate"),
            Path.of("opponent"),
            policy,
            policy,
            4,
            91826L,
            3,
            4,
            EpsilonUtilityProfile.TENHOU,
            new DecisionDuelArenaSettings(2, 2, 4),
            new InferenceBatchingSettings(0));
    Assert.assertEquals(result.metrics().completedGames(), 4);
    Assert.assertEquals(observed.keySet(), Set.of(12L, 13L, 14L, 15L));
    for (long episode : observed.keySet()) {
      var rotation = EpsilonDecisionDuelArena.rotationForGame(episode, 91826L);
      var engine = new GameEngine(rotation.wallSeed());
      GameStepResult step = engine.stepHanchan();
      List<Input> expected = new ArrayList<>();
      while (!(step instanceof GameStepResult.HanchanEnded)) {
        if (step instanceof GameStepResult.RoundSettled) {
          step = engine.stepHanchan();
          continue;
        }
        var selections = new EngineSelectionBuffer();
        for (var decision : ((GameStepResult.AwaitingDecisions) step).decisions()) {
          if (decision.legalActions().size() > 1)
            expected.add(
                new Input(
                    decision.id(),
                    decision.player(),
                    decision.legalActions().size(),
                    decision.legalActions().getFirst().toIndex()));
          selections.add(decision.id(), decision.legalActions().getFirst());
        }
        step = engine.commitDecisions(selections);
      }
      var order = Comparator.comparingLong(Input::id).thenComparingInt(Input::player);
      Assert.assertEquals(
          observed.get(episode).stream().sorted(order).toList(),
          expected.stream().sorted(order).toList(),
          "episode=" + episode);
    }
  }

  private record Input(long id, int player, int count, int firstAction) {}
}
