package com.epsilon.arena;

import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ArenaRequestTest {
  @Test(timeOut = 30000)
  public void requestsKeepEpisodeIdentityAndSeatLimitedObservations() throws Exception {
    Set<Long> episodes = ConcurrentHashMap.newKeySet();
    AtomicInteger rows = new AtomicInteger();
    BatchedPolicy policy =
        new BatchedPolicy() {
          @Override
          public Object batchKey(DecisionRequest request) {
            return 0;
          }

          @Override
          public int maxBatchSize(Object key) {
            return 16;
          }

          @Override
          public Ingress tryAcquire(Object key, Runnable ready) {
            return new Ingress() {
              @Override
              public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
                for (DecisionRequest request : requests) {
                  episodes.add(request.episodeId());
                  Assert.assertEquals(request.observation().observer(), request.player());
                  Assert.assertFalse(request.legalActions().isEmpty());
                  Assert.expectThrows(
                      IllegalArgumentException.class,
                      () -> request.observation().hand((request.player() + 1) % 4).count(0));
                  rows.incrementAndGet();
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
    Participant participant = new Participant("board", policy);
    var result =
        new ArenaRunner(
                List.of(participant, participant, participant, participant),
                new RunSettings(4, 4, 2, 23, 0))
            .run();
    Assert.assertEquals(result.players().size(), 4);
    Assert.assertTrue(rows.get() > 100);
    Assert.assertEquals(episodes, Set.of(0L, 1L, 2L, 3L));
  }
}
