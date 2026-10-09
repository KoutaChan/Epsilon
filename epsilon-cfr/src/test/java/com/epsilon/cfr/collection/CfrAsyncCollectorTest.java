package com.epsilon.cfr.collection;

import static com.epsilon.cfr.collection.CfrCollectorTestSupport.configuration;

import com.epsilon.cfr.collection.CfrCollectorTestSupport.Collector;
import com.epsilon.cfr.collection.CfrCollectorTestSupport.Result;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 推論の完了順序を制御し、エンジンの借用と収集処理のイベントループを検証する。 */
public class CfrAsyncCollectorTest {
  private static final long SEED = 901L;
  private static final BoundaryValue BOUNDARY = ignored -> new double[4];

  @Test(timeOut = 30000)
  public void completedGameAdvancesWhileAnotherGameInferenceRemainsPending() throws Exception {
    var policy = new ControlledPolicy(false);
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var collection = executor.submit(() -> collect(List.of(0L, 1L), policy, 2));
      try {
        var held = policy.next();
        var progressing = policy.next();
        long episode = progressing.context().episodeId();
        Assert.assertNotEquals(held.context().episodeId(), episode);
        progressing.complete();
        var next = policy.next();
        Assert.assertEquals(next.context().episodeId(), episode);
        Assert.assertTrue(next.context().decisionId() > progressing.context().decisionId());
        Assert.assertFalse(held.result.isDone(), "The other engine must still be paused");
      } finally {
        policy.completeAll();
      }
      Assert.assertEquals(collection.get(15, TimeUnit.SECONDS).stats().games(), 2L);
      Assert.assertTrue(policy.requests.stream().allMatch(request -> request.result.isDone()));
    }
  }

  @Test(timeOut = 30000)
  public void simultaneousDecisionsKeepTheirEnginePausedUntilEveryPredictionCompletes()
      throws Exception {
    BoundaryFixture fixture = simultaneousBoundary();
    long otherEpisode = fixture.episodeId + 10000;
    var policy = new ControlledPolicy(true);
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var collection =
          executor.submit(() -> collect(List.of(fixture.episodeId, otherEpisode), policy, 2));
      try {
        List<Request> sameBoundary = new ArrayList<>();
        while (sameBoundary.size() < fixture.decisionIds.size()) {
          var request = policy.next();
          if (request.context().episodeId() == otherEpisode) continue;
          if (fixture.decisionIds.contains(request.context().decisionId()))
            sameBoundary.add(request);
          else request.complete();
        }
        Request held = sameBoundary.removeLast();
        var before = CfrObservationCodec.encodePayload(held.context());
        for (int index = sameBoundary.size() - 1; index >= 0; index--)
          sameBoundary.get(index).complete();
        // 2つ目の対局を最初の意思決定で待機させておく。1つ目の対局の観測を借用した推論応答を
        // 待たせたまま、2つ目の対局を局末まで進める。
        policy.completeExcept(fixture.episodeId);
        Assert.assertEquals(policy.finished.poll(15, TimeUnit.SECONDS), Long.valueOf(otherEpisode));
        Assert.assertFalse(held.result.isDone());
        Assert.assertEquals(CfrObservationCodec.encodePayload(held.context()), before);
        long last = fixture.decisionIds.stream().mapToLong(Long::longValue).max().orElseThrow();
        Assert.assertTrue(
            policy.requests.stream()
                .noneMatch(
                    request ->
                        request.context().episodeId() == fixture.episodeId
                            && request.context().decisionId() > last),
            "No later decision from a partially completed engine boundary may be submitted");
      } finally {
        policy.completeAll();
      }
      Assert.assertEquals(collection.get(15, TimeUnit.SECONDS).stats().games(), 2L);
      Assert.assertTrue(policy.requests.stream().allMatch(request -> request.result.isDone()));
    }
  }

  @Test(timeOut = 30000)
  public void outOfOrderCompletionsPreserveEpisodeTracesAndUtilities() throws Exception {
    List<Long> ids = List.of(3L, 7L, 9L, 11L);
    var expected = collect(ids, batch -> predictions(batch, false), 1);
    var policy = new ControlledPolicy(false);
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var collection = executor.submit(() -> collect(ids, policy, 4));
      try {
        List<Request> first = new ArrayList<>();
        for (int index = 0; index < ids.size(); index++) first.add(policy.next());
        for (int index = first.size() - 1; index >= 0; index--) first.get(index).complete();
        while (!collection.isDone()) {
          var request = policy.pending.poll(100, TimeUnit.MILLISECONDS);
          if (request == null) continue;
          List<Request> ready = new ArrayList<>();
          ready.add(request);
          policy.pending.drainTo(ready);
          ready.sort(
              Comparator.comparingLong((Request row) -> row.context().episodeId())
                  .thenComparingLong(row -> row.context().decisionId())
                  .reversed());
          ready.forEach(Request::complete);
        }
      } finally {
        policy.completeAll();
      }
      assertEpisodes(collection.get(15, TimeUnit.SECONDS).episodes(), expected.episodes());
      Assert.assertTrue(policy.requests.stream().allMatch(request -> request.result.isDone()));
    }
  }

  @Test(timeOut = 30000)
  public void failureDrainsEverySubmittedFutureBeforeCollectionReturns() throws Exception {
    var policy = new ControlledPolicy(false);
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var collection = executor.submit(() -> collect(List.of(0L, 1L, 2L, 3L), policy, 4));
      try {
        List<Request> initial = new ArrayList<>();
        for (int index = 0; index < 4; index++) initial.add(policy.next());
        initial
            .getFirst()
            .result
            .completeExceptionally(new IllegalStateException("injected failure"));
        Assert.expectThrows(
            TimeoutException.class, () -> collection.get(200, TimeUnit.MILLISECONDS));
        Assert.assertEquals(
            policy.requests.size(), 4, "Failure must stop new inference submission");
        for (int index = 1; index < initial.size(); index++) initial.get(index).complete();
        Assert.expectThrows(ExecutionException.class, () -> collection.get(15, TimeUnit.SECONDS));
        Assert.assertTrue(policy.requests.stream().allMatch(request -> request.result.isDone()));
        Assert.assertEquals(policy.requests.size(), 4);
      } finally {
        policy.completeAll();
      }
    }
  }

  @Test(timeOut = 30000)
  public void fullSinkKeepsActiveGamesProgressingWithoutRefillingTheirCompletedSlots()
      throws Exception {
    var policy = new ControlledPolicy(false);
    var sink = new ControlledSink();
    List<Long> ids = List.of(0L, 1L, 2L, 3L);
    policy.completeExcept(1L);
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var collection = executor.submit(() -> collector(2).collect(ids, policy, BOUNDARY, sink));
      try {
        Assert.assertTrue(sink.rejected.await(10, TimeUnit.SECONDS));
        policy.completeAll();
        Long progressed;
        do {
          progressed = policy.advanced.poll(10, TimeUnit.SECONDS);
          Assert.assertNotNull(progressed);
        } while (progressed != 1L);
        Assert.assertFalse(collection.isDone(), "Collection must retain unaccepted episodes");
        Assert.assertTrue(
            policy.requests.stream().allMatch(request -> request.context().episodeId() < 2),
            "Unaccepted episodes must keep their concurrency slots");
        Assert.assertTrue(
            policy.requests.stream().allMatch(request -> request.context().roundOrdinal() == 1),
            "A game must not retain a second hand while its previous frame is blocked");
        sink.open();
        Assert.assertEquals(collection.get(10, TimeUnit.SECONDS).games(), 4L);
        var actual =
            sink.accepted.stream()
                .map(CfrCollector.CollectedEpisode::episode)
                .sorted(Comparator.comparingLong(CfrEpisode::episodeId))
                .toList();
        assertEpisodes(actual, collect(ids, batch -> predictions(batch, false), 1).episodes());
        Assert.assertTrue(policy.requests.stream().allMatch(request -> request.result.isDone()));
      } finally {
        sink.open();
        policy.completeAll();
      }
    }
  }

  @Test(timeOut = 30000)
  public void sinkFailureDrainsPendingInferenceBeforeCollectionReturns() throws Exception {
    var policy = new ControlledPolicy(false);
    var sink = new ControlledSink();
    var error = new UncheckedIOException(new IOException("simulated storage failure"));
    policy.completeExcept(1L);
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var collection =
          executor.submit(() -> collector(2).collect(List.of(0L, 1L, 2L), policy, BOUNDARY, sink));
      try {
        Assert.assertTrue(sink.rejected.await(10, TimeUnit.SECONDS));
        sink.available.completeExceptionally(error);
        Assert.expectThrows(
            TimeoutException.class, () -> collection.get(200, TimeUnit.MILLISECONDS));
        policy.completeAll();
        var failed =
            Assert.expectThrows(
                ExecutionException.class, () -> collection.get(10, TimeUnit.SECONDS));
        Assert.assertSame(failed.getCause(), error);
        Assert.assertTrue(policy.requests.stream().allMatch(request -> request.result.isDone()));
        Assert.assertTrue(sink.accepted.isEmpty());
        Assert.assertTrue(
            policy.requests.stream().allMatch(request -> request.context().episodeId() < 2));
      } finally {
        sink.open();
        policy.completeAll();
      }
    }
  }

  private static final class ControlledSink implements CfrEpisodeSink {
    final CountDownLatch rejected = new CountDownLatch(1);
    final CompletableFuture<Void> available = new CompletableFuture<>();
    final List<CfrCollector.CollectedEpisode> accepted = new CopyOnWriteArrayList<>();
    private final AtomicBoolean ready = new AtomicBoolean();

    @Override
    public boolean tryAccept(CfrCollector.CollectedEpisode episode) {
      if (!ready.get()) {
        rejected.countDown();
        return false;
      }
      accepted.add(episode);
      return true;
    }

    @Override
    public CompletionStage<Void> whenAvailable() {
      return available;
    }

    void open() {
      ready.set(true);
      available.complete(null);
    }
  }

  private static Collector collector(int concurrent) {
    return new Collector(
        configuration(CfrCollector.Mode.HAND_GRP, 0, 1, concurrent, SEED, true, true));
  }

  private static Result collect(List<Long> ids, CfrPolicy policy, int concurrent) {
    var frames = new ArrayList<CfrCollector.CollectedEpisode>();
    var stats =
        collector(concurrent)
            .collect(
                ids,
                policy,
                BOUNDARY,
                frame -> {
                  frames.add(frame);
                  if (frame.gameEnded() && policy instanceof ControlledPolicy controlled)
                    controlled.finished.add(frame.episode().episodeId());
                });
    return Result.from(frames, stats);
  }

  private static List<Prediction> predictions(List<DecisionContext> contexts, boolean firstAction) {
    return contexts.stream()
        .map(
            context -> {
              double[] target = new double[context.legalActions().size()];
              if (firstAction) target[0] = 1;
              else Arrays.fill(target, 1d / target.length);
              return new Prediction(target, CfrObservationCodec.encodePayload(context));
            })
        .toList();
  }

  private static BoundaryFixture simultaneousBoundary() {
    // 複数プレイヤーの意思決定が必要な境界を、実エンジンで探す。先頭の行動を選び、探索率を0にして、
    // 収集処理の実行順序や行動の乱数抽選に依存しないテスト条件を作る。
    for (long episode = 0; episode < 32; episode++) {
      long wall = SeedMixer.mix64(SEED ^ SeedMixer.mix64(episode) ^ 0x77616c6cL);
      var root = RootState.initial();
      var engine = new GameEngine(wall, null, root.engineStart());
      var step = engine.stepHanchan();
      for (int count = 0;
          count < 300 && step instanceof GameStepResult.AwaitingDecisions;
          count++) {
        var points = ((GameStepResult.AwaitingDecisions) step).decisions();
        if (points.size() > 1)
          return new BoundaryFixture(
              episode, Set.copyOf(points.stream().map(point -> point.id()).toList()));
        var selections = new EngineSelectionBuffer();
        for (var point : points) selections.add(point.id(), point.legalActions().getFirst());
        step = engine.commitDecisions(selections);
      }
    }
    throw new AssertionError("No simultaneous engine fixture");
  }

  private static void assertEpisodes(List<CfrEpisode> actual, List<CfrEpisode> expected) {
    Assert.assertEquals(actual.size(), expected.size());
    for (int index = 0; index < actual.size(); index++) {
      var a = actual.get(index);
      var b = expected.get(index);
      Assert.assertEquals(a.episodeId(), b.episodeId());
      Assert.assertEquals(a.utilities(), b.utilities());
      Assert.assertEquals(a.branchOutcomes(), b.branchOutcomes());
      Assert.assertEquals(a.decisions().size(), b.decisions().size());
      for (int decision = 0; decision < a.decisions().size(); decision++) {
        var x = a.decisions().get(decision);
        var y = b.decisions().get(decision);
        Assert.assertEquals(x.decisionId(), y.decisionId());
        Assert.assertEquals(x.seat(), y.seat());
        Assert.assertEquals(x.legalActionIds(), y.legalActionIds());
        Assert.assertEquals(x.selectedSlot(), y.selectedSlot());
        Assert.assertEquals(x.targetPolicy(), y.targetPolicy());
        Assert.assertEquals(x.behaviorProbability(), y.behaviorProbability());
        Assert.assertEquals(x.observation(), y.observation());
      }
    }
  }

  private record BoundaryFixture(long episodeId, Set<Long> decisionIds) {}

  private static final class Request {
    final List<DecisionContext> contexts;
    final boolean firstAction;
    final CompletableFuture<List<Prediction>> result = new CompletableFuture<>();

    Request(List<DecisionContext> contexts, boolean firstAction) {
      Assert.assertEquals(contexts.size(), 1, "Force separate futures for simultaneous decisions");
      this.contexts = List.copyOf(contexts);
      this.firstAction = firstAction;
    }

    DecisionContext context() {
      return contexts.getFirst();
    }

    void complete() {
      if (!result.isDone()) result.complete(predictions(contexts, firstAction));
    }
  }

  private static final class ControlledPolicy implements CfrPolicy {
    final boolean firstAction;
    final LinkedBlockingQueue<Request> pending = new LinkedBlockingQueue<>();
    final CopyOnWriteArrayList<Request> requests = new CopyOnWriteArrayList<>();
    final LinkedBlockingQueue<Long> finished = new LinkedBlockingQueue<>();
    final LinkedBlockingQueue<Long> advanced = new LinkedBlockingQueue<>();
    private boolean automatic;
    private Long heldEpisode;

    ControlledPolicy(boolean firstAction) {
      this.firstAction = firstAction;
    }

    @Override
    public List<Prediction> evaluate(List<DecisionContext> contexts) {
      throw new AssertionError("Collector must submit inference asynchronously");
    }

    @Override
    public synchronized CompletableFuture<List<Prediction>> submitBatch(
        List<DecisionContext> contexts, DecisionInferenceIngress ingress) {
      ingress.handoff(this).move();
      var request = new Request(contexts, firstAction);
      requests.add(request);
      if (request.context().decisionId() > 1) advanced.add(request.context().episodeId());
      if (automatic && (heldEpisode == null || heldEpisode != request.context().episodeId()))
        request.complete();
      else pending.add(request);
      return request.result;
    }

    Request next() throws InterruptedException {
      var result = pending.poll(10, TimeUnit.SECONDS);
      Assert.assertNotNull(
          result, "Collector did not submit the next independently ready decision");
      return result;
    }

    synchronized void completeExcept(Long episode) {
      heldEpisode = episode;
      automatic = true;
      for (var request : requests)
        if (episode == null || episode != request.context().episodeId()) request.complete();
    }

    void completeAll() {
      completeExcept(null);
    }
  }
}
