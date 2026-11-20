package com.epsilon.prism.kernel;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.prism.kernel.KernelProvider.Backend;
import com.epsilon.prism.kernel.KernelProvider.Batch;
import com.epsilon.prism.kernel.KernelProvider.Cache;
import com.epsilon.prism.kernel.KernelProvider.Capacity;
import com.epsilon.prism.kernel.KernelProvider.LaneCapacity;
import com.epsilon.prism.kernel.KernelProvider.Pages;
import com.epsilon.prism.kernel.KernelProvider.Request;
import com.epsilon.prism.kernel.KernelProvider.Use;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.annotations.Test;

/**
 * 収集のレーンの公開・提出・解放の約束を、CPU の偽の native で確かめる。公開し直しは同じ計画の重みを入れ替え（計画は 1 つのまま）、
 * 世代は公開のたびに新しくなければならない。失敗した公開の後と閉じた後は、どの世代も提出できない。解放が失敗しても閉じたものとして
 * 扱い、2 回目の解放はしない。イベントのページの容量は native の値の並びから 1 か所で読む。
 */
public class CollectionLaneTest {

  /** 呼ばれた順と生きている計画を控える偽の native。 */
  private static final class Recorder extends CollectionLane.Bindings {
    final Set<Long> live = new HashSet<>();
    final List<String> calls = new ArrayList<>();
    long next = 1;
    RuntimeException uploadFailure;
    boolean rejectBeforeCopy;
    long held = -1;
    RuntimeException freeFailure;
    long[] pages = {900, 600, 1000, 48, 32, 7, 11, 13, 1, 0};

    @Override
    protected long declaredPlan(
        int device,
        long stream,
        ByteBuffer declaration,
        int length,
        int abi,
        int countOffset,
        int segmentsOffset,
        int rows,
        int events,
        int transitions,
        int actions,
        long inputBytes,
        long weightBytes) {
      long handle = next++;
      live.add(handle);
      calls.add("plan " + handle);
      return handle;
    }

    @Override
    protected void freePlan(long plan) {
      if (!live.remove(plan)) throw new AssertionError("released twice: " + plan);
      calls.add("free " + plan);
      if (freeFailure != null) throw freeFailure;
    }

    @Override
    protected void uploadWeights(
        long plan, ByteBuffer pack, int position, long bytes, long version) {
      if (!live.contains(plan)) throw new AssertionError("upload to a released plan");
      calls.add("upload " + plan + " " + version);
      if (uploadFailure != null) {
        if (!rejectBeforeCopy) held = -1;
        throw uploadFailure;
      }
      held = version;
    }

    @Override
    protected long publishedVersion(long plan) {
      return held;
    }

    @Override
    protected void configureCandidateInput(long handle, int[] descriptor) {
      calls.add("restore " + handle);
    }

    @Override
    protected long[] configureEventPages(
        long handle, int remaining, int events, double reserveFraction) {
      calls.add("pages " + handle);
      return pages;
    }

    @Override
    protected void forward(
        long plan,
        ByteBuffer input,
        int inputPosition,
        int rows,
        int events,
        int transitions,
        int actions,
        long bytes,
        long[] sections,
        ByteBuffer scores,
        int scoresPosition,
        long version) {
      calls.add("forward " + plan + " " + version);
    }

    @Override
    protected void forwardPaged(
        long plan,
        ByteBuffer input,
        int inputPosition,
        int rows,
        int events,
        int transitions,
        int actions,
        long bytes,
        long[] sections,
        ByteBuffer scores,
        int scoresPosition,
        long version,
        ByteBuffer eventMap,
        int misses,
        int slots,
        int transferBytes) {
      calls.add("paged " + plan + " " + version + " " + slots);
    }
  }

  private static final Request REQUEST =
      new Request(
          DeclarationContractTest.model(),
          Backend.values()[0],
          0,
          ModelSchema.Element.BF16,
          Use.COLLECTION,
          new Capacity(2, 8, 3, 4, 64),
          Cache.CAUSAL);

  private static PackedWeights weights(long generation, int bytes, int[] restoration) {
    return new PackedWeights(
        REQUEST.model(),
        generation,
        ByteBuffer.allocateDirect(bytes),
        CollectionPlanOptions.BASE,
        restoration);
  }

  private static PackedWeights weights(long generation) {
    return weights(generation, PackedWeightLayout.HEADER_BYTES, new int[0]);
  }

  private static Batch batch() {
    return new Batch(
        1,
        0,
        1,
        1,
        new long[] {0},
        ByteBuffer.allocateDirect(16),
        ByteBuffer.allocateDirect(Float.BYTES));
  }

  @Test
  public void republishingKeepsOnePlanAndOnlyTheNewGeneration() {
    var api = new Recorder();
    var lane = new CollectionLane(REQUEST, 7, api);
    lane.publish(weights(3, PackedWeightLayout.HEADER_BYTES, new int[] {1, 2}));
    lane.submit(batch(), 3);
    lane.publish(weights(5));
    assertEquals(api.live, Set.of(1L));
    lane.submit(batch(), 5);
    var stale = expectThrows(IllegalStateException.class, () -> lane.submit(batch(), 3));
    assertTrue(stale.getMessage().contains("generation 5"), stale.getMessage());
    assertEquals(
        api.calls,
        List.of("plan 1", "upload 1 3", "restore 1", "forward 1 3", "upload 1 5", "forward 1 5"));
    lane.close();
    assertTrue(api.live.isEmpty());
  }

  @Test
  public void generationsMustIncreaseAndFitThePlan() {
    var api = new Recorder();
    var lane = new CollectionLane(REQUEST, 7, api);
    lane.publish(weights(4));
    for (long generation : new long[] {4, 2})
      expectThrows(IllegalArgumentException.class, () -> lane.publish(weights(generation)));
    expectThrows(
        IllegalArgumentException.class,
        () -> lane.publish(weights(6, PackedWeightLayout.HEADER_BYTES + 4, new int[0])));
    lane.submit(batch(), 4);
    assertEquals(api.calls, List.of("plan 1", "upload 1 4", "forward 1 4"));
    lane.close();
  }

  @Test
  public void aFailedCopyLeavesNoGenerationToSubmit() {
    var api = new Recorder();
    var lane = new CollectionLane(REQUEST, 7, api);
    lane.publish(weights(1));
    api.uploadFailure = new IllegalStateException("copy failed");
    var failure = expectThrows(IllegalStateException.class, () -> lane.publish(weights(2)));
    assertEquals(failure.getMessage(), "copy failed");
    for (long generation : new long[] {1, 2})
      expectThrows(IllegalStateException.class, () -> lane.submit(batch(), generation));
    expectThrows(IllegalStateException.class, () -> lane.configurePages(1, 8, 0));
    api.uploadFailure = null;
    lane.publish(weights(2));
    lane.submit(batch(), 2);
    assertEquals(api.live, Set.of(1L));
    lane.close();
  }

  @Test
  public void aRejectedPublicationKeepsThePreviousWeights() {
    var api = new Recorder();
    var lane = new CollectionLane(REQUEST, 7, api);
    lane.publish(weights(1));
    api.uploadFailure = new IllegalArgumentException("weight slot 3 entry 0 contains NaN");
    api.rejectBeforeCopy = true;
    expectThrows(IllegalArgumentException.class, () -> lane.publish(weights(2)));
    lane.submit(batch(), 1);
    expectThrows(IllegalStateException.class, () -> lane.submit(batch(), 2));
    api.uploadFailure = null;
    lane.publish(weights(2));
    lane.submit(batch(), 2);
    assertEquals(
        api.calls,
        List.of("plan 1", "upload 1 1", "upload 1 2", "forward 1 1", "upload 1 2", "forward 1 2"));
    lane.close();
  }

  @Test
  public void closedLanesRejectWorkAndReleaseOnce() {
    var api = new Recorder();
    var lane = new CollectionLane(REQUEST, 7, api);
    lane.publish(weights(1));
    api.freeFailure = new IllegalStateException("release failed");
    var failure = expectThrows(IllegalStateException.class, lane::close);
    assertSame(failure, api.freeFailure);
    lane.close();
    for (Runnable work :
        List.<Runnable>of(
            () -> lane.submit(batch(), 1),
            () -> lane.publish(weights(2)),
            () -> lane.configurePages(1, 8, 0))) {
      var closed = expectThrows(IllegalStateException.class, work::run);
      assertTrue(closed.getMessage().contains("closed"), closed.getMessage());
    }
    assertEquals(api.calls, List.of("plan 1", "upload 1 1", "free 1"));
    var unpublished = new CollectionLane(REQUEST, 7, api);
    unpublished.close();
    unpublished.close();
    assertEquals(api.calls.size(), 3);
  }

  @Test
  public void pagesUseTheConfiguredCapacity() {
    var api = new Recorder();
    var lane = new CollectionLane(REQUEST, 7, api);
    lane.publish(weights(1));
    var pages = new Pages(ByteBuffer.allocateDirect(16), 0, 48);
    expectThrows(IllegalArgumentException.class, () -> lane.submit(batch(), 1, pages));
    assertEquals(
        lane.configurePages(2, 8, 0.1),
        new LaneCapacity(900, 600, 1000, 48, 32, 7, 11, 13, true, false));
    lane.submit(batch(), 1, pages);
    expectThrows(
        IllegalArgumentException.class,
        () -> lane.submit(batch(), 1, new Pages(ByteBuffer.allocateDirect(16), 0, 47)));
    assertEquals(api.calls.getLast(), "paged 1 1 48");
    api.pages = new long[] {1, 2, 3};
    expectThrows(IllegalStateException.class, () -> lane.configurePages(1, 8, 0));
    lane.close();
  }
}
