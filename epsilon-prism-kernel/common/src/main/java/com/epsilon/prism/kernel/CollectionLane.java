package com.epsilon.prism.kernel;

import com.epsilon.prism.kernel.KernelProvider.Batch;
import com.epsilon.prism.kernel.KernelProvider.LaneCapacity;
import com.epsilon.prism.kernel.KernelProvider.PagedPlan;
import com.epsilon.prism.kernel.KernelProvider.Pages;
import com.epsilon.prism.kernel.KernelProvider.Request;
import java.nio.ByteBuffer;

/**
 * 指定したストリームで、公開を終えた重み世代のバッチを提出する。native の計画は最初の公開で作り、その後の公開は同じ計画の重みを
 * 入れ替える。
 */
public final class CollectionLane implements PagedPlan {
  /** 各配布ライブラリで同じ引数を受ける収集用の JNI。 */
  public abstract static class Bindings {
    protected abstract long declaredPlan(
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
        long weightBytes);

    protected abstract void freePlan(long plan);

    protected abstract void uploadWeights(
        long plan, ByteBuffer pack, int position, long bytes, long version);

    /** 計画が提出に使える重みの世代。重みが無いか、公開の途中で失敗した後は -1。 */
    protected abstract long publishedVersion(long plan);

    protected abstract void configureCandidateInput(long handle, int[] descriptor);

    protected abstract long[] configureEventPages(
        long handle, int remaining, int events, double reserveFraction);

    protected abstract void forward(
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
        long version);

    protected abstract void forwardPaged(
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
        int transferBytes);
  }

  /** 公開済みの世代が無い印。 */
  private static final long NONE = -1;

  private final Bindings api;
  private final ByteBuffer declaration;
  private int pageSlots;
  private final Request request;
  private final long stream;
  private long handle;
  private CollectionPlanOptions options;
  private long weightBytes;
  private long published = NONE;
  private boolean closed;

  public CollectionLane(Request request, long stream, Bindings api) {
    this.api = api;
    this.declaration = ModelDeclarations.encode(request.model());
    this.request = request;
    this.stream = stream;
  }

  @Override
  public synchronized void publish(PackedWeights weights) {
    requireOpen();
    if (!request.model().equals(weights.model()))
      throw new IllegalArgumentException("Weight declaration does not match the collection plan");
    if (weights.generation() <= published)
      throw new IllegalArgumentException(
          "Generation "
              + weights.generation()
              + " is not newer than the published generation "
              + published);
    var pack = weights.data();
    if (handle == 0) {
      var plan = weights.options();
      var capacity = request.capacity();
      handle =
          api.declaredPlan(
              request.device(),
              stream,
              declaration,
              declaration.remaining(),
              ModelSchema.VERSION,
              plan.count() == null ? 0 : Math.toIntExact(plan.count().offset()),
              plan.segments() == null ? 0 : Math.toIntExact(plan.segments().offset()),
              capacity.rows(),
              capacity.events(),
              capacity.transitions(),
              capacity.actions(),
              capacity.inputBytes(),
              pack.remaining());
      options = plan;
      weightBytes = pack.remaining();
    } else if (!options.equals(weights.options()) || weightBytes != pack.remaining()) {
      throw new IllegalArgumentException("Published weights do not fit the collection plan");
    }
    published = NONE;
    try {
      api.uploadWeights(handle, pack, pack.position(), pack.remaining(), weights.generation());
    } catch (RuntimeException | Error failure) {
      // 重みの検査で拒んだときは、native の計画は前の重みと世代を持ち続ける。写しの途中で失敗したときは重みを持たない。
      try {
        published = api.publishedVersion(handle);
      } catch (RuntimeException | Error query) {
        failure.addSuppressed(query);
      }
      throw failure;
    }
    int[] layout = weights.restoration();
    if (layout.length > 0) api.configureCandidateInput(handle, layout);
    published = weights.generation();
  }

  private void requireOpen() {
    if (closed) throw new IllegalStateException("The collection plan is closed");
  }

  /** 公開を終えた世代の計画を返す。 */
  private long current(long generation) {
    requireOpen();
    if (published == NONE || generation != published)
      throw new IllegalStateException(
          "Generation "
              + generation
              + " is not published; the collection plan holds "
              + (published == NONE ? "no generation" : "generation " + published));
    return handle;
  }

  @Override
  public synchronized LaneCapacity configurePages(int remaining, int events, double reserve) {
    requireOpen();
    if (published == NONE) throw new IllegalStateException("No published collection plan");
    var capacity = LaneCapacity.of(api.configureEventPages(handle, remaining, events, reserve));
    pageSlots = capacity.slots();
    return capacity;
  }

  @Override
  public void submit(Batch batch, long generation) {
    submit(batch, generation, null);
  }

  @Override
  public synchronized void submit(Batch batch, long generation, Pages pages) {
    long plan = current(generation);
    if (pages != null && pageSlots != pages.slots())
      throw new IllegalArgumentException("Unconfigured event page capacity");
    var input = batch.input();
    var scores = batch.scores();
    if (pages == null) {
      api.forward(
          plan,
          input,
          input.position(),
          batch.rows(),
          batch.events(),
          batch.transitions(),
          batch.actions(),
          input.remaining(),
          batch.sections(),
          scores,
          scores.position(),
          generation);
    } else {
      var map = pages.data();
      api.forwardPaged(
          plan,
          input,
          input.position(),
          batch.rows(),
          batch.events(),
          batch.transitions(),
          batch.actions(),
          input.remaining(),
          batch.sections(),
          scores,
          scores.position(),
          generation,
          map,
          pages.misses(),
          pages.slots(),
          map.remaining());
    }
  }

  /** 閉じた後の公開・提出は例外になる。native の計画の解放が失敗しても閉じたものとして扱い、2 回目からは何もしない。 */
  @Override
  public synchronized void close() {
    if (closed) return;
    closed = true;
    published = NONE;
    long previous = handle;
    handle = 0;
    if (previous != 0) api.freePlan(previous);
  }
}
