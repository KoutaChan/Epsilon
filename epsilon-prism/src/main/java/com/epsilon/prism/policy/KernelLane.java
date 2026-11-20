package com.epsilon.prism.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.EventCacheSettings;
import com.epsilon.prism.backend.EventPages;
import com.epsilon.prism.backend.EventTransfers;
import com.epsilon.prism.backend.KernelInputs;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.kernel.KernelProvider;
import com.epsilon.prism.kernel.KernelProvider.Cache;
import com.epsilon.prism.kernel.KernelProvider.LaneCapacity;
import com.epsilon.prism.kernel.KernelProvider.PagedPlan;
import com.epsilon.prism.kernel.KernelProvider.Pages;
import com.epsilon.prism.kernel.KernelProvider.Plan;
import com.epsilon.prism.kernel.KernelProvider.Request;
import com.epsilon.prism.kernel.PackedWeights;
import java.nio.ByteBuffer;
import java.util.Optional;

/** 受付スロットからの提出と完了イベントを、プロバイダーに渡した計算ストリームに積む。 */
final class KernelLane implements NativeInference.Lane {
  private final PtStream stream;
  private final Plan plan;
  private final EventTransfers transfers;
  private final KernelInputs inputs;

  private KernelLane(PtStream stream, Plan plan, Request request, InputFormat format) {
    this.stream = stream;
    this.plan = plan;
    inputs = new KernelInputs(request.model());
    transfers =
        new EventTransfers(format, request.capacity().events(), request.cache() == Cache.CAUSAL);
  }

  /**
   * レーンのストリームに計画を開き、重みを公開する。
   *
   * @throws IllegalStateException プロバイダーが計画を開けないとき（共有ライブラリが読めない、デバイスが合わないなど）。バックエンドと
   *     デバイスを付け、元の例外を原因に持つ
   */
  static KernelLane open(
      Device device,
      KernelProvider provider,
      Request request,
      PackedWeights weights,
      InputFormat format) {
    PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
    PtStream stream = engine.newStream(device);
    try {
      Plan plan;
      try {
        plan = provider.open(request, stream.getId());
      } catch (RuntimeException | LinkageError e) {
        throw new IllegalStateException(
            "Cannot open the "
                + request.backend()
                + " collection plan on GPU "
                + request.device()
                + ": "
                + e.getMessage(),
            e);
      }
      try {
        if (request.cache() == Cache.CAUSAL && !(plan instanceof PagedPlan))
          throw new IllegalStateException("Provider advertised event pages without a paged plan");
        plan.publish(weights);
        return new KernelLane(stream, plan, request, format);
      } catch (RuntimeException | Error e) {
        plan.close();
        throw e;
      }
    } catch (RuntimeException | Error e) {
      stream.close();
      throw e;
    }
  }

  @Override
  public void submit(HostBatch input, ByteBuffer out, long generation) {
    prepare(input, out, generation, null).run();
  }

  @Override
  public Runnable prepare(HostBatch input, ByteBuffer out, long generation, EventPages pages) {
    var batch = inputs.batch(input, out);
    return transfers.prepare(
        input,
        pages,
        () -> plan.submit(batch, generation),
        (misses, slots, bytes) ->
            ((PagedPlan) plan)
                .submit(batch, generation, new Pages(pages.map().slice(0, bytes), misses, slots)));
  }

  @Override
  public Optional<LaneCapacity> configurePages(int remaining, EventCacheSettings settings) {
    var capacity =
        ((PagedPlan) plan)
            .configurePages(
                remaining,
                settings.mode() == EventCacheSettings.Mode.OFF ? -1 : settings.eventsPerLane(),
                settings.reserveFraction());
    transfers.configure(capacity.slots());
    return Optional.of(capacity);
  }

  @Override
  public void invalidatePages() {
    transfers.invalidate();
  }

  @Override
  public void record(PtEvent event) {
    try (var ignored = stream.openScope()) {
      event.record();
    }
  }

  @Override
  public void synchronize() {
    transfers.synchronize(
        () -> {
          try (var event = stream.newEvent()) {
            record(event);
            event.synchronize();
          }
        });
  }

  @Override
  public void close() {
    try {
      transfers.close();
    } finally {
      try {
        plan.close();
      } finally {
        stream.close();
      }
    }
  }
}
