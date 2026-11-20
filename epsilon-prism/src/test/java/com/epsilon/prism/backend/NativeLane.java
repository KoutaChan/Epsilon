package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.EventCacheSettings;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.kernel.KernelProvider;
import com.epsilon.prism.kernel.KernelProvider.Backend;
import com.epsilon.prism.kernel.KernelProvider.Cache;
import com.epsilon.prism.kernel.KernelProvider.PagedPlan;
import com.epsilon.prism.kernel.KernelProvider.Pages;
import com.epsilon.prism.kernel.KernelProvider.Plan;
import com.epsilon.prism.kernel.KernelProvider.Request;
import com.epsilon.prism.kernel.KernelProviders;
import com.epsilon.prism.kernel.PackedWeights;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.policy.CollectionPlans;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * 製品の収集の入口を、テストから 1 本のストリームで動かす。{@link KernelProviders} がこの機械のランタイムに合わせて選んだプロバイダーの
 * 計画を開き、重みの公開、イベントのページの確保、全量の提出とページ付きの提出を、製品の {@link KernelInputs} と
 * {@link EventTransfers} を通して行う。スレッドセーフではない。
 */
public final class NativeLane implements AutoCloseable {
  private final Device device;
  private final PtStream stream;
  private final KernelProvider provider;
  private final Plan plan;
  private final KernelInputs inputs;
  private final EventTransfers transfers;
  private long generation = -1;

  private NativeLane(
      Request request,
      InputFormat format,
      Device device,
      PtStream stream,
      KernelProvider provider,
      Plan plan) {
    this.device = device;
    this.stream = stream;
    this.provider = provider;
    this.plan = plan;
    inputs = new KernelInputs(request.model());
    transfers =
        new EventTransfers(format, request.capacity().events(), request.cache() == Cache.CAUSAL);
  }

  /**
   * 新しいストリームに計画を開き、ネットワークの重みを公開する。イベントが公開の順に決まる入力の形式では、ページを使える計画を選び、
   * その形でページを使えなければ、ページを使わない計画を選ぶ（方策を開くときと同じ選び方）。
   *
   * @param network 重みを持つネットワーク（どのデバイスにあってもよい）
   * @param device GPU
   * @param capacity バッチの容量
   * @param generation 公開する重みの世代
   * @return 公開を済ませたレーン
   */
  public static NativeLane open(
      PrismNetwork network, Device device, BatchPlan.Capacity capacity, long generation) {
    var engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
    var stream = engine.newStream(device);
    try {
      return open(network, device, stream, stream.getId(), capacity, generation);
    } catch (RuntimeException | Error e) {
      stream.close();
      throw e;
    }
  }

  /** GPU の標準のストリームに提出する計画を開き、ネットワークの重みを公開する。 */
  public static NativeLane onDefaultStream(
      PrismNetwork network, Device device, BatchPlan.Capacity capacity, long generation) {
    return open(network, device, null, 0, capacity, generation);
  }

  private static NativeLane open(
      PrismNetwork network,
      Device device,
      PtStream stream,
      long streamId,
      BatchPlan.Capacity capacity,
      long generation) {
    // 方策を開くときと同じ選び方。イベントが公開の順に決まる形式で、宣言がページに対応すればページを使う計画を選ぶ。
    var settings = new EventCacheSettings(EventCacheSettings.Mode.CAUSAL, 0, 0);
    var plans =
        CollectionPlans.select(network, List.of(device), Precision.BFLOAT16, capacity, settings)
            .orElseThrow(
                () ->
                    new IllegalStateException(
                        "PRISM collection kernels do not run " + network.shape()));
    var selection = plans.devices().get(device);
    var request = selection.request();
    var provider = selection.provider();
    var plan = provider.open(request, streamId);
    try {
      if (request.cache() == Cache.CAUSAL && !(plan instanceof PagedPlan))
        throw new IllegalStateException("Provider advertised event pages without a paged plan");
      var lane = new NativeLane(request, network.format(), device, stream, provider, plan);
      lane.publish(network, generation);
      return lane;
    } catch (RuntimeException | Error e) {
      plan.close();
      throw e;
    }
  }

  /** 計画を開いたプロバイダーのバックエンド。 */
  public Backend backend() {
    return provider.backend();
  }

  /** 最後に公開した重みの世代。 */
  public long generation() {
    return generation;
  }

  /** ネットワークの今の重みを、指定した世代として公開する。 */
  public void publish(PrismNetwork network, long generation) {
    publish(WeightPack.publication(network, generation));
  }

  /** 重みを公開する。公開に失敗したら、前に公開した重みとページのまま残る。公開できたら、ページの履歴を空にする。 */
  public void publish(PackedWeights weights) {
    plan.publish(weights);
    generation = weights.generation();
    transfers.published();
  }

  /**
   * このレーンのイベントのページを確保する（空きメモリを分けるレーンは 1 本）。
   *
   * @return 確保できたスロットの数
   */
  public int configurePages(EventCacheSettings settings) {
    int slots = allocatePages(settings);
    transfers.configure(slots);
    return slots;
  }

  private int allocatePages(EventCacheSettings settings) {
    return ((PagedPlan) plan)
        .configurePages(
            1,
            settings.mode() == EventCacheSettings.Mode.OFF ? -1 : settings.eventsPerLane(),
            settings.reserveFraction())
        .slots();
  }

  /** 入力の全量を送ってスコアを求める（終わるのを待ってから読む）。 */
  public float[] forward(HostBatch batch) {
    return forward(batch, null);
  }

  /** ページの履歴と照らし、欠けたイベントだけを送ってスコアを求める。ページを使えないバッチは全量を送る。 */
  public float[] forward(HostBatch batch, EventPages events) {
    var out = scores(batch);
    prepare(batch, out, events).run();
    synchronize();
    float[] values = new float[batch.plan().count(Axis.ACTION)];
    out.asFloatBuffer().get(values);
    return values;
  }

  /** 今の世代で提出する処理を準備する。返した処理を走らせると、計画のストリームに積む。 */
  public Runnable prepare(HostBatch batch, ByteBuffer out, EventPages events) {
    var submission = inputs.batch(batch, out);
    long expected = generation;
    return transfers.prepare(
        batch,
        events,
        () -> plan.submit(submission, expected),
        (misses, slots, bytes) ->
            ((PagedPlan) plan)
                .submit(
                    submission, expected, new Pages(events.map().slice(0, bytes), misses, slots)));
  }

  /** バッチの手の数のスコアを書く、little-endian の direct なバッファを確保する。 */
  public static ByteBuffer scores(HostBatch batch) {
    return ByteBuffer.allocateDirect(batch.plan().count(Axis.ACTION) * Float.BYTES)
        .order(ByteOrder.LITTLE_ENDIAN);
  }

  /** 積んだ処理が終わるのを待つ。 */
  public void synchronize() {
    transfers.synchronize(
        () -> {
          if (stream == null) {
            var engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
            try (var event = engine.newEvent(device)) {
              event.record();
              event.synchronize();
            }
          } else {
            try (var event = stream.newEvent()) {
              try (var ignored = stream.openScope()) {
                event.record();
              }
              event.synchronize();
            }
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
        if (stream != null) stream.close();
      }
    }
  }
}
