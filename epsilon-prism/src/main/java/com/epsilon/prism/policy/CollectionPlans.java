package com.epsilon.prism.policy;

import ai.djl.Device;
import com.epsilon.agent.runtime.EventCacheSettings;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.backend.KernelDevices;
import com.epsilon.prism.backend.KernelInputs;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.kernel.CollectionKernels;
import com.epsilon.prism.kernel.KernelProvider;
import com.epsilon.prism.kernel.KernelProvider.Backend;
import com.epsilon.prism.kernel.KernelProvider.Cache;
import com.epsilon.prism.kernel.KernelProvider.Capacity;
import com.epsilon.prism.kernel.KernelProvider.Operation;
import com.epsilon.prism.kernel.KernelProvider.Request;
import com.epsilon.prism.kernel.KernelProvider.Use;
import com.epsilon.prism.kernel.KernelProviders;
import com.epsilon.prism.kernel.ModelSchema;
import com.epsilon.prism.kernel.PackedWeights;
import com.epsilon.prism.model.PrismNetwork;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 方策とアンカーを開くときに、各デバイスでバッチ全体を実行できる専用カーネルを選ぶ。モデルの宣言が共通の収集カーネルの対応形かは、GPU の
 * ランタイムを読まずに決める（{@link CollectionKernels#capabilities}）。対応形でなければ eager で計算する。対応形なら、デバイスの
 * バックエンドのプロバイダーが要る演算を全て持たなければならない。
 *
 * @param backend デバイスのバックエンド
 * @param cache イベントのページを使うか
 * @param devices デバイスごとに選んだプロバイダーと要求
 */
public record CollectionPlans(Backend backend, Cache cache, Map<Device, Selection> devices) {
  private static final Logger LOG = LoggerFactory.getLogger(CollectionPlans.class);

  /**
   * 1 つのデバイスで使うプロバイダーと、その計画を開く要求。
   *
   * @param provider 選んだプロバイダー
   * @param request 計画を開く要求
   */
  public record Selection(KernelProvider provider, Request request) {}

  /**
   * 方策のバッチの容量で選ぶ。
   *
   * @param network 形と入力の形式（重みは読まなくてよい）
   * @param devices GPU の列
   * @param precision 計算の精度
   * @param cache 履歴の設定。保存しない方策は null
   * @return 選んだ計画。宣言が専用カーネルの対応形でなければ空（理由をログに書く）
   * @throws IllegalArgumentException 対応形に BFLOAT16 以外の精度を指定したとき
   * @throws IllegalStateException 対応形なのに、デバイスのバックエンドのプロバイダーが無いか、要る演算を持たないとき
   */
  static Optional<CollectionPlans> select(
      PrismNetwork network, List<Device> devices, Precision precision, EventCacheSettings cache) {
    return select(
        network, devices, precision, BatchPlan.Capacity.standard(network.format()), cache);
  }

  /**
   * 指定したバッチの容量で選ぶ（アンカーは学習のマイクロバッチの容量を使う）。
   *
   * @param network 形と入力の形式（重みは読まなくてよい）
   * @param devices GPU の列
   * @param precision 計算の精度
   * @param capacity 1 回の提出の容量
   * @param cache 履歴の設定。保存しない計画は null
   * @return 選んだ計画。宣言が専用カーネルの対応形でなければ空（理由をログに書く）
   * @throws IllegalArgumentException 対応形に BFLOAT16 以外の精度を指定したとき
   * @throws IllegalStateException 対応形なのに、デバイスのバックエンドのプロバイダーが無いか、要る演算を持たないとき
   */
  public static Optional<CollectionPlans> select(
      PrismNetwork network,
      List<Device> devices,
      Precision precision,
      BatchPlan.Capacity capacity,
      EventCacheSettings cache) {
    var format = network.format();
    boolean pages =
        cache != null
            && cache.mode() == EventCacheSettings.Mode.CAUSAL
            && format.eventsSettleOnPublication();
    return select(
        network.declaration(),
        devices,
        precision,
        KernelInputs.capacity(format, capacity),
        pages,
        KernelDevices.backend(),
        KernelProviders.discover());
  }

  /**
   * バックエンドとプロバイダーの候補を渡して選ぶ。
   *
   * @param model モデルの宣言
   * @param devices GPU の列
   * @param precision 計算の精度
   * @param capacity 1 回の提出の容量
   * @param pages イベントのページを使いたいか（宣言がページに対応しなければ使わない）
   * @param backend デバイスのバックエンド
   * @param providers プロバイダーの候補
   * @return 選んだ計画。宣言が専用カーネルの対応形でなければ空（理由をログに書く）
   * @throws IllegalArgumentException 対応形に BFLOAT16 以外の精度を指定したとき
   * @throws IllegalStateException 対応形なのに、バックエンドのプロバイダーが無いか、要る演算を持たないとき
   */
  static Optional<CollectionPlans> select(
      ModelSchema.Model model,
      List<Device> devices,
      Precision precision,
      Capacity capacity,
      boolean pages,
      Backend backend,
      List<KernelProvider> providers) {
    var first = devices.getFirst();
    var declared =
        CollectionKernels.capabilities(
            request(model, backend, first, capacity, Cache.OFF), backend);
    var cache =
        pages && declared.operations().contains(Operation.EVENT_PAGES) ? Cache.CAUSAL : Cache.OFF;
    if (!declared
        .operations()
        .containsAll(KernelProviders.required(request(model, backend, first, capacity, cache)))) {
      LOG.info(
          "PRISM collection kernels do not run this model on {}; using eager inference: {}",
          backend,
          declared.detail());
      return Optional.empty();
    }
    if (precision != Precision.BFLOAT16)
      throw new IllegalArgumentException("PRISM native kernels require BFLOAT16 on " + backend);
    Map<Device, Selection> lanes = new HashMap<>();
    for (Device device : devices) {
      var request = request(model, backend, device, capacity, cache);
      lanes.put(device, new Selection(KernelProviders.collection(request, providers), request));
    }
    return Optional.of(new CollectionPlans(backend, cache, Map.copyOf(lanes)));
  }

  private static Request request(
      ModelSchema.Model model, Backend backend, Device device, Capacity capacity, Cache cache) {
    return new Request(
        model,
        backend,
        device.getDeviceId(),
        ModelSchema.Element.BF16,
        Use.COLLECTION,
        capacity,
        cache);
  }

  /** 全てのデバイスのレーンに公開する重みの束を、重みを読んだネットワークから 1 回だけ作る。 */
  PackedWeights pack(PrismNetwork network, long generation) {
    return WeightPack.publication(network, generation);
  }
}
