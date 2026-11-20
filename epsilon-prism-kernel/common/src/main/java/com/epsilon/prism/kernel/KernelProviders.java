package com.epsilon.prism.kernel;

import com.epsilon.prism.kernel.KernelProvider.Cache;
import com.epsilon.prism.kernel.KernelProvider.Operation;
import com.epsilon.prism.kernel.KernelProvider.Request;
import com.epsilon.prism.kernel.KernelProvider.Use;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.Set;

/** デバイスに対応する実装を調べ、要求した演算を一つのストリームに提出できる計画を選ぶ。 */
public final class KernelProviders {
  private KernelProviders() {}

  public static List<KernelProvider> discover() {
    return ServiceLoader.load(KernelProvider.class, KernelProviders.class.getClassLoader()).stream()
        .map(ServiceLoader.Provider::get)
        .toList();
  }

  /**
   * 収集の計画がバッチ全体を提出するのに要る演算を返す。宣言の候補の段と、イベントのページを使うならページの演算を含む。
   *
   * @param request 収集の要求
   * @return 要る演算
   */
  public static Set<Operation> required(Request request) {
    if (request.use() != Use.COLLECTION)
      throw new IllegalArgumentException("Expected a collection request");
    var needed =
        EnumSet.of(
            Operation.FULL_SUBMISSION,
            Operation.RELATION_CODES,
            Operation.TOKEN_EMBEDDING,
            Operation.RELATION_ATTENTION,
            Operation.FEED_FORWARD);
    for (var stage : request.model().stages()) needed.add(Operation.valueOf(stage.name()));
    if (request.cache() == Cache.CAUSAL) needed.add(Operation.EVENT_PAGES);
    return needed;
  }

  /**
   * 要る演算を全て持つ、要求のバックエンドの収集のプロバイダーを選ぶ。能力は GPU のランタイムを読まずに照会する。
   *
   * @param request 収集の要求
   * @param providers 候補のプロバイダー
   * @return 選んだプロバイダー
   * @throws IllegalStateException バックエンドのプロバイダーが無いか、どれも要る演算を持たないとき（各プロバイダーが返した理由を付ける）。
   *     ABI が合わないときと、要る演算を持つプロバイダーが 2 つあるとき
   */
  public static KernelProvider collection(Request request, List<KernelProvider> providers) {
    var needed = required(request);
    KernelProvider selected = null;
    var reasons = new ArrayList<String>();
    for (var provider : providers) {
      if (provider.backend() != request.backend()) continue;
      requireAbi(provider);
      var capabilities = provider.capabilities(request);
      if (!capabilities.operations().containsAll(needed)) {
        var missing = EnumSet.copyOf(needed);
        missing.removeAll(capabilities.operations());
        reasons.add(
            provider.getClass().getName() + " lacks " + missing + ": " + capabilities.detail());
        continue;
      }
      if (selected != null)
        throw new IllegalStateException("Multiple collection providers for " + request.backend());
      selected = provider;
    }
    if (selected == null)
      throw new IllegalStateException(
          reasons.isEmpty()
              ? "No PRISM "
                  + request.backend()
                  + " collection provider is available for GPU "
                  + request.device()
              : "No PRISM "
                  + request.backend()
                  + " collection provider runs this model on GPU "
                  + request.device()
                  + ": "
                  + String.join("; ", reasons));
    return selected;
  }

  /**
   * 学習の演算（関係のコード・関係の Attention とその逆伝播・RMSNorm とその逆伝播）を全て持つ、要求のバックエンドの学習のプロバイダーを選ぶ。
   *
   * @param request 学習の要求
   * @param providers 候補のプロバイダー
   * @return 選んだプロバイダー（無ければ空）
   * @throws IllegalArgumentException 学習の要求でないとき
   * @throws IllegalStateException ABI が合わないときと、要る演算を持つプロバイダーが 2 つあるとき
   */
  public static Optional<TrainingProvider> training(
      Request request, List<KernelProvider> providers) {
    if (request.use() != Use.TRAINING)
      throw new IllegalArgumentException("Expected a training request");
    var needed =
        EnumSet.of(
            Operation.RELATION_CODES,
            Operation.RELATION_ATTENTION,
            Operation.RELATION_ATTENTION_BACKWARD,
            Operation.RMS_NORM,
            Operation.RMS_NORM_BACKWARD);
    TrainingProvider found = null;
    for (var provider : providers) {
      if (!(provider instanceof TrainingProvider training)
          || provider.backend() != request.backend()) continue;
      requireAbi(provider);
      if (!provider.capabilities(request).operations().containsAll(needed)) continue;
      if (found != null)
        throw new IllegalStateException("Multiple training providers for " + request.backend());
      found = training;
    }
    return Optional.ofNullable(found);
  }

  private static void requireAbi(KernelProvider provider) {
    if (provider.abiVersion() != ModelSchema.VERSION)
      throw new IllegalStateException(
          "Incompatible PRISM kernel ABI: "
              + provider.getClass().getName()
              + " implements "
              + provider.abiVersion()
              + ", this distribution needs "
              + ModelSchema.VERSION);
  }
}
